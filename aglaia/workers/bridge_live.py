# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Live bridge server — phone as a tethered camera (#49).

Unlike the one-shot ``/import`` receiver (``bridge_server.py``), this keeps a
*persistent* session: the paired phone streams a low-res preview (``POST /v1/frame``)
and long-polls ``GET /v1/command`` for a desktop-triggered full-res still
(``POST /v1/still``). Same trust triad as the receiver — ephemeral self-signed
cert pinned by the QR fingerprint (TLS + server auth) plus a single-use bearer
token (authorization).

Transport is deliberately plain HTTP/1.1 over the stdlib ``ThreadingHTTPServer``
(no websocket dep, no asyncio): the phone POSTs preview frames sequentially on a
keep-alive connection (natural backpressure) and long-polls for commands. The
wire protocol (``bridge-live/1``) is JSON + JPEG + HTTP so a future Android
client needs no Apple-isms. Spec: ``docs/bridge.md``.
"""

from __future__ import annotations

import json
import secrets
import ssl
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from aglaia.workers.bridge_server import ReceiverInfo, lan_ip
from aglaia.workers.bridge_tls import generate_ephemeral_cert

PROTOCOL_VERSION = 1

# Preview knobs handed to the phone in the hello ack.
PREVIEW_MAX_PX = 960
PREVIEW_FPS = 12
PREVIEW_JPEG_Q = 0.6

# Long-poll hold (< common 30 s idle timeouts) and liveness window: a session
# with neither a frame nor a poll for this long is considered dead → the desktop
# shows a fresh QR.
POLL_SECONDS = 25
LIVENESS_TIMEOUT = 10.0

MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_STILL_BYTES = 64 * 1024 * 1024


class BridgeSessionLost(RuntimeError):
    """Raised by :meth:`BridgeLiveServer.request_still` when the phone session
    ended (bye / liveness timeout) while a still was pending."""


class BridgeLiveServer:
    """Persistent HTTPS server for one live phone session.

    Thread model: stdlib ``ThreadingHTTPServer`` runs one handler thread per
    request; long-polls park a thread in :meth:`_await_command` up to
    ``POLL_SECONDS``. All session state is guarded by ``self._lock`` /
    ``self._cmd_cond``. The desktop-facing API (``latest_preview`` /
    ``request_still`` / ``session_alive``) is thread-safe and meant to be
    called from the :class:`BridgeCameraThread`.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        on_session_started: Callable[[str], None] | None = None,
        on_session_ended: Callable[[str], None] | None = None,
    ) -> None:
        self._host = host or lan_ip()
        self._token = secrets.token_urlsafe(18)
        self._cert = generate_ephemeral_cert(host_ip=self._host)
        self._on_started = on_session_started
        self._on_ended = on_session_ended

        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._cmd_cond = threading.Condition()

        self._session: str | None = None
        self._device: str | None = None
        self._still_dims: tuple[int, int] | None = None
        self._last_seen = 0.0

        # Last decoded preview frame (BGR) + its monotonic sequence number.
        self._preview: np.ndarray | None = None
        self._preview_seq = 0

        # One pending command for the long-poll to hand back ("capture"/"bye").
        self._pending_command: dict[str, Any] | None = None
        # capture_id → (Event, slot) for the still round-trip.
        self._still_events: dict[str, threading.Event] = {}
        self._still_slot: dict[str, bytes] = {}

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> ReceiverInfo:
        handler = _make_handler(self)
        httpd = ThreadingHTTPServer(("0.0.0.0", 0), handler)
        httpd.daemon_threads = True
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(self._cert.cert_path), str(self._cert.key_path))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        self._httpd = httpd
        port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        return ReceiverInfo(
            host=self._host,
            port=int(port),
            token=self._token,
            fingerprint=self._cert.fingerprint_sha256,
        )

    def stop(self) -> None:
        """Tell a connected phone to end (``bye``) and shut the server down."""
        self._queue_command({"command": "bye"})
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        # Wake any pending still waiter so it doesn't hang until timeout.
        with self._lock:
            events = list(self._still_events.values())
            self._session = None
        for ev in events:
            ev.set()

    # ── desktop-facing API (called from BridgeCameraThread) ──────────
    @property
    def qr_mode(self) -> str:
        return "live"

    @property
    def device_name(self) -> str | None:
        with self._lock:
            return self._device

    @property
    def still_dims(self) -> tuple[int, int] | None:
        with self._lock:
            return self._still_dims

    @property
    def session_alive(self) -> bool:
        with self._lock:
            return self._session is not None and (
                time.monotonic() - self._last_seen < LIVENESS_TIMEOUT
            )

    def latest_preview(self) -> tuple[np.ndarray, int] | None:
        """Return ``(bgr, seq)`` of the most recent preview frame, or ``None``.
        ``seq`` is monotonic so the caller can skip already-shown frames."""
        with self._lock:
            if self._preview is None:
                return None
            return self._preview, self._preview_seq

    def request_still(self, timeout: float = 6.0) -> np.ndarray:
        """Ask the phone for a full-res still and block until it arrives.

        Raises :class:`BridgeSessionLost` if the session ended, or
        ``TimeoutError`` if the phone didn't answer within ``timeout``.
        """
        if not self.session_alive:
            raise BridgeSessionLost("no live phone session")
        capture_id = secrets.token_hex(8)
        ev = threading.Event()
        with self._lock:
            self._still_events[capture_id] = ev
        self._queue_command({"command": "capture", "capture_id": capture_id})
        got = ev.wait(timeout)
        with self._lock:
            self._still_events.pop(capture_id, None)
            body = self._still_slot.pop(capture_id, None)
        if not got or body is None:
            if not self.session_alive:
                raise BridgeSessionLost("session ended before still arrived")
            raise TimeoutError("phone did not return a still in time")
        import cv2

        arr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("phone returned an undecodable still")
        return arr

    # ── internal helpers ─────────────────────────────────────────────
    def _queue_command(self, cmd: dict[str, Any]) -> None:
        with self._cmd_cond:
            self._pending_command = cmd
            self._cmd_cond.notify_all()

    def _await_command(self) -> dict[str, Any]:
        """Block up to ``POLL_SECONDS`` for a pending command; ``none`` on timeout."""
        deadline = time.monotonic() + POLL_SECONDS
        with self._cmd_cond:
            while self._pending_command is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"command": "none"}
                self._cmd_cond.wait(remaining)
            cmd = self._pending_command
            self._pending_command = None
            return cmd

    def _touch(self) -> None:
        self._last_seen = time.monotonic()

    def _start_session(self, hello: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``/v1/session`` hello. Raises ``PermissionError`` on an
        active-session conflict (→ 409)."""
        if int(hello.get("protocol", 0)) != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol")
        with self._lock:
            alive = self._session is not None and (
                time.monotonic() - self._last_seen < LIVENESS_TIMEOUT
            )
            if alive:
                raise PermissionError("a session is already active")
            self._session = secrets.token_hex(8)
            self._device = str(hello.get("device", "phone"))
            dims = hello.get("still_max")
            if isinstance(dims, (list, tuple)) and len(dims) == 2:
                self._still_dims = (int(dims[0]), int(dims[1]))
            else:
                self._still_dims = None
            self._preview = None
            self._preview_seq = 0
            self._touch()
            session_id, device = self._session, self._device
        if self._on_started is not None:
            self._on_started(device)
        return {
            "session": session_id,
            "preview": {
                "max_px": PREVIEW_MAX_PX,
                "fps": PREVIEW_FPS,
                "jpeg_q": PREVIEW_JPEG_Q,
            },
            "poll_s": POLL_SECONDS,
        }

    def _check_session(self, session_id: str) -> None:
        with self._lock:
            if self._session is None or session_id != self._session:
                raise LookupError("session gone")
            self._touch()

    def _accept_frame(self, session_id: str, body: bytes) -> None:
        self._check_session(session_id)
        import cv2

        arr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("undecodable frame")
        with self._lock:
            self._preview = arr
            self._preview_seq += 1

    def _accept_still(self, session_id: str, capture_id: str, body: bytes) -> None:
        # A still for an abandoned/late capture_id is accepted (200) but dropped.
        self._check_session(session_id)
        with self._lock:
            ev = self._still_events.get(capture_id)
            if ev is not None:
                self._still_slot[capture_id] = body
        if ev is not None:
            ev.set()

    def _end_session(self, reason: str) -> None:
        with self._lock:
            if self._session is None:
                return
            self._session = None
            events = list(self._still_events.values())
        for ev in events:
            ev.set()
        if self._on_ended is not None:
            self._on_ended(reason)


def _make_handler(server: BridgeLiveServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:  # silence stderr spam
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _authed(self) -> bool:
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
            return secrets.compare_digest(token, server._token)

        def _read_body(self, limit: int) -> bytes | None:
            length = int(self.headers.get("Content-Length", 0))
            if length < 0 or length > limit:
                # Refuse without draining a huge body — drop the connection so
                # we don't sit reading gigabytes. The client may see the socket
                # close mid-upload rather than the 413; either way it's rejected.
                self.close_connection = True
                return None
            return self.rfile.read(length) if length else b""

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if not self._authed():
                self._send(401, {"error": "unauthorized"})
                return
            path = self.path.split("?", 1)[0]
            if path == "/v1/command":
                cmd = server._await_command()
                self._send(200, cmd)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if not self._authed():
                self._send(401, {"error": "unauthorized"})
                return
            path = self.path.split("?", 1)[0]
            if path == "/import":
                # An old push-mode app scanned a live QR. Fail gracefully.
                self._send(404, {"error": "live-bridge session — update AglaiaBridge"})
                return
            if path == "/v1/session":
                body = self._read_body(1 << 16)
                if body is None:
                    self._send(413, {"error": "too large"})
                    return
                try:
                    ack = server._start_session(json.loads(body or b"{}"))
                except PermissionError:
                    self._send(409, {"error": "session already active"})
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                else:
                    self._send(200, ack)
                return
            if path == "/v1/frame":
                body = self._read_body(MAX_FRAME_BYTES)
                if body is None:
                    self._send(413, {"error": "frame too large"})
                    return
                try:
                    server._accept_frame(self.headers.get("X-Session", ""), body)
                except LookupError:
                    self._send(410, {"error": "session gone"})
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                else:
                    self._send(200, {})
                return
            if path == "/v1/still":
                body = self._read_body(MAX_STILL_BYTES)
                if body is None:
                    self._send(413, {"error": "still too large"})
                    return
                try:
                    server._accept_still(
                        self.headers.get("X-Session", ""),
                        self.headers.get("X-Capture-Id", ""),
                        body,
                    )
                except LookupError:
                    self._send(410, {"error": "session gone"})
                else:
                    self._send(200, {})
                return
            if path == "/v1/bye":
                body = self._read_body(1 << 16) or b"{}"
                try:
                    reason = str(json.loads(body).get("reason", "user"))
                except Exception:  # noqa: BLE001
                    reason = "user"
                server._end_session(reason)
                self._send(200, {})
                return
            self._send(404, {"error": "not found"})

    return Handler
