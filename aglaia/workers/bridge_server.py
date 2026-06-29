# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""LAN receiver for the aglaia-bridge handoff (#47).

Listening mode runs a token-gated HTTPS server on an ephemeral port. The phone
scans a QR carrying ``host`` / ``port`` / ``token`` / cert ``fingerprint``, pins
the cert by fingerprint, and POSTs the ``.aglbundle`` zip to ``/import``.

- TLS → confidentiality (scans + token not readable on the wire).
- Fingerprint pinning (QR side channel) → server authentication / no MITM.
- Single-use bearer token → only the phone that scanned *this* QR can upload.
"""

from __future__ import annotations

import json
import secrets
import socket
import ssl
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from aglaia.workers.bridge_tls import generate_ephemeral_cert

# Reject absurd uploads outright (a big book of JPEGs is tens of MB).
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ReceiverInfo:
    host: str
    port: int
    token: str
    fingerprint: str

    def qr_uri(self) -> str:
        return f"aglaia://v1?h={self.host}&p={self.port}&t={self.token}&fp={self.fingerprint}"


def lan_ip() -> str:
    """Best-effort primary LAN IPv4 (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class BridgeReceiver:
    """One-shot HTTPS receiver. ``on_bundle`` is called with the path to the
    uploaded ``.aglbundle`` zip; raise to reject the upload."""

    def __init__(
        self,
        *,
        on_bundle: Callable[[Path], None],
        host: str | None = None,
    ) -> None:
        self._on_bundle = on_bundle
        self._host = host or lan_ip()
        self._token = secrets.token_urlsafe(18)
        self._cert = generate_ephemeral_cert(host_ip=self._host)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.received = threading.Event()

    def start(self) -> ReceiverInfo:
        handler = _make_handler(self)
        httpd = ThreadingHTTPServer(("0.0.0.0", 0), handler)
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
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # internal — invoked from the request handler thread
    def _accept(self, token: str, body: bytes) -> dict[str, Any]:
        if not secrets.compare_digest(token, self._token):
            raise PermissionError("bad token")
        tmp = Path(tempfile.mkdtemp(prefix="aglaia-bridge-rx-")) / "upload.aglbundle.zip"
        tmp.write_bytes(body)
        self._on_bundle(tmp)
        self.received.set()
        return {"ok": True}


def _make_handler(receiver: BridgeReceiver) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:  # silence default stderr spam
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path != "/import":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._send(413, {"error": "bad length"})
                return
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
            body = self.rfile.read(length)
            try:
                result = receiver._accept(token, body)
            except PermissionError:
                self._send(401, {"error": "unauthorized"})
                return
            except Exception as exc:  # noqa: BLE001 — report ingest failure to the phone
                self._send(400, {"error": str(exc)})
                return
            self._send(200, result)

    return Handler


def qr_png(uri: str, *, scale: int = 8) -> bytes:
    """Render the pairing URI as a PNG (for display in the listening-mode UI)."""
    import io

    import segno

    buf = io.BytesIO()
    segno.make(uri, error="m").save(buf, kind="png", scale=scale, border=2)
    return buf.getvalue()
