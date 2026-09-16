"""Live bridge server (#49): pinned TLS + token gating + preview/command/still
round-trip, mirroring the phone side of the ``bridge-live/1`` protocol."""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import threading
import time

import pytest

# The live server needs `cryptography` (gui extra) + cv2 (base) — skip on a
# headless CI without cryptography rather than error at collection.
pytest.importorskip("cryptography")
cv2 = pytest.importorskip("cv2")
import numpy as np  # noqa: E402

from aglaia.workers import bridge_live  # noqa: E402
from aglaia.workers.bridge_live import BridgeLiveServer, BridgeSessionLost  # noqa: E402


def _jpeg(w: int = 32, h: int = 24, fill: int = 200) -> bytes:
    img = np.full((h, w, 3), fill, np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    return enc.tobytes()


class FakePhone:
    """Client side of ``bridge-live/1`` — connects over TLS pinning the cert by
    fingerprint (exactly as the iOS app does), then drives the session."""

    def __init__(self, info, *, token: str | None = None) -> None:
        self.info = info
        self.token = token if token is not None else info.token
        self.session: str | None = None

    def _conn(self, timeout: float = 30.0) -> http.client.HTTPSConnection:
        ctx = ssl._create_unverified_context()  # we pin the fingerprint instead
        conn = http.client.HTTPSConnection(
            self.info.host, self.info.port, context=ctx, timeout=timeout
        )
        conn.connect()
        der = conn.sock.getpeercert(binary_form=True)
        assert der is not None
        assert hashlib.sha256(der).hexdigest() == self.info.fingerprint
        return conn

    def _request(self, method, path, *, body=b"", headers=None, timeout=30.0):
        conn = self._conn(timeout)
        h = {"Authorization": f"Bearer {self.token}", "Content-Length": str(len(body))}
        h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        resp = conn.getresponse()
        status, data = resp.status, resp.read()
        conn.close()
        return status, data

    def hello(self, *, device="Test iPhone", still_max=(4000, 3000), protocol=1):
        body = json.dumps(
            {"protocol": protocol, "device": device, "app": "test", "still_max": list(still_max)}
        ).encode()
        status, data = self._request("POST", "/v1/session", body=body)
        if status == 200:
            self.session = json.loads(data)["session"]
        return status, (json.loads(data) if data else {})

    def send_frame(self, jpeg: bytes, *, seq: int = 1):
        # An oversize body is refused server-side before the upload finishes,
        # which RSTs the connection mid-send; treat that as a rejection (-1).
        try:
            return self._request(
                "POST", "/v1/frame", body=jpeg,
                headers={"X-Session": self.session or "", "X-Seq": str(seq),
                         "Content-Type": "image/jpeg"},
            )[0]
        except (BrokenPipeError, ConnectionError, ssl.SSLError):
            return -1

    def poll(self, timeout: float = 30.0):
        _status, data = self._request("GET", "/v1/command", timeout=timeout)
        return json.loads(data)

    def send_still(self, capture_id: str, jpeg: bytes):
        return self._request(
            "POST", "/v1/still", body=jpeg,
            headers={"X-Session": self.session or "", "X-Capture-Id": capture_id},
        )[0]

    def bye(self, reason: str = "user"):
        return self._request("POST", "/v1/bye", body=json.dumps({"reason": reason}).encode())[0]


def _server(**kw) -> BridgeLiveServer:
    return BridgeLiveServer(host="127.0.0.1", **kw)


def test_qr_uri_carries_mode() -> None:
    srv = _server()
    info = srv.start()
    try:
        uri = info.qr_uri(mode="live")
        assert uri.startswith("aglaia://v1?")
        assert "&m=live" in uri
        # push mode (no arg) stays back-compatible with the receiver
        assert "&m=" not in info.qr_uri()
    finally:
        srv.stop()


def test_hello_starts_session_and_fires_callback() -> None:
    started: list[str] = []
    srv = _server(on_session_started=started.append)
    info = srv.start()
    try:
        phone = FakePhone(info)
        status, ack = phone.hello()
        assert status == 200
        assert ack["poll_s"] == bridge_live.POLL_SECONDS
        assert ack["preview"]["max_px"] == bridge_live.PREVIEW_MAX_PX
        assert srv.session_alive
        assert srv.device_name == "Test iPhone"
        assert srv.still_dims == (4000, 3000)
        assert started == ["Test iPhone"]
    finally:
        srv.stop()


def test_bad_token_is_rejected() -> None:
    srv = _server()
    info = srv.start()
    try:
        phone = FakePhone(info, token="not-the-token")
        status, _ = phone.hello()
        assert status == 401
        assert not srv.session_alive
    finally:
        srv.stop()


def test_second_phone_gets_409() -> None:
    srv = _server()
    info = srv.start()
    try:
        assert FakePhone(info).hello()[0] == 200
        assert FakePhone(info).hello()[0] == 409
    finally:
        srv.stop()


def test_frame_updates_latest_preview() -> None:
    srv = _server()
    info = srv.start()
    try:
        phone = FakePhone(info)
        phone.hello()
        assert srv.latest_preview() is None
        assert phone.send_frame(_jpeg(48, 36), seq=1) == 200
        first = srv.latest_preview()
        assert first is not None
        frame, seq = first
        assert frame.shape[:2] == (36, 48)
        assert phone.send_frame(_jpeg(48, 36), seq=2) == 200
        assert srv.latest_preview()[1] == seq + 1  # seq advanced
    finally:
        srv.stop()


def test_still_round_trip_correlates_capture_id() -> None:
    srv = _server()
    info = srv.start()
    try:
        phone = FakePhone(info)
        phone.hello()
        result: dict = {}

        def grab() -> None:
            result["img"] = srv.request_still(timeout=5)

        t = threading.Thread(target=grab)
        t.start()
        cmd = phone.poll(timeout=5)
        assert cmd["command"] == "capture"
        assert phone.send_still(cmd["capture_id"], _jpeg(4000, 3000)) == 200
        t.join(timeout=5)
        assert result["img"].shape[:2] == (3000, 4000)
    finally:
        srv.stop()


def test_poll_times_out_to_none(monkeypatch) -> None:
    monkeypatch.setattr(bridge_live, "POLL_SECONDS", 0.3)
    srv = _server()
    info = srv.start()
    try:
        phone = FakePhone(info)
        phone.hello()
        t0 = time.monotonic()
        cmd = phone.poll(timeout=5)
        assert cmd == {"command": "none"}
        assert time.monotonic() - t0 < 3  # returned near the 0.3 s deadline
    finally:
        srv.stop()


def test_silence_marks_session_dead(monkeypatch) -> None:
    monkeypatch.setattr(bridge_live, "LIVENESS_TIMEOUT", 0.2)
    srv = _server()
    info = srv.start()
    try:
        FakePhone(info).hello()
        assert srv.session_alive
        time.sleep(0.4)
        assert not srv.session_alive
        with pytest.raises(BridgeSessionLost):
            srv.request_still(timeout=1)
    finally:
        srv.stop()


def test_stop_delivers_bye_to_poller() -> None:
    srv = _server()
    info = srv.start()
    phone = FakePhone(info)
    phone.hello()
    got: dict = {}

    def poll() -> None:
        got["cmd"] = phone.poll(timeout=5)

    t = threading.Thread(target=poll)
    t.start()
    time.sleep(0.2)
    srv.stop()
    t.join(timeout=5)
    assert got["cmd"]["command"] == "bye"


def test_oversize_frame_rejected() -> None:
    srv = _server()
    info = srv.start()
    try:
        phone = FakePhone(info)
        phone.hello()
        big = b"\x00" * (bridge_live.MAX_FRAME_BYTES + 1)
        assert phone.send_frame(big, seq=1) != 200  # 413 or connection dropped
    finally:
        srv.stop()


def test_import_path_returns_404_with_hint() -> None:
    srv = _server()
    info = srv.start()
    try:
        phone = FakePhone(info)
        status, data = phone._request("POST", "/import", body=b"x")
        assert status == 404
        assert "update" in json.loads(data)["error"].lower()
    finally:
        srv.stop()
