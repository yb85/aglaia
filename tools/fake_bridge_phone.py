#!/usr/bin/env python
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Device-free driver for the live bridge (#49) — a fake phone.

Streams an image to a running desktop as the preview feed and answers each
desktop shutter with a full-res still, so the whole Bridge tab (preview, shutter,
voice, zoom, DPI calibration) can be exercised with no iPhone.

Usage::

    # 1. Open the desktop GUI, go to the Bridge tab.
    # 2. Right-click the QR → "Copy pairing URI".
    uv run python tools/fake_bridge_phone.py '<uri>' --image page.jpg [--still-scale 4]

``--still-scale N`` sends the still at N× the preview's long edge (default: the
image's native size), emulating a phone whose stills are far larger than its
preview stream.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import ssl
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2

PREVIEW_MAX = 960


def parse_uri(uri: str) -> dict[str, str]:
    u = urlparse(uri)
    if u.scheme != "aglaia":
        raise SystemExit(f"not an aglaia URI: {uri!r}")
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    for key in ("h", "p", "t", "fp"):
        if key not in q:
            raise SystemExit(f"pairing URI missing {key!r}")
    if q.get("m") != "live":
        raise SystemExit("this URI is not a live-mode QR (expected m=live)")
    return q


class FakePhone:
    def __init__(self, q: dict[str, str]) -> None:
        self.host, self.port = q["h"], int(q["p"])
        self.token, self.fp = q["t"], q["fp"]
        self.session: str | None = None
        self.running = True

    def _conn(self, timeout: float = 40.0) -> http.client.HTTPSConnection:
        ctx = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection(self.host, self.port, context=ctx, timeout=timeout)
        conn.connect()
        der = conn.sock.getpeercert(binary_form=True)
        if der is None or hashlib.sha256(der).hexdigest() != self.fp:
            raise SystemExit("cert fingerprint mismatch — refusing (this is the pin working)")
        return conn

    def _req(self, method, path, *, body=b"", headers=None, timeout=40.0):
        conn = self._conn(timeout)
        h = {"Authorization": f"Bearer {self.token}", "Content-Length": str(len(body))}
        h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        resp = conn.getresponse()
        status, data = resp.status, resp.read()
        conn.close()
        return status, data

    def hello(self, still_wh: tuple[int, int]) -> None:
        body = json.dumps({
            "protocol": 1, "device": "FakePhone", "app": "tool",
            "still_max": [still_wh[0], still_wh[1]],
        }).encode()
        status, data = self._req("POST", "/v1/session", body=body)
        if status != 200:
            raise SystemExit(f"hello failed: {status} {data!r}")
        self.session = json.loads(data)["session"]
        print(f"connected: session={self.session}  {json.loads(data)}")

    def frame_loop(self, preview_jpeg: bytes, fps: float) -> None:
        seq = 0
        period = 1.0 / max(1.0, fps)
        while self.running:
            seq += 1
            try:
                status, _ = self._req(
                    "POST", "/v1/frame", body=preview_jpeg,
                    headers={"X-Session": self.session or "", "X-Seq": str(seq),
                             "Content-Type": "image/jpeg"})
                if status == 410:
                    print("session gone (410) — stopping")
                    self.running = False
                    return
            except (ConnectionError, OSError) as exc:
                print(f"frame send error: {exc} — desktop gone?")
                self.running = False
                return
            time.sleep(period)

    def command_loop(self, still_jpeg: bytes) -> None:
        while self.running:
            try:
                status, data = self._req("GET", "/v1/command")
            except (ConnectionError, OSError) as exc:
                print(f"poll error: {exc} — desktop gone?")
                self.running = False
                return
            cmd = json.loads(data) if data else {"command": "none"}
            kind = cmd.get("command")
            if kind == "capture":
                cid = cmd["capture_id"]
                print(f"→ capture {cid}: sending {len(still_jpeg)} B still")
                self._req("POST", "/v1/still", body=still_jpeg,
                          headers={"X-Session": self.session or "", "X-Capture-Id": cid})
            elif kind == "bye":
                print("desktop said bye — stopping")
                self.running = False
                return


def main() -> None:
    ap = argparse.ArgumentParser(description="Fake phone for the aglaia live bridge")
    ap.add_argument("uri", help="pairing URI from the Bridge tab QR (right-click → Copy)")
    ap.add_argument("--image", required=True, help="image to stream as preview + still")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--still-scale", type=float, default=0.0,
                    help="still long-edge = N × preview max (0 = image native size)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    h, w = img.shape[:2]

    # Preview: downscale to PREVIEW_MAX long edge, JPEG q60.
    s = PREVIEW_MAX / float(max(w, h))
    preview = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else img
    preview_jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()

    # Still: native size, or scaled to N × preview max.
    if args.still_scale > 0:
        target = int(PREVIEW_MAX * args.still_scale)
        ss = target / float(max(w, h))
        still = cv2.resize(img, (max(1, int(w * ss)), max(1, int(h * ss))))
    else:
        still = img
    sh, sw = still.shape[:2]
    still_jpeg = cv2.imencode(".jpg", still, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    print(f"preview {preview.shape[1]}x{preview.shape[0]}  still {sw}x{sh}")

    phone = FakePhone(parse_uri(args.uri))
    phone.hello((sw, sh))
    ft = threading.Thread(target=phone.frame_loop, args=(preview_jpeg, args.fps), daemon=True)
    ct = threading.Thread(target=phone.command_loop, args=(still_jpeg,), daemon=True)
    ft.start()
    ct.start()
    try:
        while phone.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        phone.running = False
        try:
            phone._req("POST", "/v1/bye", body=json.dumps({"reason": "user"}).encode())
        except Exception:
            pass
        print("\nbye sent")


if __name__ == "__main__":
    main()
