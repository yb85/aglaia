"""End-to-end security test for the aglaia-bridge receiver (#47):
TLS + QR-pinned fingerprint + token-gated /import + bundle ingest."""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import zipfile
from pathlib import Path

import pytest

# The bridge receiver needs `cryptography` (in the `gui` extra), which a base /
# headless CI install doesn't have — skip there rather than error at collection.
pytest.importorskip("cryptography")

from aglaia.workers.bridge_bundle import read_bundle
from aglaia.workers.bridge_server import BridgeReceiver
from aglaia.workers.bridge_tls import generate_ephemeral_cert


def _make_bundle_zip(tmp_path: Path, *, dpi: int = 150, pages: int = 2) -> bytes:
    src = tmp_path / "Book.aglbundle"
    (src / "images").mkdir(parents=True)
    manifest = {
        "format_version": 1, "project": "Book", "created": "2026-06-28T12:00:00Z",
        "device": "Test iPhone", "page_count": pages, "dpi": dpi,
    }
    (src / "manifest.json").write_text(json.dumps(manifest))
    lines = []
    for i in range(1, pages + 1):
        name = f"{i:04d}.jpg"
        (src / "images" / name).write_bytes(b"\xff\xd8\xff\xd9")
        lines.append(json.dumps({"index": i, "file": f"images/{name}",
                                 "captured": "2026-06-28T12:00:00Z", "w": 100, "h": 200}))
    (src / "pages.jsonl").write_text("\n".join(lines) + "\n")

    zip_path = tmp_path / "out.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src.parent))
    return zip_path.read_bytes()


def _pinned_post(info, *, zip_bytes: bytes, token: str | None = None, path: str = "/import") -> int:
    """Connect over TLS pinning the cert by fingerprint (as the phone does)."""
    ctx = ssl._create_unverified_context()  # no CA / hostname — we pin instead
    conn = http.client.HTTPSConnection(info.host, info.port, context=ctx, timeout=5)
    conn.connect()
    der = conn.sock.getpeercert(binary_form=True)
    assert der is not None
    # Pinning: refuse anything whose fingerprint isn't the one from the QR.
    assert hashlib.sha256(der).hexdigest() == info.fingerprint
    conn.request(
        "POST", path, body=zip_bytes,
        headers={"Authorization": f"Bearer {token if token is not None else info.token}",
                 "Content-Length": str(len(zip_bytes))},
    )
    status = conn.getresponse().status
    conn.close()
    return status


def test_pinned_upload_ingests_bundle(tmp_path: Path) -> None:
    zip_bytes = _make_bundle_zip(tmp_path)
    received: list = []

    def on_bundle(path: Path) -> None:
        received.append(read_bundle(path, extract_dir=path.parent / "x"))

    receiver = BridgeReceiver(on_bundle=on_bundle, host="127.0.0.1")
    info = receiver.start()
    try:
        assert _pinned_post(info, zip_bytes=zip_bytes) == 200
        assert receiver.received.wait(timeout=5)
        assert len(received) == 1
        assert received[0].dpi == 150
        assert [p.index for p in received[0].pages] == [1, 2]
    finally:
        receiver.stop()


def test_wrong_token_is_rejected(tmp_path: Path) -> None:
    zip_bytes = _make_bundle_zip(tmp_path)
    receiver = BridgeReceiver(on_bundle=lambda _p: None, host="127.0.0.1")
    info = receiver.start()
    try:
        assert _pinned_post(info, zip_bytes=zip_bytes, token="not-the-token") == 401
        assert not receiver.received.is_set()
    finally:
        receiver.stop()


def test_qr_uri_carries_all_fields(tmp_path: Path) -> None:
    receiver = BridgeReceiver(on_bundle=lambda _p: None, host="127.0.0.1")
    info = receiver.start()
    try:
        uri = info.qr_uri()
        assert uri.startswith("aglaia://v1?")
        for token in (f"h={info.host}", f"p={info.port}", f"t={info.token}", f"fp={info.fingerprint}"):
            assert token in uri
    finally:
        receiver.stop()


def test_fingerprint_matches_der_sha256() -> None:
    cert = generate_ephemeral_cert(host_ip="127.0.0.1")
    import cryptography.x509 as x509
    from cryptography.hazmat.primitives import serialization
    loaded = x509.load_pem_x509_certificate(cert.cert_path.read_bytes())
    der = loaded.public_bytes(serialization.Encoding.DER)
    assert hashlib.sha256(der).hexdigest() == cert.fingerprint_sha256
