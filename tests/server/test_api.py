"""Aglaïa server API tests (#52, slice 1) — intake + auth + job management.

Uses FastAPI's in-process TestClient with a temp DB + data dir. Processing is
slice 2, so jobs stay `pending` here; these cover routing, auth, isolation,
storage, and the admin panel."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from aglaia.server import db as sdb
from aglaia.server.app import create_app


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "server.db"
    data_dir = tmp_path / "data"
    with sdb.session(db_path) as conn:
        key = sdb.create_api_key(conn, "user@example.com")
        admin = sdb.ensure_admin_secret(conn)
    client = TestClient(create_app(db_path=db_path, data_dir=data_dir, start_worker=False))
    return client, key, admin, db_path, data_dir


def _bundle_file():
    return {"file": ("book.aglbundle.zip", b"PK\x03\x04stub", "application/zip")}


def _pdf_file():
    return {"file": ("doc.pdf", b"%PDF-1.4 stub", "application/pdf")}


def test_run_rejects_bad_key(env):
    client, *_ = env
    r = client.post("/run", data={"api_key": "nope"}, files=_pdf_file())
    assert r.status_code == 401


def test_run_rejects_bad_filetype(env):
    client, key, *_ = env
    r = client.post("/run", data={"api_key": key}, files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_run_creates_bundle_job_and_stores_input(env):
    client, key, _admin, _db, data_dir = env
    r = client.post("/run", data={"api_key": key, "ocr": "mistral"}, files=_bundle_file())
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "pending"
    # input persisted under server_data/<job_id>/
    assert (data_dir / job_id / "input.zip").is_file()
    # listed for the owner, mistral normalized to batch on the server
    jobs = client.get("/list", params={"api_key": key}).json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["kind"] == "bundle"
    assert jobs[0]["ocr"] == "mistral:batch"


def test_run_pdf_without_ocr(env):
    client, key, *_ = env
    client.post("/run", data={"api_key": key}, files=_pdf_file())
    job = client.get("/list", params={"api_key": key}).json()["jobs"][0]
    assert job["kind"] == "pdf"
    assert job["ocr"] is None


def test_get_and_delete(env):
    client, key, _admin, _db, data_dir = env
    job_id = client.post("/run", data={"api_key": key}, files=_pdf_file()).json()["job_id"]

    got = client.get(f"/get/{job_id}", params={"api_key": key})
    assert got.status_code == 200
    assert got.json()["pdf_url"] is None and got.json()["md_url"] is None

    dele = client.post(f"/delete/{job_id}", data={"api_key": key})
    assert dele.status_code == 200
    assert client.get("/list", params={"api_key": key}).json()["jobs"] == []
    assert not (data_dir / job_id).exists()


def test_cross_key_isolation(env):
    client, key, _admin, db_path, _data = env
    with sdb.session(db_path) as conn:
        other = sdb.create_api_key(conn, "two@example.com")
    job_id = client.post("/run", data={"api_key": key}, files=_bundle_file()).json()["job_id"]
    # another key can neither see nor fetch it
    assert client.get(f"/get/{job_id}", params={"api_key": other}).status_code == 404
    assert client.get("/list", params={"api_key": other}).json()["jobs"] == []


def test_admin_auth_and_panel(env):
    client, _key, admin, *_ = env
    assert client.get("/admin", params={"secret": "wrong"}).status_code == 403
    page = client.get("/admin", params={"secret": admin})
    assert page.status_code == 200
    assert "user@example.com" in page.text
