"""Server processing + Mistral-batch backoff + downloads + admin (#52 slices 2-4).

The heavy pipeline / Mistral / SMTP calls are stubbed (module-level functions in
`aglaia.server.processor`), so these test the orchestration: state machine,
artifact handoff, transient-`.agl` cleanup, backoff scheduling, token downloads,
and admin key actions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import aglaia.server.app as appmod
import aglaia.server.processor as proc
from aglaia.server import db as sdb
from aglaia.server.app import create_app


def _setup(tmp_path):
    db_path = tmp_path / "s.db"
    data_dir = tmp_path / "data"
    with sdb.session(db_path) as conn:
        key = sdb.create_api_key(conn, "u@example.com")
        admin = sdb.ensure_admin_secret(conn)
    app = create_app(db_path=db_path, data_dir=data_dir, start_worker=False)
    return TestClient(app), key, admin, db_path, data_dir, app.state.processor


def _submit(client, key, *, kind="pdf", ocr=None):
    fname = "doc.pdf" if kind == "pdf" else "b.aglbundle.zip"
    blob = b"%PDF stub" if kind == "pdf" else b"PK\x03\x04"
    data = {"api_key": key}
    if ocr:
        data["ocr"] = ocr
    return client.post("/run", data=data, files={"file": (fname, blob, "application/octet-stream")}).json()["job_id"]


def _fake_run(write_md=False):
    def run(cfg):
        proj = Path(cfg.parent_dir)
        (proj / "job.pdf").write_bytes(b"%PDF result")
        if write_md and cfg.do_ocr and not cfg.ocr_batch:
            (proj / "job.md").write_bytes(b"# result")
        return 0
    return run


def _job(db_path, job_id):
    with sdb.session(db_path) as conn:
        return sdb.get_job(conn, job_id)


def test_process_pdf_no_ocr_makes_pdf_only(tmp_path, monkeypatch):
    client, key, _a, db_path, data_dir, p = _setup(tmp_path)
    monkeypatch.setattr(proc, "run_pipeline", _fake_run())
    job_id = _submit(client, key)
    p.process_job(job_id)

    job = _job(db_path, job_id)
    assert job["status"] == "done"
    assert Path(job["pdf_path"]).is_file()
    assert job["md_path"] is None
    assert not (data_dir / job_id / "proj").exists()          # transient .agl discarded
    assert client.get(f"/download/{job_id}/pdf", params={"api_key": key}).status_code == 200


def test_process_with_ocr_makes_pdf_and_md(tmp_path, monkeypatch):
    client, key, _a, db_path, _d, p = _setup(tmp_path)
    monkeypatch.setattr(proc, "run_pipeline", _fake_run(write_md=True))
    job_id = _submit(client, key, ocr="apple")
    p.process_job(job_id)

    job = _job(db_path, job_id)
    assert job["status"] == "done"
    assert Path(job["pdf_path"]).is_file() and Path(job["md_path"]).is_file()


def test_failure_marks_job_failed(tmp_path, monkeypatch):
    client, key, _a, db_path, _d, p = _setup(tmp_path)
    def boom(cfg):
        raise RuntimeError("chain blew up")
    monkeypatch.setattr(proc, "run_pipeline", boom)
    job_id = _submit(client, key)
    p.process_job(job_id)
    job = _job(db_path, job_id)
    assert job["status"] == "failed" and "chain blew up" in job["error"]


def test_mistral_batch_backoff_then_done(tmp_path, monkeypatch):
    client, key, _a, db_path, _d, p = _setup(tmp_path)
    monkeypatch.setattr(proc, "run_pipeline", lambda cfg: 0)   # batch submit, no artifacts
    job_id = _submit(client, key, ocr="mistral")               # normalized to mistral:batch
    p.process_job(job_id)

    job = _job(db_path, job_id)
    assert job["status"] == "ocr_pending" and job["attempt"] == 1
    assert job["ocr_spec"] == "mistral:batch"

    monkeypatch.setattr(proc, "check_batch", lambda pf: "pending")
    p.check_job(job_id)
    assert _job(db_path, job_id)["attempt"] == 2                # rescheduled with backoff

    def fake_exports(project_file, project_dir, slug, exports, *, ocr_layer, md_refine=None):
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        (Path(project_dir) / "job.pdf").write_bytes(b"%PDF")
        (Path(project_dir) / "job.md").write_bytes(b"# md")
        return 0
    monkeypatch.setattr(proc, "check_batch", lambda pf: "done")
    monkeypatch.setattr(proc, "run_exports", fake_exports)
    p.check_job(job_id)

    job = _job(db_path, job_id)
    assert job["status"] == "done"
    assert Path(job["pdf_path"]).is_file() and Path(job["md_path"]).is_file()


def test_backoff_gives_up_at_max(tmp_path, monkeypatch):
    client, key, _a, db_path, _d, p = _setup(tmp_path)
    monkeypatch.setattr(proc, "run_pipeline", lambda cfg: 0)
    job_id = _submit(client, key, ocr="mistral")
    p.process_job(job_id)
    monkeypatch.setattr(proc, "check_batch", lambda pf: "pending")
    with sdb.session(db_path) as conn:
        sdb.update_job(conn, job_id, attempt=len(proc.BACKOFF_MINUTES))
    p.check_job(job_id)
    assert _job(db_path, job_id)["status"] == "failed"


def test_delete_cancels_pending_batch(tmp_path, monkeypatch):
    client, key, _a, _db, _d, p = _setup(tmp_path)
    monkeypatch.setattr(proc, "run_pipeline", lambda cfg: 0)
    job_id = _submit(client, key, ocr="mistral")
    p.process_job(job_id)
    cancelled = {}
    monkeypatch.setattr(appmod, "cancel_batch", lambda pf: cancelled.setdefault("called", pf))
    assert client.post(f"/delete/{job_id}", data={"api_key": key}).status_code == 200
    assert "called" in cancelled


def test_download_by_token(tmp_path, monkeypatch):
    client, key, _a, db_path, _d, p = _setup(tmp_path)
    monkeypatch.setattr(proc, "run_pipeline", _fake_run())
    job_id = _submit(client, key)
    p.process_job(job_id)
    token = _job(db_path, job_id)["download_token"]
    assert client.get(f"/download/{job_id}/pdf", params={"token": token}).status_code == 200
    assert client.get(f"/download/{job_id}/pdf", params={"token": "wrong"}).status_code == 403


def test_admin_create_and_revoke_key(tmp_path):
    client, _key, admin, db_path, _d, _p = _setup(tmp_path)
    assert client.post("/admin/keys", params={"secret": "bad"}, data={"email": "x@e.com"}).status_code == 403

    new_key = client.post("/admin/keys", params={"secret": admin}, data={"email": "new@e.com"}).json()["api_key"]
    assert client.get("/list", params={"api_key": new_key}).status_code == 200

    with sdb.session(db_path) as conn:
        key_id = next(k["id"] for k in sdb.list_api_keys(conn) if k["email"] == "new@e.com")
    assert client.post(f"/admin/keys/{key_id}/revoke", params={"secret": admin}).status_code == 200
    assert client.get("/list", params={"api_key": new_key}).status_code == 401
