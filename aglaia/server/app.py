# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""FastAPI app for the Aglaïa job server (#52).

`/run` (API key) submits a job; a background worker ingests → pipelines →
exports it (PDF; +OCR+MD when OCR is on) into the job folder and discards the
transient `.agl`. Mistral *batch* OCR is polled with exponential backoff (and on
`/check`/`/get`). `/list` `/get` `/delete` manage jobs; `/download` serves the
artifacts (by API key or per-job token); `/admin` shows stats + manages keys.

`create_app(db_path=…, data_dir=…, start_worker=…)` injects paths and lets tests
drive the processor manually.
"""

from __future__ import annotations

import secrets
import shutil
import sqlite3
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from aglaia.server import db as sdb
from aglaia.server.mailer import download_url
from aglaia.server.processor import JobProcessor, cancel_batch


def _kind_from_name(name: Optional[str]) -> Optional[str]:
    n = (name or "").lower()
    if n.endswith(".pdf"):
        return "pdf"
    if n.endswith(".zip") or n.endswith(".aglbundle"):
        return "bundle"
    return None


def _normalize_ocr(ocr: Optional[str]) -> Optional[str]:
    """No `ocr` → off. On the server Mistral defaults to batch
    (`mistral` → `mistral:batch`; `mistral:streaming` keeps it sync)."""
    if not ocr or not ocr.strip():
        return None
    spec = ocr.strip()
    from aglaia.workers.cli import parse_spec
    parsed = parse_spec(spec)
    if parsed.name == "mistral" and not ({"batch", "stream", "streaming"} & {t.lower() for t in parsed.tokens}):
        return "mistral:batch"
    return spec


def _job_public(row: sqlite3.Row) -> dict:
    return {
        "job_id": row["id"],
        "status": row["status"],
        "kind": row["kind"],
        "ocr": row["ocr_spec"],
        "email_notif": bool(row["email_notif"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row["error"],
    }


def create_app(
    db_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    start_worker: bool = True,
) -> FastAPI:
    from aglaia.app_data import app_data_dir

    db_file = Path(db_path) if db_path else sdb.default_db_path()
    data_root = Path(data_dir) if data_dir else (app_data_dir() / "server_data")
    data_root.mkdir(parents=True, exist_ok=True)
    with sdb.session(db_file) as conn:
        sdb.ensure_admin_secret(conn)

    processor = JobProcessor(db_file, data_root)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        if start_worker:
            processor.start()
        try:
            yield
        finally:
            processor.stop()

    app = FastAPI(title="Aglaïa server", version="1", lifespan=_lifespan)
    app.state.processor = processor

    def _require_key(conn: sqlite3.Connection, api_key: str) -> sqlite3.Row:
        row = sdb.api_key_row(conn, api_key)
        if row is None:
            raise HTTPException(status_code=401, detail="invalid or inactive API key")
        return row

    def _owned_job(conn: sqlite3.Connection, key: sqlite3.Row, job_id: str) -> sqlite3.Row:
        job = sdb.get_job(conn, job_id)
        if job is None or job["api_key_id"] != key["id"]:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    def _require_admin(conn: sqlite3.Connection, secret: str) -> None:
        if not secrets.compare_digest(secret, sdb.ensure_admin_secret(conn)):
            raise HTTPException(status_code=403, detail="forbidden")

    def _result(conn: sqlite3.Connection, job: sqlite3.Row) -> dict:
        out = _job_public(job)
        base = sdb.get_config(conn, sdb.CONFIG_BASE_URL)
        out["pdf_url"] = download_url(base, job["id"], "pdf", job["download_token"]) if job["pdf_path"] else None
        out["md_url"] = download_url(base, job["id"], "md", job["download_token"]) if job["md_path"] else None
        return out

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/run")
    async def run_job(
        file: UploadFile = File(...),
        api_key: str = Form(...),
        email_notif: bool = Form(False),
        ocr: Optional[str] = Form(None),
        dpi: Optional[float] = Form(None),
    ) -> dict:
        kind = _kind_from_name(file.filename)
        if kind is None:
            raise HTTPException(status_code=400, detail="upload must be an .aglbundle/.zip or a .pdf")
        payload = await file.read()
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            # Cryptographically-random, unguessable id (≈128 bits, URL/path safe).
            # Access is still gated by API key / download token — the id is not a secret.
            job_id = secrets.token_urlsafe(16)
            jobdir = data_root / job_id
            jobdir.mkdir(parents=True, exist_ok=True)
            (jobdir / ("input.zip" if kind == "bundle" else "input.pdf")).write_bytes(payload)
            sdb.create_job(conn, job_id=job_id, api_key_id=key["id"], kind=kind,
                           ocr_spec=_normalize_ocr(ocr), email_notif=email_notif, dpi=dpi)
        return {"job_id": job_id, "status": sdb.STATUS_PENDING}

    @app.get("/list")
    def list_jobs(api_key: str = Query(...)) -> dict:
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            return {"jobs": [_job_public(r) for r in sdb.list_jobs(conn, key["id"])]}

    @app.get("/check/{job_id}")
    def check_job(job_id: str, api_key: str = Query(...)) -> dict:
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            job = _owned_job(conn, key, job_id)
            pending = job["status"] == sdb.STATUS_OCR_PENDING
        if pending:
            processor.check_job(job_id)   # poll Mistral now
        with sdb.session(db_file) as conn:
            return _result(conn, sdb.get_job(conn, job_id))

    @app.get("/get/{job_id}")
    def get_job(job_id: str, api_key: str = Query(...)) -> dict:
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            job = _owned_job(conn, key, job_id)
            pending = job["status"] == sdb.STATUS_OCR_PENDING
        if pending:
            processor.check_job(job_id)
        with sdb.session(db_file) as conn:
            return _result(conn, sdb.get_job(conn, job_id))

    @app.get("/download/{job_id}/{which}")
    def download(job_id: str, which: str, api_key: Optional[str] = Query(None),
                 token: Optional[str] = Query(None)) -> FileResponse:
        if which not in ("pdf", "md"):
            raise HTTPException(status_code=400, detail="which must be 'pdf' or 'md'")
        with sdb.session(db_file) as conn:
            job = sdb.get_job(conn, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            ok = False
            if token and job["download_token"] and secrets.compare_digest(token, job["download_token"]):
                ok = True
            elif api_key:
                key = sdb.api_key_row(conn, api_key)
                ok = key is not None and key["id"] == job["api_key_id"]
            if not ok:
                raise HTTPException(status_code=403, detail="forbidden")
            path = job["pdf_path"] if which == "pdf" else job["md_path"]
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=404, detail=f"{which} not available")
        media = "application/pdf" if which == "pdf" else "text/markdown"
        return FileResponse(path, media_type=media, filename=f"{job_id}.{which}")

    @app.post("/delete/{job_id}")
    def delete_job(job_id: str, api_key: str = Form(...)) -> dict:
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            job = _owned_job(conn, key, job_id)
            was_pending = job["status"] == sdb.STATUS_OCR_PENDING
        if was_pending:   # cancel the in-flight Mistral batch before removing
            cancel_batch(data_root / job_id / "proj" / "job.agl")
        with sdb.session(db_file) as conn:
            sdb.delete_job(conn, job_id)
        shutil.rmtree(data_root / job_id, ignore_errors=True)
        return {"deleted": job_id}

    # ── admin ─────────────────────────────────────────────────────────────

    @app.get("/admin", response_class=HTMLResponse)
    def admin(secret: str = Query(...)) -> HTMLResponse:
        with sdb.session(db_file) as conn:
            _require_admin(conn, secret)
            counts = sdb.status_counts(conn)
            per_key = {r["api_key_id"]: r["n"] for r in conn.execute(
                "SELECT api_key_id, COUNT(*) AS n FROM jobs GROUP BY api_key_id")}
            keys = sdb.list_api_keys(conn)
        return HTMLResponse(_admin_html(counts, keys, per_key))

    @app.post("/admin/keys")
    def admin_create_key(secret: str = Query(...), email: str = Form(...)) -> dict:
        with sdb.session(db_file) as conn:
            _require_admin(conn, secret)
            raw = sdb.create_api_key(conn, email)
        return {"email": email, "api_key": raw}   # shown once

    @app.post("/admin/keys/{key_id}/revoke")
    def admin_revoke_key(key_id: int, secret: str = Query(...)) -> dict:
        with sdb.session(db_file) as conn:
            _require_admin(conn, secret)
            sdb.revoke_api_key(conn, key_id)
        return {"revoked": key_id}

    return app


def _admin_html(counts: dict, keys, per_key: dict) -> str:
    total = sum(counts.values())
    stat_rows = "".join(
        f"<tr><td>{escape(s)}</td><td>{n}</td></tr>" for s, n in sorted(counts.items())
    ) or "<tr><td colspan=2>no jobs</td></tr>"
    key_rows = "".join(
        f"<tr><td>{k['id']}</td><td>{escape(k['email'])}</td>"
        f"<td>{'active' if k['active'] else 'revoked'}</td>"
        f"<td>{escape(k['created_at'])}</td><td>{per_key.get(k['id'], 0)}</td></tr>"
        for k in keys
    ) or "<tr><td colspan=5>no API keys</td></tr>"
    return (
        "<!doctype html><meta charset=utf-8><title>Aglaïa server — admin</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:760px}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}"
        "h1{font-size:1.4rem}</style>"
        "<h1>Aglaïa server</h1>"
        f"<p>{total} job(s) total.</p>"
        "<h2>Jobs by status</h2>"
        f"<table><tr><th>status</th><th>count</th></tr>{stat_rows}</table>"
        "<h2>API keys</h2>"
        "<table><tr><th>id</th><th>email</th><th>state</th><th>created</th>"
        f"<th>jobs</th></tr>{key_rows}</table>"
        "<p style=color:#888>Create: POST /admin/keys?secret=…&email=…  ·  "
        "Revoke: POST /admin/keys/&lt;id&gt;/revoke?secret=…</p>"
    )
