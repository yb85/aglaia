# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""FastAPI app for the Aglaïa job server (#52, slice 1).

Endpoints: ``/run`` (POST, API key) submit a job; ``/list`` / ``/check`` /
``/get`` / ``/delete`` manage jobs; ``/admin`` (admin secret) shows stats + the
API-key table. Processing, Mistral batch + backoff, and email arrive in later
slices — jobs created here sit ``pending``.

``create_app(db_path=…, data_dir=…)`` injects paths so tests use temp dirs.
"""

from __future__ import annotations

import secrets
import shutil
import sqlite3
from html import escape
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from aglaia.server import db as sdb


def _kind_from_name(name: Optional[str]) -> Optional[str]:
    n = (name or "").lower()
    if n.endswith(".pdf"):
        return "pdf"
    if n.endswith(".zip") or n.endswith(".aglbundle"):
        return "bundle"
    return None


def _normalize_ocr(ocr: Optional[str]) -> Optional[str]:
    """No `ocr` → off. On the server, Mistral defaults to batch
    (`mistral` / `mistral:streaming` → explicit; bare `mistral` → `mistral:batch`)."""
    if not ocr or not ocr.strip():
        return None
    spec = ocr.strip()
    from aglaia.workers.cli import parse_spec
    parsed = parse_spec(spec)
    if parsed.name == "mistral":
        toks = {t.lower() for t in parsed.tokens}
        if not ({"batch", "stream", "streaming"} & toks):
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


def create_app(db_path: Path | str | None = None, data_dir: Path | str | None = None) -> FastAPI:
    from aglaia.app_data import app_data_dir

    db_file = Path(db_path) if db_path else sdb.default_db_path()
    data_root = Path(data_dir) if data_dir else (app_data_dir() / "server_data")
    data_root.mkdir(parents=True, exist_ok=True)
    with sdb.session(db_file) as conn:        # create schema + admin secret eagerly
        sdb.ensure_admin_secret(conn)

    app = FastAPI(title="Aglaïa server", version="1")

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
            job_id = uuid4().hex
            jobdir = data_root / job_id
            jobdir.mkdir(parents=True, exist_ok=True)
            (jobdir / ("input.zip" if kind == "bundle" else "input.pdf")).write_bytes(payload)
            sdb.create_job(
                conn, job_id=job_id, api_key_id=key["id"], kind=kind,
                ocr_spec=_normalize_ocr(ocr), email_notif=email_notif, dpi=dpi,
            )
        # NOTE: processing is wired in slice 2 — the job is pending until then.
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
            # slice 3: poll a pending Mistral batch job here.
            return _job_public(job)

    @app.get("/get/{job_id}")
    def get_job(job_id: str, api_key: str = Query(...)) -> dict:
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            job = _owned_job(conn, key, job_id)
            result = _job_public(job)
            result["pdf_available"] = bool(job["pdf_path"])
            result["md_available"] = bool(job["md_path"])
            return result

    @app.post("/delete/{job_id}")
    def delete_job(job_id: str, api_key: str = Form(...)) -> dict:
        with sdb.session(db_file) as conn:
            key = _require_key(conn, api_key)
            _owned_job(conn, key, job_id)
            # slice 3: cancel a pending Mistral batch job before removing.
            sdb.delete_job(conn, job_id)
        shutil.rmtree(data_root / job_id, ignore_errors=True)
        return {"deleted": job_id}

    @app.get("/admin", response_class=HTMLResponse)
    def admin(secret: str = Query(...)) -> HTMLResponse:
        with sdb.session(db_file) as conn:
            if not secrets.compare_digest(secret, sdb.ensure_admin_secret(conn)):
                raise HTTPException(status_code=403, detail="forbidden")
            counts = sdb.status_counts(conn)
            per_key = {
                r["api_key_id"]: r["n"] for r in conn.execute(
                    "SELECT api_key_id, COUNT(*) AS n FROM jobs GROUP BY api_key_id")
            }
            keys = sdb.list_api_keys(conn)
        return HTMLResponse(_admin_html(counts, keys, per_key))

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
    )
