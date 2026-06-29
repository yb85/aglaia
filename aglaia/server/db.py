# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Server database — `APP_DATA/aglaia-server.db` (#52).

Three tables: ``api_keys`` (each tied to an email), ``jobs``, and a JSON
``config`` KV (admin secret, SMTP, public base URL …). Mirrors the
``aglaia.app_data.db`` style: ``isolation_level=None`` autocommit, ``Row``
factory, schema created on connect, ``session()`` context manager.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

#: Job lifecycle states.
STATUS_PENDING = "pending"          # accepted, not yet processed
STATUS_PROCESSING = "processing"    # chain running
STATUS_OCR_PENDING = "ocr_pending"  # Mistral batch submitted, awaiting result
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

CONFIG_ADMIN_SECRET = "admin_secret"
CONFIG_BASE_URL = "base_url"      # public URL for download links in emails
CONFIG_SMTP = "smtp"              # {"host","port","user","password","from","tls"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash   TEXT NOT NULL UNIQUE,
    email      TEXT NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    api_key_id     INTEGER NOT NULL REFERENCES api_keys(id),
    status         TEXT NOT NULL,
    kind           TEXT NOT NULL,            -- "bundle" | "pdf"
    ocr_spec       TEXT,                     -- NULL → no OCR (simple PDF)
    email_notif    INTEGER NOT NULL DEFAULT 0,
    dpi            REAL,
    mistral_job_id TEXT,
    attempt        INTEGER NOT NULL DEFAULT 0,
    next_check_at  TEXT,
    error          TEXT,
    pdf_path       TEXT,
    md_path        TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_key ON jobs(api_key_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    from aglaia.app_data import app_data_dir
    return app_data_dir() / "aglaia-server.db"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_db(path: Path | str | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


@contextlib.contextmanager
def session(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    conn = open_db(path)
    try:
        yield conn
    finally:
        conn.close()


# ── config KV ──────────────────────────────────────────────────────────

def get_config(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def set_config(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO config (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )


def ensure_admin_secret(conn: sqlite3.Connection) -> str:
    """Return the admin-panel secret, minting + storing one on first run."""
    secret = get_config(conn, CONFIG_ADMIN_SECRET)
    if not secret:
        secret = secrets.token_urlsafe(24)
        set_config(conn, CONFIG_ADMIN_SECRET, secret)
    return str(secret)


# ── api keys ─────────────────────────────────────────────────────────────

def create_api_key(conn: sqlite3.Connection, email: str) -> str:
    """Create a key for ``email`` and return the raw key (shown once; only its
    hash is stored)."""
    raw = "agl_" + secrets.token_urlsafe(24)
    conn.execute(
        "INSERT INTO api_keys (key_hash, email, active, created_at) VALUES (?, ?, 1, ?)",
        (hash_key(raw), email, _now()),
    )
    return raw


def api_key_row(conn: sqlite3.Connection, raw: str) -> Optional[sqlite3.Row]:
    if not raw:
        return None
    return conn.execute(
        "SELECT * FROM api_keys WHERE key_hash = ? AND active = 1", (hash_key(raw),)
    ).fetchone()


def list_api_keys(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC"))


# ── jobs ─────────────────────────────────────────────────────────────────

def create_job(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    api_key_id: int,
    kind: str,
    ocr_spec: Optional[str],
    email_notif: bool,
    dpi: Optional[float],
) -> None:
    """Create a pending job. The (unguessable, CSPRNG) ``job_id`` is itself the
    download capability — there's no separate token."""
    now = _now()
    conn.execute(
        "INSERT INTO jobs (id, api_key_id, status, kind, ocr_spec, email_notif, dpi, "
        "attempt, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (job_id, api_key_id, STATUS_PENDING, kind, ocr_spec, int(email_notif), dpi, now, now),
    )


def revoke_api_key(conn: sqlite3.Connection, key_id: int) -> None:
    conn.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))


def due_ocr_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """OCR-pending jobs whose next backoff check is due."""
    return list(conn.execute(
        "SELECT * FROM jobs WHERE status = ? AND (next_check_at IS NULL OR next_check_at <= ?)",
        (STATUS_OCR_PENDING, _now()),
    ))


def email_for_job(conn: sqlite3.Connection, job: sqlite3.Row) -> Optional[str]:
    row = conn.execute("SELECT email FROM api_keys WHERE id = ?", (job["api_key_id"],)).fetchone()
    return row["email"] if row else None


def get_job(conn: sqlite3.Connection, job_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def list_jobs(conn: sqlite3.Connection, api_key_id: int) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM jobs WHERE api_key_id = ? ORDER BY created_at DESC", (api_key_id,)
    ))


def update_job(conn: sqlite3.Connection, job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))


def delete_job(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
    return {r["status"]: r["n"] for r in rows}
