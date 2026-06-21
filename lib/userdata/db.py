# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Per-user global SQLite at $XDG_CONFIG_HOME/aglaia/user_data.sqlite (or
~/.config/aglaia/user_data.sqlite). Tracks known projects + UI preferences.
Independent of any single scan project.
"""
import os
import sqlite3
from pathlib import Path


def _default_user_data_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "aglaia" / "user_data.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id              INTEGER PRIMARY KEY,
    path            TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_opened_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_projects_opened
    ON projects(last_opened_at DESC);

CREATE TABLE IF NOT EXISTS preferences (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipelines (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    yaml_text       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


def open_userdata(path: Path | str | None = None) -> sqlite3.Connection:
    """Open or create the global user_data DB. Apply schema idempotently."""
    p = Path(path) if path else _default_user_data_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
    except sqlite3.OperationalError:
        pass
    conn.executescript(SCHEMA)
    return conn
