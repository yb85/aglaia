# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import sqlite3
from datetime import datetime, timezone
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def register(self, path: str, name: str) -> int:
        """Insert if new; bump last_opened_at on existing. Returns row id."""
        now = _now()
        self.conn.execute(
            "INSERT INTO projects (path, name, created_at, last_opened_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET last_opened_at = excluded.last_opened_at",
            (path, name, now, now),
        )
        row = self.conn.execute(
            "SELECT id FROM projects WHERE path = ?", (path,)
        ).fetchone()
        return int(row["id"])

    def touch(self, path: str) -> None:
        self.conn.execute(
            "UPDATE projects SET last_opened_at = ? WHERE path = ?",
            (_now(), path),
        )

    def list(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM projects ORDER BY "
            "COALESCE(last_opened_at, created_at) DESC"
        ).fetchall()

    def forget(self, path: str) -> None:
        self.conn.execute("DELETE FROM projects WHERE path = ?", (path,))


class PipelinesRepo:
    """Saved pipelines (yaml text) keyed by user-chosen name."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, name: str, yaml_text: str) -> int:
        now = _now()
        self.conn.execute(
            "INSERT INTO pipelines (name, yaml_text, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET yaml_text=excluded.yaml_text, "
            "updated_at=excluded.updated_at",
            (name, yaml_text, now, now),
        )
        row = self.conn.execute(
            "SELECT id FROM pipelines WHERE name = ?", (name,)
        ).fetchone()
        return int(row["id"])

    def list(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM pipelines "
            "ORDER BY updated_at DESC"
        ).fetchall()

    def get(self, pipeline_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pipelines WHERE id = ?", (pipeline_id,)
        ).fetchone()

    def delete(self, pipeline_id: int) -> None:
        self.conn.execute("DELETE FROM pipelines WHERE id = ?", (pipeline_id,))


class PreferencesRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO preferences (key, value, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, _now()),
        )
