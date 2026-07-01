# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Optional

from aglaia.storage.db import execute_write


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ProjectRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM project WHERE id = 1").fetchone()

    def init(self, name: str, slug: str, notes: Optional[str] = None) -> sqlite3.Row:
        now = _now()
        self.conn.execute(
            "INSERT OR IGNORE INTO project (id, name, slug, created_at, updated_at, notes) "
            "VALUES (1, ?, ?, ?, ?, ?)",
            (name, slug, now, now, notes),
        )
        self.conn.execute(
            "UPDATE project SET name = ?, slug = ?, updated_at = ?, notes = COALESCE(?, notes) WHERE id = 1",
            (name, slug, now, notes),
        )
        return self.get()


# Mistral batch statuses that are terminal failures (no result will arrive).
_BATCH_FAILED = ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED")


class MistralBatchRepo:
    """Submitted Mistral batch OCR jobs for this project (table
    ``mistral_batch_jobs``). Drives the OCR card's pending state and the
    Mistral Jobs tab."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, job_id: str, *, input_file_id: Optional[str] = None,
            chunk: int = 0, chunks_total: int = 1,
            page_count: Optional[int] = None, status: Optional[str] = None,
            submitted_at: Optional[str] = None,
            run_ids: Optional[Iterable[int]] = None) -> None:
        runs_json = json.dumps(list(run_ids)) if run_ids is not None else None
        self.conn.execute(
            "INSERT OR REPLACE INTO mistral_batch_jobs "
            "(job_id, input_file_id, chunk, chunks_total, page_count, "
            " status, submitted_at, run_ids, imported_at, error_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
            "  (SELECT imported_at FROM mistral_batch_jobs WHERE job_id = ?), "
            "  (SELECT error_text  FROM mistral_batch_jobs WHERE job_id = ?))",
            (job_id, input_file_id, int(chunk), int(chunks_total), page_count,
             status, submitted_at or _now(), runs_json, job_id, job_id),
        )

    @staticmethod
    def run_ids_of(row: sqlite3.Row) -> list[int]:
        try:
            return [int(x) for x in json.loads(row["run_ids"] or "[]")]
        except Exception:
            return []

    def list_all(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM mistral_batch_jobs ORDER BY submitted_at DESC, "
            "rowid DESC"
        ).fetchall()

    def get(self, job_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM mistral_batch_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

    def set_status(self, job_id: str, status: str,
                   error_text: Optional[str] = None) -> None:
        self.conn.execute(
            "UPDATE mistral_batch_jobs SET status = ?, "
            "error_text = COALESCE(?, error_text) WHERE job_id = ?",
            (status, error_text, job_id),
        )

    def mark_imported(self, job_id: str) -> None:
        self.conn.execute(
            "UPDATE mistral_batch_jobs SET imported_at = ? WHERE job_id = ?",
            (_now(), job_id),
        )

    def delete(self, job_id: str) -> None:
        self.conn.execute(
            "DELETE FROM mistral_batch_jobs WHERE job_id = ?", (job_id,))

    def pending(self) -> list[sqlite3.Row]:
        """Jobs not yet applied to the project and not terminally failed —
        includes QUEUED/RUNNING and SUCCESS-awaiting-import."""
        placeholders = ",".join("?" * len(_BATCH_FAILED))
        return self.conn.execute(
            "SELECT * FROM mistral_batch_jobs "
            "WHERE imported_at IS NULL "
            f"  AND (status IS NULL OR status NOT IN ({placeholders})) "
            "ORDER BY submitted_at DESC", _BATCH_FAILED,
        ).fetchall()

    def has_pending(self) -> bool:
        return bool(self.pending())


class PipelineRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, yaml_text: str, name: Optional[str], step_count: int, make_active: bool = True) -> int:
        h = _sha256(yaml_text.encode())
        row = self.conn.execute(
            "SELECT id FROM pipeline_versions WHERE yaml_sha256 = ?", (h,)
        ).fetchone()
        if row:
            pid = row["id"]
        else:
            cur = self.conn.execute(
                "INSERT INTO pipeline_versions (yaml_text, yaml_sha256, name, step_count, created_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (yaml_text, h, name, step_count, _now()),
            )
            pid = cur.lastrowid
        if make_active:
            self.set_active(pid)
        return pid

    def set_active(self, pipeline_id: int) -> None:
        self.conn.execute("UPDATE pipeline_versions SET is_active = 0")
        self.conn.execute("UPDATE pipeline_versions SET is_active = 1 WHERE id = ?", (pipeline_id,))

    def get_active(self) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pipeline_versions WHERE is_active = 1 LIMIT 1"
        ).fetchone()

    def get(self, pipeline_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pipeline_versions WHERE id = ?", (pipeline_id,)
        ).fetchone()


class CalibrationRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, camera_matrix, dist_coeffs, dpi: float,
               resolution: Optional[tuple] = None, new_camera_matrix=None,
               sample_count: Optional[int] = None, make_active: bool = True) -> int:
        rw, rh = (None, None)
        if resolution:
            rh, rw = resolution  # current code stores (h, w)
        cur = self.conn.execute(
            "INSERT INTO calibrations (created_at, camera_matrix_json, dist_coeffs_json, "
            "new_camera_matrix_json, dpi, resolution_w, resolution_h, sample_count, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (
                _now(),
                json.dumps(_to_list(camera_matrix)),
                json.dumps(_to_list(dist_coeffs)),
                json.dumps(_to_list(new_camera_matrix)) if new_camera_matrix is not None else None,
                float(dpi),
                rw, rh,
                sample_count,
            ),
        )
        cid = cur.lastrowid
        if make_active:
            self.set_active(cid)
        return cid

    def set_active(self, calibration_id: int) -> None:
        self.conn.execute("UPDATE calibrations SET is_active = 0")
        self.conn.execute("UPDATE calibrations SET is_active = 1 WHERE id = ?", (calibration_id,))

    def get_active(self) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM calibrations WHERE is_active = 1 LIMIT 1"
        ).fetchone()


def _to_list(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


class ImageRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, blob: bytes, fmt: str, image_type: str,
               width: int, height: int, dpi: float) -> int:
        h = _sha256(blob)
        row = self.conn.execute("SELECT id FROM images WHERE sha256 = ?", (h,)).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO images (sha256, format, type, width, height, dpi, bytes, blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (h, fmt, image_type, width, height, float(dpi), len(blob), blob, _now()),
        )
        return cur.lastrowid

    def get(self, image_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()


class ThumbRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, image_id: int, max_dim: int, width: int, height: int, blob: bytes) -> int:
        row = self.conn.execute(
            "SELECT id FROM thumbs WHERE image_id = ? AND max_dim = ?", (image_id, max_dim)
        ).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO thumbs (image_id, max_dim, width, height, blob) VALUES (?, ?, ?, ?, ?)",
            (image_id, max_dim, width, height, blob),
        )
        return cur.lastrowid

    def get(self, image_id: int, max_dim: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM thumbs WHERE image_id = ? AND max_dim = ?", (image_id, max_dim)
        ).fetchone()


class ScanRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def next_idx(self) -> int:
        """Best-effort peek of the next idx for templating. NOT race-safe;
        use ScanRepo.create for the actual write."""
        row = self.conn.execute("SELECT COALESCE(MAX(idx), 0) AS m FROM scans").fetchone()
        return int(row["m"]) + 1

    def create(self, source: str, pipeline_version_id: int, *,
               source_ref: Optional[str] = None, transform: Optional[str] = None,
               capture_dpi: Optional[float] = None, calibration_id: Optional[int] = None) -> int:
        """
        Race-safe scan insert. idx is computed inline by the database, so two
        concurrent writers (different connections, WAL mode) cannot land on
        the same idx and trip the UNIQUE constraint. page_order is seeded
        from MAX + 1.0 — `reorder` then assigns the mean of neighbours when
        the user drags a card, so siblings never need a rewrite.
        """
        cur = execute_write(
            self.conn,
            "INSERT INTO scans (idx, source, source_ref, transform, capture_dpi, "
            "calibration_id, pipeline_version_id, created_at, page_order) "
            "VALUES ((SELECT COALESCE(MAX(idx), 0) + 1 FROM scans), "
            "?, ?, ?, ?, ?, ?, ?, "
            "(SELECT COALESCE(MAX(page_order), 0) + 1.0 FROM scans))",
            (source, source_ref, transform, capture_dpi, calibration_id,
             pipeline_version_id, _now()),
        )
        return cur.lastrowid

    def set_page_order(self, scan_id: int, page_order: float) -> None:
        self.conn.execute(
            "UPDATE scans SET page_order = ? WHERE id = ?",
            (float(page_order), scan_id),
        )

    def set_root(self, scan_id: int, root_node_id: int) -> None:
        self.conn.execute("UPDATE scans SET root_node_id = ? WHERE id = ?", (root_node_id, scan_id))

    def soft_delete(self, scan_id: int) -> None:
        self.conn.execute("UPDATE scans SET deleted_at = ? WHERE id = ?", (_now(), scan_id))

    def restore(self, scan_id: int) -> None:
        self.conn.execute("UPDATE scans SET deleted_at = NULL WHERE id = ?", (scan_id,))

    def get(self, scan_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()

    def list_active(self, *, newest_first: bool = True) -> list[sqlite3.Row]:
        order = "DESC" if newest_first else "ASC"
        # page_order is set on create + on drag-reorder. idx as tiebreaker.
        return self.conn.execute(
            f"SELECT * FROM scans WHERE deleted_at IS NULL "
            f"ORDER BY page_order {order}, idx {order}"
        ).fetchall()


class NodeRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, *, scan_id: int, parent_id: Optional[int], pipeline_version_id: int,
               step_idx: int, step_name: Optional[str], processor_name: Optional[str],
               branch_label: Optional[str], depth: int, filestem: str,
               image_id: Optional[int], status_int: int = 1, elapsed_ms: Optional[float] = None,
               meta: Optional[dict] = None, is_branch_point: bool = False) -> int:
        # UPSERT: SIGKILL retries replay from start_node_idx and may collide
        # on the (scan_id, filestem, step_idx) unique constraint. Overwriting
        # ensures downstream lookups see the resumed work, not a half-written row.
        cur = self.conn.execute(
            "INSERT INTO nodes (scan_id, parent_id, pipeline_version_id, step_idx, step_name, "
            "processor_name, branch_label, depth, filestem, is_leaf, is_branch_point, image_id, "
            "status_int, elapsed_ms, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (scan_id, filestem, step_idx) DO UPDATE SET "
            "  parent_id=excluded.parent_id, "
            "  pipeline_version_id=excluded.pipeline_version_id, "
            "  step_name=excluded.step_name, "
            "  processor_name=excluded.processor_name, "
            "  branch_label=excluded.branch_label, "
            "  depth=excluded.depth, "
            "  is_branch_point=excluded.is_branch_point, "
            "  image_id=excluded.image_id, "
            "  status_int=excluded.status_int, "
            "  elapsed_ms=excluded.elapsed_ms, "
            "  meta_json=excluded.meta_json, "
            "  created_at=excluded.created_at "
            "RETURNING id",
            (scan_id, parent_id, pipeline_version_id, step_idx, step_name,
             processor_name, branch_label, depth, filestem, int(bool(is_branch_point)),
             image_id, status_int, elapsed_ms,
             json.dumps(meta) if meta else None, _now()),
        )
        row = cur.fetchone()
        return row[0] if row else cur.lastrowid

    def get(self, node_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()

    def parent_of(self, node_id: int) -> Optional[sqlite3.Row]:
        node = self.get(node_id)
        if node is None or node["parent_id"] is None:
            return None
        return self.get(node["parent_id"])

    def children_of(self, node_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE parent_id = ? ORDER BY id ASC", (node_id,)
        ).fetchall()

    def subtree(self, node_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("""
            WITH RECURSIVE t(id) AS (
                SELECT id FROM nodes WHERE id = ?
                UNION ALL
                SELECT n.id FROM nodes n JOIN t ON n.parent_id = t.id
            )
            SELECT n.* FROM nodes n JOIN t ON n.id = t.id ORDER BY n.depth, n.id
        """, (node_id,)).fetchall()

    def by_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE scan_id = ? ORDER BY depth, id", (scan_id,)
        ).fetchall()

    def by_step_name(self, step_name: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE step_name = ? ORDER BY scan_id, branch_label", (step_name,)
        ).fetchall()

    def by_depth(self, depth: int, leaves_only: bool = True) -> list[sqlite3.Row]:
        q = "SELECT * FROM nodes WHERE depth = ?"
        if leaves_only:
            q += " AND is_leaf = 1"
        return self.conn.execute(q + " ORDER BY scan_id", (depth,)).fetchall()

    def delete_subtree(self, node_id: int, *, include_self: bool = False) -> int:
        ids = [r["id"] for r in self.subtree(node_id)]
        if not include_self and ids and ids[0] == node_id:
            ids = ids[1:]
        if not ids:
            return 0
        qs = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM nodes WHERE id IN ({qs})", ids)
        return len(ids)

    def mark_branch_point(self, node_id: int) -> None:
        self.conn.execute("UPDATE nodes SET is_branch_point = 1 WHERE id = ?", (node_id,))

    def siblings_count(self, node_id: int) -> int:
        node = self.get(node_id)
        if node is None or node["parent_id"] is None:
            return 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM nodes WHERE parent_id = ? AND id <> ?",
            (node["parent_id"], node_id),
        ).fetchone()
        return int(row["c"])


class BranchRepo:
    """
    Per-branch chosen-output override.
    Branch identified by (scan_id, branch_path); branch_path = "", "A", "B", "A.1", ...
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(self, scan_id: int, branch_path: str, terminal_node_id: int) -> int:
        row = self.conn.execute(
            "SELECT id FROM branches WHERE scan_id = ? AND branch_path = ?",
            (scan_id, branch_path),
        ).fetchone()
        now = _now()
        if row:
            self.conn.execute(
                "UPDATE branches SET terminal_node_id = ?, chosen_node_id = ?, updated_at = ? "
                "WHERE id = ?",
                (terminal_node_id, terminal_node_id, now, row["id"]),
            )
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO branches (scan_id, branch_path, terminal_node_id, chosen_node_id, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (scan_id, branch_path, terminal_node_id, terminal_node_id, now, now),
        )
        return cur.lastrowid

    def get(self, branch_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM branches WHERE id = ?", (branch_id,)).fetchone()

    def by_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM branches WHERE scan_id = ? ORDER BY branch_path ASC", (scan_id,)
        ).fetchall()

    # NOTE: per-page exit-stage navigation (step_back / step_forward /
    # reset_to_leaf) was removed with issue #68 — `chosen_node_id` now
    # always tracks the rerun terminal. Output is shaped by per-page
    # processor disable (`step_overrides`), not by moving the chosen node.

    def current_export_set(self) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT b.id AS branch_id, b.branch_path, b.scan_id,
                   n.id AS node_id, n.filestem, n.step_name, n.depth, n.image_id,
                   i.format, i.width, i.height
            FROM branches b
            JOIN nodes n  ON n.id = b.chosen_node_id
            JOIN images i ON i.id = n.image_id
            JOIN scans s  ON s.id = b.scan_id
            WHERE s.deleted_at IS NULL
              AND b.trashed_at IS NULL
            ORDER BY s.idx ASC, b.branch_path ASC
        """).fetchall()


class StepOverrideRepo:
    """Per-page-layout processor disable.

    A row `(scan_id, branch_path, step_idx)` with `disabled=1` means the
    chain skips that pipeline step for that layout — `run_pipeline` emits a
    passthrough node (same image, no `process()`, no replay stamp) instead.
    `branch_path=""` is the pre-split trunk (applies to every layout);
    "A"/"B" target one PageDetector layout. `step_idx` matches
    `nodes.step_idx` (the output index the processor would occupy).

    Enabling deletes the row; disabling upserts `disabled=1`. So a missing
    row == enabled, and `map_for_scan` is the worker's skip set.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def set(self, scan_id: int, branch_path: str, step_idx: int, disabled: bool) -> None:
        if disabled:
            self.conn.execute(
                "INSERT INTO step_overrides (scan_id, branch_path, step_idx, disabled, created_at) "
                "VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(scan_id, branch_path, step_idx) DO UPDATE SET disabled = 1",
                (scan_id, branch_path, step_idx, _now()),
            )
        else:
            self.conn.execute(
                "DELETE FROM step_overrides "
                "WHERE scan_id = ? AND branch_path = ? AND step_idx = ?",
                (scan_id, branch_path, step_idx),
            )

    def is_disabled(self, scan_id: int, branch_path: str, step_idx: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM step_overrides "
            "WHERE scan_id = ? AND branch_path = ? AND step_idx = ? AND disabled = 1",
            (scan_id, branch_path, step_idx),
        ).fetchone()
        return row is not None

    def map_for_scan(self, scan_id: int) -> set[tuple[str, int]]:
        """The scan's skip set: {(branch_path, step_idx), …} where disabled."""
        return {
            (r["branch_path"], int(r["step_idx"]))
            for r in self.conn.execute(
                "SELECT branch_path, step_idx FROM step_overrides "
                "WHERE scan_id = ? AND disabled = 1",
                (scan_id,),
            ).fetchall()
        }

    def clear_scan(self, scan_id: int) -> None:
        self.conn.execute("DELETE FROM step_overrides WHERE scan_id = ?", (scan_id,))


class OcrRepo:
    """OCR runs per branch.

    Run lifecycle: `start` (status=running) → `finish` (status=done|error).
    Versioning: per (scan_id, branch_path) monotonic from 1.
    Staleness is computed at read time — a run is stale when its node_id
    no longer matches the branch's chosen_node_id.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def next_version(self, scan_id: int, branch_path: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM ocr_runs "
            "WHERE scan_id = ? AND branch_path = ?",
            (scan_id, branch_path),
        ).fetchone()
        return int(row["v"]) + 1

    def start(self, *, scan_id: int, node_id: int, branch_path: str,
              engine: str, languages: list[str]) -> int:
        v = self.next_version(scan_id, branch_path)
        now = _now()
        cur = self.conn.execute(
            "INSERT INTO ocr_runs (scan_id, node_id, branch_path, engine, "
            "languages_json, version, status, started_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)",
            (scan_id, node_id, branch_path, engine,
             json.dumps(languages), v, now, now),
        )
        return cur.lastrowid

    def finish(self, run_id: int, result: dict) -> None:
        self.conn.execute(
            "UPDATE ocr_runs SET status = 'done', result_json = ?, "
            "finished_at = ? WHERE id = ?",
            (json.dumps(result), _now(), run_id),
        )
        # A fresh `done` run is always considered current — it ran
        # against whichever node was the chosen target at the time.
        # If `chosen_node_id` moves afterwards, `recompute_stale_for_scan`
        # will flip is_stale to 1.
        self.conn.execute(
            "UPDATE ocr_runs SET is_stale = 0 WHERE id = ?", (run_id,)
        )
        # Max ONE done layer per (branch, engine): drop superseded same-engine
        # done rows so a re-run *replaces* rather than accumulates. Other
        # engines' layers are untouched — kept + independently export-selectable.
        row = self.conn.execute(
            "SELECT scan_id, branch_path, engine FROM ocr_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is not None:
            self.conn.execute(
                "DELETE FROM ocr_runs WHERE scan_id = ? AND branch_path = ? "
                "AND engine = ? AND status = 'done' AND id != ?",
                (row["scan_id"], row["branch_path"], row["engine"], run_id),
            )

    def fail(self, run_id: int, err: str) -> None:
        self.conn.execute(
            "UPDATE ocr_runs SET status = 'error', error_text = ?, "
            "finished_at = ? WHERE id = ?",
            (err, _now(), run_id),
        )

    def latest_for_branch(self, scan_id: int, branch_path: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM ocr_runs WHERE scan_id = ? AND branch_path = ? "
            "AND status = 'done' ORDER BY version DESC LIMIT 1",
            (scan_id, branch_path),
        ).fetchone()

    def latest_by_scan(self, scan_id: int) -> list[sqlite3.Row]:
        """Latest done run per branch_path of a scan."""
        return self.conn.execute("""
            SELECT o.* FROM ocr_runs o
            JOIN (
                SELECT branch_path, MAX(version) AS v
                FROM ocr_runs WHERE scan_id = ? AND status = 'done'
                GROUP BY branch_path
            ) m ON m.branch_path = o.branch_path AND m.v = o.version
            WHERE o.scan_id = ?
        """, (scan_id, scan_id)).fetchall()

    def branch_status_map(self) -> dict[tuple[int, str], dict]:
        """Per (scan_id, branch_path): {'state': 'none'|'fresh'|'stale', 'run_id': int|None}.

        Reads the persisted `is_stale` column — single source of truth.
        `set_branch_chosen` and OCR-completion writes keep it current.
        """
        rows = self.conn.execute("""
            SELECT b.scan_id, b.branch_path,
                   o.id AS run_id, o.is_stale AS is_stale
            FROM branches b
            JOIN scans s ON s.id = b.scan_id
            LEFT JOIN (
                SELECT o.* FROM ocr_runs o
                JOIN (
                    SELECT scan_id, branch_path, MAX(version) AS v
                    FROM ocr_runs WHERE status = 'done'
                    GROUP BY scan_id, branch_path
                ) m ON m.scan_id = o.scan_id
                   AND m.branch_path = o.branch_path
                   AND m.v = o.version
            ) o ON o.scan_id = b.scan_id AND o.branch_path = b.branch_path
            WHERE s.deleted_at IS NULL
              AND b.trashed_at IS NULL
        """).fetchall()
        out: dict[tuple[int, str], dict] = {}
        for r in rows:
            key = (r["scan_id"], r["branch_path"])
            if r["run_id"] is None:
                out[key] = {"state": "none", "run_id": None}
            elif int(r["is_stale"] or 0):
                out[key] = {"state": "stale", "run_id": r["run_id"]}
            else:
                out[key] = {"state": "fresh", "run_id": r["run_id"]}
        return out

    def recompute_stale_for_scan(self, scan_id: int) -> None:
        """After `branches.chosen_node_id` moves for `scan_id`, flip
        `is_stale` on every matching ocr_run to reflect the new chosen
        target. Authoritative DB maintenance — call from any writer that
        touches `chosen_node_id`."""
        self.conn.execute("""
            UPDATE ocr_runs
            SET is_stale = CASE
                WHEN ocr_runs.node_id = (
                    SELECT b.chosen_node_id FROM branches b
                    WHERE b.scan_id = ocr_runs.scan_id
                      AND b.branch_path = ocr_runs.branch_path
                    LIMIT 1
                ) THEN 0 ELSE 1 END
            WHERE scan_id = ?
        """, (int(scan_id),))

    def mark_run_fresh(self, run_id: int) -> None:
        """Clear `is_stale` on a single run (used by the OCR worker when
        a `done` row lands at the branch's current chosen_node_id)."""
        self.conn.execute(
            "UPDATE ocr_runs SET is_stale = 0 WHERE id = ?", (int(run_id),)
        )

    def mark_stale_for_engine_switch(self, new_engine: str) -> int:
        """Flip every done OCR run with a different engine to
        ``is_stale = 1``.

        Triggered when the user picks a new OCR engine in OcrTab so the
        UI badges / Live-OCR / Run-OCR-default-mode all treat the
        existing rows as needing re-OCR. Returns the row count touched
        so callers can log / toast.
        """
        cur = self.conn.execute(
            "UPDATE ocr_runs SET is_stale = 1 "
            "WHERE status = 'done' AND engine != ?",
            (str(new_engine),),
        )
        return cur.rowcount or 0

    def branches_needing_ocr(self, *, include_stale: bool,
                             engine: Optional[str] = None) -> list[sqlite3.Row]:
        """Branches missing OCR — or stale, if include_stale.

        With ``engine`` given, "needs OCR" is evaluated **per engine**: a branch
        needs OCR only if it lacks a done (and non-stale, when include_stale) run
        FOR THAT ENGINE — so re-running a different engine re-OCRs the branch
        while keeping the other engine's layer. ``engine=None`` keeps the
        original cross-engine behaviour. Reads the persisted `is_stale` column
        (same source of truth as the UI)."""
        stale_clause = " AND o.is_stale = 0" if include_stale else ""
        eng_clause = " AND o.engine = ?" if engine else ""
        params: tuple = (engine,) if engine else ()
        sql = f"""
            SELECT b.id AS branch_id, b.scan_id, b.branch_path,
                   b.chosen_node_id, n.image_id, n.filestem
            FROM branches b
            JOIN nodes n ON n.id = b.chosen_node_id
            JOIN scans s ON s.id = b.scan_id
            WHERE s.deleted_at IS NULL
              AND b.trashed_at IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM ocr_runs o
                WHERE o.scan_id = b.scan_id
                  AND o.branch_path = b.branch_path
                  AND o.status = 'done'{stale_clause}{eng_clause}
            )
            ORDER BY s.idx ASC, b.branch_path ASC
        """
        return self.conn.execute(sql, params).fetchall()

    def available_ocr_layers(self, scan_id: Optional[int] = None) -> list[sqlite3.Row]:
        """Distinct OCR engines that have a ``done`` layer, **latest-generated
        first** — feeds the export 'OCR layer' selector. Rows:
        ``{engine, created_at, n_branches}``. ``scan_id=None`` = whole project."""
        where = "WHERE status = 'done'"
        params: tuple = ()
        if scan_id is not None:
            where += " AND scan_id = ?"
            params = (scan_id,)
        # Order by MAX(id) — monotonic insertion order = true generation order,
        # robust to created_at's seconds-resolution ties.
        return self.conn.execute(
            f"SELECT engine, MAX(created_at) AS created_at, "
            f"COUNT(DISTINCT branch_path) AS n_branches "
            f"FROM ocr_runs {where} GROUP BY engine ORDER BY MAX(id) DESC",
            params,
        ).fetchall()

    def all_branches(self) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT b.id AS branch_id, b.scan_id, b.branch_path,
                   b.chosen_node_id, n.image_id, n.filestem
            FROM branches b
            JOIN nodes n ON n.id = b.chosen_node_id
            JOIN scans s ON s.id = b.scan_id
            WHERE s.deleted_at IS NULL
              AND b.trashed_at IS NULL
            ORDER BY s.idx ASC, b.branch_path ASC
        """).fetchall()

    def get(self, run_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM ocr_runs WHERE id = ?", (run_id,)
        ).fetchone()


class DebugRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, node_id: int, label: str, image_id: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO debug_artifacts (node_id, label, image_id, created_at) VALUES (?, ?, ?, ?)",
            (node_id, label, image_id, _now()),
        )
        return cur.lastrowid
