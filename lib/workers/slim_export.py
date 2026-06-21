# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Slim-export a project SQLite file.

A "slim" copy keeps only:

* every active scan's raw root node (the imported image), and
* every branch's `chosen_node_id` (the layout the user has selected for
  export — the same image a downstream PDF/markdown export would use).

All other images, thumbs, intermediate pipeline nodes, debug artifacts,
and orphaned OCR runs are dropped. The resulting file is then VACUUMed
so the on-disk size reflects the new content.

Used by:

* the Qt GUI's "Export slimmed project" button (``slim_export`` → copy),
* the GUI's "Slim-down current project" menu item (``slim_in_place`` →
  rewrites the live file once the project window has closed), and
* (eventually) a CLI flag — kept as a plain function so headless calls
  don't drag Qt in.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Optional


def slim_export(src_db: Path, dest_db: Path,
                *, progress_cb: Optional[callable] = None) -> dict:
    """Copy `src_db` → `dest_db`, then prune everything outside the slim
    set. Returns a stats dict with `kept_images`, `dropped_images`,
    `dropped_nodes`, `dropped_thumbs`, `dropped_debug`, `size_before`,
    `size_after`.

    Caller is expected to handle file-exists / overwrite confirmation;
    this function unconditionally overwrites `dest_db`.
    """
    src_db = Path(src_db).resolve()
    dest_db = Path(dest_db).resolve()
    if src_db == dest_db:
        raise ValueError("slim_export: src and dest must differ")
    if not src_db.exists():
        raise FileNotFoundError(f"slim_export: src not found: {src_db}")

    # Block writes against the live project while we copy. SQLite's
    # online backup API is the durable answer, but shutil.copy is enough
    # given the GUI's "stop pipeline first" convention.
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    if dest_db.exists():
        dest_db.unlink()
    shutil.copy2(src_db, dest_db)

    size_before = dest_db.stat().st_size
    if progress_cb:
        try:
            progress_cb("copied", size_before)
        except Exception:
            pass

    conn = sqlite3.connect(str(dest_db))
    conn.row_factory = sqlite3.Row
    try:
        counts = _prune_to_slim(conn)
    finally:
        conn.close()

    size_after = dest_db.stat().st_size
    return {**counts, "size_before": int(size_before),
            "size_after": int(size_after)}


def slim_in_place(db_path: Path, *,
                  progress_cb: Optional[callable] = None) -> dict:
    """Prune `db_path` *in place* to the slim keep-set, then VACUUM.

    Same pruning as :func:`slim_export` but no copy — the live project
    file is rewritten. The caller MUST ensure no other connection (chain
    workers, GUI persister) holds the DB open, or the VACUUM will fail
    with ``database is locked``. Returns the same stats dict.
    """
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"slim_in_place: not found: {db_path}")

    size_before = db_path.stat().st_size
    if progress_cb:
        try:
            progress_cb("start", size_before)
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        counts = _prune_to_slim(conn)
    finally:
        conn.close()

    size_after = db_path.stat().st_size
    return {**counts, "size_before": int(size_before),
            "size_after": int(size_after)}


def _prune_to_slim(conn: sqlite3.Connection) -> dict:
    """Prune an open connection down to the slim keep-set and VACUUM.

    Shared by :func:`slim_export` (on a copy) and :func:`slim_in_place`
    (on the live file). Returns the per-table count stats (no sizes —
    callers stamp those from the file).
    """
    # FK pragma OFF for the pruning window — we rewire parent_id +
    # terminal_node_id by hand and don't want cascade deletes wiping
    # the nodes we're trying to keep.
    conn.execute("PRAGMA foreign_keys = OFF")

    # ── 1. compute the keep-set ────────────────────────────────
    keep_node_ids: set[int] = set()
    # Roots of every active scan.
    for row in conn.execute(
        "SELECT root_node_id FROM scans "
        "WHERE deleted_at IS NULL AND root_node_id IS NOT NULL"
    ):
        keep_node_ids.add(int(row["root_node_id"]))
    # Plus each branch's chosen node.
    for row in conn.execute(
        "SELECT b.chosen_node_id FROM branches b "
        "JOIN scans s ON s.id = b.scan_id "
        "WHERE s.deleted_at IS NULL"
    ):
        keep_node_ids.add(int(row["chosen_node_id"]))

    # ── 2. reparent chosen → root so cascade-friendly delete works
    # branches table has no ON DELETE CASCADE on nodes, but nodes
    # cascade-delete their own subtree. If we deleted an intermediate
    # node whose subtree includes a chosen_node, the chosen would
    # disappear too — so we flatten the kept chain to root → chosen.
    for sid_row in conn.execute(
        "SELECT id, root_node_id FROM scans "
        "WHERE deleted_at IS NULL AND root_node_id IS NOT NULL"
    ):
        sid = int(sid_row["id"])
        root_id = int(sid_row["root_node_id"])
        for b in conn.execute(
            "SELECT chosen_node_id FROM branches WHERE scan_id = ?", (sid,)
        ):
            cid = int(b["chosen_node_id"])
            if cid == root_id:
                continue
            conn.execute(
                "UPDATE nodes SET parent_id = ?, is_branch_point = 0 "
                "WHERE id = ?", (root_id, cid),
            )
        # Roots can't keep is_branch_point=1 once intermediate splits
        # are gone — it would confuse can_step_back checks downstream.
        conn.execute(
            "UPDATE nodes SET is_branch_point = 0 WHERE id = ?", (root_id,),
        )

    # Collapse terminal = chosen so reset_to_leaf doesn't refer
    # to a node we're about to delete.
    conn.execute(
        "UPDATE branches SET terminal_node_id = chosen_node_id "
        "WHERE terminal_node_id != chosen_node_id"
    )

    # ── 3. delete every node not in the keep-set ──────────────
    ids_csv = ",".join(str(n) for n in keep_node_ids) or "NULL"
    dropped_nodes = conn.execute(
        f"SELECT COUNT(*) AS n FROM nodes WHERE id NOT IN ({ids_csv})"
    ).fetchone()["n"]
    conn.execute(f"DELETE FROM nodes WHERE id NOT IN ({ids_csv})")

    # ── 4. thumbs + debug_artifacts (drop wholesale) ──────────
    dropped_thumbs = conn.execute("SELECT COUNT(*) FROM thumbs").fetchone()[0]
    conn.execute("DELETE FROM thumbs")
    dropped_debug = conn.execute(
        "SELECT COUNT(*) FROM debug_artifacts"
    ).fetchone()[0]
    conn.execute("DELETE FROM debug_artifacts")

    # ── 5. drop images no kept node references ────────────────
    keep_img_rows = conn.execute(
        "SELECT DISTINCT image_id FROM nodes WHERE image_id IS NOT NULL"
    ).fetchall()
    keep_image_ids = {int(r["image_id"]) for r in keep_img_rows}
    img_csv = ",".join(str(i) for i in keep_image_ids) or "NULL"
    dropped_images = conn.execute(
        f"SELECT COUNT(*) AS n FROM images WHERE id NOT IN ({img_csv})"
    ).fetchone()["n"]
    conn.execute(f"DELETE FROM images WHERE id NOT IN ({img_csv})")
    kept_images = len(keep_image_ids)

    # ── 6. tidy OCR runs that referenced deleted nodes ────────
    # Kept (chosen / root) nodes retain their OCR — only runs pointing at
    # now-deleted intermediate nodes are dropped.
    try:
        conn.execute(
            f"DELETE FROM ocr_runs WHERE node_id NOT IN ({ids_csv})"
        )
    except sqlite3.OperationalError:
        # Older schemas may not have ocr_runs — skip silently.
        pass

    conn.commit()
    # Re-enable FK enforcement so the file behaves like any normal
    # project from here on.
    conn.execute("PRAGMA foreign_keys = ON")
    # VACUUM reclaims the slack space; otherwise the slim file is
    # the same size as the original despite the empty rows.
    conn.execute("VACUUM")

    return {
        "kept_images": kept_images,
        "dropped_images": int(dropped_images),
        "dropped_nodes": int(dropped_nodes),
        "dropped_thumbs": int(dropped_thumbs),
        "dropped_debug": int(dropped_debug),
    }


def default_slim_path(src_db: Path) -> Path:
    """Append a `-slim` suffix to the project filename, preserving the
    project extension (.agl or legacy .scanproj.sqlite)."""
    from lib.storage import PROJECT_EXT, LEGACY_PROJECT_EXT
    src = Path(src_db)
    name = src.name
    for ext in (LEGACY_PROJECT_EXT, PROJECT_EXT):
        if name.endswith(ext):
            stem = name[: -len(ext)]
            return src.with_name(f"{stem}-slim{ext}")
    return src.with_name(f"{src.stem}-slim{src.suffix}")
