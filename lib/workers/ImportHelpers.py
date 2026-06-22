# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Helpers for ingesting external inputs (PDF / image files) into a project DB
and enqueuing them on the processing chain.

Provided as reusable top-level functions so the Qt entry can reuse them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from lib.ImageBuffer import ImageBuffer, ImageType
from lib.storage.db import open_db
from lib.storage.persister import Persister
from lib.storage.repo import ScanRepo


def _persist_raw(db_path: str, pipeline_version_id: int, slug: str,
                 arr: np.ndarray, dpi: float, *,
                 source: str, source_ref: str) -> tuple[int, int, str, int, int]:
    """Insert scan + root node for a raw image, in ONE connection.
    Returns (scan_id, root_node_id, filestem, idx, image_id). idx + image_id
    are returned so callers don't reopen the DB to read them back —
    every open_db re-parses the whole schema, so on a 200-page import that
    saved hundreds of connections."""
    conn = open_db(db_path)
    try:
        persister = Persister(conn)
        scan_id = ScanRepo(conn).create(
            source, pipeline_version_id,
            source_ref=source_ref, capture_dpi=float(dpi),
        )
        image_id = persister.persist_image(arr, "COLOR", dpi=float(dpi))
        idx = int(ScanRepo(conn).get(scan_id)['idx'])
        filestem = f"{slug}_{idx:03d}"
        root_node_id = persister.persist_node(
            scan_id=scan_id, parent_id=None,
            pipeline_version_id=pipeline_version_id,
            step_idx=0, step_name=None, processor_name=None,
            branch_label=None, depth=0, filestem=filestem, image_id=image_id,
        )
        ScanRepo(conn).set_root(scan_id, root_node_id)
    finally:
        conn.close()
    return scan_id, root_node_id, filestem, idx, int(image_id)


def _embedded_dpi(path: Path) -> Optional[float]:
    """DPI declared in the image file (JPEG JFIF density / PNG pHYs / TIFF),
    or None when the file carries no DPI metadata."""
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            d = im.info.get("dpi")
    except Exception:
        return None
    if not d:
        return None
    x = float(d[0]) if isinstance(d, (tuple, list)) else float(d)
    return x if x > 0 else None


def enqueue_image_files(*, db_path: str, pipeline_version_id: int,
                        slug: str, chain, image_paths: Iterable[Path],
                        default_dpi: float = 300.0, force_dpi: bool = False,
                        progress_cb: Optional[Callable[[int, int], None]] = None,
                        log_queue=None):
    # Sort by filename so "001_..." precedes "002_..." regardless of OS
    # picker order. Case-insensitive to keep mixed-case names predictable.
    paths = sorted(image_paths, key=lambda p: Path(p).name.lower())
    for i, p in enumerate(paths, 1):
        arr = cv2.imread(str(p))
        if arr is None:
            continue
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        del arr
        # `default_dpi` fills only files with no embedded DPI, unless
        # force_dpi (the CLI `--input-dpi force:N`) overrides every input.
        embedded = None if force_dpi else _embedded_dpi(p)
        dpi = float(embedded) if embedded else float(default_dpi)
        scan_id, root_node_id, filestem, idx, raw_image_id = _persist_raw(
            db_path, pipeline_version_id, slug, rgb, dpi,
            source="import", source_ref=str(p),
        )
        # GUI hint: surface the raw immediately so the user sees the
        # scan appear before any worker finishes. idx + image_id come
        # straight from _persist_raw (no DB reopen).
        if log_queue is not None:
            try:
                log_queue.put(("scan_imported", {
                    "scan_id": scan_id,
                    "idx": idx,
                    "root_node_id": root_node_id,
                    "raw_image_id": raw_image_id,
                    "filestem": filestem,
                }))
            except Exception:
                pass
        buf = ImageBuffer(
            rgb, ImageType.COLOR, dpi=dpi,
            filestem=filestem, scan_id=scan_id,
            parent_node_id=root_node_id,
            pipeline_version_id=pipeline_version_id, depth=0,
        )
        # Backpressure: blocks while the worker pool has more than ~4× workers
        # waiting items. Prevents loading 200 raws into RAM at once.
        chain.enqueue(buf)
        del buf, rgb
        if progress_cb:
            progress_cb(i, len(paths))


def enqueue_pdf_files(*, db_path: str, pipeline_version_id: int,
                      slug: str, chain, pdf_paths: Iterable[Path],
                      render_dpi: float = 200.0,
                      progress_cb: Optional[Callable[[int, int], None]] = None,
                      log_queue=None):
    from lib.workers.pdf_extract import open_pdf

    pdfs = sorted(pdf_paths, key=lambda p: Path(p).name.lower())
    total_pages = 0
    page_counts: list[int] = []
    for p in pdfs:
        try:
            doc = open_pdf(p)
            page_counts.append(len(doc))
            total_pages += len(doc)
            doc.close()
        except Exception:
            page_counts.append(0)

    seen = 0
    for pdf, n_pages in zip(pdfs, page_counts):
        if n_pages == 0:
            continue
        from lib.workers.pdf_extract import render_page
        doc = open_pdf(pdf)
        try:
            for pno in range(n_pages):
                arr = render_page(doc, pno, render_dpi)
                scan_id, root_node_id, filestem, idx, raw_image_id = _persist_raw(
                    db_path, pipeline_version_id, slug, arr, render_dpi,
                    source="pdf", source_ref=f"{pdf}#{pno + 1}",
                )
                if log_queue is not None:
                    try:
                        log_queue.put(("scan_imported", {
                            "scan_id": scan_id,
                            "idx": idx,
                            "root_node_id": root_node_id,
                            "raw_image_id": raw_image_id,
                            "filestem": filestem,
                        }))
                    except Exception:
                        pass
                buf = ImageBuffer(
                    arr, ImageType.COLOR, dpi=float(render_dpi),
                    filestem=filestem, scan_id=scan_id,
                    parent_node_id=root_node_id,
                    pipeline_version_id=pipeline_version_id, depth=0,
                )
                chain.enqueue(buf)
                del buf, arr
                seen += 1
                if progress_cb:
                    progress_cb(seen, total_pages)
        finally:
            doc.close()


def reprocess_active_scans(*, db_path: str, pipeline_version_id: int,
                            chain, scan_ids: Optional[set[int]] = None) -> int:
    """Re-enqueue active scans' raw images with the given pipeline_version_id.
    Returns the number of scans re-enqueued. `scan_ids` (when given) limits
    the rerun to that subset — used by the Fix-input-DPI view.

    Toggles `PRAGMA foreign_keys = OFF` during the wipe: `branches.terminal_node_id`
    and `branches.chosen_node_id` reference `nodes(id)` without ON DELETE CASCADE,
    so a `DELETE FROM nodes` raises IntegrityError whenever a stale branches row
    still points at a node we're about to drop. Cross-scan references shouldn't
    exist by design — but the user has DBs where they do (workers wrote a
    branches row while a previous reprocess was mid-flight). FK-off + commit
    pattern matches `slim_export`.
    """
    from lib.storage.repo import NodeRepo, ImageRepo, BranchRepo
    conn = open_db(db_path)
    try:
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        scans = ScanRepo(conn).list_active(newest_first=False)
        node_repo = NodeRepo(conn)
        image_repo = ImageRepo(conn)
        branch_repo = BranchRepo(conn)
        count = 0
        try:
            for scan_row in scans:
                if scan_ids is not None and int(scan_row["id"]) not in scan_ids:
                    continue
                root_id = scan_row["root_node_id"]
                if root_id is None:
                    continue
                root = node_repo.get(root_id)
                if root is None or root["image_id"] is None:
                    continue
                img_row = image_repo.get(root["image_id"])
                if img_row is None:
                    continue
                # Wipe prior pipeline outputs for this scan so we don't trip
                # the (scan_id, filestem, step_idx) UNIQUE constraint when the
                # new pipeline emits identically-indexed nodes.
                conn.execute(
                    "DELETE FROM branches WHERE scan_id = ?", (scan_row["id"],),
                )
                # Re-anchor surviving OCR runs to the scan root before the
                # subtree delete cascades them away. After the rerun, the new
                # chosen_node_id won't equal root → `branch_status_map`
                # reports the run as "stale" (yellow badge) instead of "none"
                # (badge disappears). User can re-run OCR or accept.
                conn.execute(
                    "UPDATE ocr_runs SET node_id = ? "
                    "WHERE scan_id = ? AND node_id != ?",
                    (root_id, scan_row["id"], root_id),
                )
                # Reprocess invalidates any prior OCR — pipeline runs
                # from raw again, the chosen output WILL differ. Mark
                # every surviving ocr_run for this scan as stale so the
                # UI badges + bottom-bar count reflect reality
                # immediately (not only after the next chosen_node_id move).
                conn.execute(
                    "UPDATE ocr_runs SET is_stale = 1 WHERE scan_id = ?",
                    (scan_row["id"],),
                )
                # Image blobs the about-to-die subtree owns. The images
                # table is content-addressed (ImageRepo.insert dedups on
                # sha256), so without this GC the old blobs orphan forever
                # and a re-run silently re-adopts a stale row instead of
                # producing a clean one — i.e. "regenerate" would NOT
                # regenerate the images, only the nodes. Collect now, prune
                # after the node delete commits.
                stale_img_ids = [
                    r["image_id"] for r in conn.execute(
                        "SELECT DISTINCT image_id FROM nodes "
                        "WHERE scan_id = ? AND id != ? AND image_id IS NOT NULL",
                        (scan_row["id"], root_id),
                    ).fetchall()
                ]
                node_repo.delete_subtree(root_id, include_self=False)
                # Drop every image (and its thumbs) the subtree held that no
                # surviving node / debug artifact still references. The root's
                # raw image stays — root survives and still points at it.
                # thumbs has ON DELETE CASCADE but FK enforcement is OFF here,
                # so delete them explicitly.
                for iid in stale_img_ids:
                    if conn.execute(
                        "SELECT 1 FROM nodes WHERE image_id = ? LIMIT 1", (iid,)
                    ).fetchone():
                        continue
                    if conn.execute(
                        "SELECT 1 FROM debug_artifacts WHERE image_id = ? LIMIT 1",
                        (iid,),
                    ).fetchone():
                        continue
                    conn.execute("DELETE FROM thumbs WHERE image_id = ?", (iid,))
                    conn.execute("DELETE FROM images WHERE id = ?", (iid,))
                conn.commit()
                # Decode blob → numpy
                blob = bytes(img_row["blob"])
                arr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                dpi = float(img_row["dpi"] or 300.0)
                buf = ImageBuffer(
                    rgb, ImageType.COLOR, dpi=dpi,
                    filestem=root["filestem"], scan_id=scan_row["id"],
                    parent_node_id=root_id,
                    pipeline_version_id=pipeline_version_id, depth=0,
                )
                chain.enqueue(buf)
                count += 1
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
        return count
    finally:
        conn.close()


def reprocess_branch(*, db_path: str, pipeline_version_id: int, chain,
                     scan_id: int, branch_label: str) -> int:
    """Re-run only ONE page-branch of a scan, from its split point.

    Toggling a per-page step on page A used to reprocess the whole scan from
    raw — re-running PageDetector and recomputing the *other* page(s) for
    nothing. Here we resume from the branch's anchor (the shallowest node
    carrying ``branch_label`` — i.e. the PageDetector child for that page),
    wipe only that branch's downstream subtree, and re-enqueue a resume-ref so
    the worker re-runs just the post-split steps for this branch.

    Falls back to a whole-scan reprocess when the scan isn't actually split
    (≤1 branch) or the branch can't be isolated — never silently does nothing.
    Returns 1 on a branch rerun, else the fallback's count.
    """
    from lib.storage.repo import NodeRepo, ImageRepo

    def _fallback():
        return reprocess_active_scans(
            db_path=db_path, pipeline_version_id=pipeline_version_id,
            chain=chain, scan_ids={int(scan_id)})

    conn = open_db(db_path)
    try:
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        node_repo = NodeRepo(conn)
        image_repo = ImageRepo(conn)
        n_branches = conn.execute(
            "SELECT COUNT(*) AS n FROM branches WHERE scan_id = ?", (int(scan_id),)
        ).fetchone()["n"]
        anchor = conn.execute(
            "SELECT * FROM nodes WHERE scan_id = ? AND branch_label = ? "
            "ORDER BY step_idx ASC LIMIT 1", (int(scan_id), str(branch_label)),
        ).fetchone()
        # No real split, or branch not found / has no image → whole-scan path.
        if (n_branches is None or int(n_branches) <= 1 or anchor is None
                or anchor["image_id"] is None):
            conn.execute("PRAGMA foreign_keys = ON")
            conn.close()
            return _fallback()

        anchor_id = int(anchor["id"])
        anchor_step = int(anchor["step_idx"])
        # parent_stem = the pre-split (parent) node's stem, matching what the
        # normal PageDetector re-enqueue passes (out_buf.parent_stem). Used by
        # the GUI to route resumed events to the right branch card.
        parent_row = conn.execute(
            "SELECT filestem FROM nodes WHERE id = ?", (anchor["parent_id"],),
        ).fetchone() if anchor["parent_id"] is not None else None
        anchor_parent_stem = parent_row["filestem"] if parent_row else None
        # Images held by the subtree we're about to drop (anchor's descendants
        # for this branch). GC'd after the delete commits — same content-
        # addressed-orphan rationale as reprocess_active_scans.
        stale_img_ids = [
            r["image_id"] for r in conn.execute(
                "SELECT DISTINCT image_id FROM nodes WHERE scan_id = ? "
                "AND branch_label = ? AND id != ? AND image_id IS NOT NULL",
                (int(scan_id), str(branch_label), anchor_id),
            ).fetchall()
        ]
        # Drop this branch's row(s) (leaf 'A' and any nested 'A.*') + mark its
        # OCR stale; sibling branches' rows are untouched.
        conn.execute(
            "DELETE FROM branches WHERE scan_id = ? AND (branch_path = ? "
            "OR branch_path LIKE ?)",
            (int(scan_id), str(branch_label), f"{branch_label}.%"),
        )
        conn.execute(
            "UPDATE ocr_runs SET is_stale = 1 WHERE scan_id = ? AND node_id IN "
            "(SELECT id FROM nodes WHERE scan_id = ? AND branch_label = ?)",
            (int(scan_id), int(scan_id), str(branch_label)),
        )
        # Wipe the branch's downstream nodes (anchor's descendants); anchor and
        # all sibling branches survive.
        node_repo.delete_subtree(anchor_id, include_self=False)
        for iid in stale_img_ids:
            if conn.execute("SELECT 1 FROM nodes WHERE image_id = ? LIMIT 1",
                            (iid,)).fetchone():
                continue
            if conn.execute(
                "SELECT 1 FROM debug_artifacts WHERE image_id = ? LIMIT 1",
                (iid,)).fetchone():
                continue
            conn.execute("DELETE FROM thumbs WHERE image_id = ?", (iid,))
            conn.execute("DELETE FROM images WHERE id = ?", (iid,))
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
    finally:
        conn.close()

    # Resume the pipeline from the anchor for this branch only.
    chain.enqueue_resume(
        node_id=anchor_id, start_idx=anchor_step,
        branch_path=str(branch_label), scan_id=int(scan_id),
        parent_stem=anchor_parent_stem,
    )
    return 1


def scans_needing_catchup(db_path: str) -> list[int]:
    """Active scans with no branches row → never reached the pipeline objective.

    Objective = the image the GUI/headless reader would surface for the
    scan (the chosen_node_id in `branches`, defaulting to the terminal
    node). Intermediate-stage gaps don't count: the chain rebuilds them
    on demand from the raw root. A scan with at least one `branches` row
    has a serviceable objective."""
    from lib.storage.db import open_db as _open
    conn = _open(db_path)
    try:
        rows = conn.execute(
            "SELECT s.id AS id FROM scans s "
            "LEFT JOIN branches b ON b.scan_id = s.id "
            "WHERE s.deleted_at IS NULL AND s.root_node_id IS NOT NULL "
            "GROUP BY s.id "
            "HAVING COUNT(b.id) = 0 "
            "ORDER BY s.idx ASC"
        ).fetchall()
        return [int(r["id"]) for r in rows]
    finally:
        conn.close()


def catchup_active_scans(*, db_path: str, pipeline_version_id: int,
                          chain, force: bool = False) -> int:
    """Bring on-disk scans up to the current pipeline objective.

    `force=True`  → reprocess every active scan (== reprocess_active_scans).
    `force=False` → only enqueue scans with no branches row. Returns the
    number of scans actually enqueued."""
    if force:
        return reprocess_active_scans(
            db_path=db_path, pipeline_version_id=pipeline_version_id, chain=chain,
        )
    todo = set(scans_needing_catchup(db_path))
    if not todo:
        return 0
    from lib.storage.repo import NodeRepo, ImageRepo
    conn = open_db(db_path)
    try:
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        scans = ScanRepo(conn).list_active(newest_first=False)
        node_repo = NodeRepo(conn)
        image_repo = ImageRepo(conn)
        count = 0
        try:
            for scan_row in scans:
                sid = int(scan_row["id"])
                if sid not in todo:
                    continue
                root_id = scan_row["root_node_id"]
                if root_id is None:
                    continue
                root = node_repo.get(root_id)
                if root is None or root["image_id"] is None:
                    continue
                img_row = image_repo.get(root["image_id"])
                if img_row is None:
                    continue
                # Stale partial subtree (chain killed mid-flight, no branch row)
                # — wipe so UNIQUE (scan_id, filestem, step_idx) doesn't fire.
                # FK off (see `reprocess_active_scans` for rationale): drops
                # nodes even when an orphaned branches row still references
                # one of them.
                # Re-anchor OCR runs to root before the subtree wipe so the
                # cascade doesn't drop them — preserves the "stale" badge
                # after a catch-up rerun (vs. silently disappearing).
                conn.execute(
                    "UPDATE ocr_runs SET node_id = ? "
                    "WHERE scan_id = ? AND node_id != ?",
                    (root_id, sid, root_id),
                )
                # Catchup re-runs the pipeline → any prior OCR is stale.
                conn.execute(
                    "UPDATE ocr_runs SET is_stale = 1 WHERE scan_id = ?",
                    (sid,),
                )
                # Prune the orphaned step images too (see reprocess_active_scans):
                # content-addressed dedup would otherwise re-adopt stale blobs.
                stale_img_ids = [
                    r["image_id"] for r in conn.execute(
                        "SELECT DISTINCT image_id FROM nodes "
                        "WHERE scan_id = ? AND id != ? AND image_id IS NOT NULL",
                        (sid, root_id),
                    ).fetchall()
                ]
                node_repo.delete_subtree(root_id, include_self=False)
                for iid in stale_img_ids:
                    if conn.execute(
                        "SELECT 1 FROM nodes WHERE image_id = ? LIMIT 1", (iid,)
                    ).fetchone():
                        continue
                    if conn.execute(
                        "SELECT 1 FROM debug_artifacts WHERE image_id = ? LIMIT 1",
                        (iid,),
                    ).fetchone():
                        continue
                    conn.execute("DELETE FROM thumbs WHERE image_id = ?", (iid,))
                    conn.execute("DELETE FROM images WHERE id = ?", (iid,))
                conn.commit()
                blob = bytes(img_row["blob"])
                arr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                dpi = float(img_row["dpi"] or 300.0)
                buf = ImageBuffer(
                    rgb, ImageType.COLOR, dpi=dpi,
                    filestem=root["filestem"], scan_id=sid,
                    parent_node_id=root_id,
                    pipeline_version_id=pipeline_version_id, depth=0,
                )
                chain.enqueue(buf)
                count += 1
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
        return count
    finally:
        conn.close()
