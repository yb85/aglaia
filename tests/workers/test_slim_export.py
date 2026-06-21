# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Slim-export / in-place slim-down regression tests.

Builds a tiny synthetic project (root → mid → leaf chain, chosen = mid),
plus a thumb, a debug artifact and an orphan image, then asserts both
the copy path (`slim_export`) and the in-place path (`slim_in_place`)
prune to the same keep-set: raw root + chosen node (and their images +
OCR), everything else gone.
"""

from __future__ import annotations

import io
import shutil
import sqlite3

from PIL import Image

from lib.storage.db import open_db
from lib.storage.repo import (
    BranchRepo, ImageRepo, NodeRepo, PipelineRepo, ProjectRepo, ScanRepo,
)
from lib.workers.slim_export import slim_export, slim_in_place


def _png(color: int = 0) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _build_project(path) -> dict:
    """Create a one-scan project with a 3-node chain; chosen = the middle
    node so the leaf is a prunable intermediate. Returns the kept ids."""
    conn = open_db(path)
    ProjectRepo(conn).init("Test", "test")
    pid = PipelineRepo(conn).upsert("name: s\npipeline: []\n", "s", step_count=0)
    img = ImageRepo(conn)
    nodes = NodeRepo(conn)
    scans = ScanRepo(conn)

    root_img = img.insert(_png(0), "PNG", "BW", 8, 8, 72.0)
    mid_img = img.insert(_png(50), "PNG", "BW", 8, 8, 72.0)
    leaf_img = img.insert(_png(100), "PNG", "BW", 8, 8, 72.0)
    orphan_img = img.insert(_png(200), "PNG", "BW", 8, 8, 72.0)  # unreferenced

    sid = scans.create("import", pid)

    def add(parent, idx, depth, stem, image_id):
        return nodes.insert(scan_id=sid, parent_id=parent, pipeline_version_id=pid,
                            step_idx=idx, step_name=f"{idx:02d}_s", processor_name="X",
                            branch_label=None, depth=depth, filestem=stem, image_id=image_id)

    root = add(None, 0, 0, "t_001", root_img)
    mid = add(root, 1, 1, "t_001a", mid_img)
    leaf = add(mid, 2, 2, "t_001b", leaf_img)
    scans.set_root(sid, root)

    # Branch: terminal = leaf, then move chosen back to mid.
    bid = BranchRepo(conn).upsert(sid, "", leaf)
    conn.execute("UPDATE branches SET chosen_node_id = ? WHERE id = ?", (mid, bid))

    # Thumb (keyed by image_id) + debug artifact (both dropped wholesale).
    conn.execute(
        "INSERT INTO thumbs (image_id, max_dim, width, height, blob) "
        "VALUES (?, 150, 8, 8, ?)", (leaf_img, _png(10)))
    conn.execute(
        "INSERT INTO debug_artifacts (node_id, label, image_id, created_at) "
        "VALUES (?, 'dbg', ?, '2026-01-01')", (leaf, leaf_img))

    # OCR runs on chosen (kept) and leaf (dropped).
    try:
        for ver, nid in enumerate((mid, leaf), start=1):
            conn.execute(
                "INSERT INTO ocr_runs (scan_id, node_id, engine, version, "
                "status, created_at) "
                "VALUES (?, ?, 'apple_vision', ?, 'done', '2026-01-01')",
                (sid, nid, ver))
        has_ocr = True
    except sqlite3.OperationalError:
        has_ocr = False

    conn.commit()
    conn.close()
    return {"root": root, "mid": mid, "leaf": leaf, "orphan_img": orphan_img,
            "has_ocr": has_ocr}


def _counts(path) -> dict:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    out = {}
    for t in ("nodes", "images", "thumbs", "debug_artifacts"):
        out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    node_ids = {r[0] for r in conn.execute("SELECT id FROM nodes")}
    try:
        out["ocr_runs"] = conn.execute("SELECT COUNT(*) FROM ocr_runs").fetchone()[0]
    except sqlite3.OperationalError:
        out["ocr_runs"] = None
    out["integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    out["_node_ids"] = node_ids
    return out


def test_slim_in_place_matches_export(tmp_path):
    src = tmp_path / "proj.sqlite"
    ids = _build_project(src)

    # copy path
    exported = tmp_path / "export.sqlite"
    stats_e = slim_export(src, exported)

    # in-place path on an independent copy
    inplace = tmp_path / "inplace.sqlite"
    shutil.copy2(src, inplace)
    stats_i = slim_in_place(inplace)

    ce, ci = _counts(exported), _counts(inplace)

    # Both keep exactly root + chosen(mid); leaf dropped.
    assert ci["_node_ids"] == {ids["root"], ids["mid"]}
    assert ce["_node_ids"] == ci["_node_ids"]
    assert ci["nodes"] == 2
    assert ci["images"] == 2          # leaf + orphan images pruned
    assert ci["thumbs"] == 0
    assert ci["debug_artifacts"] == 0
    assert ci["integrity"] == "ok"

    # Stats parity between the two code paths.
    for k in ("kept_images", "dropped_images", "dropped_nodes",
              "dropped_thumbs", "dropped_debug"):
        assert stats_e[k] == stats_i[k]
    assert stats_i["size_after"] <= stats_i["size_before"]


def test_slim_in_place_keeps_chosen_ocr(tmp_path):
    src = tmp_path / "proj.sqlite"
    ids = _build_project(src)
    if not ids["has_ocr"]:
        import pytest
        pytest.skip("schema has no ocr_runs table")

    slim_in_place(src)
    conn = sqlite3.connect(src)
    rows = conn.execute("SELECT node_id FROM ocr_runs").fetchall()
    conn.close()
    kept = {r[0] for r in rows}
    assert ids["mid"] in kept        # chosen node's OCR survives
    assert ids["leaf"] not in kept   # dropped node's OCR is gone


def test_slim_in_place_missing_file(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        slim_in_place(tmp_path / "nope.sqlite")
