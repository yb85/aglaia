# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import io
import pytest
from PIL import Image

from aglaia.storage.repo import (
    BranchRepo, NodeRepo, ScanRepo, ImageRepo,
)


def _png():
    buf = io.BytesIO()
    Image.new("L", (4, 4), 0).save(buf, format="PNG")
    return buf.getvalue()


def _build_split_tree(db, pid):
    """
    Tree:
        root (depth=0, raw)
        └─ s1 (depth=1, linear step)
            └─ split (depth=2, branch point)
                ├─ A0 (depth=3, branch_label='A')
                │   └─ A1 (depth=4)
                │       └─ A2 (depth=5, leaf)
                └─ B0 (depth=3, branch_label='B')
                    └─ B1 (depth=4)
                        └─ B2 (depth=5, leaf)
    """
    iid = ImageRepo(db).insert(_png(), "PNG", "BW", 4, 4, 72.0)
    sid = ScanRepo(db).create("import", pid)
    n = NodeRepo(db)

    def add(parent, step_idx, depth, stem, label=None, is_branch_point=False):
        return n.insert(scan_id=sid, parent_id=parent, pipeline_version_id=pid,
                        step_idx=step_idx, step_name=f"{step_idx:02d}_s",
                        processor_name="X", branch_label=label, depth=depth,
                        filestem=stem, image_id=iid, is_branch_point=is_branch_point)

    root = add(None, 0, 0, "t_001")
    s1 = add(root, 1, 1, "t_001a")
    split = add(s1, 2, 2, "t_001b", is_branch_point=True)
    a0 = add(split, 3, 3, "t_001_A", label="A")
    a1 = add(a0, 4, 4, "t_001_A_b")
    a2 = add(a1, 5, 5, "t_001_A_c")
    b0 = add(split, 3, 3, "t_001_B", label="B")
    b1 = add(b0, 4, 4, "t_001_B_b")
    b2 = add(b1, 5, 5, "t_001_B_c")
    return sid, {"root": root, "s1": s1, "split": split,
                 "A0": a0, "A1": a1, "A2": a2,
                 "B0": b0, "B1": b1, "B2": b2}


def test_branch_upsert_initial(seeded_db):
    db, pid = seeded_db
    sid, ids = _build_split_tree(db, pid)
    b = BranchRepo(db)
    bid_a = b.upsert(sid, "A", terminal_node_id=ids["A2"])
    bid_b = b.upsert(sid, "B", terminal_node_id=ids["B2"])
    assert b.get(bid_a)["chosen_node_id"] == ids["A2"]
    assert b.get(bid_b)["chosen_node_id"] == ids["B2"]


# NOTE: exit-stage navigation (step_back / step_forward / reset_to_leaf)
# was removed with issue #68 — `chosen_node_id` now always tracks the
# rerun terminal. Per-page output is shaped by `step_overrides` (see
# tests/test_step_overrides.py), not by moving the chosen node. The old
# step_back/forward/reset tests were deleted with the feature.


def test_current_export_set(seeded_db):
    db, pid = seeded_db
    sid, ids = _build_split_tree(db, pid)
    b = BranchRepo(db)
    b.upsert(sid, "A", terminal_node_id=ids["A2"])
    b.upsert(sid, "B", terminal_node_id=ids["B2"])
    rows = b.current_export_set()
    assert {r["branch_path"] for r in rows} == {"A", "B"}
    paths = [r["node_id"] for r in rows]
    assert ids["A2"] in paths and ids["B2"] in paths
