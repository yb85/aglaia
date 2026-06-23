# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from aglaia.storage.db import open_db


def test_open_creates_schema(tmp_path):
    p = tmp_path / "x.sqlite"
    conn = open_db(p)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"project", "pipeline_versions", "calibrations", "images", "thumbs",
            "scans", "nodes", "branches", "debug_artifacts"}.issubset(tables)
    # integrity check
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_open_idempotent(tmp_path):
    p = tmp_path / "x.sqlite"
    open_db(p).close()
    conn = open_db(p)
    # second open succeeds, schema unchanged
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_trigger_marks_parent_not_leaf(seeded_db):
    from aglaia.storage.repo import ImageRepo, NodeRepo, ScanRepo
    db, pid = seeded_db
    img_id = ImageRepo(db).insert(b"\x00" * 16, "PNG", "BW", 4, 4, 72.0)
    scan_id = ScanRepo(db).create("import", pid)
    n = NodeRepo(db)
    root = n.insert(scan_id=scan_id, parent_id=None, pipeline_version_id=pid,
                    step_idx=0, step_name=None, processor_name=None,
                    branch_label=None, depth=0, filestem="t_001", image_id=img_id)
    assert n.get(root)["is_leaf"] == 1
    child = n.insert(scan_id=scan_id, parent_id=root, pipeline_version_id=pid,
                     step_idx=1, step_name="01_x", processor_name="X",
                     branch_label=None, depth=1, filestem="t_001b", image_id=img_id)
    assert n.get(root)["is_leaf"] == 0
    assert n.get(child)["is_leaf"] == 1
