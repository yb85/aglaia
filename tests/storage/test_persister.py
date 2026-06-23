# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import numpy as np

from aglaia.storage.repo import ScanRepo, ThumbRepo, ImageRepo
from aglaia.storage.persister import Persister


def test_persist_image_creates_image_without_eager_thumb(seeded_db):
    db, pid = seeded_db
    p = Persister(db)
    buf = np.full((64, 64, 3), 200, dtype=np.uint8)
    iid = p.persist_image(buf, "COLOR", dpi=72.0)
    img = ImageRepo(db).get(iid)
    assert img["format"] == "JPG"
    assert img["width"] == 64 and img["height"] == 64
    # Thumbs are lazy: nothing eagerly persisted on the hot path.
    assert ThumbRepo(db).get(iid, 256) is None


def test_persist_image_dedup_same_bytes(seeded_db):
    db, pid = seeded_db
    p = Persister(db)
    buf = np.zeros((32, 32), dtype=np.uint8)
    a = p.persist_image(buf, "BW", dpi=72.0)
    b = p.persist_image(buf, "BW", dpi=72.0)
    assert a == b


def test_persist_node_and_branch(seeded_db):
    db, pid = seeded_db
    p = Persister(db)
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    iid = p.persist_image(img, "COLOR", dpi=72.0)
    sid = ScanRepo(db).create("import", pid)
    root = p.persist_node(scan_id=sid, parent_id=None, pipeline_version_id=pid,
                          step_idx=0, step_name=None, processor_name=None,
                          branch_label=None, depth=0, filestem="t_001", image_id=iid)
    leaf = p.persist_node(scan_id=sid, parent_id=root, pipeline_version_id=pid,
                          step_idx=1, step_name="01_x", processor_name="X",
                          branch_label=None, depth=1, filestem="t_001a", image_id=iid)
    bid = p.upsert_branch(sid, "", leaf)
    row = db.execute("SELECT chosen_node_id, terminal_node_id FROM branches WHERE id=?", (bid,)).fetchone()
    assert row["chosen_node_id"] == leaf
    assert row["terminal_node_id"] == leaf
