# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Integration test: feed a small image through a one-step chain (SkewFinder),
verify DB ends with scan + nodes + branches.
"""
import multiprocessing
import queue
import time

import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.SkewFinder import SkewFinderOption
from aglaia.workers.IntegratedProcessingChain import IntegratedProcessingChain
from aglaia.workers.chain_abstraction import SimpleChainElement
from aglaia.storage.db import open_db
from aglaia.storage.repo import (
    BranchRepo, ImageRepo, NodeRepo, PipelineRepo,
    ProjectRepo, ScanRepo, ThumbRepo,
)
from aglaia.storage.persister import Persister


@pytest.fixture(scope="module")
def mp_ctx():
    multiprocessing.set_start_method("spawn", force=True)
    return multiprocessing


def _wait_for_branch(log_q: multiprocessing.Queue, timeout: float = 15.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = log_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if isinstance(msg, tuple) and msg and msg[0] == "branch_ready":
            return msg[1]
    return None


def test_one_step_chain_persists_branch(tmp_path, mp_ctx):
    db_path = tmp_path / "proj.sqlite"
    # Initial schema/seed in main process so chain workers see it on open.
    conn = open_db(db_path)
    ProjectRepo(conn).init("Test", "test")
    pid = PipelineRepo(conn).upsert("name: stub\npipeline: []\n", "stub", step_count=1)

    # Seed raw scan + root node
    persister = Persister(conn)
    raw = np.full((400, 400, 3), 255, dtype=np.uint8)
    import cv2
    cv2.line(raw, (50, 50), (350, 350), (0, 0, 0), 6)
    image_id = persister.persist_image(raw, "COLOR", dpi=72.0)
    scan_id = ScanRepo(conn).create("import", pid, capture_dpi=72.0)
    root_node_id = persister.persist_node(
        scan_id=scan_id, parent_id=None, pipeline_version_id=pid,
        step_idx=0, step_name=None, processor_name=None,
        branch_label=None, depth=0, filestem="t_001", image_id=image_id,
    )
    ScanRepo(conn).set_root(scan_id, root_node_id)
    conn.close()

    # Build chain: 1 worker, single SkewFinder step
    log_q = mp_ctx.Queue()
    skew_opts = SkewFinderOption(max_angle=20, min_angle=0.5, apply_rotation=True)
    el = SimpleChainElement("SkewFinder", skew_opts,
                            instance_name="01_skew")
    chain = IntegratedProcessingChain([el], num_workers=1,
                                       log_queue=log_q, db_path=str(db_path))
    chain.start()

    buf = ImageBuffer(raw.copy(), ImageType.COLOR, dpi=72.0,
                      path=None, filestem="t_001",
                      scan_id=scan_id, parent_node_id=root_node_id,
                      pipeline_version_id=pid, depth=0)
    chain.enqueue(buf)

    payload = _wait_for_branch(log_q, timeout=20)
    chain.stop()
    assert payload is not None, "branch_ready event never arrived"
    assert payload["scan_id"] == scan_id
    assert payload["branch_path"] == ""

    # Verify DB state
    conn = open_db(db_path)
    nodes = NodeRepo(conn).by_scan(scan_id)
    assert len(nodes) == 2  # raw (depth 0) + skew (depth 1)
    leaf = [n for n in nodes if n["depth"] == 1][0]
    assert leaf["step_name"] == "01_skew"
    assert leaf["processor_name"] == "SkewFinder"
    assert leaf["parent_id"] == root_node_id

    branches = BranchRepo(conn).by_scan(scan_id)
    assert len(branches) == 1
    assert branches[0]["chosen_node_id"] == leaf["id"]
    assert branches[0]["terminal_node_id"] == leaf["id"]

    # Image persisted; thumbs are lazy (built on first GUI request).
    img = ImageRepo(conn).get(leaf["image_id"])
    assert img is not None
    assert ThumbRepo(conn).get(leaf["image_id"], 256) is None
    from aglaia.storage.persister import make_thumb
    blob, w, h = make_thumb(bytes(img["blob"]), 256)
    ThumbRepo(conn).upsert(leaf["image_id"], 256, w, h, blob)
    assert ThumbRepo(conn).get(leaf["image_id"], 256) is not None
    conn.close()
