# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
End-to-end test: chain with PageDetector (heuristic backend) produces two
branches A, B from a synthetic two-column image. Verify DB has correct tree.
"""
import multiprocessing
import queue
import time

import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.PageDetector import PageOption
from aglaia.processors.SkewFinder import SkewFinderOption
from aglaia.workers.IntegratedProcessingChain import IntegratedProcessingChain
from aglaia.workers.chain_abstraction import SimpleChainElement
from aglaia.storage.db import open_db
from aglaia.storage.repo import BranchRepo, NodeRepo, PipelineRepo, ProjectRepo, ScanRepo
from aglaia.storage.persister import Persister


@pytest.fixture(scope="module", autouse=True)
def _spawn_ctx():
    multiprocessing.set_start_method("spawn", force=True)


def _two_col_page(w: int = 800, h: int = 600) -> np.ndarray:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(120, h - 80, 40):
        cv2.putText(img, "Lorem ipsum dolor",
                    (70, y), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Lorem ipsum dolor",
                    (470, y), font, 0.7, (0, 0, 0), 2)
    return img


def _drain_until(log_q, predicate, timeout=20.0):
    collected = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = log_q.get(timeout=0.5)
        except queue.Empty:
            continue
        collected.append(msg)
        if predicate(collected):
            return collected
    return collected


def test_layout_split_produces_two_branches(tmp_path):
    db_path = tmp_path / "p.sqlite"
    conn = open_db(db_path)
    ProjectRepo(conn).init("T", "t")
    pid = PipelineRepo(conn).upsert("name: t\npipeline: []\n", "t", step_count=2)

    raw = _two_col_page()
    persister = Persister(conn)
    image_id = persister.persist_image(raw, "COLOR", dpi=150.0)
    scan_id = ScanRepo(conn).create("import", pid, capture_dpi=150.0)
    root_id = persister.persist_node(
        scan_id=scan_id, parent_id=None, pipeline_version_id=pid,
        step_idx=0, step_name=None, processor_name=None,
        branch_label=None, depth=0, filestem="t_001", image_id=image_id,
    )
    ScanRepo(conn).set_root(scan_id, root_id)
    conn.close()

    log_q = multiprocessing.Queue()

    # 2-step chain: PageDetector (heuristic) → SkewFinder
    elements = [
        SimpleChainElement("PageDetector",
                           PageOption(margin_mm=2.0, max_pages=2,
                                        processing_dpi=None,
                                        backend="heuristic"),
                           instance_name="01_layouts"),
        SimpleChainElement("SkewFinder",
                           SkewFinderOption(max_angle=5.0, min_angle=0.5,
                                            apply_rotation=True),
                           instance_name="02_skew"),
    ]
    chain = IntegratedProcessingChain(elements, num_workers=1,
                                       log_queue=log_q, db_path=str(db_path))
    chain.start()

    buf = ImageBuffer(raw.copy(), ImageType.COLOR, dpi=150.0,
                      filestem="t_001", scan_id=scan_id,
                      parent_node_id=root_id, pipeline_version_id=pid, depth=0)
    chain.enqueue(buf)

    # Wait for two branch_ready events (one per branch A, B)
    def have_two_branches(msgs):
        return sum(1 for m in msgs if isinstance(m, tuple) and m and m[0] == "branch_ready") >= 2

    msgs = _drain_until(log_q, have_two_branches, timeout=30.0)
    chain.stop()

    branch_events = [m[1] for m in msgs if isinstance(m, tuple) and m and m[0] == "branch_ready"]
    assert len(branch_events) >= 2, f"expected 2 branches, got {len(branch_events)} (msgs={msgs})"
    paths = {e["branch_path"] for e in branch_events}
    assert paths == {"A", "B"}

    # DB verification
    conn = open_db(db_path)
    nodes = NodeRepo(conn).by_scan(scan_id)
    # raw (1) + 2 layout children (2) + 2 skew outputs (2) = 5
    assert len(nodes) == 5
    branches = BranchRepo(conn).by_scan(scan_id)
    assert {b["branch_path"] for b in branches} == {"A", "B"}

    # Each branch's chosen == its own terminal (independent per layout).
    # (Exit-stage step_back/forward navigation was removed with issue #68;
    # per-page output is now shaped by step_overrides instead.)
    a_branch = next(b for b in branches if b["branch_path"] == "A")
    b_branch = next(b for b in branches if b["branch_path"] == "B")
    assert a_branch["chosen_node_id"] == a_branch["terminal_node_id"]
    assert b_branch["chosen_node_id"] == b_branch["terminal_node_id"]
    assert a_branch["chosen_node_id"] != b_branch["chosen_node_id"]
    conn.close()
