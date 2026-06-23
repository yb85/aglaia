# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""End-to-end: a per-page `step_overrides` row makes the chain bypass a
processor with a PASSTHROUGH node (same image, meta.disabled, no replay
stamp) while downstream steps still run on the un-transformed image."""
import json
import multiprocessing
import queue
import time

import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.Binarizer import BinarizerOption
from aglaia.processors.SkewFinder import SkewFinderOption
from aglaia.workers.IntegratedProcessingChain import IntegratedProcessingChain
from aglaia.workers.chain_abstraction import SimpleChainElement
from aglaia.storage.db import open_db
from aglaia.storage.repo import (
    NodeRepo, PipelineRepo, ProjectRepo, ScanRepo, StepOverrideRepo,
)
from aglaia.storage.persister import Persister


@pytest.fixture(scope="module", autouse=True)
def _spawn_ctx():
    multiprocessing.set_start_method("spawn", force=True)


def _skewed_page(w: int = 600, h: int = 400) -> np.ndarray:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(80, h - 60, 40):
        cv2.putText(img, "Lorem ipsum dolor sit", (40, y), font, 0.7, (0, 0, 0), 2)
    # Rotate a touch so SkewFinder has something to correct (changes pixels
    # when enabled → its output image differs from the raw root).
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def _decode(blob) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)


def _drain_until(log_q, predicate, timeout=30.0):
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


def _seed(db_path):
    conn = open_db(db_path)
    ProjectRepo(conn).init("T", "t")
    pid = PipelineRepo(conn).upsert("name: t\npipeline: []\n", "t", step_count=2)
    raw = _skewed_page()
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
    return pid, scan_id, root_id, image_id, raw


def _run_chain(db_path, pid, scan_id, root_id, raw):
    log_q = multiprocessing.Queue()
    elements = [
        SimpleChainElement("SkewFinder",
                           SkewFinderOption(max_angle=8.0, min_angle=0.2,
                                            apply_rotation=True),
                           instance_name="01_skew"),
        SimpleChainElement("Binarizer", BinarizerOption(),
                           instance_name="02_bw"),
    ]
    chain = IntegratedProcessingChain(elements, num_workers=1,
                                       log_queue=log_q, db_path=str(db_path))
    chain.start()
    buf = ImageBuffer(raw.copy(), ImageType.COLOR, dpi=150.0,
                      filestem="t_001", scan_id=scan_id,
                      parent_node_id=root_id, pipeline_version_id=pid, depth=0)
    chain.enqueue(buf)
    _drain_until(
        log_q,
        lambda msgs: any(isinstance(m, tuple) and m and m[0] == "branch_ready"
                         for m in msgs),
        timeout=40.0,
    )
    chain.stop()


def test_disabled_step_emits_passthrough_node(tmp_path):
    db_path = tmp_path / "p.sqlite"
    pid, scan_id, root_id, image_id, raw = _seed(db_path)

    # Disable the SkewFinder step (step_idx=1) for this single-layout page.
    conn = open_db(db_path)
    StepOverrideRepo(conn).set(scan_id, "", 1, True)
    conn.commit()
    conn.close()

    _run_chain(db_path, pid, scan_id, root_id, raw)

    from aglaia.storage.repo import ImageRepo
    conn = open_db(db_path)
    nodes = {int(n["step_idx"]): n for n in NodeRepo(conn).by_scan(scan_id)}
    root_px = _decode(ImageRepo(conn).get(image_id)["blob"])
    skew_px = _decode(ImageRepo(conn).get(nodes[1]["image_id"])["blob"])
    conn.close()

    # step 1 = SkewFinder, disabled → passthrough: pixel-identical to the
    # raw root (a JPEG re-encode may land a new image row, but the pixels
    # must match — no deskew applied), flagged disabled, no replay stamp.
    skew = nodes[1]
    assert skew["processor_name"] == "SkewFinder"
    meta = json.loads(skew["meta_json"] or "{}")
    assert meta.get("disabled") is True
    assert "replay_kind" not in meta
    assert root_px.shape == skew_px.shape
    assert float(np.mean(np.abs(root_px.astype(int) - skew_px.astype(int)))) < 2.0

    # step 2 = Binarizer still ran on the un-deskewed image → fresh image,
    # and the disabled flag did NOT leak onto its node.
    bw = nodes[2]
    assert bw["processor_name"] == "Binarizer"
    assert int(bw["image_id"]) != int(image_id)
    assert json.loads(bw["meta_json"] or "{}").get("disabled") is not True


def test_enabled_step_runs_normally(tmp_path):
    """Control: no override → SkewFinder runs and produces a new image."""
    db_path = tmp_path / "p.sqlite"
    pid, scan_id, root_id, image_id, raw = _seed(db_path)

    _run_chain(db_path, pid, scan_id, root_id, raw)

    conn = open_db(db_path)
    nodes = {int(n["step_idx"]): n for n in NodeRepo(conn).by_scan(scan_id)}
    conn.close()

    skew = nodes[1]
    # Enabled skew rotates the page → output differs from the raw root, and
    # the node is NOT flagged disabled.
    assert int(skew["image_id"]) != int(image_id)
    meta = json.loads(skew["meta_json"] or "{}")
    assert meta.get("disabled") is not True
