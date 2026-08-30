# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Headless run completion (#64) — a slow page must not be abandoned.

`wait_for_completion` used to decide the run had settled from a SILENCE TIMER:
once every scan had emitted one `branch_ready` and no further branch event
arrived for `quiesce_s`, it returned and the caller stopped the chain. A page
whose step outlives that window was still being processed when the workers were
torn down — it produced no node, no error, and the run exited 0.

Observed twice on real corpora before the fix: 2 of 4 pages lost under the
`powell` backend (~75 s/page), and 2 of 12 lost with a 25 s dewarp retry.

Silence is not the same as finished. The chain knows whether work is in flight
(`is_idle`), so completion asks it instead of guessing.
"""
import queue
import threading
import time

import pytest

from aglaia.workers.headless import _LogDrainer


class _StubChain:
    """A chain that is busy for `busy_s`, then idle. Counts `is_idle` calls so
    a test can tell "asked and waited" from "never asked"."""

    def __init__(self, busy_s: float):
        self._until = time.monotonic() + busy_s
        self.calls = 0

    def is_idle(self) -> bool:
        self.calls += 1
        return time.monotonic() >= self._until


def _drainer_with_one_finished_scan(quiesce_s: float):
    """A drainer whose single expected scan has reported one branch — i.e. the
    exact state the old timer used to call 'done'."""
    d = _LogDrainer(queue.Queue(), total_expected=1)
    d.start()
    d._handle(("branch_ready", {"scan_id": 1, "branch_path": "B"}))
    return d


def test_completion_waits_for_the_chain_not_the_silence_timer():
    """The load-bearing case: branch events have gone quiet well past
    `quiesce_s`, but a worker is still mid-page. Returning here is what lost
    the page."""
    busy_s = 1.2
    chain = _StubChain(busy_s)
    d = _drainer_with_one_finished_scan(0.2)
    try:
        t0 = time.monotonic()
        d.wait_for_completion(timeout_s=10, quiesce_s=0.2, chain=chain)
        waited = time.monotonic() - t0
    finally:
        d.stop()
    assert chain.calls > 0, "completion never consulted the chain"
    assert waited >= busy_s, (
        f"returned after {waited:.2f}s while the chain was still busy for "
        f"{busy_s}s — the page in flight would have been killed")


def test_completion_returns_promptly_once_the_chain_is_idle():
    """The fix must not turn every run into a fixed-length wait."""
    chain = _StubChain(0.0)                     # idle immediately
    d = _drainer_with_one_finished_scan(0.2)
    try:
        t0 = time.monotonic()
        d.wait_for_completion(timeout_s=10, quiesce_s=0.2, chain=chain)
        waited = time.monotonic() - t0
    finally:
        d.stop()
    assert waited < 3.0, f"idle chain still took {waited:.2f}s to settle"


def test_a_momentary_idle_blip_does_not_end_the_run():
    """`is_idle` is best-effort — mp.Queue.empty() is approximate and a worker
    between two items reads as idle for an instant. One such reading must not
    end the run while work remains."""
    flips = iter([True, False, False, True, True, True, True, True, True])

    class _Blip:
        calls = 0

        def is_idle(self):
            _Blip.calls += 1
            try:
                return next(flips)
            except StopIteration:
                return True

    d = _drainer_with_one_finished_scan(0.05)
    try:
        d.wait_for_completion(timeout_s=10, quiesce_s=0.05, chain=_Blip())
    finally:
        d.stop()
    # Must have kept polling past the first True rather than returning on it.
    assert _Blip.calls >= 4, f"stopped after {_Blip.calls} checks"


def test_without_a_chain_the_old_timer_still_applies():
    """Callers that cannot supply a chain (or a chain without `is_idle`) keep
    the previous behaviour rather than hanging to the timeout."""
    d = _drainer_with_one_finished_scan(0.2)
    try:
        t0 = time.monotonic()
        d.wait_for_completion(timeout_s=10, quiesce_s=0.2, chain=None)
        waited = time.monotonic() - t0
    finally:
        d.stop()
    assert waited < 3.0


def test_a_chain_that_raises_does_not_hang_the_run():
    class _Broken:
        def is_idle(self):
            raise RuntimeError("manager gone")

    d = _drainer_with_one_finished_scan(0.2)
    try:
        t0 = time.monotonic()
        d.wait_for_completion(timeout_s=6, quiesce_s=0.2, chain=_Broken())
        waited = time.monotonic() - t0
    finally:
        d.stop()
    assert waited < 5.0, "a broken idle probe must degrade, not block"


# ── the signal completion now depends on, in a real spawn-worker chain ──

def test_chain_reports_busy_while_a_worker_is_mid_page(tmp_path):
    """`is_idle()` must go False while a page is being processed and True once
    it is done — it is now the thing that decides when a headless run ends.

    In-flight registration used to be conditional on the buffer carrying a
    `parent_node_id` (it is a resumable DB reference for crash recovery), so a
    worker could be busy while the chain read as idle. There is now an
    unconditional busy marker alongside it.
    """
    import multiprocessing
    import cv2
    import numpy as np

    from aglaia.ImageBuffer import ImageBuffer, ImageType
    from aglaia.processors.SkewFinder import SkewFinderOption
    from aglaia.storage.db import open_db
    from aglaia.storage.persister import Persister
    from aglaia.storage.repo import PipelineRepo, ProjectRepo, ScanRepo
    from aglaia.workers.chain_abstraction import SimpleChainElement
    from aglaia.workers.IntegratedProcessingChain import IntegratedProcessingChain

    multiprocessing.set_start_method("spawn", force=True)
    db_path = tmp_path / "proj.sqlite"
    conn = open_db(db_path)
    ProjectRepo(conn).init("Test", "test")
    pid = PipelineRepo(conn).upsert("name: stub\npipeline: []\n", "stub",
                                    step_count=1)
    # Big enough that the step takes long enough to observe.
    raw = np.full((2400, 2400, 3), 255, dtype=np.uint8)
    for y in range(120, 2300, 60):
        cv2.line(raw, (100, y), (2300, y + 30), (0, 0, 0), 5)
    persister = Persister(conn)
    image_id = persister.persist_image(raw, "COLOR", dpi=300.0)
    scan_id = ScanRepo(conn).create("import", pid, capture_dpi=300.0)
    root = persister.persist_node(
        scan_id=scan_id, parent_id=None, pipeline_version_id=pid, step_idx=0,
        step_name=None, processor_name=None, branch_label=None, depth=0,
        filestem="t_001", image_id=image_id)
    ScanRepo(conn).set_root(scan_id, root)
    conn.close()

    log_q = multiprocessing.Queue()
    el = SimpleChainElement("SkewFinder",
                            SkewFinderOption(max_angle=20, min_angle=0.5,
                                             apply_rotation=True),
                            instance_name="01_skew")
    chain = IntegratedProcessingChain([el], num_workers=1, log_queue=log_q,
                                      db_path=str(db_path))
    chain.start()
    try:
        assert chain.is_idle(), "a chain with no work should read idle"
        chain.enqueue(ImageBuffer(
            raw.copy(), ImageType.COLOR, dpi=300.0, path=None,
            filestem="t_001", scan_id=scan_id, parent_node_id=root,
            pipeline_version_id=pid, depth=0))

        saw_busy = False
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not chain.is_idle():
                saw_busy = True
                break
            time.sleep(0.005)
        assert saw_busy, "chain never reported busy while processing a page"

        # …and it must come back to idle, or a run would never terminate.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not chain.is_idle():
            time.sleep(0.05)
        assert chain.is_idle(), "chain never returned to idle"
    finally:
        chain.stop()
