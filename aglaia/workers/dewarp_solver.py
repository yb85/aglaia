# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Single GPU-owner solver process for batched page-dewarp (#28, B2).

The chain spawns ONE of these per `batcher_key` in the pipeline. Workers ship a
small solve `payload` (no image) for each parkable dewarp; this process groups
them with a `BatchEngine` and runs the GPU-batched optimiser, returning just the
solved param vector to the submitting worker. Heavy state (the page image + the
build context) never leaves the worker — only the payload comes here and only
the result goes back, so the IPC stays tiny.

`DewarpSolver` is the pure accumulate/flush/dispatch logic (BatchEngine +
DewarpBatcher + request-id→worker routing), unit-tested without multiprocessing.
`dewarp_solver_loop` is the thin spawn-process wrapper around it.

Request  (worker → solver): (worker_id, request_id, bucket_key, payload)
Result   (solver → worker): (request_id, params)   on result_queues[worker_id]
"""
from __future__ import annotations

import queue
import time
from typing import Any


class DewarpSolver:
    """Accumulate dewarp solve requests, flush as GPU batches (or CPU for
    small/tail groups), route results back to the submitting worker.

    Keyed by `(worker_id, request_id)` so `drain()` returns
    `(worker_id, request_id, params)` — the loop fans those onto per-worker
    result queues."""

    def __init__(self, batcher_cfg: dict, clock=None):
        from aglaia.processors.dewarp_batcher import DewarpBatcher
        from aglaia.processors.batching import BatchEngine
        self._batcher = DewarpBatcher(**(batcher_cfg or {}))
        self._engine = BatchEngine(self._batcher, clock=clock)

    def submit(self, worker_id: Any, request_id: Any,
               bucket_key: Any, payload: Any, now_ms=None) -> None:
        from aglaia.processors.batching import BatchItem
        self._engine.submit((worker_id, request_id),
                            BatchItem(bucket_key, payload), now_ms=now_ms)

    def drain(self, *, final: bool = False, now_ms=None) -> list:
        """Flush ready groups (or everything when `final`). Returns
        `[(worker_id, request_id, params), …]`."""
        ready = (self._engine.flush_all(now_ms=now_ms) if final
                 else self._engine.poll(now_ms=now_ms))
        return [(wid, rid, result) for (wid, rid), result in ready]

    @property
    def pending(self) -> int:
        return self._engine.pending


def dewarp_solver_loop(request_q, result_qs: dict, element,
                       stop_event, log_q=None, idle_sleep: float = 0.004) -> None:
    """Spawn-process entry: pump `request_q` → `DewarpSolver` → per-worker
    `result_qs`. Exits once `stop_event` is set AND all pending work is flushed
    (so no parked worker is left waiting on a result).

    `element` is the batchable pipeline step (a SimpleChainElement). The batcher
    config is resolved HERE (in the GPU-owner process) by instantiating the
    processor and calling make_batcher — so the heavy JAX import stays out of
    the GUI/parent process."""
    try:
        from aglaia.processors import registry as _reg
        proc = _reg.processor_classes()[element.processor_name](element.options)
        # Reproduce the worker's solve environment: PageDewarper.process()
        # stamps its whole AttrConfig (focal_length, shear_cost, OPT_MAX_ITER=
        # 2000, …) into page_dewarp's global cfg, which the objective + L-BFGS-B
        # read. Without the same stamp here the solver would optimise against
        # page_dewarp's DEFAULTS (focal 1.2, maxiter 600000) → a different fit.
        try:
            from page_dewarp.options import cfg as _lib_cfg
            for _k in proc.cfg.__struct_fields__:
                setattr(_lib_cfg, _k, getattr(proc.cfg, _k))
        except Exception:
            pass
        batcher = proc.make_batcher()
        batcher_cfg = {**batcher.cfg, "iters": batcher.iters}
        solver = DewarpSolver(batcher_cfg)
    except Exception as e:                       # pragma: no cover
        if log_q is not None:
            log_q.put(("error", f"dewarp solver init failed: {e}"))
        return

    def _route(results):
        for wid, rid, result in results:
            rq = result_qs.get(wid)
            if rq is not None:
                rq.put((rid, result))

    while True:
        stopping = stop_event.is_set()
        got = False
        # Drain a bounded burst of requests so a flood can't starve flushing.
        for _ in range(512):
            try:
                wid, rid, bucket, payload = request_q.get_nowait()
            except queue.Empty:
                break
            solver.submit(wid, rid, bucket, payload)
            got = True
        try:
            _route(solver.drain(final=stopping))
        except Exception as e:                   # one bad batch must not wedge the loop
            if log_q is not None:
                log_q.put(("error", f"dewarp solver batch failed: {e}"))
        if stopping and solver.pending == 0:
            break
        if not got and solver.pending == 0:
            time.sleep(idle_sleep)
