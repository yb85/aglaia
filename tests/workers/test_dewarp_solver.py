"""DewarpSolver: requests accumulate, flush as batches, and route back keyed by
(worker_id, request_id) — the routing/flush logic of the GPU solver process,
exercised without multiprocessing. Uses the real dewarp fixtures."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optax")
pytest.importorskip("jax")

from aglaia.workers.dewarp_solver import DewarpSolver
from aglaia.processors.dewarp_batcher import DewarpBatcher

FIX = Path(__file__).parent.parent / "processors" / "fixtures"


def _payload(i):
    d = np.load(FIX / f"dewarp_prob_{i}.npz", allow_pickle=False)
    cfg = dict(model=str(d["model"]), n_modes=int(d["n_modes"]),
               twist=bool(d["twist"]), cubic_cost=float(d["cubic_cost"]),
               huber=float(d["huber"]), knot_grading=float(d["knot_grading"]))
    payload = {"dstpoints": d["dstpoints"], "keypoint_index": d["keypoint_index"],
               "params": d["params"], "model_dims": tuple(d["model_dims"].tolist()),
               "flat_flip": bool(d["flat_flip"]),
               "flat_weights": tuple(d["flat_weights"].tolist())}
    return cfg, payload


def test_solver_routes_results_by_worker_and_request():
    cfg, p0 = _payload(0)
    _, p1 = _payload(1)
    solver = DewarpSolver({**cfg, "iters": 300})
    bk0 = DewarpBatcher(**cfg).bucket_key_for(p0)

    # three submissions across two workers, all same bucket
    solver.submit("w0", "r0", bk0, p0, now_ms=0)
    solver.submit("w1", "r1", bk0, p1, now_ms=0)
    solver.submit("w0", "r2", bk0, p0, now_ms=0)
    assert solver.pending == 3

    out = solver.drain(final=True, now_ms=0)            # flush everything
    assert solver.pending == 0
    by_key = {(w, r): res for w, r, res in out}
    assert set(by_key) == {("w0", "r0"), ("w1", "r1"), ("w0", "r2")}
    for (w, r), res in by_key.items():
        src = p1 if r == "r1" else p0
        assert np.asarray(res).shape == src["params"].shape
        assert np.all(np.isfinite(res))


def test_solver_no_flush_until_timeout_or_full():
    cfg, p0 = _payload(0)
    solver = DewarpSolver({**cfg, "iters": 200})
    bk0 = DewarpBatcher(**cfg).bucket_key_for(p0)
    solver.submit("w0", "r0", bk0, p0, now_ms=0)
    assert solver.drain(now_ms=10) == []                # 1 item, 10ms < flush_ms(40)
    assert solver.pending == 1
    out = solver.drain(now_ms=100)                       # past flush_ms → drains
    assert [(w, r) for w, r, _ in out] == [("w0", "r0")]
