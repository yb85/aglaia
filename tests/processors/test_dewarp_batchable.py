"""PageDewarper BatchableTrait: the split-out batched path (to_request →
solve → apply_result) is byte-identical to the inline process() on a real page.

CPU-gated: the inline solve and the batcher's solve_one both call the same
scipy+JAX optimiser, which is bitwise-reproducible only on CPU (GPU gradient
reductions aren't), so we assert exact equality under JAX-CPU."""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("jax")
import jax

from aglaia.ImageBuffer import ImageBuffer
from aglaia.processors.PageDewarper import PageDewarper, DewarpOption

FIX = Path(__file__).parent / "fixtures" / "dewarp_input_0.pkl"

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="byte-identical solve requires CPU determinism (set JAX_PLATFORMS=cpu)")


def _load_buf():
    m = pickle.load(open(FIX, "rb"))
    buf = cv2.imdecode(np.frombuffer(m["buf_png"], np.uint8), cv2.IMREAD_UNCHANGED)
    b = ImageBuffer(buf.copy(), m["type"], dpi=m["dpi"],
                    filestem=m["filestem"])
    b.branch_label = m["branch_label"]
    b.meta = dict(m["meta"])
    return b


def _dewarper():
    # Matches config/pipelines/book_curved_x2.yaml's PageDewarper, but forces
    # the jax backend (the batchable path).
    return PageDewarper(DewarpOption(
        backend="jax", sheet_model="cylindrical", twist=False,
        baseline_source="bottom", cubic_cost=0.0, focal_length=1.3))


def test_batched_path_matches_inline_process():
    d = _dewarper()

    out_inline = d.process(_load_buf())

    buf2 = _load_buf()
    item = d.to_request(buf2)
    assert item is not None, "cylindrical jax page should be batchable"
    assert set(item.payload) == {"dstpoints", "keypoint_index", "params",
                                 "model_dims", "flat_flip", "flat_weights"}
    params = d.make_batcher().solve_one(item.payload)
    out_trait = d.apply_result(buf2, params)

    assert out_trait.type == out_inline.type
    assert np.array_equal(np.asarray(out_trait.buffer),
                          np.asarray(out_inline.buffer)), \
        "batched-path dewarp output differs from inline process()"


def test_non_jax_backend_not_batchable():
    d = _dewarper()
    d.backend = "powell"                       # force the inline-only backend
    assert d.to_request(_load_buf()) is None
