# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""flat_spline on the optimiser backends: compile + finite value/grad,
flip as a runtime input (affects flat_spline only), penalty wiring."""
import numpy as np
import pytest


K = 5
WEIGHTS = [1.0, 0.8, 0.3, 0.05, 0.0]


@pytest.fixture(scope="module")
def problem():
    pytest.importorskip("page_dewarp")
    from page_dewarp.keypoints import make_keypoint_index

    span_counts = [10] * 4
    ki = make_keypoint_index(span_counts)
    npts = sum(span_counts)
    rng = np.random.default_rng(0)
    pvec = rng.normal(0, 0.1,
                      8 + len(span_counts) + npts + K + 1).astype(np.float32)
    pvec[3:6] = [-0.5, -0.8, 2.0]
    dst = rng.normal(0, 0.3, (npts + 1, 2)).astype(np.float32)
    return pvec, dst, ki


def test_mlx_flat_spline_objective(problem):
    mx = pytest.importorskip("mlx.core")
    import aglaia.processors.page_dewarp_mlx as m

    pvec, dst, ki = problem
    vals = {}
    for model in ("flat_spline", "bspline_twist", "cylindrical"):
        m.set_sheet_model(model, K, False)
        m.set_knot_grading(2.5)
        m.set_model_dims(1.4, 1.9)
        for flip in (False, True):
            m.set_flat(flip, WEIGHTS)
            vg = m._get_value_and_grad(1.3, 40.0, 0.0, 0.005, m._MODEL,
                                       m._N_MODES, m._TWIST,
                                       m._KNOT_GRADING)
            flat_np = np.zeros(1 + K, dtype=np.float32)
            flat_np[0] = float(flip)
            flat_np[1:] = m._FLAT_WEIGHTS
            v, g = vg(mx.array(pvec), mx.array(dst.reshape(-1, 2)),
                      mx.array(ki.astype(np.int32)),
                      mx.array(np.array([1.4, 1.9], dtype=np.float32)),
                      mx.array(flat_np))
            mx.eval(v, g)
            g_tail = np.asarray(g)[-K - 1:-1]
            assert np.isfinite(float(v.item()))
            assert np.all(np.isfinite(g_tail))
            if model != "cylindrical":
                # Spline tail must receive gradient.
                assert np.any(g_tail != 0.0)
            vals[(model, flip)] = float(v.item())
    # Flip is consumed by flat_spline only.
    assert vals[("flat_spline", False)] != vals[("flat_spline", True)]
    assert vals[("bspline_twist", False)] == vals[("bspline_twist", True)]
    assert vals[("cylindrical", False)] == vals[("cylindrical", True)]


def test_mlx_flat_penalty_raises_objective(problem):
    mx = pytest.importorskip("mlx.core")
    import aglaia.processors.page_dewarp_mlx as m

    pvec, dst, ki = problem
    pvec = pvec.copy()
    pvec[-K - 1:-1] = 0.2  # non-zero control points → penalty must bite
    m.set_sheet_model("flat_spline", K, False)
    m.set_knot_grading(2.5)
    m.set_model_dims(1.4, 1.9)
    vg = m._get_value_and_grad(1.3, 40.0, 0.0, 0.005, m._MODEL, m._N_MODES,
                               m._TWIST, m._KNOT_GRADING)

    def value(weights):
        flat_np = np.zeros(1 + K, dtype=np.float32)
        flat_np[1:] = weights
        v, _ = vg(mx.array(pvec), mx.array(dst.reshape(-1, 2)),
                  mx.array(ki.astype(np.int32)),
                  mx.array(np.array([1.4, 1.9], dtype=np.float32)),
                  mx.array(flat_np))
        mx.eval(v)
        return float(v.item())

    v_off = value(np.zeros(K, dtype=np.float32))
    w = np.asarray(WEIGHTS, dtype=np.float32)
    v_on = value(w)
    expected = float(np.sum(w * 0.2 ** 2))
    assert v_on - v_off == pytest.approx(expected, rel=1e-4)


def test_padded_jax_flat_spline_runs(problem):
    pytest.importorskip("jax")
    import aglaia.processors.page_dewarp_padded as p
    from page_dewarp.options import cfg

    pvec, dst, ki = problem
    cfg.FOCAL_LENGTH = 1.3
    cfg.SHEAR_COST = 40.0
    p.set_cubic_cost(0.0)
    p.set_huber_delta(0.005)
    p.set_sheet_model("flat_spline", K, False)
    p.set_knot_grading(2.5)
    p.set_model_dims(1.4, 1.9)
    funs = {}
    for flip in (False, True):
        p.set_flat(flip, WEIGHTS)
        res = p._run_jax_lbfgsb_padded(dst.reshape(-1, 1, 2), ki, pvec)
        assert np.all(np.isfinite(res.x))
        funs[flip] = float(res.fun)
    assert funs[False] != funs[True]
