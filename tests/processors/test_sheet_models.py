# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Sheet-model unit tests: B-spline basis correctness (vs scipy), boundary
conditions, derivative, model-name canonicalisation, bending energy."""
import numpy as np
import pytest

from aglaia.processors import sheet_models as sm


# ---------------------------------------------------------------------------
# Naming / dispatch
# ---------------------------------------------------------------------------

def test_canonical_model_legacy_alias():
    assert sm.canonical_model("spline_twist") == sm.MODEL_SINE_TWIST
    assert sm.canonical_model("sine_twist") == sm.MODEL_SINE_TWIST
    assert sm.canonical_model("bspline_twist") == sm.MODEL_BSPLINE_TWIST
    assert sm.canonical_model("flat_spline") == sm.MODEL_FLAT_SPLINE
    assert sm.canonical_model("flat-spline") == sm.MODEL_FLAT_SPLINE
    assert sm.canonical_model(None) == sm.MODEL_CYLINDRICAL
    assert sm.canonical_model("Cylindrical") == sm.MODEL_CYLINDRICAL


def test_n_extras_all_models():
    assert sm.n_extras("cylindrical", 4) == 0
    assert sm.n_extras("sine_twist", 4) == 5
    assert sm.n_extras("spline_twist", 4) == 5  # legacy alias
    assert sm.n_extras("bspline_twist", 6) == 7
    assert sm.n_extras("flat_spline", 6) == 7


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [2, 3, 4, 6, 8])
def test_bspline_basis_matches_scipy(K):
    from scipy.interpolate import BSpline

    knots = np.array(sm.bspline_knots(K))
    t = np.linspace(0.0, 0.999999, 257)
    ours = sm.bspline_interior_basis(t, K, np)
    assert len(ours) == K
    n_ctrl = K + 2
    for i in range(1, n_ctrl - 1):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        ref = BSpline(knots, c, 3)(t)
        np.testing.assert_allclose(ours[i - 1], ref, atol=1e-12)


@pytest.mark.parametrize("K", [2, 4, 6])
def test_bspline_boundary_zero_and_partition(K):
    t = np.array([0.0, 1.0])
    basis = sm.bspline_interior_basis(t, K, np)
    for b in basis:
        np.testing.assert_allclose(b, 0.0, atol=1e-12)
    # Interior basis + the two dropped endpoint basis form a partition of
    # unity; away from the clamped ends the endpoint basis vanish, so the
    # interior ones must sum to 1 there (uniform knots, m = K-1 segments:
    # endpoint basis support ends at the first/last interior knot).
    if K >= 4:
        m = K - 1
        t_mid = np.linspace(1.0 / m, 1.0 - 1.0 / m, 101)
        total = np.sum(sm.bspline_interior_basis(t_mid, K, np), axis=0)
        np.testing.assert_allclose(total, 1.0, atol=1e-12)


@pytest.mark.parametrize("K", [2, 4, 6])
def test_bspline_deriv_matches_numeric(K):
    rng = np.random.default_rng(3)
    coeffs = rng.normal(0, 0.2, K)
    t = np.linspace(0.02, 0.98, 401)
    db = sm.bspline_interior_basis_deriv(t, K, np)
    analytic = np.sum([c * d for c, d in zip(coeffs, db)], axis=0)
    eps = 1e-6
    bp = sm.bspline_interior_basis(t + eps, K, np)
    bm = sm.bspline_interior_basis(t - eps, K, np)
    numeric = np.sum([c * (p - m) / (2 * eps)
                      for c, p, m in zip(coeffs, bp, bm)], axis=0)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5)


# ---------------------------------------------------------------------------
# flat_spline: graded knots, flip, outer weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [3, 4, 6, 8])
def test_graded_knots_unit_grading_is_uniform(K):
    np.testing.assert_allclose(sm.bspline_knots(K, 1.0), sm.bspline_knots(K))


@pytest.mark.parametrize("K,g", [(4, 2.0), (6, 2.5), (8, 4.0)])
def test_graded_knots_monotone_and_dense_near_binding(K, g):
    knots = sm.bspline_knots(K, g)
    interior = knots[4:-4]
    assert all(b > a for a, b in zip(interior, interior[1:]))
    if len(interior) >= 2:
        # Knot spacing shrinks toward t = 1 (the binding edge).
        first_gap = interior[0]              # 0 → first interior knot
        last_gap = 1.0 - interior[-1]        # last interior knot → 1
        assert last_gap < first_gap


@pytest.mark.parametrize("K,g", [(4, 2.5), (6, 3.0)])
def test_graded_basis_matches_scipy(K, g):
    from scipy.interpolate import BSpline

    knots = np.array(sm.bspline_knots(K, g))
    t = np.linspace(0.0, 0.999999, 257)
    ours = sm.bspline_interior_basis(t, K, np, grading=g)
    n_ctrl = K + 2
    for i in range(1, n_ctrl - 1):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        ref = BSpline(knots, c, 3)(t)
        np.testing.assert_allclose(ours[i - 1], ref, atol=1e-12)


@pytest.mark.parametrize("K,g", [(4, 1.0), (6, 2.5)])
def test_flat_outer_weights_decrease_toward_binding(K, g):
    w = sm.flat_outer_weights(K, g)
    assert w.shape == (K,)
    assert np.all(w > 0.0)
    assert np.all(np.diff(w) < 0)        # binding at t = 1 → weights fall
    assert w[0] > 0.5                    # outer edge strongly weighted
    assert w[-1] < w[0] / 4              # gutter wall essentially free


def test_flat_spline_flip_mirrors_profile():
    coeffs = np.array([0.02, 0.01, 0.05, 0.2])  # heavy near binding
    w_m = 1.4
    x = np.linspace(0.0, w_m, 97)
    z = sm._profile(x, coeffs, w_m, model="flat_spline", grading=2.5)
    z_flip = sm._profile(w_m - x, coeffs, w_m, model="flat_spline",
                         grading=2.5, flip=True)
    np.testing.assert_allclose(z_flip, z, atol=1e-12)
    # Slope mirrors with opposite sign.
    s = sm._profile_slope(x, coeffs, w_m, model="flat_spline", grading=2.5)
    s_flip = sm._profile_slope(w_m - x, coeffs, w_m, model="flat_spline",
                               grading=2.5, flip=True)
    np.testing.assert_allclose(s_flip, -s, atol=1e-12)


def test_flat_spline_defaults_match_bspline_twist():
    # grading=1, flip=False → flat_spline IS bspline_twist.
    coeffs = np.array([0.12, -0.04, 0.015, 0.01])
    pv = _toy_pvec(coeffs, 0.25)
    md = (1.4, 1.9)
    xy = np.column_stack([np.linspace(0, 1.4, 20),
                          np.linspace(0, 1.9, 20)])
    a = sm.project_xy_model(xy, pv, model="bspline_twist", n_modes=4,
                            model_dims=md, focal_length=1.3)
    b = sm.project_xy_model(xy, pv, model="flat_spline", n_modes=4,
                            model_dims=md, focal_length=1.3)
    np.testing.assert_array_equal(a, b)


def test_bspline_profile_flat_outside_page():
    coeffs = np.array([0.1, -0.2, 0.15, 0.05])
    z = sm._profile(np.array([-0.3, 1.7]), coeffs, 1.4,
                    model=sm.MODEL_BSPLINE_TWIST)
    np.testing.assert_allclose(z, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Height field / projection plumbing
# ---------------------------------------------------------------------------

def _toy_pvec(coeffs, gamma, nspans=3, npts=5):
    head = np.array([0.02, -0.01, 0.005, -0.5, -0.8, 2.0, 0.0, 0.0])
    spans = np.linspace(0.2, 1.6, nspans)
    pts = np.linspace(0.1, 1.3, npts)
    return np.concatenate([head, spans, pts, coeffs, [gamma]])


@pytest.mark.parametrize("model", ["sine_twist", "bspline_twist"])
def test_spline_z_twist_modulation(model):
    coeffs = np.array([0.1, -0.05, 0.02, 0.0])
    md = (1.4, 1.9)
    x = np.linspace(0.05, 1.35, 64)
    z_mid = sm.spline_z(x, np.full_like(x, md[1] / 2), coeffs, 1.0, md,
                        model=model)
    z_top = sm.spline_z(x, np.zeros_like(x), coeffs, 1.0, md, model=model)
    z_bot = sm.spline_z(x, np.full_like(x, md[1]), coeffs, 1.0, md,
                        model=model)
    # γ=1, η=∓0.5 → factor 0.5 / 1.5 around the mid row.
    np.testing.assert_allclose(z_top, 0.5 * z_mid, atol=1e-12)
    np.testing.assert_allclose(z_bot, 1.5 * z_mid, atol=1e-12)


@pytest.mark.parametrize("model", ["sine_twist", "bspline_twist"])
def test_project_xy_model_runs_and_arclength_exceeds_chord(model):
    coeffs = np.array([0.12, -0.04, 0.015, 0.01])
    pv = _toy_pvec(coeffs, 0.25)
    md = (1.4, 1.9)
    xy = np.column_stack([np.linspace(0, 1.4, 30),
                          np.linspace(0, 1.9, 30)])
    proj = sm.project_xy_model(xy, pv, model=model, n_modes=4,
                               model_dims=md, focal_length=1.3)
    assert proj.shape == (30, 1, 2)
    assert np.all(np.isfinite(proj))
    xs, s = sm.arclength_x(pv, md[0], model=model, n_modes=4, model_dims=md)
    assert s[-1] > md[0]  # curved sheet is longer than its chord


def test_legacy_model_name_projects_identically():
    coeffs = np.array([0.12, -0.04, 0.015, 0.01])
    pv = _toy_pvec(coeffs, 0.25)
    md = (1.4, 1.9)
    xy = np.column_stack([np.linspace(0, 1.4, 20),
                          np.linspace(0, 1.9, 20)])
    a = sm.project_xy_model(xy, pv, model="spline_twist", n_modes=4,
                            model_dims=md, focal_length=1.3)
    b = sm.project_xy_model(xy, pv, model="sine_twist", n_modes=4,
                            model_dims=md, focal_length=1.3)
    np.testing.assert_array_equal(a, b)


def test_support_clamp_tangent_extension_bspline():
    coeffs = np.array([0.2, -0.1, 0.05, 0.02])
    md = (1.4, 1.9)
    support = (0.2, 1.2)
    x_out = np.array([0.05, 1.35])
    z = sm.spline_z(x_out, np.full_like(x_out, md[1] / 2), coeffs, 0.0, md,
                    support=support, model="bspline_twist")
    z_edge = sm.spline_z(np.array(support), np.full(2, md[1] / 2), coeffs,
                         0.0, md, support=support, model="bspline_twist")
    slope = sm.spline_dzdx_mid(np.array(support), coeffs, md,
                               support=support, model="bspline_twist")
    expected = z_edge + slope * (x_out - np.array(support))
    np.testing.assert_allclose(z, expected, atol=1e-12)


@pytest.mark.parametrize("model", ["sine_twist", "bspline_twist"])
def test_support_decay_flattens_margin(model):
    coeffs = np.array([0.3, 0.2, 0.1, 0.05])  # steep wall at left edge
    md = (1.4, 1.9)
    support = (0.3, 1.2)
    lam = 0.07
    y_mid = md[1] / 2

    def z_at(x, gamma=0.0, y=y_mid, decay=lam):
        return sm.spline_z(np.array([x]), np.array([y]), coeffs, gamma, md,
                           support=support, model=model,
                           support_decay=decay)[0]

    slope_edge = sm.spline_dzdx_mid(np.array([support[0]]), coeffs, md,
                                    support=support, model=model)[0]
    z_edge = z_at(support[0])
    # Page edges: exactly flat (cos² window hits 0 there) — nothing left
    # for the twist factor to fan, even at large γ.
    assert z_at(0.0) == pytest.approx(0.0, abs=1e-12)
    assert z_at(md[0]) == pytest.approx(0.0, abs=1e-12)
    assert z_at(0.0, gamma=3.0, y=0.4) == pytest.approx(0.0, abs=1e-12)
    # Bounded everywhere in the margin: |z| ≤ |z_edge| + |slope|·λ.
    xs = np.linspace(0.0, support[0], 64)
    z_marg = sm.spline_z(xs, np.full_like(xs, y_mid), coeffs, 0.0, md,
                         support=support, model=model, support_decay=lam)
    assert np.all(np.abs(z_marg) <= abs(z_edge) + abs(slope_edge) * lam
                  + 1e-12)
    # Pure tangent (legacy) is NOT bounded that way on a long margin.
    z_legacy = z_at(0.0, decay=None)
    assert z_legacy != pytest.approx(0.0, abs=1e-6)
    # C¹ at the support edge: numeric slope just outside ≈ edge slope
    # (bounded-tangent g′(0)=1, window derivative 0 at the edge).
    eps = 1e-6
    num_slope = (z_edge - z_at(support[0] - eps)) / eps
    assert num_slope == pytest.approx(slope_edge, abs=1e-3)
    # Descent spread over the whole margin: no point in the margin may
    # be steeper than ~the peak height over half the margin (the wiggle
    # regression was a local slope ≫ the wall slope).
    grad = np.abs(np.gradient(z_marg, xs))
    cap = (abs(z_edge) + abs(slope_edge) * lam) / (0.5 * support[0])
    assert grad.max() <= cap + abs(slope_edge)


@pytest.mark.parametrize("model", ["sine_twist", "bspline_twist"])
def test_support_y_freezes_twist_in_margins(model):
    coeffs = np.array([0.1, 0.05, 0.02, 0.01])
    md = (1.4, 1.9)
    support_y = (0.6, 1.3)
    x = np.full(2, 0.7)
    y_out = np.array([0.0, 1.9])     # top/bottom page edges
    y_edge = np.array(support_y)
    z_out = sm.spline_z(x, y_out, coeffs, 3.0, md, model=model,
                        support_y=support_y)
    z_edge = sm.spline_z(x, y_edge, coeffs, 3.0, md, model=model,
                         support_y=support_y)
    np.testing.assert_allclose(z_out, z_edge, atol=1e-12)
    # without the clamp, |γ|=3 sign-flips above the data
    z_free = sm.spline_z(x, y_out, coeffs, 3.0, md, model=model)
    assert z_free[0] < 0 < z_out[0]


# ---------------------------------------------------------------------------
# Bending energy
# ---------------------------------------------------------------------------

def test_bending_energy_sine_spectral_weighting():
    e1 = sm.spline_bending_energy(np.array([0.1, 0, 0, 0]), 0.0, "sine_twist")
    e4 = sm.spline_bending_energy(np.array([0, 0, 0, 0.1]), 0.0, "sine_twist")
    assert e4 / e1 == pytest.approx(4.0 ** 4)


def test_bending_energy_bspline_zero_for_straight_polygon():
    # Linear control polygon between zero-pinned ends is not curvature-free
    # (ends bend), but a constant-zero one is.
    assert sm.spline_bending_energy(np.zeros(5), 0.0, "bspline_twist") == 0.0
    e = sm.spline_bending_energy(np.array([0.0, 0.1, 0.0]), 0.0,
                                 "bspline_twist")
    assert e > 0.0


# ---------------------------------------------------------------------------
# Powell fallback ground-truth recovery (bspline)
# ---------------------------------------------------------------------------

def test_powell_recovers_bspline_ground_truth():
    pytest.importorskip("page_dewarp")
    from page_dewarp.keypoints import make_keypoint_index

    K = 4
    nspans = 6
    md = (1.4, 1.9)
    span_counts = [15] * nspans
    coeffs_t = np.array([0.10, 0.02, -0.03, 0.06])
    gamma_t = 0.3
    pv_true = np.concatenate([
        [0.05, -0.02, 0.01], [-0.6, -0.9, 2.2], [0, 0],
        np.linspace(0.1, 1.8, nspans),
        np.concatenate([np.linspace(0.05, 1.35, c) for c in span_counts]),
        coeffs_t, [gamma_t],
    ])
    ki = make_keypoint_index(span_counts)
    xy = pv_true[ki]
    xy[0, :] = 0
    dst = sm.project_xy_model(xy, pv_true, model="bspline_twist", n_modes=K,
                              model_dims=md, focal_length=1.3
                              ).reshape(-1, 1, 2)
    pv0 = pv_true.copy()
    pv0[-(K + 1):] = 0.0
    out = sm.optimise_params_spline_powell(
        dst, span_counts, pv0, model="bspline_twist", n_modes=K,
        model_dims=md, focal_length=1.3, huber_delta=0.005, maxiter=4000)
    err_c = np.abs(out[-(K + 1):-1] - coeffs_t).max()
    err_g = abs(out[-1] - gamma_t)
    assert err_c < 0.06
    assert err_g < 0.3


def test_powell_twist_off_pins_gamma():
    pytest.importorskip("page_dewarp")
    from page_dewarp.keypoints import make_keypoint_index

    K = 4
    nspans = 6
    md = (1.4, 1.9)
    span_counts = [15] * nspans
    coeffs_t = np.array([0.10, 0.06, 0.02, 0.01])
    pv_true = np.concatenate([
        [0.05, -0.02, 0.01], [-0.6, -0.9, 2.2], [0, 0],
        np.linspace(0.1, 1.8, nspans),
        np.concatenate([np.linspace(0.05, 1.35, c) for c in span_counts]),
        coeffs_t, [0.0],
    ])
    ki = make_keypoint_index(span_counts)
    xy = pv_true[ki]
    xy[0, :] = 0
    dst = sm.project_xy_model(xy, pv_true, model="bspline_twist", n_modes=K,
                              model_dims=md, focal_length=1.3
                              ).reshape(-1, 1, 2)
    pv0 = pv_true.copy()
    pv0[-(K + 1):] = 0.0
    out = sm.optimise_params_spline_powell(
        dst, span_counts, pv0, model="bspline_twist", n_modes=K,
        twist=False, model_dims=md, focal_length=1.3, huber_delta=0.005,
        maxiter=4000)
    assert out[-1] == 0.0
    assert np.abs(out[-(K + 1):-1] - coeffs_t).max() < 0.06


@pytest.mark.parametrize("flip", [False, True])
def test_powell_recovers_flat_spline_gutter(flip):
    """Flat page + gutter wall at the binding: graded-knot flat_spline
    with the outer penalty recovers the wall and keeps the field flat."""
    pytest.importorskip("page_dewarp")
    from page_dewarp.keypoints import make_keypoint_index

    K = 5
    g = 2.5
    nspans = 6
    md = (1.4, 1.9)
    span_counts = [15] * nspans
    # Control points ordered outer → binding (binding at basis t = 1):
    # flat field, steep wall at the gutter.
    coeffs_t = np.array([0.0, 0.0, 0.0, 0.08, 0.18])
    pv_true = np.concatenate([
        [0.05, -0.02, 0.01], [-0.6, -0.9, 2.2], [0, 0],
        np.linspace(0.1, 1.8, nspans),
        np.concatenate([np.linspace(0.05, 1.35, c) for c in span_counts]),
        coeffs_t, [0.0],
    ])
    ki = make_keypoint_index(span_counts)
    xy = pv_true[ki]
    xy[0, :] = 0
    dst = sm.project_xy_model(xy, pv_true, model="flat_spline", n_modes=K,
                              model_dims=md, focal_length=1.3,
                              grading=g, flip=flip).reshape(-1, 1, 2)
    pv0 = pv_true.copy()
    pv0[-(K + 1):] = 0.0
    out = sm.optimise_params_spline_powell(
        dst, span_counts, pv0, model="flat_spline", n_modes=K,
        twist=False, model_dims=md, focal_length=1.3, huber_delta=0.005,
        grading=g, flip=flip, flat_penalty=0.5, maxiter=4000)
    fitted = out[-(K + 1):-1]
    # Wall recovered (Powell is derivative-free over ~25 DOF — the
    # flip=True pose converges with a slightly damped wall amplitude).
    assert np.abs(fitted - coeffs_t).max() < 0.08
    # The penalised outer field stays flat.
    assert np.abs(fitted[:3]).max() < 0.03
