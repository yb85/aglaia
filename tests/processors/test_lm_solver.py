"""LM dewarp solver (issue #59) — analytic Jacobians, arrowhead/Schur sparsity.

The load-bearing claim is that the closed-form Jacobian really is the
derivative of the objective the trust region evaluates: a wrong column would
still "converge", just to the wrong surface, silently. So the core test is a
finite-difference check of the assembled gradient across every variant of the
camera/objective (8- vs 10-param camera, L2 vs pseudo-Huber, with and without
the spine term and the spine-zone weights), followed by the end-to-end claim
that LM lands at an equal-or-better objective than Powell.
"""
import numpy as np
import pytest
from scipy.optimize import minimize as sp_minimize

from aglaia.processors import lm_solver as L
from aglaia.processors.lm_solver import (Objective, SpineCurl, keypoint_index,
                                         minimize, spine_weights)

PAGE_W, PAGE_H = 1.30, 0.95
FOCAL = 1.3


def _synthetic(n_cam=8, spine=None, nspans=9, npts=22, noise=2e-4, seed=0):
    """A page whose true surface IS the model: exact params + noisy targets."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0.06, PAGE_H - 0.06, nspans)
    xs = [np.linspace(0.05, PAGE_W - 0.05, npts) for _ in range(nspans)]
    span_counts = [len(x) for x in xs]
    head = [0.05, -0.08, 0.02, -0.65, -0.45, 1.9, 0.18, -0.09]
    if n_cam > 8:
        head += [0.012, 0.14]
    p_true = np.concatenate([head, ys, np.concatenate(xs)])
    probe = Objective(span_counts, np.zeros((sum(span_counts) + 1, 2)),
                      focal=FOCAL, n_cam=n_cam, spine=spine)
    ex, ey, _ = probe.residuals(p_true)      # targets 0 → residual == projection
    dst = np.stack([ex, ey], axis=1) + rng.normal(0, noise, (probe.n, 2))
    return span_counts, dst, p_true


def _assembled_gradient(obj, p):
    """g = Jᵀr exactly as `minimize` assembles it (half the true gradient —
    the LS convention where U = JᵀJ is half the Hessian)."""
    ex, ey, geo = obj.residuals(p)
    x, y, z, zp, _a, _b, R, pc0, pc1, inv_z, fz = geo
    w = obj.weights if obj.weights is not None else np.ones_like(ex)
    if obj.huber_delta > 0:
        w = w / np.sqrt(1.0 + w * (ex * ex + ey * ey) / obj.huber_delta ** 2)
    s_w = np.sqrt(w)
    dR = L._rodrigues_derivatives(p[0:3], R)
    jx3, jy3 = -fz * pc0 * inv_z, -fz * pc1 * inv_z

    def proj(v):
        return (s_w * (fz * v[..., 0] + jx3 * v[..., 2]),
                s_w * (fz * v[..., 1] + jy3 * v[..., 2]))

    exw, eyw = s_w * ex, s_w * ey
    g = np.zeros_like(p)
    p3 = np.stack([x, y, z], axis=-1)
    cols = {a: proj(p3 @ dR[a].T) for a in range(3)}
    zero = np.zeros_like(s_w)
    cols[3] = (s_w * fz, zero)
    cols[4] = (zero, s_w * fz)
    cols[5] = (s_w * jx3, s_w * jy3)
    aa = 1.0 if -L.CURL_CLIP < p[6] < L.CURL_CLIP else 0.0
    bb = 1.0 if -L.CURL_CLIP < p[7] < L.CURL_CLIP else 0.0
    cols[6] = proj(((x ** 3 - 2 * x ** 2 + x) * aa)[:, None] * R[:, 2])
    cols[7] = proj(((x ** 3 - x ** 2) * bb)[:, None] * R[:, 2])
    if obj.n_cam > 8:
        cols[8] = (s_w, zero)
        cols[9] = (zero, s_w)
    for a, (cx, cy) in cols.items():
        g[a] = np.sum(cx * exw + cy * eyw)
    g[0] += obj.shear_cost * p[0]
    if obj.cubic_cost:
        g[6] += obj.cubic_cost * p[6]
        g[7] += obj.cubic_cost * p[7]
    jyx, jyy = proj(np.broadcast_to(R[:, 1], (obj.n, 3)))
    jxx, jxy = proj(R[:, 0] + zp[:, None] * R[:, 2])
    idx = keypoint_index(obj.span_counts, obj.n_cam)
    for i in range(1, obj.n):
        g[idx[i, 1]] += jyx[i] * exw[i] + jyy[i] * eyw[i]
        g[idx[i, 0]] += jxx[i] * exw[i] + jxy[i] * eyw[i]
    return g


def _fd_gradient(f, p, h=1e-7):
    g = np.zeros_like(p)
    for i in range(p.size):
        a, b = p.copy(), p.copy()
        a[i] += h
        b[i] -= h
        g[i] = (f(a) - f(b)) / (2 * h)
    return g


@pytest.mark.parametrize("n_cam", [8, 10])
@pytest.mark.parametrize("huber", [0.0, 0.005])
@pytest.mark.parametrize("with_spine", [False, True])
def test_analytic_jacobian_matches_finite_differences(n_cam, huber, with_spine):
    spine = SpineCurl(0.05, 0.15 * PAGE_W, PAGE_W) if with_spine else None
    span_counts, dst, p_true = _synthetic(n_cam, spine)
    obj = Objective(span_counts, dst, focal=FOCAL, shear_cost=40.0,
                    cubic_cost=0.3, huber_delta=huber, n_cam=n_cam, spine=spine,
                    weights=spine_weights(dst[:, 0], 4.0, 0.25, spine_right=True))
    p = p_true + np.random.default_rng(7).normal(0, 1e-3, p_true.size)
    analytic = _assembled_gradient(obj, p)
    numeric = 0.5 * _fd_gradient(obj, p)      # objective grad is 2·Jᵀr
    rel = np.max(np.abs(analytic - numeric)) / max(np.max(np.abs(numeric)), 1e-12)
    assert rel < 1e-5, f"Jacobian mismatch, rel={rel:.2e}"


@pytest.mark.parametrize("n_cam", [8, 10])
def test_lm_reaches_powells_objective_far_faster(n_cam):
    span_counts, dst, p_true = _synthetic(n_cam)
    obj = Objective(span_counts, dst, focal=FOCAL, shear_cost=40.0, n_cam=n_cam)
    p0 = p_true.copy()
    p0[6] = p0[7] = 0.0                       # cold curl seed, as in production
    p0[:6] += np.random.default_rng(3).normal(0, 5e-3, 6)

    lm = minimize(obj, p0)
    powell = sp_minimize(obj, p0, method="Powell",
                         options={"maxiter": 2000, "xtol": 1e-6, "ftol": 1e-7})
    assert lm.fun <= powell.fun * 1.01
    # ~10-60 Jacobian passes replace Powell's 10⁴-10⁵ objective evaluations.
    assert lm.evaluations < powell.nfev / 100


def test_principal_point_is_recovered():
    """The 10-param camera must actually FIT (cx, cy), not just carry them."""
    span_counts, dst, p_true = _synthetic(n_cam=10)
    obj = Objective(span_counts, dst, focal=FOCAL, shear_cost=0.0, n_cam=10)
    p0 = p_true.copy()
    p0[8] = p0[9] = 0.0
    fit = minimize(obj, p0)
    # A pinhole with no principal point cannot explain the same data as well.
    obj8 = Objective(span_counts, dst, focal=FOCAL, shear_cost=0.0, n_cam=8)
    p8 = np.concatenate([p_true[:8], p_true[10:]])
    assert fit.fun < obj8(minimize(obj8, p8).x)


def test_freeze_curl_holds_the_shape_and_still_fits_pose():
    span_counts, dst, p_true = _synthetic()
    obj = Objective(span_counts, dst, focal=FOCAL, shear_cost=40.0)
    p0 = p_true.copy()
    p0[6], p0[7] = 0.25, -0.15                # deliberately wrong curl
    p0[3:6] += 0.02
    fit = minimize(obj, p0, freeze_curl=True)
    assert np.allclose(fit.x[6:8], p0[6:8]), "curl must not move"
    assert fit.fun < obj(p0), "pose must still improve"


def test_keypoint_index_matches_the_library_at_n_cam_8():
    pd_keypoints = pytest.importorskip("page_dewarp.keypoints")
    counts = [5, 3, 7, 2]
    assert np.array_equal(keypoint_index(counts, 8),
                          pd_keypoints.make_keypoint_index(counts))


def test_keypoint_index_shifts_with_the_camera_block():
    counts = [4, 6]
    idx8, idx10 = keypoint_index(counts, 8), keypoint_index(counts, 10)
    assert np.array_equal(idx10[1:], idx8[1:] + 2)
    assert np.array_equal(idx10[0], [0, 0])   # origin row


def test_spine_weights_ramp_from_the_binding():
    x = np.linspace(0.0, 1.0, 11)
    w = spine_weights(x, boost=4.0, zone_frac=0.25, spine_right=True)
    assert w[-1] == pytest.approx(4.0)        # AT the binding
    assert w[0] == pytest.approx(1.0)         # far edge
    assert np.all(np.diff(w) >= 0)            # monotone toward the spine
    assert np.all(w[:6] == 1.0)               # flat outside the 25% zone
    mirrored = spine_weights(x, 4.0, 0.25, spine_right=False)
    assert np.allclose(mirrored, w[::-1])
    assert spine_weights(x, 1.0, 0.25, True) is None   # boost 1 = off


def test_spine_curl_derivative_matches_finite_differences():
    s = SpineCurl(0.05, 0.2, 1.3)
    x = np.linspace(0.0, 1.3, 40)[1:-1]
    h = 1e-6
    assert np.allclose(s.dz(x), (s.z(x + h) - s.z(x - h)) / (2 * h), atol=1e-6)


def test_spine_curl_survives_a_dict_round_trip():
    s = SpineCurl(-0.02, 0.18, 1.21)
    back = SpineCurl.from_dict(s.as_dict())
    assert (back.gamma, back.s_x, back.x0) == (s.gamma, s.s_x, s.x0)
    assert SpineCurl.from_dict(None) is None


def test_empty_span_is_rejected_rather_than_silently_miscomputed():
    """reduceat on a zero-length span would fold it into its neighbour."""
    span_counts, dst, p_true = _synthetic()
    obj = Objective(span_counts, dst, focal=FOCAL)
    obj.span_counts = [span_counts[0], 0] + span_counts[1:]
    with pytest.raises(ValueError, match="keypoint per span"):
        minimize(obj, p_true)
