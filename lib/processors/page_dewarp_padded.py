# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Padded-shape JAX optimiser drop-in for page_dewarp.

page_dewarp's stock JAX optimiser builds a fresh `@jax.jit` objective for
every (n_spans, n_points) combination. On Apple-Silicon CPU the XLA host
memory pool that backs each compiled program is *not* released by
`jax.clear_caches()`, so processing N pages with distinct span counts
leaks ~1 GB per dewarp until the worker is SIGKILLed at ~20 GB.

This module exposes `install()` which monkey-patches
`page_dewarp.optimise._jax._run_jax_lbfgsb` with a padded variant that
always traces on the same maximum shape. After the first compile every
subsequent call hits the JIT cache — no new XLA program, no pool growth.

Limits set by MAX_NSPANS / MAX_NPTS define the maximum problem size we
can absorb. Inputs larger than these limits fall through to the stock
optimiser (re-trace + leak, but functionally correct).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# Reasonable upper bounds for the project's pipeline. A typical book
# page at 150 dpi produces ~20 spans and ~300 keypoints. These caps
# leave large headroom while keeping the padded L-BFGS-B state small
# enough that per-iteration cost is close to native.
MAX_NSPANS = 80
MAX_NPTS = 2000


def _build_padded_optimiser():
    """Compile the padded objective once and return the JIT'd value+grad
    function. Defers the JAX import so this module is safe to load on
    platforms / environments without JAX."""
    import math

    import jax
    import jax.numpy as jnp
    from page_dewarp.options import cfg

    from lib.processors.sheet_models import (bspline_interior_basis,
                                             canonical_model,
                                             BSPLINE_MODELS,
                                             MODEL_FLAT_SPLINE,
                                             SPLINE_MODELS)

    focal_length = cfg.FOCAL_LENGTH
    shear_cost = cfg.SHEAR_COST
    cubic_cost = float(_CUBIC_COST)
    huber_delta = float(_HUBER_DELTA)
    model = canonical_model(_MODEL)
    n_modes = int(_N_MODES)
    twist = bool(_TWIST)
    grading = float(_KNOT_GRADING)
    is_spline = model in SPLINE_MODELS
    is_bspline = model in BSPLINE_MODELS
    is_flat = model == MODEL_FLAT_SPLINE
    ne = n_modes + 1 if is_spline else 0

    def _z_coords(xy_coords, pvec, model_dims, flat_args):
        x = xy_coords[:, 0]
        if is_spline:
            # Twist-model profile + linear-in-y twist; extras at the
            # padded pvec tail (see sheet_models.py). model_dims is a
            # runtime (2,) array — per-page values don't recompile.
            coeffs = jnp.clip(pvec[-ne:-1], -0.5, 0.5)
            # twist=False: γ baked to 0 → zero gradient on the tail slot.
            gamma = jnp.clip(pvec[-1], -4.0, 4.0) if twist else 0.0
            if is_bspline:
                # Clamped cubic B-spline basis — knots are python
                # constants, recursion unrolls to elementwise ops.
                t_raw = x / model_dims[0]
                if is_flat:
                    # flip ∈ {0, 1} runtime: binding always at t = 1.
                    t_raw = t_raw + flat_args[0] * (1.0 - 2.0 * t_raw)
                t = jnp.clip(t_raw, 0.0, 1.0)
                basis = bspline_interior_basis(t, n_modes, jnp,
                                               grading=grading)
                z = jnp.zeros_like(t)
                for k in range(n_modes):
                    z = z + coeffs[k] * basis[k]
            else:
                t = x * (math.pi / model_dims[0])
                z = jnp.zeros_like(t)
                for k in range(1, n_modes + 1):
                    z = z + coeffs[k - 1] * jnp.sin(k * t)
            eta = xy_coords[:, 1] / model_dims[1] - 0.5
            return (1.0 + gamma * eta) * z
        alpha = jnp.clip(pvec[6], -0.5, 0.5)
        beta = jnp.clip(pvec[7], -0.5, 0.5)
        return (alpha + beta) * x ** 3 + (-2.0 * alpha - beta) * x ** 2 + alpha * x

    def _project_xy_jax(xy_coords, pvec, model_dims, flat_args):
        z_coords = _z_coords(xy_coords, pvec, model_dims, flat_args)
        objpoints = jnp.column_stack([xy_coords, z_coords])

        rvec = pvec[0:3]
        tvec = pvec[3:6]
        theta = jnp.linalg.norm(rvec)
        theta_safe = jnp.where(theta < 1e-10, 1.0, theta)
        k = rvec / theta_safe
        K_mat = jnp.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
        )
        K_mat_sq = K_mat @ K_mat
        R_standard = (
            jnp.eye(3) + jnp.sin(theta) * K_mat + (1.0 - jnp.cos(theta)) * K_mat_sq
        )
        R_small = (
            jnp.eye(3) + theta * K_mat + 0.5 * theta * theta * K_mat_sq
        )
        R = jnp.where(theta < 1e-8, R_small, R_standard)
        transformed = (R @ objpoints.T).T + tvec
        z = transformed[:, 2]
        u = focal_length * transformed[:, 0] / z
        v = focal_length * transformed[:, 1] / z
        return jnp.column_stack([u, v])

    def _objective(pvec, dstpoints_flat, keypoint_index, mask, model_dims,
                   flat_args):
        xy_coords = pvec[keypoint_index]
        # Slot 0 of keypoint_index is the origin pin (original lib forces
        # xy_coords[0, :] = 0 unconditionally). Padded rows also resolve
        # to slot 0, so this single override pins them to (0, 0) and
        # the mask gates their residual to zero.
        xy_coords = xy_coords.at[0, :].set(0.0)
        proj = _project_xy_jax(xy_coords, pvec, model_dims, flat_args)
        # mask is (MAX_NPTS+1,) — zero for padded rows.
        r2 = jnp.sum((dstpoints_flat - proj) ** 2, axis=1)
        if huber_delta > 0.0:
            # Pseudo-Huber on per-keypoint residual norm; see
            # page_dewarp_mlx._make_objective for rationale.
            d2 = huber_delta * huber_delta
            per_pt = 2.0 * d2 * (jnp.sqrt(1.0 + r2 / d2) - 1.0)
        else:
            per_pt = r2
        error = jnp.sum(per_pt * mask)
        if shear_cost > 0.0:
            error = error + shear_cost * pvec[0] ** 2
        if cubic_cost > 0.0:
            if is_spline:
                # Bending energy — γ unpenalised (twist is amplitude
                # gradient, not curvature); see
                # sheet_models.spline_bending_energy for rationale.
                coeffs = jnp.clip(pvec[-ne:-1], -0.5, 0.5)
                if is_bspline:
                    ctrl = jnp.concatenate(
                        [jnp.zeros(1), coeffs, jnp.zeros(1)])
                    m2 = float(max(n_modes - 1, 1)) ** 2
                    d2 = ctrl[:-2] - 2.0 * ctrl[1:-1] + ctrl[2:]
                    reg = jnp.sum((m2 * d2) ** 2)
                else:
                    reg = jnp.array(0.0)
                    for k in range(1, n_modes + 1):
                        reg = reg + (k * k * coeffs[k - 1]) ** 2
                error = error + cubic_cost * reg
            else:
                # L2 on cubic slopes α (pvec[6]) and β (pvec[7]); see
                # page_dewarp_mlx._make_objective for rationale.
                error = error + cubic_cost * (pvec[6] ** 2 + pvec[7] ** 2)
        if is_flat:
            # Outer-flatness penalty Σ (w_i·λ)·c_i² — weights arrive
            # penalty-scaled at runtime (flat_args[1:]); zeros = off.
            coeffs = jnp.clip(pvec[-ne:-1], -0.5, 0.5)
            error = error + jnp.sum(flat_args[1:] * coeffs ** 2)
        return error

    # jit must wrap the grad transform (not the reverse): grad-of-jit
    # re-runs the linearization in Python around the pjit boundary on
    # every optimizer iteration instead of executing one fused XLA program.
    return jax.jit(jax.value_and_grad(_objective, argnums=0))


_VALUE_AND_GRAD = None
_CUBIC_COST = 0.0
_HUBER_DELTA = 0.0
# Sheet model (see lib/processors/sheet_models.py). Baked into the
# compiled objective; model page dims are a runtime input.
_MODEL = "cylindrical"
_N_MODES = 0
_TWIST = True
_MODEL_DIMS = (1.0, 1.0)
# flat_spline: grading is baked into the knot constants (cache-busting);
# flip + penalty-scaled weights are a runtime array (per-page A/B).
_KNOT_GRADING = 1.0
_FLAT_FLIP = False
_FLAT_WEIGHTS: tuple = ()
_CACHED_CONSTS = None


def set_cubic_cost(value: float) -> None:
    """Update the cubic-slope L2 weight used by the padded JAX
    objective. Invalidates the compiled fn so the next call rebuilds
    with the new constant baked in."""
    global _CUBIC_COST, _VALUE_AND_GRAD, _CACHED_CONSTS
    new = float(value)
    if new != _CUBIC_COST:
        _CUBIC_COST = new
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_huber_delta(value: float) -> None:
    """Update the pseudo-Huber scale used by the padded JAX objective.
    Same baked-constant cache semantics as set_cubic_cost."""
    global _HUBER_DELTA, _VALUE_AND_GRAD, _CACHED_CONSTS
    new = float(value)
    if new != _HUBER_DELTA:
        _HUBER_DELTA = new
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_sheet_model(model: str, n_modes: int, twist: bool = True) -> None:
    """Select the sheet model baked into the padded objective. Cache
    busts on change. twist=False bakes γ to 0 (pure cylinder with the
    chosen basis); the γ slot stays in the pvec tail."""
    from lib.processors.sheet_models import canonical_model
    global _MODEL, _N_MODES, _TWIST, _VALUE_AND_GRAD, _CACHED_CONSTS
    model = canonical_model(model)
    n_modes = int(n_modes)
    twist = bool(twist)
    if (model, n_modes, twist) != (_MODEL, _N_MODES, _TWIST):
        _MODEL = model
        _N_MODES = n_modes
        _TWIST = twist
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_model_dims(w: float, h: float) -> None:
    """Per-page model page dims for the spline parameterisation. Runtime
    input — no recompile."""
    global _MODEL_DIMS
    _MODEL_DIMS = (float(w), float(h))


def set_knot_grading(value: float) -> None:
    """flat_spline knot grading g ≥ 1 (1 = uniform). Baked into the
    compiled basis → cache busts on change."""
    global _KNOT_GRADING, _VALUE_AND_GRAD, _CACHED_CONSTS
    new = max(float(value), 1.0)
    if new != _KNOT_GRADING:
        _KNOT_GRADING = new
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_flat(flip: bool, weights=None) -> None:
    """Per-page flat_spline runtime inputs (binding-left flip + penalty-
    scaled flat weights). No recompile."""
    global _FLAT_FLIP, _FLAT_WEIGHTS
    _FLAT_FLIP = bool(flip)
    _FLAT_WEIGHTS = tuple(float(w) for w in (weights if weights is not None
                                             else ()))


def _n_extras() -> int:
    from lib.processors.sheet_models import is_spline_model
    return _N_MODES + 1 if is_spline_model(_MODEL) else 0


def _get_compiled():
    global _VALUE_AND_GRAD, _CACHED_CONSTS
    key = (_CUBIC_COST, _HUBER_DELTA, _MODEL, _N_MODES, _TWIST,
           _KNOT_GRADING)
    if _VALUE_AND_GRAD is None or _CACHED_CONSTS != key:
        _VALUE_AND_GRAD = _build_padded_optimiser()
        _CACHED_CONSTS = key
    return _VALUE_AND_GRAD


def _pad(dstpoints: np.ndarray, keypoint_index: np.ndarray,
         params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Pad inputs to fixed (MAX_NSPANS, MAX_NPTS) shape.

    page_dewarp parameter vector layout (see make_keypoint_index +
    project_keypoints in page_dewarp.keypoints):
        params[:8]                       transform + cubic coefficients
        params[8:8+nspans]               per-span Y reference
        params[8+nspans:8+nspans+npts]   per-keypoint X
    Total pvec size = 8 + nspans + npts.

    keypoint_index has shape (npts+1, 2):
        col 0: index into pvec for X (8+nspans + j)
        col 1: index into pvec for Y (8 + span_i)

    With a twist sheet model, `n_extras` model params [c_1…c_K, γ]
    sit at the pvec TAIL after the point-x block; they are carried through
    to the padded tail unchanged (the objective reads them via pvec[-ne:]).

    dstpoints has npts+1 rows — row 0 is the page-corner target for the
    origin-pinned keypoint, rows 1..npts the span points — exactly
    row-aligned with keypoint_index (this is how the stock unpadded
    objective consumes them). An earlier revision assumed npts rows and
    shifted everything by one: every projected keypoint was matched
    against the NEXT point's target, the corner constraint was dropped,
    and one unmasked garbage residual tied span params to the last
    point. Subtle systematic distortion, now fixed.

    Padded layout substitutes MAX_NSPANS / MAX_NPTS for the real counts.
    Indices into the point region (col 0) must shift by
    (MAX_NSPANS - real_nspans) so they hit the same padded slot.

    Returns (padded_params, padded_dstpoints, padded_keypoint_index,
    mask, real_nspans, real_npts) where real_npts EXCLUDES the corner row.
    """
    ne = _n_extras()
    n_rows = int(dstpoints.reshape(-1, 2).shape[0])  # npts + 1
    real_npts = n_rows - 1
    real_npvec = int(params.shape[0])
    real_nspans = real_npvec - 8 - real_npts - ne
    assert real_nspans >= 0, (
        f"derived nspans={real_nspans} < 0 (npvec={real_npvec}, "
        f"npts={real_npts}, extras={ne})"
    )

    max_pvec = 8 + MAX_NSPANS + MAX_NPTS + ne
    padded_pvec = np.zeros(max_pvec, dtype=np.float64)
    padded_pvec[:8] = params[:8]
    padded_pvec[8:8 + real_nspans] = params[8:8 + real_nspans]
    padded_pvec[8 + MAX_NSPANS:8 + MAX_NSPANS + real_npts] = (
        params[8 + real_nspans:8 + real_nspans + real_npts]
    )
    if ne:
        padded_pvec[-ne:] = params[-ne:]

    padded_dst = np.zeros((MAX_NPTS + 1, 2), dtype=np.float32)
    # Row-aligned with keypoint_index: row 0 = corner target for the
    # origin pin, rows 1..npts = span points.
    padded_dst[:n_rows] = dstpoints.reshape(-1, 2)

    # Default padded rows to a SAFE pair of slots: col 0 → first padded
    # point-x slot (always 0 to start), col 1 → first span-y slot. Both
    # are finite during the entire optimization, so the projection of a
    # padded row never produces NaN / inf even before the residual mask
    # zeroes it out. Defaulting to slot 0 (rvec[0]) instead caused
    # division-by-near-zero in the perspective divide → NaN gradients →
    # L-BFGS-B diverged → cv2.remap downstream blew SHRT_MAX.
    padded_ki = np.zeros((MAX_NPTS + 1, 2), dtype=np.int32)
    padded_ki[:, 0] = 8 + MAX_NSPANS
    padded_ki[:, 1] = 8
    real_rows = min(keypoint_index.shape[0], MAX_NPTS + 1)
    ki = keypoint_index[:real_rows].astype(np.int32).copy()
    # Point-slot indices (>= 8 + real_nspans) must shift by the
    # difference between MAX_NSPANS and real_nspans so they hit the
    # padded point region rather than the padded span region.
    point_mask = ki[:, 0] >= 8 + real_nspans
    ki[point_mask, 0] = ki[point_mask, 0] + (MAX_NSPANS - real_nspans)
    point_mask = ki[:, 1] >= 8 + real_nspans
    ki[point_mask, 1] = ki[point_mask, 1] + (MAX_NSPANS - real_nspans)
    padded_ki[:real_rows] = ki

    mask = np.zeros(MAX_NPTS + 1, dtype=np.float32)
    # Rows 0..npts are real residuals (row 0 = origin pin vs corner
    # target, exactly as in the stock objective); padded rows stay 0.
    mask[:n_rows] = 1.0
    return padded_pvec, padded_dst, padded_ki, mask, real_nspans, real_npts


def _unpad(result_x: np.ndarray, real_nspans: int, real_npts: int) -> np.ndarray:
    """Reverse the layout shift in _pad."""
    ne = _n_extras()
    out = np.zeros(8 + real_nspans + real_npts + ne, dtype=result_x.dtype)
    out[:8] = result_x[:8]
    out[8:8 + real_nspans] = result_x[8:8 + real_nspans]
    out[8 + real_nspans:8 + real_nspans + real_npts] = (
        result_x[8 + MAX_NSPANS:8 + MAX_NSPANS + real_npts]
    )
    if ne:
        out[-ne:] = result_x[-ne:]
    return out


def _run_jax_lbfgsb_padded(dstpoints: np.ndarray,
                           keypoint_index: np.ndarray,
                           params: np.ndarray):
    """Drop-in replacement for page_dewarp.optimise._jax._run_jax_lbfgsb.

    Falls back to the original implementation when the input exceeds the
    padded caps — correctness over performance.
    """
    import jax
    import jax.numpy as jnp
    from page_dewarp.options import cfg

    # dstpoints rows = npts + 1 (corner row included).
    real_npts = int(dstpoints.reshape(-1, 2).shape[0]) - 1
    real_nspans = int(params.shape[0]) - 8 - real_npts - _n_extras()

    if real_npts > MAX_NPTS or real_nspans > MAX_NSPANS:
        # NOTE: stock optimiser is cubic-only; twist-model inputs above
        # the padding caps would silently lose the model tail. Caps are
        # sized so this never triggers on real pages.
        from page_dewarp.optimise._jax import _run_jax_lbfgsb as _orig
        return _orig(dstpoints, keypoint_index, params)

    padded_pvec, padded_dst, padded_ki, mask, _, _ = _pad(
        dstpoints, keypoint_index, params
    )
    dst_j = jnp.array(padded_dst)
    ki_j = jnp.array(padded_ki, dtype=jnp.int32)
    mask_j = jnp.array(mask)
    dims_j = jnp.array(np.asarray(_MODEL_DIMS, dtype=np.float32))
    # [flip, w_1·λ … w_K·λ] — fixed (1 + K) shape per model config.
    flat_np = np.zeros(1 + _N_MODES, dtype=np.float32)
    flat_np[0] = 1.0 if _FLAT_FLIP else 0.0
    if _FLAT_WEIGHTS:
        flat_np[1:1 + len(_FLAT_WEIGHTS)] = _FLAT_WEIGHTS
    flat_j = jnp.array(flat_np)

    value_and_grad = _get_compiled()

    def obj_with_grad_np(p):
        p_jax = jnp.array(p)
        val, grad = value_and_grad(p_jax, dst_j, ki_j, mask_j, dims_j,
                                   flat_j)
        val_np = float(val)
        grad_np = np.array(grad, dtype=np.float64)
        if not np.isfinite(val_np):
            return 1e10, np.zeros_like(p)
        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
        return val_np, grad_np

    result = minimize(
        obj_with_grad_np,
        padded_pvec,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": cfg.OPT_MAX_ITER, "maxcor": cfg.MAX_CORR},
    )
    result.x = _unpad(result.x, real_nspans, real_npts)
    return result


_INSTALLED = False


def install() -> bool:
    """Replace page_dewarp.optimise._jax._run_jax_lbfgsb with the padded
    variant. Idempotent. Returns True on success."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from page_dewarp.optimise import _jax as _pd_jax_mod
        _pd_jax_mod._run_jax_lbfgsb = _run_jax_lbfgsb_padded
        _INSTALLED = True
        return True
    except Exception:
        return False
