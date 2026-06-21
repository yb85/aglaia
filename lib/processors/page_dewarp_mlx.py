# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""MLX-backed drop-in for page_dewarp's JAX optimiser.

page_dewarp's stock `_run_jax_lbfgsb` uses JAX for value+grad and
scipy's L-BFGS-B for the outer loop. On Apple-Silicon CPU the JAX-CPU
XLA host-allocator leaks ~1 GB / call regardless of `clear_caches()`,
forcing watchdog SIGKILLs every 10-15 dewarps.

MLX is Apple's native ML framework. It uses unified memory with eager
deallocation through Python GC, so per-call buffers actually return to
the OS. This module mirrors `_run_jax_lbfgsb` but with MLX for the
value+grad path, keeping scipy.minimize(method="L-BFGS-B") as the
outer loop (mature, well-tuned, no porting cost).

Triggered via `install()` which monkey-patches
`page_dewarp.optimise._jax._run_jax_lbfgsb` exactly the way the padded
variant does. Either patch is active at any one time; the last one
installed wins.

MLX runs on the Metal GPU by default on Apple Silicon. Memory is
unified with the CPU, so the existing watchdog phys_footprint counter
captures the right total.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


_INSTALLED = False
_VALUE_AND_GRAD = None
# L2 penalty on cubic α (pvec[6]) and β (pvec[7]). Set by the host
# (PageDewarper.process) before each optimise call. Changing it busts
# the cached value+grad closure since the constant is JIT-baked.
_CUBIC_COST = 0.0
# Pseudo-Huber transition scale (pix2norm units) on the per-keypoint
# reprojection residual. 0 = plain squared-L2.
_HUBER_DELTA = 0.0
# Sheet model: "cylindrical" (stock cubic), "sine_twist" (sine-basis
# height profile + linear-in-y twist) or "bspline_twist" (clamped cubic
# B-spline profile + twist) — K+1 extra params at the pvec tail, see
# lib/processors/sheet_models.py. Baked into the compiled objective.
_MODEL = "cylindrical"
_N_MODES = 0
_TWIST = True
# Model page dims (W, H) for the spline parameterisation. Passed to the
# compiled fn as a runtime array — changing them per page does NOT bust
# the compile cache.
_MODEL_DIMS = (1.0, 1.0)
# flat_spline: knot grading is part of the traced basis (python-float
# knots) → JIT-baked, cache-busting. Flip (binding left/right) and the
# penalty-scaled flat weights alternate per page (A/B spreads) → runtime
# array [flip, w_1·λ … w_K·λ], no recompile.
_KNOT_GRADING = 1.0
_FLAT_FLIP = False
_FLAT_WEIGHTS: tuple = ()
_CACHED_CONSTS: tuple | None = None


def set_cubic_cost(value: float) -> None:
    """Update the cubic-slope L2 weight used by the MLX objective.
    Caller sets this before invoking optimise_params; the cache is
    invalidated so the new constant is JIT-baked on next call."""
    global _CUBIC_COST, _VALUE_AND_GRAD, _CACHED_CONSTS
    new = float(value)
    if new != _CUBIC_COST:
        _CUBIC_COST = new
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_huber_delta(value: float) -> None:
    """Update the pseudo-Huber scale used by the MLX objective. Same
    JIT-baked-constant cache semantics as set_cubic_cost."""
    global _HUBER_DELTA, _VALUE_AND_GRAD, _CACHED_CONSTS
    new = float(value)
    if new != _HUBER_DELTA:
        _HUBER_DELTA = new
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_sheet_model(model: str, n_modes: int, twist: bool = True) -> None:
    """Select the sheet model baked into the MLX objective. Cache busts
    on change (model structure is part of the traced graph). twist=False
    bakes γ to 0 (pure cylinder with the chosen basis); the γ slot stays
    in the pvec tail."""
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
    input — no cache invalidation."""
    global _MODEL_DIMS
    _MODEL_DIMS = (float(w), float(h))


def set_knot_grading(value: float) -> None:
    """flat_spline knot grading g ≥ 1 (1 = uniform). Knots are JIT-baked
    python constants → cache busts on change (constant per pipeline run)."""
    global _KNOT_GRADING, _VALUE_AND_GRAD, _CACHED_CONSTS
    new = max(float(value), 1.0)
    if new != _KNOT_GRADING:
        _KNOT_GRADING = new
        _VALUE_AND_GRAD = None
        _CACHED_CONSTS = None


def set_flat(flip: bool, weights=None) -> None:
    """Per-page flat_spline runtime inputs: flip=True puts the binding at
    the page's LEFT edge (basis t = 1 − x/W); `weights` are the penalty-
    scaled flat weights (flat_outer_penalty × flat_outer_weights). Runtime
    array — alternating A/B pages does NOT recompile."""
    global _FLAT_FLIP, _FLAT_WEIGHTS
    _FLAT_FLIP = bool(flip)
    _FLAT_WEIGHTS = tuple(float(w) for w in (weights if weights is not None
                                             else ()))


def _project_xyz_mlx(xy_coords, z_coords, pvec, focal_length, mx):
    """Mirror of page_dewarp._jax._project_xy_jax in MLX ops, with the
    height field z computed by the caller (model-dependent)."""
    objpoints = mx.stack([xy_coords[:, 0], xy_coords[:, 1], z_coords], axis=1)

    rvec = pvec[0:3]
    tvec = pvec[3:6]

    # Rodrigues formula. theta-safe normalisation matches the JAX
    # variant; small-angle branch keeps gradients well-behaved near 0.
    theta = mx.linalg.norm(rvec)
    theta_safe = mx.where(theta < 1e-10, mx.array(1.0), theta)
    k = rvec / theta_safe
    K = mx.stack([
        mx.stack([mx.array(0.0), -k[2], k[1]]),
        mx.stack([k[2], mx.array(0.0), -k[0]]),
        mx.stack([-k[1], k[0], mx.array(0.0)]),
    ])
    K2 = K @ K
    eye3 = mx.eye(3)
    R_std = eye3 + mx.sin(theta) * K + (1.0 - mx.cos(theta)) * K2
    R_small = eye3 + theta * K + 0.5 * theta * theta * K2
    R = mx.where(theta < 1e-8, R_small, R_std)

    transformed = (R @ objpoints.T).T + tvec
    z = transformed[:, 2]
    u = focal_length * transformed[:, 0] / z
    v = focal_length * transformed[:, 1] / z
    return mx.stack([u, v], axis=1)


def _make_objective(focal_length: float, shear_cost: float,
                    cubic_cost: float, huber_delta: float,
                    model: str, n_modes: int, twist: bool = True,
                    grading: float = 1.0):
    """Build the closure-over-constants MLX objective.

    `model_dims` is a runtime (2,) array argument — per-page values do
    not bust the compile cache. `flat_args` likewise:
    [flip, w_1·λ … w_K·λ] (flat_spline only; zeros otherwise)."""
    import math
    import mlx.core as mx

    from lib.processors.sheet_models import (bspline_interior_basis,
                                             canonical_model,
                                             BSPLINE_MODELS,
                                             MODEL_FLAT_SPLINE,
                                             SPLINE_MODELS)

    model = canonical_model(model)
    is_spline = model in SPLINE_MODELS
    is_bspline = model in BSPLINE_MODELS
    is_flat = model == MODEL_FLAT_SPLINE
    ne = n_modes + 1 if is_spline else 0

    def objective(pvec, dstpoints_flat, keypoint_index, model_dims,
                  flat_args):
        # xy_coords[i] = (pvec[ki[i,0]], pvec[ki[i,1]]). MLX has no
        # advanced-indexing-into-1D-with-2D-array semantics matching
        # numpy's pvec[ki], so build columns explicitly. Same result.
        xy_x = pvec[keypoint_index[:, 0]]
        xy_y = pvec[keypoint_index[:, 1]]
        xy_coords = mx.stack([xy_x, xy_y], axis=1)

        # Pin row 0 to origin — matches page_dewarp's convention where
        # the first keypoint sits at (0, 0) and other params absorb the
        # transform.
        row0_pin = mx.zeros((1, 2))
        xy_coords = mx.concatenate(
            [row0_pin, xy_coords[1:]], axis=0
        )

        x = xy_coords[:, 0]
        if is_spline:
            # z(x,y) = (1 + γ·η)·profile(x), η = y/H − 0.5. Extras live
            # at the pvec tail (see sheet_models.py); the keypoint_index
            # never reaches them.
            coeffs = mx.clip(pvec[-ne:-1], -0.5, 0.5)
            # twist=False: γ baked to 0 → zero gradient on the tail slot,
            # pvec[-1] stays at its init (0).
            gamma = (mx.clip(pvec[-1], -4.0, 4.0) if twist
                     else mx.array(0.0))
            if is_bspline:
                # Clamped cubic B-spline, K free interior control points.
                # Knots are python constants → recursion unrolls to
                # elementwise where/mul ops, autodiff-safe.
                t_raw = x / model_dims[0]
                if is_flat:
                    # flip ∈ {0, 1} runtime: t = t_raw or 1 − t_raw
                    # (binding always at basis t = 1).
                    flip = flat_args[0]
                    t_raw = t_raw + flip * (1.0 - 2.0 * t_raw)
                t = mx.clip(t_raw, 0.0, 1.0)
                basis = bspline_interior_basis(t, n_modes, mx,
                                               grading=grading)
                z_coords = mx.zeros_like(t)
                for k in range(n_modes):
                    z_coords = z_coords + coeffs[k] * basis[k]
            else:
                t = x * (math.pi / model_dims[0])
                z_coords = mx.zeros_like(t)
                for k in range(1, n_modes + 1):
                    z_coords = z_coords + coeffs[k - 1] * mx.sin(k * t)
            eta = xy_coords[:, 1] / model_dims[1] - 0.5
            z_coords = (1.0 + gamma * eta) * z_coords
        else:
            alpha = mx.clip(pvec[6], -0.5, 0.5)
            beta = mx.clip(pvec[7], -0.5, 0.5)
            z_coords = ((alpha + beta) * x ** 3
                        + (-2.0 * alpha - beta) * x ** 2 + alpha * x)

        proj = _project_xyz_mlx(xy_coords, z_coords, pvec, focal_length, mx)
        r2 = mx.sum((dstpoints_flat - proj) ** 2, axis=1)
        if huber_delta > 0.0:
            # Pseudo-Huber on the per-keypoint residual norm: quadratic
            # near 0 (matches L2 inlier behaviour), linear past δ —
            # one stray span (footer/caption that survived the width
            # filter) can no longer drag the whole sheet. Smooth
            # everywhere, so L-BFGS-B stays happy.
            d2 = huber_delta * huber_delta
            error = mx.sum(2.0 * d2 * (mx.sqrt(1.0 + r2 / d2) - 1.0))
        else:
            error = mx.sum(r2)
        if shear_cost > 0.0:
            error = error + shear_cost * pvec[0] ** 2
        if cubic_cost > 0.0:
            if is_spline:
                # Bending energy — see sheet_models.spline_bending_energy.
                # γ unpenalised: twist is an amplitude gradient, not
                # curvature (clip is the runaway guard).
                reg = mx.array(0.0)
                if is_bspline:
                    # Σ (m²·Δ²cᵢ)² over zero-padded control polygon.
                    m2 = float(max(n_modes - 1, 1)) ** 2
                    for i in range(n_modes):
                        prev = coeffs[i - 1] if i > 0 else mx.array(0.0)
                        nxt = (coeffs[i + 1] if i < n_modes - 1
                               else mx.array(0.0))
                        reg = reg + (m2 * (prev - 2.0 * coeffs[i] + nxt)) ** 2
                else:
                    # Σ (k²·c_k)² — phantom high-mode ripple suppressed,
                    # real low-mode curl survives.
                    for k in range(1, n_modes + 1):
                        reg = reg + (k * k * coeffs[k - 1]) ** 2
                error = error + cubic_cost * reg
            else:
                # L2 on cubic slopes α (pvec[6]) and β (pvec[7]). Stock
                # page-dewarp has no regularizer here, so on flat inputs
                # the optimizer drives α, β to ±0.5 absorbing span-mean-y
                # noise — phantom curl in the inverse remap. Penalty
                # collapses them toward 0 unless data demand curvature.
                error = error + cubic_cost * (pvec[6] ** 2 + pvec[7] ** 2)
        if is_flat:
            # Outer-flatness penalty: Σ (w_i·λ)·c_i². Weights arrive
            # penalty-scaled at runtime (flat_args[1:]); zeros = off.
            reg_flat = mx.array(0.0)
            for k in range(n_modes):
                reg_flat = reg_flat + flat_args[1 + k] * coeffs[k] ** 2
            error = error + reg_flat
        return error

    return objective


def _get_value_and_grad(focal_length: float, shear_cost: float,
                        cubic_cost: float, huber_delta: float,
                        model: str, n_modes: int, twist: bool = True,
                        grading: float = 1.0):
    """Cache the constants-specific MLX function. Constants are
    JIT-baked, so the cache key includes all of them — the set_* helpers
    invalidate on change."""
    global _VALUE_AND_GRAD, _CACHED_CONSTS
    key = (focal_length, shear_cost, cubic_cost, huber_delta,
           model, n_modes, twist, grading)
    if _VALUE_AND_GRAD is not None and _CACHED_CONSTS == key:
        return _VALUE_AND_GRAD
    import mlx.core as mx
    objective = _make_objective(focal_length, shear_cost, cubic_cost,
                                huber_delta, model, n_modes, twist,
                                grading)
    # mx.compile fuses the ~40-op eager graph into one Metal kernel
    # launch per L-BFGS-B iteration. Caches per input shape.
    _VALUE_AND_GRAD = mx.compile(mx.value_and_grad(objective, argnums=0))
    _CACHED_CONSTS = key
    return _VALUE_AND_GRAD


def _run_mlx_lbfgsb(dstpoints: np.ndarray,
                    keypoint_index: np.ndarray,
                    params: np.ndarray):
    """Drop-in replacement for `_run_jax_lbfgsb`. Computes value+grad
    on MLX (Metal GPU, unified memory), runs scipy L-BFGS-B as the
    outer loop. Returns the scipy OptimizeResult so the rest of
    page_dewarp's pipeline is unchanged."""
    import mlx.core as mx
    from page_dewarp.options import cfg

    focal_length = float(cfg.FOCAL_LENGTH)
    shear_cost = float(cfg.SHEAR_COST)
    cubic_cost = float(_CUBIC_COST)
    huber_delta = float(_HUBER_DELTA)

    dst_mx = mx.array(np.asarray(dstpoints, dtype=np.float32).reshape(-1, 2))
    ki_mx = mx.array(np.asarray(keypoint_index, dtype=np.int32))
    dims_mx = mx.array(np.asarray(_MODEL_DIMS, dtype=np.float32))
    # [flip, w_1·λ … w_K·λ] — fixed (1 + K) shape per model config.
    flat_np = np.zeros(1 + _N_MODES, dtype=np.float32)
    flat_np[0] = 1.0 if _FLAT_FLIP else 0.0
    if _FLAT_WEIGHTS:
        flat_np[1:1 + len(_FLAT_WEIGHTS)] = _FLAT_WEIGHTS
    flat_mx = mx.array(flat_np)
    value_and_grad = _get_value_and_grad(focal_length, shear_cost,
                                         cubic_cost, huber_delta,
                                         _MODEL, _N_MODES, _TWIST,
                                         _KNOT_GRADING)

    def obj_with_grad_np(p):
        p_mx = mx.array(np.asarray(p, dtype=np.float32))
        val, grad = value_and_grad(p_mx, dst_mx, ki_mx, dims_mx, flat_mx)
        mx.eval(val, grad)  # force lazy graph to realise before .item()
        val_np = float(val.item())
        grad_np = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(val_np):
            return 1e10, np.zeros_like(p)
        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
        return val_np, grad_np

    result = minimize(
        obj_with_grad_np,
        params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": cfg.OPT_MAX_ITER, "maxcor": cfg.MAX_CORR},
    )
    return result


def clear_pool() -> None:
    """Free the MLX Metal allocator pool. Called after each
    PageDewarper.process() to keep per-worker phys_footprint bounded —
    every MLX call pins a few hundred MB of unified memory; after ~10
    scans the worker watchdog SIGKILLs at the 3 GB cap. The pool is the
    actual leak source; the compiled value_and_grad closure is tiny and
    kept so subsequent pages skip re-tracing.
    Idempotent; safe if mlx is not installed."""
    try:
        import mlx.core as mx
        # Prefer mx.clear_cache; fall back to mx.metal.clear_cache for
        # older mlx versions.
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def clear_caches() -> None:
    """Full clear: drop the compiled value_and_grad closure AND free the
    Metal allocator pool. Reserved for cache-policy resets; per-image
    cleanup should call clear_pool() instead."""
    global _VALUE_AND_GRAD, _CACHED_CONSTS
    _VALUE_AND_GRAD = None
    _CACHED_CONSTS = None
    clear_pool()


def install() -> bool:
    """Replace page_dewarp.optimise._jax._run_jax_lbfgsb with the MLX
    variant. Idempotent. Returns True on success.

    The patch site is identical to the padded-JAX variant (they target
    the same internal). Installing both will leave only the last one
    active; current code keeps them as mutually-exclusive options."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        # Probe mlx up front. Without this guard, install() patches
        # page_dewarp's JAX entry point unconditionally — then the first
        # dewarp call later crashes with ModuleNotFoundError("mlx") deep
        # in the worker. Probing here lets PageDewarper fall back to JAX
        # / Powell cleanly on machines without MLX (Linux, Intel macs).
        import mlx.core  # noqa: F401
        from page_dewarp.optimise import _jax as _pd_jax_mod
        _pd_jax_mod._run_jax_lbfgsb = _run_mlx_lbfgsb
        _INSTALLED = True
        return True
    except Exception:
        return False
