# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Sheet-surface models for PageDewarper.

page-dewarp hardcodes a *cylindrical* sheet: z(x) = (α+β)x³ − (2α+β)x² + αx
(generalised cylinder — every horizontal slice has the same height profile,
roots pinned at x = 0 and x = 1). Real book pages violate this two ways:

- curl amplitude varies top-to-bottom (spine binding pulls harder near the
  ends — a *twist*), and
- the height profile is not a cubic (steep gutter wall + long flat field
  needs more than two shape DOF).

Two twist models address both. Each is a height profile modulated linearly
in y:

    z(x, y) = (1 + γ·η) · profile(x)
    η = y / H − 0.5                       (−0.5 top edge … +0.5 bottom edge)

with profile(x):

- `sine_twist` (formerly `spline_twist` — legacy name still accepted):
  Fourier-sine series Σ_{k=1..K} c_k · sin(kπ · x / W). Global support,
  spectral ordering (mode k carries curvature ∝ k²).
- `bspline_twist`: clamped cubic B-spline with K free interior control
  points, both endpoint control points pinned to 0. Local support — a
  steep gutter wall does not ripple into the flat field the way high sine
  modes do.

(W, H) are the *model page dims* (rough dims at fit time — stored in
replay params so replay reconstructs the identical surface). Both bases
are zero at the page edges by construction (same boundary condition as
the stock cubic). γ is the twist gain.

Parameter-vector layout (identical for both models): the K+1 extras
[c_1 … c_K, γ] are appended at the TAIL of page-dewarp's pvec —
`make_keypoint_index` indices (8 + nspans + npts) stay valid, and the
library's own numpy path simply ignores the tail.

All projections here take the focal length EXPLICITLY instead of reading
page_dewarp's global cfg — the global is save/restored around the optimise
call and reading it later silently picks up the library default (1.2).
"""
from __future__ import annotations

import numpy as np

MODEL_CYLINDRICAL = "cylindrical"
MODEL_SINE_TWIST = "sine_twist"
MODEL_BSPLINE_TWIST = "bspline_twist"
# flat_spline = bspline_twist specialised for post-TrapezoidalCorrection
# pages: the sheet is assumed FLAT except for curl near the binding.
# Three additions over bspline_twist (all default-off → identical):
#   - graded knots: knot density concentrated near the binding edge
#     (coarse spline over the flat field, high resolution at the gutter);
#   - a flip convention: the binding always sits at basis parameter t = 1,
#     pages bound on the LEFT evaluate at t = 1 − x/W;
#   - flat_outer_penalty: L2 on control points weighted by (squared)
#     Greville distance from the binding — pulls the outer field to z = 0.
MODEL_FLAT_SPLINE = "flat_spline"
SPLINE_MODELS = (MODEL_SINE_TWIST, MODEL_BSPLINE_TWIST, MODEL_FLAT_SPLINE)
BSPLINE_MODELS = (MODEL_BSPLINE_TWIST, MODEL_FLAT_SPLINE)

# Pre-rename replay stamps / configs say "spline_twist"; canonicalise so
# old DB nodes replay byte-identical under the new name.
_LEGACY_ALIASES = {"spline_twist": MODEL_SINE_TWIST,
                   "flat-spline": MODEL_FLAT_SPLINE}

# Clip bounds mirror page-dewarp's α/β clamp (runaway-stretch guard).
COEFF_CLIP = 0.5
# (1 + γ·η), η ∈ [−0.5, +0.5]. |γ| > 2 lets the twist factor cross zero
# INSIDE the page — sign-flipping curl (top lines bow up, bottom lines
# bow down: the open-book fan). With γ capped below 2 such pages cancel
# to c ≈ 0 and γ rails (observed on dboc7 p18).
GAMMA_CLIP = 4.0


def canonical_model(model: str | None) -> str:
    """Normalise a sheet-model name (legacy aliases, case, None)."""
    name = str(model or MODEL_CYLINDRICAL).lower()
    return _LEGACY_ALIASES.get(name, name)


def is_spline_model(model: str | None) -> bool:
    return canonical_model(model) in SPLINE_MODELS


def n_extras(model: str, n_modes: int) -> int:
    """Number of tail parameters the model appends to the page-dewarp pvec."""
    return int(n_modes) + 1 if is_spline_model(model) else 0


def split_extras(pvec: np.ndarray, model: str, n_modes: int
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """(base_pvec, coeffs, gamma) — coeffs/gamma clipped to model bounds."""
    ne = n_extras(model, n_modes)
    if ne == 0:
        return np.asarray(pvec), np.empty(0), 0.0
    pvec = np.asarray(pvec)
    coeffs = np.clip(pvec[-ne:-1].astype(np.float64), -COEFF_CLIP, COEFF_CLIP)
    gamma = float(np.clip(pvec[-1], -GAMMA_CLIP, GAMMA_CLIP))
    return pvec[:-ne], coeffs, gamma


# --------------------------------------------------------------------------
# B-spline basis (clamped cubic, both endpoint control points pinned to 0)
# --------------------------------------------------------------------------

def bspline_knots(n_free: int, grading: float = 1.0) -> list[float]:
    """Clamped cubic knot vector on [0, 1] for K = n_free interior control
    points (total control points K+2; the first and last are pinned to 0
    so z vanishes at both page edges). K=2 degenerates to the Bernstein
    basis (single cubic segment).

    `grading` g ≥ 1 concentrates knots near t = 1 (the binding edge under
    the flat_spline convention) via u → 1 − (1 − u)^g: the flat outer
    field gets a coarse spline, the gutter wall a fine one. g = 1 is the
    uniform vector (bspline_twist)."""
    m = int(n_free) - 1  # number of segments
    if m < 1:
        raise ValueError(f"bspline needs >= 2 free control points, got {n_free}")
    g = max(float(grading), 1.0)
    return ([0.0] * 4
            + [1.0 - (1.0 - i / m) ** g for i in range(1, m)]
            + [1.0] * 4)


def _cox_de_boor(t, knots: list[float], degree: int, xp):
    """Cox–de Boor basis up to `degree` at parameter t (array, any backend
    module xp ∈ {numpy, jax.numpy, mlx.core}). Knots are python floats so
    the recursion unrolls to elementwise ops only — autodiff/JIT friendly.
    Zero-width spans (repeated clamp knots) yield None entries, dropped by
    the guarded recursion. Half-open intervals: every basis evaluates to 0
    at exactly t = 1, which equals the interior-basis limit there."""
    nb = len(knots) - 1
    N: list = []
    for i in range(nb):
        if knots[i + 1] > knots[i]:
            lo = xp.where(t >= knots[i], 1.0, 0.0)
            hi = xp.where(t < knots[i + 1], 1.0, 0.0)
            N.append(lo * hi)
        else:
            N.append(None)
    for p in range(1, degree + 1):
        Np: list = []
        for i in range(nb - p):
            term = None
            d1 = knots[i + p] - knots[i]
            if d1 > 0.0 and N[i] is not None:
                term = ((t - knots[i]) / d1) * N[i]
            d2 = knots[i + p + 1] - knots[i + 1]
            if d2 > 0.0 and N[i + 1] is not None:
                t2 = ((knots[i + p + 1] - t) / d2) * N[i + 1]
                term = t2 if term is None else term + t2
            Np.append(term)
        N = Np
    return N


def bspline_interior_basis(t, n_free: int, xp=np, grading: float = 1.0) -> list:
    """The K free (interior) clamped-cubic basis functions at t ∈ [0, 1].
    Endpoint basis functions are dropped — pinning their control points
    to 0 enforces z(0) = z(W) = 0 exactly. Returns a list of K arrays.
    Backend-generic (numpy / jax.numpy / mlx.core)."""
    knots = bspline_knots(n_free, grading)
    N = _cox_de_boor(t, knots, 3, xp)
    return [b if b is not None else xp.zeros_like(t) for b in N[1:-1]]


def bspline_interior_basis_deriv(t, n_free: int, xp=np,
                                 grading: float = 1.0) -> list:
    """d/dt of the K interior basis functions:
    N'_{i,3} = 3·( N_{i,2}/(k_{i+3}−k_i) − N_{i+1,2}/(k_{i+4}−k_{i+1}) ),
    zero-denominator terms dropped."""
    knots = bspline_knots(n_free, grading)
    N2 = _cox_de_boor(t, knots, 2, xp)
    out = []
    for i in range(1, len(N2) - 2):  # interior cubic basis indices
        term = None
        d1 = knots[i + 3] - knots[i]
        if d1 > 0.0 and N2[i] is not None:
            term = (3.0 / d1) * N2[i]
        d2 = knots[i + 4] - knots[i + 1]
        if d2 > 0.0 and N2[i + 1] is not None:
            t2 = (-3.0 / d2) * N2[i + 1]
            term = t2 if term is None else term + t2
        out.append(term if term is not None else xp.zeros_like(t))
    return out


def flat_outer_weights(n_free: int, grading: float = 1.0) -> np.ndarray:
    """Per-control-point weights for the flat_spline outer-flatness
    penalty: w_i = (1 − ξ_i)², ξ_i = Greville abscissa of free control
    point i over the (graded) knot vector, binding at t = 1.

    Quadratic in the distance from the binding: control points at the
    gutter wall are essentially free (w → 0), the far edge is fully
    weighted (w → ~1). The penalty term is then
        flat_outer_penalty · Σ w_i · c_i²
    — z pulled to 0 (the post-trapezoidal flat plane) exactly where the
    flatness assumption holds. Coefficient 0 disables the term."""
    knots = bspline_knots(n_free, grading)
    xi = np.array([(knots[i + 1] + knots[i + 2] + knots[i + 3]) / 3.0
                   for i in range(1, int(n_free) + 1)])
    return (1.0 - xi) ** 2


# --------------------------------------------------------------------------
# Height profile (basis dispatch)
# --------------------------------------------------------------------------

def _basis_t(x: np.ndarray, w_m: float, flip: bool) -> np.ndarray:
    """Page-x → basis parameter. flat_spline convention: binding at
    t = 1; flip=True for pages bound on the LEFT (t = 1 − x/W)."""
    t = x / w_m
    if flip:
        t = 1.0 - t
    return np.clip(t, 0.0, 1.0)


def _profile(x: np.ndarray, coeffs: np.ndarray, w_m: float,
             model: str = MODEL_SINE_TWIST,
             grading: float = 1.0, flip: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if canonical_model(model) in BSPLINE_MODELS:
        # Outside [0, W] the clamped t pins z to the edge value 0 (flat
        # extension) — the support clamp handles in-page margins anyway.
        t = _basis_t(x, w_m, flip)
        basis = bspline_interior_basis(t, len(coeffs), np, grading)
        z = np.zeros_like(t)
        for c, b in zip(coeffs, basis):
            z += float(c) * b
        return z
    t = x * (np.pi / w_m)
    z = np.zeros_like(t)
    for k, c in enumerate(coeffs, start=1):
        z += float(c) * np.sin(k * t)
    return z


def _profile_slope(x: np.ndarray, coeffs: np.ndarray, w_m: float,
                   model: str = MODEL_SINE_TWIST,
                   grading: float = 1.0, flip: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if canonical_model(model) in BSPLINE_MODELS:
        t = _basis_t(x, w_m, flip)
        dbasis = bspline_interior_basis_deriv(t, len(coeffs), np, grading)
        zp = np.zeros_like(t)
        for c, db in zip(coeffs, dbasis):
            zp += float(c) * db
        # Chain rule t = x/W (or 1 − x/W when flipped).
        return -zp / w_m if flip else zp / w_m
    t = x * (np.pi / w_m)
    zp = np.zeros_like(t)
    for k, c in enumerate(coeffs, start=1):
        zp += float(c) * (k * np.pi / w_m) * np.cos(k * t)
    return zp


# Slope-bound length of the support extension, as a fraction of the model
# page width. Margin-extension history (all observed on dboc7, wide left
# margin):
#   1. pure tangent — grows linearly; a steep gutter wall fitted right at
#      the support edge (B-spline especially) blows the margin grid up.
#   2. exponential slope decay — bounded, but asymptotes to a CONSTANT
#      non-zero z, which the twist factor (1 + γ·η) scales differently
#      per row → fanned margin columns.
#   3. Gaussian window over λ — z → 0, but the whole descent happens
#      inside λ: local extension slope steeper than the wall itself →
#      S-wiggled grid.
# Current scheme: bounded tangent (excursion ≤ |slope|·λ) multiplied by a
# cos² window over the ENTIRE margin (support edge → page edge), reaching
# exactly 0 at the page edge. C¹ at both ends, descent spread as wide as
# the geometry allows.
SUPPORT_DECAY_FRAC = 0.05


def _profile_ext(x: np.ndarray, coeffs: np.ndarray, w_m: float,
                 support, model: str = MODEL_SINE_TWIST,
                 decay=None, grading: float = 1.0,
                 flip: bool = False) -> np.ndarray:
    """Profile with C¹ tangent extension outside the data support.

    The optimiser only ever evaluates the sheet at span keypoints, so the
    basis is unconstrained in the page margins — it wiggles freely there
    (observed: wild curl in a wide left margin while the spans sit flat).
    Inside [x0, x1] this is the plain profile. Outside, with
    d = x − clip(x):

      legacy (decay None/0):
          z_edge + slope·d                          (pure tangent)
      decaying (λ = decay):
          [z_edge + slope·sign(d)·λ·(1 − e^{−|d|/λ})] · cos²(π·u/2)
          u = clip(|d| / margin, 0, 1)
          margin = x0 (left side) or W − x1 (right side)

    The bracket leaves the edge along its tangent but caps the excursion
    at |slope|·λ; the cos² window is 1 with zero derivative at the
    support edge (C¹), descends over the WHOLE margin, and is exactly 0
    (with zero derivative) at the page edge — the margin ends as flat
    sheet, so the twist factor has nothing to fan, and the descent is as
    gentle as the margin width allows."""
    x = np.asarray(x, dtype=np.float64)
    if support is None:
        return _profile(x, coeffs, w_m, model, grading, flip)
    x0, x1 = float(support[0]), float(support[1])
    xc = np.clip(x, x0, x1)
    d = x - xc
    if decay:
        lam = float(decay)
        g = np.sign(d) * lam * (1.0 - np.exp(-np.abs(d) / lam))
    else:
        g = d
    z_ext = (_profile(xc, coeffs, w_m, model, grading, flip)
             + _profile_slope(xc, coeffs, w_m, model, grading, flip) * g)
    if decay:
        margin = np.where(d < 0, max(x0, 1e-9), max(w_m - x1, 1e-9))
        u = np.clip(np.abs(d) / margin, 0.0, 1.0)
        z_ext = z_ext * np.cos(0.5 * np.pi * u) ** 2
    return z_ext


def spline_z(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, gamma: float,
             model_dims: tuple[float, float], support=None,
             model: str = MODEL_SINE_TWIST, support_y=None,
             support_decay=None, grading: float = 1.0,
             flip: bool = False) -> np.ndarray:
    """Evaluate the twist-model height field at model coords (x, y).

    `support_y` = y-range of the fitted spans. The twist factor (1 + γ·η)
    is linear in y, so above the first / below the last span it keeps
    growing (or sign-flips at large γ) with nothing constraining it —
    amplified phantom curl in the top/bottom margins. Clamping y to the
    data range freezes the amplitude at its edge value (a tangent
    extension of a linear function IS the function — flat clamp is the
    only non-trivial extension)."""
    w_m, h_m = float(model_dims[0]), float(model_dims[1])
    z = _profile_ext(x, coeffs, w_m, support, model, decay=support_decay,
                     grading=grading, flip=flip)
    y = np.asarray(y, dtype=np.float64)
    if support_y is not None:
        y = np.clip(y, float(support_y[0]), float(support_y[1]))
    eta = y / h_m - 0.5
    return (1.0 + gamma * eta) * z


def spline_dzdx_mid(x: np.ndarray, coeffs: np.ndarray,
                    model_dims: tuple[float, float],
                    support=None, model: str = MODEL_SINE_TWIST,
                    grading: float = 1.0, flip: bool = False) -> np.ndarray:
    """∂z/∂x along the mid row (η = 0, twist term vanishes), support
    clamped (slope frozen at the edge value in the margins). The
    arc-length grid does NOT use this in the margins — it differentiates
    the extended profile numerically (see arclength_x)."""
    w_m = float(model_dims[0])
    x = np.asarray(x, dtype=np.float64)
    if support is None:
        return _profile_slope(x, coeffs, w_m, model, grading, flip)
    xc = np.clip(x, float(support[0]), float(support[1]))
    return _profile_slope(xc, coeffs, w_m, model, grading, flip)


def project_xy_model(xy_coords: np.ndarray, pvec: np.ndarray, *,
                     model: str = MODEL_CYLINDRICAL, n_modes: int = 0,
                     model_dims=None, focal_length: float = 1.2,
                     support=None, support_y=None,
                     support_decay=None, grading: float = 1.0,
                     flip: bool = False) -> np.ndarray:
    """Model-aware replacement for page_dewarp.projection.project_xy.

    Returns an (N, 1, 2) array of projected image points (pix2norm units).
    Focal length is explicit; cx = cy = 0 as in the library K matrix.
    `support` / `support_y` (twist models only): x/y-range of the fitted
    span keypoints — outside it the surface is tangent-extended in x
    (decaying over `support_decay` when set) and amplitude-frozen in y
    (see _profile_ext / spline_z).
    """
    from cv2 import projectPoints

    model = canonical_model(model)
    xy_coords = np.asarray(xy_coords, dtype=np.float64).reshape((-1, 2))
    if model in SPLINE_MODELS:
        _, coeffs, gamma = split_extras(pvec, model, n_modes)
        z_coords = spline_z(xy_coords[:, 0], xy_coords[:, 1],
                            coeffs, gamma, model_dims, support=support,
                            model=model, support_y=support_y,
                            support_decay=support_decay,
                            grading=grading, flip=flip)
    else:
        alpha = float(np.clip(pvec[6], -0.5, 0.5))
        beta = float(np.clip(pvec[7], -0.5, 0.5))
        x = xy_coords[:, 0]
        z_coords = ((alpha + beta) * x ** 3
                    + (-2.0 * alpha - beta) * x ** 2 + alpha * x)

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))
    K = np.array([[focal_length, 0, 0],
                  [0, focal_length, 0],
                  [0, 0, 1]], dtype=np.float32)
    rvec = np.asarray(pvec[0:3], dtype=np.float64)
    tvec = np.asarray(pvec[3:6], dtype=np.float64)
    image_points, _ = projectPoints(objpoints, rvec, tvec, K, np.zeros(5))
    return image_points


def project_keypoints_model(pvec: np.ndarray, keypoint_index: np.ndarray, *,
                            model: str = MODEL_CYLINDRICAL, n_modes: int = 0,
                            model_dims=None, focal_length: float = 1.2,
                            grading: float = 1.0, flip: bool = False
                            ) -> np.ndarray:
    """Model-aware replacement for page_dewarp.keypoints.project_keypoints."""
    xy_coords = np.asarray(pvec)[np.asarray(keypoint_index)]
    xy_coords[0, :] = 0
    return project_xy_model(xy_coords, pvec, model=model, n_modes=n_modes,
                            model_dims=model_dims, focal_length=focal_length,
                            grading=grading, flip=flip)


def arclength_x(params: np.ndarray, page_w: float, *,
                model: str = MODEL_CYLINDRICAL, n_modes: int = 0,
                model_dims=None, n: int = 4096, support=None,
                support_decay=None, grading: float = 1.0,
                flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative mid-row arc length of the sheet over x ∈ [0, page_w].

    Used to build an arc-length-uniform output x grid (paper is
    inextensible — uniform-x remap stretches text by √(1+z′²) where the
    sheet is steep). For the twist models the mid row (η = 0) is used so
    the grid is shared by all rows and output verticals stay straight.
    """
    model = canonical_model(model)
    if model in SPLINE_MODELS:
        _, coeffs, _ = split_extras(params, model, n_modes)
        xs = np.linspace(0.0, float(page_w), int(n))
        # Mid-row height IS the extended profile (twist factor 1 at
        # η = 0). Numerical gradient keeps the arc grid consistent with
        # whatever extension/window _profile_ext applies — no analytic
        # margin-slope twin to keep in sync.
        z_mid = _profile_ext(xs, coeffs, float(model_dims[0]), support,
                             model, decay=support_decay,
                             grading=grading, flip=flip)
        zp = np.gradient(z_mid, xs)
        ds = np.sqrt(1.0 + zp * zp)
        s = np.concatenate([[0.0],
                            np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(xs))])
        return xs, s
    from aglaia.processors.geometry import dewarp_arclength_x
    return dewarp_arclength_x(params, page_w, n=n)


def get_page_dims_model(corners: np.ndarray, rough_dims, params: np.ndarray, *,
                        model: str = MODEL_CYLINDRICAL, n_modes: int = 0,
                        model_dims=None, focal_length: float = 1.2,
                        support=None, support_y=None,
                        support_decay=None, grading: float = 1.0,
                        flip: bool = False) -> np.ndarray:
    """Model-aware replacement for page_dewarp.image.get_page_dims.

    The library version projects the bottom-right corner through its
    cylindrical-only project_xy (global-cfg focal) — under a twist model
    that fits the output dims against the wrong surface."""
    from scipy.optimize import minimize

    dst_br = np.asarray(corners[2]).flatten()

    def objective(dims):
        proj = project_xy_model(
            np.asarray(dims, dtype=np.float64).reshape(1, 2), params,
            model=model, n_modes=n_modes, model_dims=model_dims,
            focal_length=focal_length, support=support,
            support_y=support_y, support_decay=support_decay,
            grading=grading, flip=flip)
        return float(np.sum((dst_br - proj.flatten()) ** 2))

    res = minimize(objective, np.array(rough_dims, dtype=np.float64),
                   method="Powell")
    return res.x


def spline_bending_energy(coeffs: np.ndarray, gamma: float,
                          model: str = MODEL_SINE_TWIST) -> float:
    """Curvature-weighted L2 used as the twist-model analogue of
    cubic_cost. γ is NOT penalised: twist is an amplitude gradient, not
    curvature, and a γ² term was observed fighting genuine one-sided curl
    (the clip is the runaway guard).

    sine_twist:    Σ (k²·c_k)² — high modes cost most (Fourier form of
                   ∫(z″)²): phantom ripple > genuine low-mode curl.
    bspline_twist / flat_spline: Σ (m²·Δ²cᵢ)² over the zero-padded
                   control polygon [0, c_1 … c_K, 0] (m = K−1 segments) —
                   the discrete ∫(z″)² of the control polygon (uniform-
                   spacing approximation, kept for graded knots too)."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if canonical_model(model) in BSPLINE_MODELS:
        ctrl = np.concatenate([[0.0], coeffs, [0.0]])
        m = max(len(coeffs) - 1, 1)
        d2 = ctrl[:-2] - 2.0 * ctrl[1:-1] + ctrl[2:]
        return float(np.sum((m * m * d2) ** 2))
    e = 0.0
    for k, c in enumerate(coeffs, start=1):
        e += (k * k * c) ** 2
    return e


def optimise_params_spline_powell(dstpoints: np.ndarray,
                                  span_counts: list[int],
                                  params: np.ndarray, *,
                                  n_modes: int,
                                  model_dims,
                                  focal_length: float,
                                  model: str = MODEL_SINE_TWIST,
                                  twist: bool = True,
                                  shear_cost: float = 0.0,
                                  cubic_cost: float = 0.0,
                                  huber_delta: float = 0.0,
                                  grading: float = 1.0,
                                  flip: bool = False,
                                  flat_penalty: float = 0.0,
                                  maxiter: int = 2000) -> np.ndarray:
    """SciPy-Powell fallback optimiser for the twist sheet models.

    The library's Powell path projects through the cubic-only objective and
    would silently ignore the model tail. Derivative-free over the full
    pvec — slow, correctness-only fallback (mirrors the library's own
    Powell role)."""
    from scipy.optimize import minimize
    from page_dewarp.keypoints import make_keypoint_index

    model = canonical_model(model)
    ki = make_keypoint_index(span_counts)
    target = np.asarray(dstpoints, dtype=np.float64).reshape((-1, 2))
    ne = n_extras(model, n_modes)
    flat_w = (flat_penalty * flat_outer_weights(n_modes, grading)
              if flat_penalty > 0.0 and model == MODEL_FLAT_SPLINE
              else None)

    def _obj(pvec):
        if not twist:
            # γ pinned to 0 — objective independent of the tail slot, so
            # Powell finds no improvement direction along it.
            pvec = np.asarray(pvec, dtype=np.float64).copy()
            pvec[-1] = 0.0
        ppts = project_keypoints_model(
            pvec, ki, model=model, n_modes=n_modes,
            model_dims=model_dims, focal_length=focal_length,
            grading=grading, flip=flip,
        ).reshape((-1, 2))
        r2 = np.sum((target - ppts) ** 2, axis=1)
        if huber_delta > 0.0:
            d2 = huber_delta * huber_delta
            err = float(np.sum(2.0 * d2 * (np.sqrt(1.0 + r2 / d2) - 1.0)))
        else:
            err = float(np.sum(r2))
        if shear_cost > 0.0:
            err += shear_cost * float(pvec[0]) ** 2
        if cubic_cost > 0.0:
            coeffs = pvec[-ne:-1]
            err += cubic_cost * spline_bending_energy(coeffs, pvec[-1], model)
        if flat_w is not None:
            coeffs = np.clip(pvec[-ne:-1], -COEFF_CLIP, COEFF_CLIP)
            err += float(np.sum(flat_w * coeffs * coeffs))
        return err

    res = minimize(_obj, np.asarray(params, dtype=np.float64),
                   method="Powell",
                   options={"maxiter": int(maxiter),
                            "xtol": 1e-6, "ftol": 1e-7})
    out = res.x.astype(np.float32)
    if not twist:
        out[-1] = 0.0
    return out
