# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Levenberg-Marquardt sheet-fit solver for the cylindrical dewarp model.

Backport of the iOS port's `LMSolver.swift` (issue #59, validated in
yb85/aglaia-ios#20) plus the camera/objective upgrades from the on-device
research in #60: principal-point DOF, spine-zone residual weights and a
spine-localized directrix term.

Why it is fast — the page_dewarp objective is a NLLS problem with bipartite
sparsity:

- **global** params: rvec, tvec, α, β (8), optionally + (cx, cy) (10)
- **local** nuisance params: one span height `y_i` per span (S) and one
  abscissa `x_ij` per keypoint (N)

Points inside a span share only `y_i`, and `x_ij ⟂ x_ik` across residuals, so
each local Hessian block `V_i` is an **arrowhead** matrix — eliminated in
O(n_i) by Sherman-Morrison instead of O(n_i³). The Schur complement then
reduces every LM iteration to one O(N) assembly pass, S arrowhead
eliminations, one `n_cam × n_cam` solve and an O(N) back-substitution.
~10-60 LM iterations replace the ~10⁵–10⁶ objective evaluations Powell needs.

Measured on iOS (M4, release, aglaia test_data fixtures): 7.4 ms/page vs
539 ms for Powell and 3.0 s for MLX L-BFGS, at an equal-or-better final
objective on every fixture page.

Same variables, clip semantics and rvec (axis-angle) parameterisation as the
Powell path, so warm start and replay params are untouched. The Rodrigues
derivative is the Gallego-Yezzi closed form; the α/β clip is handled like
autodiff-of-clamp (zero derivative when railed).

Robust loss: the desktop objective is pseudo-Huber by default (unlike the iOS
port's plain L2). Gauss-Newton on a robust loss is IRLS — the per-residual
weight `dρ/d(r²) = 1/√(1 + r²/δ²)` is recomputed at each Jacobian assembly,
while the accept/reject check evaluates the true robust cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# page-dewarp's runaway-stretch guard on the cubic slopes.
CURL_CLIP = 0.5


@dataclass(frozen=True)
class SpineCurl:
    """Spine-localized directrix term added to the cubic height profile:

        z += γ · exp(−|x − x₀| / s)

    x₀ is the binding edge in model page-x, s the decay length. Rejected as
    unidentifiable under the 8-param camera (#60, first pass); with the
    principal-point DOF it becomes strongly identifiable and supplies the
    surface slope the arc-length grid needs at the fold (#60, comment 3).
    γ is NOT a free parameter — the caller grid-searches it and keeps the
    best objective, exactly as the iOS port does."""

    gamma: float
    s_x: float
    x0: float

    def z(self, x):
        return self.gamma * np.exp(-np.abs(np.asarray(x, dtype=np.float64)
                                           - self.x0) / self.s_x)

    def dz(self, x):
        x = np.asarray(x, dtype=np.float64)
        return np.where(x >= self.x0, -1.0, 1.0) * self.z(x) / self.s_x

    def as_dict(self) -> dict:
        return {"gamma": float(self.gamma), "s_x": float(self.s_x),
                "x0": float(self.x0)}

    @staticmethod
    def from_dict(d):
        if not d:
            return None
        return SpineCurl(float(d["gamma"]), float(d["s_x"]), float(d["x0"]))


def keypoint_index(span_counts, n_cam: int = 8) -> np.ndarray:
    """`page_dewarp.keypoints.make_keypoint_index` with the camera block size
    made explicit — the library hardcodes 8, which the principal-point layout
    (10) shifts. Column 0 = index of `x_ij` in pvec, column 1 = index of the
    span height `y_i`; row 0 is the page origin (both zeroed by the caller)."""
    span_counts = [int(c) for c in span_counts]
    nspans, npts = len(span_counts), sum(span_counts)
    idx = np.zeros((npts + 1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        idx[start:start + count, 1] = n_cam + i
        start += count
    idx[1:, 0] = np.arange(npts) + n_cam + nspans
    return idx


def spine_weights(dst_x, boost: float, zone_frac: float,
                  spine_right: bool) -> np.ndarray | None:
    """Per-keypoint residual weights, ramped from `boost` AT the binding to 1
    at the far edge of the spine-side `zone_frac` of the x-range (#60).

    The spine tail is a small minority of the least-squares sum, so an
    unweighted fit systematically under-curls there (measured 2-4 px at
    300 dpi). Boost 4 over the spine-side 25% drops the by-third spine bias
    from (−2.9, −3.8, −1.7) px to (−0.1, −0.7, +0.2) at ~1 px mid-page cost."""
    if not boost or boost <= 1.0:
        return None
    x = np.asarray(dst_x, dtype=np.float64).ravel()
    x_min, x_max = float(x.min()), float(x.max())
    rng = max(x_max - x_min, 1e-9)
    u = (x - x_min) / rng                      # 0 at left edge, 1 at right
    d = (1.0 - u) if spine_right else u        # 0 AT the spine
    t = np.maximum(0.0, 1.0 - d / max(float(zone_frac), 1e-9))
    return 1.0 + (float(boost) - 1.0) * t


class Objective:
    """Weighted (optionally pseudo-Huber) reprojection cost of the cylindrical
    sheet, evaluated straight off the parameter vector.

    Mirrors what the LM assembly linearises, so the trust-region accept/reject
    test and the Gauss-Newton step never disagree about what is being
    minimised."""

    def __init__(self, span_counts, dstpoints, *, focal: float,
                 shear_cost: float = 0.0, cubic_cost: float = 0.0,
                 huber_delta: float = 0.0, weights=None,
                 spine: SpineCurl | None = None, n_cam: int = 8):
        self.focal = float(focal)
        self.shear_cost = float(shear_cost)
        self.cubic_cost = float(cubic_cost)
        self.huber_delta = float(huber_delta)
        self.spine = spine
        self.n_cam = int(n_cam)
        self.span_counts = [int(c) for c in span_counts]
        idx = keypoint_index(self.span_counts, self.n_cam)
        self.xi = idx[:, 0].copy()
        self.yi = idx[:, 1].copy()
        self.xi[0] = -1                       # origin row: x = y = 0
        self.yi[0] = -1
        dst = np.asarray(dstpoints, dtype=np.float64).reshape((-1, 2))
        self.dst_x = dst[:, 0].copy()
        self.dst_y = dst[:, 1].copy()
        self.n = dst.shape[0]
        if weights is None:
            self.weights = None
        else:
            w = np.asarray(weights, dtype=np.float64).ravel().copy()
            w[0] = 1.0                        # the origin row is not a span pt
            self.weights = w

    # -- shared geometry ---------------------------------------------------

    def unpack(self, pvec):
        """(x, y, α, β, cx, cy) with the clips applied and the origin row
        pinned to (0, 0)."""
        p = np.asarray(pvec, dtype=np.float64)
        x = np.where(self.xi >= 0, p[np.maximum(self.xi, 0)], 0.0)
        y = np.where(self.yi >= 0, p[np.maximum(self.yi, 0)], 0.0)
        alpha = float(np.clip(p[6], -CURL_CLIP, CURL_CLIP))
        beta = float(np.clip(p[7], -CURL_CLIP, CURL_CLIP))
        cx = float(p[8]) if self.n_cam > 8 else 0.0
        cy = float(p[9]) if self.n_cam > 9 else 0.0
        return x, y, alpha, beta, cx, cy

    def height(self, x, alpha: float, beta: float):
        """z(x) and z′(x) for the cubic sheet (+ the fixed spine term)."""
        c3 = alpha + beta
        c2 = -2.0 * alpha - beta
        z = ((c3 * x + c2) * x + alpha) * x
        zp = 3.0 * c3 * x * x + 2.0 * c2 * x + alpha
        if self.spine is not None:
            z = z + self.spine.z(x)
            zp = zp + self.spine.dz(x)
        return z, zp

    def residuals(self, pvec):
        """Unweighted (ex, ey) = projection − target, plus the geometry the
        Jacobian assembly reuses."""
        p = np.asarray(pvec, dtype=np.float64)
        x, y, alpha, beta, cx, cy = self.unpack(p)
        z, zp = self.height(x, alpha, beta)
        R = _rodrigues(p[0:3])
        t = p[3:6]
        pc0 = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0]
        pc1 = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1]
        pc2 = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2]
        inv_z = 1.0 / pc2
        fz = self.focal * inv_z
        ex = fz * pc0 + cx - self.dst_x
        ey = fz * pc1 + cy - self.dst_y
        return ex, ey, (x, y, z, zp, alpha, beta, R, pc0, pc1, inv_z, fz)

    # -- cost --------------------------------------------------------------

    def __call__(self, pvec) -> float:
        ex, ey, _ = self.residuals(pvec)
        r2 = ex * ex + ey * ey
        if self.weights is not None:
            r2 = self.weights * r2
        if self.huber_delta > 0.0:
            d2 = self.huber_delta * self.huber_delta
            err = float(np.sum(2.0 * d2 * (np.sqrt(1.0 + r2 / d2) - 1.0)))
        else:
            err = float(np.sum(r2))
        p = np.asarray(pvec, dtype=np.float64)
        if self.shear_cost > 0.0:
            err += self.shear_cost * float(p[0]) ** 2
        if self.cubic_cost > 0.0:
            err += self.cubic_cost * (float(p[6]) ** 2 + float(p[7]) ** 2)
        return err


@dataclass
class LMResult:
    x: np.ndarray
    fun: float
    iterations: int
    evaluations: int


def _rodrigues(rvec) -> np.ndarray:
    """Axis-angle → 3×3 rotation (same convention as cv2.Rodrigues)."""
    r = np.asarray(rvec, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return (np.eye(3) + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K))


def _rodrigues_derivatives(rvec, R: np.ndarray) -> np.ndarray:
    """∂R/∂rᵢ for the axis-angle vector (Gallego & Yezzi 2015, eq. 10):

        ∂R/∂rᵢ = ( rᵢ[r]ₓ + [ r × ((I−R)eᵢ) ]ₓ ) / ‖r‖² · R

    with the θ→0 limit [eᵢ]ₓ. Returns a (3, 3, 3) array indexed [i, row, col]."""
    r = np.asarray(rvec, dtype=np.float64).ravel()
    theta2 = float(r @ r)

    def skew(v):
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    if theta2 < 1e-14:
        return np.stack([skew(e) for e in np.eye(3)])
    rx = skew(r)
    out = np.empty((3, 3, 3))
    for i in range(3):
        imr = np.eye(3)[i] - R[:, i]           # (I − R)·eᵢ
        m = (r[i] * rx + skew(np.cross(r, imr))) / theta2
        out[i] = m @ R
    return out


# Accelerate's vectorised matmul raises SPURIOUS FPE warnings (divide-by-zero /
# invalid) on perfectly finite operands — reproducible with a bare random
# (8, N) @ (N, 8) product. Correctness here is guarded by the explicit isfinite
# checks on the step, not by the flags, so silence them for the solver.
@np.errstate(divide="ignore", invalid="ignore", over="ignore")
def minimize(objective: Objective, x0, *, max_iterations: int = 60,
             initial_lambda: float = 1e-3, cost_tolerance: float = 1e-6,
             freeze_curl: bool = False) -> LMResult:
    """Damped Gauss-Newton (Levenberg-Marquardt) on `objective`, exploiting
    the arrowhead/Schur sparsity described in the module docstring.

    `freeze_curl` zeroes the α/β Jacobian columns and pins their step, so the
    solver fits pose (and the page dims that follow) around a caller-fixed
    curl — used by the divergence retry and by manual-curl overrides."""
    n_cam = objective.n_cam
    span_counts = np.asarray(objective.span_counts, dtype=int)
    nspans = span_counts.size
    # Row 0 is the page origin (globals only); local rows start at 1 and are
    # contiguous per span, so every per-span reduction is one reduceat.
    if nspans == 0 or np.any(span_counts <= 0):
        raise ValueError(f"LM needs >= 1 keypoint per span, got {span_counts}")
    starts = np.concatenate([[0], np.cumsum(span_counts)[:-1]])
    sid = np.repeat(np.arange(nspans), span_counts)

    p = np.asarray(x0, dtype=np.float64).copy()
    dim = p.size
    cost = objective(p)
    lam = float(initial_lambda)
    rejects = 0
    evals = 1
    iters = 0
    stall = 0
    delta = np.zeros(dim)

    # Accelerate's vectorised matmul raises SPURIOUS FPE flags (divide-by-zero
    # / invalid) on perfectly finite operands — reproducible with a random
    # (8, N) @ (N, 8) product. Correctness is guarded by the explicit isfinite
    # checks below, not by the flags, so silence them for the numeric core.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
      while iters < max_iterations:
          iters += 1
          # ---- Jacobian + normal-equation assembly (one O(N) pass) ----------
          ex, ey, geo = objective.residuals(p)
          x, y, z, zp, _alpha, _beta, R, pc0, pc1, inv_z, fz = geo

          # IRLS: Gauss-Newton on the pseudo-Huber loss is plain least squares
          # with weight dρ/d(r²); the accept test still uses the robust cost.
          w = (objective.weights if objective.weights is not None
               else np.ones_like(ex))
          if objective.huber_delta > 0.0:
              d2 = objective.huber_delta ** 2
              r2 = w * (ex * ex + ey * ey)
              w = w / np.sqrt(1.0 + r2 / d2)
          s_w = np.sqrt(w)

          dR = _rodrigues_derivatives(p[0:3], R)
          alpha_raw, beta_raw = float(p[6]), float(p[7])
          # Clip handled like autodiff-of-clamp: zero derivative when railed.
          a_active = 0.0 if (freeze_curl or not (-CURL_CLIP < alpha_raw < CURL_CLIP)) else 1.0
          b_active = 0.0 if (freeze_curl or not (-CURL_CLIP < beta_raw < CURL_CLIP)) else 1.0

          ex = s_w * ex
          ey = s_w * ey
          jx3 = -fz * pc0 * inv_z
          jy3 = -fz * pc1 * inv_z

          def proj(v):
              """(N,3) world-space column → weighted image-space Jacobian pair."""
              return (s_w * (fz * v[..., 0] + jx3 * v[..., 2]),
                      s_w * (fz * v[..., 1] + jy3 * v[..., 2]))

          p3 = np.stack([x, y, z], axis=-1)                    # (N, 3)
          JG = np.zeros((n_cam, 2, objective.n))
          for a in range(3):
              JG[a] = proj(p3 @ dR[a].T)
          JG[3, 0] = s_w * fz
          JG[4, 1] = s_w * fz
          JG[5, 0] = s_w * jx3
          JG[5, 1] = s_w * jy3
          dz_a = (x ** 3 - 2.0 * x ** 2 + x) * a_active
          dz_b = (x ** 3 - x ** 2) * b_active
          rz = R[:, 2]
          JG[6] = proj(dz_a[:, None] * rz)
          JG[7] = proj(dz_b[:, None] * rz)
          if n_cam > 8:
              JG[8, 0] = s_w
              JG[9, 1] = s_w

          # local columns: ∂X/∂y_i = R·e₂, ∂X/∂x_ij = R·(1, 0, z′)
          jy_x, jy_y = proj(np.broadcast_to(R[:, 1], (objective.n, 3)))
          jx_vec = R[:, 0] + zp[:, None] * R[:, 2]
          jx_x, jx_y = proj(jx_vec)

          U = np.einsum("aci,bci->ab", JG, JG)
          gg = np.einsum("aci,ci->a", JG, np.stack([ex, ey]))
          # shear penalty: E += λs·p₀² → U₀₀ += λs, g₀ += λs·p₀
          U[0, 0] += objective.shear_cost
          gg[0] += objective.shear_cost * float(p[0])
          if objective.cubic_cost > 0.0 and not freeze_curl:
              # L2 on the RAW slopes — the clamp does not gate this penalty
              # (it is what pulls a railed coefficient back inside).
              for c in (6, 7):
                  U[c, c] += objective.cubic_cost
                  gg[c] += objective.cubic_cost * float(p[c])

          # ---- per-span arrowhead blocks (origin row carries no locals) -----
          JGl = JG[:, :, 1:]
          jyx, jyy = jy_x[1:], jy_y[1:]
          jxx, jxy = jx_x[1:], jx_y[1:]
          exl, eyl = ex[1:], ey[1:]
          a_i = np.add.reduceat(jyx * jyx + jyy * jyy, starts)
          g_y = np.add.reduceat(jyx * exl + jyy * eyl, starts)
          WY = np.add.reduceat(JGl[:, 0] * jyx + JGl[:, 1] * jyy, starts, axis=1)
          b_j = jyx * jxx + jyy * jxy
          c_j = jxx * jxx + jxy * jxy
          g_x = jxx * exl + jxy * eyl
          WX = JGl[:, 0] * jxx + JGl[:, 1] * jxy          # (n_cam, N_local)

          while True:
              # ---- damped arrowhead elimination (Sherman-Morrison) ----------
              c_d = c_j * (1.0 + lam) + 1e-15
              f_j = b_j / c_d
              s_i = a_i * (1.0 + lam) + 1e-15 - np.add.reduceat(b_j * f_j, starts)
              hat_g = g_y - np.add.reduceat(f_j * g_x, starts)
              HW = WY - np.add.reduceat(f_j * WX, starts, axis=1)

              # ---- Schur complement: S = U − Σ WxWxᵀ/c̃ − Σ ŴŴᵀ/s ----------
              Sc = U.copy()
              # Multiplicative Marquardt damping + a relative ridge: a railed
              # α/β has a structurally ZERO row (autodiff-of-clamp), which
              # scaling alone leaves singular. The ridge makes that row solve
              # to a zero step instead — exactly the clamp semantics.
              d_u = np.diag(Sc).copy()
              ridge = 1e-12 * max(1.0, float(np.max(np.abs(d_u))))
              np.fill_diagonal(Sc, d_u * (1.0 + lam) + ridge)
              Sc -= (WX / c_d) @ WX.T
              Sc -= (HW / s_i) @ HW.T
              rhs = -gg + WX @ (g_x / c_d) + HW @ (hat_g / s_i)
              if freeze_curl:
                  # Zeroed curl columns would leave the system singular.
                  for f in (6, 7):
                      Sc[f, :] = 0.0
                      Sc[:, f] = 0.0
                      Sc[f, f] = 1.0
                      rhs[f] = 0.0
              try:
                  dg = np.linalg.solve(Sc, rhs)
              except np.linalg.LinAlgError:
                  dg = None
              if dg is not None and np.all(np.isfinite(dg)):
                  break
              # A singular system will not heal by damping alone — count it
              # like a rejection (measured on iOS: 53/60 iterations spun here
              # on one page) but retry the cheap elimination at higher λ before
              # paying for another Jacobian pass.
              lam *= 4.0
              rejects += 1
              if rejects >= 6 or lam > 1e10:
                  dg = None
                  break
          if dg is None:
              break

          # ---- back-substitution -------------------------------------------
          delta[:] = 0.0
          delta[:n_cam] = dg
          d_y = -(hat_g + HW.T @ dg) / s_i
          delta[n_cam:n_cam + nspans] = d_y
          delta[n_cam + nspans:] = -(g_x + b_j * d_y[sid] + WX.T @ dg) / c_d

          # ---- trust region -------------------------------------------------
          p_new = p + delta
          if not np.all(np.isfinite(p_new)):
              lam *= 4.0
              rejects += 1
              if rejects >= 6 or lam > 1e10:
                  break
              continue
          new_cost = objective(p_new)
          evals += 1
          if new_cost < cost:
              rel = (cost - new_cost) / max(cost, 1e-300)
              p = p_new
              cost = new_cost
              lam = max(lam / 3.0, 1e-12)
              rejects = 0
              if rel < cost_tolerance:
                  stall += 1
                  if stall >= 2:
                      break
              else:
                  stall = 0
          else:
              lam *= 2.5
              # `stall` only counts ACCEPTED steps, so a rejection-dominated
              # solve would otherwise thrash to max_iterations: six consecutive
              # rejections (λ grown ×244) means the direction is dead.
              rejects += 1
              if rejects >= 6 or lam > 1e10:
                  break

    return LMResult(x=p, fun=float(cost), iterations=iters, evaluations=evals)
