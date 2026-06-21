# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Geometry helpers for projective rectification.

Pure-numpy primitives reused by `TrapezoidalCorrection` and (later) the Qt
manual-corner dialog. Kept side-effect free and test-friendly: every public
function takes plain numpy arrays and returns plain numpy arrays.

Conventions
-----------
- Points are 2D `(x, y)` in image pixel coordinates. Homogeneous extensions
  carry an explicit `w` column.
- Lines are 3-vectors `(a, b, c)` with the implicit constraint `a x + b y + c = 0`.
- Homographies are 3x3, applied as `m' = H @ m`. They map images via
  `cv2.warpPerspective(img, H, ...)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ───────────────────────────────────────────────────────────────────────────
# Basic projective primitives
# ───────────────────────────────────────────────────────────────────────────

def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    """Append a 1-column to a (N, 2) array of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        return np.append(pts, 1.0)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.concatenate([pts, ones], axis=1)


def line_through(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Homogeneous line through two image points. Returns (3,) vector (a, b, c).

    Both inputs may be plain `(x, y)` or homogeneous `(x, y, w)`.
    """
    a = to_homogeneous(np.asarray(p1, dtype=np.float64).reshape(-1)[:2])
    b = to_homogeneous(np.asarray(p2, dtype=np.float64).reshape(-1)[:2])
    return np.cross(a, b)


def normalize_line(line: np.ndarray) -> np.ndarray:
    """Scale a line so (a, b) has unit norm. Lets us interpret distances directly."""
    line = np.asarray(line, dtype=np.float64).reshape(3)
    n = math.hypot(line[0], line[1])
    if n < 1e-12:
        return line
    return line / n


def point_line_distance(point: np.ndarray, line: np.ndarray) -> float:
    """Perpendicular distance from a 2D point to a normalised line."""
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    if p.size == 2:
        x, y = p
    else:
        x, y, w = p
        x /= w; y /= w
    a, b, c = normalize_line(line)
    return abs(a * x + b * y + c)


def line_angle_deg(line: np.ndarray) -> float:
    """Angle of a 2D line in degrees, in `[-90, 90)`.

    Defined as the direction of the line (perpendicular to its normal),
    so that horizontal lines return 0.
    """
    a, b, _ = np.asarray(line, dtype=np.float64).reshape(3)
    # Direction vector is (-b, a). Use atan2 then fold to [-90, 90).
    ang = math.degrees(math.atan2(a, -b))
    while ang >= 90.0:
        ang -= 180.0
    while ang < -90.0:
        ang += 180.0
    return ang


# ───────────────────────────────────────────────────────────────────────────
# Text-line geometry primitives
# ───────────────────────────────────────────────────────────────────────────

def bbox_baseline(bbox: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Baseline endpoints from an axis-aligned bbox `(x, y, w, h)`.

    Returns `(p_left, p_right)` — the two endpoints of the bottom edge.

    NOTE: This gives a horizontal line at y_max regardless of actual text
    slope — useless for VP estimation. Use `baseline_from_ink` instead
    when working with binary text crops.
    """
    x, y, w, h = bbox
    return np.array([x, y + h - 1], dtype=np.float64), \
           np.array([x + w - 1, y + h - 1], dtype=np.float64)


def baseline_from_ink(ink: np.ndarray, bbox: Sequence[int],
                      *, min_samples: int = 8,
                      span_mask: Optional[np.ndarray] = None,
                      ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Robustly fit a baseline through the bottom contour of a text line.

    For each column x inside `bbox`, find the bottom-most ink pixel
    (ink > 127, with ink as text-on-black). Filter outliers (descenders
    are dropped via a windowed lower-quantile pass), then RANSAC-fit a
    line through the remaining `(x, y_bottom(x))` samples.

    span_mask: optional binary mask same shape as ink. When provided,
    only ink pixels where span_mask is non-zero are considered — used
    to keep this span's RANSAC fit unaffected by ink from neighbouring
    text lines that happen to fall inside the same bbox (curl, morph
    bridging, or skewed bboxes).

    Returns `(p_left, p_right)` — two points spanning the line — or None
    when there is too little data.
    """
    x, y, w, h = bbox
    crop = ink[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    if span_mask is not None:
        crop = np.where(span_mask[y:y + h, x:x + w] > 0, crop, 0)
    # Per-column lowest ink row. -1 sentinel for columns with no ink.
    mask = crop > 127
    if not mask.any():
        return None
    rows = np.arange(crop.shape[0])[:, None]
    rows_when_ink = np.where(mask, rows, -1)
    bottoms = rows_when_ink.max(axis=0)  # (w,) ints, -1 = empty column
    cols = np.arange(w)
    sel = bottoms >= 0
    if sel.sum() < min_samples:
        return None
    xs = cols[sel].astype(np.float64)
    ys = bottoms[sel].astype(np.float64)

    # Drop descenders: per windowed segment, anything more than ~25 % of
    # line height below the local median is treated as a descender and
    # excluded. Cheap median-filter pass.
    if ys.size >= 24:
        win = max(8, ys.size // 16)
        from numpy.lib.stride_tricks import sliding_window_view
        if ys.size >= win:
            windowed = sliding_window_view(ys, win)
            local_med = np.median(windowed, axis=1)
            # Pad so it lines up with ys.
            pad = win - 1
            local_med = np.concatenate([
                np.full(pad // 2, local_med[0]),
                local_med,
                np.full(pad - pad // 2, local_med[-1]),
            ])
            descender_thresh = local_med + 0.25 * h
            keep = ys <= descender_thresh
            if keep.sum() >= min_samples:
                xs = xs[keep]
                ys = ys[keep]

    # RANSAC-fit a line through (xs, ys). At each trial pick 2 points,
    # count inliers within `eps` pixels. eps = 0.15 × h lets RANSAC fit
    # a single line through perspective-tilted baselines whose end-to-end
    # drop exceeds the tight 0.05×h band (a ~3° tilt over 2000 px drops
    # ~100 px, which a 10-px band rejects, fragmenting the inlier set
    # into a near-horizontal sub-cluster that underestimates the slope).
    rng = np.random.default_rng(0)
    n = xs.size
    best_inliers = np.array([], dtype=np.int64)
    eps = max(1.5, 0.15 * h)
    for _ in range(60):
        i, j = rng.choice(n, size=2, replace=False)
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]
        if abs(x2 - x1) < 1e-3:
            continue
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        residuals = np.abs(ys - (a * xs + b))
        inliers = np.where(residuals < eps)[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            if inliers.size > 0.85 * n:
                break

    if best_inliers.size < min_samples:
        return None
    # Least-squares re-fit on inliers.
    A = np.column_stack([xs[best_inliers], np.ones(best_inliers.size)])
    sol, *_ = np.linalg.lstsq(A, ys[best_inliers], rcond=None)
    a, b = sol
    # Two endpoints in the bbox-relative frame, translated back to image coords.
    x_lo, x_hi = xs.min(), xs.max()
    pL = np.array([x + x_lo, y + (a * x_lo + b)], dtype=np.float64)
    pR = np.array([x + x_hi, y + (a * x_hi + b)], dtype=np.float64)
    return pL, pR


def xband_baseline_per_col(mask: np.ndarray) -> np.ndarray:
    """Per-column **baseline-y** that *follows curl* and rejects descenders.

    Three steps:

      1. Per column, take the bottom-most ink row (`bot[c]`). For
         x-height letters this is the baseline; for descenders
         (`p, g, q, j, …`) it's the descender bottom — wrong, but we'll
         filter those out next.

      2. Estimate the *local baseline trend* across the contour with a
         rolling median of `bot`. Descender columns are isolated
         outliers (a single `p` won't bend the local median); the
         baseline curl IS the median trend.

      3. Drop columns where `bot[c]` sits more than `0.25 * x_height`
         below the local median — those are descenders. Surviving
         columns carry true-baseline y values, varying with page curl
         so the cubic-sheet optimiser sees real curvature instead of
         a flat clipped line.

    Why this beats the previous global-band-clip approach: that one
    capped `bot[c]` at `band_hi` (a contour-global row), which flattened
    real curl wherever the baseline curved past `band_hi` — visible as
    a perfectly straight red baseline through a clearly-curling line.

    Inputs:
        mask  H×W binary (0/1 or 0/255 — only `>0` matters).

    Returns:
        baselines  W-vector. NaN in (a) empty columns and (b) columns
                   identified as descenders. Caller filters NaNs.

    Fallback: contour too small for a robust median (< 16 valid cols) →
    raw `bot`; the optimiser swallows a few noisy contours.
    """
    if mask.ndim != 2 or mask.size == 0:
        return np.full((mask.shape[1] if mask.ndim == 2 else 0,),
                       np.nan, dtype=np.float32)
    bin_mask = (mask > 0)
    H, W = bin_mask.shape

    def _bottommost(m: np.ndarray) -> np.ndarray:
        any_ink = m.any(axis=0)
        offsets = np.argmax(m[::-1], axis=0)
        bottom = (m.shape[0] - 1) - offsets
        return np.where(any_ink, bottom.astype(np.float32), np.nan)

    def _topmost(m: np.ndarray) -> np.ndarray:
        any_ink = m.any(axis=0)
        top = np.argmax(m, axis=0)
        return np.where(any_ink, top.astype(np.float32), np.nan)

    bot = _bottommost(bin_mask)
    valid = np.isfinite(bot)
    valid_idx = np.where(valid)[0]
    n_valid = valid_idx.size
    if n_valid < 16:
        return bot  # too sparse to median-filter

    top = _topmost(bin_mask)
    # x-height ≈ median per-column ink height. Descenders inflate this
    # slightly; median is robust enough.
    heights = bot[valid] - top[valid]
    h_med = float(np.median(heights))
    if h_med <= 0:
        return bot

    bot_valid = bot[valid]

    # Rolling median over a window sized to letter spacing — wide enough
    # to span single descender columns, narrow enough to track curl.
    # ~1/8 of the contour's valid-column count is a good default.
    win = max(8, n_valid // 8)
    win = min(win, n_valid)
    from numpy.lib.stride_tricks import sliding_window_view
    windowed = sliding_window_view(bot_valid, win)
    local_med = np.median(windowed, axis=1)
    pad = win - 1
    pad_lo = pad // 2
    pad_hi = pad - pad_lo
    local_med = np.concatenate([
        np.full(pad_lo, local_med[0]),
        local_med,
        np.full(pad_hi, local_med[-1]),
    ])

    descender_thresh = local_med + 0.25 * h_med
    keep_in_valid = bot_valid <= descender_thresh

    # Edge fragment filter: near the contour's left/right end (within
    # `h_med` pixels of the first/last valid column), drop columns whose
    # ink height is below `0.5 * h_med`. Targets stray short bits that
    # tend to live at the edge of a contour — apostrophes, footnote
    # markers, leading `-` dashes, trailing commas. They bias the
    # baseline up (no body to anchor it to) and the rolling median can
    # only do so much when they sit alone outside the body span.
    heights_valid = bot[valid] - top[valid]
    left_edge = int(valid_idx[0])
    right_edge = int(valid_idx[-1])
    edge_window = h_med
    edge_zone = ((valid_idx - left_edge) < edge_window) | \
                ((right_edge - valid_idx) < edge_window)
    short = heights_valid < 0.5 * h_med
    edge_short = edge_zone & short
    keep_in_valid = keep_in_valid & (~edge_short)

    out = np.full(W, np.nan, dtype=np.float32)
    out[valid_idx[keep_in_valid]] = bot_valid[keep_in_valid]
    return out


def span_bottom_series(cinfos, side: str = "bottom"
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-column ink-extremum samples over all contours of
    a text-line span.

    `cinfos`: iterable of objects with `.mask` (H×W binary) and
    `.rect` (xmin, ymin, …) — duck-typed to page_dewarp's ContourInfo.

    `side`: "bottom" → bottom-most ink y per column (baseline signal,
    descenders are the outliers); "top" → top-most ink y per column
    (x-height topline signal, ascenders / capitals / accents are the
    outliers).

    Returns `(xs, ys, heights)` in *global* (span-image) coords:
        xs       (N,) int64    column x of each ink column
        ys       (N,) float64  bottom- or top-most ink y per column
        heights  (N,) float64  per-column ink height (bot − top)
    """
    xs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    h_all: list[np.ndarray] = []
    for ci in cinfos:
        m = getattr(ci, "mask", None)
        if m is None or m.size == 0 or m.ndim != 2:
            continue
        b = (m > 0)
        any_ink = b.any(axis=0)
        if not any_ink.any():
            continue
        bot = (b.shape[0] - 1) - np.argmax(b[::-1], axis=0)
        top = np.argmax(b, axis=0)
        cols = np.where(any_ink)[0]
        xmin, ymin = int(ci.rect[0]), int(ci.rect[1])
        sel = top if side == "top" else bot
        xs_all.append(cols.astype(np.int64) + xmin)
        y_all.append(sel[cols].astype(np.float64) + ymin)
        h_all.append((bot[cols] - top[cols]).astype(np.float64))
    if not xs_all:
        z = np.empty(0)
        return np.empty(0, np.int64), z, z
    return np.concatenate(xs_all), np.concatenate(y_all), np.concatenate(h_all)


def fit_span_baseline(xs: np.ndarray, ys: np.ndarray, heights: np.ndarray,
                      *, degree: int = 3, iters: int = 3,
                      min_cols: int = 16, min_width_factor: float = 3.0):
    """Robust span-level baseline fit — IRLS polynomial with Tukey biweight.

    Replaces the per-contour drop-the-outliers strategy
    (`xband_baseline_per_col`): instead of discarding descender / edge
    columns (which truncated baselines at line ends, exactly where page
    curl is strongest), fit one smooth curve per text line and let the
    robust loss zero-out descenders (below) and floating glyphs like
    dashes / apostrophes (above). The fit is then evaluable at *every*
    column, so sampled keypoints reach the true line ends.

    Degree 3 matches the cubic-sheet dewarp model; narrow spans
    (< 6 × x-height) fall back to a line.

    Returns `numpy.polynomial.Polynomial` (call it with global x), or
    `None` when the span is too small/degenerate — caller falls back to
    the legacy per-contour path.
    """
    from numpy.polynomial import Polynomial

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    n = xs.size
    if n < min_cols:
        return None
    h_med = float(np.median(heights))
    if h_med <= 0:
        return None
    width = float(xs.max() - xs.min())
    if width < min_width_factor * h_med:
        return None
    deg = int(degree) if width >= 6.0 * h_med else 1
    deg = min(deg, max(1, n - 2))
    # Tukey cutoff: true-baseline columns scatter within a few px;
    # descenders sit ~0.3–0.5 x-height below the trend.
    c = max(1.5, 0.3 * h_med)

    w = np.ones(n)
    p = None
    for _ in range(iters + 1):
        try:
            # Polynomial.fit weights multiply *unsquared* residuals →
            # pass sqrt of the IRLS weights.
            p = Polynomial.fit(xs, ys, deg, w=np.sqrt(w))
        except (np.linalg.LinAlgError, ValueError):
            return None
        r = ys - p(xs)
        u = r / c
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if np.count_nonzero(w) < min_cols:
            return None
    inliers = w > 0
    rms = float(np.sqrt(np.mean((ys[inliers] - p(xs[inliers])) ** 2)))
    if rms > 0.5 * h_med:
        return None
    return p


def dewarp_arclength_x(params, page_w: float, n: int = 4096
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative arc length of the cubic-sheet height profile.

    page-dewarp models the page as z(x) = (α+β)x³ − (2α+β)x² + αx but
    remaps with x as the flat-page coordinate. Paper is inextensible —
    the true flat coordinate is the arc length s(x) = ∫√(1+z′²)dx, so
    a uniform-x remap stretches output text horizontally by √(1+z′²)
    wherever the sheet is steep (gutter side).

    Returns `(xs, s)` — n model-x samples over [0, page_w] and their
    cumulative arc length. Callers build an arc-length-uniform x grid
    with `np.interp(np.linspace(0, s[-1], w), s, xs)` and size the
    output width from `s[-1]` instead of `page_w`.
    """
    alpha = float(np.clip(params[6], -0.5, 0.5))
    beta = float(np.clip(params[7], -0.5, 0.5))
    xs = np.linspace(0.0, float(page_w), int(n))
    zp = 3.0 * (alpha + beta) * xs ** 2 - 2.0 * (2.0 * alpha + beta) * xs + alpha
    ds = np.sqrt(1.0 + zp * zp)
    s = np.concatenate([[0.0],
                        np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(xs))])
    return xs, s


def lines_from_segments(segments: np.ndarray) -> np.ndarray:
    """Convert an (N, 4) array of `(x1, y1, x2, y2)` segments to (N, 3) homogeneous lines."""
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    out = np.empty((segments.shape[0], 3), dtype=np.float64)
    for i, (x1, y1, x2, y2) in enumerate(segments):
        out[i] = line_through((x1, y1), (x2, y2))
    return out


# ───────────────────────────────────────────────────────────────────────────
# Vanishing point estimation
# ───────────────────────────────────────────────────────────────────────────

def vp_from_lines_svd(lines: np.ndarray) -> np.ndarray:
    """Total least-squares vanishing point from N lines.

    `lines`: (N, 3) homogeneous lines. Returns a (3,) homogeneous point.
    The VP is the right singular vector of the line matrix corresponding to
    the smallest singular value — that is, the point closest to lying on
    every line in the least-squares sense.
    """
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    if lines.shape[0] < 2:
        raise ValueError("vp_from_lines_svd needs at least 2 lines")
    # Normalise each line so its weighting is uniform.
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    A = lines / norms
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    vp = vt[-1]  # smallest singular vector
    # Convention: keep w >= 0 so downstream code can read x / w consistently.
    if vp[2] < 0:
        vp = -vp
    return vp


def _line_through_vp_and_point(vp: np.ndarray, point: np.ndarray) -> np.ndarray:
    """The line joining a homogeneous VP and a 2D point."""
    p = to_homogeneous(np.asarray(point).reshape(-1)[:2])
    return np.cross(vp, p)


def ransac_vp(
    lines: np.ndarray,
    *,
    trials: int = 200,
    eps_deg: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC vanishing-point fit on a set of homogeneous lines.

    Each input line is also represented by a midpoint sample (mean of the
    two endpoints, recovered from line × (image x-axis) is not stable, so
    we expect the caller to pass `lines` paired with the segment array).

    Lighter variant: each iteration picks two random lines, intersects to
    form a candidate VP, and counts how many other lines pass within
    `eps_deg` of it. Final VP is re-estimated by SVD on the inliers.

    Returns
    -------
    vp : (3,) ndarray or None
        Best-found vanishing point, or None if the inlier set ends up
        below 3 lines.
    inlier_idxs : (M,) int ndarray
        Indices of lines in the inlier set.
    """
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    N = lines.shape[0]
    if N < 3:
        if N == 2:
            return np.cross(lines[0], lines[1]), np.array([0, 1])
        return None, np.array([], dtype=np.int64)

    rng = rng or np.random.default_rng(12345)
    # Pre-normalise lines so the angle check is direct.
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    L = lines / norms

    best_inliers: np.ndarray = np.array([], dtype=np.int64)
    cos_eps = math.cos(math.radians(eps_deg))

    thresh = math.sin(math.radians(eps_deg))
    for _ in range(trials):
        i, j = rng.choice(N, size=2, replace=False)
        vp = np.cross(L[i], L[j])
        vp_norm = np.linalg.norm(vp)
        if vp_norm < 1e-12:
            continue
        vp_unit = vp / vp_norm
        # With L having unit (a, b) and vp_unit being a unit 3-vector,
        # |L_i . vp_unit| ≈ sin(angle between line and VP direction) on the
        # projective sphere — a scale-free residual.
        residuals = np.abs(L @ vp_unit)
        inliers = np.where(residuals < thresh)[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            if inliers.size > 0.9 * N:
                break

    if best_inliers.size < 3:
        return None, best_inliers

    vp = vp_from_lines_svd(L[best_inliers])
    return vp, best_inliers


# ───────────────────────────────────────────────────────────────────────────
# Rectification homographies
# ───────────────────────────────────────────────────────────────────────────

def affine_rect_from_line(vanishing_line: np.ndarray) -> np.ndarray:
    """3x3 affine-rectification homography for a given image-of-the-infinity-line.

    Given `ℓ∞' = (l1, l2, l3)`, returns

            ⎡ 1   0   0 ⎤
       H =  ⎢ 0   1   0 ⎥
            ⎣ l1  l2  l3⎦

    which maps `ℓ∞'` back to the canonical `(0, 0, 1)`. Renormalised so
    `l3 = 1` when possible.
    """
    l = np.asarray(vanishing_line, dtype=np.float64).reshape(3)
    if abs(l[2]) > 1e-12:
        l = l / l[2]
    H = np.eye(3, dtype=np.float64)
    H[2] = l
    return H


def horizontal_vanishing_line(vp_x: np.ndarray, y_centroid: float,
                              infinity_eps: float = 1e-6) -> np.ndarray:
    """Vanishing line for the affine-only fallback (one VP known).

    When `vp_x` is essentially at infinity (baselines already parallel in
    the image — no horizontal perspective), return the canonical
    line-at-infinity `(0, 0, 1)`, so `affine_rect_from_line` produces an
    identity homography. Otherwise return a horizontal line through
    `vp_x.y / vp_x.w`.
    """
    xy_norm = max(abs(vp_x[0]), abs(vp_x[1]), 1.0)
    if abs(vp_x[2]) < infinity_eps * xy_norm:
        return np.array([0.0, 0.0, 1.0])
    y_v = vp_x[1] / vp_x[2]
    return np.array([0.0, 1.0, -y_v])


# ───────────────────────────────────────────────────────────────────────────
# Vertical VP from line-pitch constraint (Phase-2 metric upgrade)
# ───────────────────────────────────────────────────────────────────────────

def vertical_vp_from_line_pitch(
    baseline_ys: np.ndarray,
    line_indices: Optional[np.ndarray] = None,
    x_anchor: float = 0.0,
) -> Optional[np.ndarray]:
    """Recover the vertical VP using only the y-coordinates of successive baselines.

    Assumption: line *index* i is an affine function of the page-plane
    y-coordinate, so its image y obeys a 1D Mobius map

        i = (a * y + b) / (c * y + 1)

    Fit `(a, b, c)` by least squares to `(y_i, i)`. The image-y of the
    vertical vanishing point is then `y_v = -1/c` (the y at which `i`
    blows up — the direction of "+infinity many lines per unit").

    Parameters
    ----------
    baseline_ys : (N,) array of float
        y-coordinates of detected baselines, **sorted ascending**.
    line_indices : (N,) array of int, optional
        Index labels for each baseline. If omitted, use `0, 1, …, N-1`,
        which assumes the input is contiguous (no big gaps between lines).
    x_anchor : float
        x-coordinate to use for the VP. Typically the median x of all
        baselines (a "centre" of the page). Tilt is assumed to be along
        the y axis — fine for the common "looking down at desk" geometry.

    Returns
    -------
    (3,) homogeneous VP, or None if the fit is degenerate.
    """
    ys = np.asarray(baseline_ys, dtype=np.float64).reshape(-1)
    if line_indices is None:
        idx = np.arange(ys.size, dtype=np.float64)
    else:
        idx = np.asarray(line_indices, dtype=np.float64).reshape(-1)
    if ys.size < 4:
        return None  # underdetermined

    # Linearise `i = (a y + b) / (c y + 1)`  =>  i c y + i = a y + b
    # =>  a y + b - i c y - i = 0
    # =>  [ y  1  -i*y ] [a, b, c]^T = i
    A = np.column_stack([ys, np.ones_like(ys), -idx * ys])
    sol, *_ = np.linalg.lstsq(A, idx, rcond=None)
    a, b, c = sol
    if abs(c) < 1e-10:
        return None  # essentially constant pitch → VP at infinity already
    y_v = -1.0 / c
    return np.array([x_anchor, y_v, 1.0])


# ───────────────────────────────────────────────────────────────────────────
# Composite warp utilities
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class WarpedExtent:
    """Bounding-box of an image after applying a homography."""
    H_corrected: np.ndarray   # H with a translation prepended so output sits at (0, 0)
    width: int
    height: int


def warp_extent(H: np.ndarray, src_w: int, src_h: int, margin_px: int = 0) -> WarpedExtent:
    """Compute the bbox of `H @ src_corners` and return a translation-prefixed H.

    Useful to pass `(W, H)` to `cv2.warpPerspective` without clipping.
    """
    corners = np.array([
        [0, 0, 1],
        [src_w - 1, 0, 1],
        [src_w - 1, src_h - 1, 1],
        [0, src_h - 1, 1],
    ], dtype=np.float64).T  # 3x4
    mapped = H @ corners
    mapped /= mapped[2:3]
    xs = mapped[0]
    ys = mapped[1]
    x_min = math.floor(xs.min()) - margin_px
    y_min = math.floor(ys.min()) - margin_px
    x_max = math.ceil(xs.max()) + margin_px
    y_max = math.ceil(ys.max()) + margin_px
    T = np.array([
        [1.0, 0.0, -x_min],
        [0.0, 1.0, -y_min],
        [0.0, 0.0, 1.0],
    ])
    return WarpedExtent(T @ H, int(x_max - x_min), int(y_max - y_min))


def oob_fraction(H: np.ndarray, src_w: int, src_h: int,
                 dst_w: int, dst_h: int, samples: int = 2000,
                 rng: Optional[np.random.Generator] = None) -> float:
    """Fraction of source-derived destination samples whose H^-1 leaves the source.

    Samples are drawn uniformly inside the **forward-mapped source quad**
    (not the whole destination canvas). Padding/margin pixels that the
    warp fills via `BORDER_REPLICATE` are intentional, not OOB, so they
    must not count.

    Cheap quality check for the homography — analog of PageDewarper's
    `oob` stats.
    """
    rng = rng or np.random.default_rng(1)
    # Forward-map the source quad to get a 4-vertex polygon in dst space.
    src_corners = np.array([
        [0, 0, 1],
        [src_w - 1, 0, 1],
        [src_w - 1, src_h - 1, 1],
        [0, src_h - 1, 1],
    ], dtype=np.float64).T
    mapped = H @ src_corners
    mapped /= mapped[2:3]
    quad = mapped[:2].T  # (4, 2) in dst pixel coords

    H_inv = np.linalg.inv(H)
    # Reject-sample uniform points inside the quad's bounding rect.
    x_lo, y_lo = quad.min(axis=0)
    x_hi, y_hi = quad.max(axis=0)
    accepted_x: list[float] = []
    accepted_y: list[float] = []
    # Heuristic: oversample 3x then keep those inside the quad.
    while len(accepted_x) < samples:
        xs = rng.uniform(x_lo, x_hi, samples * 3)
        ys = rng.uniform(y_lo, y_hi, samples * 3)
        mask = _points_in_quad(xs, ys, quad)
        accepted_x.extend(xs[mask].tolist())
        accepted_y.extend(ys[mask].tolist())
    xs = np.asarray(accepted_x[:samples])
    ys = np.asarray(accepted_y[:samples])
    P = np.stack([xs, ys, np.ones(samples)], axis=0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        M = H_inv @ P
        M = M / M[2:3]
        inside = (M[0] >= 0) & (M[0] < src_w) & (M[1] >= 0) & (M[1] < src_h)
        inside = np.where(np.isfinite(inside), inside, False)
    return float(1.0 - inside.mean())


# ───────────────────────────────────────────────────────────────────────────
# Column-rectangle (Zhang–He style) detection on a binarized text crop
# ───────────────────────────────────────────────────────────────────────────

def _fit_line_ransac(points: np.ndarray, *,
                     trials: int = 200, eps_px: float = 18.0,
                     rng: Optional[np.random.Generator] = None,
                     ) -> Optional[np.ndarray]:
    """RANSAC straight-line fit through 2D `points` → homogeneous line.

    Picks the candidate line passing through the maximum-cardinality
    inlier set (perpendicular distance < `eps_px`). Re-fits TLS on
    those inliers. Returns None when fewer than 3 inliers survive.

    Use over `_fit_line_lstsq` when the point cloud is contaminated by
    structurally different sub-populations — e.g. blockquote rights
    sitting inside the body's right margin (scan-43 B): TLS would
    blend, RANSAC sticks to the dominant cluster.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n < 3:
        return None
    rng = rng or np.random.default_rng(7)
    best_inliers = np.array([], dtype=np.int64)
    for _ in range(trials):
        i, j = rng.choice(n, size=2, replace=False)
        p1, p2 = pts[i], pts[j]
        d = p2 - p1
        norm = np.hypot(d[0], d[1])
        if norm < 1e-6:
            continue
        # Normal of the line = (-dy, dx) / |d|.
        nrm = np.array([-d[1], d[0]]) / norm
        # Signed perpendicular distance to every point.
        dist = np.abs((pts - p1) @ nrm)
        inliers = np.where(dist < eps_px)[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            if inliers.size > 0.85 * n:
                break
    if best_inliers.size < 3:
        return None
    return _fit_line_lstsq(pts[best_inliers])


def _theil_sen_slope_bounded(points: np.ndarray, *,
                             max_tilt_deg: float = 7.0,
                             max_pairs: int = 2000,
                             rng: Optional[np.random.Generator] = None,
                             ) -> Optional[float]:
    """Pairwise-median slope b of x = a + b·y over 2D `points`.

    Margin endpoints are a mix of parallel sub-populations (body margin,
    blockquote indent — both near-vertical, both sharing the page's
    vertical vanishing direction), so the *slope* is consensus-stable
    even when the *intercept* is bimodal: same-cluster pairs vote the
    true slope, cross-cluster pairs scatter, the median lands on the
    consensus. Pairs with |Δy| < span/4 are skipped (slope noise blows
    up as 1/Δy). Result clipped to |b| ≤ tan(max_tilt_deg) — the page
    is deskewed upstream, near-vertical is a hard prior.

    Returns None when no usable pair exists (degenerate y-span)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        return None
    y = pts[:, 1]
    x = pts[:, 0]
    span = float(y.max() - y.min())
    if span < 1e-6:
        return None
    dy_floor = span / 4.0
    slopes: list[float] = []
    if n * (n - 1) // 2 <= max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        rng = rng or np.random.default_rng(7)
        ii = rng.integers(0, n, size=max_pairs)
        jj = rng.integers(0, n, size=max_pairs)
    dy = y[jj] - y[ii]
    keep = np.abs(dy) >= dy_floor
    if not np.any(keep):
        return None
    b = (x[jj][keep] - x[ii][keep]) / dy[keep]
    b_med = float(np.median(b))
    bound = math.tan(math.radians(max_tilt_deg))
    return float(np.clip(b_med, -bound, bound))


def fit_margin_line_clustered(points: np.ndarray, *, side: str,
                              cluster_gap_px: float,
                              max_tilt_deg: float = 7.0,
                              min_support: int = 3,
                              min_support_frac: float = 0.25,
                              ) -> Optional[tuple[np.ndarray, dict]]:
    """Slope-bounded, cluster-aware column-margin fit.

    Replaces hard-anchor-box + max-cardinality RANSAC, both of which
    fail on blockquote-heavy pages (see docs/algorithms.md §3.6.7):
    the anchor box can't distinguish perspective drift from indentation
    in raw x, and unconstrained RANSAC prefers a *tilted* band mixing
    body-margin and blockquote endpoints over the true near-vertical
    margin cluster (athanase 070 B / 256 A / 198 B / 191 B / 161 A).

    1. Global slope b via bounded Theil-Sen over ALL endpoints (slope
       is consensus-stable across parallel sub-populations).
    2. De-drift: residual r_i = x_i − b·y_i collapses each straight
       margin trace to a tight 1-D cluster. Sort r, split at gaps
       > `cluster_gap_px`.
    3. Cluster choice = OUTERMOST WITH SUPPORT: among clusters with
       count ≥ max(min_support, min_support_frac · largest), pick the
       smallest median r (side="left") / largest (side="right").
       Outermost-supported beats max-cardinality (blockquote lines may
       outnumber body lines) and beats pure extremality (1-3 page-edge
       ink points fail the support floor — the scan-44 A regression
       that killed p95 seeding).
    4. TLS refit on the chosen cluster; if the refit slope strays
       > 2° from the global consensus (tiny cluster y-span), keep the
       global slope through the cluster's median intercept.

    Returns (homogeneous line, info) or None when degenerate; caller
    falls back to the legacy anchor + RANSAC path.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n < max(min_support, 3) or side not in ("left", "right"):
        return None
    b = _theil_sen_slope_bounded(pts, max_tilt_deg=max_tilt_deg)
    if b is None:
        return None
    r = pts[:, 0] - b * pts[:, 1]
    order = np.argsort(r)
    r_sorted = r[order]
    # Split sorted residuals into clusters at gaps > cluster_gap_px.
    gap = max(float(cluster_gap_px), 1e-6)
    breaks = np.where(np.diff(r_sorted) > gap)[0]
    bounds = np.concatenate([[0], breaks + 1, [n]])
    clusters = [order[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]
    largest = max(len(c) for c in clusters)
    support = max(min_support, int(math.ceil(min_support_frac * largest)))
    eligible = [c for c in clusters if len(c) >= support]
    if not eligible:
        return None
    med_r = [float(np.median(r[c])) for c in eligible]
    pick = int(np.argmin(med_r)) if side == "left" else int(np.argmax(med_r))
    members = eligible[pick]
    cluster_pts = pts[members]
    line = _fit_line_lstsq(cluster_pts)
    # Slope sanity: line (a, b_l, c) has direction (−b_l, a); tilt from
    # vertical = atan2(|dx|, |dy|). A tiny cluster y-span lets TLS pick
    # an arbitrary tilt — scan back to the consensus slope then.
    dx, dy = -line[1], line[0]
    if abs(dy) < 1e-9:
        tilt = 90.0
    else:
        tilt = math.degrees(math.atan(abs(dx / dy)))
    consensus_tilt = math.degrees(math.atan(abs(b)))
    if abs(tilt - consensus_tilt) > 2.0:
        m = float(np.median(r[members]))
        # x = b·y + m  →  x − b·y − m = 0.
        line = np.array([1.0, -b, -m], dtype=np.float64)
    info = {
        "members": members.tolist(),
        "n_clusters": len(clusters),
        "n_eligible": len(eligible),
        "cluster_size": int(len(members)),
        "median_r": float(np.median(r[members])),
        "slope": float(b),
    }
    return line, info


def margin_line_tilt_deg(line: np.ndarray) -> float:
    """Tilt of a homogeneous line from image-vertical, in degrees."""
    dx, dy = -float(line[1]), float(line[0])
    if abs(dy) < 1e-9:
        return 90.0
    return math.degrees(math.atan(abs(dx / dy)))


def _fit_line_lstsq(points: np.ndarray) -> np.ndarray:
    """Total least-squares fit of a homogeneous line through 2D points.

    `points`: (N, 2). Returns `(a, b, c)` with `a x + b y + c = 0`.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[-1]  # the "thinnest" direction = line normal
    a, b = direction
    c = -(a * centroid[0] + b * centroid[1])
    return np.array([a, b, c], dtype=np.float64)


def _envelope_intercept(line: np.ndarray, pts: np.ndarray, outer_sign: int,
                        *, margin: float = 2.0, pct: float = 98.0) -> np.ndarray:
    """Shift a fitted column edge OUTWARD to enclose its members.

    `_fit_line_lstsq` returns the cluster *centerline*, so roughly half of a
    justified column's endpoints fall outside it and get clipped by the warp
    ROI (athanase 014_A / 046_B: body text shaved on the right). Keep the TLS
    slope but move the intercept to the outer percentile of the member
    residuals (+ a small margin) so the edge envelopes the body text instead
    of bisecting it. A high percentile (not max) ignores a lone over-extended
    endpoint. `outer_sign=+1` for the right edge, `-1` for the left.
    """
    if abs(line[0]) < 1e-9 or len(pts) == 0:
        return line
    s = -line[1] / line[0]                 # dx/dy slope
    r = pts[:, 0] - s * pts[:, 1]          # per-point intercept
    q = pct if outer_sign > 0 else (100.0 - pct)
    m = float(np.percentile(r, q)) + outer_sign * margin
    return np.array([1.0, -s, -m], dtype=np.float64)


def select_full_width_lines(line_bboxes: list[tuple[int, int, int, int]],
                            *, tol_pct: float = 0.06) -> list[int]:
    """Return indices of bboxes whose left/right edges sit at the dominant
    column margins — i.e. justified body-text lines.

    Robust to noisy line detectors that emit many short fragments: we
    seed left/right margins from the widest bboxes (top 40 %), so partial
    fragments don't drag the medians off.
    """
    if len(line_bboxes) < 3:
        return list(range(len(line_bboxes)))
    lefts = np.array([b[0] for b in line_bboxes], dtype=np.float64)
    rights = np.array([b[0] + b[2] for b in line_bboxes], dtype=np.float64)
    widths = rights - lefts

    # Seed margins from widest bboxes only — these are the full-width
    # body-text lines we want to lock onto.
    order = np.argsort(-widths)
    n_top = max(3, int(0.4 * len(line_bboxes)))
    top_idx = order[:n_top]
    width_seed = float(np.median(widths[top_idx]))
    left_seed = float(np.median(lefts[top_idx]))
    right_seed = float(np.median(rights[top_idx]))
    tol = tol_pct * width_seed

    return [i for i in range(len(line_bboxes))
            if abs(lefts[i] - left_seed) <= tol and abs(rights[i] - right_seed) <= tol]


def select_per_side_anchors(
    baselines: list[tuple[np.ndarray, np.ndarray]],
    *, tol_pct: float = 0.06,
) -> tuple[list[int], list[int], list[int]]:
    """Decoupled left + right column-margin anchor selection.

    Seeds come from the **median of the top-40 %-widest baselines**
    — same robust statistic the old `select_full_width_baselines`
    used. Extremum (p5 / p95) seeds were tried (Track A in
    `docs/algorithms.md` §3.6.4) but regressed pages with
    page-edge ink outliers (scan-44 A: rights [..., 1660, 1663,
    1663] dragged p95 onto noise). Median-of-top-40 %-widest
    rejects those without losing the per-side decoupling, which is
    the structural improvement of this function over the old one.

    A baseline is a *left-anchor* if its left endpoint sits within
    `tol` of `left_seed`. Similarly for right-anchor. The two index
    sets are **independent**: a hyphenated line that reaches the
    left margin but stops 70 px short of the right margin
    contributes to `left_idxs` only, not `right_idxs` — its left
    endpoint is still valid evidence about the left column edge.

    Returns
    -------
    (left_idxs, right_idxs, full_idxs) where `full_idxs` is the
    intersection. Callers use the intersection only when they need
    lines that reach BOTH margins; top + bottom y-extreme anchors
    should use the **union** to avoid collapsing the quad on
    pages where few lines hit both margins simultaneously.
    """
    if len(baselines) < 3:
        idx = list(range(len(baselines)))
        return idx, idx, idx
    lefts = np.array([pL[0] for pL, _ in baselines], dtype=np.float64)
    rights = np.array([pR[0] for _, pR in baselines], dtype=np.float64)
    widths = rights - lefts

    order = np.argsort(-widths)
    n_top = max(3, int(0.4 * len(baselines)))
    top_idx = order[:n_top]
    width_seed = float(np.median(widths[top_idx]))
    left_seed = float(np.median(lefts[top_idx]))
    right_seed = float(np.median(rights[top_idx]))
    tol = tol_pct * width_seed

    left_idxs = [i for i in range(len(baselines))
                 if abs(lefts[i] - left_seed) <= tol]
    right_idxs = [i for i in range(len(baselines))
                  if abs(rights[i] - right_seed) <= tol]
    full_idxs = sorted(set(left_idxs) & set(right_idxs))
    return left_idxs, right_idxs, full_idxs


def select_full_width_baselines(
    baselines: list[tuple[np.ndarray, np.ndarray]],
    *, tol_pct: float = 0.06,
) -> list[int]:
    """Indices of baselines that reach BOTH dominant column margins.

    Thin wrapper over `select_per_side_anchors` for back-compat with
    code that only needs the intersection.
    """
    _, _, full = select_per_side_anchors(baselines, tol_pct=tol_pct)
    return full


def _supported_x_mask(xs: np.ndarray, *, gap_px: float,
                      min_support: int) -> np.ndarray:
    """1D gap-clustering. True where a value sits in a cluster (members
    within ``gap_px`` of each other) of at least ``min_support`` lines —
    i.e. its x is a *shared margin*, not a one-off."""
    n = len(xs)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    order = np.argsort(xs)
    sx = xs[order]
    splits = np.where(np.diff(sx) > gap_px)[0]
    for g in np.split(np.arange(n), splits + 1):
        if len(g) >= min_support:
            mask[order[g]] = True
    return mask


def block_justified_mask(all_left: np.ndarray, all_right: np.ndarray,
                         *, gap_px: float, min_support: int = 3) -> np.ndarray:
    """Block-aware "is this line justified to a column?" test.

    A line counts as justified iff BOTH its endpoints sit on a *supported*
    margin (≥ ``min_support`` lines share that left AND that right). This
    implicitly segments the page into blocks by margin: body paragraphs,
    indented blockquotes, and a centred Greek block each justify to their
    OWN left/right margins, while running heads, page numbers, and
    paragraph-final short lines — whose endpoints are one-offs — do not.

    Used to pick which baselines vote for the horizontal vanishing point:
    short/biased non-block lines were rotating the quad on blockquote-heavy
    pages (athanase 055 A)."""
    lx = np.asarray(all_left, dtype=np.float64)[:, 0]
    rx = np.asarray(all_right, dtype=np.float64)[:, 0]
    return (_supported_x_mask(lx, gap_px=gap_px, min_support=min_support)
            & _supported_x_mask(rx, gap_px=gap_px, min_support=min_support))


def segment_baselines_vblocks(
    baselines: list[tuple[np.ndarray, np.ndarray]],
    *, gap_mult: float = 2.5, min_pitch_px: float = 4.0,
) -> list[list[int]]:
    """Split baselines into vertically-separated text blocks.

    A page can stack several justified blocks — body text, then a
    critical apparatus, then footnotes/commentary — divided by wide
    vertical gaps. They have different baselines and line pitch, so
    feeding them all to one column-quad fit stretches the top/bottom
    anchors over the whole page and biases the VP (observed on
    augustin-confessions-vii_002: the apparatus dragged the quad bottom
    to the page foot).

    Walking the baselines top→bottom by midpoint-y, a gap larger than
    ``gap_mult ×`` the median line pitch starts a new block. Paragraph
    blank lines (~2× pitch) stay within a block; a real block boundary is
    several line-heights, so ``gap_mult`` ≈ 2.5 separates blocks without
    splitting paragraphs.

    Returns blocks as lists of indices into ``baselines``, ordered
    top→bottom. The common single-block page returns one list.
    """
    n = len(baselines)
    if n < 2:
        return [list(range(n))]
    ys = np.array([0.5 * (pL[1] + pR[1]) for pL, pR in baselines],
                  dtype=np.float64)
    order = np.argsort(ys, kind="stable")
    gaps = np.diff(ys[order])
    pos = gaps[gaps > 0]
    pitch = float(np.median(pos)) if pos.size else 0.0
    thr = gap_mult * max(pitch, min_pitch_px)
    blocks: list[list[int]] = []
    cur = [int(order[0])]
    for k in range(1, n):
        if gaps[k - 1] > thr:
            blocks.append(cur)
            cur = []
        cur.append(int(order[k]))
    blocks.append(cur)
    return blocks


def split_block_by_left_margin(
    baselines: list[tuple[np.ndarray, np.ndarray]],
    block: list[int],
    *, x_tol: float,
) -> list[list[int]]:
    """Split a vertical block into left-margin-coherent runs.

    `segment_baselines_vblocks` splits on vertical GAPS, so a block can mix
    columns that abut with no gap — e.g. an indented blockquote directly
    followed by full-width body (athanase 100: the quote's 8 lines + 5
    body-after lines land in one block, whose median left margin is
    quote-dominated, so the body-after never re-merges with the body above
    and the column quad is fit on the top half only → a rotated warp).

    Left-aligned text means the LEFT margin identifies the column. Mode-seek
    the dominant left margin, peel off the lines within `x_tol` of it, repeat
    on the remainder. Returns sub-blocks (index lists); a clean single-column
    block returns unchanged."""
    if len(block) <= 2:
        return [list(block)]
    remaining = list(block)
    out: list[list[int]] = []
    while remaining:
        lx = np.array([baselines[i][0][0] for i in remaining], dtype=np.float64)
        # Center covering the most lines within x_tol (1D mode), refined to
        # the median of its members.
        best_c, best_n = lx[0], -1
        for c in lx:
            n = int(np.sum(np.abs(lx - c) <= x_tol))
            if n > best_n:
                best_n, best_c = n, c
        center = float(np.median(lx[np.abs(lx - best_c) <= x_tol]))
        keep = [remaining[k] for k in range(len(remaining))
                if abs(lx[k] - center) <= x_tol]
        rest = [remaining[k] for k in range(len(remaining))
                if abs(lx[k] - center) > x_tol]
        out.append(keep)
        remaining = rest
    return out


def cluster_blocks_by_margins(
    baselines: list[tuple[np.ndarray, np.ndarray]],
    blocks: list[list[int]],
    *, x_tol: float, ang_tol_deg: float = 3.0,
) -> list[list[int]]:
    """Merge vertical blocks that belong to the **same text column**.

    Two blocks join when their LEFT margins agree and their RIGHT margins
    agree — in position (median endpoint x within ``x_tol`` px) and angle
    (median baseline tilt within ``ang_tol_deg``). So a body column that a
    section gap split into several blocks re-merges into one, while a
    marginal gloss (short → different right x) or a differently-set
    apparatus stays its own cluster. Blocks are NOT discarded — every
    block lands in exactly one cluster.

    Returns clusters as index lists into ``baselines``, **largest first**
    (most baselines). The page deskew upstream means margins are ~vertical,
    so median endpoint x is comparable across blocks at different y.
    """
    nb = len(blocks)
    if nb <= 1:
        return [list(b) for b in blocks]

    def _desc(b: list[int]) -> tuple[float, float, float]:
        lx = float(np.median([baselines[i][0][0] for i in b]))
        rx = float(np.median([baselines[i][1][0] for i in b]))
        ang = float(np.median([
            np.degrees(np.arctan2(baselines[i][1][1] - baselines[i][0][1],
                                  baselines[i][1][0] - baselines[i][0][0]))
            for i in b]))
        return lx, rx, ang

    desc = [_desc(b) for b in blocks]
    parent = list(range(nb))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(nb):
        li, ri, ai = desc[i]
        for j in range(i + 1, nb):
            lj, rj, aj = desc[j]
            if (abs(li - lj) <= x_tol and abs(ri - rj) <= x_tol
                    and abs(ai - aj) <= ang_tol_deg):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for k, b in enumerate(blocks):
        groups.setdefault(find(k), []).extend(b)
    return sorted(groups.values(), key=lambda g: -len(g))


def _dominant_mode(vals: np.ndarray, bw: float) -> tuple[int, float, np.ndarray]:
    """1-D dominant mode: the center maximizing the count within ±bw.
    Returns (count, refined_center, membership_mask)."""
    order = np.sort(vals)
    best = (0, float(order[0]), np.zeros(vals.shape, bool))
    for c in order:
        m = np.abs(vals - c) <= bw
        cnt = int(m.sum())
        if cnt > best[0]:
            best = (cnt, float(np.mean(vals[m])), m)
    return best


def estimate_column_edge_members(
    xs: np.ndarray, ys: np.ndarray, y_ref: float, *,
    bw: float, outer_sign: int, min_support: int,
    max_slant: float = 0.35, n_slants: int = 141,
) -> np.ndarray:
    """Select the body column's side-endpoint members, block-free.

    A side edge is a straight line in the page plane, so a slant ``b`` (dx/dy)
    exists for which ``x − b·(y − y_ref)`` collapses that edge's endpoints to
    one tight value. The BODY edge is the OUTERMOST well-supported margin —
    leftmost for the left side (``outer_sign=-1``), rightmost for the right
    (``outer_sign=+1``) — so an indented quote / footnote sits INSIDE it and
    never wins even when it outnumbers the body. For each candidate slant we
    take the outermost mode with ``>= min_support`` endpoints; the slant whose
    outermost mode is tightest (most members) wins. No vertical-block
    segmentation, no per-page gap threshold. Returns the membership mask."""
    best_count, best_mask = -1, None
    for b in np.linspace(-max_slant, max_slant, n_slants):
        corr = xs - b * (ys - y_ref)
        oc, om, ocount = None, None, 0
        for c in corr:
            m = np.abs(corr - c) <= bw
            cnt = int(m.sum())
            if cnt < min_support:
                continue
            ctr = float(np.mean(corr[m]))
            if oc is None or outer_sign * ctr > outer_sign * oc:
                oc, om, ocount = ctr, m, cnt
        if om is not None and ocount > best_count:
            best_count, best_mask = ocount, om
    if best_mask is None:  # nothing met support → densest mode at slant 0
        _, _, best_mask = _dominant_mode(xs, bw)
    return best_mask


def detect_column_quad_from_baselines(
    baselines: list[tuple[np.ndarray, np.ndarray]],
    *, tol_pct: float = 0.06,
    ransac_trials: int = 200,
    ransac_eps_deg: float = 0.3,
    edge_max_tilt_deg: float = 7.0,
    edge_cluster_gap_pct: float = 0.03,
    vblock_gap_mult: float = 2.5,
    vblock_edge_tol_pct: float = 0.04,
) -> Optional[tuple[np.ndarray, dict]]:
    """Column quadrilateral from precomputed line baselines.

    Decoupled estimator (see `docs/algorithms.md` §3.6.2):

    * **Horizontal VP** fitted on *every* baseline via RANSAC + TLS
      refit on the inlier set. Short lines (blockquotes, chapter
      headings, paragraph tails) contribute their angle without
      polluting the column-edge fit.
    * **Left / right column edges** via the slope-bounded clustered
      margin fit (`fit_margin_line_clustered`, §3.6.7) on ALL
      endpoints — blockquote indents form their own intercept cluster
      and the outermost supported cluster wins. Legacy anchor-box +
      RANSAC path is the fallback when clustering degenerates.
    * **Top / bottom column edges** anchored at the extreme y of the
      margin-cluster members and pinned to also pass through the VP.

    Returns
    -------
    (quad, info) where
        quad : (4, 2) ndarray, TL/TR/BR/BL.
        info : dict with `n_all`, `n_full_width`, `n_vp_inliers`,
               `vp_inlier_frac` for downstream telemetry.
        None if any decoupled step is irrecoverably degenerate.
    """
    n_all = len(baselines)
    if n_all < 3:
        return None

    # 0. BLOCK-JUSTIFIED filtering (the page is segmented into blocks by
    # margin; a line counts only if BOTH endpoints sit on a shared margin
    # — its own block's). Running heads, page numbers and paragraph-final
    # short lines have one-off endpoints and are dropped. This is the key
    # robustness step for blockquote-heavy pages: on athanase 055 A the
    # page number "118" + italic running head tilted the L/R margin fit
    # into a phantom ~6° rotation and biased the VP. Everything below —
    # VP, margins, top/bottom anchors — then runs on real block lines
    # only. Guarded: keep all lines when too few justify (sparse pages).
    _ol = np.array([pL for pL, _ in baselines], dtype=np.float64)
    _or = np.array([pR for _, pR in baselines], dtype=np.float64)
    _w = _or[:, 0] - _ol[:, 0]
    _seed = float(np.median(_w[np.argsort(-_w)[:max(3, int(0.4 * n_all))]]))
    _gap = max(edge_cluster_gap_pct * _seed, 6.0)
    justified = block_justified_mask(_ol, _or, gap_px=_gap, min_support=3)
    n_justified = int(justified.sum())
    n_original = n_all
    import os as _os
    if (_os.environ.get("AGLAIA_TRAP_BLOCK", "1") != "0"
            and n_justified >= max(6, int(0.5 * n_all))):
        baselines = [baselines[i] for i in np.where(justified)[0]]
        n_all = len(baselines)

    # Vertical-block segmentation is retired: "what gap starts a new block"
    # has no book-independent value (leading, footnote spacing, section
    # breaks all differ), so it was brittle. Column identity now comes from
    # endpoint clustering below, which needs no blocks and no gap threshold.
    n_vblocks = 1

    # Geometry stats on the (block-filtered) set.
    all_left = np.array([pL for pL, _ in baselines], dtype=np.float64)
    all_right = np.array([pR for _, pR in baselines], dtype=np.float64)
    # 1+2. Column side edges by ENDPOINT CLUSTERING (block-free). The body's
    # many endpoints form the dominant mode of x − slant·(y − y_ref); indented
    # quotes, footnotes and running heads only ever form SMALLER modes and
    # never win, so no vertical-block segmentation / cluster-picking is needed.
    W_est = float(max(all_right[:, 0].max(), 1.0))
    # Sparsity-promoting band: the slant search rewards the slant that
    # CONCENTRATES the outermost endpoints (the justification margin is a
    # tight cluster), not the one that rakes in the most. A wide band let a
    # wrong slant gather short / indented stragglers and win on raw count,
    # tilting the edge (athanase 062 B: rightmost ends are clustered to ~2 px
    # std yet the fit tilted +2.5°). A tight band is the L1/L0 criterion —
    # the densest concentration wins, the stragglers fall outside.
    bw = max(6.0, 0.008 * W_est)
    y_ref = float(np.median(np.concatenate([all_left[:, 1], all_right[:, 1]])))
    min_support = max(4, int(0.2 * len(all_left)))
    lmask = estimate_column_edge_members(
        all_left[:, 0], all_left[:, 1], y_ref, bw=bw,
        outer_sign=-1, min_support=min_support)
    rmask = estimate_column_edge_members(
        all_right[:, 0], all_right[:, 1], y_ref, bw=bw,
        outer_sign=+1, min_support=min_support)
    left_idxs = list(np.where(lmask)[0])
    right_idxs = list(np.where(rmask)[0])
    fw_idxs = sorted(set(left_idxs) & set(right_idxs))
    column_edge_source = "endpoint_clustering"
    if len(left_idxs) < 3 or len(right_idxs) < 3:
        return None
    # The slant search robustly SELECTS the body endpoints; the precise edge
    # line is a TLS fit on those members (exact slope + intercept, no grid
    # quantisation).
    left_line = _fit_line_lstsq(all_left[lmask])
    right_line = _fit_line_lstsq(all_right[rmask])

    # Cross-side tilt consistency: both side edges are vertical in page space
    # and converge only by the (small) keystone, so a tilt disagreement > 2.5°
    # means one side's mode locked onto a slanted minority. The weaker side
    # (fewer inliers) adopts the stronger side's slope, keeping its own
    # intercept (median residual of its members).
    tilt_l = margin_line_tilt_deg(left_line)
    tilt_r = margin_line_tilt_deg(right_line)
    if abs(tilt_l - tilt_r) > 2.5:
        if len(left_idxs) >= len(right_idxs):
            strong_line, weak_pts, weak_members, weak_is_right = (
                left_line, all_right, right_idxs, True)
        else:
            strong_line, weak_pts, weak_members, weak_is_right = (
                right_line, all_left, left_idxs, False)
        if abs(strong_line[0]) > 1e-9 and weak_members:
            b_s = -strong_line[1] / strong_line[0]
            r_weak = (weak_pts[weak_members, 0]
                      - b_s * weak_pts[weak_members, 1])
            m = float(np.median(r_weak))
            fixed = np.array([1.0, -b_s, -m], dtype=np.float64)
            if weak_is_right:
                right_line = fixed
            else:
                left_line = fixed

    # Envelope, don't bisect: push each edge out to the outer extent of its
    # justified endpoints so the warp ROI encloses the body text. (Slopes from
    # the tilt-consistent fit above are preserved; only the intercept moves.)
    left_line = _envelope_intercept(left_line, all_left[lmask], -1)
    right_line = _envelope_intercept(right_line, all_right[rmask], +1)

    # Horizontal VP from the COLUMN-inlier lines (union of side members), so a
    # running head / footnote outside the column can't tug the convergence.
    inl = sorted(set(left_idxs) | set(right_idxs))
    inl_lines = np.array(
        [line_through(baselines[i][0], baselines[i][1]) for i in inl],
        dtype=np.float64)
    vp, vp_inliers = ransac_vp(
        inl_lines, trials=ransac_trials, eps_deg=ransac_eps_deg)
    n_inliers = int(vp_inliers.size)
    if vp is None:
        try:
            vp = vp_from_lines_svd(inl_lines)
            n_inliers = len(inl)
        except Exception:
            return None

    # Rotation-consistency. SkewFinder runs immediately before the trap and
    # already removed page rotation, so the trap must correct only KEYSTONE
    # (margin convergence) — never re-introduce rotation. On multi-width
    # pages the clustered fit can lock both margins onto a shared NON-zero
    # tilt (parallel = pure rotation) that disagrees with the near-
    # horizontal baselines (athanase 055 A: baselines ~0°, margins +5.9° →
    # the page got rotated 6°). Re-centre the margins' MEAN tilt onto the
    # baseline tilt while preserving their RELATIVE convergence (the true
    # keystone signal). base_ang (baseline tilt from horizontal) equals the
    # desired margin tilt from vertical under a rigid page rotation.
    base_ang = float(np.median([
        np.arctan2(pR[1] - pL[1], pR[0] - pL[0]) for pL, pR in baselines]))
    s_l = (-left_line[1] / left_line[0]) if abs(left_line[0]) > 1e-9 else 0.0
    s_r = (-right_line[1] / right_line[0]) if abs(right_line[0]) > 1e-9 else 0.0
    th_l, th_r = float(np.arctan(s_l)), float(np.arctan(s_r))
    delta = base_ang - 0.5 * (th_l + th_r)
    # Only de-rotate when the margins are near-PARALLEL (convergence ≈ 0,
    # i.e. a pure rotation/shear with no perspective width change) AND the
    # mismatch with the baselines is GROSS (> 3°). Genuine keystone has the
    # two margins CONVERGING (different slopes) — left untouched, so we
    # never flatten a real perspective. A > 3° parallel margin tilt on a
    # page whose baselines say ~0° rotation is the invented-rotation bug.
    convergence = abs(th_l - th_r)
    if convergence < np.radians(1.0) and abs(delta) > np.radians(3.0):
        def _retilt(members, fallback_pts, new_th):
            pts = (all_left if fallback_pts is all_left else all_right)
            pts = pts[list(members)] if len(members) else pts
            c = pts.mean(axis=0)
            ns = float(np.tan(new_th))
            return np.array([1.0, -ns, -(c[0] - ns * c[1])], dtype=np.float64)
        left_line = _retilt(left_idxs, all_left, th_l + delta)
        right_line = _retilt(right_idxs, all_right, th_r + delta)

    # Top/bot anchor at UNION(left, right) extremes — intersection
    # excludes lines reaching only one margin, costing vertical reach on
    # hyphenation-heavy or blockquote pages.
    yspan_idxs = sorted(set(left_idxs) | set(right_idxs))
    sel = [baselines[i] for i in yspan_idxs]

    # 3. Top / bottom: anchor at extreme y of the full-width set and
    # pin through the VP. Using full-width avoids being dragged off by
    # a top-of-page header or footnote that is not part of the body
    # column.
    mid_y = [0.5 * (pL[1] + pR[1]) for pL, pR in sel]
    top_i = int(np.argmin(mid_y))
    bot_i = int(np.argmax(mid_y))
    x_center = float(np.median([0.5 * (pL[0] + pR[0]) for pL, pR in sel]))

    def _line_through_vp_at(y_anchor: float) -> np.ndarray:
        if abs(vp[2]) < 1e-9:
            return np.array([0.0, -1.0, y_anchor])
        return _line_through_vp_and_point(vp, np.array([x_center, y_anchor]))

    top_line = _line_through_vp_at(mid_y[top_i])
    bottom_line = _line_through_vp_at(mid_y[bot_i])

    def _intersect(l1, l2) -> Optional[np.ndarray]:
        p = np.cross(l1, l2)
        if abs(p[2]) < 1e-9:
            return None
        return np.array([p[0] / p[2], p[1] / p[2]])

    tl = _intersect(top_line, left_line)
    tr = _intersect(top_line, right_line)
    br = _intersect(bottom_line, right_line)
    bl = _intersect(bottom_line, left_line)
    if any(c is None for c in (tl, tr, br, bl)):
        return None
    quad = np.array([tl, tr, br, bl], dtype=np.float64)
    info = {
        "n_all": n_all,
        "n_original": n_original,
        "n_full_width": len(fw_idxs),
        "n_justified": n_justified,
        "n_vblocks": n_vblocks,
        "n_vp_inliers": n_inliers,
        "vp_inlier_frac": float(n_inliers) / max(n_all, 1),
        "full_width_idxs": fw_idxs,
        "column_edge_source": column_edge_source,
    }
    return quad, info


def detect_column_quad(line_bboxes: list[tuple[int, int, int, int]],
                       ink: np.ndarray,
                       *, tol_pct: float = 0.06) -> Optional[np.ndarray]:
    """Estimate the 4-corner column quadrilateral from justified text bboxes.

    Returns a `(4, 2)` array of corners in image pixels ordered
    [top-left, top-right, bottom-right, bottom-left] — the same convention
    `cv2.getPerspectiveTransform` expects.

    Approach: filter to full-width lines (so each carries the true column
    left/right margin), fit straight lines through

        - left-margin samples: bottom-left corner of each full line bbox
        - right-margin samples: bottom-right corner of each full line bbox
        - top edge: the ink-fit baseline of the first full line, then shifted
          up by half its bbox height (so we capture the typographic top,
          not the baseline)
        - bottom edge: ink-fit baseline of the last full line

    Returns None when we can't isolate enough full-width lines or the
    fitted lines are degenerate (parallel = page is not perspective-distorted,
    no need to warp).
    """
    idxs = select_full_width_lines(line_bboxes, tol_pct=tol_pct)
    if len(idxs) < 3:
        return None

    # Per-line baselines via the real ink contour. We DO NOT use bbox
    # top/height: bbox height is sensitive to warp (cap height + descender
    # depth both wander with perspective), so anchoring the top edge on it
    # makes the homography poorly conditioned. Baselines are the stable
    # feature — bottom-contour of ink, fit via RANSAC — so we use them on
    # BOTH the top and bottom edges of the column quad.
    baselines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in idxs:
        bb = line_bboxes[i]
        ep = baseline_from_ink(ink, bb)
        if ep is None:
            continue
        baselines.append(ep)
    if len(baselines) < 3:
        return None

    # Left and right column edges from the endpoints of all full-line baselines.
    left_pts = np.array([pL for pL, _ in baselines])
    right_pts = np.array([pR for _, pR in baselines])
    left_line = _fit_line_lstsq(left_pts)
    right_line = _fit_line_lstsq(right_pts)

    # Median baseline slope (low-variance, 50 % breakdown).
    slopes = []
    for pL, pR in baselines:
        dx = pR[0] - pL[0]
        if abs(dx) > 1e-3:
            slopes.append((pR[1] - pL[1]) / dx)
    slope = float(np.median(slopes)) if slopes else 0.0

    # Top + bottom edges: BASELINE of the topmost/bottommost line (averaged
    # over K neighbours for robustness). Pick the extremum by baseline
    # midpoint-y so descenders / ascenders don't decide.
    n = len(baselines)
    k = max(3, n // 6)
    mid_y = [0.5 * (pL[1] + pR[1]) for pL, pR in baselines]
    order = sorted(range(n), key=lambda i: mid_y[i])
    top_idxs = order[:k]
    bot_idxs = order[-k:]

    top_ys = [mid_y[i] for i in top_idxs]
    top_y = float(np.median(top_ys))
    x_center_top = float(np.median([0.5 * (baselines[i][0][0] + baselines[i][1][0])
                                    for i in top_idxs]))
    top_line = np.array([slope, -1.0, top_y - slope * x_center_top])

    bot_ys = [mid_y[i] for i in bot_idxs]
    bot_y = float(np.median(bot_ys))
    x_center_bot = float(np.median([0.5 * (baselines[i][0][0] + baselines[i][1][0])
                                    for i in bot_idxs]))
    bottom_line = np.array([slope, -1.0, bot_y - slope * x_center_bot])

    # 4 corners as line intersections.
    def _intersect(l1, l2) -> Optional[np.ndarray]:
        p = np.cross(l1, l2)
        if abs(p[2]) < 1e-9:
            return None
        return np.array([p[0] / p[2], p[1] / p[2]])

    tl = _intersect(top_line, left_line)
    tr = _intersect(top_line, right_line)
    br = _intersect(bottom_line, right_line)
    bl = _intersect(bottom_line, left_line)
    if any(c is None for c in (tl, tr, br, bl)):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float64)
    return quad


# ───────────────────────────────────────────────────────────────────────────
# Zhang-He aspect ratio recovery from a quadrilateral
# ───────────────────────────────────────────────────────────────────────────

def zhang_he_aspect_and_focal(quad: np.ndarray,
                              image_size: tuple[int, int],
                              focal_px: Optional[float] = None,
                              ) -> tuple[Optional[float], Optional[float]]:
    """Estimate the true rectangle aspect (w/h) and (optionally) focal length
    from a perspective-distorted quadrilateral.

    Implements Zhang & He, *Whiteboard It!* (2002), §4 — equations (11)–(21).

    Parameters
    ----------
    quad : (4, 2) ndarray
        The four corners of the imaged rectangle, ordered TL, TR, BR, BL —
        i.e. `M1=TL=(0,0)`, `M2=TR=(w,0)`, `M3=BL=(0,h)`, `M4=BR=(w,h)`
        in the page-plane coordinate frame.
    image_size : (W, H)
        Image dimensions in pixels (used for the principal-point assumption).
    focal_px : float or None
        Known focal length in pixels. If None, the equation (21) closed form
        is used to derive it.

    Returns
    -------
    (w_over_h, focal_px) — either may be None on degeneracy.
    """
    if quad.shape != (4, 2):
        raise ValueError(f"quad must be (4, 2), got {quad.shape}")
    W, H = image_size
    # Map Zhang-He labels: M1=TL, M2=TR, M3=BL, M4=BR.
    m1 = np.array([quad[0, 0], quad[0, 1], 1.0])
    m2 = np.array([quad[1, 0], quad[1, 1], 1.0])
    m4 = np.array([quad[2, 0], quad[2, 1], 1.0])  # BR
    m3 = np.array([quad[3, 0], quad[3, 1], 1.0])  # BL

    def safe_div(a: float, b: float) -> Optional[float]:
        if abs(b) < 1e-9:
            return None
        return a / b

    k2_num = float(np.dot(np.cross(m1, m4), m3))
    k2_den = float(np.dot(np.cross(m2, m4), m3))
    k3_num = float(np.dot(np.cross(m1, m4), m2))
    k3_den = float(np.dot(np.cross(m3, m4), m2))
    k2 = safe_div(k2_num, k2_den)
    k3 = safe_div(k3_num, k3_den)
    if k2 is None or k3 is None:
        return None, focal_px

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    # Focal length from (21) if not provided.
    if focal_px is None:
        u0 = W / 2.0
        v0 = H / 2.0
        n21, n22, n23 = n2
        n31, n32, n33 = n3
        denom = n23 * n33
        if abs(denom) < 1e-9:
            # k2 == 1 or k3 == 1 → degenerate, skip.
            return None, None
        sq = -(1.0 / denom) * (
            (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0)
            + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0)
        )
        if not (sq > 0):
            return None, None
        focal_px = float(math.sqrt(sq))

    # Build A^{-T} A^{-1} (cheap because A is diagonal-ish with u0, v0 offsets).
    u0 = W / 2.0
    v0 = H / 2.0
    A = np.array([
        [focal_px, 0, u0],
        [0, focal_px, v0],
        [0, 0, 1.0],
    ])
    A_inv = np.linalg.inv(A)
    M = A_inv.T @ A_inv

    num = float(n2 @ M @ n2)
    den = float(n3 @ M @ n3)
    if den <= 0 or num <= 0:
        return None, focal_px
    w_over_h = math.sqrt(num / den)
    return w_over_h, focal_px


# ───────────────────────────────────────────────────────────────────────────
# Reused inside oob_fraction
# ───────────────────────────────────────────────────────────────────────────

def _points_in_quad(xs: np.ndarray, ys: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Vectorised point-in-convex-polygon test using cross products."""
    n = quad.shape[0]
    inside = np.ones(xs.shape, dtype=bool)
    sign_ref = None
    for i in range(n):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % n]
        cross = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        if sign_ref is None:
            sign_ref = np.sign(np.mean(cross))
            if sign_ref == 0:
                sign_ref = 1
        inside &= (np.sign(cross) == sign_ref) | (cross == 0)
    return inside
