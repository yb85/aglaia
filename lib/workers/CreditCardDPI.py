# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""DPI estimation from a credit-card ISO/IEC 7810 ID-1 sized rectangle.

ID-1 dimensions: 85.60 mm × 53.98 mm (3.370 in × 2.125 in). Aspect ratio
≈ 1.586:1. Same physical size for every standard credit / debit / ID
card, so users almost always have one available.

Detection runs several segmentation strategies in parallel (Otsu light,
Otsu dark, Canny + close, adaptive threshold) and merges the candidates.
This handles low-contrast scenes (white card on grey table) and busy
textures equally — single-pipeline detectors fail when the texture
overwhelms Canny.

Per candidate contour, the polygon-approx epsilon is swept (1–6 % of
perimeter) so a slightly bent perspective quad still resolves to four
vertices.
"""
from __future__ import annotations

import cv2
import numpy as np

ID1_LONG_MM = 85.60
ID1_SHORT_MM = 53.98
ID1_ASPECT = ID1_LONG_MM / ID1_SHORT_MM  # ≈ 1.586

# Detection tolerances. Loose on aspect: a card photographed off-axis
# distorts to anywhere in [1.2, 2.0] before perspective recovery.
ASPECT_TOLERANCE = 0.25
# Plausible card area range vs frame area.
MIN_AREA_FRAC = 0.003
MAX_AREA_FRAC = 0.6
# Polygon-approx epsilon sweep (as a fraction of perimeter).
EPS_SWEEP = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return quad corners ordered tl, tr, br, bl."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def _candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    """Build several binary masks; each is a plausible foreground guess."""
    masks: list[np.ndarray] = []
    # Smooth without losing card edges.
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)

    # 1. Otsu, card brighter than bg.
    _, otsu = cv2.threshold(smooth, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(otsu)
    # 2. Otsu inverted, card darker than bg.
    masks.append(cv2.bitwise_not(otsu))

    # 3. Canny with median-based auto thresholds, dilated to close gaps.
    med = float(np.median(smooth))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(smooth, lo, hi)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )
    masks.append(edges)

    # 4. Adaptive threshold (block size scaled to frame), then close to
    # merge texture speckle into card blob.
    h, w = gray.shape[:2]
    block = max(31, (min(h, w) // 16) | 1)  # odd
    adp = cv2.adaptiveThreshold(
        smooth, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        block, -10,
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    adp_closed = cv2.morphologyEx(adp, cv2.MORPH_CLOSE, k, iterations=2)
    masks.append(adp_closed)
    return masks


def _quad_from_contour(c: np.ndarray) -> np.ndarray | None:
    """Sweep approxPolyDP epsilon; return first convex 4-vertex polygon."""
    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return None
    for eps in EPS_SWEEP:
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx
    return None


def _evaluate_quad(quad: np.ndarray) -> tuple[float, float, float] | None:
    """Return (long_px, short_px, aspect) if the quad's aspect is within
    tolerance of the ID-1 ratio; otherwise None."""
    ordered = _order_quad(quad)
    tl, tr, br, bl = ordered
    top = np.linalg.norm(tr - tl)
    bot = np.linalg.norm(br - bl)
    lft = np.linalg.norm(bl - tl)
    rgt = np.linalg.norm(br - tr)
    long_px = (top + bot) / 2.0
    short_px = (lft + rgt) / 2.0
    if long_px < short_px:
        long_px, short_px = short_px, long_px
    if short_px <= 0:
        return None
    aspect = long_px / short_px
    if abs(aspect - ID1_ASPECT) / ID1_ASPECT > ASPECT_TOLERANCE:
        return None
    return long_px, short_px, aspect


def _refine_edge(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                 *, samples: int = 24, search_px: int = 18,
                 inset_frac: float = 0.18) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample along edge p0→p1, find max-gradient point on the perpendicular
    for each sample, return (line_point, line_dir) from a robust LS fit.

    Skips the first/last `inset_frac` of the edge to avoid the rounded
    corners of an ID-1 card (≈3.2 mm radius). Returns None if too few
    inliers (degenerate / no edge detected).
    """
    p0 = p0.astype(np.float32)
    p1 = p1.astype(np.float32)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 4.0:
        return None
    direction = direction / length
    # Perpendicular (rotate 90°). Sign doesn't matter — we scan ±.
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    h, w = gray.shape[:2]
    # Cross-edge gradient via Sobel along the perpendicular. We use Scharr
    # (more accurate at small kernel) on the gray buffer once, then sample.
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    refined_pts: list[np.ndarray] = []
    t_start = inset_frac
    t_end = 1.0 - inset_frac
    if t_end <= t_start:
        t_start, t_end = 0.1, 0.9
    for i in range(samples):
        t = t_start + (t_end - t_start) * (i / (samples - 1))
        centre = p0 + direction * (t * length)
        best_score = -1.0
        best_d = 0.0
        for d in range(-search_px, search_px + 1):
            x = centre[0] + perp[0] * d
            y = centre[1] + perp[1] * d
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            # |gradient · perp| = strength of the edge perpendicular to
            # the line direction at this point.
            mag = abs(gx[iy, ix] * perp[0] + gy[iy, ix] * perp[1])
            if mag > best_score:
                best_score = float(mag)
                best_d = float(d)
        if best_score < 8.0:  # noise floor
            continue
        # Subpixel parabola peak on the perpendicular score.
        # (Refine only if neighbours are in-bounds and finite.)
        x0 = centre[0] + perp[0] * (best_d - 1)
        y0 = centre[1] + perp[1] * (best_d - 1)
        x2 = centre[0] + perp[0] * (best_d + 1)
        y2 = centre[1] + perp[1] * (best_d + 1)
        if (0 <= int(x0) < w and 0 <= int(y0) < h
                and 0 <= int(x2) < w and 0 <= int(y2) < h):
            s_m = abs(gx[int(round(y0)), int(round(x0))] * perp[0]
                      + gy[int(round(y0)), int(round(x0))] * perp[1])
            s_p = abs(gx[int(round(y2)), int(round(x2))] * perp[0]
                      + gy[int(round(y2)), int(round(x2))] * perp[1])
            denom = (s_m - 2 * best_score + s_p)
            if abs(denom) > 1e-6:
                delta = 0.5 * (s_m - s_p) / denom
                if -1.0 < delta < 1.0:
                    best_d += delta
        refined = centre + perp * best_d
        refined_pts.append(refined)

    if len(refined_pts) < 6:
        return None
    pts = np.array(refined_pts, dtype=np.float32)
    # cv2.fitLine with HUBER → robust to a few outlier samples (corner
    # rounding, glare, fingers obscuring part of the edge).
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    line_point = np.array([x0, y0], dtype=np.float32)
    line_dir = np.array([vx, vy], dtype=np.float32)
    return line_point, line_dir


def _line_intersection(p1: np.ndarray, d1: np.ndarray,
                       p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    """Intersect two parametric lines (point, direction). None if parallel."""
    # p1 + t*d1 = p2 + s*d2  →  solve 2x2 linear system in (t, s).
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float32)
    b = (p2 - p1).astype(np.float32)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-6:
        return None
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    return p1 + d1 * t


def refine_and_measure(bgr: np.ndarray,
                       corners: np.ndarray) -> tuple[float, np.ndarray]:
    """Refine each of the 4 edges via cross-edge gradient sampling, then
    intersect adjacent lines to recover sharp corners (the card's actual
    rounded ID-1 corners would mislead `cornerSubPix`).

    Returns (dpi, ordered_quad). `corners` is (4,2) in image pixel coords.
    Order doesn't matter — `_order_quad` reorders to (tl, tr, br, bl)
    before edge refinement runs.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    # Light blur to stabilise the gradient against sensor noise.
    gray_s = cv2.GaussianBlur(gray, (5, 5), 1.0)
    rough = _order_quad(corners)
    tl, tr, br, bl = rough

    # Edge order matches sides between consecutive ordered corners:
    #   top:    tl → tr
    #   right:  tr → br
    #   bottom: br → bl
    #   left:   bl → tl
    edges_in = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
    refined_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for p0, p1 in edges_in:
        result = _refine_edge(gray_s, p0, p1)
        if result is None:
            # Fallback to the user-clicked edge itself.
            d = (p1 - p0).astype(np.float32)
            d /= max(np.linalg.norm(d), 1e-6)
            refined_lines.append((p0.astype(np.float32), d))
        else:
            refined_lines.append(result)

    # Re-derive corners as line intersections (top ∩ left = tl, etc.).
    top_pt, top_dir = refined_lines[0]
    right_pt, right_dir = refined_lines[1]
    bot_pt, bot_dir = refined_lines[2]
    left_pt, left_dir = refined_lines[3]
    refined_corners = []
    for a, b in (((top_pt, top_dir), (left_pt, left_dir)),
                 ((top_pt, top_dir), (right_pt, right_dir)),
                 ((bot_pt, bot_dir), (right_pt, right_dir)),
                 ((bot_pt, bot_dir), (left_pt, left_dir))):
        pt = _line_intersection(a[0], a[1], b[0], b[1])
        refined_corners.append(pt if pt is not None else rough[len(refined_corners)])
    quad = _order_quad(np.array(refined_corners, dtype=np.float32))

    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    bot = np.linalg.norm(br - bl)
    lft = np.linalg.norm(bl - tl)
    rgt = np.linalg.norm(br - tr)
    long_px = (top + bot) / 2.0
    short_px = (lft + rgt) / 2.0
    if long_px < short_px:
        long_px, short_px = short_px, long_px
    dpi = float(long_px) * 25.4 / ID1_LONG_MM
    return dpi, quad


def _detect_card_apple_vision(bgr: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Tier-1 detector via Apple Vision's `VNDetectRectanglesRequest`.

    Returns `(dpi, ordered_quad)` or `(None, None)` if Vision is not
    available (non-macOS, missing pyobjc) or no card-shaped rectangle
    survives the aspect-ratio filter. The fallback to the classical
    OpenCV pipeline is handled by the caller — this function never
    raises.

    Aspect-ratio convention: Vision uses `min_side / max_side` in
    [0, 1]. ID-1 ≈ 53.98 / 85.60 ≈ 0.631. Bounds ±0.10 absorb
    perspective tilt; off-axis shots warp the visible aspect a fair
    bit before the corners stop forming a quad.
    """
    try:
        import io
        import objc
        import Vision
        from Foundation import NSData
        from PIL import Image
    except Exception:
        return None, None
    if bgr is None or bgr.size == 0:
        return None, None
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr
    pil_img = Image.fromarray(rgb)

    # Apple Vision expects `min_side / max_side`; ID-1 → 0.6306. Loose
    # range to swallow perspective foreshortening from a typical webcam
    # mounted ~30 cm above a desk surface.
    target_aspect = ID1_SHORT_MM / ID1_LONG_MM
    aspect_min = max(0.30, target_aspect - 0.18)
    aspect_max = min(0.95, target_aspect + 0.18)

    candidates: list[tuple[np.ndarray, float]] = []
    try:
        with objc.autorelease_pool():
            req = Vision.VNDetectRectanglesRequest.alloc().init()
            req.setMinimumAspectRatio_(float(aspect_min))
            req.setMaximumAspectRatio_(float(aspect_max))
            req.setMinimumSize_(0.05)            # 5% of the shortest image side
            req.setMaximumObservations_(8)
            req.setQuadratureTolerance_(20.0)    # degrees off square corners
            req.setMinimumConfidence_(0.55)

            b = io.BytesIO()
            pil_img.save(b, format="JPEG", quality=95)
            img_bytes = b.getvalue()
            ns_data = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                ns_data, None,
            )
            success, _err = handler.performRequests_error_([req], None)
            if not success:
                return None, None
            results = req.results() or []
            for obs in results:
                conf = float(obs.confidence())
                # Vision corners are CGPoints with y-origin at bottom-left,
                # normalised to [0, 1]. Convert to pixel coords with the
                # OpenCV convention (origin top-left).
                def _pt(p):
                    return np.array([p.x * W, (1.0 - p.y) * H], dtype=np.float32)
                tl = _pt(obs.topLeft())
                tr = _pt(obs.topRight())
                br = _pt(obs.bottomRight())
                bl = _pt(obs.bottomLeft())
                quad = np.stack([tl, tr, br, bl], axis=0)
                candidates.append((quad, conf))
            del handler
            del req
            del ns_data
    except Exception:
        return None, None

    if not candidates:
        return None, None

    img_area = float(H * W)
    min_area = MIN_AREA_FRAC * img_area
    max_area = MAX_AREA_FRAC * img_area

    best = None  # (score, ordered_quad, long_px)
    for quad, conf in candidates:
        # Vision quads are already ordered, but _order_quad re-canonicalises
        # robustly against the handful of edge cases where the labels and
        # geometry disagree (e.g. mirror-flipped frames).
        ordered = _order_quad(quad)
        ev = _evaluate_quad(ordered)
        if ev is None:
            continue
        long_px, short_px, aspect = ev
        area = long_px * short_px
        if area < min_area or area > max_area:
            continue
        aspect_err = abs(aspect - ID1_ASPECT) / ID1_ASPECT
        # Score weights: Vision confidence × aspect match × sqrt(area).
        # Confidence dominates because Vision already rejects implausible
        # shapes upstream; aspect is a secondary tiebreaker.
        score = conf * float(np.sqrt(area)) / (1.0 + aspect_err * 4.0)
        if best is None or score > best[0]:
            best = (score, ordered, long_px)

    if best is None:
        return None, None
    _, quad, long_px = best
    dpi = float(long_px) * 25.4 / ID1_LONG_MM
    return dpi, quad


def detect_card_dpi(bgr: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Detect a credit-card-shaped rectangle and return (dpi, ordered_quad).

    Returns (None, None) if no convincing card-shaped quadrilateral found.
    The quad is in (tl, tr, br, bl) order for downstream overlay/debug.

    Two-tier strategy: try Apple Vision's `VNDetectRectanglesRequest`
    first (shape-only, ~20 ms, no model weights, no training data),
    fall back to the classical OpenCV mask-sweep pipeline if Vision is
    unavailable or returns nothing usable. The fallback is what runs
    on Linux and on macOS scenes where Vision rejects the card (rare —
    e.g. extreme tilt, severe glare).
    """
    if bgr is None or bgr.size == 0:
        return None, None

    av_dpi, av_quad = _detect_card_apple_vision(bgr)
    if av_dpi is not None and av_quad is not None:
        return av_dpi, av_quad

    H, W = bgr.shape[:2]
    img_area = float(H * W)
    min_area = MIN_AREA_FRAC * img_area
    max_area = MAX_AREA_FRAC * img_area

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    best = None  # (score, ordered_quad, long_px)
    for mask in _candidate_masks(gray):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        # Sort big-first so we hit the most plausible blobs early.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            quad = _quad_from_contour(c)
            if quad is None:
                continue
            ev = _evaluate_quad(quad)
            if ev is None:
                continue
            long_px, _short_px, aspect = ev
            # Score: aspect match (closer to 1.586 → larger) × sqrt(area).
            # sqrt keeps a huge texture blob from dominating a clean card.
            aspect_err = abs(aspect - ID1_ASPECT) / ID1_ASPECT
            score = float(np.sqrt(area)) / (1.0 + aspect_err * 5.0)
            if best is None or score > best[0]:
                best = (score, _order_quad(quad), long_px)

    if best is None:
        return None, None
    _, quad, long_px = best
    dpi = float(long_px) * 25.4 / ID1_LONG_MM
    return dpi, quad
