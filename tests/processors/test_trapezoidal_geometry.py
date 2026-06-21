# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Unit tests for `lib.processors.geometry` — the math primitives behind TrapezoidalCorrection."""
import math

import numpy as np
import pytest

from lib.processors.geometry import (
    affine_rect_from_line, bbox_baseline, horizontal_vanishing_line,
    line_through, oob_fraction, ransac_vp, vertical_vp_from_line_pitch,
    vp_from_lines_svd, warp_extent,
)


# ───────────────────────── helpers ────────────────────────────────────

def _line_through_vp_and_point(vp, point):
    """Build a homogeneous line passing through a finite VP and a 2D point."""
    p = np.array([point[0], point[1], 1.0])
    return np.cross(p, vp)


# ───────────────────────── vp_from_lines_svd ──────────────────────────

def test_vp_exact_three_lines():
    vp_true = np.array([1000.0, -500.0, 1.0])
    lines = [_line_through_vp_and_point(vp_true, p) for p in [(0, 0), (100, 200), (500, 80)]]
    vp = vp_from_lines_svd(np.array(lines))
    vp = vp / vp[2]
    np.testing.assert_allclose(vp[:2], vp_true[:2], atol=1e-6)


def test_vp_robust_to_small_noise():
    rng = np.random.default_rng(7)
    vp_true = np.array([800.0, 300.0, 1.0])
    lines = []
    for _ in range(40):
        p = rng.uniform(0, 1000, size=2)
        lines.append(_line_through_vp_and_point(vp_true, p))
    # Add ~1 px noise to the lines.
    lines = np.array(lines) + rng.normal(scale=0.05, size=(40, 3))
    vp = vp_from_lines_svd(lines)
    vp = vp / vp[2]
    assert math.hypot(vp[0] - vp_true[0], vp[1] - vp_true[1]) < 10.0


# ───────────────────────── ransac_vp ───────────────────────────────────

def test_ransac_rejects_outlier_lines():
    vp_true = np.array([1500.0, -200.0, 1.0])
    lines = [_line_through_vp_and_point(vp_true, p) for p in [(0, 0), (100, 200), (500, 80), (700, 400)]]
    # Add 2 outliers — purely vertical lines through unrelated points.
    lines.append(np.array([1.0, 0.0, -50.0]))
    lines.append(np.array([1.0, 0.0, -900.0]))
    vp, inliers = ransac_vp(np.array(lines), trials=300, eps_deg=0.5,
                            rng=np.random.default_rng(0))
    assert vp is not None
    # Inliers should be the first 4 lines.
    assert set(int(i) for i in inliers) == {0, 1, 2, 3}
    vp = vp / vp[2]
    np.testing.assert_allclose(vp[:2], vp_true[:2], atol=1e-3)


def test_ransac_returns_none_for_too_few_lines():
    vp, inl = ransac_vp(np.empty((0, 3)))
    assert vp is None and inl.size == 0


# ───────────────────────── vertical_vp_from_line_pitch ────────────────

def test_line_pitch_recovers_known_vertical_vp():
    """Synthesize baseline y-coordinates consistent with a known vertical VP."""
    # Construct ground-truth Mobius: i = (a y + b)/(c y + 1) with c = -1/y_v.
    y_v_true = -800.0          # vertical VP above the page (negative y)
    a, b = 0.02, 0.5
    c = -1.0 / y_v_true
    # Sample 8 lines at uniformly spaced integer indices, invert to get y.
    indices = np.arange(8, dtype=np.float64)
    # i (c y + 1) = a y + b  =>  y (a - i c) = i - b  =>  y = (i - b)/(a - i c)
    ys = (indices - b) / (a - indices * c)
    vp = vertical_vp_from_line_pitch(ys, indices, x_anchor=0.0)
    assert vp is not None
    y_v_est = vp[1] / vp[2]
    assert abs(y_v_est - y_v_true) < 1.0


def test_line_pitch_underdetermined_returns_none():
    assert vertical_vp_from_line_pitch(np.array([10.0, 30.0])) is None


# ───────────────────────── homography utilities ───────────────────────

def test_affine_rect_maps_vanishing_line_to_infinity():
    l = np.array([0.5, 1.0, -300.0])
    H = affine_rect_from_line(l)
    l_norm = l / l[2]
    # H^-T maps lines: the image-line ℓ should land on (0, 0, 1).
    l_after = np.linalg.inv(H).T @ l_norm
    np.testing.assert_allclose(l_after / l_after[2], np.array([0, 0, 1]), atol=1e-9)


def test_warp_extent_translates_to_origin():
    H = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.001, 0.0, 1.0]])
    ext = warp_extent(H, src_w=500, src_h=400)
    # Apply translated H to (0,0) → must land at or near (0,0).
    p = ext.H_corrected @ np.array([0.0, 0.0, 1.0])
    p /= p[2]
    assert p[0] >= -1 and p[1] >= -1
    assert ext.width > 0 and ext.height > 0


def test_oob_fraction_is_low_for_identity():
    H = np.eye(3)
    frac = oob_fraction(H, src_w=300, src_h=200, dst_w=300, dst_h=200,
                        samples=5000, rng=np.random.default_rng(0))
    assert frac < 0.02


# ───────────────────────── bbox_baseline ──────────────────────────────

def test_bbox_baseline_endpoints():
    pL, pR = bbox_baseline((10, 20, 100, 30))
    np.testing.assert_array_equal(pL, [10, 49])
    np.testing.assert_array_equal(pR, [109, 49])


def test_horizontal_vanishing_line_is_horizontal():
    vp_x = np.array([2000.0, 50.0, 1.0])
    l = horizontal_vanishing_line(vp_x, y_centroid=999.0)
    # Form a x + b y + c = 0 with a == 0 → horizontal.
    assert abs(l[0]) < 1e-9
    # Line passes through y = 50.
    assert abs(-l[2] / l[1] - 50.0) < 1e-9
