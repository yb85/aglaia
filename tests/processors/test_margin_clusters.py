# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Slope-bounded clustered margin fit (geometry §3.6.7).

Failure modes covered:
- blockquote-heavy pages where indented lines OUTNUMBER body lines
  (athanase 070 B / 256 A / 198 B / 191 B / 161 A — old anchor-box +
  max-cardinality RANSAC tilted into the indent cluster);
- page-edge ink noise (scan-44 A — killed the old p95 extremal seeds);
- clean pages (parity with the straight TLS answer).
"""
import numpy as np
import pytest

from lib.processors import geometry as g


W, H = 1400.0, 2000.0
B_L, M_L = 0.020, 150.0     # true left margin  x = M + B·y
B_R, M_R = 0.035, 1250.0    # true right margin
INDENT = 90.0


def _make_baselines(*, n_body=20, n_quote=0, quote_band=(700, 1400),
                    noise_right=0, jitter=3.0, seed=0):
    rng = np.random.default_rng(seed)
    baselines = []
    ys = np.linspace(100, 1900, n_body)
    for y in ys:
        xl = M_L + B_L * y + rng.uniform(-jitter, jitter)
        xr = M_R + B_R * y + rng.uniform(-jitter, jitter)
        baselines.append((np.array([xl, y]), np.array([xr, y])))
    if n_quote:
        yq = np.linspace(*quote_band, n_quote)
        for y in yq:
            xl = M_L + B_L * y + INDENT + rng.uniform(-jitter, jitter)
            xr = M_R + B_R * y - INDENT + rng.uniform(-jitter, jitter)
            baselines.append((np.array([xl, y]), np.array([xr, y])))
    for _ in range(noise_right):
        y = rng.uniform(200, 1800)
        xl = M_L + B_L * y + rng.uniform(-jitter, jitter)
        baselines.append((np.array([xl, y]), np.array([W - 5.0, y])))
    return baselines


def _x_on_line(line, y):
    a, b, c = line
    assert abs(a) > 1e-9
    return (-b * y - c) / a


def _quad_edge_x(quad, side, y):
    """x of the quad's left/right edge at height y (linear interp)."""
    tl, tr, br, bl = quad
    top, bot = (tl, bl) if side == "left" else (tr, br)
    t = (y - top[1]) / (bot[1] - top[1])
    return top[0] + t * (bot[0] - top[0])


# ---------------------------------------------------------------------------
# Unit: bounded Theil-Sen + clustered fit
# ---------------------------------------------------------------------------

def test_theil_sen_slope_immune_to_parallel_subpopulation():
    bls = _make_baselines(n_body=12, n_quote=18)
    pts = np.array([pL for pL, _ in bls])
    b = g._theil_sen_slope_bounded(pts, max_tilt_deg=7.0)
    assert b == pytest.approx(B_L, abs=0.006)


def test_theil_sen_slope_clipped_to_bound():
    rng = np.random.default_rng(1)
    y = np.linspace(0, 1000, 20)
    pts = np.column_stack([0.5 * y + rng.normal(0, 1, 20), y])  # ~26° tilt
    b = g._theil_sen_slope_bounded(pts, max_tilt_deg=7.0)
    assert abs(b) <= np.tan(np.radians(7.0)) + 1e-9


def test_clustered_fit_picks_outermost_supported_cluster():
    bls = _make_baselines(n_body=12, n_quote=18)  # quotes outnumber body
    left_pts = np.array([pL for pL, _ in bls])
    right_pts = np.array([pR for _, pR in bls])
    lf = g.fit_margin_line_clustered(left_pts, side="left",
                                     cluster_gap_px=33.0)
    rf = g.fit_margin_line_clustered(right_pts, side="right",
                                     cluster_gap_px=33.0)
    assert lf is not None and rf is not None
    left_line, li = lf
    right_line, ri = rf
    assert li["n_clusters"] >= 2 and ri["n_clusters"] >= 2
    assert li["cluster_size"] == 12 and ri["cluster_size"] == 12
    for y in (100.0, 1000.0, 1900.0):
        assert _x_on_line(left_line, y) == pytest.approx(
            M_L + B_L * y, abs=5.0)
        assert _x_on_line(right_line, y) == pytest.approx(
            M_R + B_R * y, abs=5.0)


def test_clustered_fit_rejects_low_support_edge_noise():
    bls = _make_baselines(n_body=20, noise_right=3)  # scan-44 A pattern
    right_pts = np.array([pR for _, pR in bls])
    rf = g.fit_margin_line_clustered(right_pts, side="right",
                                     cluster_gap_px=33.0)
    assert rf is not None
    right_line, ri = rf
    # The 3 page-edge points fail the 25 %-of-largest support floor.
    assert ri["cluster_size"] == 20
    for y in (100.0, 1900.0):
        assert _x_on_line(right_line, y) == pytest.approx(
            M_R + B_R * y, abs=5.0)


def test_clustered_fit_clean_page_single_cluster():
    bls = _make_baselines(n_body=20)
    left_pts = np.array([pL for pL, _ in bls])
    lf = g.fit_margin_line_clustered(left_pts, side="left",
                                     cluster_gap_px=33.0)
    assert lf is not None
    line, info = lf
    assert info["n_clusters"] == 1
    ref = g._fit_line_lstsq(left_pts)
    for y in (100.0, 1900.0):
        assert _x_on_line(line, y) == pytest.approx(
            _x_on_line(ref, y), abs=1.0)


# ---------------------------------------------------------------------------
# End-to-end: detect_column_quad_from_baselines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_body,n_quote,noise", [
    (20, 0, 0),     # clean
    (12, 18, 0),    # blockquote-heavy (quotes outnumber body)
    (20, 10, 3),    # quotes + page-edge noise
])
def test_quad_edges_track_true_margins(n_body, n_quote, noise):
    bls = _make_baselines(n_body=n_body, n_quote=n_quote,
                          noise_right=noise)
    res = g.detect_column_quad_from_baselines(bls)
    assert res is not None
    quad, info = res
    assert info["column_edge_source"] == "endpoint_clustering"
    y_lo = min(min(pL[1], pR[1]) for pL, pR in bls)
    y_hi = max(max(pL[1], pR[1]) for pL, pR in bls)
    for y in (y_lo, y_hi):
        assert _quad_edge_x(quad, "left", y) == pytest.approx(
            M_L + B_L * y, abs=8.0)
        assert _quad_edge_x(quad, "right", y) == pytest.approx(
            M_R + B_R * y, abs=8.0)


def test_quad_edge_tilts_consistent():
    bls = _make_baselines(n_body=12, n_quote=18)
    res = g.detect_column_quad_from_baselines(bls)
    assert res is not None
    quad, _ = res
    tl, tr, br, bl = quad
    tilt_l = np.degrees(np.arctan2(abs(bl[0] - tl[0]), abs(bl[1] - tl[1])))
    tilt_r = np.degrees(np.arctan2(abs(br[0] - tr[0]), abs(br[1] - tr[1])))
    assert abs(tilt_l - tilt_r) < 2.5
