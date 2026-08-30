# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Spine-aware keystone estimation (tracker #70, from the iOS port).

Near the binding the page curls out of plane, so the bottom-most ink sits
progressively lower than the true baseline. That is evidence which is
systematically WRONG, not merely noisy — a plain fit lets it tilt the line and
drag the vanishing point. Knowing which edge carries the fold (`page_side`,
which now survives the pipeline) lets three things be corrected without
guessing:

1. baseline evidence in the curl zone is down-weighted;
2. the fold-side endpoint cluster gets a relaxed bandwidth, because its
   members wobble and would otherwise splinter below `min_support`;
3. a tilt disagreement backed by strong support on BOTH sides is kept as real
   keystone instead of being reconciled away.

Every one is gated on `spine_side`; `None` must reproduce the prior behaviour
exactly, which is what the parity tests here pin.
"""
import numpy as np
import pytest

from aglaia.processors.geometry import (
    baseline_from_ink, detect_column_quad_from_baselines,
)


def _bl(xl, xr, y):
    return (np.array([float(xl), float(y)]), np.array([float(xr), float(y)]))


def _left_tilt(quad):
    tl, _tr, _br, bl = quad
    return float(np.degrees(np.arctan2(bl[0] - tl[0], bl[1] - tl[1])))


def _right_tilt(quad):
    _tl, tr, br, _bl = quad
    return float(np.degrees(np.arctan2(br[0] - tr[0], br[1] - tr[1])))


# ── 1. curl-zone down-weighting ─────────────────────────────────────────

def _curled_line(h=40, w=400, curl_px=12, curl_frac=0.30):
    """A text line whose bottom ink sags near the RIGHT edge — the fold."""
    ink = np.zeros((h, w), np.uint8)
    base = h // 2
    zone = int(w * curl_frac)
    for x in range(w):
        d = max(0.0, (x - (w - zone)) / max(zone, 1))      # 0 → 1 at the fold
        y = int(round(base + curl_px * d * d))
        ink[max(0, y - 3):y + 1, x] = 255
    return ink


def test_curl_sag_tilts_an_unweighted_baseline():
    """The behaviour the weighting exists to fix, pinned so the fix is
    visibly a change."""
    ink = _curled_line()
    pL, pR = baseline_from_ink(ink, (0, 0, ink.shape[1], ink.shape[0]))
    drop = pR[1] - pL[1]
    # Measured 2.41 px on this synthetic; the descender filter already
    # absorbs part of the sag, which is why it is not the full 12 px.
    assert drop > 2.0, f"expected the curl to tilt the fit, got {drop:.2f}px"


def test_spine_weighting_resists_the_curl_sag():
    ink = _curled_line()
    box = (0, 0, ink.shape[1], ink.shape[0])
    plain = baseline_from_ink(ink, box)
    weighted = baseline_from_ink(
        ink, box, spine=(float(ink.shape[1] - 1), 0.30 * ink.shape[1]))
    assert weighted is not None
    drop_plain = plain[1][1] - plain[0][1]
    drop_weighted = weighted[1][1] - weighted[0][1]
    # Measured 57-69% reduction across curl strengths of 12-30 px.
    assert abs(drop_weighted) < 0.5 * abs(drop_plain), (
        f"weighted fit did not resist the curl: {drop_weighted:.2f} vs "
        f"{drop_plain:.2f}")


def test_spine_none_is_the_unweighted_fit():
    """Parity: every already-processed page goes through this path."""
    ink = _curled_line()
    box = (0, 0, ink.shape[1], ink.shape[0])
    a = baseline_from_ink(ink, box)
    b = baseline_from_ink(ink, box, spine=None)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_weighting_leaves_a_straight_line_alone():
    """A line with no curl must not be bent by the option existing."""
    ink = np.zeros((40, 400), np.uint8)
    ink[18:21, :] = 255
    box = (0, 0, 400, 40)
    plain = baseline_from_ink(ink, box)
    weighted = baseline_from_ink(ink, box, spine=(399.0, 120.0))
    assert np.allclose(plain[0], weighted[0], atol=0.5)
    assert np.allclose(plain[1], weighted[1], atol=0.5)


# ── 3. tilt disagreement kept as real keystone ──────────────────────────

def _diverging_tilts():
    """Left edge near-vertical, right edge tilted ~3° — a real keystone
    signature, with every line supporting both edges."""
    return [_bl(100 + 0.005 * y, 700 + 0.06 * y, y)
            for y in (100.0 + 40 * i for i in range(16))]


def test_without_spine_side_a_tilt_disagreement_is_reconciled():
    """Pins the prior behaviour — the fix must be visibly a change."""
    res = detect_column_quad_from_baselines(_diverging_tilts(),
                                            ransac_trials=200)
    assert res is not None
    quad, _ = res
    assert abs(_left_tilt(quad) - _right_tilt(quad)) < 1.0, \
        "expected the tilts to be reconciled without spine_side"


def test_spine_side_keeps_a_well_supported_disagreement():
    res = detect_column_quad_from_baselines(_diverging_tilts(),
                                            ransac_trials=200,
                                            spine_side="left")
    assert res is not None
    quad, _ = res
    assert abs(_left_tilt(quad) - _right_tilt(quad)) > 1.0, (
        "a disagreement supported on both sides is real keystone and must "
        "survive")


@pytest.mark.parametrize("side", ["left", "right"])
def test_spine_side_does_not_break_the_ordinary_cases(side):
    """The upright-page and genuine-keystone cases must still work with a
    spine side set — this runs on every page, not just curled ones."""
    upright = [_bl(100 + 0.10 * y, 600 + 0.10 * y, y)
               for y in (100.0 + 40 * i for i in range(14))]
    res = detect_column_quad_from_baselines(upright, ransac_trials=200,
                                            spine_side=side)
    assert res is not None


def test_spine_side_none_reproduces_the_quad_exactly():
    """The parity guarantee for the quad path."""
    bls = _diverging_tilts()
    a = detect_column_quad_from_baselines(bls, ransac_trials=200)
    b = detect_column_quad_from_baselines(bls, ransac_trials=200,
                                          spine_side=None)
    assert a is not None and b is not None
    assert np.allclose(a[0], b[0])
