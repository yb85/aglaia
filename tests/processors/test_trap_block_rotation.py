# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Block-justified VP filtering + rotation-consistency in the trap quad
detector (athanase 055 A failure: invented ~6° rotation on a blockquote-
heavy upright page)."""

import numpy as np

from aglaia.processors.geometry import (
    block_justified_mask, detect_column_quad_from_baselines,
)


def _bl(xl, xr, y):
    return (np.array([float(xl), float(y)]), np.array([float(xr), float(y)]))


def test_block_justified_excludes_singleton_endpoints():
    bls = [_bl(100, 600, 100 + i * 40) for i in range(10)]   # shared margins
    bls.append(_bl(20, 120, 60))    # page number: unique left AND right
    L = np.array([p[0] for p in bls])
    R = np.array([p[1] for p in bls])
    mask = block_justified_mask(L, R, gap_px=15, min_support=3)
    assert mask[:10].all()          # body lines justified
    assert not mask[10]             # page number dropped (one-off endpoints)


def test_block_justified_keeps_blockquote_block():
    # body block + an indented blockquote block (its OWN shared margins).
    bls = [_bl(100, 600, 100 + i * 30) for i in range(8)]
    bls += [_bl(180, 520, 400 + i * 30) for i in range(5)]   # indent block
    L = np.array([p[0] for p in bls])
    R = np.array([p[1] for p in bls])
    mask = block_justified_mask(L, R, gap_px=15, min_support=3)
    assert mask.all()               # both blocks justified to their margins


def _left_tilt(quad):
    tl, _tr, _br, bl = quad
    return float(np.degrees(np.arctan2(bl[0] - tl[0], bl[1] - tl[1])))


def _right_tilt(quad):
    _tl, tr, br, _bl = quad
    return float(np.degrees(np.arctan2(br[0] - tr[0], br[1] - tr[1])))


def test_rotation_consistency_flattens_invented_rotation():
    # Horizontal baselines (upright page) but PARALLEL +5.7° margins — the
    # 055 A signature. The detector must NOT rotate the page: edges ~vertical.
    bls = [_bl(100 + 0.10 * y, 600 + 0.10 * y, y)
           for y in (100.0 + 40 * i for i in range(14))]
    res = detect_column_quad_from_baselines(bls, ransac_trials=200)
    assert res is not None
    quad, _ = res
    assert abs(_left_tilt(quad)) < 1.5
    assert abs(_right_tilt(quad)) < 1.5


def test_real_keystone_is_preserved():
    # Converging margins (different slopes) = genuine keystone → kept.
    bls = [_bl(100 + 0.02 * y, 600 + 0.06 * y, y)
           for y in (100.0 + 40 * i for i in range(14))]
    res = detect_column_quad_from_baselines(bls, ransac_trials=200)
    assert res is not None
    quad, _ = res
    # still converging — not flattened to a rectangle
    assert abs(_left_tilt(quad) - _right_tilt(quad)) > 0.8
