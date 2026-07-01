# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""XY-cut X-projection page splitter.

Groups text-detection boxes into columns/pages at vertical whitespace gutters
wider than `min_gap_frac × page_w`. Replaces the old pairwise x-overlap
clustering, whose ANY-gap split orphaned a flush page-number box into its own
page (the DBnet 2-up mis-split)."""

from __future__ import annotations

from aglaia.processors.PageDetector import xy_cut

W = 1920


def _col(x0, x1, n=15, y0=300, dy=40):
    return [(x0, y0 + dy * i, x1, y0 + 18 + dy * i) for i in range(n)]


def test_empty():
    assert xy_cut([], page_w=W) == []


def test_single_page_one_group():
    out = xy_cut(_col(200, 1700), page_w=W)
    assert len(out) == 1
    assert out[0][0] == 200 and out[0][2] == 1700


def test_two_page_spread_splits_at_gutter():
    out = xy_cut(_col(354, 874) + _col(977, 1510), page_w=W)
    assert len(out) == 2
    assert out[0][2] <= 900 and out[1][0] >= 950   # split at the gutter


def test_flush_page_number_speck_folds_into_column():
    # The regression: a page-number box touching the body column (0 px gap)
    # must NOT become its own group — it rides with the left page.
    speck = (340, 90, 354, 98)
    out = xy_cut([speck] + _col(354, 874) + _col(977, 1510), page_w=W)
    assert len(out) == 2, f"speck must not spawn a 3rd group: {out}"
    left = min(out, key=lambda r: r[0])
    assert left[0] == 340 and left[2] == 874   # speck absorbed into left bounds


def test_sub_gutter_gap_does_not_split():
    # A gap narrower than min_gap_frac × W (here 2% = 38 px) stays one column.
    out = xy_cut(_col(200, 700) + _col(720, 1200), page_w=W, min_gap_frac=0.02)
    assert len(out) == 1, f"20 px gap < 38 px gutter must not split: {out}"


def test_real_gutter_above_threshold_splits():
    # Same columns, but now a 120 px gap (> 38 px) is a real gutter.
    out = xy_cut(_col(200, 700) + _col(820, 1320), page_w=W, min_gap_frac=0.02)
    assert len(out) == 2


def test_three_columns():
    out = xy_cut(_col(100, 500) + _col(700, 1100) + _col(1300, 1700), page_w=W)
    assert len(out) == 3
    assert [r[0] for r in out] == sorted(r[0] for r in out)   # left→right
