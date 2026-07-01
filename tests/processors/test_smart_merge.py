# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Score-based page merge — canonical scenarios.

Cases:
  TOC          — wide main column + narrow page-number column → merge.
  2-page spread — two balanced columns separated by gutter   → keep.
  Header stack — wide narrow header above wide tall body      → merge.
  Two-col body — true 2-column page, balanced + close gap   → keep.
  Sidebar      — narrow sidebar w/ large margin               → keep.
  Cap overshoot — 4 pages, max=2, all below threshold        → force-merge twice.
"""
from __future__ import annotations

import pytest

from aglaia.processors.PageDetector import _pair_score, smart_merge


PAGE_W = 1000
PAGE_H = 1400


def _merge(pages, *, max_pages=2, threshold=0.60):
    return list(smart_merge(
        pages,
        max_pages=max_pages,
        page_w=PAGE_W, page_h=PAGE_H,
        threshold=threshold,
    ))


# ── score sanity ──────────────────────────────────────────────────────

def test_score_toc_high():
    """TOC: wide text column + narrow page-number column, same Y range,
    tiny X gap → score must be high (≥ default threshold)."""
    text = (60, 100, 760, 1300)        # w ≈ 700
    nums = (820, 100, 920, 1300)       # w ≈ 100, narrow
    s = _pair_score(text, nums,
                    page_w=PAGE_W, page_h=PAGE_H,
                    gap_weight=0.4, width_weight=0.6, gap_norm_cap=0.15)
    assert s >= 0.60, f"TOC pair score too low: {s:.3f}"


def test_score_two_page_spread_low():
    """Balanced 2-page spread, sane gutter → score below threshold."""
    left = (60, 100, 460, 1300)        # w = 400
    right = (540, 100, 940, 1300)      # w = 400, gutter ~ 80
    s = _pair_score(left, right,
                    page_w=PAGE_W, page_h=PAGE_H,
                    gap_weight=0.4, width_weight=0.6, gap_norm_cap=0.15)
    assert s < 0.60, f"spread pair score too high: {s:.3f}"


def test_score_header_stacked_high():
    """Thin header above wide body, same X range, small Y gap → merge."""
    header = (100, 60, 900, 110)       # h = 50, thin
    body = (100, 130, 900, 1300)       # h = 1170, tall
    s = _pair_score(header, body,
                    page_w=PAGE_W, page_h=PAGE_H,
                    gap_weight=0.4, width_weight=0.6, gap_norm_cap=0.15)
    assert s >= 0.60, f"header/body pair score too low: {s:.3f}"


def test_score_two_col_body_low():
    """Balanced 2-column body w/ small gap → keep (score low)."""
    left = (60, 100, 460, 1300)
    right = (500, 100, 900, 1300)      # gap 40, both 400 wide
    s = _pair_score(left, right,
                    page_w=PAGE_W, page_h=PAGE_H,
                    gap_weight=0.4, width_weight=0.6, gap_norm_cap=0.15)
    assert s < 0.60, f"2-col body score too high: {s:.3f}"


# ── full merge loop ───────────────────────────────────────────────────

def test_merge_toc_collapses_to_one():
    text = (60, 100, 760, 1300)
    nums = (820, 100, 920, 1300)
    out = _merge([text, nums], max_pages=2)
    assert len(out) == 1, f"TOC should merge: {out}"
    x1, y1, x2, y2 = out[0]
    assert x1 == 60 and x2 == 920
    assert y1 == 100 and y2 == 1300


def test_merge_spread_kept_separate():
    left = (60, 100, 460, 1300)
    right = (540, 100, 940, 1300)
    out = _merge([left, right], max_pages=2)
    assert len(out) == 2, f"spread must not merge: {out}"


def test_merge_header_body_collapses():
    header = (100, 60, 900, 110)
    body = (100, 130, 900, 1300)
    out = _merge([header, body], max_pages=2)
    assert len(out) == 1, f"header+body should merge: {out}"


def test_merge_two_col_body_kept():
    left = (60, 100, 460, 1300)
    right = (500, 100, 900, 1300)
    out = _merge([left, right], max_pages=2)
    assert len(out) == 2, f"two-col body must not merge: {out}"


def test_merge_cap_forces_below_threshold():
    """Cap trigger fires even when no pair meets threshold.

    Threshold is bumped sky-high so only the capacity rule can act —
    confirms cap merges happen regardless of score. With 3 separated
    cols and max=2, cap forces exactly one merge."""
    cols = [
        (40, 100, 200, 1300),
        (400, 100, 560, 1300),
        (760, 100, 920, 1300),
    ]
    out = _merge(cols, max_pages=2, threshold=10.0)
    assert len(out) == 2, f"capacity cap must enforce: {out}"


def test_merge_speck_absorbed_not_pages_fused():
    """Capacity-forced merge folds a small fragment into a neighbour rather
    than fusing the two real pages.

    Regression for the DBnet 2-up mis-split: a detector drops an isolated
    page-number box a hair left of the body column, so it survives as its own
    group → 3 groups, max=2. The old code force-merged the highest-scoring
    pair (the two balanced pages → one giant 'right' page, speck → 'left').
    The fix absorbs the SMALLEST group into its best neighbour, so the two
    side-by-side pages stay split and the number rides with the left page."""
    speck = (340, 90, 354, 98)        # page number, just left of the column
    left = (354, 77, 874, 975)
    right = (977, 83, 1510, 914)
    out = _merge([speck, left, right], max_pages=2, threshold=0.60)
    assert len(out) == 2, f"two real pages must survive: {out}"
    out = sorted(out, key=lambda p: p[0])
    # Left page swallowed the speck (its left edge moved out to the speck).
    assert out[0][0] <= 354 and out[0][2] >= 874
    # Right page is untouched — NOT fused with the left.
    assert out[1][0] >= 970 and out[1][2] >= 1510
    # Centres stay well separated → a real side-by-side spread.
    cl = (out[0][0] + out[0][2]) / 2
    cr = (out[1][0] + out[1][2]) / 2
    assert cr - cl > 600


def test_merge_greedy_collapses_imbalanced_chain():
    """Documents (not regrets) the greedy behavior: a chain of close
    columns telescopes via width-imbalance once any merge widens one
    side. Real-world callers should set max_pages to match the
    intended column count to prevent this."""
    cols = [
        (60, 100, 230, 1300),
        (260, 100, 430, 1300),
        (560, 100, 730, 1300),
        (760, 100, 930, 1300),
    ]
    out = _merge(cols, max_pages=2)
    assert len(out) == 1, (
        f"greedy width-imbalance is expected to telescope chain: {out}"
    )


def test_merge_no_cap_keeps_separate():
    """max_pages=0 disables capacity trigger; nothing should merge if
    no pair exceeds threshold."""
    left = (60, 100, 460, 1300)
    right = (540, 100, 940, 1300)
    out = _merge([left, right], max_pages=0)
    assert len(out) == 2


def test_merge_toc_plus_spread_combined():
    """2-page spread, each page has its own page-number column.
    Expect: 4 input → 2 final (per-page text+nums collapse, spread kept)."""
    pages = [
        # left page
        (60, 100, 380, 1300),    # text
        (400, 100, 460, 1300),   # page #
        # right page
        (540, 100, 860, 1300),   # text
        (880, 100, 940, 1300),   # page #
    ]
    out = _merge(pages, max_pages=2)
    assert len(out) == 2, f"expected 2 final pages: {out}"
    # Each survivor should span at least 220 px wide (text+nums combined).
    widths = sorted(r[2] - r[0] for r in out)
    assert widths[0] >= 320, f"survivors too narrow: {widths}"


# ── contrast bias (optional knob, off by default in caller) ───────────

def test_contrast_bonus_param_still_supported():
    """`smart_merge` accepts a `contrast_bonuses` list — the PageDetector
    no longer feeds it (hard-drop filter runs first instead) but the
    primitive must still honour it for callers that want the bias path."""
    real = (60, 100, 480, 1300)
    ghost = (520, 100, 940, 1300)
    no_bias = list(smart_merge(
        [real, ghost],
        max_pages=2, page_w=PAGE_W, page_h=PAGE_H,
        threshold=0.60, contrast_bonuses=[0.0, 0.0],
    ))
    assert len(no_bias) == 2

    with_bias = list(smart_merge(
        [real, ghost],
        max_pages=2, page_w=PAGE_W, page_h=PAGE_H,
        threshold=0.60, contrast_bonuses=[0.0, 0.40],
    ))
    assert len(with_bias) == 1


# ── degenerate ────────────────────────────────────────────────────────

def test_merge_single_layout_no_op():
    out = _merge([(60, 60, 900, 1300)], max_pages=2)
    assert len(out) == 1


def test_merge_empty():
    out = _merge([], max_pages=2)
    assert out == []
