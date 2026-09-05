# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""One estimator, shared; meta as its cache (#143).

TrapezoidalCorrection and PageDewarper each carried a copy of the same
character-height estimator. Two copies of one estimate drift apart. Now both
call `text_metrics`, and the dewarper reads the keystone step's result from
meta when it is there — so the pipeline works without the keystone step, the
two agree by construction, and the only thing meta buys is skipping a
recomputation.
"""
import cv2
import numpy as np

from aglaia.processors import text_metrics as tm


def _page(char_h=24, n_lines=12, dpi=300.0):
    """Rows of rectangular 'glyphs' of one height on a white page."""
    img = np.zeros((900, 700), np.uint8)
    for i in range(n_lines):
        y = 60 + i * 60
        for x in range(40, 660, 22):
            img[y:y + char_h, x:x + 12] = 255
    return img


def test_the_median_is_the_glyph_height():
    ink = _page(char_h=24)
    stats = cv2.connectedComponentsWithStats(ink, connectivity=4)[2]
    assert tm.median_char_height(stats, 300.0) == 24.0


def test_too_few_components_is_no_estimate():
    ink = _page(char_h=24, n_lines=1)[0:100]          # one short row
    stats = cv2.connectedComponentsWithStats(ink, connectivity=4)[2]
    assert stats.shape[0] - 1 < tm.MIN_COMPONENTS
    assert tm.median_char_height(stats, 300.0) == 0.0


def test_bounds_scale_with_dpi():
    assert tm.cc_bounds(300.0) == (12, 135, 6, 180)
    assert tm.cc_bounds(100.0) == (4, 45, 2, 60)
    # 10 dpi: round(0.45*10) is 4 under banker's rounding; floors and the
    # min+1 guards hold either way.
    assert tm.cc_bounds(10.0) == (3, 4, 2, 6)


def test_the_meta_form_is_dimensionless_and_round_trips():
    frac = tm.char_h_frac(24.0, 900)
    assert abs(frac - 24 / 900) < 1e-9
    assert tm.cached_char_height({"char_h_frac": frac}, 900) == 24.0
    assert tm.cached_char_height({"char_h_frac": frac}, 450) == 12.0   # after a 0.5x resample


def test_no_cache_means_compute():
    assert tm.cached_char_height({}, 900) is None
    assert tm.cached_char_height(None, 900) is None
    assert tm.cached_char_height({"char_h_frac": 0}, 900) is None
    assert tm.cached_char_height({"char_h_frac": "junk"}, 900) is None


def test_the_dewarper_uses_the_cached_value_when_present():
    """Same mask, with and without the hint: with it, the CC median is not
    what decides — the hint is."""
    from aglaia.processors.PageDewarper import _text_mask_dpi
    ink = _page(char_h=24)
    rgb = cv2.cvtColor(cv2.bitwise_not(ink), cv2.COLOR_GRAY2RGB)
    pagemask = np.full(ink.shape, 255, np.uint8)
    _, h_computed = _text_mask_dpi(rgb, pagemask, 300.0, 2.0)
    _, h_hinted = _text_mask_dpi(rgb, pagemask, 300.0, 2.0, h_med_hint=40.0)
    assert h_computed == 24.0
    assert h_hinted == 40.0


def test_neither_processor_carries_its_own_estimator_any_more():
    """The duplication is what this replaced; it must not grow back."""
    import pathlib
    for f in ("TrapezoidalCorrection.py", "PageDewarper.py"):
        src = pathlib.Path("aglaia/processors", f).read_text("utf-8")
        assert "dpi * 0.04" not in src and "analysis_dpi * 0.04" not in src, f
        assert "text_metrics" in src, f
