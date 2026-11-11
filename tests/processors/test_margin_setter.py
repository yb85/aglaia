# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The margin you set is the margin you get (#112).

`MarginSetter` crops to ink and pads. `_enforce_width_floor` then padded the
page back out to the step's INPUT width whenever the crop came out narrower —
which is every page, because that input is the dewarp canvas and the crop's
whole job is to strip the whitespace it carries. So the horizontal margin was
never the requested one. Measured over 40 real pages asking for 5 mm: top and
bottom exactly 5.0 mm, left and right 10.2-16.8 mm and varying page to page.

The floor stays available (`width_floor`), off by default.
"""
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.MarginSetter import MarginSetter, MarginSetterOption

DPI = 300.0


def _page(*, content=(200, 300), canvas=(1200, 900)):
    """A black content block centred in a much wider white canvas — the shape
    the dewarp hands over: real ink, lots of leftover whitespace."""
    ch, cw = content
    h, w = canvas
    img = np.full((h, w), 255, dtype=np.uint8)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    img[y0:y0 + ch, x0:x0 + cw] = 0
    return ImageBuffer(img, ImageType.BW, dpi=DPI)


def _borders(arr):
    ink = arr < 128
    cols, rws = ink.any(axis=0), ink.any(axis=1)
    x0 = int(np.argmax(cols))
    x1 = len(cols) - 1 - int(np.argmax(cols[::-1]))
    y0 = int(np.argmax(rws))
    y1 = len(rws) - 1 - int(np.argmax(rws[::-1]))
    return x0, len(cols) - 1 - x1, y0, len(rws) - 1 - y1


def _mm_px(mm):
    return int(round(mm * DPI / 25.4))


def test_all_four_margins_equal_the_requested_value():
    out = MarginSetter(MarginSetterOption(margin_mm="2")).process(_page())
    pad = _mm_px(2)
    assert _borders(out.buffer) == (pad, pad, pad, pad)


def test_the_default_is_two_millimetres():
    """No `margin_mm` in the YAML must not mean "no margin" — the shipped
    pipelines all state 2, and the bare default agrees with them."""
    out = MarginSetter(MarginSetterOption()).process(_page())
    pad = _mm_px(2)
    assert _borders(out.buffer) == (pad, pad, pad, pad)


def test_a_wide_canvas_is_cropped_away_not_padded_back():
    """The regression itself: the output is content + 2×margin, NOT the
    canvas it arrived on."""
    buf = _page(content=(200, 300), canvas=(1200, 900))
    out = MarginSetter(MarginSetterOption(margin_mm="2")).process(buf)
    assert out.buffer.shape[1] == 300 + 2 * _mm_px(2)
    assert out.buffer.shape[1] < 900


def test_the_width_floor_still_works_when_asked_for():
    buf = _page(content=(200, 300), canvas=(1200, 900))
    out = MarginSetter(
        MarginSetterOption(margin_mm="2", width_floor=True)).process(buf)
    assert out.buffer.shape[1] == 900
    # Vertical margin is untouched; horizontal is now the floor's padding.
    _l, _r, t, b = _borders(out.buffer)
    assert (t, b) == (_mm_px(2), _mm_px(2))


def test_the_stamp_says_whether_a_floor_applied():
    """`apply_replay` reproduces the forward pass from this stamp, and the
    shipped pipelines all run `replay: true`."""
    off = MarginSetter(MarginSetterOption(margin_mm="2")).process(_page())
    assert off.meta["replay_params"]["min_width_px"] == 0
    on = MarginSetter(
        MarginSetterOption(margin_mm="2", width_floor=True)).process(_page())
    assert on.meta["replay_params"]["min_width_px"] == 900


def test_replay_matches_the_forward_pass_both_ways():
    for floor in (False, True):
        buf = _page()
        fwd = MarginSetter(
            MarginSetterOption(margin_mm="2", width_floor=floor)
        ).process(_page())
        mask = np.full(buf.buffer.shape, 255, dtype=np.uint8)
        out, _out_mask = MarginSetter.apply_replay(
            buf.buffer, mask, fwd.meta["replay_params"], {})
        assert out.shape == fwd.buffer.shape, floor


def test_an_old_stamp_without_the_key_keeps_its_floor():
    """Chains stamped before `width_floor` existed carry an unconditional
    `min_width_px`; replaying one must still reproduce what it produced."""
    buf = _page()
    mask = np.full(buf.buffer.shape, 255, dtype=np.uint8)
    params = {"ltrb_px": [24, 24, 24, 24],
              "content_bbox_xywh": [300, 350, 300, 200],
              "min_width_px": 900}
    out, _ = MarginSetter.apply_replay(buf.buffer, mask, params, {})
    assert out.shape[1] == 900


def test_css_shorthand_still_parses():
    out = MarginSetter(MarginSetterOption(margin_mm="2 6")).process(_page())
    left, right, top, bottom = _borders(out.buffer)
    assert (top, bottom) == (_mm_px(2), _mm_px(2))
    assert (left, right) == (_mm_px(6), _mm_px(6))


@pytest.mark.parametrize("path", [
    "book_curved_x2", "book_flat_x1", "book_flat_x2", "sheet_flat_x1",
])
def test_every_shipped_pipeline_asks_for_two_millimetres(path):
    import yaml
    from pathlib import Path
    import aglaia
    doc = yaml.safe_load(
        (Path(aglaia.__file__).parent / "config" / "pipelines"
         / f"{path}.yaml").read_text())
    steps = [s for s in doc["pipeline"] if s["processor"] == "MarginSetter"]
    assert steps, path
    for s in steps:
        assert float(s["options"]["margin_mm"]) == 2.0
        assert not s["options"].get("width_floor")


# ── fixed page size (#197) ───────────────────────────────────────────

A5 = (_mm_px(148), _mm_px(210))


def _fixed(**kw):
    return MarginSetter(MarginSetterOption(margin_mm="2", fixed_size=True, **kw))


def test_fixed_size_defaults_to_a5_centred_and_top_aligned():
    out = _fixed().process(_page(content=(200, 300))).buffer
    assert out.shape[::-1] == A5
    l, r, t, b = _borders(out)
    assert abs(l - r) <= 1 and t == _mm_px(2) and b > t


@pytest.mark.parametrize("h,v,expect", [
    ("left", "top", lambda l, r, t, b: (l, t) == (_mm_px(2), _mm_px(2))),
    ("right", "bottom", lambda l, r, t, b: (r, b) == (_mm_px(2), _mm_px(2))),
])
def test_alignment_puts_the_block_against_the_named_edges(h, v, expect):
    out = _fixed(h_align=h, v_align=v).process(_page()).buffer
    assert out.shape[::-1] == A5
    assert expect(*_borders(out))


def _big():
    # Content wider than A5 (148 mm = 1748 px at 300 dpi), height fits.
    return _page(content=(400, 2000), canvas=(800, 2400))


def test_overflow_without_clip_grows_the_page_on_that_axis_only():
    out = _fixed().process(_big()).buffer
    assert out.shape[1] == 2000 + 2 * _mm_px(2)
    assert out.shape[0] == A5[1]


@pytest.mark.parametrize("h", ["left", "center", "right"])
def test_overflow_with_clip_keeps_the_page_size(h):
    out = _fixed(clip=True, h_align=h).process(_big()).buffer
    assert out.shape[::-1] == A5
    l, r, _, _ = _borders(out)
    if h == "left":
        assert l == _mm_px(2) and r == 0       # cut on the right
    elif h == "right":
        assert r == _mm_px(2) and l == 0       # cut on the left
    else:
        assert l == r == 0                     # cut on both sides


def test_fixed_size_ignores_the_width_floor():
    out = _fixed(width_floor=True).process(_page(canvas=(1200, 3000))).buffer
    assert out.shape[::-1] == A5


def test_replay_reproduces_the_fixed_page():
    buf = _page(content=(200, 300))
    src = buf.buffer.copy()
    out = _fixed(h_align="left", v_align="bottom").process(buf)
    mask = np.full(src.shape, 255, dtype=np.uint8)
    rep, rep_mask = MarginSetter.apply_replay(
        src, mask, out.meta["replay_params"], None)
    assert np.array_equal(rep, out.buffer)
    assert rep_mask.shape == rep.shape


def test_a_bad_page_size_is_refused():
    with pytest.raises(ValueError):
        _fixed(page_size_mm="148")


@pytest.mark.parametrize("shape", [(1, 1), (3000, 2500)])
def test_an_empty_page_always_comes_out_at_the_page_size(shape):
    """A blank page (#196) — 1 px placeholder or larger than A5 — matches the
    fixed size even with clip off: there is no content to keep."""
    buf = ImageBuffer(np.full(shape, 255, np.uint8), ImageType.BW, dpi=DPI)
    src = buf.buffer.copy()
    out = _fixed().process(buf)
    assert out.buffer.shape[::-1] == A5
    assert (out.buffer == 255).all()
    rep, _ = MarginSetter.apply_replay(
        src, np.full(src.shape, 255, np.uint8), out.meta["replay_params"], None)
    assert rep.shape == out.buffer.shape
