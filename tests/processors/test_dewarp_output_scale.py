# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Output sizing of the dewarp remap grid.

`page_dims` lives in norm2pix units, whose scale is 0.5 * max(h, w). Upstream
page-dewarp sized the output height against `img.shape[0]` instead, which is
the same number only while the crop is portrait. On a LANDSCAPE crop — a
chapter opening, a part-title, any short page whose text block is wider than
tall — the whole remap came out scaled by ref_h / ref_w, and the node was
still stamped 300 dpi. A 1289x533 page landed at 520x224.
"""
import numpy as np

from aglaia.processors.PageDewarper import PageDewarper


def _params():
    params = np.zeros(20, dtype=np.float32)
    params[0:3] = [0.02, -0.03, 0.01]        # rvec
    params[3:6] = [-0.6, -0.45, 1.9]         # tvec
    params[6], params[7] = 0.05, -0.02       # curl
    return params


def _target(ref_h, ref_w, page_dims_h):
    _pts, _shp, target_w, target_h, _ws, _hs = PageDewarper._sample_grid(
        ref_h, ref_w, params=_params(), page_dims_w=1.3,
        page_dims_h=page_dims_h, decimate=4, zoom=1.0, focal=1.3)
    return target_w, target_h


def test_landscape_page_keeps_its_resolution():
    """The regression. A wide-and-short crop must be measured against the
    long side, exactly as norm2pix scales the projected points."""
    _w, target_h = _target(653, 1409, 0.6827)
    assert abs(target_h - 0.5 * 0.6827 * 1409) <= 4  # decimate rounding


def test_portrait_sizing_is_unchanged():
    """Parity: on a portrait crop max(h, w) IS ref_h, so every already
    stamped project replays to the pixel size it has today."""
    _w, target_h = _target(2400, 1600, 0.95)
    assert target_h == 1140


def test_output_scale_does_not_depend_on_crop_orientation():
    """Same page height in norm units, same reference long side → same
    output height, whichever way round the crop happens to be."""
    assert _target(1400, 900, 0.8)[1] == _target(900, 1400, 0.8)[1]


# ── #202: the output keeps the input's pixel scale ──────────────────────────
#
# `page_dims` is FITTED, and the fit trades camera depth against page size:
# on two facing pages of one book (lubac.agl, scan 4) it came out 1.758 and
# 2.126 high while the height measured in the image (`rough_dims`, stamped as
# `model_dims`) was 1.960 on both. Sized from page_dims alone, the left page
# shrank 11 % and the right one grew 9 %.

def _height(ref_h, ref_w, rough_h, page_h):
    zoom = PageDewarper._scale_zoom((1.1, rough_h), (1.1, page_h))
    _pts, _shp, _w, target_h, _ws, _hs = PageDewarper._sample_grid(
        ref_h, ref_w, params=_params(), page_dims_w=1.1,
        page_dims_h=page_h, decimate=4, zoom=zoom, focal=1.3)
    return target_h


def test_facing_pages_come_out_at_the_same_scale():
    """The regression: same input, same measured height, different fitted
    heights → same output height (within decimate rounding). Before #202
    these came out 2424 and 2932 px."""
    left = _height(2758, 1569, 1.9599, 1.7582)
    right = _height(2758, 1569, 1.9598, 2.1259)
    assert abs(left - right) <= 4


def test_output_height_is_the_measured_page_height():
    """The output is sized from the height measured in the image."""
    assert abs(_height(2758, 1569, 1.96, 1.758) - 0.5 * 1.96 * 2758) <= 4


def test_zoom_is_one_when_the_fit_fell_back_to_rough_dims():
    assert PageDewarper._scale_zoom((1.1, 1.96), (1.1, 1.96)) == 1.0


def test_zoom_ignores_degenerate_dims():
    assert PageDewarper._scale_zoom((1.1, 0.0), (1.1, 1.9)) == 1.0
    assert PageDewarper._scale_zoom((1.1, 1.9), (1.1, -1.0)) == 1.0


def test_zoom_is_clamped():
    assert PageDewarper._scale_zoom((1, 10.0), (1, 1.0)) == PageDewarper._ZOOM_MAX
    assert PageDewarper._scale_zoom((1, 0.1), (1, 1.0)) == PageDewarper._ZOOM_MIN


def test_replay_reads_the_stamped_zoom():
    """Replay sizes from the stamped zoom: an old stamp (1.0) keeps its
    size, a new stamp reproduces the forward size."""
    base = {"params": _params().tolist(), "src_shape": [2758, 1569],
            "pad_px": 0, "decimate": 4, "focal_length": 1.3,
            "page_dims": [1.1, 1.758], "model_dims": [1.1, 1.96],
            "sheet_model": "cylindrical"}
    old = PageDewarper._replay_sample_map((2758, 1569), {**base, "zoom": 1.0})
    zoom = PageDewarper._scale_zoom((1.1, 1.96), (1.1, 1.758))
    new = PageDewarper._replay_sample_map((2758, 1569), {**base, "zoom": zoom})
    assert old[0].shape[0] == _target_h_plain(1.758)
    assert abs(new[0].shape[0] - _height(2758, 1569, 1.96, 1.758)) == 0


def _target_h_plain(page_h):
    _pts, _shp, _w, target_h, _ws, _hs = PageDewarper._sample_grid(
        2758, 1569, params=_params(), page_dims_w=1.1, page_dims_h=page_h,
        decimate=4, zoom=1.0, focal=1.3)
    return target_h
