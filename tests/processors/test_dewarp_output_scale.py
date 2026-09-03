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
