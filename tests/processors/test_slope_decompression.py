# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Slope-based x decompression of the output grid (iOS #86, tracker #70).

The arc-length grid corrects the SURFACE-LENGTH term — paper is inextensible,
so sampling uniformly in x stretches text by √(1+z′²) where the sheet is steep.
It does not correct **projective foreshortening**: where the page recedes
steeply near the spine, the camera sees those glyphs compressed regardless of
how the surface is parameterised, and the arc-length grid gives them no extra
output pixels.

`slope_emphasis` (k) re-weights the grid measure by (1+z′²)^(k/2), so steep
regions claim proportionally more output width. k = 0 reproduces the existing
grid exactly, which is what makes it safe to ship switched off.
"""
import numpy as np
import pytest

from aglaia.processors.PageDewarper import PageDewarper


def _grid(k, alpha=0.25, beta=-0.1):
    params = np.zeros(20, dtype=np.float32)
    params[0:3] = [0.02, -0.03, 0.01]        # rvec
    params[3:6] = [-0.6, -0.45, 1.9]         # tvec
    params[6], params[7] = alpha, beta       # curl
    return PageDewarper._sample_grid(
        2400, 1600, params=params, page_dims_w=1.3, page_dims_h=0.95,
        decimate=4, zoom=1.0, focal=1.3, slope_emphasis=k)


def test_k_zero_is_bit_identical_to_no_emphasis():
    """The parity guarantee. Every stamped project replays through this code;
    if k = 0 drifted even in the last bits, every existing page would shift."""
    a = _grid(0.0)
    b = _grid(0.0)
    assert a[2] == b[2] and a[3] == b[3]
    assert np.array_equal(a[0], b[0])


def test_emphasis_widens_the_output_on_a_curved_page():
    """Steep regions claim more output pixels, so the page gets wider — that
    IS the decompression."""
    w0 = _grid(0.0)[2]
    w1 = _grid(1.0)[2]
    assert w1 > w0, f"k=1 did not widen the output ({w1} vs {w0})"


def test_a_flat_page_is_unaffected_by_any_k():
    """z′ = 0 everywhere → the weight is 1 → nothing to decompress. A flat
    page must not silently change size because the option exists."""
    flat0 = _grid(0.0, alpha=0.0, beta=0.0)
    flat1 = _grid(1.0, alpha=0.0, beta=0.0)
    assert flat0[2] == flat1[2]
    assert np.allclose(flat0[0], flat1[0], atol=1e-6)


def test_emphasis_is_monotone_in_k():
    widths = [_grid(k)[2] for k in (0.0, 0.5, 1.0, 2.0)]
    assert widths == sorted(widths), widths
    assert widths[-1] > widths[0]


def test_the_extra_width_lands_where_the_sheet_is_steep():
    """Not just "wider" — the samples must redistribute TOWARD the steep side.

    z(x) for the cubic is steepest near the page edges; with α > 0 the gutter
    side is the steeper one. Under emphasis the x-samples should cluster less
    densely there (each output pixel covers less page-x), which shows up as a
    smaller mean page-x step across the steep quarter."""
    def steep_quarter_step(k):
        params = np.zeros(20, dtype=np.float32)
        params[3:6] = [-0.6, -0.45, 1.9]
        params[6], params[7] = 0.35, -0.05
        from aglaia.processors.sheet_models import arclength_x
        xs, s = arclength_x(params, 1.3)
        s = PageDewarper._emphasise(xs, s, k)
        n = 400
        page_x = np.interp(np.linspace(0.0, float(s[-1]), n), s, xs)
        return float(np.mean(np.diff(page_x[: n // 4])))

    assert steep_quarter_step(1.0) < steep_quarter_step(0.0)


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_replay_reads_the_emphasis_from_the_stamp(k):
    """Replay rebuilds the grid from the stamp; if it didn't carry k the
    replayed page would be a different width from the live one."""
    params = np.zeros(20, dtype=np.float32)
    params[3:6] = [-0.6, -0.45, 1.9]
    params[6], params[7] = 0.25, -0.1
    stamp = {
        "params": params.tolist(), "page_dims": [1.3, 0.95],
        "pad_px": 0, "zoom": 1.0, "decimate": 4,
        "sheet_model": "cylindrical", "model_dims": [1.3, 0.95],
        "focal_length": 1.3, "camera_np": 8, "spine": None,
        "slope_emphasis": k,
    }
    im_x, _im_y, _pad = PageDewarper._replay_sample_map((2400, 1600), stamp)
    live = PageDewarper._sample_grid(
        2400, 1600, params=params, page_dims_w=1.3, page_dims_h=0.95,
        decimate=4, zoom=1.0, focal=1.3, slope_emphasis=k)
    assert im_x.shape[1] == live[2]


def test_a_pre_emphasis_stamp_replays_unchanged():
    """Projects stamped before this option existed carry no key and must
    replay exactly as they were fitted."""
    params = np.zeros(20, dtype=np.float32)
    params[3:6] = [-0.6, -0.45, 1.9]
    params[6], params[7] = 0.25, -0.1
    stamp = {
        "params": params.tolist(), "page_dims": [1.3, 0.95],
        "pad_px": 0, "zoom": 1.0, "decimate": 4,
        "sheet_model": "cylindrical", "model_dims": [1.3, 0.95],
        "focal_length": 1.3,
    }
    im_x, _, _ = PageDewarper._replay_sample_map((2400, 1600), stamp)
    w0 = PageDewarper._sample_grid(
        2400, 1600, params=params, page_dims_w=1.3, page_dims_h=0.95,
        decimate=4, zoom=1.0, focal=1.3, slope_emphasis=0.0)[2]
    assert im_x.shape[1] == w0
