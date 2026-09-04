# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Editable geometry beside the rasterised diagnostics (M9 #96).

The composite a renderer returns is a picture: right for span masks, baselines
and the fitted grid, useless for anything the user must grab. `geom` carries
the same values as numbers, in the coordinates of the node's own stage frame,
so the debug view can paint draggable handles over them.

Only the three tunable stages carry it. `frame_wh` travels with every edit
made on that frame, so a polygon can be validated against the frame it was
drawn on rather than silently rescaled.
"""
import numpy as np

from aglaia.storage.debug_renderers import (_dewarp_renderer, _page_renderer,
                                            _skew_renderer, _default_renderer)


def _img(w=120, h=200):
    a = np.full((h, w, 3), 240, np.uint8)
    a[40:160, 20:100] = 30
    return a


def test_skew_geom_is_the_angle_in_the_node_frame():
    geom = _skew_renderer(_img(), None, {"skew": -1.75})[0]["geom"]
    assert geom["frame_wh"] == [120, 200]
    assert geom["skew_deg"] == -1.75
    # The composite carries a label bar above the image; a handle layer that
    # ignored the offset would sit a bar-height high.
    assert geom["origin"][0] == 0 and geom["origin"][1] > 0


def test_skew_geom_omits_an_angle_it_does_not_have():
    """A missing value must be absent, not zero — zero is a measurement."""
    assert "skew_deg" not in _skew_renderer(_img(), None, {})[0]["geom"]


def test_page_geom_is_the_roi_in_the_CHILD_frame():
    """The parent composite shows every layout at once, which is the right
    picture to look at and the wrong one to edit on: the polygon the pipeline
    consumes is in child coordinates."""
    roi = [[5, 5], [100, 5], [100, 180], [5, 180]]
    child = _img()
    meta = {"roi": roi, "parent_crop_xywh": [10, 20, 120, 200]}
    on_child = _page_renderer(child, None, meta)[0]["geom"]
    on_parent = _page_renderer(child, _img(400, 400), meta)[0]["geom"]
    # Same polygon and same frame either way — only WHERE that frame sits in
    # the composite differs, which is what `origin` is for.
    assert on_child["frame_wh"] == on_parent["frame_wh"] == [120, 200]
    assert on_child["roi"] == on_parent["roi"] == [
        [5.0, 5.0], [100.0, 5.0], [100.0, 180.0], [5.0, 180.0]]
    assert on_child["origin"][0] == 0
    assert on_parent["origin"][0] == 10          # the child's crop offset


def test_dewarp_geom_is_the_curl_of_the_fit():
    params = [0.0] * 10
    params[6], params[7] = 0.21, -0.07
    meta = {"replay_params": {
        "params": params, "page_dims": [1.0, 1.5], "src_shape": [320, 240],
        "pad_px": 60, "focal_length": 1.3,
        "spine": {"gamma": 0.05, "s_x": 0.16, "x0": 1.0}}}
    parent = _img(200, 200)
    geom = _dewarp_renderer(_img(), parent, meta)[0]["geom"]
    assert geom["curl"] == {"alpha": 0.21, "beta": -0.07, "gamma": 0.05}
    # The grid is drawn on the step's INPUT, so that is the frame to edit on.
    assert geom["frame_wh"] == [200, 200]


def test_dewarp_geom_reads_gamma_as_zero_without_a_spine():
    params = [0.0] * 10
    params[6], params[7] = 0.1, 0.0
    meta = {"replay_params": {
        "params": params, "page_dims": [1.0, 1.5], "src_shape": [320, 240],
        "pad_px": 60, "focal_length": 1.3, "spine": None}}
    geom = _dewarp_renderer(_img(), _img(200, 200), meta)[0]["geom"]
    assert geom["curl"]["gamma"] == 0.0


def test_a_stage_with_nothing_to_edit_carries_no_geom():
    assert "geom" not in _default_renderer(_img(), None, {})[0]
