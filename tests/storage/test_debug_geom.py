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


def test_page_geom_without_a_parent_is_the_roi_in_the_child_frame():
    """Nothing to draw the set on — edit this one polygon, in its own frame."""
    roi = [[5, 5], [100, 5], [100, 180], [5, 180]]
    meta = {"roi": roi, "parent_crop_xywh": [10, 20, 120, 200]}
    geom = _page_renderer(_img(), None, meta)[0]["geom"]
    assert geom["frame_wh"] == [120, 200]
    assert geom["roi"] == [[5.0, 5.0], [100.0, 5.0],
                           [100.0, 180.0], [5.0, 180.0]]
    assert geom["origin"][0] == 0


def test_page_geom_on_a_parent_is_the_layout_SET_in_parent_coords():
    """With a parent to draw on, the handles work on every layout at once, in
    PARENT coordinates (#118).

    They used to live in ONE child's frame, which is why a vertex could not be
    dragged outside that child's crop — the clamp WAS the crop, so the page
    could only be corrected inwards, never outwards."""
    roi = [[5, 5], [100, 5], [100, 180], [5, 180]]
    meta = {"roi": roi, "parent_crop_xywh": [10, 20, 120, 200]}
    geom = _page_renderer(_img(), _img(400, 400), meta)[0]["geom"]
    assert geom["frame_wh"] == [400, 400]        # the PARENT
    # Each point carries its child's crop offset — (10, 20) here.
    assert geom["layouts"] == [[[15.0, 25.0], [110.0, 25.0],
                                [110.0, 200.0], [15.0, 200.0]]]
    assert geom["origin"][0] == 0                # only the label bar remains
    assert geom["origin"][1] > 0


def test_a_layout_without_a_stamped_roi_falls_back_to_its_crop():
    """An older node carries no ROI. Give the crop rect instead — a layout
    with no handles at all could not be deleted or corrected."""
    meta = {"parent_crop_xywh": [10, 20, 120, 200]}
    geom = _page_renderer(_img(), _img(400, 400), meta)[0]["geom"]
    assert geom["layouts"] == [[[10.0, 20.0], [130.0, 20.0],
                                [130.0, 220.0], [10.0, 220.0]]]


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


def test_geom_reports_the_downscale_the_url_applies():
    """The picture a viewer decodes is NOT the composite: `_png_data_url`
    shrinks it under Qt's allocation cap. `geom` is in full-resolution
    coordinates, so a consumer that ignored `scale` would see every handle
    drift further from its pixel the further it sat from the origin — the
    "everything is shifted down-right" report."""
    small = _skew_renderer(_img(120, 200), None, {"skew": 0.0})[0]["geom"]
    assert small["scale"] == 1.0
    big = _skew_renderer(_img(4000, 2000), None, {"skew": 0.0})[0]["geom"]
    assert 0.0 < big["scale"] < 1.0


def test_trap_geom_is_the_column_quad():
    from aglaia.storage.debug_renderers import _trap_renderer
    quad = [[10, 10], [180, 12], [178, 290], [12, 288]]
    meta = {"column_quad": quad,
            "replay_params": {"H": np.eye(3).tolist(),
                              "canvas_wh": [200, 300], "src_wh": [200, 300]}}
    geom = _trap_renderer(_img(200, 300), _img(200, 300), meta)[0]["geom"]
    assert geom["quad"] == [[float(x), float(y)] for x, y in quad]
    assert geom["frame_wh"] == [200, 300]


def test_trap_geom_seeds_a_quad_when_the_step_fell_back():
    """A page with no quad is exactly the page a user wants to draw one on.
    Handing back nothing would leave nothing to grab."""
    from aglaia.storage.debug_renderers import _trap_renderer
    geom = _trap_renderer(_img(200, 300), _img(200, 300),
                          {"trapezoid_success": False})[0]["geom"]
    assert len(geom["quad"]) == 4
    xs = [p[0] for p in geom["quad"]]
    assert min(xs) > 0 and max(xs) < 200        # inset, grabbable
