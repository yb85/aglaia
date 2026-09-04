# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A hand-drawn layout set replaces detection (#118).

Three things the old per-branch `roi` override could not do, all of which
reduce to the same missing idea — that the SET of layouts is itself editable,
and belongs to the scan rather than to any one branch:

* a polygon could not leave the crop the detector had chosen, because the
  frame it was expressed in WAS that crop;
* a layout could not be deleted, because detection would find it again;
* a layout could not be added, because branches came only from detection.
"""
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.PageDetector import (PageDetector, PageOption,
                                            _manual_layouts_for,
                                            _poly_bbox_int)


class _Buf:
    def __init__(self, meta):
        self.meta = meta


def _payload(layouts, frame=None):
    p = {"layouts": layouts}
    if frame:
        p["layouts_frame_wh"] = list(frame)
    return _Buf({"manual_overrides_all": {"": p}})


SQUARE = [[10, 10], [90, 10], [90, 90], [10, 90]]


# ── the trunk payload ────────────────────────────────────────────────

def test_the_set_is_read_from_the_trunk_not_a_branch():
    """It decides how many branches exist, so it cannot live on one of them."""
    on_trunk = _Buf({"manual_overrides_all": {"": {"layouts": [SQUARE]}}})
    on_branch = _Buf({"manual_overrides_all": {"A": {"layouts": [SQUARE]}}})
    assert _manual_layouts_for(on_trunk, frame_wh=(100, 100)) is not None
    assert _manual_layouts_for(on_branch, frame_wh=(100, 100)) is None


def test_no_override_reads_as_no_set():
    assert _manual_layouts_for(_Buf({}), frame_wh=(100, 100)) is None
    assert _manual_layouts_for(_Buf({"manual_overrides_all": {}}),
                               frame_wh=(100, 100)) is None


def test_a_set_drawn_on_another_frame_is_refused():
    """Same rule as every other spatial override: a polygon applied to an
    image of another size would be silently shifted."""
    buf = _payload([SQUARE], frame=(100, 100))
    assert _manual_layouts_for(buf, frame_wh=(100, 100)) is not None
    assert _manual_layouts_for(buf, frame_wh=(640, 480)) is None


def test_a_set_with_no_frame_stamp_is_accepted():
    """Unvalidatable, not wrong — matches `validate_frame`."""
    assert _manual_layouts_for(_payload([SQUARE]),
                               frame_wh=(640, 480)) is not None


def test_degenerate_polygons_are_dropped_and_an_empty_set_is_none():
    buf = _payload([SQUARE, [[1, 1], [2, 2]]])       # a segment
    assert len(_manual_layouts_for(buf, frame_wh=(100, 100))) == 1
    assert _manual_layouts_for(_payload([[[1, 1], [2, 2]]]),
                               frame_wh=(100, 100)) is None


# ── the bbox the crop is derived from ────────────────────────────────

def test_the_bbox_is_clamped_into_the_image():
    """A vertex dragged past the edge must not produce a crop outside it."""
    poly = [[-50, -50], [500, -20], [500, 500], [-50, 500]]
    assert _poly_bbox_int(poly, 100, 80) == (0, 0, 100, 80)


def test_a_degenerate_bbox_is_none_not_a_zero_size_crop():
    assert _poly_bbox_int([[5, 5], [5.5, 5.5], [5, 5.5]], 100, 100) is None


def test_the_bbox_rounds_outwards():
    """Floor the minimum, ceil the maximum — a crop must never cut the
    polygon it was derived from."""
    assert _poly_bbox_int([[10.7, 10.2], [50.1, 10.2], [50.1, 49.9]],
                          100, 100) == (10, 10, 51, 50)


# ── end to end through the processor ─────────────────────────────────

class _FakeDetector:
    """Two text columns side by side — the spread the detector would split."""

    uses_gpu = False

    def detect(self, _img):
        boxes = []
        for x0 in (40, 230):
            for row in range(60, 240, 18):
                boxes.append((x0, row, x0 + 130, row + 8))
        return boxes


def _page(w=400, h=300):
    img = np.full((h, w, 3), 245, np.uint8)
    for x0 in (40, 230):
        for row in range(60, 240, 18):
            img[row:row + 8, x0:x0 + 130] = 20
    return img


def _detect(manual=None, frame=None, **opts):
    """Returns (children, proc). `process` hands back the child list."""
    buf = ImageBuffer(_page(), ImageType.COLOR, dpi=300.0)
    buf.filestem = "page"
    if manual is not None:
        payload = {"layouts": manual}
        if frame:
            payload["layouts_frame_wh"] = list(frame)
        buf.meta["manual_overrides_all"] = {"": payload}
    proc = PageDetector(PageOption(**opts))
    proc.detector = _FakeDetector()
    out = proc.process(buf)
    kids = out if isinstance(out, list) else [out]
    return kids, buf


def test_the_set_decides_how_many_children_there_are():
    """One polygon over a spread the detector splits in two."""
    whole = [[20, 40], [380, 40], [380, 260], [20, 260]]
    kids, _ = _detect(manual=[whole], frame=(400, 300))
    assert len(kids) == 1


def test_three_polygons_make_three_children_past_the_max_pages_cap():
    """`max_pages` is a guard on DETECTION. A set the user drew is not a
    guess to be capped."""
    polys = [[[10, 10], [120, 10], [120, 280], [10, 280]],
             [[140, 10], [250, 10], [250, 280], [140, 280]],
             [[270, 10], [390, 10], [390, 280], [270, 280]]]
    kids, _ = _detect(manual=polys, frame=(400, 300), max_pages=2)
    assert len(kids) == 3


def test_the_child_roi_is_the_polygon_the_user_drew():
    """Not the text-tight bbox re-derived from the boxes — the user has
    already said where the page is."""
    poly = [[30.0, 50.0], [300.0, 40.0], [310.0, 250.0], [25.0, 260.0]]
    kids, _ = _detect(manual=[poly], frame=(400, 300))
    kid = kids[0]
    crop = kid.meta["parent_crop_xywh"]
    roi_parent = [[x + crop[0], y + crop[1]] for x, y in kid.meta["roi"]]
    for got, want in zip(roi_parent, poly):
        assert got == pytest.approx(want, abs=1.5)


def test_a_polygon_wider_than_the_detected_page_still_governs_the_crop():
    """The whole point of the parent frame: the ROI can grow OUTWARDS. The
    crop is derived from the polygon, so a bigger polygon means a bigger
    crop — the old clamp made this impossible."""
    small = [[100, 100], [200, 100], [200, 200], [100, 200]]
    big = [[10, 10], [390, 10], [390, 290], [10, 290]]
    areas = []
    for poly in (small, big):
        kids, _ = _detect(manual=[poly], frame=(400, 300))
        _x, _y, cw, ch = kids[0].meta["parent_crop_xywh"]
        areas.append(cw * ch)
    assert areas[1] > areas[0] * 3


def test_a_stale_frame_falls_back_to_detection():
    kids, _ = _detect(manual=[[[10, 10], [390, 10], [390, 290], [10, 290]]],
                      frame=(999, 999))
    assert len(kids) == 2                       # detected, not manual


def test_the_edited_layouts_are_marked_as_hand_edited():
    """The scan views mark a hand-edited page; a layout set is one."""
    kids, _ = _detect(manual=[[[20, 40], [380, 40], [380, 260], [20, 260]]],
                      frame=(400, 300))
    # Named for the instrument: only "layouts" explains a changed page count.
    assert kids[0].meta.get("manual") == ["layouts"]


def test_a_detected_layout_is_not_marked_as_hand_edited():
    kids, _ = _detect()
    assert not kids[0].meta.get("manual")
