# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Convex-hull page ROI (`PageDetector._hull_roi`).

The ROI is what the Binarizer KEEPS; everything outside it is erased. On a
slanted capture the axis-aligned bbox of the text swallows the corners — where
the fingers holding the book sit — so the hull of the text boxes is used
instead. These tests pin the properties the downstream chain relies on:
tighter than the bbox, never wider, correctly padded, correctly wound, and
expressed in the child's cropped coordinates."""

from __future__ import annotations

import cv2
import numpy as np

from aglaia.processors.PageDetector import PageDetector


def _bbox(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _slanted(n=10, drift=8):
    """A text block photographed at an angle: each line shifted right."""
    return [(100 + drift * i, 100 + 40 * i, 400 + drift * i, 130 + 40 * i)
            for i in range(n)]


def _square_on(n=10):
    return [(100, 100 + 40 * i, 400, 130 + 40 * i) for i in range(n)]


def _hull(boxes, pad=10, crop=None, rect=None, clamp_to=None):
    rect = rect or _bbox(boxes)
    if crop is None:
        crop = (rect[0] - 100, rect[1] - 100, rect[2] + 100, rect[3] + 100)
    if clamp_to is None:
        # Mirror `process` exactly: the rect ROI is the tight bounds + pad
        # clamped to the image, then expressed in child coords and clamped
        # to the crop.
        roi = (max(0, rect[0] - pad), max(0, rect[1] - pad),
               rect[2] + pad, rect[3] + pad)
        clamp_to = (max(0, roi[0] - crop[0]),
                    max(0, roi[1] - crop[1]),
                    min(crop[2] - crop[0], roi[2] - crop[0]),
                    min(crop[3] - crop[1], roi[3] - crop[1]))
    return PageDetector._hull_roi(boxes, rect=rect, pad=pad, crop=crop,
                                  clamp_to=clamp_to)


def test_too_few_boxes_falls_back_to_rect():
    # Caller keeps its bbox ROI; a single box's padded hull IS that bbox.
    assert PageDetector._hull_roi([], rect=(0, 0, 500, 500), pad=10,
                                  crop=(0, 0, 500, 500),
                                  clamp_to=(0, 0, 500, 500)) is None
    # Under 3 boxes the hull is a segment or one padded box — no better than
    # the rect, so the caller keeps the rect.
    assert _hull(_slanted(1)) is None
    assert _hull(_slanted(2)) is None


def test_slanted_block_is_tighter_than_bbox():
    boxes = _slanted()
    hull = np.array(_hull(boxes), dtype=np.float32)
    x1, y1, x2, y2 = _bbox(boxes)
    padded_bbox_area = (x2 - x1 + 20) * (y2 - y1 + 20)
    assert cv2.contourArea(hull) < 0.9 * padded_bbox_area


def test_square_on_block_matches_bbox():
    # No slant → the hull IS the padded rect, so nothing is lost on a
    # square-on capture by leaving roi_hull on.
    hull = np.array(_hull(_square_on()), dtype=np.float32)
    x1, y1, x2, y2 = _bbox(_square_on())
    assert cv2.contourArea(hull) == (x2 - x1 + 20) * (y2 - y1 + 20)


def test_never_wider_than_padded_bbox():
    boxes = _slanted()
    x1, y1, x2, y2 = _bbox(boxes)
    for x, y in _hull(boxes):
        assert x1 - 10 <= x <= x2 + 10
        assert y1 - 10 <= y <= y2 + 10


def test_pad_reaches_every_corner():
    # Exact dilation: every box corner, pushed out by `pad` on both axes,
    # must lie inside (or on) the hull — a centroid-outward nudge would
    # under-pad the vertices between hull corners.
    boxes = _slanted()
    hull = np.array(_hull(boxes, pad=10), dtype=np.float32)
    for bx1, by1, bx2, by2 in boxes:
        for pt in ((bx1 - 10, by1 - 10), (bx2 + 10, by1 - 10),
                   (bx2 + 10, by2 + 10), (bx1 - 10, by2 + 10)):
            assert cv2.pointPolygonTest(hull, pt, True) >= -1e-3


def test_child_coords_and_clamped_to_crop():
    boxes = _slanted()
    rect = _bbox(boxes)
    # Crop tighter than rect+pad on every side → the hull must clamp to it
    # and be expressed relative to its top-left, since it gets rasterised
    # on the cropped child buffer.
    crop = (rect[0], rect[1], rect[2], rect[3])
    hull = _hull(boxes, pad=10, crop=crop)
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    for x, y in hull:
        assert 0 <= x <= w
        assert 0 <= y <= h


def test_winding_matches_rect_path():
    # The rect ROI is emitted TL,TR,BR,BL; `cv2.intersectConvexConvex`
    # against the parent ROI is fed both, so the hull must wind the same way.
    boxes = _slanted()
    hull = np.array(_hull(boxes), dtype=np.float32)
    x1, y1, x2, y2 = _bbox(boxes)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32)
    assert np.sign(cv2.contourArea(hull, True)) == \
        np.sign(cv2.contourArea(rect, True))


def test_outlier_box_outside_rect_is_ignored():
    # A finger / cable box the caller's gap test already trimmed out of the
    # tight rect must not drag the hull back over it.
    boxes = _slanted()
    rect = _bbox(boxes)
    finger = (rect[2] + 300, rect[1] + 100, rect[2] + 500, rect[1] + 400)
    crop = (rect[0] - 100, rect[1] - 100, rect[2] + 600, rect[3] + 100)
    hull = np.array(_hull(boxes + [finger], rect=rect, crop=crop),
                    dtype=np.float32)
    cx = (finger[0] + finger[2]) / 2 - crop[0]
    cy = (finger[1] + finger[3]) / 2 - crop[1]
    assert cv2.pointPolygonTest(hull, (cx, cy), False) < 0


def test_box_straddling_the_tight_bound_is_clamped_in():
    # A box only PARTLY trimmed (centre inside the tight bounds, edge past
    # them) is clamped to the bounds before dilation, so the hull stops at
    # the bound rather than bulging out to the box's far edge.
    boxes = _square_on()
    rect = _bbox(boxes)
    straddler = (rect[2] - 20, rect[1] + 100, rect[2] + 120, rect[1] + 140)
    hull = np.array(_hull(boxes + [straddler], rect=rect), dtype=np.float32)
    crop = (rect[0] - 100, rect[1] - 100, rect[2] + 100, rect[3] + 100)
    # Right edge of the bulge, in child coords, must not pass rect[2] + pad.
    assert max(x for x, _ in hull.tolist()) <= (rect[2] + 10) - crop[0] + 1e-6
