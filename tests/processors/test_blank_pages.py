# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A spread page with no text is kept on demand (#196).

The detector sees only text, so a blank verso has no boxes and the spread
comes out one page short. `blank_page` puts a white page back into the
largest empty area of the frame.
"""
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.PageDetector import PageDetector, PageOption, blank_slots

W, H = 2000, 1400


def _right_page_boxes():
    """Text lines of a right-hand page only: x 1100-1800."""
    return [(1100, y, 1800, y + 30) for y in range(200, 1200, 60)]


class _Stub:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img):
        return list(self.boxes)


def _run(mode, boxes, max_pages=2):
    det = PageDetector(PageOption(backend="heuristic", max_pages=max_pages,
                                  blank_page=mode, processing_dpi=None,
                                  roi_hull=False))
    det.detector = _Stub(boxes)
    buf = ImageBuffer(np.full((H, W, 3), 255, np.uint8), ImageType.COLOR,
                      dpi=200.0, filestem="s")
    return det.process(buf)


# ── the heuristic ───────────────────────────────────────────────────

def test_the_blank_page_goes_into_the_largest_empty_area():
    (x1, _, x2, _), = blank_slots([(1100, 200, 1800, 1200)],
                                  frame_w=W, target=2)
    assert x2 <= 1100 and x2 - x1 == 700          # left of the page


def test_a_centred_single_page_gets_no_blank():
    """A cover in the middle of the frame: both gaps are too narrow."""
    assert blank_slots([(500, 100, 1500, 1300)], frame_w=W, target=2) == []


def test_three_pages_fill_the_widest_gap_first():
    pages = [(0, 0, 600, 900), (1400, 0, 2000, 900)]
    (x1, _, x2, _), = blank_slots(pages, frame_w=W, target=3)
    assert 600 <= x1 and x2 <= 1400                # the middle gap


def test_no_blank_when_the_count_is_reached():
    pages = [(0, 0, 900, 900), (1100, 0, 2000, 900)]
    assert blank_slots(pages, frame_w=W, target=2) == []


# ── the modes ───────────────────────────────────────────────────────

def test_delete_is_the_default_and_keeps_one_page():
    assert PageOption().blank_page == "delete"
    assert len(_run("delete", _right_page_boxes())) == 1


def test_match_adds_a_white_left_page_the_size_of_the_other():
    kids = _run("match", _right_page_boxes())
    assert [k.branch_label for k in kids] == ["A", "B"]
    blank, page = kids
    assert blank.meta["blank_page"] is True
    assert "blank_page" not in page.meta
    assert (blank.meta["page_side"], page.meta["page_side"]) == ("left", "right")
    assert (blank.buffer == 255).all()
    assert blank.buffer.shape[1] == 700           # the text width of the other


def test_1px_adds_a_one_pixel_placeholder():
    blank, _ = _run("1px", _right_page_boxes())
    assert blank.buffer.shape[:2] == (1, 1)
    assert blank.meta["blank_page"] is True


@pytest.mark.parametrize("saved,cap", [(0, 3), (8, 3), (1, 1), (3, 3)])
def test_max_pages_is_capped_at_three(saved, cap):
    det = PageDetector(PageOption(backend="heuristic", max_pages=saved))
    assert det.max_pages == cap
