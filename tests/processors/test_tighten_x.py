# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Horizontal tightening of a detected page (`PageDetector.tighten_x`).

The gap test drops a box sitting a large gap past the dense text cluster — a
cable, a hand, the edge of a cup. It cannot tell such an intruder from a
legitimate element that simply stands alone, and it was silently eating text:
a lone date in a chronology column, and the two longest lines of a short
ragged block (both observed on `delbrel-oc9`, #86).

The discriminator is the SIDE. An intruder comes from outside the book, never
from the gutter, so on a two-page spread the spine side is bounded by the
crease instead of by the gap test.
"""
from aglaia.processors.PageDetector import tighten_x


def _body(x0, x1, n=6, y=100, step=40):
    """A dense block of `n` full-width lines."""
    return [(x0, y + i * step, x1, y + i * step + 24) for i in range(n)]


def test_intruder_past_a_large_gap_is_dropped():
    """The behaviour the gap test exists for, on a single page."""
    boxes = _body(100, 500) + [(700, 140, 760, 220)]   # a thumb
    assert tighten_x(boxes, rect_x=(100, 760)) == (100, 500)


def test_lone_date_on_the_spine_side_survives():
    """#86. A right page whose only left-hand element is a date: the gap test
    trimmed it, the crease bound keeps it."""
    boxes = _body(400, 900) + [(250, 180, 310, 204)]   # "1996"
    assert tighten_x(boxes, rect_x=(250, 900)) == (400, 900)
    assert tighten_x(boxes, rect_x=(250, 900),
                     spine_side="left", gutter_x=200) == (250, 900)


def test_long_lines_of_a_ragged_block_survive():
    """#86, second case: a short block whose two longest lines are each a
    >10%-width step out. The gap test cut them mid-word."""
    boxes = [(100, 100, 500, 124), (100, 140, 480, 164),
             (100, 180, 470, 204), (100, 220, 610, 244),
             (100, 260, 718, 284)]
    assert tighten_x(boxes, rect_x=(100, 718))[1] == 500
    assert tighten_x(boxes, rect_x=(100, 718),
                     spine_side="right", gutter_x=800)[1] == 718


def test_the_crease_bound_can_only_be_the_gutter():
    """The clamp never widens a page past its own rect, and never lets it
    reach across the gutter into the facing page."""
    boxes = _body(400, 900)
    # rect stops short of the gutter -> the rect wins
    assert tighten_x(boxes, rect_x=(400, 900),
                     spine_side="right", gutter_x=950)[1] == 900
    # rect straddles the gutter -> the gutter wins
    assert tighten_x(boxes, rect_x=(400, 900),
                     spine_side="right", gutter_x=850)[1] == 850


def test_outer_side_keeps_the_gap_test_on_a_spread():
    """Only the spine side stands down: a hand on the outer margin of the
    same page is still trimmed."""
    boxes = _body(400, 900) + [(1100, 140, 1180, 320)]
    got = tighten_x(boxes, rect_x=(400, 1180),
                    spine_side="left", gutter_x=350)
    assert got == (400, 900)


def test_too_few_boxes_leaves_the_rect_alone():
    boxes = [(100, 100, 500, 124), (100, 140, 480, 164), (700, 100, 760, 124)]
    assert tighten_x(boxes, rect_x=(100, 760)) == (100, 760)
