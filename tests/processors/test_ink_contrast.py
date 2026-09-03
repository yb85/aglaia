# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Page-level contrast used by `min_contrast` (`PageDetector.ink_contrast`).

The filter exists to delete a bleed-through ghost: the mirror of the other
side of the leaf, pale wherever its ink is. It measured `p95 - p5` over the
page's whole BBOX, which made it a density proxy instead — a sparse page is
mostly paper, so p5 lands on paper too and the range collapses however black
the ink. It deleted a title page, two chapter openings and two chronology
date columns on `delbrel-oc9`.

Measuring the same range under the page's DETECTION BOXES separates the two:
the ghost stays low, the sparse page reads like any other.
"""
import numpy as np

from aglaia.processors.PageDetector import ink_contrast


def _canvas(w=800, h=1000, paper=235):
    return np.full((h, w), paper, dtype=np.uint8)


def _write(img, boxes, ink):
    """Paint glyph-like strokes: a real text box holds ink AND the paper
    between the letters, which is the spread the measure reads."""
    for x1, y1, x2, y2 in boxes:
        for x in range(x1, x2, 6):
            img[y1:y2, x:x + 3] = ink


def _lines(x0, y0, n, *, w=300, h=20, step=60):
    return [(x0, y0 + i * step, x0 + w, y0 + i * step + h) for i in range(n)]


def test_a_sparse_page_of_black_ink_is_not_a_ghost():
    """The regression. Three lines of black ink on the left, fourteen on the
    right: same ink, and the sparse page must not be deleted."""
    img = _canvas()
    dense = _lines(420, 60, 14)
    sparse = _lines(60, 380, 3)
    _write(img, dense + sparse, 20)
    pages = [(40, 360, 380, 560), (400, 40, 760, 940)]
    rels = ink_contrast(img, pages, dense + sparse)
    assert min(rels) > 0.9


def test_a_pale_ghost_is_still_caught():
    """What the filter is for: the same layout, but the left page's ink is
    the pale mirror of the facing leaf."""
    img = _canvas()
    dense = _lines(420, 60, 14)
    ghost = _lines(60, 380, 3)
    _write(img, dense, 20)
    _write(img, ghost, 215)
    pages = [(40, 360, 380, 560), (400, 40, 760, 940)]
    rels = ink_contrast(img, pages, dense + ghost)
    assert rels[0] < 0.2
    assert rels[1] == 1.0


def test_the_bbox_measure_would_have_deleted_the_sparse_page():
    """Pins WHY the measure moved. Over the bbox the sparse page scores far
    under the 0.5 default; under the boxes it does not."""
    img = _canvas()
    dense = _lines(420, 60, 14)
    sparse = _lines(60, 380, 1, w=120)
    _write(img, dense + sparse, 20)
    pages = [(40, 360, 380, 900), (400, 40, 760, 940)]

    def bbox_rels():
        rng = []
        for (x1, y1, x2, y2) in pages:
            p5, p95 = np.percentile(img[y1:y2, x1:x2], (5, 95))
            rng.append(float(p95 - p5))
        top = max(rng)
        return [r / top for r in rng]

    assert bbox_rels()[0] < 0.5
    assert ink_contrast(img, pages, dense + sparse)[0] > 0.9


def test_a_page_with_no_box_scores_zero():
    img = _canvas()
    dense = _lines(420, 60, 14)
    _write(img, dense, 20)
    pages = [(40, 40, 380, 940), (400, 40, 760, 940)]
    assert ink_contrast(img, pages, dense) == [0.0, 1.0]
