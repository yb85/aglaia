# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Clicking a handle grabs the handle you clicked.

The old rule walked the polygons in order and took the FIRST vertex inside the
grab radius. That is only the right answer when handles sit further apart than
that radius, and on a traced stamp outline they do not — a circle of twenty
vertices at working zoom has neighbours well inside it. So a click reliably
grabbed whichever came earlier in the list, which reads as a constant offset of
a few millimetres up and to the left.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint                                   # noqa: E402
from PySide6.QtGui import QPixmap                                   # noqa: E402
from PySide6.QtWidgets import QApplication                          # noqa: E402

from aglaia.gui.DebugEditCanvas import EditCanvas                  # noqa: E402

W = H = 400


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def canvas(app):
    c = EditCanvas()
    pix = QPixmap(W, H)
    pix.fill()
    c.set_image(pix)
    c.resize(W, H)
    return c


def _ring(cx, cy, r, n):
    import math
    return [[cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)] for i in range(n)]


def test_the_nearest_handle_wins(canvas):
    """A dense ring — the case that broke. Every vertex must be reachable."""
    ring = _ring(200, 200, 60, 20)
    canvas.set_layouts([ring], frame_wh=(W, H))
    for i, (x, y) in enumerate(ring):
        got = canvas._nearest_vertex(
            QPoint(int(round(canvas._to_view(x, y).x())),
                   int(round(canvas._to_view(x, y).y()))), [ring])
        assert got == (0, i), f"clicking vertex {i} selected {got}"


def test_a_click_between_two_handles_takes_the_closer(canvas):
    poly = [[100, 100], [130, 100], [130, 130]]
    canvas.set_layouts([poly], frame_wh=(W, H))
    a = canvas._to_view(100, 100)
    b = canvas._to_view(130, 100)
    # 40% of the way from a to b: still a's.
    p = QPoint(int(a.x() + 0.4 * (b.x() - a.x())), int(a.y()))
    assert canvas._nearest_vertex(p, [poly]) == (0, 0)
    p = QPoint(int(a.x() + 0.6 * (b.x() - a.x())), int(a.y()))
    assert canvas._nearest_vertex(p, [poly]) == (0, 1)


def test_a_click_in_open_space_grabs_nothing(canvas):
    poly = [[10, 10], [30, 10], [30, 30]]
    canvas.set_layouts([poly], frame_wh=(W, H))
    assert canvas._nearest_vertex(QPoint(W - 5, H - 5), [poly]) is None


def test_it_searches_every_polygon(canvas):
    a = [[20, 20], [40, 20], [40, 40]]
    b = [[300, 300], [320, 300], [320, 320]]
    canvas.set_layouts([a, b], frame_wh=(W, H))
    v = canvas._to_view(320, 300)
    assert canvas._nearest_vertex(QPoint(int(v.x()), int(v.y())),
                                  [a, b]) == (1, 1)


def test_the_grab_area_is_round_not_diamond(canvas):
    """`manhattanLength` made the reachable region a diamond, so a handle was
    a third harder to grab on the diagonal than straight on."""
    from aglaia.gui.DebugEditCanvas import GRAB_PX
    poly = [[200, 200], [260, 200], [260, 260]]
    canvas.set_layouts([poly], frame_wh=(W, H))
    c = canvas._to_view(200, 200)
    d = GRAB_PX * 2 * 0.6          # inside the circle, outside the diamond
    p = QPoint(int(c.x() + d * 0.707), int(c.y() + d * 0.707))
    assert canvas._nearest_vertex(p, [poly]) == (0, 0)
