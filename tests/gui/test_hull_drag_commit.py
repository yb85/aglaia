# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""One drag is one commit (#116).

`edited` fires per mouse-move so the handle tracks the cursor. The host used
to persist AND rerun the page on it, so a single vertex drag launched a chain
rerun per move event — hundreds stacked up, memory filled, the app died.
`edit_finished` is the commit signal; these tests pin the ratio.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt                   # noqa: E402
from PySide6.QtGui import QMouseEvent, QPixmap                   # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.DebugEditCanvas import EditCanvas                # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def canvas(app):
    c = EditCanvas()
    c.resize(400, 300)
    pix = QPixmap(200, 150)
    pix.fill()
    c.set_image(pix)
    c.set_editable(polygon=[[20, 20], [180, 20], [180, 130], [20, 130]],
                   origin=(0, 0), frame_wh=(200, 150))
    return c


def _ev(kind, pos):
    pt = QPointF(float(pos[0]), float(pos[1]))
    return QMouseEvent(kind, pt, pt,
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


class _Counter:
    def __init__(self, c):
        self.steps, self.commits = 0, 0
        c.edited.connect(lambda *_: setattr(self, "steps", self.steps + 1))
        c.edit_finished.connect(lambda: setattr(self, "commits",
                                                self.commits + 1))


def _drag(canvas, frm, to, n=40):
    canvas.mousePressEvent(_ev(QEvent.Type.MouseButtonPress, frm))
    for i in range(1, n + 1):
        t = i / n
        canvas.mouseMoveEvent(_ev(
            QEvent.Type.MouseMove,
            (frm[0] + (to[0] - frm[0]) * t, frm[1] + (to[1] - frm[1]) * t)))
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease, to))


def test_a_long_drag_commits_exactly_once(canvas):
    start = canvas._to_view(20, 20).toPoint()
    end = canvas._to_view(60, 70).toPoint()
    c = _Counter(canvas)
    _drag(canvas, (start.x(), start.y()), (end.x(), end.y()), n=40)
    assert c.steps > 10, "the handle must still track the cursor"
    assert c.commits == 1


def test_the_canvas_reports_editing_only_during_the_drag(canvas):
    start = canvas._to_view(20, 20).toPoint()
    canvas.mousePressEvent(_ev(QEvent.Type.MouseButtonPress,
                               (start.x(), start.y())))
    canvas.mouseMoveEvent(_ev(QEvent.Type.MouseMove,
                              (start.x() + 20, start.y() + 20)))
    assert canvas.is_editing() is True
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease,
                                 (start.x() + 20, start.y() + 20)))
    assert canvas.is_editing() is False


def test_commit_lands_after_the_drag_flag_is_cleared(canvas):
    """The host tells a step from a commit with `is_editing()`, so the
    release must clear the flag BEFORE it emits."""
    seen = []
    canvas.edit_finished.connect(lambda: seen.append(canvas.is_editing()))
    start = canvas._to_view(20, 20).toPoint()
    _drag(canvas, (start.x(), start.y()), (start.x() + 30, start.y() + 30), n=5)
    assert seen == [False]


def test_an_inserted_vertex_commits_immediately(canvas):
    """A double-click insert is atomic — there is no drag to wait for."""
    a = canvas._to_view(20, 20)
    b = canvas._to_view(180, 20)
    mid = ((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
    c = _Counter(canvas)
    before = len(canvas.polygon())
    canvas.mouseDoubleClickEvent(_ev(QEvent.Type.MouseButtonDblClick, mid))
    assert len(canvas.polygon()) == before + 1
    assert c.commits == 1


def test_a_click_that_never_moves_commits_nothing_to_write(canvas):
    """Press + release on a handle: the commit fires, but with no drag step
    behind it the host has nothing pending, so nothing is written."""
    start = canvas._to_view(20, 20).toPoint()
    c = _Counter(canvas)
    canvas.mousePressEvent(_ev(QEvent.Type.MouseButtonPress,
                               (start.x(), start.y())))
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease,
                                 (start.x(), start.y())))
    assert c.steps == 0
    assert c.commits == 1


def test_the_rotation_handle_debounces_too(app):
    """`skew_deg` was emitted per move on the same path."""
    c = EditCanvas()
    c.resize(400, 300)
    pix = QPixmap(200, 150)
    pix.fill()
    c.set_image(pix)
    c.set_editable(rotation_deg=0.0, origin=(0, 0), frame_wh=(200, 150))
    end = c._rot_end().toPoint()
    counter = _Counter(c)
    _drag(c, (end.x(), end.y()), (end.x() - 40, end.y() + 40), n=30)
    assert counter.steps > 10
    assert counter.commits == 1
