# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Editing the layout SET on the parent frame (#118).

The handles used to live in ONE child's crop, so a vertex could not be dragged
outside it — the clamp WAS the crop, and the crop was the detector's answer.
Here they live in the parent frame: every layout is reachable, each carries a
trash badge at its barycentre, and one add badge sits at the picture's
top-right corner.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt                   # noqa: E402
from PySide6.QtGui import QMouseEvent, QPixmap                   # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.DebugEditCanvas import EditCanvas                # noqa: E402
from aglaia.gui.DebugViewerTab import DebugViewerWidget          # noqa: E402

A = [[20.0, 20.0], [180.0, 20.0], [180.0, 130.0], [20.0, 130.0]]
B = [[200.0, 20.0], [360.0, 20.0], [360.0, 130.0], [200.0, 130.0]]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def canvas(app):
    c = EditCanvas()
    c.resize(500, 400)
    pix = QPixmap(400, 300)
    pix.fill()
    c.set_image(pix)
    c.set_layouts([A, B], labels=["A", "B"], origin=(0, 0),
                  frame_wh=(400, 300))
    return c


def _ev(kind, pos):
    pt = QPointF(float(pos[0]), float(pos[1]))
    return QMouseEvent(kind, pt, pt, Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def _press(canvas, view_pt):
    canvas.mousePressEvent(_ev(QEvent.Type.MouseButtonPress,
                               (view_pt.x(), view_pt.y())))


# ── the set, and the frame it lives in ───────────────────────────────

def test_the_set_round_trips(canvas):
    assert canvas.layouts() == [A, B]


def test_installing_a_set_clears_the_single_shape_handles(canvas):
    """One editor at a time — a stale quad under the layouts would be
    draggable and would write to a different field."""
    assert canvas.polygon() is None
    assert canvas.rotation_deg() is None


def test_a_single_shape_clears_the_set(canvas):
    canvas.set_editable(polygon=A, origin=(0, 0), frame_wh=(400, 300))
    assert canvas.layouts() == []


# ── dragging a vertex, in PARENT coordinates ─────────────────────────

def test_a_vertex_drags_within_the_whole_parent_frame(canvas):
    """The old clamp was one child's crop. Here a corner of layout A can be
    pushed to the far side of the page, over where layout B sits."""
    start = canvas._to_view(*A[1])
    target = canvas._to_view(390.0, 290.0)
    _press(canvas, start)
    canvas.mouseMoveEvent(_ev(QEvent.Type.MouseMove,
                              (target.x(), target.y())))
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease,
                                 (target.x(), target.y())))
    moved = canvas.layouts()[0][1]
    assert moved[0] == pytest.approx(390.0, abs=2.0)
    assert moved[1] == pytest.approx(290.0, abs=2.0)


def test_a_vertex_is_still_clamped_into_the_image(canvas):
    """Reachable everywhere on the page, but not off it — a vertex under no
    pixel could not be grabbed back."""
    start = canvas._to_view(*A[0])
    _press(canvas, start)
    canvas.mouseMoveEvent(_ev(QEvent.Type.MouseMove, (-500, -500)))
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease, (-500, -500)))
    x, y = canvas.layouts()[0][0]
    assert 0 <= x <= 399 and 0 <= y <= 299


def test_dragging_one_layout_leaves_the_other_alone(canvas):
    start = canvas._to_view(*A[0])
    to = canvas._to_view(60.0, 60.0)
    _press(canvas, start)
    canvas.mouseMoveEvent(_ev(QEvent.Type.MouseMove, (to.x(), to.y())))
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease,
                                 (to.x(), to.y())))
    assert canvas.layouts()[1] == B


def test_a_set_drag_commits_once(canvas):
    """Same contract as #116 — `edited` per step, one `edit_finished`."""
    steps, commits = [], []
    canvas.edited.connect(lambda k, v: steps.append(k))
    canvas.edit_finished.connect(lambda: commits.append(1))
    start = canvas._to_view(*A[0])
    _press(canvas, start)
    for i in range(1, 21):
        canvas.mouseMoveEvent(_ev(QEvent.Type.MouseMove,
                                  (start.x() + i, start.y() + i)))
    canvas.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease,
                                 (start.x() + 20, start.y() + 20)))
    assert steps and set(steps) == {"layouts"}
    assert len(commits) == 1


# ── the badges ───────────────────────────────────────────────────────

def test_the_trash_badge_sits_at_the_barycentre(canvas):
    centre = canvas._trash_badge_centre(0)
    expect = canvas._to_view(100.0, 75.0)         # centroid of A
    assert (centre - expect).manhattanLength() < 1.0


def test_pressing_a_trash_badge_asks_for_that_layout(canvas):
    seen = []
    canvas.layout_action.connect(lambda a, i: seen.append((a, i)))
    _press(canvas, canvas._trash_badge_centre(1))
    assert seen == [("delete", 1)]


def test_pressing_the_add_badge_asks_for_one_more(canvas):
    seen = []
    canvas.layout_action.connect(lambda a, i: seen.append((a, i)))
    _press(canvas, canvas._add_badge_centre())
    assert seen == [("add", None)]


def test_a_badge_press_starts_no_drag(canvas):
    """The badge is the whole gesture. Otherwise the trash would also grab a
    vertex underneath and the page would move as it was deleted."""
    _press(canvas, canvas._trash_badge_centre(0))
    assert canvas.is_editing() is False


def test_the_last_layout_keeps_no_trash_badge(canvas):
    """Deleting it would leave the page with nothing to process."""
    canvas.set_layouts([A], labels=["A"], origin=(0, 0), frame_wh=(400, 300))
    seen = []
    canvas.layout_action.connect(lambda a, i: seen.append((a, i)))
    _press(canvas, canvas._to_view(100.0, 75.0))
    assert seen == []


def test_the_add_badge_is_inside_the_picture_top_right(canvas):
    fit = canvas._fit_rect()
    add = canvas._add_badge_centre()
    assert fit.left() < add.x() < fit.right()
    assert fit.top() < add.y() < fit.bottom()
    assert add.x() > fit.center().x() and add.y() < fit.center().y()


# ── inserting a vertex ───────────────────────────────────────────────

def test_double_click_inserts_into_the_nearest_layout(canvas):
    before = [len(p) for p in canvas.layouts()]
    edge = canvas._to_view(280.0, 20.0)           # top edge of B
    canvas.mouseDoubleClickEvent(_ev(QEvent.Type.MouseButtonDblClick,
                                     (edge.x(), edge.y())))
    after = [len(p) for p in canvas.layouts()]
    assert after == [before[0], before[1] + 1]


def test_double_click_far_from_any_edge_inserts_nothing(canvas):
    before = canvas.layouts()
    canvas.mouseDoubleClickEvent(_ev(QEvent.Type.MouseButtonDblClick, (5, 395)))
    assert canvas.layouts() == before


# ── the seed rectangle a new layout starts from ──────────────────────

def test_a_new_layout_is_inset_and_fully_grabbable():
    poly = DebugViewerWidget._new_layout_poly([], (400, 300))
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) > 0 and max(xs) < 400
    assert min(ys) > 0 and max(ys) < 300
    assert len(poly) == 4


def test_a_second_new_layout_does_not_hide_under_the_first():
    first = DebugViewerWidget._new_layout_poly([], (400, 300))
    second = DebugViewerWidget._new_layout_poly([first], (400, 300))
    assert second[0] != first[0]


def test_a_new_layout_stays_inside_a_tiny_frame():
    poly = DebugViewerWidget._new_layout_poly([], (40, 30))
    assert max(p[0] for p in poly) <= 39
    assert max(p[1] for p in poly) <= 29
