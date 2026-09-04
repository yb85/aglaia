# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The debug view as the per-page editor (M9 #97).

The handle layer is a coordinate problem before it is a UI one: the geometry
is expressed in the STAGE frame, the picture it is drawn on is a composite
with a label bar above it and, for a layout, the child sitting at its crop
offset on the parent. Get the mapping wrong and every handle is a bar-height
— or a crop — away from the pixels it describes.

`EditCanvas` is tested on its own here; the surrounding widget needs a real
project DB and is exercised by hand.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF                               # noqa: E402
from PySide6.QtGui import QPixmap                                # noqa: E402

from aglaia.gui.DebugEditCanvas import EditCanvas                # noqa: E402


@pytest.fixture()
def canvas(qapp_or_skip):
    c = EditCanvas()
    c.resize(400, 300)
    pix = QPixmap(200, 150)          # the composite
    pix.fill()
    c.set_image(pix)
    return c


@pytest.fixture(scope="session")
def qapp_or_skip():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def test_no_geometry_means_no_handles(canvas):
    canvas.set_editable()
    assert canvas.polygon() is None
    assert canvas.rotation_deg() is None


def test_the_origin_offsets_every_handle(canvas):
    """The stage frame sits below the composite's label bar. A handle layer
    that ignored the offset would sit a bar-height high."""
    canvas.set_editable(polygon=[[10, 10]], origin=(0, 0), frame_wh=(200, 150))
    at_zero = canvas._to_view(10, 10)
    canvas.set_editable(polygon=[[10, 10]], origin=(5, 40), frame_wh=(200, 150))
    shifted = canvas._to_view(10, 10)
    assert shifted.x() > at_zero.x()
    assert shifted.y() > at_zero.y()


def test_view_and_source_are_inverses(canvas):
    """Whatever the origin, a point mapped out and back must land on itself —
    otherwise a drag would creep away from the cursor."""
    canvas.set_editable(polygon=[[10, 10]], origin=(7, 41), frame_wh=(200, 150))
    for src in ((0.0, 0.0), (30.0, 90.0), (120.0, 12.0)):
        back = canvas._to_src(canvas._to_view(*src).toPoint())
        assert back == pytest.approx(src, abs=1.0)


def test_a_dragged_vertex_is_clamped_into_the_stage_frame(canvas):
    """`frame_wh` is the page, not the composite. Clamping to the composite
    would let a vertex wander into the label bar."""
    canvas.set_editable(polygon=[[10, 10], [50, 10], [50, 50]],
                        origin=(0, 20), frame_wh=(100, 80))
    canvas._drag_vertex = 0
    far = canvas._to_view(1000, 1000).toPoint()

    class _Ev:
        @staticmethod
        def position():
            return QPointF(far)

    canvas.mouseMoveEvent(_Ev())
    x, y = canvas.polygon()[0]
    assert x == pytest.approx(99.0)
    assert y == pytest.approx(79.0)


def test_double_click_inserts_a_vertex_on_the_nearest_edge(canvas):
    """A hull that follows the text usually needs a point ADDED: the
    detector's polygon is convex and the page rarely is."""
    canvas.set_editable(polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
                        origin=(0, 0), frame_wh=(200, 150))
    mid = canvas._to_view(50, 0).toPoint()

    class _Ev:
        @staticmethod
        def button():
            from PySide6.QtCore import Qt
            return Qt.MouseButton.LeftButton

        @staticmethod
        def position():
            return QPointF(mid)

    canvas.mouseDoubleClickEvent(_Ev())
    poly = canvas.polygon()
    assert len(poly) == 5
    assert poly[1][0] == pytest.approx(50.0, abs=2.0)


def test_setting_the_rotation_does_not_echo_back(canvas):
    """The slider and the handle drive the same value. If moving one emitted
    at the other, the two would fight while the user drags."""
    seen = []
    canvas.edited.connect(lambda k, v: seen.append((k, v)))
    canvas.set_editable(rotation_deg=0.0, frame_wh=(200, 150))
    canvas.set_rotation_deg(-3.0)
    assert canvas.rotation_deg() == -3.0
    assert seen == []


def test_the_rotation_handle_follows_the_angle(canvas):
    canvas.set_editable(rotation_deg=0.0, frame_wh=(200, 150))
    flat = canvas._rot_end()
    canvas.set_rotation_deg(45.0)
    tilted = canvas._rot_end()
    assert tilted.y() > flat.y()          # +45 deg points down-right
    assert tilted.x() < flat.x()


def test_the_composite_downscale_is_applied(canvas):
    """The renderer hands geometry in full-resolution composite pixels while
    the picture on screen is downscaled. Miss the factor and the error grows
    with the coordinate."""
    canvas.set_editable(polygon=[[100, 100]], origin=(0, 0),
                        frame_wh=(200, 150), scale=1.0)
    full = canvas._to_view(100, 100)
    canvas.set_editable(polygon=[[100, 100]], origin=(0, 0),
                        frame_wh=(200, 150), scale=0.5)
    half = canvas._to_view(100, 100)
    fit = canvas._fit_rect()
    assert (half.x() - fit.x()) == pytest.approx((full.x() - fit.x()) / 2)
    assert (half.y() - fit.y()) == pytest.approx((full.y() - fit.y()) / 2)


def test_view_and_source_are_inverses_under_a_downscale(canvas):
    """A drag reads the cursor back through `_to_src`; if it did not undo the
    scale the vertex would creep away from the pointer."""
    canvas.set_editable(polygon=[[10, 10]], origin=(7, 41),
                        frame_wh=(400, 300), scale=0.5)
    for src in ((0.0, 0.0), (60.0, 180.0), (240.0, 24.0)):
        back = canvas._to_src(canvas._to_view(*src).toPoint())
        assert back == pytest.approx(src, abs=2.0)
