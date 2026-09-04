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

from PySide6.QtCore import QEvent, QPointF, Qt                   # noqa: E402
from PySide6.QtGui import QMouseEvent, QPixmap                    # noqa: E402

from aglaia.gui.DebugEditCanvas import EditCanvas                # noqa: E402


def _mouse(kind, pos):
    """A real QMouseEvent — the handlers fall through to Qt's own, which
    rejects a stand-in object."""
    return QMouseEvent(kind, QPointF(pos), QPointF(pos),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


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
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, far))
    x, y = canvas.polygon()[0]
    assert x == pytest.approx(99.0)
    assert y == pytest.approx(79.0)


def test_double_click_inserts_a_vertex_on_the_nearest_edge(canvas):
    """A hull that follows the text usually needs a point ADDED: the
    detector's polygon is convex and the page rarely is."""
    canvas.set_editable(polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
                        origin=(0, 0), frame_wh=(200, 150))
    mid = canvas._to_view(50, 0).toPoint()
    canvas.mouseDoubleClickEvent(_mouse(QEvent.Type.MouseButtonDblClick, mid))
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


def test_a_quad_refuses_a_fifth_corner(canvas):
    """A keystone is a projective map from FOUR points; a fifth would have no
    meaning, so the keystone's polygon does not accept insertion."""
    canvas.set_editable(polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
                        origin=(0, 0), frame_wh=(200, 150),
                        allow_insert=False)
    mid = canvas._to_view(50, 0).toPoint()
    canvas.mouseDoubleClickEvent(_mouse(QEvent.Type.MouseButtonDblClick, mid))
    assert len(canvas.polygon()) == 4


# ── the dewarp's derived controls ─────────────────────────────────────

def test_arch_and_tilt_round_trip_through_the_fitted_pair(qapp_or_skip):
    from aglaia.gui.DebugViewerTab import DebugViewerWidget as W
    for a, b in ((0.07, 0.168), (-0.35, 0.0), (0.0, 0.0), (0.12, -0.04)):
        d = W._from_curl({"alpha": a, "beta": b, "gamma": 0.03})
        back = W._to_curl(d["arch"], d["tilt"], d["gamma"])
        assert back["alpha"] == pytest.approx(a)
        assert back["beta"] == pytest.approx(b)
        assert back["gamma"] == pytest.approx(0.03)


def test_arch_alone_sets_the_mid_page_rise(qapp_or_skip):
    """The point of the reparametrisation. α and β are the sheet's slopes at
    the two page EDGES: neither moves one visible thing on its own, which is
    what made them unusable by hand. `arch` moves the mid-page rise and
    nothing else; `tilt` slides the crest and leaves the rise alone."""
    from aglaia.gui.DebugViewerTab import DebugViewerWidget as W
    from aglaia.processors.sheet_models import cylindrical_z

    def mid(arch, tilt):
        c = W._to_curl(arch, tilt, 0.0)
        return float(cylindrical_z(0.5, c["alpha"], c["beta"]))

    # tilt does not touch the mid-page rise
    assert mid(0.20, -0.10) == pytest.approx(mid(0.20, 0.10))
    # arch does, linearly, and exactly at arch/4
    assert mid(0.20, 0.0) == pytest.approx(0.20 / 4.0)
    assert mid(-0.08, 0.05) == pytest.approx(-0.08 / 4.0)


def test_the_slider_ranges_cover_the_corpus(qapp_or_skip):
    """Measured over 276 fitted pages of `delbrel-oc9`: |arch| <= 0.325,
    |tilt| <= 0.250, |gamma| <= 0.100. A range that clipped a real page would
    make it untunable."""
    from aglaia.gui.DebugViewerTab import DebugViewerWidget as W
    for key, seen in (("arch", 0.325), ("tilt", 0.250), ("gamma", 0.100)):
        lo, hi, step = W._RANGES[key]
        assert lo <= -seen and hi >= seen
        assert step <= 0.001            # ~one step per slider pixel


def test_a_preview_can_be_drawn_and_cleared(canvas):
    canvas.set_editable(frame_wh=(200, 150))
    assert canvas._preview == []
    canvas.set_preview([[(0, 0), (100, 10)], [(0, 50), (100, 60)]])
    assert len(canvas._preview) == 2
    canvas.set_preview(None)
    assert canvas._preview == []
