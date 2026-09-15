# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Mode artwork follows the palette (#114).

`assets/modes/*.svg` paint with `fill="currentColor"`, which QSvgRenderer
does not resolve — it paints BLACK. Drawn straight through `QIcon`, the book
icons were near-invisible on the dark palette, while the Lucide icon beside
them (already routed through the tinting renderer) looked right.
"""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QIcon, QImage                          # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.app_data.modes import MODES                          # noqa: E402
from aglaia.gui.theme import svg_pixmap_path                     # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _ink_rgb(pix):
    """Mean colour of the most opaque pixels — the drawn strokes."""
    img = pix.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(
        img.height(), img.width(), 4)
    alpha = arr[..., 3]
    assert alpha.max() > 0, "nothing was drawn"
    sel = alpha > alpha.max() * 0.8
    b, g, r = (int(arr[..., i][sel].mean()) for i in range(3))
    return r, g, b


def _mode_icons():
    return [(m.key, m.icon_path()) for m in MODES if m.icon_path() is not None]


def test_there_is_artwork_to_tint():
    assert _mode_icons(), "no bundled mode icons found"


@pytest.mark.parametrize("key,path", _mode_icons())
def test_artwork_takes_the_colour_it_is_given(app, key, path):
    light = svg_pixmap_path(path, color="#f0f0f0", size=96)
    dark = svg_pixmap_path(path, color="#18181b", size=96)
    assert _ink_rgb(light) == pytest.approx((240, 240, 240), abs=3)
    assert _ink_rgb(dark) == pytest.approx((24, 24, 27), abs=3)


@pytest.mark.parametrize("key,path", _mode_icons())
def test_the_old_qicon_path_would_have_been_black(app, key, path):
    """Pins the cause, so a well-meaning revert to `QIcon(path).pixmap()`
    fails loudly instead of quietly going black again."""
    assert _ink_rgb(QIcon(str(path)).pixmap(96, 96)) == (0, 0, 0)


def test_an_rgba_colour_survives_the_renderer(app):
    """`rgba()` is the other thing QSvgRenderer blackens; the alpha has to
    come back as opacity, not as a black stroke. This is what the 26 px card
    icons are drawn with (COLOR_FONT_MUTED)."""
    _key, path = _mode_icons()[0]
    pix = svg_pixmap_path(path, color="rgba(255, 255, 255, 0.55)", size=26)
    assert _ink_rgb(pix) == (255, 255, 255)
    img = pix.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(
        img.height(), img.width(), 4)
    assert 100 < int(arr[..., 3].max()) < 200      # ~0.55 of full opacity


def test_a_non_svg_icon_is_shown_untinted(app, tmp_path):
    """A user's own coloured PNG has no `currentColor` to substitute, and
    repainting their artwork would be wrong."""
    png = tmp_path / "custom.png"
    src = svg_pixmap_path(_mode_icons()[0][1], color="#ff0000", size=32)
    assert src.save(str(png), "PNG")
    out = svg_pixmap_path(png, color="#00ff00", size=32)
    assert not out.isNull()
    assert _ink_rgb(out)[0] > _ink_rgb(out)[1]     # still red, not green


def test_a_missing_path_returns_an_empty_pixmap(app, tmp_path):
    assert svg_pixmap_path(None).isNull()
    assert svg_pixmap_path(tmp_path / "nope.svg").isNull()


# ── size and aspect (regression of #114) ──────────────────────────────

def _ink_bbox(pix):
    """(x0, y0, x1, y1) of the drawn pixels, in device pixels."""
    img = pix.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(
        img.height(), img.width(), 4)
    ys, xs = np.nonzero(arr[..., 3] > 16)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_mode_artwork_has_the_requested_logical_size(app):
    """#114 swapped `QIcon.pixmap(96, 96)`, which returned 96 px, for a 2x
    render with no device pixel ratio — 192 px reported, so a 96-px label
    showed only the centre of the book. Every mode card came out cropped."""
    for mode in MODES:
        path = mode.icon_path()
        if path is None or path.suffix != ".svg":
            continue
        pix = svg_pixmap_path(path, color="#ffffff", size=96)
        assert pix.deviceIndependentSize().width() == pytest.approx(96)
        assert pix.deviceIndependentSize().height() == pytest.approx(96)


def test_non_square_artwork_keeps_its_aspect(app):
    """A bare `QSvgRenderer.render(p)` stretches the viewBox to the square.
    The mode artwork is not square — up to 1203 x 762 — so the ink must span
    the full width and only part of the height, centred."""
    wide = None
    for mode in MODES:
        path = mode.icon_path()
        if path is not None and path.name == "book_flat_x2.svg":
            wide = path              # viewBox 1203 x 762 — the widest one
    if wide is None:
        pytest.skip("book_flat_x2.svg not bundled")
    pix = svg_pixmap_path(wide, color="#ffffff", size=96)
    x0, y0, x1, y1 = _ink_bbox(pix)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    side = pix.width()
    assert h < side * 0.8                   # not stretched to fill the height
    assert abs((y0 + y1) / 2 - side / 2) < side * 0.08   # vertically centred
    # The ink does not fill its own viewBox (the artwork has margins), so
    # compare against a reference render on a canvas of the viewBox's OWN
    # aspect — where a plain render cannot stretch anything.
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer
    renderer = QSvgRenderer(QByteArray(wide.read_bytes()))
    vb = renderer.viewBoxF()
    ref = QPixmap(int(vb.width() / 4), int(vb.height() / 4))
    ref.fill(Qt.GlobalColor.transparent)
    painter = QPainter(ref)
    renderer.render(painter)
    painter.end()
    rx0, ry0, rx1, ry1 = _ink_bbox(ref)
    ref_aspect = (rx1 - rx0 + 1) / (ry1 - ry0 + 1)
    assert w / h == pytest.approx(ref_aspect, rel=0.05)
