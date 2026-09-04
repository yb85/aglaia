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
