# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Every icon the GUI asks for is actually bundled.

A missing icon fails **silently**: `lucide()` returns an empty QIcon and the
button simply has no picture. Nothing raises, nothing logs, and it survives
review because the code reads fine. Three had been shipping blank — `cloud`
(the Mistral jobs tab), `key-round` (Set API key) and `layers` (the mode-picker
fallback) — and nobody noticed, which is exactly the point of this test.
"""
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                          # noqa: E402

ICON_DIR = Path("aglaia/assets/icons")

#: How the GUI asks for one. Kept in one place so a new helper name is a
#: one-line change here rather than a hole in the coverage.
CALL_RE = re.compile(
    r'(?:lucide|lucide_pixmap|theme_icon|_lucide_tab|_icon|icon)'
    r'\(\s*"([a-z0-9-]+)"')


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _referenced() -> set[str]:
    names: set[str] = set()
    for f in Path("aglaia/gui").rglob("*.py"):
        names |= set(CALL_RE.findall(f.read_text("utf-8")))
    return names


def test_the_scan_finds_something():
    """A regex that matched nothing would make every assertion below pass."""
    assert len(_referenced()) > 20


def test_every_referenced_icon_is_bundled():
    have = {p.stem for p in ICON_DIR.glob("*.svg")}
    missing = sorted(n for n in _referenced() if n not in have)
    assert not missing, f"referenced but not bundled: {', '.join(missing)}"


def test_every_bundled_icon_renders(app):
    """An SVG that will not parse is as blank as one that is absent."""
    from aglaia.gui.theme import lucide_pixmap
    dead = [p.stem for p in sorted(ICON_DIR.glob("*.svg"))
            if lucide_pixmap(p.stem, color="#ffffff", size=16).isNull()]
    assert not dead, f"bundled but will not render: {', '.join(dead)}"


def test_the_bundled_icons_are_lucide(app):
    """They must all be tintable and the same visual family: `currentColor`
    is what `_tint_and_render` substitutes, and an icon without it comes out
    black on the dark theme (#114)."""
    untinted = [p.stem for p in sorted(ICON_DIR.glob("*.svg"))
                if "currentColor" not in p.read_text("utf-8")]
    assert not untinted, f"not tintable: {', '.join(untinted)}"
