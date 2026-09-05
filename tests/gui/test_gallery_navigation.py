# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Getting to the first or last scan in the gallery.

Stepping one page at a time is fine for neighbours and useless for a 300-page
book: reaching the end meant holding a key or clicking a chevron 299 times.

Home / End, and ⌘↑ / ⌘↓ because Home and End need Fn on most Mac keyboards and
⌘↑/⌘↓ is the native start/end-of-document idiom there. Shift-click on the
existing up/down chevron is the mouse route — the same "all the way" convention
as a shift-click in a list, and it reuses a control that is already on screen
rather than putting two more floating buttons over the page.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt                                       # noqa: E402
from PySide6.QtGui import QKeyEvent                                 # noqa: E402
from PySide6.QtWidgets import QApplication                          # noqa: E402

from aglaia.gui.ScansGalleryView import ScansGalleryView            # noqa: E402

N = 12
STAGES = ["raw", "01_a", "02_b"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def view(app):
    v = ScansGalleryView(
        scans_provider=lambda: [(i + 1, f"book_{i:03d}") for i in range(N)],
        stages_provider=lambda: list(STAGES),
        stage_resolver=lambda scan_id, stage: [("A", 1, 1)],
        thumb_loader=lambda *a, **k: None,
    )
    v.reload()
    return v


def _key(view, key, mods=Qt.KeyboardModifier.NoModifier):
    view.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, key, mods))


def test_the_fixture_starts_at_the_first_scan(view):
    """Every assertion below is about movement; if it started at the end,
    `go_first` would pass by doing nothing."""
    assert view._scan_idx == 0
    assert len(view._scans) == N


class TestJumping:
    def test_end_goes_to_the_last_scan(self, view):
        _key(view, Qt.Key.Key_End)
        assert view._scan_idx == N - 1

    def test_home_comes_back(self, view):
        view.go_last()
        _key(view, Qt.Key.Key_Home)
        assert view._scan_idx == 0

    def test_cmd_down_and_cmd_up(self, view):
        """The Mac idiom — Home/End need Fn on most Mac keyboards."""
        _key(view, Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)
        assert view._scan_idx == N - 1
        _key(view, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
        assert view._scan_idx == 0

    def test_a_plain_arrow_still_steps_by_one(self, view):
        """The jump must not swallow the ordinary key."""
        _key(view, Qt.Key.Key_Down)
        assert view._scan_idx == 1
        _key(view, Qt.Key.Key_Up)
        assert view._scan_idx == 0


class TestItStaysInRange:
    def test_home_at_the_first_scan_does_nothing(self, view):
        view.go_first()
        assert view._scan_idx == 0

    def test_end_at_the_last_scan_does_nothing(self, view):
        view.go_last()
        view.go_last()
        assert view._scan_idx == N - 1

    def test_an_empty_gallery_is_not_an_error(self, app):
        """Reachable: a project with every scan deleted, or one still
        importing."""
        v = ScansGalleryView(
            scans_provider=lambda: [],
            stages_provider=lambda: list(STAGES),
            stage_resolver=lambda *a: [],
            thumb_loader=lambda *a, **k: None,
        )
        v.reload()
        v.go_first()
        v.go_last()          # must not raise, and must not index -1
        assert v._scan_idx == 0

    def test_an_out_of_range_index_is_clamped(self, view):
        view._jump_to(9999)
        assert view._scan_idx == N - 1
        view._jump_to(-5)
        assert view._scan_idx == 0


def test_the_stage_is_carried_across_the_jump(view):
    """Landing on page 300 must not also throw away the stage being examined
    — that is the whole reason a single step carries it too."""
    view._stage_idx = 2
    view.go_last()
    assert view._scan_idx == N - 1
    assert view._stage_idx == 2


def test_the_chevrons_say_how(view):
    """A keyboard-only feature is an invisible one. The tooltip on the control
    the user is already reaching for is where the shortcut gets learned."""
    assert "Home" in view._btn_up.toolTip()
    assert "End" in view._btn_down.toolTip()
    assert "shift-click" in view._btn_up.toolTip().lower()
