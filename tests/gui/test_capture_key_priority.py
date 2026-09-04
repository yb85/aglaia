# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Capture keys outrank the focused widget (#119).

`MainWindow.keyPressEvent` is the LAST stop in Qt's propagation: whatever has
focus sees the press first, and anything it accepts never reaches the window.
A presentation remote bound to PgUp/PgDown/Ctrl+Return therefore paged the
list view's scroll area instead of capturing, and the first press after a
click was eaten by whatever had just taken focus.

The bindings themselves were fine — what was missing is priority. These tests
cover the two halves that can be tested without a camera: the binding table
resolving a press to an action, and the guard that decides when the capture
keys may pre-empt the focused widget.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QKeyCombination, Qt              # noqa: E402
from PySide6.QtGui import QKeyEvent                                 # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QLineEdit,  # noqa: E402
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QTextEdit)

from aglaia.gui import keybindings as kb                            # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _press(key, mod=Qt.KeyboardModifier.NoModifier, autorep=False):
    return QKeyEvent(QEvent.Type.KeyPress, key, mod, 0, 0, 0,
                     autorep=autorep)


REMOTE = {"scan": ["PgDown", "Ctrl+Return"],
          "trash": ["PgUp"],
          "rotate": ["R"]}


# ── the remote's keys resolve at all ─────────────────────────────────

@pytest.mark.parametrize("text", ["PgUp", "PgDown", "Ctrl+Return",
                                  "Shift+F5", "Esc", "Space", "S"])
def test_the_remote_spellings_round_trip(app, text):
    assert kb.normalise(text) == text


@pytest.mark.parametrize("key,mod,want", [
    (Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier, "scan"),
    (Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier, "scan"),
    (Qt.Key.Key_PageUp, Qt.KeyboardModifier.NoModifier, "trash"),
    (Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier, "rotate"),
    (Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier, None),
    # Bound bare, pressed with a modifier — a different combination.
    (Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier, None),
])
def test_a_press_resolves_to_its_action(app, key, mod, want):
    assert kb.action_for(_press(key, mod), REMOTE) == want


def test_a_bare_modifier_resolves_to_nothing(app):
    """Holding Ctrl on the way to Return must not fire anything by itself."""
    assert kb.action_for(_press(Qt.Key.Key_Control), REMOTE) is None


def test_action_for_agrees_with_matches(app):
    """`action_for` is the one-pass form of `matches`; they must not drift."""
    ev = _press(Qt.Key.Key_PageUp)
    assert kb.action_for(ev, REMOTE) == "trash"
    assert kb.matches(ev, REMOTE, "trash") is True
    assert kb.matches(ev, REMOTE, "scan") is False


# ── where the keys must NOT be taken ─────────────────────────────────

def test_text_entries_keep_their_keystrokes(app):
    """Typing `s` into a filename field is an `s`, not a capture. The
    keybinding recorder is a QLineEdit too, so this is also what lets it
    record a bound key instead of firing it."""
    for w in (QLineEdit(), QTextEdit(), QPlainTextEdit(), QSpinBox()):
        assert kb.is_text_entry(w) is True


def test_an_editable_combo_counts_but_a_fixed_one_does_not(app):
    fixed, editable = QComboBox(), QComboBox()
    editable.setEditable(True)
    assert kb.is_text_entry(fixed) is False
    assert kb.is_text_entry(editable) is True


def test_a_button_is_not_a_text_entry(app):
    """This is the widget that was eating the press — it must NOT be
    protected, or the fix does nothing."""
    assert kb.is_text_entry(QPushButton("Capture")) is False


def test_nothing_focused_is_not_a_text_entry(app):
    assert kb.is_text_entry(None) is False


def test_a_subclass_of_a_text_entry_still_counts(app):
    class _Slot(QLineEdit):
        pass
    assert kb.is_text_entry(_Slot()) is True


# ── auto-repeat ──────────────────────────────────────────────────────

def test_an_autorepeat_press_is_marked_as_one(app):
    """A held key must not spray captures. The filter checks this flag, so
    pin that Qt reports it the way we read it."""
    assert _press(Qt.Key.Key_PageDown, autorep=True).isAutoRepeat() is True
    assert _press(Qt.Key.Key_PageDown).isAutoRepeat() is False
