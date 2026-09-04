# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Editable capture keybindings (#103).

The binding this feature exists for is a **presentation remote**: its
fullscreen button cycles between `Shift+F5` and `Esc`, so capture must accept
both. The old matcher could express neither — it compared key NAMES and never
looked at the modifiers.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt                          # noqa: E402
from PySide6.QtGui import QKeyEvent                            # noqa: E402

from aglaia.gui import keybindings as kb                       # noqa: E402

YAML = {"keycontrols": {"scan": ["Space", "S"],
                        "trash": ["Backspace", "D"],
                        "rotate": ["R"]}}


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def app_data(tmp_path, monkeypatch):
    """An empty app-data dir, so the stored bindings start unset."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import importlib
    import aglaia.app_data as ad
    import aglaia.app_data.db as db
    importlib.reload(ad)
    importlib.reload(db)
    return db


def _press(key, mods=Qt.KeyboardModifier.NoModifier):
    return QKeyEvent(QEvent.Type.KeyPress, key, mods, "")


# ── spelling ──────────────────────────────────────────────────────────

def test_a_binding_has_one_spelling(qapp):
    """"esc", "ESC" and "Escape" must be ONE binding, not three of which
    only one matches."""
    assert kb.normalise("esc") == kb.normalise("Escape") == "Esc"
    assert kb.normalise("shift+f5") == "Shift+F5"
    assert kb.normalise("  ") == ""


def test_every_legacy_default_still_parses(qapp):
    """A config written before this module must keep working untouched."""
    assert kb.defaults_from_config(YAML) == {
        "scan": ["Space", "S"], "trash": ["Backspace", "D"], "rotate": ["R"]}


# ── recording ─────────────────────────────────────────────────────────

def test_a_bare_modifier_records_nothing(qapp):
    """A user holding Shift on the way to F5 must not end up bound to
    "Shift"; the slot stays armed instead."""
    for mod_key in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt,
                    Qt.Key.Key_Meta):
        assert kb.from_event(_press(mod_key)) == ""


def test_a_combination_records_its_modifiers(qapp):
    assert kb.from_event(
        _press(Qt.Key.Key_F5, Qt.KeyboardModifier.ShiftModifier)) == "Shift+F5"
    assert kb.from_event(_press(Qt.Key.Key_Escape)) == "Esc"


# ── matching ──────────────────────────────────────────────────────────

def test_capture_answers_to_both_remote_keys(qapp):
    """The whole point: one action, two bindings, because the remote sends a
    different one each press."""
    bindings = {"scan": ["Shift+F5", "Esc"]}
    assert kb.matches(_press(Qt.Key.Key_F5,
                             Qt.KeyboardModifier.ShiftModifier),
                      bindings, "scan")
    assert kb.matches(_press(Qt.Key.Key_Escape), bindings, "scan")
    # …and not to the unmodified key, which the old name-matcher could not
    # have told apart.
    assert not kb.matches(_press(Qt.Key.Key_F5), bindings, "scan")


def test_an_unbound_action_never_fires(qapp):
    assert not kb.matches(_press(Qt.Key.Key_Space), {"scan": []}, "scan")


# ── persistence ───────────────────────────────────────────────────────

def test_stored_bindings_override_the_yaml(qapp, app_data):
    assert kb.resolve(YAML)["scan"] == ["Space", "S"]
    kb.save({"scan": ["Shift+F5", "Esc"]})
    resolved = kb.resolve(YAML)
    assert resolved["scan"] == ["Shift+F5", "Esc"]
    assert resolved["trash"] == ["Backspace", "D"]      # untouched → YAML


def test_clearing_an_action_is_a_decision_not_an_absence(qapp, app_data):
    """A user who cleared capture wants NO capture key. Falling back to the
    YAML default would hand it straight back."""
    kb.save({"scan": []})
    assert kb.resolve(YAML)["scan"] == []


def test_at_most_two_bindings_are_kept(qapp, app_data):
    kb.save({"scan": ["A", "B", "C"]})
    assert len(kb.stored()["scan"]) == kb.SLOTS


# ── the editor ────────────────────────────────────────────────────────

def test_the_slot_records_the_first_combination_pressed(qapp):
    from aglaia.gui.KeybindingDialog import KeySlot
    slot = KeySlot("Space")
    slot.keyPressEvent(_press(Qt.Key.Key_Shift))          # on the way there
    assert slot.sequence() == "Space"
    slot.keyPressEvent(_press(Qt.Key.Key_F5,
                              Qt.KeyboardModifier.ShiftModifier))
    assert slot.sequence() == "Shift+F5"


def test_escape_leaves_a_slot_instead_of_binding_to_it(qapp):
    """A field that swallowed every key would trap the user in it. Escape is
    the way out — so binding TO Escape is done from the other slot, or from
    the same one after moving focus back in."""
    from aglaia.gui.KeybindingDialog import KeySlot
    slot = KeySlot("Space")
    slot.keyPressEvent(_press(Qt.Key.Key_Escape))
    assert slot.sequence() == "Space"


def test_the_dialog_drops_a_duplicate_within_one_action(qapp):
    from aglaia.gui.KeybindingDialog import KeybindingDialog
    dlg = KeybindingDialog({"scan": ["Esc", "Esc"]},
                           kb.defaults_from_config(YAML))
    assert dlg.bindings()["scan"] == ["Esc"]
