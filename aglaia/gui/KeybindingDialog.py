# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The keybinding editor (#103).

Two slots per action, each an input showing its current combination. There is
no "record" button: **focusing a slot arms it**, the next key or combination
pressed is what it becomes, and focus leaves. That is one gesture instead of
three, and it is the gesture a user already expects from every game and every
IDE that lets them rebind a key.

The awkward part is that a field which swallows keys must not swallow the ones
that operate the dialog. So an armed slot keeps Tab (move on) and Escape
(disarm) for the dialog, and takes everything else.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGridLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout,
)

from aglaia.gui import keybindings as kb
from aglaia.gui.colors import COLOR_FONT_MUTED, COLOR_PRIMARY


class KeySlot(QLineEdit):
    """One binding. Focus arms it; the next combination is recorded."""

    PLACEHOLDER = "click, then press a key"

    def __init__(self, seq: str = "", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)          # typed text is never the value
        self.setPlaceholderText(self.tr(self.PLACEHOLDER))
        self.setMinimumWidth(150)
        self.set_sequence(seq)

    def set_sequence(self, seq: str) -> None:
        self.setText(kb.normalise(seq))
        self._restyle()

    def sequence(self) -> str:
        return kb.normalise(self.text())

    # ── arming ────────────────────────────────────────────────────
    def focusInEvent(self, ev):  # noqa: N802
        super().focusInEvent(ev)
        self._restyle()

    def focusOutEvent(self, ev):  # noqa: N802
        super().focusOutEvent(ev)
        self._restyle()

    def keyPressEvent(self, ev):  # noqa: N802
        # Tab and Escape stay the dialog's: a field that swallows every key
        # would trap the user inside it with no way out but the mouse.
        if ev.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            super().keyPressEvent(ev)
            return
        if ev.key() == Qt.Key.Key_Escape and not ev.modifiers():
            self.clearFocus()
            return
        seq = kb.from_event(ev)
        if not seq:
            # A bare modifier on the way to the real key. Stay armed.
            ev.accept()
            return
        self.setText(seq)
        self._restyle()
        self.clearFocus()
        ev.accept()

    def _restyle(self) -> None:
        armed = self.hasFocus()
        colour = COLOR_PRIMARY if armed else COLOR_FONT_MUTED
        self.setStyleSheet(
            f"QLineEdit {{ border: 1px solid {colour}; border-radius: 4px; "
            f"padding: 3px 6px; }}")


class KeybindingDialog(QDialog):
    """Edit every capture binding. Returns the map on accept."""

    def __init__(self, bindings: dict, defaults: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Capture shortcuts"))
        self._defaults = {k: list(v) for k, v in (defaults or {}).items()}

        v = QVBoxLayout(self)
        v.setSpacing(10)
        intro = QLabel(self.tr(
            "Click a field, then press the key or combination. Two per "
            "action — a presentation remote often sends a different one each "
            "press, and both can drive the same action."))
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        v.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self._slots: dict[str, list[KeySlot]] = {}
        for row, (action, label) in enumerate(kb.ACTIONS):
            grid.addWidget(QLabel(self.tr(label)), row, 0)
            seqs = list((bindings or {}).get(action) or [])
            slots = []
            for i in range(kb.SLOTS):
                slot = KeySlot(seqs[i] if i < len(seqs) else "")
                grid.addWidget(slot, row, 1 + i)
                slots.append(slot)
            self._slots[action] = slots
            clear = QPushButton(self.tr("Clear"))
            clear.setFixedWidth(60)
            clear.clicked.connect(
                lambda _=False, a=action: self._clear(a))
            grid.addWidget(clear, row, 1 + kb.SLOTS)
        v.addLayout(grid)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.RestoreDefaults
                  ).clicked.connect(self._restore_defaults)
        v.addWidget(bb)

    def _clear(self, action: str) -> None:
        for slot in self._slots.get(action, ()):
            slot.set_sequence("")

    def _restore_defaults(self) -> None:
        for action, slots in self._slots.items():
            seqs = list(self._defaults.get(action) or [])
            for i, slot in enumerate(slots):
                slot.set_sequence(seqs[i] if i < len(seqs) else "")

    def bindings(self) -> dict[str, list[str]]:
        """``{action: [seq, …]}``. A cleared action maps to an empty list,
        which is a decision the user made — not the absence of one — so it
        must not silently fall back to the YAML default."""
        out: dict[str, list[str]] = {}
        for action, slots in self._slots.items():
            seen: list[str] = []
            for slot in slots:
                seq = slot.sequence()
                if seq and seq not in seen:
                    seen.append(seq)
            out[action] = seen
        return out
