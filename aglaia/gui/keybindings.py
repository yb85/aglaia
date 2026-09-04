# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Capture keybindings: resolve, match, persist (#103).

The bindings used to be key NAMES in `args.config["keycontrols"]`, matched by
hand against `event.text()` and a table of seven names. That matcher never
looked at the modifiers, so a combination was not expressible at all — and a
combination is exactly what a **presentation remote** sends: the one this was
built for has a fullscreen button that cycles between `Shift+F5` and `Esc`, so
driving capture from it needs BOTH bound to the same action.

Everything here goes through `QKeySequence`, which parses and prints portable,
platform-correct names, understands modifiers, and already accepts every
legacy default (`Space`, `S`, `Backspace`, `D`, `R`) — so a config written
before this module keeps working untouched.

Resolution order per action, first hit wins:

1. the user's binding in the app-data config DB (`KEY_KEYBINDINGS`);
2. `args.config["keycontrols"]` — the YAML default.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

#: The bindable actions, in the order the editor lists them: what they do
#: comes first, because that is what a user is looking for.
ACTIONS: tuple[tuple[str, str], ...] = (
    ("scan", "Capture"),
    ("trash", "Delete last scan"),
    ("rotate", "Rotate camera"),
)

#: Slots per action. Two, because that is what the presenter-remote case
#: needs and what keeps the panel legend readable.
SLOTS = 2


def normalise(seq: str) -> str:
    """A binding string in `QKeySequence`'s own portable spelling, or ``""``.

    Round-tripping through `QKeySequence` is what makes "esc", "ESC" and
    "Escape" one binding rather than three that only one of which matches.
    """
    text = (seq or "").strip()
    if not text:
        return ""
    ks = QKeySequence(text)
    if ks.isEmpty():
        return ""
    return ks.toString(QKeySequence.SequenceFormat.PortableText)


def from_event(event) -> str:
    """The binding a key press describes, or ``""`` for a bare modifier.

    A user holding Shift on the way to F5 must not have "Shift" recorded as
    their binding, so a press whose key IS a modifier records nothing and the
    slot stays armed.
    """
    key = event.key()
    if key in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt,
               Qt.Key.Key_Meta, Qt.Key.Key_AltGr, Qt.Key.Key_unknown, 0):
        return ""
    return QKeySequence(event.keyCombination()).toString(
        QKeySequence.SequenceFormat.PortableText)


def defaults_from_config(config: Optional[dict]) -> dict[str, list[str]]:
    """The YAML `keycontrols` as normalised binding strings."""
    raw = ((config or {}).get("keycontrols") or {})
    out: dict[str, list[str]] = {}
    for action, _label in ACTIONS:
        seqs = [normalise(str(k)) for k in (raw.get(action) or [])]
        out[action] = [s for s in seqs if s][:SLOTS]
    return out


def stored() -> dict[str, list[str]]:
    """The user's bindings from the app-data config DB."""
    try:
        from aglaia.app_data import db as cfg
        with cfg.session() as conn:
            value = cfg.get(conn, cfg.KEY_KEYBINDINGS, {}) or {}
    except Exception:
        return {}
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for action, _label in ACTIONS:
        seqs = value.get(action)
        if not isinstance(seqs, list):
            continue
        clean = [normalise(str(s)) for s in seqs]
        out[action] = [s for s in clean if s][:SLOTS]
    return out


def save(bindings: dict[str, list[str]]) -> None:
    """Persist the user's bindings. An action mapped to an empty list is
    stored as such — "the user cleared this", which is not the same as "the
    user never touched it" and must not fall back to the YAML default."""
    payload = {}
    for action, _label in ACTIONS:
        if action in bindings:
            seqs = [normalise(str(s)) for s in (bindings.get(action) or [])]
            payload[action] = [s for s in seqs if s][:SLOTS]
    from aglaia.app_data import db as cfg
    with cfg.session() as conn:
        cfg.set(conn, cfg.KEY_KEYBINDINGS, payload)
        conn.commit()


def resolve(config: Optional[dict]) -> dict[str, list[str]]:
    """``{action: [seq, …]}`` — the bindings actually in force."""
    out = defaults_from_config(config)
    out.update(stored())
    return out


def matches(event, bindings: dict[str, list[str]], action: str) -> bool:
    """Does this key press fire `action`?"""
    pressed = from_event(event)
    if not pressed:
        return False
    return pressed in (bindings.get(action) or [])


def legend(bindings: dict[str, list[str]]) -> dict[str, list[str]]:
    """The bindings shaped for the panel legend: labelled, and only the
    actions that have one."""
    return {label: bindings.get(action) or []
            for action, label in ACTIONS
            if bindings.get(action)}
