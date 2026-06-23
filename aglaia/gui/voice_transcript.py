# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
"""Shared rendering for the voice transcript.

Each recent word is colour-coded by how the worker handled it, so the user
can see at a glance whether a spoken command actually fired:

* **green** (success)  — recognised command that triggered an action
* **yellow** (warning) — recognised command suppressed by the debounce window
* **red** (error)      — word not recognised as a command (incl. Vosk `[unk]`)

The Vosk backend (`VoskVoiceWorker`) builds a rolling list of ``(word, state)``
pairs and emits ``words_html(pairs)`` through ``transcription_update``; the
capture tab renders it as rich text.
"""
from __future__ import annotations

import html as _html

from aglaia.gui import colors

FIRED = "fire"        # command word that triggered an action
DEBOUNCED = "debounce"  # command word swallowed by the cooldown
UNKNOWN = "unknown"   # not a command word

_COLOR_TOKEN = {
    FIRED: "COLOR_SUCCESS",
    DEBOUNCED: "COLOR_WARNING",
    UNKNOWN: "COLOR_ERROR",
}


def classify(clean_word: str, commands, *, fired: bool) -> str:
    """State for one cleaned word given the command map and whether it fired
    (cooldown allowed it). Non-commands are UNKNOWN."""
    if clean_word in commands:
        return FIRED if fired else DEBOUNCED
    return UNKNOWN


def words_html(pairs) -> str:
    """`pairs`: iterable of ``(word, state)`` → HTML with coloured spans.
    Colours are resolved live so a dark/light theme switch is reflected."""
    out = []
    for word, state in pairs:
        col = getattr(colors, _COLOR_TOKEN.get(state, "COLOR_ERROR"))
        out.append(
            f'<span style="color:{col}; font-weight:600">'
            f'{_html.escape(str(word))}</span>'
        )
    return " ".join(out)
