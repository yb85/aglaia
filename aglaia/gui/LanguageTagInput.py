# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Language-tag picker.

``LanguageTagInput`` is a one-line BCP-47 chip input used by the OCR
sidebar tab and the Settings tab. Was previously bundled inside
``aglaia/gui/OcrFrame.py`` — extracted so the now-defunct OcrFrame can be
deleted without orphaning Settings.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, QStringListModel
from PySide6.QtWidgets import (
    QCompleter, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)
from aglaia.gui.FlowLayout import FlowLayout
from aglaia.gui.colors import (
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_ON_BUTTON,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_PRIMARY_BORDER,
)


# Vision supports a long list of locales; expose the realistic European /
# major-Asian subset. Codes accepted as BCP-47 by VNRecognizeTextRequest.
LANGUAGES: list[tuple[str, str]] = [
    ("en-US", "English (US)"),
    ("en-GB", "English (UK)"),
    ("fr-FR", "French"),
    ("es-ES", "Spanish"),
    ("de-DE", "German"),
    ("it-IT", "Italian"),
    ("pt-PT", "Portuguese (PT)"),
    ("pt-BR", "Portuguese (BR)"),
    ("nl-NL", "Dutch"),
    ("sv-SE", "Swedish"),
    ("nb-NO", "Norwegian Bokmål"),
    ("da-DK", "Danish"),
    ("fi-FI", "Finnish"),
    ("pl-PL", "Polish"),
    ("cs-CZ", "Czech"),
    ("hu-HU", "Hungarian"),
    ("ro-RO", "Romanian"),
    ("tr-TR", "Turkish"),
    ("ru-RU", "Russian"),
    ("uk-UA", "Ukrainian"),
    ("el-GR", "Greek"),
    ("ar-SA", "Arabic"),
    ("he-IL", "Hebrew"),
    ("zh-Hans", "Chinese (Simplified)"),
    ("zh-Hant", "Chinese (Traditional)"),
    ("ja-JP", "Japanese"),
    ("ko-KR", "Korean"),
    ("th-TH", "Thai"),
    ("vi-VN", "Vietnamese"),
    ("la", "Latin"),
]

# Apple Vision-only codes (queried at runtime) that aren't in the editorial
# catalogue above — give them friendly names for the chip tooltips / picker.
LANGUAGES += [
    ("yue-Hans", "Cantonese (Simplified)"),
    ("yue-Hant", "Cantonese (Traditional)"),
    ("ars-SA", "Najdi Arabic"),
]

# Whole-ISO discovery: augment the curated (region-specific) catalogue above
# with the common world languages by ISO 639 code, named via langcodes. This is
# the completer list only — the input accepts ANY valid BCP-47 tag (see
# _parse_code), so no language is off-limits, and plugin VLMs / unknown
# languages on a known script are covered.
_COMMON_CODES = (
    "en fr es de it pt nl sv nb nn da fi is pl cs sk hu ro bg hr sr sl et lv lt "
    "mk sq el grc la ru uk be kk ky uz az tk tt tg mn ka hy he ar fa ur ps ckb "
    "sd ug th lo km my bo si ta te kn ml hi mr ne sa bn as gu pa or am ti zh yue "
    "ja ko vi id ms tl jv su ceb sw zu xh yo ig ha so rw mg ht eo cy ga gd br oc "
    "ca eu gl af fo lb mt gv kw tr"
).split()


def _augment_catalogue(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = {c for c, _ in items}
    try:
        import langcodes
        for code in _COMMON_CODES:
            if code in seen:
                continue
            try:
                name = langcodes.Language.get(code).display_name()
            except Exception:
                name = code
            items.append((code, name))
            seen.add(code)
    except Exception:
        pass
    return items


LANGUAGES = _augment_catalogue(LANGUAGES)
CODE_TO_NAME = {c: n for c, n in LANGUAGES}
NAME_TO_CODE = {n.lower(): c for c, n in LANGUAGES}


class LanguageTagInput(QWidget):
    """One-line tag picker. `QLineEdit` with a completer; selecting an
    entry (or pressing Enter on a matching string) pushes a chip into the
    row left of the input.

    The completer model contains both ``"code"`` and ``"code — name"``
    strings, so the user can type either and the chip stores the canonical
    code.
    """

    tags_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tags: list[str] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._input = QLineEdit()
        self._input.setPlaceholderText(self.tr("Add language… (fr-FR, French)"))
        self._input.setClearButtonEnabled(True)
        self._input.setMinimumWidth(240)

        model_items = [f"{code} — {name}" for code, name in LANGUAGES]
        completer = QCompleter(self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setModel(QStringListModel(model_items, self))
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.activated[str].connect(self._on_completer_picked)
        self._input.setCompleter(completer)
        self._input.returnPressed.connect(self._on_return)

        self._chip_host = QWidget()
        self._chip_row = FlowLayout(self._chip_host, margin=0,
                                     h_spacing=4, v_spacing=4)

        outer.addWidget(self._input)
        outer.addWidget(self._chip_host)

    def _on_completer_picked(self, text: str) -> None:
        code = self._parse_code(text)
        if code:
            self.add_tag(code)
        self._input.clear()

    def _on_return(self) -> None:
        raw = self._input.text().strip()
        if not raw:
            return
        code = self._parse_code(raw)
        if code:
            self.add_tag(code)
        self._input.clear()

    @staticmethod
    def _parse_code(text: str) -> Optional[str]:
        head = text.split("—", 1)[0].strip()
        if head in CODE_TO_NAME:
            return head
        if head.lower() in NAME_TO_CODE:
            return NAME_TO_CODE[head.lower()]
        if "—" in text:
            tail = text.split("—", 1)[1].strip()
            if tail in CODE_TO_NAME:
                return tail
            if tail.lower() in NAME_TO_CODE:
                return NAME_TO_CODE[tail.lower()]
        # Whole-ISO coverage: accept ANY well-formed BCP-47 / ISO 639 tag
        # (fr-FR, sr-Cyrl, zh-Hant, grc, …), not just the completer catalogue —
        # but reject junk (langcodes validates against the registry).
        try:
            import langcodes
            if langcodes.tag_is_valid(head):
                return head
        except Exception:
            # langcodes unavailable — fall back to the lenient shape check.
            if 2 <= len(head) <= 3 and head.isalpha():
                return head.lower()
        return None

    def set_allowed_languages(self, codes: list[str]) -> None:
        """Restrict the completer to ``codes`` (e.g. Apple Vision's actual
        supported set). Friendly names come from the static catalogue,
        falling back to the code. No-op on an empty list (keep the full
        catalogue) so non-macOS still shows something."""
        if not codes:
            return
        pairs = [(c, CODE_TO_NAME.get(c, c)) for c in codes]
        items = [f"{c} — {n}" for c, n in pairs]
        comp = self._input.completer()
        if comp is not None and comp.model() is not None:
            comp.model().setStringList(items)

    def add_tag(self, code: str) -> None:
        if code in self._tags:
            return
        self._tags.append(code)
        chip = self._build_chip(code)
        self._chip_row.insertWidget(len(self._tags) - 1, chip)
        self._chip_host.update()
        self.tags_changed.emit()

    def remove_tag(self, code: str) -> None:
        if code not in self._tags:
            return
        self._tags.remove(code)
        while self._chip_row.count():
            item = self._chip_row.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        for i, c in enumerate(self._tags):
            self._chip_row.insertWidget(i, self._build_chip(c))
        self._chip_host.update()
        self.tags_changed.emit()

    def _build_chip(self, code: str) -> QWidget:
        chip = QFrame()
        chip.setObjectName("LangChip")
        chip.setStyleSheet(
            "QFrame#LangChip {"
            f"  background-color: {COLOR_PRIMARY_BG_SOFT};"
            f"  border: 1px solid {COLOR_PRIMARY_BORDER};"
            "  border-radius: 8px;"
            "}"
            f"QFrame#LangChip QLabel {{ color: {COLOR_FONT_ON_BUTTON}; padding: 0; }}"
            "QFrame#LangChip QToolButton {"
            f"  border: none; background: transparent; color: {COLOR_FONT_MUTED};"
            "  padding: 0 2px;"
            "}"
            f"QFrame#LangChip QToolButton:hover {{ color: {COLOR_FONT_INVERSE}; }}"
        )
        row = QHBoxLayout(chip)
        row.setContentsMargins(6, 1, 4, 1)
        row.setSpacing(2)
        lbl = QLabel(code)
        lbl.setToolTip(CODE_TO_NAME.get(code, code))
        row.addWidget(lbl)
        close = QToolButton()
        close.setText("×")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(lambda _, c=code: self.remove_tag(c))
        row.addWidget(close)
        chip.setSizePolicy(QSizePolicy.Policy.Maximum,
                            QSizePolicy.Policy.Maximum)
        return chip

    def tags(self) -> list[str]:
        return list(self._tags)

    def set_tags(self, codes: list[str]) -> None:
        while self._chip_row.count():
            item = self._chip_row.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._tags.clear()
        for c in codes:
            self.add_tag(c)
