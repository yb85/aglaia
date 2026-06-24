# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""First-run "install recommended models" invite.

Shown right after the welcome/permissions screen on every launch until the
models are present or the user ticks "don't show again". Off macOS the app
leans on two small offline models — EAST (page/text-region detection) and
Vosk (offline voice control); on macOS the Apple Vision backend covers
detection, so only Vosk is offered.

`maybe_prompt(parent)` returns the list of model keys the user chose to
install (so the caller can open the downloader and autostart them), or
``None`` when nothing should happen."""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


# (key, emoji, title, body) for every model the invite can offer.
_MODELS = {
    "east": ("🗺️", "EAST — page detection",
             "Finds the text region on each photo so pages crop cleanly. "
             "~95 MB."),
    "vosk_en": ("🎙️", "Vosk — voice control",
                "Offline, hands-free capture by voice. ~40 MB."),
}


def _target_keys() -> list[str]:
    # macOS gets page detection from Apple Vision → only Vosk is useful.
    return ["vosk_en"] if sys.platform == "darwin" else ["east", "vosk_en"]


class ModelInstallPrompt(QDialog):
    """First-run invite to fetch the recommended offline models."""

    def __init__(self, keys: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Recommended models"))
        self.setModal(True)
        self.setMinimumWidth(500)
        self._dont_show = False

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 22, 24, 18)
        v.setSpacing(14)

        title = QLabel(self.tr("Install recommended models"))
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        v.addWidget(title)

        intro = QLabel(self.tr(
            "Aglaïa runs everything offline. These small models unlock its "
            "best results — download them now in the background, or skip and "
            "grab them later from <i>View → Show Downloader</i>."))
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        v.addWidget(intro)

        for key in keys:
            emoji, head, body = _MODELS[key]
            v.addWidget(self._model_row(emoji, self.tr(head), self.tr(body)))

        self._chk = QCheckBox(self.tr("Don't show this again"))
        v.addWidget(self._chk)

        row = QHBoxLayout()
        later = QPushButton(self.tr("Maybe later"))
        later.clicked.connect(self.reject)
        row.addWidget(later)
        row.addStretch(1)
        install = QPushButton(self.tr("Install"))
        install.setDefault(True)
        install.clicked.connect(self.accept)
        row.addWidget(install)
        v.addLayout(row)

    def dont_show_again(self) -> bool:
        return self._chk.isChecked()

    def _model_row(self, emoji: str, head: str, body: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        icon = QLabel(emoji)
        icon.setStyleSheet("font-size: 22px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        icon.setFixedWidth(30)
        h.addWidget(icon)
        txt = QLabel(f"<b>{head}</b><br>{body}")
        txt.setWordWrap(True)
        txt.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(txt, 1)
        return w

    @classmethod
    def maybe_prompt(cls, parent: Optional[QWidget] = None) -> Optional[list[str]]:
        """Gate + show the invite. Returns the model keys to install if the
        user clicked Install, else ``None``. Any failure is swallowed — a
        missing invite must never block launch."""
        try:
            from aglaia.app_data import db
            from aglaia.gui.ModelDownloaderTab import is_model_installed
            with db.session() as conn:
                if db.get(conn, db.KEY_MODELS_PROMPT_DISMISSED, False):
                    return None
                missing = [k for k in _target_keys()
                           if not is_model_installed(k)]
                if not missing:
                    return None
                dlg = cls(missing, parent)
                accepted = dlg.exec() == QDialog.DialogCode.Accepted
                if dlg.dont_show_again():
                    db.set(conn, db.KEY_MODELS_PROMPT_DISMISSED, True)
                return missing if accepted else None
        except Exception:
            return None
