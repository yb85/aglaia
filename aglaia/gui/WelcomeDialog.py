# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""First-run welcome / permissions screen.

Shown once, the first time the GUI launches (gated by the ``welcome_seen``
config flag). Its job is to set expectations *before* macOS pops its own
camera / keychain prompts — a notarized-but-unknown app asking for the system
password is intimidating without context. We explain exactly when and why each
prompt appears, and stress that nothing leaves the machine and the keychain is
touched only if the user opts into Cloud OCR."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


_INTRO = (
    "Aglaïa turns photos of book and document pages into clean, "
    "<b>searchable PDFs and Markdown</b>."
)

# (emoji, title, body) — kept plain so it reads the same in any theme.
_ROWS = [
    ("📷", "Permission to use your camera and microphone",
     "The camera captures pages with your webcam; the microphone powers "
     "optional hands-free voice capture. macOS asks the first time you use "
     "each — or skip both and just import existing images and PDFs."),
    ("🔑", "Permission to use your keychain",
     "Asked <b>only if you choose to save a Cloud OCR API key</b>. The key is "
     "kept in your macOS Keychain, and macOS asks your password to protect it "
     "— not Aglaïa. Everything else, including offline OCR, works without it."),
    ("📁", "Permission to access local files",
     "Projects, settings and models live under "
     "<i>~/Library/Application Support/Aglaia</i>. Pages stay on your Mac "
     "unless you explicitly pick a cloud OCR engine."),
]


class WelcomeDialog(QDialog):
    """One-shot first-run welcome. Use :meth:`show_if_first_run`."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Welcome to Aglaïa"))
        self.setModal(True)
        self.setMinimumWidth(520)

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 22, 24, 18)
        v.setSpacing(14)

        title = QLabel(self.tr("Welcome to Aglaïa"))
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        v.addWidget(title)

        intro = QLabel(self.tr(_INTRO))
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        v.addWidget(intro)

        why = QLabel(self.tr("Why do we ask for permissions?"))
        why.setStyleSheet("font-size: 16px; font-weight: 600;")
        why.setContentsMargins(0, 6, 0, 0)
        v.addWidget(why)

        for emoji, head, body in _ROWS:
            v.addWidget(self._permission_row(emoji, self.tr(head), self.tr(body)))

        foot = QLabel(self.tr(
            "Source-available, signed &amp; notarized, built by CI. "
            "You stay in control of every permission."))
        foot.setTextFormat(Qt.TextFormat.RichText)
        foot.setWordWrap(True)
        foot.setStyleSheet("font-size: 11px; color: palette(mid);")
        v.addWidget(foot)

        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton(self.tr("Get started"))
        btn.setDefault(True)
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        v.addLayout(row)

    def _permission_row(self, emoji: str, head: str, body: str) -> QWidget:
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
    def show_if_first_run(cls, parent: Optional[QWidget] = None) -> None:
        """Show the dialog once, then never again. Any failure (no DB, etc.)
        is swallowed — a missing welcome screen must never block launch."""
        try:
            from aglaia.app_data import db
            with db.session() as conn:
                if db.get(conn, db.KEY_WELCOME_SEEN, False):
                    return
                cls(parent).exec()
                db.set(conn, db.KEY_WELCOME_SEEN, True)
        except Exception:
            pass
