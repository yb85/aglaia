# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Scrim + spinner + caption painted over a panel while something runs.

Lived inside OcrTab until the Export tab needed the same thing: a send to a
Kindle or a calibre server takes as long as the far end takes, and a button
that does nothing visible for forty seconds reads as a broken button. Sharing
the widget is also what keeps "something is happening" looking the same
everywhere — the alpha and the frames match `_SpinnerOverlay` in
ScanItemWidget for exactly that reason.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from aglaia.gui.colors import COLOR_FONT_INVERSE, COLOR_PRIMARY, qcolor


class BusyOverlay(QWidget):
    """Scrim + spinner + caption painted on top of OcrTab while a long
    op (OCR run / pipeline run) is in flight. Blocks input — the
    underlying controls are also ``setEnabled(False)`` for keyboard
    focus + a11y, but the overlay communicates the lock visually."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # Eats clicks (no WA_TransparentForMouseEvents) — disabled
        # widgets underneath wouldn't react anyway, but blocking here
        # avoids misleading hover cues.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._idx = 0
        self._caption = self.tr("Working…")
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self.hide()
        if parent is not None:
            parent.installEventFilter(self)

    def set_caption(self, text: str) -> None:
        self._caption = text
        self.update()

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self.show()
        self.raise_()
        self._resize_to_parent()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._idx = (self._idx + 1) % len(self._FRAMES)
        self.update()

    def _resize_to_parent(self) -> None:
        p = self.parentWidget()
        if p is not None:
            self.setGeometry(0, 0, p.width(), p.height())

    def eventFilter(self, obj, ev):  # noqa: N802 — Qt API
        if obj is self.parentWidget() and ev.type() in (
            ev.Type.Resize, ev.Type.Show, ev.Type.Move,
        ):
            self._resize_to_parent()
        return False

    def paintEvent(self, _ev):  # noqa: N802
        self._resize_to_parent()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Match `_SpinnerOverlay` (ScanItemWidget): same scrim alpha so
        # the busy state reads identically across the app.
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        font = QFont()
        font.setPixelSize(40)
        font.setBold(True)
        p.setFont(font)
        p.setPen(qcolor(COLOR_PRIMARY))
        spinner_rect = self.rect().adjusted(0, 0, 0, -32)
        p.drawText(spinner_rect, int(Qt.AlignmentFlag.AlignCenter),
                   self._FRAMES[self._idx])
        font.setPixelSize(13)
        font.setBold(True)
        p.setFont(font)
        p.setPen(qcolor(COLOR_FONT_INVERSE))
        caption_rect = self.rect().adjusted(0, 32, 0, 0)
        p.drawText(caption_rect, int(Qt.AlignmentFlag.AlignCenter),
                   self._caption)
        p.end()
