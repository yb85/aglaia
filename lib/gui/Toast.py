# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Toast — non-blocking, fade-in/out status notification.

Anchored to the bottom-right of the host widget by default. The toast
auto-positions on every show + on host `resizeEvent`. New calls to
`show_message` cancel any in-flight animation and reuse the same
widget — only one toast on screen at a time, so a fast sequence of
events doesn't pile up.

Use it for "the action succeeded, but there's no other visual cue":
saving settings, applying pipeline changes, exporting a PDF, etc. The
button/tab closes; the toast confirms what just happened.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, Qt, QTimer,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

from lib.gui.colors import (
    COLOR_BG_TOAST_QCOLOR,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_GHOST,
)


class Toast(QLabel):
    """One-shot toast notification overlay.

    Lifecycle: `show_message(text, duration_ms)` →
      fade-in (200 ms) → hold (duration_ms) → fade-out (400 ms) → hide.
    A second call while the toast is visible cancels the current
    animation and restarts at full opacity with the new text.
    """

    FADE_IN_MS = 200
    FADE_OUT_MS = 400
    DEFAULT_DURATION_MS = 2000
    MARGIN_PX = 24

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "QLabel#Toast {"
            f"  background-color: {COLOR_BG_TOAST_QCOLOR};"
            # Theme-aware: the toast bg follows the theme (dark on dark,
            # light on light), so the text must too — COLOR_FONT_INVERSE was
            # hardcoded white and vanished on the light-theme toast.
            f"  color: {COLOR_FONT_PRIMARY};"
            "  padding: 10px 18px;"
            "  border-radius: 10px;"
            f"  border: 1px solid {COLOR_OUTLINE_GHOST};"
            "}"
        )
        font = self.font()
        font.setPointSize(font.pointSize() + 1)
        font.setWeight(QFont.Weight.Medium)
        self.setFont(font)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fx = QGraphicsOpacityEffect(self)
        self._fx.setOpacity(0.0)
        self.setGraphicsEffect(self._fx)
        self._fade = QPropertyAnimation(self._fx, b"opacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Track whether the fade-out → hide chain is currently armed so we
        # can disconnect cleanly. PySide6 emits a RuntimeWarning on
        # disconnect() with no slots; try/except doesn't catch warnings.
        self._hide_armed = False
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.timeout.connect(self._fade_out)
        self.hide()

    def show_message(self, text: str, duration_ms: int | None = None) -> None:
        if not text:
            return
        self.setText(text)
        self.adjustSize()
        self._position_self()
        self._fade.stop()
        # Drop any lingering finished→hide from a previous fade-out so the
        # fade-in doesn't auto-hide on completion.
        if self._hide_armed:
            self._fade.finished.disconnect(self.hide)
            self._hide_armed = False
        self._fade.setDuration(self.FADE_IN_MS)
        self._fade.setStartValue(self._fx.opacity())
        self._fade.setEndValue(1.0)
        self.show()
        self.raise_()
        self._fade.start()
        self._hold.start(duration_ms or self.DEFAULT_DURATION_MS)

    def _fade_out(self) -> None:
        self._fade.stop()
        self._fade.setDuration(self.FADE_OUT_MS)
        self._fade.setStartValue(self._fx.opacity())
        self._fade.setEndValue(0.0)
        if self._hide_armed:
            self._fade.finished.disconnect(self.hide)
        self._fade.finished.connect(self.hide)
        self._hide_armed = True
        self._fade.start()

    def _position_self(self) -> None:
        p = self.parentWidget()
        if p is None:
            return
        x = p.width() - self.width() - self.MARGIN_PX
        y = p.height() - self.height() - self.MARGIN_PX
        self.move(max(0, x), max(0, y))

    def reposition(self) -> None:
        """Public hook for the host to call from its `resizeEvent`."""
        if self.isVisible():
            self._position_self()
