# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""iOS-style toggle switch.

QAbstractButton subclass — drop-in replacement for QCheckBox where a
softer, label-on-the-right pill is wanted (sidebar options, settings
toggles). The pill track + thumb are hand-painted so colors track the
active palette and the geometry stays crisp at any DPI.

Layout: pill on the left (40×22 px), label on the right (font weight
500). Click anywhere on the widget toggles. Setting ``checked`` runs a
short slide animation; ``setChecked`` without animation is also fine
(checked changes apply on the next paint).

API mirrors QCheckBox closely so callers can swap one for the other:

    self.chk = ToggleSwitch("Use JBIG2 for monochrome")
    self.chk.setChecked(True)
    self.chk.toggled.connect(handler)
    if self.chk.isChecked(): ...
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
)
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy

from aglaia.gui.colors import (
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_BUTTON,
    COLOR_PRIMARY,
    qcolor,
)


_TRACK_W = 36
_TRACK_H = 20
_THUMB_PAD = 2
_GAP = 8  # space between pill and label


class ToggleSwitch(QAbstractButton):
    """Pill toggle. Same signal surface as QCheckBox."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # Inherit card background instead of the qdarktheme QAbstractButton
        # default (white fill on light theme leaks behind the label).
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent; border: none;")
        self.setText(text)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Fixed)
        self._thumb_pos = float(_THUMB_PAD)
        self._anim = QPropertyAnimation(self, b"thumb_pos", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._on_toggled)

    # ── thumb_pos animated property ────────────────────────────────

    def get_thumb_pos(self) -> float:
        return self._thumb_pos

    def set_thumb_pos(self, val: float) -> None:
        self._thumb_pos = float(val)
        self.update()

    thumb_pos = Property(float, get_thumb_pos, set_thumb_pos)

    # ── geometry ────────────────────────────────────────────────────

    def _thumb_off_x(self) -> float:
        return float(_THUMB_PAD)

    def _thumb_on_x(self) -> float:
        return float(_TRACK_W - _TRACK_H + _THUMB_PAD)

    def sizeHint(self) -> QSize:  # noqa: N802
        fm = QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(self.text()) if self.text() else 0
        w = _TRACK_W + (_GAP + text_w if text_w else 0)
        h = max(_TRACK_H, fm.height())
        return QSize(w, h + 4)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    # ── state plumbing ──────────────────────────────────────────────

    def _on_toggled(self, checked: bool) -> None:
        target = self._thumb_on_x() if checked else self._thumb_off_x()
        if not self.isVisible():
            # Skip animation when the widget hasn't been shown yet — the
            # initial state shouldn't appear to "slide in".
            self._thumb_pos = target
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(self._thumb_pos)
        self._anim.setEndValue(target)
        self._anim.start()

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(checked)
        # Scan thumb position when set programmatically before first show.
        if not self.isVisible():
            self._thumb_pos = (
                self._thumb_on_x() if checked else self._thumb_off_x()
            )

    # ── painting ────────────────────────────────────────────────────

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        checked = self.isChecked()

        # Vertical centre the pill in the widget rect.
        cy = self.height() / 2
        track_top = cy - _TRACK_H / 2
        track = QRectF(0, track_top, _TRACK_W, _TRACK_H)

        if checked:
            track_color = qcolor(COLOR_PRIMARY)
        else:
            track_color = qcolor(COLOR_OUTLINE_BUTTON)
        if not enabled:
            track_color.setAlphaF(0.35)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track_color)
        p.drawRoundedRect(track, _TRACK_H / 2, _TRACK_H / 2)

        # Thumb — white circle.
        thumb_r = (_TRACK_H - 2 * _THUMB_PAD) / 2
        thumb_cx = self._thumb_pos + thumb_r
        thumb_cy = cy
        thumb_color = qcolor(COLOR_FONT_INVERSE)
        if not enabled:
            thumb_color.setAlphaF(0.85)
        p.setBrush(thumb_color)
        p.drawEllipse(
            QRectF(thumb_cx - thumb_r, thumb_cy - thumb_r,
                   thumb_r * 2, thumb_r * 2)
        )

        # Label.
        text = self.text()
        if text:
            label_color = qcolor(
                COLOR_FONT_PRIMARY if enabled else COLOR_FONT_DIM
            )
            p.setPen(QPen(label_color))
            text_rect = QRectF(
                _TRACK_W + _GAP, 0,
                self.width() - _TRACK_W - _GAP, self.height(),
            )
            p.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                text,
            )
        p.end()

    def hitButton(self, pos) -> bool:  # noqa: N802
        return self.rect().contains(pos)
