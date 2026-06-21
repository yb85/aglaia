# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Small custom widgets used by the modernized Qt UI.

Currently exposes:

  * `Switch` — animated toggle pill, drop-in replacement for QCheckBox
    where the boolean is a setting (persist, enabled, etc.). Painted
    manually because QSS can't draw the thumb circle reliably.
  * `Card` — rounded QFrame with `objectName == "Card"` so the QSS in
    `lib.gui.theme` styles it as an elevated container.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, Property,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QAbstractButton, QFrame, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from lib.gui.colors import (
    COLOR_FONT_INVERSE,
    COLOR_OUTLINE,
    COLOR_OUTLINE_BUTTON_STRONG,
    COLOR_PRIMARY,
    qcolor,
)


def make_icon_button(
    icon_name: str,
    *,
    size: int = 24,
    icon_size: int = 16,
    color: str = COLOR_OUTLINE_BUTTON_STRONG,
    bg: str = "transparent",
    hover_bg: Optional[str] = None,
    border: str = "none",
    parent: Optional[QWidget] = None,
    tooltip: Optional[str] = None,
) -> QPushButton:
    """Round-pill icon button. Replaces the hand-rolled QPushButton +
    setIcon + setIconSize + setFixedSize + setStyleSheet recipes
    scattered through ScanItemWidget."""
    from lib.gui.theme import lucide

    btn = QPushButton("", parent)
    btn.setIcon(lucide(icon_name, color=color, size=icon_size))
    btn.setIconSize(QSize(icon_size, icon_size))
    btn.setFixedSize(size, size)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    radius = size // 2
    hover_rule = (
        f"QPushButton:hover {{ background-color: {hover_bg}; }}"
        if hover_bg else ""
    )
    btn.setStyleSheet(
        f"QPushButton {{ background-color: {bg}; border: {border};"
        f"  border-radius: {radius}px; }}"
        f"{hover_rule}"
    )
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def place_overlay(btn: QPushButton, container: QWidget, x: int, y: int) -> None:
    """Reparent + position a button as an overlay on `container`. Used
    for the trash / nav buttons that float on top of thumbnails."""
    btn.setParent(container)
    btn.move(x, y)
    btn.show()


class Switch(QAbstractButton):
    """iOS/Material-style toggle pill.

    Inherits QAbstractButton.toggled(bool) — connect to it the same as
    QCheckBox. Animates thumb position when the state changes. Colors are
    exposed as Qt properties so the QSS in `lib.gui.theme` can theme them
    without touching this file.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._track_off = qcolor(COLOR_OUTLINE)
        self._track_on = QColor(COLOR_PRIMARY)
        self._thumb = QColor(COLOR_FONT_INVERSE)
        self._offset = 2.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._on_toggled)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(44, 24)

    # ── animated thumb position ─────────────────────────────────────
    def _on_toggled(self, on: bool):
        end = (self.width() - self.height() + 2) if on else 2.0
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(float(end))
        self._anim.start()

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, v: float):
        self._offset = float(v)
        self.update()

    offset = Property(float, fget=get_offset, fset=set_offset)

    # Theme-controlled colour properties
    def _get_track_off(self):
        return self._track_off

    def _set_track_off(self, c: QColor):
        self._track_off = QColor(c)
        self.update()

    trackOff = Property(QColor, fget=_get_track_off, fset=_set_track_off)

    def _get_track_on(self):
        return self._track_on

    def _set_track_on(self, c: QColor):
        self._track_on = QColor(c)
        self.update()

    trackOn = Property(QColor, fget=_get_track_on, fset=_set_track_on)

    def _get_thumb(self):
        return self._thumb

    def _set_thumb(self, c: QColor):
        self._thumb = QColor(c)
        self.update()

    thumb = Property(QColor, fget=_get_thumb, fset=_set_thumb)

    # ── paint ───────────────────────────────────────────────────────
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        radius = rect.height() / 2.0
        # Track
        track = QColor(self._track_on if self.isChecked() else self._track_off)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        p.fillPath(path, track)
        # Thumb (offset along x, slightly inset)
        thumb_size = rect.height() - 4
        x = self._offset
        thumb_rect = QRectF(x, 2, thumb_size, thumb_size)
        tpath = QPainterPath()
        tpath.addEllipse(thumb_rect)
        p.fillPath(tpath, self._thumb)
        p.end()

    def resizeEvent(self, ev):
        # Re-sync offset on size change so the thumb sits at the right edge.
        super().resizeEvent(ev)
        if not self._anim.state() == QPropertyAnimation.State.Running:
            self._offset = (self.width() - self.height() + 2) if self.isChecked() else 2.0
            self.update()


class Card(QFrame):
    """Rounded container styled via QSS in `lib.gui.theme` (selector
    `QFrame#Card`). Use for grouping a section of settings — pairs well
    with a `QLabel#SectionTitle` heading.

    Pass `elevated=True` for the slightly brighter background that signals
    the currently active card (e.g. the params panel).
    """

    def __init__(self, parent: QWidget | None = None, *, elevated: bool = False):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setProperty("elevated", "true" if elevated else "false")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._inner = QVBoxLayout(self)
        self._inner.setContentsMargins(14, 12, 14, 12)
        self._inner.setSpacing(8)

    def layout(self) -> QVBoxLayout:  # type: ignore[override]
        return self._inner
