# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""VS Code-style vertical activity bar.

Thin icon strip on the right edge of MainWindow. Top-of-bar holds a
discreet collapse/expand toggle, followed by tab icons, a vertical
stretch, and the bottom block (tip + settings). Click on a tab icon
emits ``activated(name)``; click the active icon collapses (emits
``collapse_toggled(True)``); click the dedicated toggle emits
``collapse_toggled(...)`` directly.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from lib.gui.colors import (
    COLOR_ERROR_STRONG,
    COLOR_FONT_PLACEHOLDER,
    COLOR_FONT_SECTION_LABEL,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
)
from lib.gui.theme import icon as theme_icon

# Strip width — matches design doc.
BAR_WIDTH = 48
# Tab buttons.
TAB_BTN = 40
# Toggle button (smaller; chrome control, not a tab).
TOGGLE_BTN = 32
# Left-edge accent strip width (active tab indicator).
ACCENT_W = 3
# Single accent — keep in sync with RadioCardGroup so the sidebar
# never paints two different "selected" blues at once.
ACCENT_COLOR = COLOR_PRIMARY
TOGGLE_COLOR = COLOR_FONT_SECTION_LABEL


class _SideTooltip:
    """Mixin: anchor the tooltip to the LEFT of the widget (vertically
    centred), not at the cursor. The activity bar sits on the window's
    RIGHT edge, so a right- or cursor-anchored tooltip spills off-screen;
    placing it just inside the left edge keeps it on the window. The left
    offset is the tooltip text width (in the tooltip font) so its right
    edge lands beside the icon."""

    def event(self, e):  # noqa: N802 — Qt override
        if e.type() == QEvent.Type.ToolTip:
            tip = self.toolTip()
            if tip:
                from PySide6.QtGui import QFontMetrics
                tip_w = QFontMetrics(QToolTip.font()).horizontalAdvance(tip)
                gp = self.mapToGlobal(
                    QPoint(-tip_w - 24, self.height() // 2 - 10))
                QToolTip.showText(gp, tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)


class _ActivityButton(_SideTooltip, QToolButton):
    """One tab button — left-edge accent paints when active."""

    def __init__(self, name: str, icon_name: str, tooltip: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name = name
        self._icon_name = icon_name
        self._active = False
        self.setToolTip(tooltip)
        self.setFixedSize(TAB_BTN, TAB_BTN)
        self.setIconSize(QSize(22, 22))
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_icon()

    @property
    def name(self) -> str:
        return self._name

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self._refresh_icon()
        self.update()

    def is_active(self) -> bool:
        return self._active

    def _refresh_icon(self) -> None:
        # Active = brand accent (palette-aware). Inverse white worked on
        # dark mode but vanished into the light-mode page bg, hence the
        # switch to ``COLOR_PRIMARY`` which contrasts both ways.
        if self._active:
            color = COLOR_PRIMARY
        else:
            color = COLOR_FONT_PLACEHOLDER
        self.setIcon(theme_icon(self._icon_name, color=color, size=22))

    def paintEvent(self, event) -> None:  # noqa: D401 — Qt override
        super().paintEvent(event)
        if not self._active:
            return
        p = QPainter(self)
        p.fillRect(0, 4, ACCENT_W, self.height() - 8, QColor(ACCENT_COLOR))
        p.end()


class _ToggleButton(_SideTooltip, QToolButton):
    """Collapse/expand chrome control — distinct from tab icons."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(TOGGLE_BTN, TOGGLE_BTN)
        self.setIconSize(QSize(18, 18))
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapsed = False
        self._refresh()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._refresh()

    def _refresh(self) -> None:
        name = "panel-right-open" if self._collapsed else "panel-right-close"
        self.setIcon(theme_icon(name, color=TOGGLE_COLOR, size=18))
        self.setToolTip(
            self.tr("Expand sidebar") if self._collapsed
            else self.tr("Collapse sidebar")
        )


class _TipButton(_SideTooltip, QWidget):
    """Vertical heart-glyph + 'Tip' label, rosy-red glow, used in the
    centred mid block of the activity bar. Slot for the bottom-bar tip
    widget that previously lived in StatusBarWidget."""

    def __init__(self, on_click, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(self.tr("Tip the developer"))
        self.setFixedWidth(BAR_WIDTH)

        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        accent = COLOR_ERROR_STRONG

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(2)
        v.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        icon = QLabel()
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            ic = theme_icon("heart", color=accent, size=20)
            icon.setPixmap(ic.pixmap(20, 20))
        except Exception:
            icon.setText("♥")
            icon.setStyleSheet(f"color:{accent}; font-size:18px;")
        icon.setStyleSheet(
            (icon.styleSheet() or "")
            + "background: transparent; border: none;"
        )
        glow = QGraphicsDropShadowEffect(icon)
        glow.setOffset(0, 0)
        glow.setBlurRadius(40)
        glow.setColor(QColor(accent))
        icon.setGraphicsEffect(glow)
        v.addWidget(icon)

        text = QLabel(self.tr("Tip"))
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setStyleSheet(
            f"color:{accent}; font-size:10px; font-weight:700;"
            "background: transparent; border: none;"
        )
        text_glow = QGraphicsDropShadowEffect(text)
        text_glow.setOffset(0, 0)
        text_glow.setBlurRadius(24)
        text_glow.setColor(QColor(accent))
        text.setGraphicsEffect(text_glow)
        v.addWidget(text)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton and self._on_click:
            try:
                self._on_click()
            except Exception:
                pass
        super().mouseReleaseEvent(ev)


class _HairlineDivider(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(
            f"background: {COLOR_OUTLINE_SUBTLE}; border: none;"
        )


class ActivityBar(QWidget):
    """Vertical icon strip — activity tabs + bottom toolbox.

    Signals:
      * ``activated(name)`` — user clicked a tab icon (not the active one).
      * ``collapse_toggled(collapsed: bool)`` — collapsed state changed.
    """

    activated = Signal(str)
    collapse_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ActivityBar")
        self.setFixedWidth(BAR_WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Transparent so the bar inherits the window/tab background.
        # Any extra tint here produces a visible seam against the
        # adjacent content stack.
        self.setStyleSheet(
            "QWidget#ActivityBar { background: transparent; }"
        )

        self._buttons: dict[str, _ActivityButton] = {}
        self._active: Optional[str] = None
        self._collapsed = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 6, 0, 6)
        self._layout.setSpacing(2)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Toggle on top.
        self._toggle = _ToggleButton(self)
        self._toggle.clicked.connect(self._on_toggle_clicked)
        self._wrap_centered(self._toggle)

        self._divider = _HairlineDivider(self)
        self._layout.addWidget(self._divider)
        self._layout.addSpacing(4)

        # Slot for tab buttons — keep a stretch at the end for the
        # bottom block.
        self._tab_container = QWidget(self)
        self._tab_layout = QVBoxLayout(self._tab_container)
        self._tab_layout.setContentsMargins(0, 0, 0, 0)
        self._tab_layout.setSpacing(2)
        self._tab_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._layout.addWidget(self._tab_container)

        self._layout.addStretch(1)

        # Mid block — the Tip heart, centred between the top tabs and the
        # bottom toolbox (equal stretch above and below).
        self._mid_container = QWidget(self)
        self._mid_layout = QVBoxLayout(self._mid_container)
        self._mid_layout.setContentsMargins(0, 0, 0, 0)
        self._mid_layout.setSpacing(2)
        self._mid_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._layout.addWidget(self._mid_container)

        self._layout.addStretch(1)

        self._bottom_container = QWidget(self)
        self._bottom_layout = QVBoxLayout(self._bottom_container)
        self._bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._bottom_layout.setSpacing(2)
        self._bottom_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._layout.addWidget(self._bottom_container)

    # ── public API ──────────────────────────────────────────────────

    def add_tip_button(self, on_click: Callable[[], None]) -> None:
        """Rosy-red heart + 'Tip' label, centred between the top tabs and
        the bottom toolbox."""
        btn = _TipButton(on_click, self._mid_container)
        self._add_centered(btn, self._mid_layout)

    def add_activity(self, name: str, icon_name: str, tooltip: str,
                     *, bottom: bool = False,
                     on_click: Optional[Callable[[], None]] = None) -> None:
        """Append one button. ``bottom=True`` puts it under the stretch."""
        btn = _ActivityButton(name, icon_name, tooltip, self)
        btn.clicked.connect(lambda: self._on_tab_clicked(name, on_click))
        self._buttons[name] = btn

        target_layout = self._bottom_layout if bottom else self._tab_layout
        self._add_centered(btn, target_layout)

    def set_active(self, name: Optional[str]) -> None:
        if self._active == name:
            return
        self._active = name
        for n, btn in self._buttons.items():
            btn.set_active(n == name)

    def active(self) -> Optional[str]:
        return self._active

    def set_collapsed(self, collapsed: bool) -> None:
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self._toggle.set_collapsed(collapsed)

    def collapsed(self) -> bool:
        return self._collapsed

    # ── internals ───────────────────────────────────────────────────

    def _wrap_centered(self, w: QWidget) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(w)
        row.addStretch(1)
        self._layout.addLayout(row)

    def _add_centered(self, w: QWidget, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)

    def _on_toggle_clicked(self) -> None:
        self.set_collapsed(not self._collapsed)
        self.collapse_toggled.emit(self._collapsed)

    def _on_tab_clicked(self, name: str,
                        on_click: Optional[Callable[[], None]]) -> None:
        # Bottom items (tip / settings) have a side effect rather than
        # swapping the content pane — invoke their callback and stop.
        if on_click is not None:
            on_click()
            return

        # True tab — toggle collapse if clicking active, else activate.
        if self._active == name and not self._collapsed:
            self.set_collapsed(True)
            self.collapse_toggled.emit(True)
            return
        if self._collapsed:
            self.set_collapsed(False)
            self.collapse_toggled.emit(False)
        self.set_active(name)
        self.activated.emit(name)
