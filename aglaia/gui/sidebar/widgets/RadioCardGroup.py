# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Card-style radio group.

A vertical (or grid) stack of selectable cards. Each card has:

* an optional left-side icon (lucide name),
* a bold title + one-line description,
* a radio indicator on the right (filled when active),
* an optional expandable inner panel that becomes visible only while
  the card is selected — meant for "additional parameters" tied to
  that choice (e.g. Surya engine settings, compression-specific knobs).

Selected state: blue outline (``#4f46e5``) + filled radio dot, the
same idiom as the Spotify/Lovable-style picker the user referenced.
Hover on unselected cards lifts the background slightly.

Usage::

    grp = RadioCardGroup()
    grp.add_card("apple_vision", "Apple Vision",
                 "Native macOS OCR. Fast, no model download.",
                 icon_name="scan-text")
    surya_extras = QWidget()  # any QWidget — shown when surya is picked
    grp.add_card("surya", "Surya",
                 "VLM-based. Structure-aware MD export.",
                 icon_name="scan-text",
                 extras=surya_extras)
    grp.set_current_key("apple_vision")
    grp.currentChanged.connect(lambda key: ...)

The widget emits ``currentChanged(key: str)`` on every selection swap
(both programmatic and user-driven).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_INPUT_FOCUS,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_BG_SURFACE_ALT,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_BUTTON,
    COLOR_OUTLINE_GHOST,
    COLOR_PRIMARY,
)


class _ElidedLabel(QLabel):
    """A word-wrapped label clamped to ``max_lines`` lines, ending in an
    ellipsis when the text overflows. The full text is always available as
    the tooltip, so nothing is lost — just compacted in the card."""

    def __init__(self, text: str, *, max_lines: int = 2, parent=None) -> None:
        super().__init__(parent)
        self._full = text
        self._max_lines = max_lines
        self.setWordWrap(True)
        self.setToolTip(text)
        super().setText(text)

    def _fits(self, txt: str, w: int, max_h: int) -> bool:
        from PySide6.QtCore import Qt as _Qt
        r = self.fontMetrics().boundingRect(
            0, 0, w, 1_000_000, _Qt.TextFlag.TextWordWrap, txt)
        return r.height() <= max_h

    def _relayout(self) -> None:
        w = self.width()
        if w <= 0 or not self._full:
            return
        fm = self.fontMetrics()
        max_h = fm.lineSpacing() * self._max_lines + 2
        if self._fits(self._full, w, max_h):
            if self.text() != self._full:
                super().setText(self._full)
            return
        # Binary-search the longest prefix that fits in max_lines + "…".
        lo, hi, best = 0, len(self._full), "…"
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = self._full[:mid].rstrip() + "…"
            if self._fits(cand, w, max_h):
                best, lo = cand, mid + 1
            else:
                hi = mid - 1
        super().setText(best)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._relayout()


# Tokens — kept tight so callers don't have to think about colors.
# Accent matches ``ActivityBar.ACCENT_COLOR`` so the entire sidebar
# uses one blue for every "selected" affordance.
_ACCENT = COLOR_PRIMARY
_BORDER_IDLE = COLOR_OUTLINE_GHOST
_BG_IDLE = COLOR_BG_OVERLAY_SOFT
_BG_HOVER = COLOR_BG_SURFACE_ALT
_BG_ACTIVE = COLOR_BG_INPUT_FOCUS
_RADIO_OUTER = 16
_RADIO_INNER = 8
_ICON_BOX = 32


class _RadioDot(QWidget):
    """Hand-drawn radio glyph — donut ring + filled dot when active."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._active = False
        self.setFixedSize(_RADIO_OUTER + 2, _RADIO_OUTER + 2)

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self.width() / 2
        cy = self.height() / 2
        outer_r = _RADIO_OUTER / 2

        if self._active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(_ACCENT))
            p.drawEllipse(int(cx - outer_r), int(cy - outer_r),
                          _RADIO_OUTER, _RADIO_OUTER)
            inner_r = _RADIO_INNER / 2
            p.setBrush(QColor(COLOR_FONT_INVERSE))
            p.drawEllipse(int(cx - inner_r), int(cy - inner_r),
                          _RADIO_INNER, _RADIO_INNER)
        else:
            pen = p.pen()
            pen.setColor(QColor(COLOR_OUTLINE_BUTTON))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx - outer_r) + 1, int(cy - outer_r) + 1,
                          _RADIO_OUTER - 2, _RADIO_OUTER - 2)
        p.end()


@dataclass
class _Card:
    key: str
    frame: QFrame
    radio: _RadioDot
    extras: Optional[QWidget]
    extras_always_visible: bool = False


class _CardFrame(QFrame):
    """Click-to-select QFrame. Forwards clicks to the parent group."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("RadioCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._apply_style(False)

    def set_selected(self, selected: bool) -> None:
        self._apply_style(selected)

    def _apply_style(self, selected: bool) -> None:
        if selected:
            qss = (
                "QFrame#RadioCard {"
                f"  background: {_BG_ACTIVE};"
                f"  border: 1.5px solid {_ACCENT};"
                "  border-radius: 10px;"
                "}"
            )
        else:
            qss = (
                "QFrame#RadioCard {"
                f"  background: {_BG_IDLE};"
                f"  border: 1px solid {_BORDER_IDLE};"
                "  border-radius: 10px;"
                "}"
                "QFrame#RadioCard:hover {"
                f"  background: {_BG_HOVER};"
                "}"
            )
        self.setStyleSheet(qss)

    def mousePressEvent(self, ev) -> None:  # noqa: N802 — Qt API
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class RadioCardGroup(QWidget):
    """Vertical stack of card-style radio choices."""

    currentChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 *, orientation: str = "vertical") -> None:
        super().__init__(parent)
        self._cards: dict[str, _Card] = {}
        self._current: Optional[str] = None
        self._order: list[str] = []
        # Orientation controls the outer layout direction. Horizontal is
        # used by the StartupWindow's source picker (capture / files
        # side-by-side); vertical is the default for sidebar usage where
        # cards stack down the panel.
        if orientation == "horizontal":
            self._layout = QHBoxLayout(self)
        else:
            self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

    # ── public API ──────────────────────────────────────────────────

    def add_card(self, key: str, title: str, description: str = "",
                 *, icon_name: Optional[str] = None,
                 title_badges: Optional[list[str]] = None,
                 extras: Optional[QWidget] = None,
                 extras_always_visible: bool = False,
                 enabled: bool = True) -> None:
        """Append one card.

        ``extras`` (any QWidget) is hidden by default and shown when the
        card becomes the active selection. Use it to host "additional
        parameters" tied to this choice.

        ``title_badges`` (list of Lucide icon names) is rendered as a
        row of small mono-tinted icons right after the title — used for
        speed / accuracy cues on the engine cards.
        """
        frame = _CardFrame(self)
        if not enabled:
            frame.setEnabled(False)
            frame.setCursor(Qt.CursorShape.ArrowCursor)
        # Outer layout: header row + (optional) extras row.
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(10)

        if icon_name:
            icon_box = QLabel()
            icon_box.setFixedSize(_ICON_BOX, _ICON_BOX)
            icon_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_box.setStyleSheet(
                "QLabel {"
                f"  background: {COLOR_BG_OVERLAY_SOFT};"
                "  border-radius: 6px;"
                "}"
            )
            try:
                from aglaia.gui.theme import icon as _icon
                ic = _icon(icon_name, color=COLOR_FONT_MUTED, size=18)
                icon_box.setPixmap(ic.pixmap(18, 18))
            except Exception:
                pass
            header.addWidget(icon_box)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; "
            "font-weight: 700; font-size: 13px;"
        )
        title_row.addWidget(title_lbl)
        if title_badges:
            try:
                from aglaia.gui.theme import icon as _theme_icon
            except Exception:
                _theme_icon = None
            for badge in title_badges:
                # Accept ``"icon"`` (default slate tint) OR
                # ``("icon", "#hexcolor")`` so per-badge colors carry
                # semantic meaning (gold / silver / bronze medals).
                if isinstance(badge, (tuple, list)) and len(badge) == 2:
                    badge_name, badge_color = badge
                else:
                    # Default badge tint is the primary text color (solid
                    # hex) rather than ``COLOR_FONT_MUTED``: lucide SVGs
                    # set their stroke via the ``color=`` arg which is
                    # interpolated into the SVG attribute, and many SVG
                    # renderers reject ``rgba(...)`` there — turtle /
                    # rabbit went invisible on light mode as a result.
                    badge_name, badge_color = badge, COLOR_FONT_PRIMARY
                badge_lbl = QLabel()
                badge_lbl.setFixedSize(18, 18)
                badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                badge_lbl.setStyleSheet("background: transparent;")
                if _theme_icon is not None:
                    try:
                        ic = _theme_icon(
                            badge_name, color=badge_color, size=16,
                        )
                        badge_lbl.setPixmap(ic.pixmap(16, 16))
                    except Exception:
                        pass
                title_row.addWidget(badge_lbl)
        title_row.addStretch(1)
        text_col.addLayout(title_row)
        if description:
            # Clamp to 2 lines + ellipsis; full text lives in the tooltip.
            desc_lbl = _ElidedLabel(description, max_lines=2)
            desc_lbl.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: 11px;"
            )
            text_col.addWidget(desc_lbl)
        header.addLayout(text_col, 1)

        radio = _RadioDot(frame)
        header.addWidget(radio, 0, Qt.AlignmentFlag.AlignTop)

        outer.addLayout(header)

        if extras is not None:
            extras.setParent(frame)
            extras.setVisible(extras_always_visible)
            # Kill the extras host's own bg only (not descendants — combo
            # / drop-zone / inputs need their painted bg). qdarktheme
            # gives every bare QWidget a palette(window) fill, which
            # paints over the card's QSS bg and produces a pale strip at
            # the bottom of the active card. Translucent + explicit
            # `background: transparent` on the QWidget tag-selector
            # leaves child controls untouched.
            extras.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            outer.addWidget(extras)

        # Let the frame grow vertically so the description QLabel's
        # word-wrap is honoured — a Fixed policy would clip multi-line
        # descriptions (Surya's 3-line tagline was getting cut off).
        frame.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)

        card = _Card(key=key, frame=frame, radio=radio, extras=extras,
                       extras_always_visible=extras_always_visible)
        self._cards[key] = card
        self._order.append(key)
        self._layout.addWidget(frame)
        frame.clicked.connect(lambda k=key: self._select(k))

    def set_current_key(self, key: Optional[str]) -> bool:
        if key is not None and key not in self._cards:
            return False
        self._select(key, emit=False)
        return True

    def current_key(self) -> Optional[str]:
        return self._current

    def set_card_description(self, key: str, description: str) -> None:
        """Useful for the "(not installed)" suffix when state changes."""
        # The description widget is the 2nd child of the text column;
        # walk the frame's children to find the QLabel matching it. To
        # keep the API simple we just re-create the card if needed; for
        # now this is intentionally unimplemented and callers re-add.
        raise NotImplementedError("rebuild the card instead")

    def set_card_enabled(self, key: str, enabled: bool) -> None:
        c = self._cards.get(key)
        if c is None:
            return
        c.frame.setEnabled(enabled)
        c.frame.setCursor(
            Qt.CursorShape.PointingHandCursor if enabled
            else Qt.CursorShape.ArrowCursor
        )

    def set_card_tooltip(self, key: str, tooltip: str) -> None:
        """Set (or clear, with "") the hover tooltip on a card — used to
        explain why a disabled card can't be picked."""
        c = self._cards.get(key)
        if c is None:
            return
        c.frame.setToolTip(tooltip)

    def keys(self) -> list[str]:
        return list(self._order)

    # ── internals ───────────────────────────────────────────────────

    def _select(self, key: Optional[str], *, emit: bool = True) -> None:
        if key == self._current:
            return
        self._current = key
        for k, c in self._cards.items():
            active = (k == key)
            c.frame.set_selected(active)
            c.radio.set_active(active)
            if c.extras is not None:
                c.extras.setVisible(active or c.extras_always_visible)
        if emit and key is not None:
            self.currentChanged.emit(key)
