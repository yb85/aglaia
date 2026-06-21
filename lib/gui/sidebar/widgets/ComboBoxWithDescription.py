# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""QComboBox where each item shows a bold title + a one-line description.

Closed state still renders only the title (standard QComboBox lineEdit
text), so the popup is the only place the longer descriptions surface.

Usage::

    combo = ComboBoxWithDescription()
    combo.add_item("apple_vision", "Apple Vision",
                   "Native macOS OCR. Fast, no model download.")
    combo.add_item("surya", "Surya",
                   "VLM-based. Handwriting + 90+ scripts.")
    combo.set_current_key("apple_vision")
    key = combo.current_key()

Designed for ``≤5`` items — anything bigger and the popup gets unwieldy.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QModelIndex, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)


_TITLE_ROLE = Qt.ItemDataRole.UserRole + 1
_DESC_ROLE = Qt.ItemDataRole.UserRole + 2
_KEY_ROLE = Qt.ItemDataRole.UserRole + 3


class _TwoLineDelegate(QStyledItemDelegate):
    """Paints title (bold 13px) over description (10px, muted)."""

    _ROW_PAD_V = 6
    _ROW_PAD_H = 10
    _GAP = 2

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        painter.save()

        style = opt.widget.style() if opt.widget else None
        if style is not None:
            style.drawPrimitive(
                QStyle.PrimitiveElement.PE_PanelItemViewItem,
                opt, painter, opt.widget,
            )

        title = index.data(_TITLE_ROLE) or index.data(Qt.ItemDataRole.DisplayRole) or ""
        desc = index.data(_DESC_ROLE) or ""

        rect = opt.rect.adjusted(self._ROW_PAD_H, self._ROW_PAD_V,
                                 -self._ROW_PAD_H, -self._ROW_PAD_V)

        palette = opt.palette
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        title_color = palette.color(
            QPalette.ColorRole.HighlightedText if selected
            else QPalette.ColorRole.Text
        )
        desc_color = QColor(title_color)
        desc_color.setAlphaF(0.55 if not selected else 0.75)

        title_font = QFont(opt.font)
        title_font.setBold(True)
        title_font.setPointSize(max(title_font.pointSize(), 11))
        fm_title = painter.fontMetrics()
        painter.setFont(title_font)
        fm_title = painter.fontMetrics()

        title_h = fm_title.height()
        title_rect = rect.adjusted(0, 0, 0, -(rect.height() - title_h))

        painter.setPen(title_color)
        painter.drawText(
            title_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            fm_title.elidedText(str(title), Qt.TextElideMode.ElideRight,
                                title_rect.width()),
        )

        if desc:
            desc_font = QFont(opt.font)
            desc_font.setPointSize(max(desc_font.pointSize() - 2, 8))
            painter.setFont(desc_font)
            fm_desc = painter.fontMetrics()

            desc_rect = rect.adjusted(0, title_h + self._GAP, 0, 0)
            painter.setPen(desc_color)
            painter.drawText(
                desc_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                fm_desc.elidedText(str(desc), Qt.TextElideMode.ElideRight,
                                   desc_rect.width()),
            )

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem,
                 index: QModelIndex) -> QSize:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        title_font = QFont(opt.font)
        title_font.setBold(True)
        title_font.setPointSize(max(title_font.pointSize(), 11))
        desc_font = QFont(opt.font)
        desc_font.setPointSize(max(desc_font.pointSize() - 2, 8))

        from PySide6.QtGui import QFontMetrics
        fm_title = QFontMetrics(title_font)
        fm_desc = QFontMetrics(desc_font)

        desc = index.data(_DESC_ROLE) or ""
        h = fm_title.height() + (self._GAP + fm_desc.height() if desc else 0)
        h += 2 * self._ROW_PAD_V
        return QSize(max(opt.rect.width(), 240), h)


class ComboBoxWithDescription(QComboBox):
    """QComboBox whose popup rows show ``title + description``."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        view = QListView(self)
        view.setUniformItemSizes(False)
        view.setSpacing(0)
        self.setView(view)
        self._delegate = _TwoLineDelegate(view)
        view.setItemDelegate(self._delegate)
        self.setMinimumHeight(32)

    # ── public API ──────────────────────────────────────────────────

    def add_item(self, key: str, title: str, description: str = "") -> None:
        """Append one entry. ``key`` is what ``current_key()`` returns."""
        self.addItem(title)
        idx = self.count() - 1
        self.setItemData(idx, key, _KEY_ROLE)
        self.setItemData(idx, title, _TITLE_ROLE)
        self.setItemData(idx, description, _DESC_ROLE)

    def set_description(self, index: int, description: str) -> None:
        if 0 <= index < self.count():
            self.setItemData(index, description, _DESC_ROLE)

    def set_current_key(self, key: str) -> bool:
        for i in range(self.count()):
            if self.itemData(i, _KEY_ROLE) == key:
                self.setCurrentIndex(i)
                return True
        return False

    def current_key(self) -> Optional[str]:
        i = self.currentIndex()
        if i < 0:
            return None
        return self.itemData(i, _KEY_ROLE)

    def key_at(self, index: int) -> Optional[str]:
        if 0 <= index < self.count():
            return self.itemData(index, _KEY_ROLE)
        return None
