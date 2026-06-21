# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Wrapping flow layout — arranges child widgets left→right, wrapping to a
new row when horizontal space runs out. Cards keep their sizeHint width
(zoom-based) and the parent stretches vertically as rows accumulate."""

from PySide6.QtCore import Qt, QRect, QPoint, QSize, Signal
from PySide6.QtGui import QPainter, QColor, QPalette
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidgetItem, QWidget, QApplication


class FlowContentWidget(QWidget):
    """QScrollArea host that delegates heightForWidth to its FlowLayout.

    Without this, a QScrollArea + FlowLayout pairs the layout against a
    fixed-height widget — rows never wrap correctly because the layout's
    `heightForWidth` recomputation is ignored. This widget enables HFW
    sizing so the scroll content grows vertically as rows accumulate.

    Also handles drag-and-drop card reordering: emits `card_dropped`
    (scan_id, target_index) when a card is released over the grid.
    """

    card_dropped = Signal(int, int)

    DRAG_MIME = "application/x-aglaia-scan-id"

    def __init__(self, parent=None):
        super().__init__(parent)
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setAcceptDrops(True)
        # Vertical drop-target line painted during drag-hover; None otherwise.
        self._drop_indicator: QRect | None = None

    def hasHeightForWidth(self) -> bool:
        layout = self.layout()
        return layout is not None and layout.hasHeightForWidth()

    def heightForWidth(self, w: int) -> int:
        layout = self.layout()
        return layout.heightForWidth(w) if layout is not None else 0

    # ── drag and drop ─────────────────────────────────────────────────
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(self.DRAG_MIME):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasFormat(self.DRAG_MIME):
            ev.acceptProposedAction()
            pos = ev.position().toPoint()
            idx = self._index_for_point(pos)
            self._set_indicator(self._indicator_rect_for_index(idx))
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev):
        self._set_indicator(None)
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev):
        mime = ev.mimeData()
        if not mime.hasFormat(self.DRAG_MIME):
            ev.ignore()
            return
        try:
            scan_id = int(bytes(mime.data(self.DRAG_MIME)).decode())
        except Exception:
            ev.ignore()
            return
        pos = ev.position().toPoint()
        target_idx = self._index_for_point(pos)
        self._set_indicator(None)
        ev.acceptProposedAction()
        self.card_dropped.emit(scan_id, target_idx)

    def _set_indicator(self, rect):
        if rect == self._drop_indicator:
            return
        old = self._drop_indicator
        self._drop_indicator = rect
        # Repaint old + new bounding union so we don't smear stale lines.
        for r in (old, rect):
            if r is not None:
                self.update(r.adjusted(-2, -2, 2, 2))

    def _indicator_rect_for_index(self, idx: int) -> QRect | None:
        """Map an insertion index → vertical line rect to paint. Sits
        flush between the previous and the next card in the row, spanning
        the row's height."""
        layout = self.layout()
        if layout is None or layout.count() == 0:
            return None
        n = layout.count()
        if idx <= 0:
            r = layout.itemAt(0).geometry()
            x = r.left() - 4
            return QRect(x, r.top(), 2, r.height())
        if idx >= n:
            r = layout.itemAt(n - 1).geometry()
            x = r.right() + 2
            return QRect(x, r.top(), 2, r.height())
        r_prev = layout.itemAt(idx - 1).geometry()
        r_next = layout.itemAt(idx).geometry()
        if r_prev.top() == r_next.top():
            # Same row → between them.
            x = (r_prev.right() + r_next.left()) // 2 - 1
            return QRect(x, r_prev.top(), 2, r_prev.height())
        # Row break: paint to the right of the last card of the previous row.
        x = r_prev.right() + 2
        return QRect(x, r_prev.top(), 2, r_prev.height())

    def paintEvent(self, ev):  # noqa: N802 — Qt API
        super().paintEvent(ev)
        if self._drop_indicator is None:
            return
        p = QPainter(self)
        col: QColor = QApplication.palette().color(QPalette.ColorRole.Text)
        p.fillRect(self._drop_indicator, col)
        p.end()

    def _index_for_point(self, pos: QPoint) -> int:
        """Map a drop point in widget coordinates to an insertion index in
        the FlowLayout. Uses center-x within the row that contains the
        drop's y, so left-half = before, right-half = after."""
        layout = self.layout()
        if layout is None:
            return 0
        n = layout.count()
        if n == 0:
            return 0
        # Pick the row whose vertical extent contains pos.y (or the
        # nearest one if we're between rows).
        row_y = None
        row_items: list[tuple[int, QRect]] = []
        best_row_dist = None
        for i in range(n):
            item = layout.itemAt(i)
            r = item.geometry()
            dist = 0 if r.top() <= pos.y() <= r.bottom() else (
                r.top() - pos.y() if pos.y() < r.top() else pos.y() - r.bottom()
            )
            if best_row_dist is None or dist < best_row_dist:
                best_row_dist = dist
                row_y = r.top()
        for i in range(n):
            item = layout.itemAt(i)
            r = item.geometry()
            if r.top() == row_y:
                row_items.append((i, r))
        # Within the row, drop before the first card whose center.x > pos.x.
        for idx, r in row_items:
            if pos.x() < r.center().x():
                return idx
        # Past the last card on this row → insert after it (index = last+1).
        return row_items[-1][0] + 1 if row_items else n


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin: int = 0,
                 h_spacing: int = 10, v_spacing: int = 10):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._h_space = h_spacing
        self._v_space = v_spacing
        self._items: list = []

    def __del__(self):
        while self.count():
            self.takeAt(0)

    def addItem(self, item):
        self._items.append(item)

    def insertWidget(self, index: int, widget):
        self.addChildWidget(widget)
        item = QWidgetItem(widget)
        if index < 0 or index > len(self._items):
            index = len(self._items)
        self._items.insert(index, item)
        self.invalidate()

    def removeWidget(self, widget):
        for i, item in enumerate(self._items):
            if item.widget() is widget:
                self._items.pop(i)
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent; with no parent that's a bare
                # top-level window flash.
                widget.hide()
                widget.setParent(None)
                self.invalidate()
                return

    def horizontalSpacing(self) -> int:
        return self._h_space

    def verticalSpacing(self) -> int:
        return self._v_space

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        # Ignore child minimums — propagating them would force the scroll
        # host (and window) wider than needed. Height comes from `heightForWidth`.
        m = self.contentsMargins()
        return QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = eff.x()
        y = eff.y()
        line_h = 0
        for item in self._items:
            w = item.widget()
            sp_x = self._h_space
            sp_y = self._v_space
            hint = item.sizeHint()
            next_x = x + hint.width() + sp_x
            if next_x - sp_x > eff.right() and line_h > 0:
                x = eff.x()
                y = y + line_h + sp_y
                next_x = x + hint.width() + sp_x
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()
