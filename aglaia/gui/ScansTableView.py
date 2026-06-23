# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Compact tabular view of scans.

Each scan renders as ONE table-style row, single line:
  [ idx + raw filestem + OCR badge ] [ layout sub-rows ] [ trash ]

Each cell in the layout sub-rows is a fixed-width 50-px mini-thumb
center-cropped to its slot, height controlled by `_thumb_h`. Multiple
layouts (branches) of the same scan stack as multiple sub-rows inside
the central column.

Click on a thumb: pops a 3-action menu [Trash layout | Debug | Select].
The trash action dims the entire row (a layout = all stages share the
same fate).

Global height-scaling knob: `set_thumb_height(int)` rebuilds with the
new cell height; the host wires it to the same slider that resizes
grid cards (50-600 mapped to thumb height).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QRect, Qt, Signal, QSize
from PySide6.QtGui import QCursor, QDrag, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMenu,
    QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG,
    COLOR_BG_ZEBRA_ODD,
    COLOR_ERROR,
    COLOR_ERROR_BG_SOFT,
    COLOR_FONT_ON_BUTTON,
    COLOR_FONT_PLACEHOLDER,
    COLOR_OUTLINE_GHOST,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
)

# Same MIME as the grid view — drop receivers in either view can resolve
# the dragged scan id without redundant constants.
SCAN_DRAG_MIME = "application/x-aglaia-scan-id"


CELL_W = 50                  # fixed thumb width — center-cropped to fit
DEFAULT_THUMB_H = 32         # default thumb height
MIN_THUMB_H = 16
MAX_THUMB_H = 96
ROW_V_SPACING = 2
SELECTED_BORDER = COLOR_PRIMARY


def _palette_row_colors() -> tuple[str, str]:
    """Return (even, odd) row backgrounds.

    Even row = `transparent` → inherits the scroll viewport bg.
    Odd row = palette-aware `COLOR_BG_ZEBRA_ODD` (#1d1d1d on dark,
    #e4e4e7 on light). Reads via the token instead of `QPalette.Base`
    because qdarktheme's Base on light theme matches the window bg —
    no contrast = no zebra."""
    return ("transparent", COLOR_BG_ZEBRA_ODD)


def _pix_fit_crop(blob: bytes, w: int, h: int) -> Optional[QPixmap]:
    """Decode + scale-to-fill + center-crop to exactly w × h."""
    if not blob:
        return None
    img = QImage.fromData(blob)
    if img.isNull() or img.width() == 0 or img.height() == 0:
        return None
    # Scale-to-fill (KeepAspectRatioByExpanding) then center-crop.
    scaled = img.scaled(w, h,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation)
    x = max(0, (scaled.width() - w) // 2)
    y = max(0, (scaled.height() - h) // 2)
    crop = scaled.copy(x, y, w, h)
    return QPixmap.fromImage(crop)


class _DragGrip(QWidget):
    """grip-vertical icon on the row's left edge — the only hit region
    that initiates a drag. Same visual idiom as `_DragHandle` in the grid
    card: source row dims to 0.35 during the drag, restores on drop/cancel.
    """

    def __init__(self, *, block: "_SnapBlock"):
        super().__init__(block)
        self._block = block
        self._press_pos: Optional[QPoint] = None
        self.setFixedSize(18, 28)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setToolTip(self.tr("Drag to reorder"))
        from aglaia.gui.theme import lucide_pixmap as _lp
        # Lines horizontal vs the grid card's vertical handle? Lucide's
        # `grip-vertical` glyph is two columns of dots — a universal
        # drag-handle motif. Reads the same in both views.
        pix = _lp("grip-vertical", color=COLOR_FONT_PLACEHOLDER, size=18)
        pix.setDevicePixelRatio(2.0)
        self._pix = pix

    def paintEvent(self, _ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        x = (self.width() - 18) // 2
        y = (self.height() - 18) // 2
        p.drawPixmap(QPoint(x, y), self._pix)
        p.end()

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position().toPoint()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        self._press_pos = None
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):  # noqa: N802
        if self._press_pos is None:
            return
        delta = ev.position().toPoint() - self._press_pos
        if delta.manhattanLength() < QApplication.startDragDistance():
            return
        pix = self._block.grab()
        max_h = 96
        if pix.height() > max_h:
            pix = pix.scaledToHeight(max_h, Qt.TransformationMode.SmoothTransformation)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(SCAN_DRAG_MIME, QByteArray(str(self._block.scan_id).encode()))
        drag.setMimeData(mime)
        drag.setPixmap(pix)
        drag.setHotSpot(QPoint(8, pix.height() // 2))
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._press_pos = None
        prev_effect = self._block.graphicsEffect()
        dim = QGraphicsOpacityEffect(self._block)
        dim.setOpacity(0.35)
        self._block.setGraphicsEffect(dim)
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self._block.setGraphicsEffect(prev_effect)


class _ThumbCell(QLabel):
    """One stage's mini-thumb. 50 × thumb_h fixed; outlined when this
    is the selected stage. Click toggles the step's per-page disable
    (toggleable steps only).

    Two independent overlays: when the row is `hidden`, a crossed-eye
    glyph reads "hidden by user"; when this step is `disabled`, a red
    diagonal strike reads "processor skipped for this page"."""

    clicked = Signal()

    def __init__(self, pix: Optional[QPixmap], *, selected: bool,
                 stem: str, step_name: str, node_id: Optional[int],
                 thumb_h: int, hidden: bool = False,
                 disabled: bool = False, toggleable: bool = True):
        super().__init__()
        self.stem = stem
        self.step_name = step_name
        self.node_id = node_id
        self._selected = selected
        self._hidden = bool(hidden)
        self._disabled = bool(disabled)
        self._toggleable = bool(toggleable)
        self.setFixedSize(CELL_W, thumb_h)
        if pix is not None:
            self.setPixmap(pix)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if toggleable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            tip = (self.tr("{stem} · {step} — click to re-enable")
                   if disabled else
                   self.tr("{stem} · {step} — click to disable this step"))
        else:
            tip = self.tr("{stem} · {step}")
        self.setToolTip(tip.format(stem=stem, step=step_name))
        self._restyle()

    def set_hidden(self, hidden: bool) -> None:
        if hidden == self._hidden:
            return
        self._hidden = bool(hidden)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        if self._disabled:
            # Red diagonal strike + faint red wash: "this step is skipped".
            from PySide6.QtGui import QColor as _QC, QPen as _QP
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            wash = _QC(COLOR_ERROR)
            wash.setAlpha(40)
            p.fillRect(self.rect(), wash)
            pen = _QP(_QC(COLOR_ERROR))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawLine(3, 3, self.width() - 3, self.height() - 3)
            p.end()
        if self._hidden:
            # Crossed-eye overlay so a dimmed row reads as "user hid" rather
            # than a dark scan.
            from aglaia.gui.theme import lucide_pixmap as _lp
            glyph_size = max(12, min(self.height(), self.width()) - 8)
            pix = _lp("eye-off", color=COLOR_FONT_ON_BUTTON, size=glyph_size)
            pix.setDevicePixelRatio(2.0)
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)
            p.setOpacity(0.75)
            x = (self.width() - glyph_size) // 2
            y = (self.height() - glyph_size) // 2
            p.drawPixmap(QPoint(x, y), pix)
            p.end()

    def _restyle(self) -> None:
        if self._disabled:
            self.setStyleSheet(
                f"QLabel{{border: 2px solid {COLOR_ERROR}; "
                "border-radius: 2px; background: transparent;}}")
        elif self._selected:
            self.setStyleSheet(
                f"QLabel{{border: 2px solid {SELECTED_BORDER}; "
                f"border-radius: 2px; background: transparent;}}")
        else:
            self.setStyleSheet(
                f"QLabel{{border: 1px solid {COLOR_OUTLINE_GHOST}; "
                "border-radius: 2px; background: transparent;}}")

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton and self._toggleable:
            self.clicked.emit()
        super().mousePressEvent(ev)


class _RowWidget(QWidget):
    """Per-layout sub-row: branch label, OCR badge slot, stage cells."""

    def __init__(self, *, stem: str, item: dict, raw_filestem: str,
                 ocr_state: str, thumb_loader, thumb_h: int,
                 global_history: Optional[list[str]] = None,
                 on_cell_toggle=None, cell_states: Optional[dict] = None,
                 on_toggle_visibility=None,
                 on_debug=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.stem = stem
        self._dimmed = bool(item.get("trashed", False))
        self._on_toggle_visibility = on_toggle_visibility
        self._on_debug = on_debug
        self._leaf_node_id: Optional[int] = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        # Per-row eye toggle — show/hide layout. Reserved trash icon for
        # scan-level destructive delete (scan block, right edge).
        from aglaia.gui.theme import lucide_pixmap as _lp
        self._eye_btn = QToolButton()
        self._eye_btn.setFixedSize(28, 28)
        self._eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._eye_btn.setStyleSheet(
            "QToolButton{background:transparent; border:none; padding:0; margin:0;}"
            f"QToolButton:hover{{background:{COLOR_OUTLINE_SUBTLE}; border-radius:4px;}}"
        )
        self._refresh_eye_icon()
        self._eye_btn.clicked.connect(self._on_eye_clicked)
        h.addWidget(self._eye_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        # Magnifier — opens the debug viewer on the row's leaf node.
        self._debug_btn = QToolButton()
        self._debug_btn.setFixedSize(24, 24)
        self._debug_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._debug_btn.setToolTip(self.tr("Open debug viewer for this page"))
        self._debug_btn.setStyleSheet(
            "QToolButton{background:transparent; border:none; padding:0; margin:0;}"
            f"QToolButton:hover{{background:{COLOR_OUTLINE_SUBTLE}; border-radius:4px;}}"
        )
        _pix = _lp("search", color=COLOR_FONT_PLACEHOLDER, size=18)
        _pix.setDevicePixelRatio(2.0)
        self._debug_btn.setIcon(_pix)
        self._debug_btn.setIconSize(QSize(18, 18))
        self._debug_btn.clicked.connect(self._on_debug_clicked)
        h.addWidget(self._debug_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        # Branch label.
        suffix = stem[len(raw_filestem) + 1:] if stem.startswith(raw_filestem + "_") else stem
        lbl = QLabel(suffix or "·")
        lbl.setStyleSheet(f"color:{COLOR_FONT_PLACEHOLDER}; font-size:11px;")
        lbl.setFixedWidth(20)
        h.addWidget(lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        # OCR badge slot.
        self._ocr_lbl = QLabel()
        self._ocr_lbl.setFixedSize(14, 14)
        self._ocr_lbl.setStyleSheet("background:transparent; border:none;")
        self._apply_ocr_badge(ocr_state)
        h.addWidget(self._ocr_lbl)
        # Stage cells.
        self._cells: list[_ThumbCell] = []
        history = item.get("history", [])
        nodes = item.get("nodes", {})
        # `current_idx` is an index into the *global* pipeline history
        # (raw + every pipeline step) — not into this stem's local
        # `history` list. Translate so the blue "selected" frame tracks
        # the grid widget's actual current step.
        cur_idx = int(item.get("current_idx", 0))
        if global_history and 0 <= cur_idx < len(global_history):
            current_step = global_history[cur_idx]
        elif history:
            current_step = history[-1]
        else:
            current_step = ""
        cell_states = cell_states or {}
        for step in history:
            node_info = nodes.get(step) or {}
            image_id = node_info.get("image_id")
            nid = node_info.get("node_id")
            tog, dis = (cell_states.get(int(nid), (False, False))
                        if nid is not None else (False, False))
            pix = None
            if image_id is not None:
                # Pull a 256-px thumb (already cached in DB) — plenty
                # of source resolution to center-crop down to 50 × thumb_h.
                blob = thumb_loader(image_id, 256)
                pix = _pix_fit_crop(blob, CELL_W, thumb_h) if blob else None
            sel = ((step == current_step)
                   or (history and step == history[-1] and not current_step))
            cell = _ThumbCell(pix, selected=bool(sel), stem=stem,
                              step_name=step, node_id=nid,
                              thumb_h=thumb_h, hidden=self._dimmed,
                              disabled=bool(dis), toggleable=bool(tog))
            # Click = toggle this step's per-page disable (toggleable steps
            # only). Debug lives on the magnifier button, trash on the eye.
            if tog and nid is not None and on_cell_toggle is not None:
                cell.clicked.connect(lambda _nid=int(nid): on_cell_toggle(_nid))
            # Last cell's node id = the leaf the debug button targets.
            if cell.node_id is not None:
                self._leaf_node_id = int(cell.node_id)
            h.addWidget(cell)
            self._cells.append(cell)
        h.addStretch(1)
        if self._dimmed:
            self._restyle()

    def _apply_ocr_badge(self, state: str) -> None:
        from aglaia.gui.theme import lucide_pixmap as _lp
        if state not in ("fresh", "stale"):
            self._ocr_lbl.clear()
            self._ocr_lbl.setToolTip("")
            return
        color = COLOR_SUCCESS if state == "fresh" else COLOR_WARNING
        # lucide_pixmap renders at 2× for HiDPI sharpness (size=14 → 28-px
        # pixmap). Tag DPR so the 14×14 QLabel paints it as 14 logical px
        # instead of cropping to the top-left corner.
        pix = _lp("scan-text", color=color, size=14)
        pix.setDevicePixelRatio(2.0)
        self._ocr_lbl.setPixmap(pix)
        self._ocr_lbl.setToolTip(
            self.tr("OCR up to date") if state == "fresh"
            else self.tr("OCR stale — selected page changed")
        )

    def is_dimmed(self) -> bool:
        return self._dimmed

    def set_dimmed(self, dim: bool) -> None:
        if dim == self._dimmed:
            return
        self._dimmed = dim
        self._refresh_eye_icon()
        self._restyle()

    def _refresh_eye_icon(self) -> None:
        from aglaia.gui.theme import lucide_pixmap as _lp
        glyph = "eye-off" if self._dimmed else "eye"
        pix = _lp(glyph, color=COLOR_FONT_PLACEHOLDER, size=22)
        pix.setDevicePixelRatio(2.0)
        self._eye_btn.setIcon(pix)
        self._eye_btn.setIconSize(QSize(22, 22))
        self._eye_btn.setToolTip(self.tr("Show page") if self._dimmed else self.tr("Hide page"))

    def _on_eye_clicked(self) -> None:
        self.set_dimmed(not self._dimmed)
        if self._on_toggle_visibility is not None:
            self._on_toggle_visibility(self.stem, self._dimmed)

    def _on_debug_clicked(self) -> None:
        if self._on_debug is None or self._leaf_node_id is None:
            return
        self._on_debug(int(self._leaf_node_id), self.stem)

    def _restyle(self) -> None:
        # `opacity` is NOT a valid Qt QSS property — setting it via stylesheet
        # is silently ignored AND spams "Could not parse stylesheet of object
        # _ThumbCell" for every child cell. Use a graphics effect instead.
        fx = self.graphicsEffect()
        if not isinstance(fx, QGraphicsOpacityEffect):
            fx = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(fx)
        fx.setOpacity(0.45 if self._dimmed else 1.0)
        for cell in self._cells:
            cell.setEnabled(not self._dimmed)
            cell.set_hidden(self._dimmed)


class _SnapBlock(QFrame):
    """One scan → single horizontal table row.

    Layout: [ title block | layouts column | trash btn ].
    """

    delete_requested = Signal(int)
    debug_requested = Signal(int, str)
    step_toggle_requested = Signal(int, int)   # scan_id, node_id
    trash_requested = Signal(int, str, bool)

    def __init__(self, *, scan_id: int, idx: int, raw_filestem: str,
                 items: dict, thumb_loader, thumb_h: int,
                 global_history: Optional[list[str]] = None,
                 ocr_state: str = "none",
                 ocr_branch_state: Optional[dict] = None,
                 cell_states: Optional[dict] = None,
                 bg: str = COLOR_BG, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.scan_id = scan_id
        self.raw_filestem = raw_filestem
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Block bg + every nested QWidget inherits transparent so the row
        # frame's colour shows through the title block, sub-rows, and the
        # branch label slot — no more darker rectangles inside lighter rows.
        self._bg = bg
        self._apply_bg()
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 4, 12, 4)
        h.setSpacing(8)
        # Leftmost: drag grip — the ONLY region that initiates a row drag.
        self._grip = _DragGrip(block=self)
        h.addWidget(self._grip, 0, Qt.AlignmentFlag.AlignVCenter)
        # Title + scan-level OCR badge.
        h.addWidget(self._title_widget(idx, raw_filestem, ocr_state), 0,
                    Qt.AlignmentFlag.AlignVCenter)
        # Middle: vertically-stacked layout rows.
        self._rows: dict[str, _RowWidget] = {}
        ocr_branch_state = ocr_branch_state or {}
        rows_host = QWidget()
        rv = QVBoxLayout(rows_host)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(ROW_V_SPACING)
        for stem in self._ordered_stems(items, raw_filestem):
            branch_path = self._stem_to_branch_path(stem, raw_filestem)
            # Single-layout scans: the DB may record `branch_path = ""`
            # (no branching event fired) even though the stem carries an
            # `_A` suffix. Fall through to the empty-key entry in that
            # case so the OCR badge still resolves.
            row_state = (ocr_branch_state.get(branch_path)
                         or ocr_branch_state.get("")
                         or "none")
            row = _RowWidget(
                stem=stem, item=items[stem],
                raw_filestem=raw_filestem,
                ocr_state=row_state,
                thumb_loader=thumb_loader,
                thumb_h=thumb_h,
                global_history=global_history,
                on_cell_toggle=self._on_cell_toggled,
                cell_states=cell_states,
                on_toggle_visibility=self._on_row_visibility_toggled,
                on_debug=self._on_row_debug_requested,
                parent=rows_host,
            )
            self._rows[stem] = row
            rv.addWidget(row)
        h.addWidget(rows_host, 1, Qt.AlignmentFlag.AlignVCenter)
        # Right: trash button.
        h.addWidget(self._trash_widget(), 0,
                    Qt.AlignmentFlag.AlignVCenter)

    def _apply_bg(self) -> None:
        self.setStyleSheet(
            f"_SnapBlock{{background:{self._bg}; border:none;}} "
            f"_SnapBlock QWidget{{background:transparent;}}"
        )

    def set_row_bg(self, bg: str) -> None:
        """Cheap zebra re-stripe after an incremental insert/remove —
        restyles the row frame only, no thumbnail rebuild."""
        if bg == self._bg:
            return
        self._bg = bg
        self._apply_bg()

    @staticmethod
    def _stem_to_branch_path(stem: str, raw_filestem: str) -> str:
        if not stem.startswith(raw_filestem + "_"):
            return ""
        return stem[len(raw_filestem) + 1:].replace("_", ".")

    def _ordered_stems(self, items: dict, raw_filestem: str) -> list[str]:
        out = []
        for stem in sorted(items.keys()):
            if stem == raw_filestem:
                continue
            out.append(stem)
        if not out and raw_filestem in items:
            out.append(raw_filestem)
        return out

    def _title_widget(self, idx: int, raw_filestem: str, ocr_state: str) -> QWidget:
        # Scan-level OCR badge intentionally omitted — staleness is per
        # layout (per row) and a scan-level badge would lie when one
        # layout is fresh while another is stale.
        host = QWidget()
        h = QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        title = QLabel(self.tr("#{idx}  {stem}").format(idx=idx, stem=raw_filestem))
        title.setStyleSheet(f"color:{COLOR_FONT_ON_BUTTON}; font-weight:600; font-size:12px;")
        title.setFixedWidth(180)
        h.addWidget(title)
        return host

    def _trash_widget(self) -> QWidget:
        from aglaia.gui.theme import lucide_pixmap as _lp
        btn = QToolButton()
        btn.setIcon(QPixmap(_lp("trash-2", color=COLOR_ERROR, size=16)))
        btn.setIconSize(QSize(16, 16))
        btn.setFixedSize(24, 24)
        btn.setStyleSheet(
            "QToolButton{background:transparent; border:none;}"
            f"QToolButton:hover{{background:{COLOR_ERROR_BG_SOFT}; border-radius:4px;}}"
        )
        btn.setToolTip(self.tr("Discard scan"))
        btn.clicked.connect(lambda: self.delete_requested.emit(self.scan_id))
        return btn

    def _on_row_visibility_toggled(self, stem: str, dimmed: bool) -> None:
        """Per-row eye toggled → fire the same trash signal that the
        cell-menu uses so the host writes the same DB field. ``dimmed``
        is the NEW post-toggle state, propagated through so the host
        doesn't have to re-derive it from a DB query (which silently
        misses single-page scans whose branch row has a non-empty
        ``branch_path`` like ``A``)."""
        self.trash_requested.emit(self.scan_id, stem, bool(dimmed))

    def _on_cell_toggled(self, node_id: int) -> None:
        """Click on a (toggleable) stage cell = flip its per-page disable.
        Debug + trash are standalone buttons on the row."""
        self.step_toggle_requested.emit(self.scan_id, int(node_id))

    def _on_row_debug_requested(self, leaf_node_id: int, stem: str) -> None:
        self.debug_requested.emit(int(leaf_node_id), stem)


class _TableDropHost(QWidget):
    """Inner host widget for the table scroll area — accepts row drops,
    paints a horizontal indicator line at the target index, and fires
    `card_dropped(scan_id, target_idx)` on release. Mirrors the grid view's
    `FlowContentWidget` pattern with horizontal-band visuals instead of
    vertical-bar."""

    def __init__(self, dropped_signal: Signal, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._dropped = dropped_signal
        self.setAcceptDrops(True)
        self._indicator_y: Optional[int] = None

    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(SCAN_DRAG_MIME):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):  # noqa: N802
        if not ev.mimeData().hasFormat(SCAN_DRAG_MIME):
            ev.ignore()
            return
        ev.acceptProposedAction()
        y = ev.position().toPoint().y()
        self._set_indicator(self._indicator_y_for(y))

    def dragLeaveEvent(self, ev):  # noqa: N802
        self._set_indicator(None)
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev):  # noqa: N802
        mime = ev.mimeData()
        if not mime.hasFormat(SCAN_DRAG_MIME):
            ev.ignore()
            return
        try:
            scan_id = int(bytes(mime.data(SCAN_DRAG_MIME)).decode())
        except Exception:
            ev.ignore()
            return
        idx = self._index_for_y(ev.position().toPoint().y())
        self._set_indicator(None)
        ev.acceptProposedAction()
        self._dropped.emit(scan_id, idx)

    def _blocks(self) -> list[QWidget]:
        out = []
        lay = self.layout()
        if lay is None:
            return out
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if w is not None:
                out.append(w)
        return out

    def _index_for_y(self, y: int) -> int:
        # Drop above mid-row of block i → index i. Below mid-row of last
        # block → after all blocks.
        for i, w in enumerate(self._blocks()):
            top = w.y()
            mid = top + w.height() // 2
            if y < mid:
                return i
        return len(self._blocks())

    def _indicator_y_for(self, y: int) -> int:
        idx = self._index_for_y(y)
        blocks = self._blocks()
        if not blocks:
            return 0
        if idx >= len(blocks):
            last = blocks[-1]
            return last.y() + last.height()
        return blocks[idx].y()

    def _set_indicator(self, val: Optional[int]) -> None:
        if val == self._indicator_y:
            return
        self._indicator_y = val
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        if self._indicator_y is None:
            return
        p = QPainter(self)
        from PySide6.QtGui import QColor as _QC, QPen as _QP
        pen = _QP(_QC(COLOR_PRIMARY))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawLine(8, self._indicator_y, self.width() - 8, self._indicator_y)
        p.end()


class ScansTableView(QScrollArea):
    """Compact table view of all scans.

    Pulls live data from `get_snap_widgets`; rebuilds on `refresh()`.
    `set_thumb_height(int)` provides the global height scaling factor.
    """

    delete_requested = Signal(int)
    debug_requested = Signal(int, str)
    step_toggle_requested = Signal(int, int)   # scan_id, node_id
    trash_requested = Signal(int, str, bool)
    # scan_id, target_index — same shape as `FlowContentWidget.card_dropped`.
    card_dropped = Signal(int, int)

    def __init__(self, *, get_snap_widgets, thumb_loader,
                 ocr_state_provider=None, cell_states_provider=None,
                 thumb_h: int = DEFAULT_THUMB_H,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._get_snap_widgets = get_snap_widgets
        self._thumb_loader = thumb_loader
        self._ocr_state_provider = ocr_state_provider
        # scan_id → {node_id: (toggleable, disabled)} for the stage strip.
        self._cell_states_provider = cell_states_provider
        self._thumb_h = max(MIN_THUMB_H, min(MAX_THUMB_H, int(thumb_h)))
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._host = _TableDropHost(self.card_dropped)
        self._v = QVBoxLayout(self._host)
        self._v.setContentsMargins(8, 8, 8, 8)
        self._v.setSpacing(0)             # zero-gap rows for the table feel
        self._v.addStretch(1)
        self.setWidget(self._host)

    def set_thumb_height(self, h: int) -> None:
        h = max(MIN_THUMB_H, min(MAX_THUMB_H, int(h)))
        if h == self._thumb_h:
            return
        self._thumb_h = h
        if self.isVisible():
            self.refresh()

    def thumb_height(self) -> int:
        return self._thumb_h

    def _full_ocr(self) -> dict:
        if self._ocr_state_provider is None:
            return {}
        try:
            return self._ocr_state_provider() or {}
        except Exception:
            return {}

    def _cell_states(self, sid: int) -> dict:
        if self._cell_states_provider is None:
            return {}
        try:
            return self._cell_states_provider(int(sid)) or {}
        except Exception:
            return {}

    def _make_block(self, sid: int, w, full_ocr: dict, bg: str) -> "_SnapBlock":
        branch_state: dict[str, str] = {}
        for (msid, bp), info in full_ocr.items():
            if int(msid) == int(sid):
                branch_state[str(bp)] = str(info.get("state", "none"))
        block = _SnapBlock(
            scan_id=int(sid),
            idx=int(getattr(w, "idx", sid)),
            raw_filestem=str(getattr(w, "raw_filestem", f"scan-{sid}")),
            items=dict(getattr(w, "items", {})),
            thumb_loader=self._thumb_loader,
            thumb_h=self._thumb_h,
            global_history=list(getattr(w, "global_history", []) or []),
            ocr_state=str(getattr(w, "_ocr_state", "none")),
            ocr_branch_state=branch_state,
            cell_states=self._cell_states(sid),
            bg=bg,
            # Parent straight to the eventual host — a parentless QWidget
            # flashes as a bare top-level window on macOS between
            # construction and addWidget().
            parent=self._host,
        )
        block.delete_requested.connect(self.delete_requested)
        block.debug_requested.connect(self.debug_requested)
        block.step_toggle_requested.connect(self.step_toggle_requested)
        block.trash_requested.connect(self.trash_requested)
        return block

    def _snap_blocks(self) -> list["_SnapBlock"]:
        out: list[_SnapBlock] = []
        for i in range(self._v.count()):
            w = self._v.itemAt(i).widget()
            if isinstance(w, _SnapBlock):
                out.append(w)
        return out

    def _restripe(self) -> None:
        """Re-apply zebra backgrounds after an incremental add/remove.
        Cheap — restyles row frames only, no thumbnail rebuild."""
        bg_even, bg_odd = _palette_row_colors()
        for i, b in enumerate(self._snap_blocks()):
            b.set_row_bg(bg_even if (i % 2 == 0) else bg_odd)

    def refresh(self) -> None:
        while self._v.count() > 0:
            it = self._v.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        widgets = self._get_snap_widgets() or {}
        full_ocr = self._full_ocr()
        # Pull row backgrounds from the active palette so the alt color
        # tracks dark/light mode instead of being hardcoded.
        bg_even, bg_odd = _palette_row_colors()
        for i, (sid, w) in enumerate(sorted(widgets.items())):
            bg = bg_even if (i % 2 == 0) else bg_odd
            self._v.addWidget(self._make_block(sid, w, full_ocr, bg))
        self._v.addStretch(1)

    def remove_snap(self, scan_id: int) -> None:
        """Drop one scan's row in place — no full rebuild. Keeps a 300+
        scan table responsive on delete."""
        scan_id = int(scan_id)
        for b in self._snap_blocks():
            if b.scan_id == scan_id:
                self._v.removeWidget(b)
                b.deleteLater()
                self._restripe()
                return

    def add_snap(self, scan_id: int) -> None:
        """Insert one new scan's row at its sorted position — no full
        rebuild. Mirror of `remove_snap` for the capture/import path."""
        scan_id = int(scan_id)
        if any(b.scan_id == scan_id for b in self._snap_blocks()):
            return
        w = (self._get_snap_widgets() or {}).get(scan_id)
        if w is None:
            return
        block = self._make_block(scan_id, w, self._full_ocr(),
                                 _palette_row_colors()[0])
        # Insert before the first block with a higher scan_id (sorted order),
        # else just before the trailing stretch.
        insert_at = self._v.count() - 1
        for i in range(self._v.count()):
            wdg = self._v.itemAt(i).widget()
            if isinstance(wdg, _SnapBlock) and wdg.scan_id > scan_id:
                insert_at = i
                break
        self._v.insertWidget(insert_at, block)
        self._restripe()
