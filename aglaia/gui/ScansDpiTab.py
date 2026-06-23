# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Fix input DPI — a sortable scans table for batch-correcting import DPI.

One row per active scan: checkbox · thumbnail · name · input DPI (editable) ·
source · import time. Multi-select (shift/ctrl-click) is the batch lever:
toggling a checkbox or editing the DPI of a selected row applies to every
selected row. "Apply & reprocess" writes the new DPI (raw `images.dpi` +
`scans.capture_dpi`) for the checked rows and reprocesses just those scans.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QSpinBox, QStyledItemDelegate, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from aglaia.gui.colors import COLOR_PRIMARY
from aglaia.gui.theme import lucide
from aglaia.storage.db import db_session

COL_CHECK, COL_THUMB, COL_NAME, COL_DPI, COL_SOURCE, COL_TIME = range(6)
_THUMB_PX = 44


class _DpiDelegate(QStyledItemDelegate):
    """Integer-DPI editor for the DPI column. A plain QSpinBox with an
    OPAQUE background filling the whole cell — the default editor let the
    cell's own text + pencil icon show through (the "doubled text" overlay)."""

    def createEditor(self, parent, option, index):  # noqa: N802
        sb = QSpinBox(parent)
        sb.setRange(1, 4000)
        sb.setAccelerated(True)
        sb.setAutoFillBackground(True)   # opaque → no bleed-through
        sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        return sb

    def setEditorData(self, editor, index):  # noqa: N802
        try:
            editor.setValue(int(round(float(
                index.data(Qt.ItemDataRole.EditRole) or 0))))
        except Exception:
            editor.setValue(0)

    def setModelData(self, editor, model, index):  # noqa: N802
        editor.interpretText()
        model.setData(index, float(editor.value()), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):  # noqa: N802
        # Fill the entire cell so nothing underneath peeks out.
        editor.setGeometry(option.rect)


def _format_source(source: str, source_ref: Optional[str]) -> str:
    ref = source_ref or ""
    if source == "capture":
        cam = ref.split("#", 1)[1] if "#" in ref else ""
        return f"capture #{cam}" if cam else "capture"
    if source == "pdf":
        path, _, page = ref.partition("#")
        name = Path(path).name or ref
        try:
            return f"pdf {name} p{int(page):03d}"
        except ValueError:
            return f"pdf {name}"
    if source == "import":
        return f"image {Path(ref).name or ref}"
    return source or "?"


def _format_time(created_at: Optional[str]) -> str:
    if not created_at:
        return ""
    # ISO8601 → "YYYY-MM-DD HH:MM" (drop seconds / tz noise for the column).
    s = str(created_at).replace("T", " ")
    return s[:16]


class ScansDpiTab(QWidget):
    """Batch input-DPI editor over all active scans."""

    def __init__(self, db_path: str,
                 thumb_loader: Callable[[int, int], Optional[bytes]],
                 reprocess_cb: Callable[[set], None],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db_path = str(db_path)
        self._thumb_loader = thumb_loader
        self._reprocess_cb = reprocess_cb
        self._updating = False  # reentrancy guard for itemChanged propagation

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel(self.tr("Fix input DPI"))
        title.setObjectName("SectionTitle")
        root.addWidget(title)

        # ── toolbar: select-all + apply ────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(10)
        self._select_all = QCheckBox(self.tr("Select all"))
        self._select_all.stateChanged.connect(self._on_select_all)
        bar.addWidget(self._select_all)
        bar.addStretch(1)
        self._apply_btn = QPushButton(self.tr("Set DPI and reprocess scans"))
        self._apply_btn.clicked.connect(self._on_apply)
        bar.addWidget(self._apply_btn)
        root.addLayout(bar)

        # Editable-cell affordance: a pencil tint shown on the DPI cells and
        # column header so it reads as "click to change me".
        self._edit_icon = lucide("pencil", color=COLOR_PRIMARY, size=13)

        # ── table ──────────────────────────────────────────────────────
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "", self.tr("Thumb"), self.tr("Name"), self.tr("Input DPI"),
            self.tr("Source"), self.tr("Imported"),
        ])
        # Cue the editable column in its header too.
        _dpi_hdr = self.table.horizontalHeaderItem(COL_DPI)
        _dpi_hdr.setIcon(self._edit_icon)
        _dpi_hdr.setToolTip(self.tr("Editable — click a cell to set the import DPI"))
        self.table.setAlternatingRowColors(True)  # zebra
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(QPixmap(_THUMB_PX, _THUMB_PX).size())
        self.table.setSortingEnabled(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeMode.Stretch)
        # The thumbnail column is not meaningfully sortable.
        hh.sectionClicked.connect(self._on_header_clicked)
        self.table.setItemDelegateForColumn(COL_DPI, _DpiDelegate(self.table))
        self.table.itemChanged.connect(self._on_item_changed)
        # Single click on a DPI cell opens its editor (no double-click needed).
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.table, 1)

        self.reload()

    # ── data ────────────────────────────────────────────────────────────
    def reload(self) -> None:
        self._updating = True
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        rows = self._query_rows()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.table.setRowHeight(r, _THUMB_PX + 8)

            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Unchecked)
            chk.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
            chk.setData(Qt.ItemDataRole.UserRole + 1, int(row["image_id"]))
            self.table.setItem(r, COL_CHECK, chk)

            thumb = QTableWidgetItem()
            thumb.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            pix = self._thumb_pixmap(int(row["image_id"]))
            if pix is not None:
                thumb.setIcon(QIcon(pix))
            self.table.setItem(r, COL_THUMB, thumb)

            name = QTableWidgetItem(str(row["filestem"] or f"scan-{row['id']}"))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, COL_NAME, name)

            dpi_val = float(row["img_dpi"] or row["capture_dpi"] or 0)
            dpi = QTableWidgetItem()
            dpi.setData(Qt.ItemDataRole.EditRole, dpi_val)  # numeric sort + edit
            dpi.setData(Qt.ItemDataRole.UserRole, dpi_val)  # original, to detect change
            dpi.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                         | Qt.ItemFlag.ItemIsEditable)
            # Editable cue: pencil icon + accent colour + tooltip.
            dpi.setIcon(self._edit_icon)
            dpi.setForeground(QColor(COLOR_PRIMARY))
            dpi.setToolTip(self.tr("Click to edit"))
            self.table.setItem(r, COL_DPI, dpi)

            src = QTableWidgetItem(_format_source(str(row["source"]), row["source_ref"]))
            src.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, COL_SOURCE, src)

            tm = QTableWidgetItem(_format_time(row["created_at"]))
            tm.setData(Qt.ItemDataRole.UserRole, str(row["created_at"] or ""))
            tm.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, COL_TIME, tm)

        self.table.resizeColumnToContents(COL_CHECK)
        self.table.resizeColumnToContents(COL_THUMB)
        self.table.resizeColumnToContents(COL_DPI)
        self.table.setSortingEnabled(True)
        self._select_all.setChecked(False)
        self._updating = False

    def _query_rows(self) -> list:
        with db_session(self._db_path) as conn:
            return conn.execute(
                "SELECT s.id, s.idx, s.source, s.source_ref, s.created_at, "
                "       s.capture_dpi, n.image_id, n.filestem, i.dpi AS img_dpi "
                "FROM scans s "
                "JOIN nodes n ON n.id = s.root_node_id "
                "JOIN images i ON i.id = n.image_id "
                "WHERE s.deleted_at IS NULL AND s.root_node_id IS NOT NULL "
                "ORDER BY s.idx"
            ).fetchall()

    def _thumb_pixmap(self, image_id: int) -> Optional[QPixmap]:
        try:
            blob = self._thumb_loader(image_id, _THUMB_PX * 2)
        except Exception:
            blob = None
        if not blob:
            return None
        pix = QPixmap()
        if pix.loadFromData(blob):
            return pix.scaled(_THUMB_PX, _THUMB_PX, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        return None

    # ── selection-aware editing ─────────────────────────────────────────
    def _selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})

    def _checked_rows(self) -> list[int]:
        """Rows whose checkbox is ticked — the explicit multi-edit set
        (independent of which row is highlighted)."""
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, COL_CHECK)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                out.append(r)
        return out

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Single click on a DPI cell jumps straight into its editor."""
        if col != COL_DPI:
            return
        item = self.table.item(row, col)
        if item is not None and (item.flags() & Qt.ItemFlag.ItemIsEditable):
            self.table.editItem(item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        col, row = item.column(), item.row()
        self._updating = True
        try:
            if col == COL_CHECK:
                # Toggling one checkbox while several rows are highlighted
                # toggles all of them (selection-driven, as before).
                sel = self._selected_rows()
                if row in sel and len(sel) > 1:
                    state = item.checkState()
                    for r in sel:
                        self.table.item(r, COL_CHECK).setCheckState(state)
            elif col == COL_DPI:
                val = float(item.data(Qt.ItemDataRole.EditRole) or 0)
                if val <= 0:
                    return
                # Editing a CHECKED row's DPI applies to every checked row —
                # the checkbox is the explicit multi-edit set (the user may
                # have checked rows without highlighting them).
                checked = self._checked_rows()
                if row in checked and len(checked) > 1:
                    # Capture item refs before mutating — setData re-sorts the
                    # table live, so row indices would go stale mid-loop;
                    # QTableWidgetItem refs survive the re-sort.
                    cells = [self.table.item(r, COL_DPI) for r in checked]
                    for cell in cells:
                        if cell is not None and cell is not item:
                            cell.setData(Qt.ItemDataRole.EditRole, val)
        finally:
            self._updating = False

    def _on_select_all(self, state: int) -> None:
        if self._updating:
            return
        self._updating = True
        cs = Qt.CheckState.Checked if state == Qt.CheckState.Checked.value \
            else Qt.CheckState.Unchecked
        for r in range(self.table.rowCount()):
            self.table.item(r, COL_CHECK).setCheckState(cs)
        self._updating = False

    def _on_header_clicked(self, section: int) -> None:
        if section == COL_THUMB:
            # Re-sort by name instead — the thumbnail column has no order.
            self.table.sortItems(COL_NAME)

    # ── apply ────────────────────────────────────────────────────────────
    def _on_apply(self) -> None:
        # Reprocess only scans whose DPI actually CHANGED — and only among
        # checked rows. Setting the DPI deletes that scan's processing data
        # and reruns it from raw (reprocess_active_scans wipes branches +
        # the node subtree, then re-enqueues).
        edits: dict[int, float] = {}   # scan_id → new dpi
        images: dict[int, float] = {}  # image_id → new dpi
        for r in range(self.table.rowCount()):
            if self.table.item(r, COL_CHECK).checkState() != Qt.CheckState.Checked:
                continue
            dpi_item = self.table.item(r, COL_DPI)
            dpi = float(dpi_item.data(Qt.ItemDataRole.EditRole) or 0)
            orig = float(dpi_item.data(Qt.ItemDataRole.UserRole) or 0)
            if dpi <= 0 or dpi == orig:
                continue  # unchanged → nothing to reprocess
            chk = self.table.item(r, COL_CHECK)
            edits[int(chk.data(Qt.ItemDataRole.UserRole))] = dpi
            images[int(chk.data(Qt.ItemDataRole.UserRole + 1))] = dpi
        if not edits:
            return
        with db_session(self._db_path) as conn:
            for image_id, dpi in images.items():
                conn.execute("UPDATE images SET dpi = ? WHERE id = ?", (dpi, image_id))
            for scan_id, dpi in edits.items():
                conn.execute("UPDATE scans SET capture_dpi = ? WHERE id = ?", (dpi, scan_id))
            conn.commit()
        self._reprocess_cb(set(edits.keys()))
        self.reload()
