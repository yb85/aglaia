# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
"""Mistral OCR Jobs tab — a zebra table of every Aglaïa batch job on the
account (newest first), refreshed from the Batch API. The linked project
path (from job metadata) is clickable to open that project."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_ZEBRA_ODD, COLOR_FONT_DIM, COLOR_FONT_PLACEHOLDER, COLOR_PRIMARY,
)
from aglaia.gui.theme import lucide
from aglaia.gui.timeago import time_ago

COL_SUBMITTED, COL_STATUS, COL_PROJECT, COL_CHUNK, COL_REQ = range(5)

_STATUS_COLOR = {
    "SUCCESS": "#2e7d32", "RUNNING": COLOR_PRIMARY, "QUEUED": COLOR_PRIMARY,
    "FAILED": "#c62828", "TIMEOUT_EXCEEDED": "#c62828",
    "CANCELLED": COLOR_FONT_PLACEHOLDER,
    "CANCELLATION_REQUESTED": COLOR_FONT_PLACEHOLDER,
}


class MistralJobsTab(QWidget):
    """Account-wide Mistral batch jobs. ``open_project_requested(path)`` fires
    when the user clicks a linked project."""

    open_project_requested = Signal(str)

    def __init__(self, db_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db_path = str(db_path)
        self._worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel(self.tr("Mistral OCR jobs"))
        title.setObjectName("SectionTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        bar.addWidget(self._status_lbl)
        self._refresh_btn = QPushButton(self.tr("Refresh"))
        self._refresh_btn.setIcon(lucide("refresh-cw", color=COLOR_PRIMARY, size=13))
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self._refresh_btn)
        root.addLayout(bar)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            self.tr("Submitted"), self.tr("Status"), self.tr("Project"),
            self.tr("Chunk"), self.tr("Requests"),
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            f"QTableWidget {{ alternate-background-color: {COLOR_BG_ZEBRA_ODD}; }}")
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            COL_PROJECT, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._on_cell_clicked)
        root.addWidget(self.table, 1)

        self.refresh()

    def showEvent(self, e):  # noqa: N802 — refresh whenever the tab opens
        super().showEvent(e)
        self.refresh()

    def refresh(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        from aglaia.workers.MistralBatchWorker import MistralBatchWorker
        self._status_lbl.setText(self.tr("Loading…"))
        self._worker = MistralBatchWorker(action="list", db_path=self._db_path)
        self._worker.list_done.connect(self._populate)
        self._worker.start()

    def _populate(self, rows: list, error: str) -> None:
        if error:
            self._status_lbl.setText(error)
            return
        self._status_lbl.setText(
            self.tr("{n} job(s)").format(n=len(rows)))
        self.table.setRowCount(len(rows))
        for r, j in enumerate(rows):
            sub = QTableWidgetItem(time_ago(j.get("created_at")))
            sub.setData(Qt.ItemDataRole.ToolTipRole, str(j.get("created_at")))
            self.table.setItem(r, COL_SUBMITTED, sub)

            status = str(j.get("status") or "")
            st = QTableWidgetItem(status)
            st.setForeground(QColor(_STATUS_COLOR.get(status, COLOR_FONT_DIM)))
            self.table.setItem(r, COL_STATUS, st)

            proj = str(j.get("project") or "")
            name = os.path.basename(proj) if proj else self.tr("(unknown)")
            pit = QTableWidgetItem(name)
            if proj:
                pit.setForeground(QColor(COLOR_PRIMARY))
                pit.setToolTip(self.tr("Open {p}").format(p=proj))
                pit.setData(Qt.ItemDataRole.UserRole, proj)
            self.table.setItem(r, COL_PROJECT, pit)

            chunk = j.get("chunk")
            tot = j.get("chunks_total")
            ctxt = (f"{int(chunk) + 1}/{tot}" if chunk not in ("", None)
                    and tot not in ("", None) else "—")
            self.table.setItem(r, COL_CHUNK, QTableWidgetItem(ctxt))

            done, total = j.get("succeeded"), j.get("total")
            rtxt = (f"{done}/{total}" if total is not None else "—")
            self.table.setItem(r, COL_REQ, QTableWidgetItem(rtxt))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            COL_PROJECT, QHeaderView.ResizeMode.Stretch)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col != COL_PROJECT:
            return
        it = self.table.item(row, COL_PROJECT)
        path = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if path:
            self.open_project_requested.emit(str(path))
