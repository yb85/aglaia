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
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_ZEBRA_ODD, COLOR_FONT_DIM, COLOR_FONT_PLACEHOLDER, COLOR_PRIMARY,
)
from aglaia.gui.theme import lucide
from aglaia.gui.timeago import time_ago

COL_SUBMITTED, COL_STATUS, COL_PROJECT, COL_CHUNK, COL_REQ, COL_DISMISS = range(6)

#: A job in one of these is over — nothing more will happen to it, so it can
#: be cleared from the view. RUNNING / QUEUED / CANCELLATION_REQUESTED are
#: still live and keep no dismiss button: cancel them first.
TERMINAL = frozenset({"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"})

_STATUS_COLOR = {
    "SUCCESS": "#2e7d32", "RUNNING": COLOR_PRIMARY, "QUEUED": COLOR_PRIMARY,
    "FAILED": "#c62828", "TIMEOUT_EXCEEDED": "#c62828",
    "CANCELLED": COLOR_FONT_PLACEHOLDER,
    "CANCELLATION_REQUESTED": COLOR_FONT_PLACEHOLDER,
}


def _load_dismissed() -> set:
    try:
        from aglaia.app_data import db as cfg
        with cfg.session() as conn:
            value = cfg.get(conn, cfg.KEY_MISTRAL_JOBS_DISMISSED, []) or []
    except Exception:
        return set()
    return {str(v) for v in value} if isinstance(value, list) else set()


def _save_dismissed(ids) -> None:
    try:
        from aglaia.app_data import db as cfg
        with cfg.session() as conn:
            cfg.set(conn, cfg.KEY_MISTRAL_JOBS_DISMISSED, sorted(ids))
            conn.commit()
    except Exception:
        pass


class MistralJobsTab(QWidget):
    """Account-wide Mistral batch jobs. ``open_project_requested(path)`` fires
    when the user clicks a linked project."""

    open_project_requested = Signal(str)

    def __init__(self, db_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db_path = str(db_path)
        self._worker = None
        self._rows: list = []
        self._dismissed: set[str] = _load_dismissed()
        self._show_dismissed = False

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
        self._show_chk = QCheckBox(self.tr("Show dismissed"))
        self._show_chk.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        self._show_chk.setToolTip(self.tr(
            "Dismissed jobs are hidden here only — the Mistral Batch API has "
            "no delete, so the job itself stays in your account history."))
        self._show_chk.toggled.connect(self._on_show_dismissed)
        bar.addWidget(self._show_chk)
        self._refresh_btn = QPushButton(self.tr("Refresh"))
        self._refresh_btn.setIcon(lucide("refresh-cw", color=COLOR_PRIMARY, size=13))
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self._refresh_btn)
        root.addLayout(bar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            self.tr("Submitted"), self.tr("Status"), self.tr("Project"),
            self.tr("Chunk"), self.tr("Requests"), "",
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
        self._rows = list(rows)
        self._render()

    def _render(self) -> None:
        rows = [j for j in self._rows
                if self._show_dismissed
                or str(j.get("id") or "") not in self._dismissed]
        hidden = len(self._rows) - len(rows)
        self._status_lbl.setText(
            self.tr("{n} job(s)").format(n=len(rows)) if not hidden else
            self.tr("{n} job(s) · {h} dismissed").format(n=len(rows), h=hidden))
        self._show_chk.setVisible(bool(hidden) or self._show_dismissed)
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

            jid = str(j.get("id") or "")
            self.table.setCellWidget(r, COL_DISMISS,
                                     self._dismiss_cell(jid, status))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            COL_PROJECT, QHeaderView.ResizeMode.Stretch)

    def _dismiss_cell(self, job_id: str, status: str):
        """Trash (or restore) button for a finished job.

        A live job keeps none: there is nothing to clear away yet, and the
        button beside a RUNNING row would read as "cancel", which it is not."""
        if not job_id or status not in TERMINAL:
            return None
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        btn = QPushButton()
        btn.setFlat(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(24, 22)
        if job_id in self._dismissed:
            btn.setIcon(lucide("undo", color=COLOR_FONT_DIM, size=13))
            btn.setToolTip(self.tr("Show this job again"))
            btn.clicked.connect(lambda _=False, j=job_id: self._restore(j))
        else:
            btn.setIcon(lucide("trash-2", color=COLOR_FONT_DIM, size=13))
            btn.setToolTip(self.tr(
                "Dismiss this finished job from the list. Mistral keeps no "
                "delete endpoint, so the job stays in your account history."))
            btn.clicked.connect(lambda _=False, j=job_id: self._dismiss(j))
        row.addWidget(btn)
        row.addStretch(1)
        return wrap

    def _dismiss(self, job_id: str) -> None:
        """Hide a finished job, and drop the project's record of it.

        The local row is what makes a job "pending" for the OCR card's *Check
        result*; leaving it behind would keep asking about a job the user has
        just said they are done with."""
        self._dismissed.add(str(job_id))
        _save_dismissed(self._dismissed)
        try:
            from aglaia.storage.db import db_session
            from aglaia.storage.repo import MistralBatchRepo
            if self._db_path:
                with db_session(self._db_path) as conn:
                    MistralBatchRepo(conn).delete(str(job_id))
                    conn.commit()
        except Exception:
            pass
        self._render()

    def _restore(self, job_id: str) -> None:
        self._dismissed.discard(str(job_id))
        _save_dismissed(self._dismissed)
        self._render()

    def _on_show_dismissed(self, on: bool) -> None:
        self._show_dismissed = bool(on)
        self._render()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col != COL_PROJECT:
            return
        it = self.table.item(row, COL_PROJECT)
        path = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if path:
            self.open_project_requested.emit(str(path))
