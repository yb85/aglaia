# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Bridge sidebar tab — pair a phone, then use it as a live camera (#49).

Two pages in a ``QStackedWidget``:

* **Pairing** (index 0) — the pairing QR, a status line, and a "Restart pairing"
  button. Right-click the QR → "Copy pairing URI" (drives the device-free
  ``tools/fake_bridge_phone.py`` E2E).
* **Live** (index 1) — hosts the live ``CaptureTab`` that ``MainWindow`` installs
  once the phone connects; identical capture controls to the webcam, minus the
  camera/format pickers.

This widget is deliberately dumb: it renders what it's told and emits
``restart_requested``. All session/server logic lives in ``BridgeLiveController``
and ``MainWindow``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class BridgeTab(QWidget):
    restart_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._uri: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        root.addWidget(self._stack)

        # ── page 0: pairing ────────────────────────────────────────
        pairing = QWidget()
        v = QVBoxLayout(pairing)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        title = QLabel(self.tr("Bridge"))
        title.setObjectName("SectionTitle")
        v.addWidget(title)

        self._qr = QLabel()
        self._qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr.setMinimumSize(260, 260)
        # Right-click → "Copy pairing URI" (device-free E2E hook).
        self._qr.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._qr.customContextMenuRequested.connect(self._show_qr_menu)
        v.addWidget(self._qr)

        self._status = QLabel(self.tr("Starting…"))
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        v.addWidget(self._status)

        self._btn_restart = QPushButton(self.tr("Restart pairing"))
        self._btn_restart.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_restart.clicked.connect(self.restart_requested)
        v.addWidget(self._btn_restart)
        v.addStretch(1)

        self._stack.addWidget(pairing)  # index 0
        self._live: QWidget | None = None

    # ── pairing page API ───────────────────────────────────────────
    def show_pairing(self, qr: QPixmap | None, status: str, *, uri: str | None = None) -> None:
        self._uri = uri
        if qr is not None and not qr.isNull():
            self._qr.setPixmap(qr)
        else:
            self._qr.clear()
        self._status.setText(status)
        self._stack.setCurrentIndex(0)

    def set_status(self, status: str) -> None:
        self._status.setText(status)

    # ── live page API ──────────────────────────────────────────────
    def show_live(self, widget: QWidget) -> None:
        """Install the live capture widget as page 1 and switch to it."""
        self.clear_live()
        self._live = widget
        self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)

    def clear_live(self) -> QWidget | None:
        """Detach and return the live widget (caller owns ``deleteLater``)."""
        w = self._live
        if w is not None:
            self._stack.removeWidget(w)
            self._live = None
        self._stack.setCurrentIndex(0)
        return w

    # ── internal ───────────────────────────────────────────────────
    def _show_qr_menu(self, pos) -> None:
        if not self._uri:
            return
        menu = QMenu(self)
        act = menu.addAction(self.tr("Copy pairing URI"))
        chosen = menu.exec(self._qr.mapToGlobal(pos))
        if chosen is act:
            QGuiApplication.clipboard().setText(self._uri)
