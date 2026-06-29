# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Listening-mode UI for the aglaia-bridge handoff (#47).

Shows the pairing QR and waits for the phone to push an ``.aglbundle``. The
HTTP upload arrives on a worker thread; we marshal it to the Qt main thread via
a signal so the ingest (chain/DB access) happens where it's safe.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from aglaia.workers.bridge_server import BridgeReceiver, qr_png


class BridgeReceiveController(QObject):
    """Owns the one-shot receiver and re-emits uploads on the main thread."""

    bundle_received = Signal(str)  # path to the uploaded .aglbundle zip

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._receiver: BridgeReceiver | None = None

    def start(self):
        self._receiver = BridgeReceiver(on_bundle=self._on_bundle)
        return self._receiver.start()

    def _on_bundle(self, path: Path) -> None:
        # Called on the server's handler thread — emit() is queued to the
        # receiver's (main) thread by Qt, so the slot runs there.
        self.bundle_received.emit(str(path))

    def stop(self) -> None:
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None


class BridgeReceiveDialog(QDialog):
    """Displays the pairing QR; ``ingest`` is a main-thread callable that takes
    the uploaded zip path and returns the imported page count."""

    def __init__(self, ingest: Callable[[str], int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ingest = ingest
        self.setWindowTitle(self.tr("Receive from phone"))
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._qr = QLabel()
        self._qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr.setMinimumSize(280, 280)
        layout.addWidget(self._qr)

        self._status = QLabel(self.tr("Starting…"))
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._controller = BridgeReceiveController(self)
        self._controller.bundle_received.connect(self._on_bundle_path)
        self._start()

    def _start(self) -> None:
        try:
            info = self._controller.start()
        except Exception as exc:  # noqa: BLE001 — surface any startup failure in the UI
            self._status.setText(self.tr("Couldn't start the receiver: %s") % exc)
            return
        pix = QPixmap()
        pix.loadFromData(qr_png(info.qr_uri()))
        self._qr.setPixmap(pix)
        self._status.setText(
            self.tr("Open aglaia-bridge on your phone, tap Push, and scan this code.\n"
                    "Waiting on %s:%d…\n"
                    "(If your Wi-Fi blocks device-to-device traffic, share a hotspot.)")
            % (info.host, info.port)
        )

    def _on_bundle_path(self, zip_path: str) -> None:
        try:
            pages = self._ingest(zip_path)
            self._status.setText(self.tr("Received %d page(s). Importing…") % pages)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(self.tr("Import failed: %s") % exc)
        self._controller.stop()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt API
        self._controller.stop()
        super().closeEvent(event)
