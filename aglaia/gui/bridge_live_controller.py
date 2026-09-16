# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Qt-side owner of the live bridge server (#49).

Mirrors ``BridgeReceiveController`` (bridge_receive.py): the server fires its
``on_session_started`` / ``on_session_ended`` callbacks on HTTP handler threads;
this re-emits them as Qt signals so ``MainWindow`` handles them on the main
thread (building/tearing down the live capture UI is Qt-thread-only).

``bridge_live`` is imported lazily so importing this module never hard-requires
the ``gui`` extra's ``cryptography`` / ``segno``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from aglaia.workers.bridge_live import BridgeLiveServer
    from aglaia.workers.bridge_server import ReceiverInfo


class BridgeLiveController(QObject):
    session_started = Signal(str)  # device name
    session_ended = Signal(str)    # reason

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._server: BridgeLiveServer | None = None

    @property
    def server(self) -> BridgeLiveServer | None:
        return self._server

    def arm(self) -> ReceiverInfo:
        """Start a fresh live server (new cert + token) and return its QR info."""
        from aglaia.workers.bridge_live import BridgeLiveServer

        self.disarm()
        server = BridgeLiveServer(
            on_session_started=lambda device: self.session_started.emit(device),
            on_session_ended=lambda reason: self.session_ended.emit(reason),
        )
        info = server.start()
        self._server = server
        return info

    def disarm(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
