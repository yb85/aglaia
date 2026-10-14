# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Bridge tab + MainWindow wiring (#49): the live-tab build hides device
selection, session start/teardown swaps the capture routing, and a live phone
session is refused while a local webcam is running (mutual exclusion).

Uses the bypassed-``__init__`` MainWindow pattern from test_capture_unification;
only the attributes each code path reads are stubbed."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("cryptography")

from types import SimpleNamespace  # noqa: E402

from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from aglaia.gui.bridge_live_controller import BridgeLiveController  # noqa: E402
from aglaia.gui.sidebar.tabs.BridgeTab import BridgeTab  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def mw(qapp):
    from aglaia.gui.MainWindow import MainWindow
    m = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(m)
    m._transform_items = [(None, "none", None), ("rotate-cw-square", "R90", 90)]
    m.args = SimpleNamespace(config={"keycontrols": {"scan": ["space"]}})
    m.toast = lambda *a, **k: None
    m._freehand_overlay = lambda img: img
    return m


# ── BridgeTab (pure widget) ────────────────────────────────────────
def test_bridge_tab_pairing_then_live_swap(qapp):
    tab = BridgeTab()
    pix = QPixmap(16, 16)
    pix.fill()
    tab.show_pairing(pix, "scan me", uri="aglaia://v1?h=1&p=2&t=3&fp=4&m=live")
    assert tab._stack.currentIndex() == 0

    from PySide6.QtWidgets import QLabel
    live = QLabel("live")
    tab.show_live(live)
    assert tab._stack.currentWidget() is live

    returned = tab.clear_live()
    assert returned is live
    assert tab._stack.currentIndex() == 0  # back to pairing


def test_bridge_tab_copy_uri_to_clipboard(qapp):
    tab = BridgeTab()
    tab.show_pairing(None, "waiting", uri="aglaia://v1?h=h&p=1&t=t&fp=f&m=live")
    # _show_qr_menu pops a menu; exercise the clipboard branch directly.
    from PySide6.QtGui import QGuiApplication
    QGuiApplication.clipboard().setText(tab._uri)
    assert QGuiApplication.clipboard().text().endswith("m=live")


# ── live-tab build (bridge=True) ───────────────────────────────────
def test_make_bridge_capture_tab_hides_device_pickers(mw):
    ct = mw._make_live_capture_tab(bridge=True)
    ct.show()
    assert ct.camera_row.isHidden()
    assert ct.format_row.isHidden()
    assert ct.btn_freehand.isHidden()
    assert ct.btn_deactivate.text() == "End bridge session"


def test_make_webcam_capture_tab_keeps_pickers(mw):
    ct = mw._make_live_capture_tab(bridge=False)
    ct.show()
    assert not ct.camera_row.isHidden()
    assert ct.camera_combo.count() >= 1  # populated with devices


# ── session start / teardown ───────────────────────────────────────
def test_session_refused_while_local_webcam_active(mw):
    mw.webcam_thread = SimpleNamespace(is_bridge=False)  # a local webcam is live
    mw._bridge_tab = BridgeTab()
    mw._bridge_controller = BridgeLiveController()
    mw._bridge_controller.arm()  # armed pairing exists
    mw._bridge_live = False
    try:
        mw._on_bridge_session_started("Intruder")
        # Refused: no live UI built, local webcam untouched, QR re-armed.
        assert mw._bridge_live is False
        assert mw.webcam_thread.is_bridge is False
        assert mw._bridge_controller.server is not None
    finally:
        mw._bridge_controller.disarm()


def test_session_start_and_teardown_roundtrip(mw):
    mw.webcam_thread = None
    mw._bridge_tab = BridgeTab()
    mw._bridge_controller = BridgeLiveController()
    mw._bridge_controller.arm()
    mw._bridge_live = False
    mw._bridge_prev_capture_tab = "SENTINEL"
    mw._capture_tab = "SENTINEL"
    mw.sidebar = SimpleNamespace(active=lambda: "pipeline")  # not on bridge tab
    try:
        mw._on_bridge_session_started("Pixel")
        assert mw._bridge_live is True
        assert getattr(mw.webcam_thread, "is_bridge", False) is True
        assert mw._capture_tab is mw._bridge_tab._live  # routing repointed
        assert mw._active_cam_id == "bridge:Pixel"

        mw._end_bridge_session()
        assert mw._bridge_live is False
        assert mw.webcam_thread is None
        assert mw._capture_tab == "SENTINEL"  # restored
    finally:
        mw._bridge_controller.disarm()
        if getattr(mw, "webcam_thread", None) is not None:
            mw.webcam_thread.stop()
