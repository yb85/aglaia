# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Smoke tests for the unified capture-tab bring-up (launch-from-camera
and late-activation share one placeholder/stack + live-tab builder).

Capture mode can't be driven end-to-end here (it needs the startup dialog
+ a real camera), so these exercise the pure widget logic — building the
live tab, the camera pre-select, and the stack swap — on a MainWindow
whose heavy ``__init__`` is bypassed (only the QMainWindow C++ base is
initialised so QTimer / child widgets work).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QComboBox, QMainWindow, QStackedWidget,
)
from types import SimpleNamespace  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def mw(qapp):
    """A MainWindow with __init__ bypassed — only the QMainWindow base is
    initialised, plus the few attributes the capture helpers read."""
    from aglaia.gui.MainWindow import MainWindow
    m = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(m)
    m._transform_items = [(None, "none", None), ("rotate-cw-square", "R90", 90)]
    m.args = SimpleNamespace(config={"keycontrols": {"scan": ["space"]}})
    return m


def test_make_live_capture_tab(mw):
    ct = mw._make_live_capture_tab()
    ct.show()
    # transform combo seeded from _transform_items
    assert ct.transform_combo.count() == 2
    # Deactivate button is visible in the live tab (the launch-from-camera
    # regression: it used to stay hidden).
    assert not ct.btn_deactivate.isHidden()
    # Legacy aliases point at the live tab's widgets.
    assert mw.zoom_slider is ct.zoom_slider
    assert mw.btn_full_calibrate is ct.btn_full_calibrate
    assert mw.transform_combo is ct.transform_combo
    # Camera + format pickers are populated; format list leads with "Auto".
    assert ct.camera_combo.count() >= 1
    assert ct.format_combo.count() >= 1
    assert ct.format_combo.itemData(0) is None      # first = auto-pick


def test_preview_adapts_aspect_ratio(qapp):
    """The preview height tracks the frame AR (no crop, no AR change)."""
    from PySide6.QtGui import QImage, QPixmap
    from aglaia.gui.sidebar.tabs.CaptureTab import CaptureTab
    ct = CaptureTab()
    ct.show()
    # Feed a 2:1 frame; height must come out at width/2, not the fixed slot.
    img = QImage(400, 200, QImage.Format.Format_RGB888)
    img.fill(0)
    ct.set_preview_pixmap(QPixmap.fromImage(img))
    assert ct.preview_label.height() == round(ct.PREVIEW_W * 200 / 400)


def test_select_capture_camera(mw):
    combo = QComboBox()
    combo.addItem("MacBook Air Camera (id 0)", 0)
    combo.addItem("ybp Camera (id 1)", 1)
    mw._capture_cam_combo = combo

    mw._select_capture_camera(1)
    assert combo.currentData() == 1
    mw._select_capture_camera(0)
    assert combo.currentData() == 0
    # Unknown id is a no-op, not a crash.
    mw._select_capture_camera(99)
    assert combo.currentData() == 0


def test_select_capture_camera_no_combo(mw):
    # Missing combo must not raise.
    mw._select_capture_camera(0)


def test_dpi_readout(qapp):
    """The capture tab shows the DPI, flagged when uncalibrated."""
    from aglaia.gui.sidebar.tabs.CaptureTab import CaptureTab
    ct = CaptureTab()
    ct.set_dpi(150.0, calibrated=True)
    assert "150" in ct.dpi_label.text()
    assert "uncalibrated" not in ct.dpi_label.text()
    ct.set_dpi(100.0, calibrated=False)
    assert "uncalibrated" in ct.dpi_label.text()


def test_stack_swap_to_live(mw):
    ct = mw._make_live_capture_tab()
    stack = QStackedWidget()
    stack.addWidget(QMainWindow())   # index 0 = picker stand-in
    mw._capture_stack = stack
    stack.addWidget(ct)              # index 1 = live
    stack.setCurrentIndex(1)
    assert stack.currentWidget() is ct
