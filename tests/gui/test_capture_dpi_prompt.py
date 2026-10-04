# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Ask for the capture DPI while the rig is still in place (#174).

`effective_dpi()` falls back to `input_dpi` (100) until someone calibrates,
and nothing insists — so a whole book gets shot at the default, the chain
upsamples every page to 300, and the mistake is only visible once the export
is too big. By then the rig has moved and the pages have to be shot again.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow      # noqa: E402

from aglaia.gui.MainWindow import MainWindow                 # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _mw(qapp, *, camera: bool, dpi_base=None):
    m = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(m)
    m.webcam_thread = object() if camera else None
    m._dpi_base = dpi_base
    m.opened = []
    # The prompt defers through QTimer; record the call instead of running it.
    m.calibrate_dpi = lambda: m.opened.append(True)
    return m


def test_a_camera_with_no_dpi_is_asked(qapp):
    m = _mw(qapp, camera=True)
    assert m.prompt_dpi_if_uncalibrated() is True


def test_an_already_calibrated_session_is_not(qapp):
    m = _mw(qapp, camera=True, dpi_base=287.0)
    assert m.prompt_dpi_if_uncalibrated() is False


def test_no_camera_asks_nothing(qapp):
    m = _mw(qapp, camera=False)
    assert m.prompt_dpi_if_uncalibrated() is False


def test_it_asks_once_per_activation(qapp):
    """A reminder, not a gate: closing the dialog without calibrating must
    not reopen it on the next frame."""
    m = _mw(qapp, camera=True)
    assert m.prompt_dpi_if_uncalibrated() is True
    assert m.prompt_dpi_if_uncalibrated() is False
    assert m.prompt_dpi_if_uncalibrated() is False


def test_a_new_camera_session_asks_again(qapp):
    """Deactivate/reactivate may mean a new distance, so a new DPI."""
    m = _mw(qapp, camera=True)
    m.prompt_dpi_if_uncalibrated()
    m._dpi_prompted = False          # what _deactivate_capture_clicked does
    assert m.prompt_dpi_if_uncalibrated() is True
