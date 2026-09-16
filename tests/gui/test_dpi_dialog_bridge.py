# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""DPI calibration freeze-frame flow for a bridge source (#49): live ticks run
on the low-res preview (no network), and only a measurement pulls one full-res
still. Verified by counting get_frame vs get_preview_frame calls."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication  # noqa: E402

from aglaia.gui.CalibrationDialogs import DpiCalibrationDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class CountingBridgeCam:
    """A bridge-like source: get_preview_frame() is the cheap live stream,
    get_frame() is the (counted) full-res still."""

    def __init__(self) -> None:
        self.preview_calls = 0
        self.get_frame_calls = 0
        self.still_dims = (4000, 3000)
        self._preview = np.full((240, 320, 3), 200, np.uint8)
        self._still = np.full((3000, 4000, 3), 200, np.uint8)

    def get_preview_frame(self):
        self.preview_calls += 1
        return self._preview.copy()

    def get_frame(self):
        self.get_frame_calls += 1
        return self._still.copy()


def _dialog(qapp, cam):
    return DpiCalibrationDialog(cam, id1_long_mm=85.6, id1_short_mm=53.98)


def test_live_ticks_never_pull_a_still(qapp) -> None:
    cam = CountingBridgeCam()
    dlg = _dialog(qapp, cam)
    for _ in range(10):
        dlg._tick_preview()
    assert cam.get_frame_calls == 0      # live ticks stayed on the preview stream
    assert cam.preview_calls == 10


def test_measure_scale_lifts_preview_dpi(qapp) -> None:
    cam = CountingBridgeCam()
    dlg = _dialog(qapp, cam)
    dlg._tick_preview()                  # populate _live_frame (long edge 320)
    assert dlg._measure_scale() == pytest.approx(4000 / 320)


def test_capture_refine_pulls_exactly_one_still(qapp) -> None:
    cam = CountingBridgeCam()
    dlg = _dialog(qapp, cam)
    dlg._tick_preview()                  # need a live frame first
    dlg._on_capture_refine()             # detect fails on the blank still, but the grab counts
    assert cam.get_frame_calls == 1


def test_ruler_pulls_exactly_one_still(qapp) -> None:
    cam = CountingBridgeCam()
    dlg = _dialog(qapp, cam)
    dlg._tick_preview()
    dlg._on_method_ruler()
    assert cam.get_frame_calls == 1


def test_webcam_source_uses_get_frame_for_live(qapp) -> None:
    # A plain webcam (no get_preview_frame) → live ticks fall back to get_frame,
    # and measure scale is 1.0 (its live frame is already full-res).
    class Webcam:
        def __init__(self):
            self.get_frame_calls = 0

        def get_frame(self):
            self.get_frame_calls += 1
            return np.full((480, 640, 3), 200, np.uint8)

    cam = Webcam()
    dlg = _dialog(qapp, cam)
    dlg._tick_preview()
    assert cam.get_frame_calls == 1
    assert dlg._measure_scale() == 1.0
