# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Freehand-capture registration dialog.

Shows a live webcam preview with a red 2 cm × 2 cm overlay square. The
user nudges a high-contrast pattern into the square and clicks
"Register". The dialog converts the box from millimetres to pixels
using the current effective DPI, samples the centred patch from the
latest webcam frame, and hands it back to the caller so the SIFT
tracker can store the reference descriptor set.

Closes with either Accepted (a registration succeeded — caller reads
`patch_bgr` + `roi_xywh`) or Rejected (user cancelled or no keypoints
found).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from lib.gui.colors import COLOR_FONT_MUTED, COLOR_FONT_SECONDARY


class _PreviewLabel(QLabel):
    """QLabel that paints the live BGR frame fit-to-widget and overlays
    a centred ROI rectangle in red. The rectangle is the *registration
    target* — what the user must drop a pattern into."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._frame_bgr: Optional[np.ndarray] = None
        self._roi_frac: float = 0.0   # ROI half-side / min(w, h)/2 in frame coords.

    def set_frame(self, bgr: Optional[np.ndarray]) -> None:
        self._frame_bgr = bgr
        self.update()

    def set_roi_pixels(self, side_px: int) -> None:
        if self._frame_bgr is None:
            return
        h, w = self._frame_bgr.shape[:2]
        ref = min(w, h)
        self._roi_frac = max(0.03, min(0.6, side_px / max(1, ref)))

    def roi_in_frame(self) -> Optional[tuple[int, int, int, int]]:
        """ROI rect in the *original* frame coords."""
        if self._frame_bgr is None or self._roi_frac <= 0:
            return None
        h, w = self._frame_bgr.shape[:2]
        ref = min(w, h)
        side = max(8, int(self._roi_frac * ref))
        x = (w - side) // 2
        y = (h - side) // 2
        return x, y, side, side

    def paintEvent(self, _ev) -> None:  # noqa: N802 — Qt API
        if self._frame_bgr is None:
            super().paintEvent(_ev)
            return
        rgb = cv2.cvtColor(self._frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        ww, wh = self.width(), self.height()
        scale = min(ww / w, wh / h)
        dw, dh = int(w * scale), int(h * scale)
        dx = (ww - dw) // 2
        dy = (wh - dh) // 2

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(dx, dy, dw, dh, pix)

        roi = self.roi_in_frame()
        if roi is not None:
            from PySide6.QtGui import QColor
            x, y, side, _ = roi
            rx = dx + int(x * scale)
            ry = dy + int(y * scale)
            rs = int(side * scale)
            # Dim the rest of the preview so the registration square
            # reads as the focal point. Cuts out a hole over the square
            # by stroking a thick black frame, then paints the
            # high-contrast red outline on top.
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            shade = QColor(0, 0, 0, 110)
            p.fillRect(dx, dy, dw, ry - dy, shade)              # above
            p.fillRect(dx, ry + rs, dw, dy + dh - (ry + rs), shade)  # below
            p.fillRect(dx, ry, rx - dx, rs, shade)              # left
            p.fillRect(rx + rs, ry, dx + dw - (rx + rs), rs, shade)  # right
            # Thick bright-red rectangle + matching corner ticks.
            pen = QPen(QColor(255, 0, 0, 255), 6)
            p.setPen(pen)
            p.drawRect(rx, ry, rs, rs)
            p.setPen(QPen(QColor(255, 0, 0, 255), 4))
            tick = max(10, rs // 6)
            for cx, cy in ((rx, ry), (rx + rs, ry),
                           (rx, ry + rs), (rx + rs, ry + rs)):
                p.drawLine(cx - tick, cy, cx + tick, cy)
                p.drawLine(cx, cy - tick, cx, cy + tick)
        p.end()


class FreehandRegistrationTab(QWidget):
    """Tab-hosted registration of the freehand-tracking reference pattern.

    `webcam_thread` is the live `WebcamThread`; the tab polls its
    `get_frame()` 30× / s, draws the overlay, and on Register samples
    the latest frame + the ROI rectangle, then emits
    `registered(patch_bgr, roi_xywh)`. The host owns the tab lifecycle
    and removes it on either signal.

    The ROI is defined in **frame pixels**, not millimetres. Using
    physical units (cm × dpi) meant the box shrank when the user zoomed
    in via the AVCaptureDevice — the registration patch became too
    small to extract stable SIFT features. Pixel-based sizing keeps
    the box constant on screen regardless of zoom, which is also what
    the user expects visually.
    """

    REFRESH_MS = 33
    DEFAULT_SIDE_PX = 160

    registered = Signal(object, tuple)   # (patch_bgr, roi_xywh)
    cancel_requested = Signal()

    def __init__(self, webcam_thread, *, side_px: int = DEFAULT_SIDE_PX,
                 parent=None):
        super().__init__(parent)
        self._webcam = webcam_thread
        self._side_px = max(40, int(side_px))
        self.patch_bgr: Optional[np.ndarray] = None
        self.roi_xywh: Optional[tuple[int, int, int, int]] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        caption = QLabel(self.tr(
            "Place a high-contrast pattern (sticker, drawing, label) "
            "inside the red square below.\n"
            "Once it sits still, click <b>Register</b>. The tracker will use it as a"
            " reference; briefly covering it will trigger a capture."
        ))
        caption.setWordWrap(True)
        caption.setStyleSheet(f"color: {COLOR_FONT_SECONDARY};")
        v.addWidget(caption)

        self._preview = _PreviewLabel()
        v.addWidget(self._preview, 1)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {COLOR_FONT_MUTED};")
        v.addWidget(self._status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self.cancel_requested)
        btns.addWidget(self._cancel_btn)
        self._register_btn = QPushButton(self.tr("Register"))
        self._register_btn.setDefault(True)
        self._register_btn.clicked.connect(self._on_register)
        btns.addWidget(self._register_btn)
        v.addLayout(btns)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.REFRESH_MS)

    def stop(self) -> None:
        """Called by the host before removing this tab so the timer
        stops polling the dead-soon webcam reference."""
        self._timer.stop()

    # ── live preview ──────────────────────────────────────────────
    def _tick(self) -> None:
        if self._webcam is None:
            return
        frame = self._webcam.get_frame()
        if frame is None:
            return
        self._preview.set_frame(frame)
        # Fixed pixel side — same screen footprint regardless of the
        # camera's current AVF zoom.
        self._preview.set_roi_pixels(self._side_px)

    # ── actions ───────────────────────────────────────────────────
    def _on_register(self) -> None:
        frame = self._webcam.get_frame() if self._webcam else None
        if frame is None:
            self._status.setText(self.tr("No live frame available."))
            return
        roi = self._preview.roi_in_frame()
        if roi is None:
            self._status.setText(self.tr("Could not compute ROI — try again."))
            return
        x, y, w, h = roi
        self.patch_bgr = frame[y:y + h, x:x + w].copy()
        self.roi_xywh = roi
        self.registered.emit(self.patch_bgr, self.roi_xywh)
