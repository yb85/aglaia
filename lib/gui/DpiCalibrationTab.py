# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
DPI calibration tab.

Single-frame credit-card-based DPI estimation, hosted in the right-side
tab strip of `MainWindow`. Two buttons:

  * **Capture and detect** — grab one webcam frame, run Apple Vision's
    rectangle detector, drop a 4-corner overlay on the captured frame.
    If Vision rejects the scene the corners are seeded at the centre of
    the frame so the user can drag them into place.
  * **Calibrate DPI** — runs `refine_and_measure` against the current
    corner positions and emits `calibration_committed(dpi, base_dpi,
    zoom, frame)`. Disabled until corners exist.

The corner overlay is fully draggable: hover near a handle → cursor
turns into a size-all glyph, hold + move to reposition. Clicks outside
the existing handles seed new corners until 4 exist; thereafter only
drags work, so the user can't accidentally add a fifth.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from lib.gui.colors import COLOR_FONT_MUTED


class _PickLabel(QLabel):
    """Scanned-frame canvas. Once 4 corners exist (auto-detect or manual
    click), they are draggable: hover near a handle, hold primary
    button, move. Clicks outside any handle add a corner until 4 exist."""

    HANDLE_R = 9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._base_pixmap: Optional[QPixmap] = None
        self._img_clicks: list[tuple[float, float]] = []
        self._image_size: tuple[int, int] = (0, 0)
        self._scale: float = 1.0
        self._offset: tuple[float, float] = (0.0, 0.0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._drag_idx: Optional[int] = None
        self._on_change: Optional[Callable[[list[tuple[float, float]]], None]] = None

    def set_image(self, pix: QPixmap, image_size: tuple[int, int]) -> None:
        self._base_pixmap = pix
        self._image_size = image_size
        self._img_clicks = []
        self._drag_idx = None
        self._repaint()
        self._emit_change()

    def set_corners(self, corners) -> None:
        self._img_clicks = [(float(x), float(y)) for x, y in corners]
        self._drag_idx = None
        self._repaint()
        self._emit_change()

    def clear_clicks(self) -> None:
        self._img_clicks = []
        self._drag_idx = None
        self._repaint()
        self._emit_change()

    def corners(self) -> list[tuple[float, float]]:
        return list(self._img_clicks)

    def _emit_change(self) -> None:
        if self._on_change is not None:
            self._on_change(list(self._img_clicks))

    def _widget_to_image(self, wx: float, wy: float
                         ) -> Optional[tuple[float, float]]:
        ox, oy = self._offset
        if self._scale <= 0:
            return None
        ix = (wx - ox) / self._scale
        iy = (wy - oy) / self._scale
        iw, ih = self._image_size
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None

    def _hit_handle(self, wx: float, wy: float) -> Optional[int]:
        ox, oy = self._offset
        r2 = self.HANDLE_R * self.HANDLE_R * 2.25
        for i, (ix, iy) in enumerate(self._img_clicks):
            cx = ix * self._scale + ox
            cy = iy * self._scale + oy
            if (cx - wx) ** 2 + (cy - wy) ** 2 <= r2:
                return i
        return None

    def mousePressEvent(self, ev):  # noqa: N802
        if self._base_pixmap is None:
            return
        wx, wy = ev.position().x(), ev.position().y()
        idx = self._hit_handle(wx, wy)
        if idx is not None:
            self._drag_idx = idx
            return
        if len(self._img_clicks) >= 4:
            return
        ip = self._widget_to_image(wx, wy)
        if ip is None:
            return
        self._img_clicks.append(ip)
        self._repaint()
        self._emit_change()

    def mouseMoveEvent(self, ev):  # noqa: N802
        wx, wy = ev.position().x(), ev.position().y()
        if self._drag_idx is None:
            if self._hit_handle(wx, wy) is not None:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.unsetCursor()
            return
        ip = self._widget_to_image(wx, wy)
        if ip is None:
            ox, oy = self._offset
            iw, ih = self._image_size
            cx = max(ox, min(self.width() - ox, wx))
            cy = max(oy, min(self.height() - oy, wy))
            ip = ((cx - ox) / self._scale, (cy - oy) / self._scale)
            ip = (max(0.0, min(iw - 1.0, ip[0])),
                  max(0.0, min(ih - 1.0, ip[1])))
        self._img_clicks[self._drag_idx] = ip
        self._repaint()

    def mouseReleaseEvent(self, _ev):  # noqa: N802
        if self._drag_idx is not None:
            self._drag_idx = None
            self._emit_change()

    def _repaint(self) -> None:
        if self._base_pixmap is None:
            return
        w = self.width() or self._base_pixmap.width()
        h = self.height() or self._base_pixmap.height()
        iw, ih = self._image_size
        self._scale = min(w / iw, h / ih) if iw and ih else 1.0
        disp_w = iw * self._scale
        disp_h = ih * self._scale
        self._offset = ((w - disp_w) / 2.0, (h - disp_h) / 2.0)

        scaled = self._base_pixmap.scaled(
            int(disp_w), int(disp_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QPixmap(w, h)
        canvas.fill(QColor(0, 0, 0))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ox, oy = self._offset
        p.drawPixmap(int(ox), int(oy), scaled)

        def proj(ix, iy):
            return (ix * self._scale + ox, iy * self._scale + oy)
        projected = [proj(ix, iy) for ix, iy in self._img_clicks]

        if len(projected) >= 2:
            pen = QPen(QColor(255, 64, 64))
            pen.setWidth(2)
            p.setPen(pen)
            for i in range(len(projected)):
                j = (i + 1) % len(projected)
                if j == 0 and len(projected) < 4:
                    continue
                x0, y0 = projected[i]
                x1, y1 = projected[j]
                p.drawLine(int(x0), int(y0), int(x1), int(y1))

        for i, (cx, cy) in enumerate(projected):
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(QColor(255, 64, 64))
            p.drawEllipse(QPoint(int(cx), int(cy)),
                          self.HANDLE_R, self.HANDLE_R)
            p.setPen(QPen(QColor(255, 255, 255)))
            p.drawText(int(cx) + self.HANDLE_R + 4,
                       int(cy) - self.HANDLE_R - 2, str(i + 1))
        p.end()
        self.setPixmap(canvas)

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self._repaint()


class DpiCalibrationTab(QWidget):
    """Tab-hosted DPI calibration with Apple Vision auto-detect.

    Emits `calibration_committed(dpi, base_dpi, zoom, frame_bgr,
    corners_quad)` when the user clicks "Calibrate DPI" with a valid
    set of corners. The host (MainWindow) is responsible for saving the
    calibration to disk + signalling worker reload.
    """

    calibration_committed = Signal(float, float, float, object, object)
    closed = Signal()

    def __init__(self, webcam_thread, *,
                 id1_long_mm: float, id1_short_mm: float,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._webcam = webcam_thread
        self._id1_long_mm = float(id1_long_mm)
        self._id1_short_mm = float(id1_short_mm)
        self._frame_bgr: Optional[np.ndarray] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        caption = QLabel(self.tr(
            "Place a credit-card-sized object (ISO ID-1, "
            "{long} × {short} mm) flat in the "
            "frame and click <b>Capture and detect</b>. Apple Vision tries "
            "to find the corners automatically; drag any handle to nudge "
            "a wrong one. Then click <b>Calibrate DPI</b>."
        ).format(long=self._id1_long_mm, short=self._id1_short_mm))
        caption.setWordWrap(True)
        v.addWidget(caption)

        self._pick = _PickLabel()
        self._pick.setMinimumHeight(420)
        self._pick._on_change = self._on_corners_changed
        v.addWidget(self._pick, 1)

        self._status = QLabel(self.tr("Click <b>Capture and detect</b> to start."))
        self._status.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-style: italic;")
        v.addWidget(self._status)

        row = QHBoxLayout()
        self._btn_capture = QPushButton(self.tr("Capture and detect"))
        self._btn_capture.clicked.connect(self._on_capture)
        row.addWidget(self._btn_capture)

        self._btn_reset = QPushButton(self.tr("Reset corners"))
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_reset.setEnabled(False)
        row.addWidget(self._btn_reset)

        row.addStretch(1)

        self._btn_calibrate = QPushButton(self.tr("Calibrate DPI"))
        self._btn_calibrate.setEnabled(False)
        self._btn_calibrate.clicked.connect(self._on_calibrate)
        row.addWidget(self._btn_calibrate)
        v.addLayout(row)

    # ── handlers ──────────────────────────────────────────────────
    def _on_capture(self) -> None:
        if self._webcam is None:
            self._status.setText(self.tr("No webcam available."))
            return
        frame = self._webcam.get_frame()
        if frame is None:
            self._status.setText(self.tr("No frame from camera."))
            return
        self._frame_bgr = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._pick.set_image(pix, (w, h))

        from lib.workers.CreditCardDPI import detect_card_dpi
        _dpi, quad = detect_card_dpi(self._frame_bgr)
        if quad is not None and len(quad) == 4:
            self._pick.set_corners([(float(p[0]), float(p[1])) for p in quad])
            self._status.setText(self.tr(
                "Auto-detected card. Drag any wrong corner, then click "
                "<b>Calibrate DPI</b>."
            ))
        else:
            # Seed a centred rectangle at the canonical ID-1 aspect so
            # the user can drag the 4 handles into place instead of
            # starting from scratch.
            self._pick.set_corners(self._seed_centre_quad(w, h))
            self._status.setText(self.tr(
                "Auto-detect failed. Drag the 4 corners onto the card, "
                "then click <b>Calibrate DPI</b>."
            ))

    def _seed_centre_quad(self, w: int, h: int) -> list[tuple[float, float]]:
        long_px = 0.35 * w
        short_px = long_px * self._id1_short_mm / self._id1_long_mm
        cx, cy = w / 2.0, h / 2.0
        hx, hy = long_px / 2.0, short_px / 2.0
        return [(cx - hx, cy - hy), (cx + hx, cy - hy),
                (cx + hx, cy + hy), (cx - hx, cy + hy)]

    def _on_corners_changed(self, corners: list[tuple[float, float]]) -> None:
        ready = (self._frame_bgr is not None and len(corners) == 4)
        self._btn_calibrate.setEnabled(ready)
        self._btn_reset.setEnabled(len(corners) > 0)

    def _on_reset(self) -> None:
        self._pick.clear_clicks()
        if self._frame_bgr is not None:
            self._status.setText(self.tr("Corners cleared. Click 4 points or recapture."))

    def _on_calibrate(self) -> None:
        from lib.workers.CreditCardDPI import refine_and_measure
        if self._frame_bgr is None:
            return
        corners = self._pick.corners()
        if len(corners) != 4:
            return
        quad = np.array(corners, dtype=np.float32)
        dpi, quad_ordered = refine_and_measure(self._frame_bgr, quad)
        zoom = float(getattr(self._webcam, "current_zoom", 1.0))
        base_dpi = dpi / max(zoom, 1e-6)
        self._status.setText(self.tr(
            "Measured {dpi:.1f} dpi at zoom {zoom:.2f}x → "
            "base {base_dpi:.1f} dpi @1.0x"
        ).format(dpi=dpi, zoom=zoom, base_dpi=base_dpi))
        self.calibration_committed.emit(
            float(dpi), float(base_dpi), float(zoom),
            self._frame_bgr, quad_ordered,
        )
