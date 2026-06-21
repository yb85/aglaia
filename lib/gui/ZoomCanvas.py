# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Reusable image viewer with cursor-following Picture-in-Picture zoom.

`ZoomCanvas` paints a QPixmap fit to its bounds. When the cursor hovers
inside the image rect, a fixed-size PiP rectangle anchored bottom-left
shows a magnified crop centred on the cursor; a dashed marker on the
main image highlights the source region of that crop.

`ZoomToolbar` is a thin row of [label, slider, value] bound to a canvas
— hosts that already own a toolbar can skip it and call
`canvas.set_zoom(...)` directly.

Used by the pipeline editor preview panel and the per-node debug viewer.
"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QSizePolicy, QSlider, QWidget,
)

from lib.gui.colors import (
    COLOR_ERROR,
    COLOR_FONT_SECONDARY,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_SCRIM_MEDIUM,
    qcolor,
)


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert a BGR/Gray ndarray to a QPixmap, copy-safe.

    Color images are assumed BGR (OpenCV convention). 4-channel BGRA is
    flattened to RGB. `QPixmap.fromImage` deep-copies into the pixmap's
    native format, so the QImage (a view onto the ndarray) needs no extra
    `.copy()` — the ndarray can be freed right after."""
    if img.ndim == 2:
        img = np.ascontiguousarray(img)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, c = img.shape
        if c == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ZoomCanvas(QWidget):
    """Image surface with bottom-left PiP zoom under cursor.

    `set_image` accepts either a QPixmap or an ndarray. `set_zoom`
    accepts a float; values ≤1 hide the PiP."""

    PIP_W = 240
    PIP_H = 240
    PIP_MARGIN = 12

    def __init__(self, parent: Optional[QWidget] = None,
                 *, placeholder: Optional[str] = None,
                 min_size: tuple[int, int] = (360, 280)):
        super().__init__(parent)
        if placeholder is None:
            placeholder = self.tr("No image")
        self._pix: Optional[QPixmap] = None
        self._zoom: float = 2.0
        self._cursor: Optional[QPoint] = None
        # When set, the PiP follows these source-pixel coords instead of
        # the live cursor. Click on the canvas to pin/unpin. Survives
        # auto-refresh: on a new image we clamp the pin minimally back
        # into bounds rather than dropping it.
        self._pinned_src: Optional[tuple[int, int]] = None
        # PiP anchor side. Flips left↔right each time the cursor enters
        # the current PiP rect (only while unpinned) so the overlay never
        # traps the cursor against a wall.
        self._pip_side: str = "left"
        self._pip_was_inside: bool = False
        self._placeholder = placeholder
        self.setMinimumSize(QSize(*min_size))
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background-color: {COLOR_SCRIM_MEDIUM};"
            f"border: 1px solid {COLOR_OUTLINE_SUBTLE};"
            "border-radius: 8px;"
        )

    # ── public API ─────────────────────────────────────────────────
    def set_image(self, src: Union[QPixmap, np.ndarray, None]) -> None:
        if src is None:
            self._pix = None
        elif isinstance(src, QPixmap):
            self._pix = src if not src.isNull() else None
        elif isinstance(src, np.ndarray):
            self._pix = numpy_to_qpixmap(src) if src.size else None
        else:
            raise TypeError(f"ZoomCanvas.set_image: unsupported type {type(src)}")
        # Auto-refresh path: clamp the pinned source pixel back into the
        # new image's bounds rather than dropping the pin. Best effort —
        # if the new image is smaller in either axis we shift minimally.
        if self._pinned_src is not None and self._pix is not None:
            sx, sy = self._pinned_src
            self._pinned_src = (
                max(0, min(self._pix.width() - 1, sx)),
                max(0, min(self._pix.height() - 1, sy)),
            )
        elif self._pix is None:
            self._pinned_src = None
        self.update()

    def set_zoom(self, z: float) -> None:
        self._zoom = max(1.0, float(z))
        self.update()

    def has_image(self) -> bool:
        return self._pix is not None

    # ── geometry helpers ──────────────────────────────────────────
    def _fit_rect(self) -> QRect:
        if self._pix is None:
            return QRect()
        ww, wh = self.width(), self.height()
        pw, ph = self._pix.width(), self._pix.height()
        if pw == 0 or ph == 0:
            return QRect()
        scale = min(ww / pw, wh / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        return QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def _fit_scale(self) -> float:
        if self._pix is None:
            return 1.0
        r = self._fit_rect()
        return r.width() / self._pix.width() if self._pix.width() else 1.0

    def _pip_dst_rect(self) -> QRect:
        if self._pip_side == "right":
            x = self.width() - self.PIP_W - self.PIP_MARGIN
        else:
            x = self.PIP_MARGIN
        y = self.height() - self.PIP_H - self.PIP_MARGIN
        return QRect(x, y, self.PIP_W, self.PIP_H)

    # ── events ────────────────────────────────────────────────────
    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        pos = (
            ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        )
        # Flip PiP side on enter (unpinned only). Edge-triggered so
        # boundary-hovering doesn't oscillate.
        if (
            self._pix is not None
            and self._pinned_src is None
            and self._zoom > 1.001
        ):
            inside = self._pip_dst_rect().contains(pos)
            if inside and not self._pip_was_inside:
                self._pip_side = "right" if self._pip_side == "left" else "left"
            self._pip_was_inside = inside
        else:
            self._pip_was_inside = False
        self._cursor = pos
        self.update()

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Left-click pins / unpins the PiP location.

        Pinned: PiP locks on the clicked source-pixel; the marker wire
        turns red. Click anywhere on the canvas while pinned to release.
        Right-click does nothing here so the host can wire context menus.
        """
        if ev.button() != Qt.MouseButton.LeftButton or self._pix is None:
            super().mousePressEvent(ev)
            return
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if self._pinned_src is not None:
            self._pinned_src = None
            self.update()
            return
        fit = self._fit_rect()
        if not fit.contains(pos):
            return
        rel_x = (pos.x() - fit.x()) / fit.width()
        rel_y = (pos.y() - fit.y()) / fit.height()
        sx = int(rel_x * self._pix.width())
        sy = int(rel_y * self._pix.height())
        self._pinned_src = (sx, sy)
        self.update()

    def leaveEvent(self, _ev) -> None:  # noqa: N802
        # Pinned PiP must survive the cursor leaving the canvas — only
        # blow away the live-cursor track.
        self._cursor = None
        self._pip_was_inside = False
        self.update()

    # ── paint ─────────────────────────────────────────────────────
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._pix is None:
            p.setPen(QPen(Qt.GlobalColor.gray))
            p.drawText(self.rect(),
                       int(Qt.AlignmentFlag.AlignCenter),
                       self._placeholder)
            p.end()
            return
        fit = self._fit_rect()
        p.drawPixmap(fit, self._pix, self._pix.rect())
        self._paint_pip(p, fit)
        p.end()

    def _paint_pip(self, p: QPainter, fit: QRect) -> None:
        if self._pix is None or self._zoom <= 1.001:
            return
        # Pin takes priority over the live cursor — that's the whole
        # point of pinning.
        pinned = self._pinned_src is not None
        if pinned:
            cx, cy = self._pinned_src
        else:
            if self._cursor is None or not fit.contains(self._cursor):
                return
            rel_x = (self._cursor.x() - fit.x()) / fit.width()
            rel_y = (self._cursor.y() - fit.y()) / fit.height()
            cx = int(rel_x * self._pix.width())
            cy = int(rel_y * self._pix.height())
        src_w = self._pix.width()
        src_h = self._pix.height()
        eff = self._fit_scale() * self._zoom
        if eff <= 0:
            return
        crop_w = max(8, int(self.PIP_W / eff))
        crop_h = max(8, int(self.PIP_H / eff))
        sx = max(0, min(src_w - crop_w, cx - crop_w // 2))
        sy = max(0, min(src_h - crop_h, cy - crop_h // 2))
        src_rect = QRect(sx, sy, crop_w, crop_h)
        dst_rect = self._pip_dst_rect()
        # Red when pinned, primary-blue when tracking the cursor — pure
        # yellow read as washed-out on both themes. The brand accent
        # gives the user a quick visual receipt that the next click
        # will release.
        wire_color = qcolor(COLOR_ERROR) if pinned else qcolor(COLOR_PRIMARY)
        p.save()
        p.setPen(QPen(wire_color, 1))
        p.drawPixmap(dst_rect, self._pix, src_rect)
        p.drawRect(dst_rect.adjusted(0, 0, -1, -1))
        p.setPen(QPen(wire_color, 1, Qt.PenStyle.DashLine))
        fit_scale = self._fit_scale()
        marker = QRect(
            fit.x() + int(sx * fit_scale),
            fit.y() + int(sy * fit_scale),
            max(2, int(crop_w * fit_scale)),
            max(2, int(crop_h * fit_scale)),
        )
        p.drawRect(marker)
        p.restore()


class ZoomToolbar(QWidget):
    """Row of [label, slider, value-readout] bound to a `ZoomCanvas`.

    Slider raw range is fixed in tenths (10..100 → 1.0×..10.0×) to match
    the rest of the UI. `zoom_changed` re-fires the float value too in
    case the host wants its own readout."""

    zoom_changed = Signal(float)

    def __init__(self, canvas: ZoomCanvas, *,
                 label_text: Optional[str] = None,
                 default: float = 2.0,
                 slider_width: int = 110,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        if label_text is None:
            label_text = self.tr("Zoom")
        self._canvas = canvas
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        row.addWidget(QLabel(label_text))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(10, 100)
        self._slider.setValue(max(10, min(100, int(round(default * 10)))))
        self._slider.setFixedWidth(slider_width)
        self._slider.valueChanged.connect(self._on_changed)
        row.addWidget(self._slider)

        self._val = QLabel(self.tr("{value:.1f}×").format(value=default))
        self._val.setStyleSheet(
            f"color: {COLOR_FONT_SECONDARY}; min-width: 32px;"
        )
        row.addWidget(self._val)

        # Sync initial state.
        self._canvas.set_zoom(default)

    def _on_changed(self, raw: int) -> None:
        z = raw / 10.0
        self._val.setText(self.tr("{value:.1f}×").format(value=z))
        self._canvas.set_zoom(z)
        self.zoom_changed.emit(z)

    def value(self) -> float:
        return self._slider.value() / 10.0
