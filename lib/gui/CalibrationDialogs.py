# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Medium-size modal dialogs for one-shot camera-side workflows.

Two dialogs live here:

* :class:`DpiCalibrationDialog` — live webcam preview with a running
  Apple-Vision card overlay. Two action buttons:

    * **Capture and refine** — freezes the current frame, keeps the
      auto-detected card quad, and runs ``refine_and_measure``
      straight away so the dialog closes one click after the user
      sees the box.
    * **Trace manually** — freezes the current frame but drops the
      auto-detected quad. The pane swaps to a draggable 4-corner
      canvas so the user can outline the card themselves. A
      **Calibrate** button finalises the trace.

* :class:`FreehandRegistrationDialog` — thin modal wrapper around
  :class:`lib.gui.FreehandTab.FreehandRegistrationTab` so the SIFT
  freehand-capture setup also lives in a self-contained popup rather
  than the main-window tab strip.

Both dialogs are sized to a comfortable medium (~720 × 560) so they
sit naturally next to the live capture without obscuring the whole
window.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)

from lib.gui.colors import (
    COLOR_BG_BUTTON_CHECKED, COLOR_BG_TOGGLE_ON, COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
)
from lib.gui.DpiCalibrationTab import _PickLabel
from lib.gui.FreehandTab import FreehandRegistrationTab


_DIALOG_W, _DIALOG_H = 720, 560

# Amber for "measure won't be reliable yet" hints (card too small / oblique).
_COLOR_HINT_WARN = "#d97706"

# Accent style for the primary action button at each stage, so the eye lands
# on it instead of scanning a flat row of identical buttons.
_PRIMARY_QSS = f"""
QPushButton {{
    background: {COLOR_BG_BUTTON_CHECKED};
    color: #ffffff;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:disabled {{ background: rgba(255,255,255,0.10); color: {COLOR_FONT_MUTED}; }}
"""


class _LivePreviewLabel(QLabel):
    """QLabel that paints the latest BGR frame fitted to its current
    size + an optional overlay polyline (the live card-detection quad)
    in the original frame's coords."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(360)
        self._frame: Optional[np.ndarray] = None
        self._quad: Optional[np.ndarray] = None

    def set_frame(self, bgr: Optional[np.ndarray]) -> None:
        self._frame = bgr
        self.update()

    def set_quad(self, quad: Optional[np.ndarray]) -> None:
        self._quad = quad
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        if self._frame is None:
            super().paintEvent(_ev)
            return
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        ww, wh = self.width(), self.height()
        scale = min(ww / w, wh / h) if w and h else 1.0
        dw, dh = int(w * scale), int(h * scale)
        dx = (ww - dw) // 2
        dy = (wh - dh) // 2

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(dx, dy, dw, dh, pix)
        if self._quad is not None and len(self._quad) == 4:
            pen = QPen(Qt.GlobalColor.green, 3)
            pen.setCosmetic(True)
            p.setPen(pen)
            pts = [(dx + q[0] * scale, dy + q[1] * scale) for q in self._quad]
            for i in range(4):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % 4]
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.end()


class DpiCalibrationDialog(QDialog):
    """Medium modal: live preview → capture-and-refine OR trace-manually."""

    calibration_committed = Signal(float, float, float, object, object)

    REFRESH_MS = 60          # live preview tick (~16 Hz)
    DETECT_MS = 400          # card-detect tick (~2.5 Hz)
    STEADY_TICKS = 3         # ~1.2 s of stable detection → auto-capture
    STEADY_CV = 0.05         # max coeff. of variation for "stable" DPI
    MIN_LONG_FRAC = 0.25     # card long side must fill ≥ this of frame width
    MAX_PERSP_RATIO = 1.25   # opposite-side length ratio above this = oblique

    def __init__(self, webcam_thread, *, id1_long_mm: float,
                 id1_short_mm: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Calibrate DPI"))
        self.setModal(False)
        self.resize(_DIALOG_W, _DIALOG_H)

        self._webcam = webcam_thread
        self._id1_long_mm = float(id1_long_mm)
        self._id1_short_mm = float(id1_short_mm)
        self._live_frame: Optional[np.ndarray] = None
        self._live_quad: Optional[np.ndarray] = None
        self._captured_frame: Optional[np.ndarray] = None
        self._captured_quad: Optional[np.ndarray] = None
        # Rolling live DPI estimates (for the median cross-check shown in
        # the refine/review step) + the pending measurement awaiting
        # confirmation.
        self._live_dpi_samples: deque = deque(maxlen=15)
        self._pending_dpi: Optional[float] = None
        self._pending_quad: Optional[np.ndarray] = None
        # Hold-steady auto-capture: count consecutive detect ticks where the
        # card is present, the geometry is reliable, and the DPI estimate is
        # stable. Fire capture-and-refine once it holds for STEADY_TICKS.
        self._steady_ticks = 0
        self._auto_fired = False

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 12)
        v.setSpacing(8)

        caption = QLabel(self.tr(
            "Hold an ISO ID-1 credit-card-sized object flat in the "
            "frame. The green outline is the auto-detected card."
        ))
        caption.setWordWrap(True)
        v.addWidget(caption)

        # Stack: 0 = live, 1 = manual trace.
        self._stack = QStackedWidget()
        v.addWidget(self._stack, 1)

        self._live_preview = _LivePreviewLabel()
        self._stack.addWidget(self._live_preview)

        self._pick = _PickLabel()
        self._pick.setMinimumHeight(360)
        self._pick._on_change = self._on_corners_changed
        self._stack.addWidget(self._pick)

        self._status = QLabel(self.tr("Waiting for camera…"))
        self._status.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-style: italic;"
        )
        self._status.setWordWrap(True)
        v.addWidget(self._status)

        # Hold-steady auto-capture toggle (live stage only).
        self._chk_auto = QCheckBox(self.tr("Auto-capture when the card holds steady"))
        self._chk_auto.setChecked(True)
        self._chk_auto.setStyleSheet(f"color: {COLOR_FONT_MUTED};")
        v.addWidget(self._chk_auto)

        # Button row — different sets depending on stage.
        self._btn_capture_refine = QPushButton(self.tr("Capture now"))
        self._btn_capture_refine.setDefault(True)
        self._btn_capture_refine.setStyleSheet(_PRIMARY_QSS)
        self._btn_capture_refine.clicked.connect(self._on_capture_refine)

        self._btn_trace_manual = QPushButton(self.tr("Trace manually"))
        self._btn_trace_manual.clicked.connect(self._on_trace_manual)

        self._btn_calibrate = QPushButton(self.tr("Calibrate DPI"))
        self._btn_calibrate.setEnabled(False)
        self._btn_calibrate.setStyleSheet(_PRIMARY_QSS)
        self._btn_calibrate.clicked.connect(self._on_calibrate_manual)

        self._btn_reset = QPushButton(self.tr("Reset corners"))
        self._btn_reset.clicked.connect(self._on_reset_corners)
        self._btn_reset.setEnabled(False)

        self._btn_back = QPushButton(self.tr("Back to live"))
        self._btn_back.clicked.connect(self._on_back_to_live)

        # Review step (after a measurement): confirm the value or redo.
        self._btn_confirm = QPushButton(self.tr("Use this DPI"))
        self._btn_confirm.setDefault(True)
        self._btn_confirm.setStyleSheet(_PRIMARY_QSS)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_recapture = QPushButton(self.tr("Recapture"))
        self._btn_recapture.clicked.connect(self._on_recapture)

        self._btn_cancel = QPushButton(self.tr("Cancel"))
        self._btn_cancel.clicked.connect(self.reject)

        # Primary action(s) on the left, a separator, Cancel pinned right —
        # only the 2-3 buttons relevant to the current stage are ever shown,
        # so the row reads cleanly instead of as a flat bank of choices.
        row = QHBoxLayout()
        row.setSpacing(8)
        for b in (self._btn_capture_refine, self._btn_trace_manual,
                  self._btn_back, self._btn_reset, self._btn_calibrate,
                  self._btn_confirm, self._btn_recapture):
            row.addWidget(b)
        row.addStretch(1)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.10);")
        row.addWidget(sep)
        row.addWidget(self._btn_cancel)
        v.addLayout(row)

        self._show_live_buttons()

        # Live preview timer + slower card-detect timer.
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._tick_preview)
        self._preview_timer.start(self.REFRESH_MS)

        self._detect_timer = QTimer(self)
        self._detect_timer.timeout.connect(self._tick_detect)
        self._detect_timer.start(self.DETECT_MS)

    # ── view state ──────────────────────────────────────────────────
    def _show_live_buttons(self) -> None:
        self._stack.setCurrentIndex(0)
        self._steady_ticks = 0
        self._auto_fired = False
        self._chk_auto.show()
        self._btn_capture_refine.show()
        self._btn_trace_manual.show()
        self._btn_calibrate.hide()
        self._btn_reset.hide()
        self._btn_back.hide()
        self._btn_confirm.hide()
        self._btn_recapture.hide()

    def _show_manual_buttons(self) -> None:
        self._stack.setCurrentIndex(1)
        self._chk_auto.hide()
        self._btn_capture_refine.hide()
        self._btn_trace_manual.hide()
        self._btn_calibrate.show()
        self._btn_reset.show()
        self._btn_back.show()
        self._btn_confirm.hide()
        self._btn_recapture.hide()

    def _show_review_buttons(self) -> None:
        # Frozen captured frame + refined quad on the live stack page.
        self._stack.setCurrentIndex(0)
        self._chk_auto.hide()
        self._btn_capture_refine.hide()
        self._btn_trace_manual.hide()
        self._btn_calibrate.hide()
        self._btn_reset.hide()
        self._btn_back.hide()
        self._btn_confirm.show()
        self._btn_recapture.show()

    # ── live ticks ──────────────────────────────────────────────────
    def _tick_preview(self) -> None:
        if self._webcam is None:
            return
        frame = self._webcam.get_frame()
        if frame is None:
            return
        self._live_frame = frame
        self._live_preview.set_frame(frame)

    def _tick_detect(self) -> None:
        if self._live_frame is None:
            return
        dpi = None
        try:
            from lib.workers.CreditCardDPI import detect_card_dpi
            dpi, quad = detect_card_dpi(self._live_frame)
        except Exception:
            quad = None
        self._live_quad = (
            np.asarray(quad, dtype=np.float32)
            if quad is not None and len(quad) == 4 else None
        )
        self._live_preview.set_quad(self._live_quad)
        if self._live_quad is None:
            self._live_dpi_samples.clear()
            self._steady_ticks = 0
            self._auto_fired = False
            self._set_status(self.tr(
                "Looking for a card — hold an ID-1 card flat in the frame."),
                COLOR_FONT_MUTED)
            return

        if dpi and dpi > 0:
            self._live_dpi_samples.append(float(dpi))
        med = self._median_live_dpi()

        # Geometry guidance — is this frame good enough to measure from?
        ok, hint = self._assess_quad(self._live_quad)
        stable = ok and self._dpi_is_stable()
        if not stable:
            self._steady_ticks = 0
            self._auto_fired = False

        if not ok:
            # Card present but the measure would be unreliable — say why.
            self._set_status(hint, _COLOR_HINT_WARN)
            return

        shown = self.tr(" — ≈ {med:.0f} dpi").format(med=med) if med else ""
        if self._chk_auto.isChecked() and stable:
            self._steady_ticks += 1
            remaining = max(0, self.STEADY_TICKS - self._steady_ticks)
            if self._steady_ticks >= self.STEADY_TICKS and not self._auto_fired:
                self._auto_fired = True
                self._set_status(self.tr(
                    "Card steady{dpi} — capturing…").format(dpi=shown),
                    COLOR_BG_TOGGLE_ON)
                self._on_capture_refine()
                return
            secs = remaining * self.DETECT_MS / 1000.0
            self._set_status(self.tr(
                "Card steady{dpi} — auto-capturing in {secs:.1f}s "
                "(or click <b>Capture now</b>).").format(dpi=shown, secs=secs),
                COLOR_BG_TOGGLE_ON)
        else:
            self._set_status(self.tr(
                "Card detected{dpi} — hold steady, or click "
                "<b>Capture now</b> / <b>Trace manually</b>."
            ).format(dpi=shown), COLOR_FONT_PRIMARY)

    def _set_status(self, text: str, color: str) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-style: italic;")

    def _dpi_is_stable(self) -> bool:
        """True when enough recent DPI samples agree (low coeff. of variation)."""
        if len(self._live_dpi_samples) < 4:
            return False
        arr = np.asarray(self._live_dpi_samples, dtype=np.float64)
        mean = float(arr.mean())
        if mean <= 0:
            return False
        return float(arr.std() / mean) <= self.STEADY_CV

    def _assess_quad(self, quad: np.ndarray) -> tuple[bool, str]:
        """Judge whether the detected card quad gives a reliable measure.

        Returns (ok, hint). Flags a card that is too small in frame (DPI from
        few pixels is noisy) or too oblique (strong perspective biases the
        long-edge length the DPI is derived from)."""
        from lib.workers.CreditCardDPI import ID1_ASPECT
        if self._live_frame is None or quad is None or len(quad) != 4:
            return False, self.tr("Looking for a card…")
        fw = float(self._live_frame.shape[1]) or 1.0
        q = np.asarray(quad, dtype=np.float64)
        # Side lengths around the quad (already corner-ordered by the detector).
        sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        long_side = max(sides)
        if long_side < self.MIN_LONG_FRAC * fw:
            return False, self.tr(
                "Card too small — move it closer so it fills more of the frame.")
        # Perspective: opposite sides should match for a square-on card.
        top, right, bottom, left = sides
        persp = max(top / max(bottom, 1e-6), bottom / max(top, 1e-6),
                    left / max(right, 1e-6), right / max(left, 1e-6))
        if persp > self.MAX_PERSP_RATIO:
            return False, self.tr(
                "Card looks tilted — hold it flat and square to the camera.")
        # Aspect sanity: a wildly wrong aspect means a mis-detection.
        aspect = long_side / max(min(sides), 1e-6)
        if abs(aspect - ID1_ASPECT) / ID1_ASPECT > 0.25:
            return False, self.tr(
                "Hold the whole card flat in view — the outline looks off.")
        return True, ""

    def _median_live_dpi(self) -> Optional[float]:
        if not self._live_dpi_samples:
            return None
        return float(np.median(np.asarray(self._live_dpi_samples)))

    # ── action handlers ─────────────────────────────────────────────
    def _on_capture_refine(self) -> None:
        if self._live_frame is None:
            self._status.setText(self.tr("No frame from camera."))
            return
        self._captured_frame = self._live_frame.copy()
        # Use the latest live detect — re-run detect on the captured
        # frame so we're refining the exact pixels we measured against.
        try:
            from lib.workers.CreditCardDPI import detect_card_dpi
            _dpi, quad = detect_card_dpi(self._captured_frame)
        except Exception:
            quad = None
        if quad is None or len(quad) != 4:
            self._status.setText(self.tr(
                "Auto-detect lost the card on capture — switch to "
                "<b>Trace manually</b>."
            ))
            return
        self._finalize(np.asarray(quad, dtype=np.float32))

    def _on_trace_manual(self) -> None:
        if self._live_frame is None:
            self._status.setText(self.tr("No frame from camera."))
            return
        self._captured_frame = self._live_frame.copy()
        # Stop the live + detect timers so the preview freezes and the
        # CPU isn't double-using SIFT/Vision on a frame we're not
        # showing any more.
        self._preview_timer.stop()
        self._detect_timer.stop()
        rgb = cv2.cvtColor(self._captured_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self._pick.set_image(QPixmap.fromImage(qimg), (w, h))
        self._pick.set_corners(self._seed_centre_quad(w, h))
        self._show_manual_buttons()
        self._status.setText(self.tr(
            "Drag the 4 corners onto the card, then click "
            "<b>Calibrate DPI</b>."
        ))

    def _on_back_to_live(self) -> None:
        self._captured_frame = None
        self._pick.clear_clicks()
        self._show_live_buttons()
        self._preview_timer.start(self.REFRESH_MS)
        self._detect_timer.start(self.DETECT_MS)

    def _on_reset_corners(self) -> None:
        self._pick.clear_clicks()
        self._status.setText(self.tr(
            "Corners cleared. Click 4 points or use <b>Back to live</b>."
        ))

    def _on_corners_changed(self, corners: list[tuple[float, float]]) -> None:
        self._btn_calibrate.setEnabled(
            self._captured_frame is not None and len(corners) == 4
        )
        self._btn_reset.setEnabled(len(corners) > 0)

    def _on_calibrate_manual(self) -> None:
        if self._captured_frame is None:
            return
        corners = self._pick.corners()
        if len(corners) != 4:
            return
        self._finalize(np.array(corners, dtype=np.float32))

    # ── helpers ─────────────────────────────────────────────────────
    def _seed_centre_quad(self, w: int, h: int
                          ) -> list[tuple[float, float]]:
        long_px = 0.35 * w
        short_px = long_px * self._id1_short_mm / self._id1_long_mm
        cx, cy = w / 2.0, h / 2.0
        hx, hy = long_px / 2.0, short_px / 2.0
        return [(cx - hx, cy - hy), (cx + hx, cy - hy),
                (cx + hx, cy + hy), (cx - hx, cy + hy)]

    def _finalize(self, quad: np.ndarray) -> None:
        """Measure (edge-refined) and enter the REVIEW step — the user sees
        the DPI + a live-median cross-check and confirms before it commits,
        instead of the old blind auto-commit."""
        from lib.workers.CreditCardDPI import refine_and_measure
        dpi, quad_ordered = refine_and_measure(self._captured_frame, quad)
        self._pending_dpi = float(dpi)
        self._pending_quad = quad_ordered
        # Freeze the captured frame + refined quad in the preview.
        self._preview_timer.stop()
        self._detect_timer.stop()
        self._live_preview.set_frame(self._captured_frame)
        self._live_preview.set_quad(
            np.asarray(quad_ordered, dtype=np.float32)
            if quad_ordered is not None else None
        )
        med = self._median_live_dpi()
        cross = self.tr(" · live median ≈ {med:.0f}").format(med=med) if med else ""
        self._status.setText(self.tr(
            "Measured ≈ <b>{dpi:.0f} dpi</b> (edge-refined){cross}. "
            "Click <b>Use this DPI</b> to apply, or <b>Recapture</b>."
        ).format(dpi=self._pending_dpi, cross=cross))
        self._show_review_buttons()

    def _on_confirm(self) -> None:
        if self._pending_dpi is None:
            return
        zoom = float(getattr(self._webcam, "current_zoom", 1.0))
        base_dpi = self._pending_dpi / max(zoom, 1e-6)
        self.calibration_committed.emit(
            float(self._pending_dpi), float(base_dpi), float(zoom),
            self._captured_frame, self._pending_quad,
        )
        self.accept()

    def _on_recapture(self) -> None:
        self._pending_dpi = None
        self._pending_quad = None
        self._captured_frame = None
        self._live_dpi_samples.clear()
        self._pick.clear_clicks()
        self._live_preview.set_quad(None)
        self._show_live_buttons()
        self._preview_timer.start(self.REFRESH_MS)
        self._detect_timer.start(self.DETECT_MS)

    # ── cleanup ─────────────────────────────────────────────────────
    def reject(self) -> None:
        self._preview_timer.stop()
        self._detect_timer.stop()
        super().reject()

    def accept(self) -> None:
        self._preview_timer.stop()
        self._detect_timer.stop()
        super().accept()


class FreehandRegistrationDialog(QDialog):
    """Medium modal wrapping :class:`FreehandRegistrationTab`. The
    inner widget already polls the webcam + paints its own overlay; we
    just give it a sized window + a Cancel button."""

    registered = Signal(object, tuple)

    def __init__(self, webcam_thread, *, side_px: int = 160,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Hands-free capture — register pattern"))
        self.setModal(False)
        self.resize(_DIALOG_W, _DIALOG_H)

        self._inner = FreehandRegistrationTab(
            webcam_thread, side_px=side_px, parent=self,
        )
        self._inner.registered.connect(self._on_registered)
        self._inner.cancel_requested.connect(self.reject)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._inner)

    def _on_registered(self, patch, roi) -> None:
        self.registered.emit(patch, roi)
        self.accept()

    def reject(self) -> None:
        try:
            self._inner.stop()
        except Exception:
            pass
        super().reject()

    def accept(self) -> None:
        try:
            self._inner.stop()
        except Exception:
            pass
        super().accept()
