# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Bridge camera thread — a ``WebcamThread`` look-alike backed by a phone (#49).

``MainWindow`` duck-types on the webcam thread (``get_frame`` = full-res still,
``change_pixmap_signal`` = downscaled preview, ``current_zoom``/``max_zoom``,
``set_transform``, ``stop``). By mirroring that surface over a
:class:`~aglaia.workers.bridge_live.BridgeLiveServer`, live capture, voice, and
zoom flow through the *same* ``capture()`` path as the webcam — no pipeline
changes.

- Preview: the phone POSTs low-res JPEG frames; ``run`` re-emits the newest one
  (after transform + digital-zoom crop) via ``change_pixmap_signal``.
- Full-res still: ``get_frame`` asks the phone for a still (blocking, ~0.5–2 s)
  and applies the same transform + zoom crop, so a shutter/voice/DPI capture
  gets sensor-resolution pixels.

Zoom is a **desktop-side digital crop** (center-crop ``1/zoom`` then resize back
up) applied identically to preview and still — no wire command — so
``MainWindow.effective_dpi() = base × zoom`` keeps working unchanged.
"""

from __future__ import annotations

import time

import cv2
from PySide6.QtCore import Signal

from aglaia.gui.WebcamThread import WebcamThread
from aglaia.workers.bridge_live import BridgeLiveServer, BridgeSessionLost

# Digital zoom only — past the still's real resolution this is empty
# magnification, so cap lower than a webcam's optical range.
BRIDGE_MAX_ZOOM = 3.0
# Reuse one remote still across capture triggers this close together (a shutter
# and a follow-up read) so we don't fire two round-trips for one intent.
STILL_CACHE_SECONDS = 1.0


class BridgeCameraThread(WebcamThread):
    """Feeds :class:`MainWindow` from a paired phone instead of a local camera."""

    is_bridge = True

    session_lost = Signal(str)   # reason — the phone/session went away
    still_failed = Signal()      # a full-res still request timed out

    def __init__(self, server: BridgeLiveServer) -> None:
        super().__init__(camera_id=-1)
        self._server = server
        self.max_zoom = BRIDGE_MAX_ZOOM
        self.current_zoom = 1.0
        self._last_seq = -1
        self._preview_frame = None          # last transformed+cropped preview
        self._still: cv2.typing.MatLike | None = None
        self._still_at = 0.0

    # ── preview loop ─────────────────────────────────────────────────
    def run(self) -> None:
        self._run_flag = True
        while self._run_flag:
            if not self._server.session_alive:
                self.session_lost.emit("phone disconnected")
                break
            got = self._server.latest_preview()
            if got is not None:
                frame, seq = got
                if seq != self._last_seq:
                    self._last_seq = seq
                    disp = self._zoom_crop(self._apply_transform(frame), cv2.INTER_LINEAR)
                    with self.lock:
                        self._preview_frame = disp
                    self._emit_preview(disp)
            self.msleep(33)  # ~30 Hz repaint; the phone caps the real frame rate

    def get_preview_frame(self):
        """Last low-res preview frame (transformed + zoom-cropped). Used by the
        DPI dialog's live ticks so they never trigger a remote still."""
        with self.lock:
            return None if self._preview_frame is None else self._preview_frame.copy()

    # ── full-res still ───────────────────────────────────────────────
    def get_frame(self):
        """Fetch a full-res still from the phone (blocking). Returns ``None`` and
        signals on timeout / lost session so callers degrade gracefully."""
        now = time.monotonic()
        with self.lock:
            if self._still is not None and now - self._still_at < STILL_CACHE_SECONDS:
                return self._still.copy()
        try:
            still = self._server.request_still(timeout=6.0)
        except BridgeSessionLost:
            self.session_lost.emit("phone disconnected")
            return None
        except (TimeoutError, ValueError):
            self.still_failed.emit()
            return None
        # CUBIC on the output path (the still feeds the pipeline); LINEAR is only
        # for the throwaway preview.
        still = self._zoom_crop(self._apply_transform(still), cv2.INTER_CUBIC)
        with self.lock:
            self._still = still
            self._still_at = time.monotonic()
        return still.copy()

    # ── zoom (desktop-side digital crop) ─────────────────────────────
    def _zoom_crop(self, img, interp):
        z = self.current_zoom
        if z <= 1.0 + 1e-6:
            return img
        h, w = img.shape[:2]
        cw, ch = max(1, round(w / z)), max(1, round(h / z))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        crop = img[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=interp)

    def set_zoom(self, factor: float) -> float:
        # Pure desktop-side state — no AVFoundation device, no wire command.
        self.current_zoom = max(1.0, min(float(factor), self.max_zoom))
        return self.current_zoom

    # ── passthroughs to the session ──────────────────────────────────
    @property
    def device_name(self) -> str | None:
        return self._server.device_name

    @property
    def still_dims(self) -> tuple[int, int] | None:
        return self._server.still_dims
