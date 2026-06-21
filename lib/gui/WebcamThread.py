# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
import sys
import threading
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage


def _avf_device(camera_id: int):
    """Return the AVCaptureDevice for a given cv2 AVFoundation index, or None.
    Index ordering matches CAP_AVFOUNDATION enumeration."""
    try:
        import AVFoundation as AV
    except ImportError:
        return None
    types = [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    for opt in ("AVCaptureDeviceTypeExternal",
                "AVCaptureDeviceTypeContinuityCamera"):
        if hasattr(AV, opt):
            types.append(getattr(AV, opt))
    session = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AV.AVMediaTypeVideo, AV.AVCaptureDevicePositionUnspecified
    )
    devs = list(session.devices())
    if 0 <= camera_id < len(devs):
        return devs[camera_id]
    return None


def _camera_label(camera_id: int) -> str:
    """Resolve the AVFoundation localizedName for a given cv2 device index.
    Returns "?" on non-macOS / when AVFoundation enumeration fails."""
    try:
        dev = _avf_device(camera_id)
        if dev is not None:
            return str(dev.localizedName())
    except Exception:
        pass
    return "?"


class WebcamThread(QThread):
    change_pixmap_signal = Signal(QImage)

    def __init__(self, camera_id=0, format_index=None):
        super().__init__()
        self.camera_id = camera_id
        # Optional AVCaptureDevice.formats() index to force a specific
        # format. None → auto-pick the widest field-of-view landscape
        # format (matches Photo Booth; avoids the cropped/zoomed default
        # some Continuity Cameras hand out).
        self.format_index = format_index
        self._run_flag = True
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.rotation = 0 # 0, 90, 180, 270
        self.mirror = False
        self.flip = False
        # AVCaptureDevice handle for zoom/exposure/focus control. Resolved
        # at start() time; None for cv2 backends that lack AVF.
        self._avf_dev = None
        self.max_zoom = 1.0
        self.current_zoom = 1.0
        # Optional per-frame BGR mutator. Called with the post-transform
        # frame just before the BGR→RGB→QImage conversion; returns the
        # frame to actually display. Used by the freehand-capture
        # tracker to paint SIFT keypoints over the live preview.
        self._overlay_fn = None
        # Preview is downscaled to this longest-edge before convert/overlay/
        # emit — the live feed never needs full sensor res (capture grabs
        # latest_frame at full res). ~960 px ≈ 480p/720p range, plenty for the
        # small preview pane, and cuts the per-frame cvtColor + overlay copy
        # from ~12 MP to <1 MP.
        self.preview_max = 960

    def set_preview_max(self, px: int) -> None:
        self.preview_max = max(320, int(px))

    def _emit_preview(self, cv_img) -> None:
        """Downscale-first preview emit shared by the real + fake loops."""
        ph, pw = cv_img.shape[:2]
        longest = max(pw, ph)
        if longest > self.preview_max:
            s = self.preview_max / float(longest)
            cv_img = cv2.resize(
                cv_img, (max(1, int(pw * s)), max(1, int(ph * s))),
                interpolation=cv2.INTER_AREA)
        disp = cv_img
        if self._overlay_fn is not None:
            # Overlay (capture flash) runs on the small image — the flash is
            # resolution-independent. SIFT keypoint coords would be wrong on a
            # downscaled frame, but SIFT is disabled; re-enabling it needs a
            # scale-aware overlay.
            try:
                out = self._overlay_fn(cv_img.copy())
                if out is not None:
                    disp = out
            except Exception:
                pass
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # `.copy()` detaches the QImage from the numpy buffer before emit.
        qimg = QImage(rgb.data, w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()
        self.change_pixmap_signal.emit(qimg)

    def set_overlay_fn(self, fn) -> None:
        """`fn(bgr) -> bgr` runs on every emitted frame. Pass `None` to
        clear. The callback runs on the webcam thread, so it must be
        thread-safe and quick (sub-frame budget)."""
        self._overlay_fn = fn

    def set_transform(self, transform_str):
        transform_str = str(transform_str).lower()
        
        # Reset
        self.rotation = 0
        self.mirror = False
        self.flip = False
        
        # Parse Rotation
        if "-90" in transform_str:
            self.rotation = 270 # -90 is 270 CW
        elif "90" in transform_str:
            self.rotation = 90
        elif "180" in transform_str:
            self.rotation = 180
            
        # Parse Modifiers
        if "mirror" in transform_str:
            self.mirror = True
        if "flip" in transform_str:
            self.flip = True

    def run(self):
        # Debug / CI: feed a still image as the camera so the capture UI +
        # pipeline can be driven headlessly with no hardware.
        # AGLAIA_FAKE_CAMERA=/path/img.(jpg|png) — emits that frame on a loop.
        import os as _os
        fake = _os.environ.get("AGLAIA_FAKE_CAMERA")
        if fake:
            self._run_fake(fake)
            return
        # Force AVFoundation backend on macOS so the device index aligns
        # with AVCaptureDeviceDiscoverySession's ordering (same list the
        # picker shows). CAP_ANY can silently fall back to a different
        # backend with a different enumeration.
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        label = "?"
        if sys.platform == "darwin":
            self._avf_dev = _avf_device(self.camera_id)
            if self._avf_dev is not None:
                label = str(self._avf_dev.localizedName())
                try:
                    self.max_zoom = float(self._avf_dev.activeFormat().videoMaxZoomFactor())
                    self.current_zoom = float(self._avf_dev.videoZoomFactor())
                except Exception:
                    pass
        # Select the capture format + reset zoom. Continuity Cameras
        # (iPhone) advertise several landscape formats with different
        # crops/zoom and persist videoZoomFactor across sessions, so the
        # default cv2 hands out is often zoomed/cropped vs. Photo Booth.
        # Pick the widest field-of-view format (or self.format_index if the
        # user chose one), force it via AVFoundation, and reset zoom to 1.
        if self._avf_dev is not None:
            fmt, fw, fh = self._choose_format(self._avf_dev)
            try:
                ok, _err = self._avf_dev.lockForConfiguration_(None)
                if ok:
                    # Reset zoom FIRST — it's the main fix (Continuity
                    # persists videoZoomFactor across sessions) and must not
                    # be skipped if the format change below fails.
                    try:
                        self._avf_dev.setVideoZoomFactor_(1.0)
                    except Exception as e:
                        print(f"[WebcamThread] zoom reset err: {e}", flush=True)
                    if fmt is not None:
                        try:
                            self._avf_dev.setActiveFormat_(fmt)
                        except Exception as e:
                            print(f"[WebcamThread] setActiveFormat err: {e}", flush=True)
                    self._avf_dev.unlockForConfiguration()
            except Exception as e:
                print(f"[WebcamThread] config lock err: {e}", flush=True)
            # Mirror the chosen dims into cv2 as a fallback in case the
            # active-format change didn't propagate to its buffer.
            if fw and fh and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
            # Refresh zoom bounds from the (possibly new) active format.
            try:
                self.max_zoom = float(self._avf_dev.activeFormat().videoMaxZoomFactor())
                self.current_zoom = float(self._avf_dev.videoZoomFactor())
            except Exception:
                pass
        opened = self.cap.isOpened()
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        print(f"[WebcamThread] camera_id={self.camera_id} backend={backend} "
              f"opened={opened} device={label!r} max_zoom={self.max_zoom:.1f} "
              f"res={actual_w}x{actual_h}",
              flush=True)
        first_frame_logged = False
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                if not first_frame_logged:
                    fh, fw = cv_img.shape[:2]
                    print(f"[WebcamThread] first frame shape={fw}x{fh} "
                          f"aspect={fw / max(fh, 1):.2f}",
                          flush=True)
                    first_frame_logged = True
                # Apply rotation
                if self.rotation == 90:
                    cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 180:
                    cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
                elif self.rotation == 270:
                    cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                if self.mirror:
                    cv_img = cv2.flip(cv_img, 1)
                if self.flip:
                    cv_img = cv2.flip(cv_img, 0)
                    
                with self.lock:
                    self.latest_frame = cv_img.copy()

                # latest_frame (above) stays full-res for capture/get_frame;
                # the preview is downscaled-first inside _emit_preview.
                self._emit_preview(cv_img)

                # Cap FPS (e.g. 30 FPS)
                self.msleep(30)
            else:
                self.sleep(1)
        self.cap.release()

    def _run_fake(self, path: str) -> None:
        """Emit a still image as the live camera (AGLAIA_FAKE_CAMERA). Same
        transform + overlay + emit path as ``run`` so the capture Uj behaves
        identically, minus the hardware. Stops when ``stop()`` clears the
        run flag, like the real loop."""
        img = cv2.imread(path)
        if img is None:
            import numpy as np
            img = np.full((1200, 900, 3), 245, np.uint8)
            cv2.putText(img, "FAKE", (60, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        4, (40, 40, 40), 6)
        self.max_zoom = 1.0
        self.current_zoom = 1.0
        print(f"[WebcamThread] FAKE camera: {path} "
              f"shape={img.shape[1]}x{img.shape[0]}", flush=True)
        while self._run_flag:
            cv_img = img.copy()
            if self.rotation == 90:
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
            elif self.rotation == 270:
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if self.mirror:
                cv_img = cv2.flip(cv_img, 1)
            if self.flip:
                cv_img = cv2.flip(cv_img, 0)
            with self.lock:
                self.latest_frame = cv_img.copy()
            self._emit_preview(cv_img)
            self.msleep(100)

    def _choose_format(self, dev):
        """Return ``(avformat, width, height)`` to activate. Honours
        ``self.format_index`` when set, else picks the widest field-of-view
        landscape format (tie-break: tallest, then largest). Returns
        ``(None, 0, 0)`` when AVFoundation isn't available."""
        try:
            import AVFoundation as AV
        except Exception:
            return (None, 0, 0)
        formats = list(dev.formats())

        def dims(fmt):
            d = AV.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
            return int(d.width), int(d.height)

        if self.format_index is not None and 0 <= self.format_index < len(formats):
            fmt = formats[self.format_index]
            fw, fh = dims(fmt)
            return (fmt, fw, fh)

        best = None  # (key, fmt, fw, fh)
        for fmt in formats:
            try:
                fw, fh = dims(fmt)
                # Landscape, ≥ 4:3 — drops square/portrait centre-crop modes.
                if fw <= 0 or fh <= 0 or fw < fh * 4 // 3:
                    continue
                try:
                    fov = float(fmt.videoFieldOfView())
                except Exception:
                    fov = 0.0
                key = (fov, fw * fh)   # widest FOV if known, else full sensor
                if best is None or key > best[0]:
                    best = (key, fmt, fw, fh)
            except Exception:
                continue
        if best is None:
            return (None, 0, 0)
        return (best[1], best[2], best[3])

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def rotate(self, delta=90):
        self.rotation = (self.rotation + delta) % 360

    def toggle_mirror(self):
        self.mirror = not self.mirror

    def toggle_flip(self):
        self.flip = not self.flip

    def set_zoom(self, factor: float) -> float:
        """Set videoZoomFactor on the underlying AVCaptureDevice. Returns
        the actual zoom applied (clamped to [1, max_zoom]). Affects cv2's
        live capture because AVCaptureDevice config is per-device, not
        per-session."""
        if self._avf_dev is None:
            return self.current_zoom
        f = max(1.0, min(float(factor), self.max_zoom))
        try:
            ok, _err = self._avf_dev.lockForConfiguration_(None)
            if not ok:
                return self.current_zoom
            self._avf_dev.setVideoZoomFactor_(f)
            self._avf_dev.unlockForConfiguration()
            self.current_zoom = float(self._avf_dev.videoZoomFactor())
        except Exception as e:
            print(f"[WebcamThread] zoom set err: {e}", flush=True)
        return self.current_zoom

    def stop(self):
        self._run_flag = False
        self.wait()
