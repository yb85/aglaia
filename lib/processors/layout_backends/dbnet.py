# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
PP-OCRv4 mobile text detection backend via OpenCV dnn (DBNet++ family).

Smaller (~5 MB) and more accurate than EAST. Loaded as ONNX via cv2.dnn so
no Paddle / onnxruntime install is required.

Model file: `en_PP-OCRv4_mobile_det.onnx`. Pre-converted ONNX releases are
available from the RapidOCR project:
  https://github.com/RapidAI/RapidOCR/releases

Resolution order:
  1. `AGLAIA_DBNET_MODEL` env var (absolute path)
  2. `./model/en_PP-OCRv4_mobile_det.onnx`
  3. Common absolute paths used by this repo
"""
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from lib.processors.layout_backends.base import LayoutBackend, BBox

# PP-OCR det model — any of v3/v4/v5, mobile or server. Same DBNet pipeline.
KNOWN_FILENAMES = [
    "en_PP-OCRv5_mobile_det.onnx",
    "en_PP-OCRv5_server_det.onnx",
    "PP-OCRv5_mobile_det.onnx",
    "PP-OCRv5_server_det.onnx",
    "en_PP-OCRv4_mobile_det.onnx",
    "en_PP-OCRv4_server_det.onnx",
    "PP-OCRv4_mobile_det.onnx",
    "PP-OCRv4_server_det.onnx",
    "en_PP-OCRv3_det_mobile.onnx",
    "en_PP-OCRv3_mobile_det.onnx",
    "PP-OCRv3_mobile_det.onnx",
]
# cwd-relative first, then the repo root (parents[3] = repo) so models drop
# into ./model[s] resolve regardless of where the app is launched from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_DIRS = [
    "./model",
    "./models",
    str(_REPO_ROOT / "model"),
    str(_REPO_ROOT / "models"),
]

# ImageNet normalization (Paddle convention, scaled to 0-255 space).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def _resolve_model_path() -> Path:
    env = os.environ.get("AGLAIA_DBNET_MODEL")
    if env and Path(env).exists():
        return Path(env)
    dirs: list[str] = []
    try:
        from lib.app_data import models_dir as _md
        dirs.append(str(_md()))
    except Exception:
        pass
    dirs.extend(SEARCH_DIRS)
    for d in dirs:
        d_path = Path(d)
        if not d_path.exists():
            continue
        for fname in KNOWN_FILENAMES:
            p = d_path / fname
            if p.exists():
                return p
        # Fallback: anything matching PP-OCR*det*.onnx in the dir
        matches = sorted(d_path.glob("*PP-OCR*det*.onnx")) + \
                  sorted(d_path.glob("*PP-OCR*Det*.onnx"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "PP-OCR detection ONNX not found. Looked in: "
        + ", ".join(SEARCH_DIRS)
        + ". Set AGLAIA_DBNET_MODEL to the absolute path, or drop the "
        "det model under model/ or models/ (any PP-OCR v3/v4/v5 det ONNX is fine)."
    )


def _round32(n: int) -> int:
    return max(32, ((n + 31) // 32) * 32)


def _normalize(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 image → (1, 3, H, W) float32 blob."""
    x = img_rgb.astype(np.float32)
    x -= _MEAN
    x /= _STD
    return x.transpose(2, 0, 1)[np.newaxis, ...]


def _try_enable_gpu(net) -> bool:
    """Enable CUDA backend if the local OpenCV build supports it."""
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Some builds silently fall back; the only sure check is a dummy forward.
        dummy = np.zeros((1, 3, 32, 32), dtype=np.float32)
        net.setInput(dummy)
        net.forward()
        return True
    except Exception:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass
        return False


class DbnetBackend(LayoutBackend):
    """
    PP-OCRv4 mobile detection (~5 MB ONNX). DBNet++-based.

    Output of the network is a probability map. We threshold, find contours,
    fit minAreaRects and return axis-aligned bboxes scaled back to original
    image coordinates. Unclip (PaddleOCR's Vatti polygon dilation) is omitted
    — `PageDetector.margin_mm` already pads the merged-column crop.
    """

    name = "dbnet"

    def __init__(self, target_size: int = 960, bin_thr: float = 0.3,
                 box_thr: float = 0.5, min_area_px: float = 16.0):
        path = _resolve_model_path()
        try:
            self.net = cv2.dnn.readNetFromONNX(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load PP-OCRv4 det model at {path}: {e}")
        self.target_size = int(target_size)
        self.bin_thr = float(bin_thr)
        self.box_thr = float(box_thr)
        self.min_area_px = float(min_area_px)
        # Probe for a GPU backend (CUDA-built OpenCV).
        self.uses_gpu = _try_enable_gpu(self.net)

    def detect(self, img_rgb: np.ndarray) -> List[BBox]:
        if self.net is None or img_rgb is None or img_rgb.size == 0:
            return []
        H, W = img_rgb.shape[:2]
        scale = self.target_size / max(H, W)
        newH = _round32(int(H * scale))
        newW = _round32(int(W * scale))
        if newH == 0 or newW == 0:
            return []
        rH = H / float(newH)
        rW = W / float(newW)

        resized = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
        blob = _normalize(resized)
        self.net.setInput(blob)
        pred = self.net.forward()  # (1, 1, newH, newW) typically
        prob = pred[0, 0] if pred.ndim == 4 else pred[0]

        binary = (prob > self.bin_thr).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out: List[BBox] = []
        for c in contours:
            if len(c) < 3:
                continue
            area = cv2.contourArea(c)
            if area < self.min_area_px:
                continue
            # Score: mean network probability inside the contour mask.
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [c], -1, 255, -1)
            inside = prob[mask > 0]
            if inside.size == 0:
                continue
            score = float(inside.mean())
            if score < self.box_thr:
                continue

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            box[:, 0] *= rW
            box[:, 1] *= rH
            x0 = int(max(0, box[:, 0].min()))
            y0 = int(max(0, box[:, 1].min()))
            x1 = int(min(W, box[:, 0].max()))
            y1 = int(min(H, box[:, 1].max()))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            out.append((x0, y0, x1, y1))
        return out
