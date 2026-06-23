# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
EAST text detector backend via OpenCV dnn.

Model file: `frozen_east_text_detection.pb` (~95 MB). Resolved from:
  1. `AGLAIA_EAST_MODEL` env var (absolute path)
  2. `./model/frozen_east_text_detection.pb` (project-relative)
  3. Common absolute paths used in this repo's data layout
"""
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from aglaia.processors.layout_backends.base import LayoutBackend, BBox

MODEL_FILENAME = "frozen_east_text_detection.pb"
# cwd-relative first, then the repo root (parents[3] = repo) so models drop
# into ./model[s] resolve regardless of where the app is launched from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_DIRS = [
    "./model",
    "./models",
    str(_REPO_ROOT / "model"),
    str(_REPO_ROOT / "models"),
]


def _resolve_model_path() -> Path:
    env = os.environ.get("AGLAIA_EAST_MODEL")
    if env and Path(env).exists():
        return Path(env)
    # User-configurable models dir wins over repo-relative defaults.
    try:
        from aglaia.app_data import models_dir as _md
        cand = _md() / MODEL_FILENAME
        if cand.exists():
            return cand
    except Exception:
        pass
    for d in SEARCH_DIRS:
        p = Path(d) / MODEL_FILENAME
        if p.exists():
            return p
    raise FileNotFoundError(
        f"EAST model not found. Place {MODEL_FILENAME} under the configured "
        f"models dir or set AGLAIA_EAST_MODEL to its absolute path."
    )


def _decode(scores, geometry, min_confidence: float):
    """Decode EAST output into (rect, score) lists. Rects are [x, y, w, h]."""
    n_rows, n_cols = scores.shape[2:4]
    rects: list[list[int]] = []
    confidences: list[float] = []
    for y in range(n_rows):
        s_row = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(n_cols):
            score = s_row[x]
            if score < min_confidence:
                continue
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            end_x = int(offset_x + (cos_a * x1[x]) + (sin_a * x2[x]))
            end_y = int(offset_y - (sin_a * x1[x]) + (cos_a * x2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            rects.append([start_x, start_y, int(w), int(h)])
            confidences.append(float(score))
    return rects, confidences


class EastBackend(LayoutBackend):
    name = "east"

    def __init__(self, min_confidence: float = 0.5, nms_thr: float = 0.4):
        path = _resolve_model_path()
        try:
            self.net = cv2.dnn.readNet(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load EAST model at {path}: {e}")
        self.min_confidence = float(min_confidence)
        self.nms_thr = float(nms_thr)
        from aglaia.processors.layout_backends.dbnet import _try_enable_gpu
        self.uses_gpu = _try_enable_gpu(self.net)

    def detect(self, img_rgb: np.ndarray) -> List[BBox]:
        if self.net is None or img_rgb is None or img_rgb.size == 0:
            return []
        orig_h, orig_w = img_rgb.shape[:2]
        # EAST input dims must be multiples of 32.
        newW = (orig_w // 32) * 32
        newH = (orig_h // 32) * 32
        if newW == 0 or newH == 0:
            return []
        rW = orig_w / float(newW)
        rH = orig_h / float(newH)

        blob = cv2.dnn.blobFromImage(
            img_rgb, 1.0, (newW, newH),
            (123.68, 116.78, 103.94), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ])
        rects, confidences = _decode(scores, geometry, self.min_confidence)
        if not rects:
            return []

        idxs = cv2.dnn.NMSBoxes(rects, confidences, self.min_confidence, self.nms_thr)
        if idxs is None or len(idxs) == 0:
            return []

        out: List[BBox] = []
        for i in np.array(idxs).flatten():
            x, y, w, h = rects[i]
            x0 = int(max(0, x * rW))
            y0 = int(max(0, y * rH))
            x1 = int(min(orig_w, (x + w) * rW))
            y1 = int(min(orig_h, (y + h) * rH))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            out.append((x0, y0, x1, y1))
        return out
