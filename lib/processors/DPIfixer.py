# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
from dataclasses import dataclass
from lib.ImageBuffer import ImageBuffer, ImageType
from lib.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait
from lib.processors.option_specs import _i
from typing import Optional

@dataclass
class DPIfixerOption(AbstractProcessorOption):
    min_dpi: int = 100
    max_dpi: int = 300

class DPIfixer(AbstractImageProcessor):
    name: str = "DPIfixer"
    SUMMARY = "Clamp DPI to [min, max] by resampling."
    REPLAY_TRAIT = ReplayTrait.COORDINATE  # uniform scale resample → fuses with other warps
    OPTION_CLASS = DPIfixerOption
    _ESSENTIAL_PARAMS = ("min_dpi", "max_dpi")
    OPTIONS = {
        "min_dpi": _i(100, 50, 600,
                      "If the buffer's DPI is below this, upsample to it."),
        "max_dpi": _i(300, 50, 1200,
                      "If the buffer's DPI is above this, downsample to it."),
    }

    @classmethod
    def replay_transform(cls, params, in_wh):
        """Uniform DPI scale → an affine (diagonal) 3×3, src→dst."""
        import numpy as np

        from lib.processors.replay_transform import AffineTransform
        w, h = in_wh
        in_w, in_h = params["in_wh"]
        out_w, out_h = params["out_wh"]
        sx, sy = out_w / in_w, out_h / in_h
        H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        return AffineTransform(H, (int(round(w * sx)), int(round(h * sy))))

    def __init__(self, options: DPIfixerOption):
        super().__init__(options)
        self.min_dpi = options.min_dpi
        self.max_dpi = options.max_dpi

    def process(self, img_buf: ImageBuffer) -> ImageBuffer:
        if not img_buf or img_buf.buffer is None:
            return img_buf
            
        current_dpi = float(img_buf.dpi)
        target_dpi = current_dpi
        
        # Determine if we need to change DPI
        if current_dpi < self.min_dpi:
            target_dpi = float(self.min_dpi)
        elif current_dpi > self.max_dpi:
            target_dpi = float(self.max_dpi)
            
        # If no change needed, return early
        if abs(target_dpi - current_dpi) < 1.0:
            return img_buf
            
        scale_factor = target_dpi / current_dpi
        
        # Calculate new dimensions
        h, w = img_buf.buffer.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        if new_w <= 0 or new_h <= 0:
            return img_buf
            
        # Choose interpolation method
        if scale_factor > 1.0:
            inter = cv2.INTER_CUBIC  # Upsampling
        else:
            inter = cv2.INTER_AREA   # Downsampling
            
        # Resize
        resized = cv2.resize(img_buf.buffer, (new_w, new_h), interpolation=inter)
        
        # Update buffer and metadata
        img_buf.buffer = resized
        img_buf.dpi = target_dpi

        # Replay: a uniform scale (COORDINATE trait). Records in/out sizes so
        # the replay engine folds the resample into the fused warp instead of
        # re-interpolating.
        if img_buf.meta is None:
            img_buf.meta = {}
        img_buf.meta["replay_kind"] = "resample"
        img_buf.meta["replay_params"] = {"in_wh": [w, h], "out_wh": [new_w, new_h]}


        # Update ROI if present
        if img_buf.meta and "roi" in img_buf.meta:
            roi = img_buf.meta["roi"]
            if isinstance(roi, list):
                new_roi = []
                for pt in roi:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        new_roi.append([pt[0] * scale_factor, pt[1] * scale_factor])
                    else:
                        new_roi.append(pt) # Fallback / Passthrough
                img_buf.meta["roi"] = new_roi
        
        return img_buf
