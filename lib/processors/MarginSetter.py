# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
MarginSetter — pads the image with a white border of a configurable
width, CSS-style.

Accepts the margin as either `margin_mm` (millimetres) or `margin_px`
(pixels). The value is parsed CSS-shorthand-style:

    "10"                   →  10 on all four sides
    "10 20"                →  10 vertical (top + bottom) / 20 horizontal
    "10 20 30 40"          →  10 left, 20 right, 30 top, 40 bottom
                              (LRTB — different from CSS top-right-bottom-left)

Pure number values work too (`margin_mm: 5` → 5 on all sides).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from lib.ImageBuffer import ImageBuffer, ImageType
from lib.Status import Status
from lib.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait


@dataclass
class MarginSetterOption(AbstractProcessorOption):
    margin_mm: Optional[Union[float, int, str]] = None
    margin_px: Optional[Union[float, int, str]] = None


def _parse_margin(value: Union[float, int, str]) -> tuple[float, float, float, float]:
    """Parse a CSS-shorthand margin spec to (left, right, top, bottom)."""
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v, v, v
    parts = str(value).split()
    if len(parts) == 1:
        v = float(parts[0])
        return v, v, v, v
    if len(parts) == 2:
        vert = float(parts[0])
        horiz = float(parts[1])
        return horiz, horiz, vert, vert
    if len(parts) == 4:
        l, r, t, b = (float(p) for p in parts)
        return l, r, t, b
    raise ValueError(f"unrecognized margin spec: {value!r}")


def _resolve_pixels(opt: MarginSetterOption, dpi: float) -> tuple[int, int, int, int]:
    if opt.margin_px is not None:
        l, r, t, b = _parse_margin(opt.margin_px)
    elif opt.margin_mm is not None:
        l, r, t, b = _parse_margin(opt.margin_mm)
        scale = dpi / 25.4
        l, r, t, b = l * scale, r * scale, t * scale, b * scale
    else:
        return 0, 0, 0, 0
    return int(round(l)), int(round(r)), int(round(t)), int(round(b))


from lib.processors.option_specs import _s


class MarginSetter(AbstractImageProcessor):
    name: str = "MarginSetter"
    SUMMARY = "Crop to content bbox + pad with a CSS-style margin (mm or px)."
    REPLAY_TRAIT = ReplayTrait.ROI  # crop + pad → ROI change
    OPTION_CLASS = MarginSetterOption
    _ESSENTIAL_PARAMS = ("margin_mm", "margin_px")
    OPTIONS = {
        "margin_mm": _s("5",
                        "CSS-style margin in mm. \"10\" all sides; "
                        "\"10 20\" V H; \"10 20 30 40\" L R T B."),
        "margin_px": _s("",
                        "Same as margin_mm but in pixels (overrides margin_mm when set).",
                        advanced=True),
    }

    @classmethod
    def apply_replay(cls, buf, mask, params, ctx):
        """Crop to content, then pad the stamped left/right/top/bottom margin,
        enforcing the dewarp-width floor — matches the forward pass."""
        in_w = buf.shape[1]
        cropped, bbox = _crop_to_content(buf)
        x0, y0, w0, h0 = bbox
        cropped_mask = mask[y0:y0 + h0, x0:x0 + w0]
        l, r, t, b = params["ltrb_px"]
        if (l, r, t, b) == (0, 0, 0, 0):
            out, out_mask = cropped, cropped_mask
        else:
            border_val = 255 if cropped.ndim == 2 else (255, 255, 255)
            out = cv2.copyMakeBorder(cropped, t, b, l, r,
                                     cv2.BORDER_CONSTANT, value=border_val)
            out_mask = cv2.copyMakeBorder(cropped_mask, t, b, l, r,
                                          cv2.BORDER_CONSTANT, value=0)
        # Width floor: dewarping a curved page can only widen it; cropping
        # whitespace + a tight pad must not shrink below the dewarp output.
        min_w = int(params.get("min_width_px", in_w))
        if out.shape[1] < min_w:
            out = _enforce_width_floor(out, min_w, fill=255)
            out_mask = _enforce_width_floor(out_mask, min_w, fill=0)
        return out, out_mask

    def __init__(self, options: MarginSetterOption):
        super().__init__(options)
        self.opt = options
        self.uses_gpu = False

    def process(self, buf: ImageBuffer) -> ImageBuffer:
        in_h, in_w = buf.buffer.shape[:2]
        left, right, top, bottom = _resolve_pixels(self.opt, buf.dpi)
        # 1. Crop to content bbox (drop whitespace borders).
        cropped, bbox = _crop_to_content(buf.buffer)
        # 2. Pad with desired margin (white).
        border_val = 255 if cropped.ndim == 2 else (255, 255, 255)
        if (left, right, top, bottom) == (0, 0, 0, 0):
            out = cropped
        else:
            out = cv2.copyMakeBorder(
                cropped, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=border_val,
            )
        out = _enforce_width_floor(out, in_w)
        buf.buffer = out
        buf.meta["status"] = int(Status.SUCCESS)
        buf.meta["replay_kind"] = "margin"
        buf.meta["replay_params"] = {
            "ltrb_px": [int(left), int(right), int(top), int(bottom)],
            "content_bbox_xywh": [int(b) for b in bbox],
            "min_width_px": int(in_w),
        }
        return buf


def _enforce_width_floor(arr: np.ndarray, min_w: int,
                         fill: int = 255) -> np.ndarray:
    """Pad `arr` horizontally so its width >= `min_w`.

    Physically, flattening a curved page can only *grow* the page width
    (arc length unrolls). Yet content-bbox cropping after PageDewarper —
    which strips the whitespace margin page-dewarp adds — can drop the
    final width below the source. Earlier this was patched with a
    uniform up-scale, but when dewarp left a lot of horizontal
    whitespace inside its canvas the scale ratio went 1.15-1.34, which
    stretched height by the same factor and produced visibly squished
    glyphs (scans 95, 108, ...). Replacing the up-scale with symmetric
    padding restores the width promise without distorting content.
    `fill` is 255 for image (white), 0 for mask (outside ROI).
    """
    h, w = arr.shape[:2]
    if w >= min_w or w <= 0:
        return arr
    pad_total = int(min_w - w)
    pad_l = pad_total // 2
    pad_r = pad_total - pad_l
    border_val = fill if arr.ndim == 2 else (fill, fill, fill)
    return cv2.copyMakeBorder(arr, 0, 0, pad_l, pad_r,
                              cv2.BORDER_CONSTANT, value=border_val)


def _crop_to_content(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (cropped, (x, y, w, h)) where the crop is the content bbox.

    Content = pixels darker than 250 on grayscale (handles JPEG noise).
    Empty image → no crop, bbox = full extent.
    """
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    h_img, w_img = gray.shape[:2]
    mask = gray < 250
    if not mask.any():
        return arr, (0, 0, w_img, h_img)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = arr[y0:y1, x0:x1]
    return cropped, (x0, y0, x1 - x0, y1 - y0)
