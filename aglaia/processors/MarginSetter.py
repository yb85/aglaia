# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
MarginSetter — pads the image with a white border of a configurable
width, CSS-style.

The image is first cropped to its ink content, then padded — so the margin
is measured from the text, not from whatever canvas the previous step left.
`width_floor` (off) is the one thing that can override that: see
`_enforce_width_floor`.

Accepts the margin as either `margin_mm` (millimetres) or `margin_px`
(pixels). The value is parsed CSS-shorthand-style:

    "10"                   →  10 on all four sides
    "10 20"                →  10 vertical (top + bottom) / 20 horizontal
    "10 20 30 40"          →  10 left, 20 right, 30 top, 40 bottom
                              (LRTB — different from CSS top-right-bottom-left)

Pure number values work too (`margin_mm: 5` → 5 on all sides).

`fixed_size` (off) places that block on a page of `page_size_mm` (default A5,
"148 210") instead, aligned by `h_align` / `v_align`. The margin is then the
minimum margin. A block larger than the page grows the page on that axis, or
with `clip` is cut to the page following the alignment (#197). A page with no
ink always comes out at exactly the page size, whatever `clip` says.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.Status import Status
from aglaia.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait


@dataclass
class MarginSetterOption(AbstractProcessorOption):
    margin_mm: Optional[Union[float, int, str]] = "2"
    margin_px: Optional[Union[float, int, str]] = None
    # Pad back out to the step's INPUT width when the crop came out narrower.
    # OFF by default: it overrides the horizontal margin with "whatever
    # whitespace the dewarp canvas happened to carry", so a page asking for
    # 5 mm measured 10.2-16.8 mm left/right while top/bottom were exactly 5.0
    # (#112). See `_enforce_width_floor` for what it is for.
    width_floor: bool = False
    # Fixed page size (#197): place the content+margin block on a W×H mm page.
    fixed_size: bool = False
    page_size_mm: str = "148 210"   # A5
    h_align: str = "center"         # left | center | right
    v_align: str = "center"         # top | center | bottom
    clip: bool = False              # overflow: cut to the page, not grow it


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


def _parse_page_size(value) -> tuple[float, float]:
    """`"148 210"` → (148.0, 210.0), width then height, in mm."""
    parts = str(value).replace("x", " ").replace("×", " ").split()
    if len(parts) != 2:
        raise ValueError(f"page size must be 'W H' in mm: {value!r}")
    w, h = (float(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"page size must be positive: {value!r}")
    return w, h


def _offset(align: str, page: int, block: int) -> int:
    """Where the block starts on the page axis. Negative = cut at the start."""
    if align in ("left", "top"):
        return 0
    if align in ("right", "bottom"):
        return page - block
    return (page - block) // 2


def _place(block: np.ndarray, page_wh: tuple[int, int], h_align: str,
           v_align: str, clip: bool, fill: int) -> np.ndarray:
    """Put `block` on a `fill` page of `page_wh`, aligned. Without `clip` an
    axis the block overflows takes the block's size instead."""
    bh, bw = block.shape[:2]
    pw, ph = page_wh
    if not clip:
        pw, ph = max(pw, bw), max(ph, bh)
    page = np.full((ph, pw) + block.shape[2:], fill, dtype=block.dtype)
    x, y = _offset(h_align, pw, bw), _offset(v_align, ph, bh)
    # Intersection of the block with the page, in both frames.
    px0, py0 = max(0, x), max(0, y)
    px1, py1 = min(pw, x + bw), min(ph, y + bh)
    if px1 > px0 and py1 > py0:
        page[py0:py1, px0:px1] = block[py0 - y:py1 - y, px0 - x:px1 - x]
    return page


from aglaia.processors.option_specs import _b, _e, _s

_FIXED = {"fixed_size": [True]}


class MarginSetter(AbstractImageProcessor):
    name: str = "MarginSetter"
    SUMMARY = "Crop to content bbox + pad with a CSS-style margin (mm or px)."
    REPLAY_TRAIT = ReplayTrait.ROI  # crop + pad → ROI change
    OPTION_CLASS = MarginSetterOption
    PROVIDES_META = {
        "status": "int Status enum for the crop+pad outcome",
    }
    _ESSENTIAL_PARAMS = ("margin_mm", "margin_px")
    OPTIONS = {
        "margin_mm": _s("2",
                        "CSS-style margin in mm. \"10\" all sides; "
                        "\"10 20\" V H; \"10 20 30 40\" L R T B."),
        "margin_px": _s("",
                        "Same as margin_mm but in pixels (overrides margin_mm when set).",
                        advanced=True),
        "width_floor": _b(False,
                          "Pad back out to the step's input width when the "
                          "cropped page is narrower. Overrides the horizontal "
                          "margin with the dewarp canvas's leftover "
                          "whitespace — leave off unless you need every page "
                          "at least as wide as the dewarp made it.",
                          advanced=True),
        "fixed_size": _b(False,
                         "Place the content and its margin on a page of a "
                         "fixed size, so every page has the same size. The "
                         "margin becomes the minimum margin."),
        "page_size_mm": _s("148 210",
                           "Page size \"W H\" in mm. A5 = \"148 210\", "
                           "A4 = \"210 297\".", visible_when=_FIXED),
        "h_align": _e("center", ["left", "center", "right"],
                      "Horizontal position of the content on the page.",
                      visible_when=_FIXED),
        "v_align": _e("center", ["top", "center", "bottom"],
                      "Vertical position of the content on the page.",
                      visible_when=_FIXED),
        "clip": _b(False,
                   "When the content and its margin are larger than the "
                   "page: on = cut them to the page, following the "
                   "alignment; off = make the page larger on that axis.",
                   visible_when=_FIXED),
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
        # Width floor, only when the forward pass stamped one. It defaults
        # off (#112) — but a chain stamped before this option existed carries
        # the old unconditional value, and replaying it must still reproduce
        # what that run produced.
        if page := params.get("page"):
            wh = tuple(page["wh_px"])
            args = (page["h_align"], page["v_align"], bool(page["clip"]))
            return (_place(out, wh, *args, fill=255),
                    _place(out_mask, wh, *args, fill=0))
        min_w = int(params.get("min_width_px", in_w) or 0)
        if min_w and out.shape[1] < min_w:
            out = _enforce_width_floor(out, min_w, fill=255)
            out_mask = _enforce_width_floor(out_mask, min_w, fill=0)
        return out, out_mask

    def __init__(self, options: MarginSetterOption):
        super().__init__(options)
        self.opt = options
        self.width_floor = bool(getattr(options, "width_floor", False))
        self.fixed_size = bool(getattr(options, "fixed_size", False))
        if self.fixed_size:
            self.page_mm = _parse_page_size(options.page_size_mm)
            self.h_align = str(options.h_align)
            self.v_align = str(options.v_align)
            self.clip = bool(options.clip)
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
        # 3. Optional width floor. Off by default: it pads the page back out
        # to the dewarp canvas, which silently replaces the horizontal margin
        # the user asked for (#112).
        page = None
        if self.fixed_size:
            scale = buf.dpi / 25.4
            # A page with no ink (a blank page, #196) has nothing to protect
            # from the cut: it always comes out at exactly the page size, so
            # a large blank never grows the page. Stamped as the clip the
            # replay applies.
            empty = not _has_ink(cropped)
            clip = self.clip or empty
            page = {"wh_px": [int(round(self.page_mm[0] * scale)),
                              int(round(self.page_mm[1] * scale))],
                    "h_align": self.h_align, "v_align": self.v_align,
                    "clip": clip}
            out = _place(out, tuple(page["wh_px"]), self.h_align,
                         self.v_align, clip, fill=255)
        # The floor is about the dewarp canvas width; a fixed page replaces it.
        floor_px = int(in_w) if self.width_floor and page is None else 0
        if floor_px:
            out = _enforce_width_floor(out, floor_px)
        buf.buffer = out
        buf.meta["status"] = int(Status.SUCCESS)
        buf.meta["replay_kind"] = "margin"
        buf.meta["replay_params"] = {
            "ltrb_px": [int(left), int(right), int(top), int(bottom)],
            "content_bbox_xywh": [int(b) for b in bbox],
            # Stamped for the replay pass, which must reproduce the forward
            # result exactly — 0 means "no floor", not "floor at 0".
            "min_width_px": floor_px,
            "page": page,
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

    OFF by default since #112. The promise it keeps is about *page* width,
    while this step's contract is content-crop plus a stated margin — so
    horizontally it just re-adds the whitespace the crop removed, and the
    margin the user set stops being the margin they get. Measured over 40
    pages asking for 5 mm: top and bottom exactly 5.0, left and right
    10.2-16.8 and varying page to page. Turn `width_floor` on only when every
    page must stay at least as wide as the dewarp made it.
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


def _has_ink(arr: np.ndarray) -> bool:
    """Same ink test as `_crop_to_content`: any pixel darker than 250."""
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    return bool((gray < 250).any())


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
