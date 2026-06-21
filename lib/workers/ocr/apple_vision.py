# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Apple Vision OCR engine (macOS-only, via pyobjc).

Wraps `VNRecognizeTextRequest`. Language codes follow BCP-47 — Vision
accepts e.g. `fr-FR`, `en-US`. Any unsupported code is silently dropped
by Vision; we don't pre-filter.
"""

from __future__ import annotations

import io
import sys
from typing import Optional

import numpy as np

from .engine import (
    OcrEngine, OcrResult, register,
    resolve_ocr_dpi, downsample_to_dpi,
)

_AVAILABLE: Optional[bool] = None
_SUPPORTED_LANGS: Optional[list[str]] = None


def supported_languages() -> list[str]:
    """Apple Vision's ACTUAL recognition languages on this macOS (accurate
    level), queried once + cached. Empty off-macOS / on error — callers
    fall back to a static catalogue. This is the real list (no Greek/Latin,
    which is why apple_docs needs a complement)."""
    global _SUPPORTED_LANGS
    if _SUPPORTED_LANGS is not None:
        return _SUPPORTED_LANGS
    out: list[str] = []
    if sys.platform == "darwin":
        try:
            import Vision
            import objc
            with objc.autorelease_pool():
                req = Vision.VNRecognizeTextRequest.alloc().init()
                req.setRecognitionLevel_(
                    Vision.VNRequestTextRecognitionLevelAccurate)
                langs, _err = (
                    req.supportedRecognitionLanguagesAndReturnError_(None))
                if langs:
                    out = [str(x) for x in langs]
        except Exception:
            out = []
    _SUPPORTED_LANGS = out
    return out


def _check() -> bool:
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    if sys.platform != "darwin":
        _AVAILABLE = False
        return False
    try:
        import Vision  # noqa: F401
        import objc    # noqa: F401
        from Foundation import NSData  # noqa: F401
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


@register
class AppleVisionEngine(OcrEngine):
    name = "apple_vision"
    display = "Apple Vision"
    description = ("Fast native macOS OCR, no download. "
                   "Best for printed Latin scripts.")

    def __init__(self) -> None:
        self.available = _check()

    def recognize(self, image_rgb: np.ndarray, languages: list[str],
                   *, src_dpi: float | None = None) -> OcrResult:
        if not self.available:
            raise RuntimeError("Apple Vision unavailable (non-macOS or pyobjc missing)")

        import Vision
        from Foundation import NSData
        from PIL import Image
        import objc

        if image_rgb.ndim == 2:
            arr = np.stack([image_rgb] * 3, axis=-1)
        elif image_rgb.shape[2] == 4:
            arr = image_rgb[:, :, :3]
        else:
            arr = image_rgb
        # Honour the unified OCR-DPI knob — Vision is fast enough at any
        # DPI, but downsampling cuts JPEG encode + memory and matches
        # what Surya / Paddle do, so the picker has a uniform effect.
        arr = downsample_to_dpi(arr, src_dpi or 0, resolve_ocr_dpi())
        h, w = arr.shape[:2]
        pil = Image.fromarray(arr.astype(np.uint8))

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        img_bytes = buf.getvalue()

        lines: list[dict] = []
        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(True)
            if languages:
                req.setRecognitionLanguages_(list(languages))
            ns = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns, None)
            ok, err = handler.performRequests_error_([req], None)
            if not ok:
                raise RuntimeError(f"VNImageRequestHandler failed: {err}")
            results = req.results() or []
            for r in results:
                cands = r.topCandidates_(1)
                if not cands:
                    continue
                top = cands[0]
                text = top.string()
                bbox = r.boundingBox()
                # Vision: origin bottom-left, normalised [0,1].
                x0 = int(bbox.origin.x * w)
                y0 = int((1.0 - bbox.origin.y - bbox.size.height) * h)
                x1 = int(x0 + bbox.size.width * w)
                y1 = int(y0 + bbox.size.height * h)
                conf = float(top.confidence())
                lines.append({
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "confidence": conf,
                })

        return {
            "engine": self.name,
            "languages": list(languages),
            "page_w": w,
            "page_h": h,
            "lines": lines,
            "meta": {
                "recognition_level": "accurate",
                "ocr_dpi": int(resolve_ocr_dpi() or 0),
            },
        }
