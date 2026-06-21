# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Apple Vision text detection backend (macOS only).

Lifts the logic from the deprecated lib/processors/AppleOCREngine.py into the
LayoutBackend interface.
"""
import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from lib.processors.layout_backends.base import LayoutBackend, BBox

try:
    import objc
    import Vision
    from Foundation import NSData
    HAS_VISION = True
except ImportError:
    HAS_VISION = False


class AppleVisionBackend(LayoutBackend):
    name = "apple_vision"
    # Vision schedules on the Neural Engine / integrated GPU on Apple Silicon
    # and the dGPU on Intel Macs. Always counts as accelerated for our purposes.
    uses_gpu = True

    def __init__(self):
        if not HAS_VISION:
            raise ImportError("Apple Vision unavailable. Install pyobjc-framework-vision (macOS only).")

    def _request(self, img_rgb: np.ndarray) -> List[Tuple[str, BBox]]:
        pil = Image.fromarray(img_rgb)
        results: List[Tuple[str, BBox]] = []
        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(False)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=95)
            data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
            success, _ = handler.performRequests_error_([req], None)
            if not success:
                return results
            w, h = pil.size
            for r in req.results():
                text = r.topCandidates_(1)[0].string()
                bb = r.boundingBox()
                x0 = int(bb.origin.x * w)
                y0 = int((1.0 - bb.origin.y - bb.size.height) * h)
                x1 = int(x0 + bb.size.width * w)
                y1 = int(y0 + bb.size.height * h)
                results.append((text, (x0, y0, x1, y1)))
        return results

    def detect(self, img_rgb: np.ndarray) -> List[BBox]:
        return [bb for _, bb in self._request(img_rgb)]

    def recognize(self, img_rgb: np.ndarray) -> List[Tuple[str, BBox]]:
        return self._request(img_rgb)
