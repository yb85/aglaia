# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1


class LayoutBackend(ABC):
    """Cross-platform interface for text detection / recognition."""

    name: str = "base"
    # Whether this backend actually runs on GPU / accelerated hardware.
    # Subclasses override (e.g. apple_vision → True, heuristic → False);
    # cv2.dnn-based backends probe at init.
    uses_gpu: bool = False

    @abstractmethod
    def detect(self, img_rgb: np.ndarray) -> List[BBox]:
        """Return list of text bounding boxes in pixel coordinates."""

    def recognize(self, img_rgb: np.ndarray) -> List[Tuple[str, BBox]]:
        """Return list of (text, bbox). Default: backend lacks OCR — return empty."""
        return []
