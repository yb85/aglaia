# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
import numpy as np

def to_rgb(buf):
    """Convert a buffer (ImageBuffer or numpy array) to RGB."""
    if hasattr(buf, 'to_rgb'):
        return buf.to_rgb()
    # Handle raw numpy
    if len(buf.shape) == 2:
        return cv2.cvtColor(buf, cv2.COLOR_GRAY2RGB)
    elif len(buf.shape) == 3 and buf.shape[2] == 4:
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
    return buf

def to_gray(buf):
    """Convert a buffer (ImageBuffer or numpy array) to Gray."""
    if hasattr(buf, 'to_gray'):
        return buf.to_gray()
    # Handle raw numpy
    if len(buf.shape) == 3:
        return cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
    return buf

def to_bw(buf):
    """Convert a buffer (ImageBuffer or numpy array) to BW."""
    if hasattr(buf, 'to_bw'):
        return buf.to_bw()
    # Handle raw numpy
    gray = to_gray(buf)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw

def count_distinct_values(img) -> int:
    """Number of distinct pixel values in a 2-D image. For uint8 uses a
    cv2 histogram (~ms even at 12 MP) instead of np.unique's full sort
    (~100-300 ms)."""
    if img.dtype == np.uint8:
        hist = cv2.calcHist([np.ascontiguousarray(img)], [0], None,
                            [256], [0, 256])
        return int(np.count_nonzero(hist))
    return int(np.unique(img).size)

def is_binary(buf):
    """Check if a buffer (ImageBuffer or numpy array) is binary."""
    if hasattr(buf, 'check_binary'):
        return buf.check_binary()
    # Handle raw numpy
    if len(buf.shape) == 2:
        return count_distinct_values(buf) <= 2
    return False

def binarize_fixed(buf, threshold=127):
    """Apply a fixed threshold binarization."""
    gray = to_gray(buf)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return bw
