# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The text-scale estimate two processors share.

`TrapezoidalCorrection` and `PageDewarper` both size their line-joining
morphology from the median height of character-like connected components.
Each carried its own copy of the estimator — same DPI-scaled bounds, same
≥30-component floor, same median — and two copies of one estimate drift apart.

The rule (Yann's): **meta is a cache, never the only source.** A step that
needs the statistic reads `meta["char_h_frac"]` if an earlier step left it,
else computes it here. The pipeline works with or without the upstream step;
the two agree by construction because there is one function; and the only
thing meta buys is skipping a recomputation.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

#: Character-like component bounds, as fractions of the analysis DPI. A glyph
#: on a 300 dpi page is 12–135 px tall; the same fractions hold at 100 dpi.
CC_H_MIN_DPI, CC_H_MAX_DPI = 0.04, 0.45
CC_W_MIN_DPI, CC_W_MAX_DPI = 0.02, 0.60
#: Fewer components than this and the median is noise, not a text scale.
MIN_COMPONENTS = 30

META_KEY = "char_h_frac"


def cc_bounds(dpi: float) -> tuple[int, int, int, int]:
    """(h_min, h_max, w_min, w_max) in pixels for character-like components."""
    h_min = max(3, int(round(dpi * CC_H_MIN_DPI)))
    h_max = max(h_min + 1, int(round(dpi * CC_H_MAX_DPI)))
    w_min = max(2, int(round(dpi * CC_W_MIN_DPI)))
    w_max = max(w_min + 1, int(round(dpi * CC_W_MAX_DPI)))
    return h_min, h_max, w_min, w_max


def median_char_height(stats: np.ndarray, dpi: float) -> float:
    """Median height of character-like components, or 0.0 if too few.

    `stats` is `cv2.connectedComponentsWithStats(...)[2]`, row 0 being the
    background. Takes the stats rather than the mask so a caller that already
    ran the labelling for another reason — both do, for the large-blob wipe —
    does not run it twice."""
    h_min, h_max, w_min, w_max = cc_bounds(dpi)
    char_h = [int(s[3]) for s in stats[1:]
              if h_min <= s[3] <= h_max and w_min <= s[2] <= w_max]
    if len(char_h) < MIN_COMPONENTS:
        return 0.0
    return float(np.median(char_h))


def char_h_frac(h_med: float, frame_h: int) -> float:
    """The dimensionless form that goes in meta: median glyph height as a
    fraction of the analysis frame height, so it survives a resample."""
    return (float(h_med) / float(frame_h)) if h_med > 0 and frame_h > 0 else 0.0


def cached_char_height(meta: Optional[dict], frame_h: int) -> Optional[float]:
    """The median char height in THIS frame's pixels, from an upstream
    estimate in meta — or None, meaning compute it."""
    try:
        frac = float((meta or {}).get(META_KEY) or 0.0)
    except (TypeError, ValueError):
        return None
    if frac <= 0 or frame_h <= 0:
        return None
    return frac * float(frame_h)
