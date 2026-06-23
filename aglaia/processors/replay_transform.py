# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Replay contract types shared by the processors and the replay engine.

A COORDINATE processor describes its transform as a *composable geometric
primitive* — without touching pixels — by returning one of these from
``replay_transform(params, in_wh)``:

  * ``AffineTransform``     — a forward (src→dst) 3×3 homography + output canvas.
    Contiguous affines compose into ONE 3×3 the engine applies in a single
    interpolation (uniform DPI scale, skew rotate, keystone perspective).
  * ``SampleMapTransform``  — a nonlinear backward sampling map (dewarp). The
    engine folds any pending upstream affine into the map's source coords so
    a warp-then-dewarp run still costs one ``cv2.remap``.

PIXEL_VALUE / ROI processors instead implement ``apply_replay(buf, mask,
params, ctx)`` and act on pixels directly; ``ReplayContext`` carries the bits
they need (DPI for binarisation, debug dump dir/tag).

Keeping these primitives here — not in ``Replay.py`` — is what lets a plugin
processor join replay by implementing the method on its own class, with no
edit to the central engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class AffineTransform:
    """Forward (src→dst) 3×3 homography computed for the input size the engine
    passed to ``replay_transform``, plus the output canvas ``(w, h)``."""
    H: np.ndarray                       # 3×3 float64, forward src→dst
    out_wh: tuple[int, int]


@dataclass(frozen=True)
class SampleMapTransform:
    """Nonlinear backward map. ``make_map(in_hw) -> (im_x, im_y, pad_px)``:
    for each output pixel, the float coordinate to sample in the input padded
    by ``pad_px`` white px on every side. Analytic in the fitted params (no
    pixel reads), so an upstream affine can be folded into its coords."""
    make_map: Callable[[tuple[int, int]], tuple[np.ndarray, np.ndarray, int]]


# Either kind a COORDINATE processor may return.
ReplayTransform = "AffineTransform | SampleMapTransform"


@dataclass(frozen=True)
class ReplayContext:
    """Side inputs an ``apply_replay`` step may need beyond (buf, mask,
    params): output DPI (binariser window sizing) and an optional debug dump
    location + per-step tag."""
    dpi: float = 300.0
    debug_dir: Optional[str] = None
    debug_tag: Optional[str] = None


def interp_for(buf: np.ndarray) -> int:
    """NN for binary buffers (no smudging across the 0/255 jump); cubic for
    grayscale / colour where smooth resampling beats stair-stepping."""
    if buf.ndim == 2:
        uniq = np.unique(buf[::8, ::8])
        if uniq.size <= 2:
            return cv2.INTER_NEAREST
    return cv2.INTER_CUBIC


def remap_by_coords(buf: np.ndarray, mask: np.ndarray,
                    sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample buf/mask at per-output-pixel source coords (sx, sy)."""
    border_val = (255, 255, 255) if buf.ndim == 3 else 255
    out = cv2.remap(buf, sx, sy, interp_for(buf), None,
                    cv2.BORDER_CONSTANT, border_val)
    out_mask = cv2.remap(mask, sx, sy, cv2.INTER_NEAREST, None,
                         cv2.BORDER_CONSTANT, 0)
    return out, out_mask
