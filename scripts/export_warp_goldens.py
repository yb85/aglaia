#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Emit cv2-rotated reference images for aglaia-ios affine-warp parity tests.

Applies exactly the SkewFinder rotation recipe (getRotationMatrix2D about the
integer center, INTER_CUBIC on gray, BORDER_CONSTANT white) to each set's
scan1 raw, at a fixed test angle.

    uv run python scripts/export_warp_goldens.py ../aglaia-ios/AglaiaCore/Tests/AglaiaCoreTests/Fixtures
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ANGLE = -2.5  # degrees, arbitrary non-trivial test angle


def main() -> int:
    root = Path(sys.argv[1])
    for raw in sorted(root.glob("*/scan1/00_raw.jpg")):
        img = cv2.imread(str(raw))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w // 2, h // 2), ANGLE, 1.0)
        rot = cv2.warpAffine(
            gray, m, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
        out = raw.with_name(f"00_raw.rot{ANGLE}.png")
        cv2.imwrite(str(out), rot)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
