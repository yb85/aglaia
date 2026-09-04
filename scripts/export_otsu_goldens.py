#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Compute cv2 Otsu thresholds + gray stats for exported golden fixture images.

Companion to export_goldens.py: walks a fixtures tree, and for every stage
image computes the desktop-reference grayscale conversion (cv2 BGR→GRAY,
BT.601) and Otsu threshold (cv2.THRESH_OTSU). Output lands next to each
set's manifest as otsu_goldens.json, consumed by aglaia-ios parity tests.

    uv run python scripts/export_otsu_goldens.py ../aglaia-ios/AglaiaCore/Tests/AglaiaCoreTests/Fixtures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2


def main() -> int:
    root = Path(sys.argv[1])
    for manifest in sorted(root.glob("*/manifest.json")):
        set_dir = manifest.parent
        out: dict[str, dict] = {}
        for img_path in sorted(set_dir.glob("scan*/*.jpg")) + sorted(set_dir.glob("scan*/*.png")):
            if img_path.name.endswith(".otsu.png"):
                continue  # our own output from a previous run
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            thresh, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if img_path.stem == "00_raw":
                # full binarized reference for the SSIM criterion (raws only)
                cv2.imwrite(str(img_path.with_suffix(".otsu.png")), bw)
            key = f"{img_path.parent.name}/{img_path.name}"
            out[key] = {
                "otsu": thresh,
                "gray_mean": round(float(gray.mean()), 4),
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
            }
        (set_dir / "otsu_goldens.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"{set_dir.name}: {len(out)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
