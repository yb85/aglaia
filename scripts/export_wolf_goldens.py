#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Emit wolf_masked reference outputs for aglaia-ios parity tests.

Runs the desktop's actual `wolf_masked` (Binarizer.py) on each set's
scan1 dpi-normalized page A with a synthetic partial mask (left band
missing + polygon-free right), plus the full-coverage case. Window/k from
the frozen pipeline (window_mm_wolf=5 → px at the image dpi, k=0.25).

    uv run python scripts/export_wolf_goldens.py ../aglaia-ios/AglaiaCore/Tests/AglaiaCoreTests/Fixtures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aglaia.processors.Binarizer import morpho_close, wolf_masked  # noqa: E402

K = 0.25
WINDOW_MM = 5.0
ROI_SHRINK = 10
MORPHO_CLOSE = 2


def forward_stage(gray: np.ndarray, roi, window: int) -> np.ndarray:
    """The wolf++ forward stage as the iOS port defines it: wolf_masked with
    full coverage, ROI white-wipe (fillPoly + roi_shrink erosion), morpho
    close — the reference for aglaia-ios full-stage parity."""
    bw = wolf_masked(gray, np.full_like(gray, 255), window, k=K)
    if roi:
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        if cv2.countNonZero(mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=ROI_SHRINK)
            bw = cv2.bitwise_or(bw, cv2.bitwise_not(mask))
    return morpho_close(bw, MORPHO_CLOSE)


def main() -> int:
    root = Path(sys.argv[1])
    for set_dir in sorted(p.parent for p in root.glob("*/manifest.json")):
        manifest = json.loads((set_dir / "manifest.json").read_text())
        for scan in manifest["scans"]:
            for node in scan["nodes"]:
                if node["step_name"] != "04_dpi_normalize_output":
                    continue
                branch = node["branch"]
                img = cv2.imread(str(set_dir / scan["dir"] / node["image"]))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                h, w = gray.shape
                # The pipeline ran post-normalize stages at dpi 300; the .agl
                # images-table dpi is stale for some sets (aglaia-ios#17).
                dpi = 300.0
                window = max(4, int(round(WINDOW_MM / 25.4 * dpi)))
                meta = json.loads((set_dir / scan["dir"] / node["meta"]).read_text())
                roi = meta.get("roi")

                # full forward-stage reference for every page
                fwd = forward_stage(gray, roi, window)
                cv2.imwrite(
                    str(set_dir / scan["dir"] / f"wolfpp_forward.{branch}.png"), fwd
                )

                if scan["idx"] == 1 and branch == "A":
                    # partial mask: left 25% missing, plus a corner notch
                    mask = np.full((h, w), 255, dtype=np.uint8)
                    mask[:, : w // 4] = 0
                    mask[: h // 6, w - w // 5 :] = 0
                    bw = wolf_masked(gray, mask, window, k=K)
                    cv2.imwrite(str(set_dir / "scan1" / "wolf_masked_partial.png"), bw)
                    cv2.imwrite(str(set_dir / "scan1" / "wolf_masked_partial.mask.png"), mask)
                    full = np.full((h, w), 255, dtype=np.uint8)
                    bw_full = wolf_masked(gray, full, window, k=K)
                    cv2.imwrite(str(set_dir / "scan1" / "wolf_masked_full.png"), bw_full)
                print(f"{set_dir.name} {scan['dir']}/{branch}: window={window}px dpi={dpi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
