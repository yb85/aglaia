#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Emit PageDetector split references for aglaia-ios parity tests.

Runs the desktop PageDetector with the apple_vision backend (the only
backend the iOS port has) on each set's 02_capture_deskew fixtures, with
the frozen book_curved_x2 options, and records each child's crop rect,
page_side and ROI. This is the like-for-like reference: the .agl node
goldens may have been produced by another backend (auto → dbnet) or by an
older detector revision (e.g. test_athanase carries a degenerate tiny-crop
detection).

    uv run python scripts/export_split_goldens.py ../aglaia-ios/AglaiaCore/Tests/AglaiaCoreTests/Fixtures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aglaia.ImageBuffer import ImageBuffer, ImageType  # noqa: E402
from aglaia.processors.PageDetector import PageDetector, PageOption  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1])
    opts = PageOption(
        margin_mm=2, max_pages=2, rescale_threshold=0.01, processing_dpi=100.0,
        backend="apple_vision", min_contrast=0.5,
    )
    for set_dir in sorted(p.parent for p in root.glob("*/manifest.json")):
        manifest = json.loads((set_dir / "manifest.json").read_text())
        out: dict = {}
        for scan in manifest["scans"]:
            node = next(n for n in scan["nodes"] if n["step_name"] == "02_capture_deskew")
            img = cv2.imread(str(set_dir / scan["dir"] / node["image"]))
            buf = ImageBuffer(img, ImageType.COLOR, dpi=float(node["dpi"] or 110),
                              filestem="golden", path=None, parent=None)
            det = PageDetector(opts)
            det.process(buf)
            out[scan["dir"]] = [
                {
                    "crop_xywh": child.meta.get("parent_crop_xywh"),
                    "page_side": child.meta.get("page_side"),
                    "roi": child.meta.get("roi"),
                    "branch": child.branch_label,
                }
                for child in buf.children
            ]
        (set_dir / "split_goldens.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        print(set_dir.name, {k: len(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
