#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Emit dewarp span-analysis references for aglaia-ios parity tests.

Replicates the analysis half of PageDewarper._build_dewarp_problem (pad →
downscale → pagemask → _text_mask_dpi → get_contours/assemble_spans with
char-scaled cfg → span-width filter → _sample_spans_xband, baseline_source
"bottom") on each 07_pages_trap fixture page, using the desktop's own
functions, and dumps the sampled span points (normalized page_dewarp
coords) to JSON.

    uv run python scripts/export_dewarp_span_goldens.py ../aglaia-ios/AglaiaCore/Tests/AglaiaCoreTests/Fixtures
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aglaia.processors.PageDewarper import (  # noqa: E402
    _sample_spans_xband,
    _text_mask_dpi,
)
from page_dewarp.contours import get_contours  # noqa: E402
from page_dewarp.options.core import cfg as pd_cfg  # noqa: E402
from page_dewarp.spans import assemble_spans  # noqa: E402

# frozen pages_dewarp options
PROCESSING_DPI = 150.0
PAGE_MARGIN_MM = 5.0
DEWARP_MARGIN_MM = 5.0
LINE_JOIN_MM = 4.0
KERNEL_CHAR_MULT = 1.5
THICKNESS_CHAR_MULT = 3.0
EDGE_MAX_LENGTH_CHAR_MULT = 3.0
LARGE_BLOB_LIMIT = 10.0
MIN_SPAN_WIDTH_RATIO = 0.2


def analyze(gray: np.ndarray, dpi: float, is_bw: bool) -> dict:
    pad_px = int(math.ceil(dpi * (DEWARP_MARGIN_MM / 25.4)))
    img = cv2.copyMakeBorder(gray, pad_px, pad_px, pad_px, pad_px,
                             cv2.BORDER_CONSTANT, value=255)
    interp = cv2.INTER_NEAREST if is_bw else cv2.INTER_AREA
    if PROCESSING_DPI < dpi:
        scale = PROCESSING_DPI / dpi
        small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                           interpolation=interp)
    else:
        small = img
    small_rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
    h, w = small.shape[:2]
    analysis_dpi = dpi * (w / img.shape[1])
    pagemask = np.zeros((h, w), dtype=np.uint8)
    mx = int(PAGE_MARGIN_MM * (analysis_dpi / 25.4) / 2)
    cv2.rectangle(pagemask, (mx, mx), (w - mx, h - mx), 255, -1)

    text_mask, h_med = _text_mask_dpi(small_rgb, pagemask, analysis_dpi,
                                      LINE_JOIN_MM, is_bw=is_bw,
                                      kernel_char_mult=KERNEL_CHAR_MULT,
                                      large_blob_limit=LARGE_BLOB_LIMIT)
    saved = {k: getattr(pd_cfg, k) for k in (
        "TEXT_MAX_THICKNESS", "TEXT_MIN_WIDTH", "TEXT_MIN_HEIGHT",
        "EDGE_MAX_LENGTH", "EDGE_MAX_OVERLAP", "EDGE_MAX_ANGLE", "SPAN_MIN_WIDTH")}
    try:
        if h_med > 0:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(THICKNESS_CHAR_MULT * h_med)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(0.5 * h_med)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(0.5 * h_med)))
            pd_cfg.EDGE_MAX_LENGTH = max(20, int(round(EDGE_MAX_LENGTH_CHAR_MULT * h_med)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, 0.1 * h_med)
            pd_cfg.SPAN_MIN_WIDTH = max(30, int(round(10.0 * h_med)))
        else:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(analysis_dpi * 0.25)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(analysis_dpi * 0.10)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(analysis_dpi * 0.01)))
            pd_cfg.EDGE_MAX_LENGTH = max(100, int(round(analysis_dpi * 0.5)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, analysis_dpi * 0.02)
            pd_cfg.SPAN_MIN_WIDTH = max(30, w // 20)
        pd_cfg.EDGE_MAX_ANGLE = 7.5
        cinfos = get_contours("golden", small_rgb, text_mask)
        spans = assemble_spans("golden", small_rgb, pagemask, cinfos)
        if MIN_SPAN_WIDTH_RATIO > 0 and spans:
            widths = []
            for s in spans:
                xs = []
                for c in s:
                    xs.extend(c.local_xrng)
                widths.append(max(xs) - min(xs) if xs else 0.0)
            w_max = max(widths)
            spans = [s for s, sw in zip(spans, widths)
                     if sw >= MIN_SPAN_WIDTH_RATIO * w_max]
        span_points = _sample_spans_xband(small.shape[:2], spans, "bottom")
    finally:
        for k, v in saved.items():
            setattr(pd_cfg, k, v)
    return {
        "analysis_wh": [w, h],
        "analysis_dpi": analysis_dpi,
        "pad_px": pad_px,
        "h_med": h_med,
        "n_contours": len(cinfos),
        "n_spans": len(span_points),
        "spans": [np.asarray(sp).reshape(-1, 2).tolist() for sp in span_points],
    }


def main() -> int:
    root = Path(sys.argv[1])
    for set_dir in sorted(p.parent for p in root.glob("*/manifest.json")):
        manifest = json.loads((set_dir / "manifest.json").read_text())
        out: dict = {}
        for scan in manifest["scans"]:
            for node in scan["nodes"]:
                if node["step_name"] != "07_pages_trap":
                    continue
                img = cv2.imread(str(set_dir / scan["dir"] / node["image"]),
                                 cv2.IMREAD_GRAYSCALE)
                is_bw = node["image_type"] == "BW"
                # true pipeline dpi (images-table value stale, aglaia-ios#17)
                res = analyze(img, 300.0, is_bw)
                out[f"{scan['dir']}/{node['branch']}"] = res
                print(f"{set_dir.name} {scan['dir']}/{node['branch']}: "
                      f"spans={res['n_spans']} h_med={res['h_med']:.1f}")
        (set_dir / "dewarp_span_goldens.json").write_text(
            json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
