# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
import numpy as np
from typing import List

from lib.processors.layout_backends.base import LayoutBackend, BBox


class HeuristicBackend(LayoutBackend):
    """
    Cross-platform layout detector with no ML dependency.

    Pipeline:
      1. Otsu-binarize → ink mask (ink=255).
      2. Horizontal closing only (merge intra-word strokes into text lines,
         while keeping inter-line gaps intact).
      3. Column projection (sum_y of ink) → candidate column groups.
      4. For each candidate, look at the ORIGINAL binary slice and count
         distinct horizontal text-line runs in its row projection.
         A real text column has many alternating ink/gap rows; a hand or
         solid blob has 1-3 fat runs. Reject low-line-count candidates.

    Tunables:
      - `min_box_px`: minimum side length of a returned bbox.
      - `gap_px`: max allowed horizontal gap inside a single column group.
      - `min_text_rows`: minimum distinct text-line runs to accept a column.
    """

    name = "heuristic"
    uses_gpu = False  # pure numpy / OpenCV CPU ops

    def __init__(self, min_box_px: int = 60, gap_px: int = 30,
                 min_text_rows: int = 5):
        self.min_box_px = min_box_px
        self.gap_px = gap_px
        self.min_text_rows = min_text_rows

    def detect(self, img_rgb: np.ndarray) -> List[BBox]:
        if img_rgb.ndim == 3:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb
        h, w = gray.shape

        # ink=255 after inversion
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal closing only — merge words into text-line strips,
        # keep inter-line whitespace so row-projection still oscillates.
        kx = max(15, w // 40)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        closed_h = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_h)

        # Column projection
        col_sum = (closed_h > 0).sum(axis=0)
        col_thr = max(h * 0.02, 5)
        col_mask = col_sum > col_thr
        col_groups = _runs(col_mask, min_len=self.min_box_px, gap=self.gap_px)
        if not col_groups:
            return []

        # For each candidate column, verify text-pattern via row-line count
        kept: list[BBox] = []
        strip_w = lambda cx0, cx1: cx1 - cx0
        for cx0, cx1 in col_groups:
            strip = bw[:, cx0:cx1]
            sw = strip_w(cx0, cx1)
            # Horizontal closing inside the strip merges glyph fragments
            # within each row but does not bleed across text-line gaps.
            kernel_strip = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, kx), 1))
            strip_closed = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, kernel_strip)
            row_sum = (strip_closed > 0).sum(axis=1)
            # Per-row ink density: 1.0 = entire row is ink (solid bar);
            # text rows typically sit in 0.05–0.6.
            row_density = row_sum / max(sw, 1)
            row_thr = max(sw * 0.05, 3)
            row_mask = (row_sum > row_thr)
            # Exclude rows that look like solid bars / dense blobs (top/bottom
            # book edges, scanlines, hands). Text rows in real photos rarely
            # exceed ~70% ink density across the column width.
            text_row_mask = row_mask & (row_density < 0.85)

            # Count text-line runs WITHOUT merging across line gaps.
            line_runs = _runs(text_row_mask, min_len=max(3, h // 200), gap=2)
            # Reject blobs (low line count) — they're hands, shadows, edges.
            if len(line_runs) < self.min_text_rows:
                continue
            # Reject if a single run covers most of the strip height (solid blob).
            if max((e - s for s, e in line_runs), default=0) > h * 0.6:
                continue

            # y-extent: span of text rows only (solid bars excluded).
            extent_runs = _runs(text_row_mask, min_len=max(3, h // 200),
                                gap=max(8, h // 40))
            if not extent_runs:
                continue
            y0 = extent_runs[0][0]
            y1 = extent_runs[-1][1]
            if y1 - y0 < self.min_box_px:
                continue
            kept.append((int(cx0), int(y0), int(cx1), int(y1)))
        return kept


def _runs(mask: np.ndarray, *, min_len: int, gap: int) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) runs of True in mask, merging across gaps <= gap."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]
