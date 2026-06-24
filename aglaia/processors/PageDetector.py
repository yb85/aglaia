# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import os
import cv2
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait
from aglaia.processors.utils import is_binary, to_gray, to_rgb

from aglaia.processors.layout_backends import get_backend

@dataclass
class PageOption(AbstractProcessorOption):
    margin_mm: float = 2.0
    # Extra padding (mm) around the text bbox for the child ROI. The crop
    # rect remains page bbox + margin_mm; the ROI used downstream by
    # Binarizer is page bbox + roi_margin_mm. Should be ≤ margin_mm so
    # outside-ROI pixels still exist as a halo around the ROI rectangle
    # (those get masked to white after binarisation).
    roi_margin_mm: float = 1.0
    max_pages: int = 2
    # When more pages than max_pages are found: "merge" them into one
    # (default), or "discard" the extras (keep the largest) — single-page
    # modes use discard so marginal text from the facing page is dropped.
    over_cap: str = "merge"
    rescale_threshold: float = 0.01
    binarize_threshold: int = 127
    processing_dpi: Optional[float] = None
    backend: str = "auto"
    # Apple Vision only: smallest detectable text as a fraction of image
    # height. Lower = catches small running heads / page numbers. 0 = default.
    min_text_height: float = 0.01
    # Reject pages whose pixel dynamic range is below this fraction of
    # the MAX range across all merged pages in the same scan.
    # Polarity-agnostic, robust to global contrast variation, and works
    # for sparse text (one-line bboxes still span ink-to-paper range):
    #   range_i  = p95 - p5 inside page i
    #   relative = range_i / max(range_j)
    # Single-page scans always normalise to 1.0 → never dropped.
    # Well-lit pages sit > 0.7, but a real page in the gutter shadow / under
    # uneven lighting can dip to ~0.6 (e.g. a 2-page spread where one side is
    # brighter); bleed-through ghosts stay < 0.5. So the cutoff is 0.5 — drop
    # ghosts, keep dim-but-real pages. 0 = disabled.
    min_contrast: float = 0.5
    # Smart-merge tunables. See `smart_merge` / `_pair_score`.
    merge_threshold: float = 0.60
    merge_gap_weight: float = 0.4
    merge_width_weight: float = 0.6
    merge_gap_norm_cap: float = 0.15

    # We can pass specific config dict if needed, or mapping fields here.
    # The existing code utilized a nested 'config' dict.
    # We'll flatten what's needed.

def merge_layouts(boxes):
    if not boxes: return []
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    merged_groups = []
    if not sorted_boxes: return []
    curr_group = list(sorted_boxes[0])
    for box in sorted_boxes[1:]:
        bx1, by1, bx2, by2 = box
        gx1, gy1, gx2, gy2 = curr_group
        if gx1 < bx2 and bx1 < gx2:
            curr_group[0] = min(gx1, bx1)
            curr_group[1] = min(gy1, by1)
            curr_group[2] = max(gx2, bx2)
            curr_group[3] = max(gy2, by2)
        else:
            merged_groups.append(tuple(curr_group))
            curr_group = list(box)
    if curr_group: merged_groups.append(tuple(curr_group))
    return merged_groups

def _pair_score(a, b, *, page_w, page_h,
                gap_weight, width_weight, gap_norm_cap,
                contrast_bonus=0.0):
    """Mergeability score for two page rects.

    Combines axis IoU, size imbalance, and edge proximity. Score lives in
    ~[0,1] before contrast_bonus is added. Horizontal pair (Y-aligned,
    side-by-side) and vertical pair (X-aligned, stacked) are scored
    symmetrically; the max wins. Imbalance dominates so a thin column
    next to a wide one merges even when the gap is modest."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw, bw = max(0, ax2 - ax1), max(0, bx2 - bx1)
    ah, bh = max(0, ay2 - ay1), max(0, by2 - by1)

    y_inter = max(0, min(ay2, by2) - max(ay1, by1))
    y_union = max(ay2, by2) - min(ay1, by1)
    y_iou = (y_inter / y_union) if y_union > 0 else 0.0

    x_inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    x_union = max(ax2, bx2) - min(ax1, bx1)
    x_iou = (x_inter / x_union) if x_union > 0 else 0.0

    if ax2 < bx1:
        gap_h = bx1 - ax2
    elif bx2 < ax1:
        gap_h = ax1 - bx2
    else:
        gap_h = 0
    if ay2 < by1:
        gap_v = by1 - ay2
    elif by2 < ay1:
        gap_v = ay1 - by2
    else:
        gap_v = 0

    gap_h_norm = (gap_h / page_w) if page_w > 0 else 0.0
    gap_v_norm = (gap_v / page_h) if page_h > 0 else 0.0
    gap_cap = max(gap_norm_cap, 1e-6)

    w_max, w_min = max(aw, bw), min(aw, bw)
    h_max, h_min = max(ah, bh), min(ah, bh)
    w_imb = (1.0 - w_min / w_max) if w_max > 0 else 0.0
    h_imb = (1.0 - h_min / h_max) if h_max > 0 else 0.0

    prox_h = max(0.0, 1.0 - gap_h_norm / gap_cap)
    prox_v = max(0.0, 1.0 - gap_v_norm / gap_cap)

    score_h = y_iou * (gap_weight * prox_h + width_weight * w_imb)
    score_v = x_iou * (gap_weight * prox_v + width_weight * h_imb)
    return max(score_h, score_v) + contrast_bonus


def smart_merge(pages, *, max_pages, page_w, page_h,
                threshold=0.60, gap_weight=0.4, width_weight=0.6,
                gap_norm_cap=0.15, contrast_bonuses=None,
                over_cap_strategy="merge"):
    """Iteratively merge the best-scoring page pair.

    Two triggers, OR-combined:
      1. The top pair scores ≥ `threshold` (auto-merge true over-splits).
      2. `len(pages) > max_pages` (capacity cap).
    `contrast_bonuses[i]` is added to every score involving page i —
    use to bias ghost / low-contrast pages toward absorption.
    `max_pages <= 0` disables the capacity trigger.

    `over_cap_strategy` controls the capacity trigger (genuine over-splits
    at/above `threshold` always merge regardless):
      * `"merge"`   — merge the highest-scoring pair (default).
      * `"discard"` — drop the smallest page (ghost / marginal text from
        the facing page) instead of absorbing it. For single-page modes."""
    pages = [tuple(int(v) for v in l) for l in pages]
    bonuses = list(contrast_bonuses) if contrast_bonuses is not None \
        else [0.0] * len(pages)
    while len(pages) > 1:
        best_score = -1.0
        best_i = best_j = -1
        for i in range(len(pages)):
            for j in range(i + 1, len(pages)):
                s = _pair_score(
                    pages[i], pages[j],
                    page_w=page_w, page_h=page_h,
                    gap_weight=gap_weight, width_weight=width_weight,
                    gap_norm_cap=gap_norm_cap,
                    contrast_bonus=max(bonuses[i], bonuses[j]),
                )
                if s > best_score:
                    best_score = s
                    best_i, best_j = i, j
        if best_i < 0:
            break
        over_cap = max_pages > 0 and len(pages) > max_pages
        if best_score < threshold:
            if not over_cap:
                break
            if over_cap_strategy == "discard":
                # Over capacity, no genuine over-split: drop the smallest
                # page (facing-page ghost / marginal text) rather than
                # merge it in. Ties prefer dropping the higher-bonus ghost.
                drop = min(
                    range(len(pages)),
                    key=lambda k: ((pages[k][2] - pages[k][0])
                                   * (pages[k][3] - pages[k][1]),
                                   -bonuses[k]))
                pages.pop(drop)
                bonuses.pop(drop)
                continue
        a = pages[best_i]
        b = pages[best_j]
        merged = (
            min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]),
        )
        merged_bonus = max(bonuses[best_i], bonuses[best_j])
        survivors = []
        new_bonuses = []
        for k in range(len(pages)):
            if k in (best_i, best_j):
                continue
            survivors.append(pages[k])
            new_bonuses.append(bonuses[k])
        survivors.append(merged)
        new_bonuses.append(merged_bonus)
        pages = survivors
        bonuses = new_bonuses
    return sorted(pages, key=lambda l: (l[0], l[1]))


def reduce_pages(pages, target_count=2, *, page_w=None, page_h=None):
    """Legacy shim. Forwards to `smart_merge` with default weights."""
    if not pages:
        return []
    if page_w is None:
        page_w = max((l[2] for l in pages), default=1) or 1
    if page_h is None:
        page_h = max((l[3] for l in pages), default=1) or 1
    return [list(l) for l in
            smart_merge(pages, max_pages=target_count,
                        page_w=page_w, page_h=page_h)]

from aglaia.processors.option_specs import _b, _e, _f, _i


class PageDetector(AbstractImageProcessor):
    SUMMARY = "Text detection → child crops (splits 2-page spreads)."
    REPLAY_TRAIT = ReplayTrait.ROI  # crop + branch → fixed barrier / source anchor
    OPTION_CLASS = PageOption
    PROVIDES_META = {
        "roi": "detected page polygon [[x,y],...] in the child's coords",
        "page_side": "'left' | 'right' for a split 2-page spread",
        "parent_crop_xywh": "[x,y,w,h] of this child's bbox in parent coords",
    }
    _ESSENTIAL_PARAMS = ("backend", "max_pages", "margin_mm")
    OPTIONS = {
        "margin_mm": _f(2.0, 0.0, 50.0, 0.5,
                        "Padding around each detected text bbox before cropping."),
        "max_pages": _i(2, 0, 8,
                          "Max child crops per page (1 = single page, 2 = two-page spread, 0 = no cap)."),
        "over_cap": _e("merge", ["merge", "discard"],
                       "When more pages than max_pages are found: merge them "
                       "into one, or discard the extras (keep the largest). "
                       "Single-page modes use discard to drop facing-page text."),
        "rescale_threshold": _f(0.01, 0.001, 1.0, 0.001,
                                "Minimum DPI scale-factor delta to trigger a resize.",
                                advanced=True),
        "processing_dpi": _f(150.0, 36.0, 600.0, 10.0,
                             "Downsample to this DPI for detection; full-res used for cropping."),
        "backend": _e("auto",
                      ["auto", "east", "dbnet", "apple_vision", "heuristic"],
                      "Text detector backend. auto = apple_vision on macOS else EAST → DBnet → heuristic."),
        "min_text_height": _f(0.01, 0.0, 0.1, 0.005,
                              "Apple Vision only: smallest text to detect, as a fraction of "
                              "image height. Lower = catches running heads / page numbers the "
                              "page crop would otherwise clip. 0 = Vision's default (~0.03).",
                              advanced=True),
        "min_contrast": _f(0.5, 0.0, 1.0, 0.05,
                           "Drop pages whose (p95−p5) pixel range is below this fraction "
                           "of the max across all merged pages in the scan. Well-lit pages "
                           "> 0.7, dim/shadowed-but-real pages ~0.6; bleed-through < 0.5. "
                           "0 = disabled."),
        "merge_threshold": _f(0.60, 0.0, 1.5, 0.05,
                              "Auto-merge any adjacent page pair whose mergeability "
                              "score is ≥ this. Lower = merge more aggressively.",
                              advanced=True),
        "merge_gap_weight": _f(0.4, 0.0, 1.0, 0.05,
                               "Weight of gap-proximity in the merge score (vs width imbalance).",
                               advanced=True),
        "merge_width_weight": _f(0.6, 0.0, 1.0, 0.05,
                                 "Weight of width-imbalance in the merge score. Narrow-vs-wide "
                                 "pairs (e.g. page-number column) score higher.",
                                 advanced=True),
        "merge_gap_norm_cap": _f(0.15, 0.01, 0.5, 0.01,
                                 "Cap for gap normalisation (fraction of page extent). Gaps "
                                 "beyond this contribute nothing to the gap-proximity term.",
                                 advanced=True),
    }

    @classmethod
    def inject_step_options(cls, step_opts: dict, args) -> dict:
        """CLI override for max_pages. The pipeline editor sets the
        YAML default; `--max-pages` stomps it at chain-build time."""
        if hasattr(args, "max_pages"):
            step_opts["max_pages"] = args.max_pages
        return step_opts

    name: str = "PageDetector"

    def __init__(self, options: PageOption):
        super().__init__(options)
        from aglaia.processors.layout_backends.factory import LayoutModelUnavailable
        try:
            self.detector = get_backend(getattr(options, "backend", "auto"))
        except LayoutModelUnavailable:
            # No model + auto: pass pages through untouched rather than crop
            # them with the (removed-from-auto) heuristic. The GUI warns and
            # offers the downloader before processing; headless gates on
            # `aglaia --setup`. process() falls through via `if not detector`.
            self.detector = None
        # Apple Vision sensitivity to small text (running heads); harmless on
        # backends that don't expose the attribute.
        if self.detector is not None and hasattr(self.detector, "min_text_height"):
            self.detector.min_text_height = float(getattr(options, "min_text_height", 0.01))
        self.margin_mm = options.margin_mm
        self.roi_margin_mm = options.roi_margin_mm
        self.max_pages = options.max_pages
        self.over_cap = str(getattr(options, "over_cap", "merge"))
        self.rescale_threshold = options.rescale_threshold
        self.processing_dpi = options.processing_dpi
        self.min_contrast = float(getattr(options, "min_contrast", 0.0))
        self.merge_threshold = float(getattr(options, "merge_threshold", 0.60))
        self.merge_gap_weight = float(getattr(options, "merge_gap_weight", 0.4))
        self.merge_width_weight = float(getattr(options, "merge_width_weight", 0.6))
        self.merge_gap_norm_cap = float(getattr(options, "merge_gap_norm_cap", 0.15))
        self.uses_gpu = bool(getattr(self.detector, "uses_gpu", False))

    def process(self, input_buf: ImageBuffer) -> ImageBuffer:
        if not self.detector:
             return input_buf

        img_cv = input_buf.buffer
        h_orig, w_orig = img_cv.shape[:2]
        dpi = input_buf.dpi

        # Decision: do we downscale for detection?
        scale = 1.0
        if self.processing_dpi and self.processing_dpi < dpi:
            scale = self.processing_dpi / dpi
            h_proc, w_proc = int(h_orig * scale), int(w_orig * scale)
            img_proc = cv2.resize(img_cv, (w_proc, h_proc), interpolation=cv2.INTER_AREA)
            _work_dpi = float(self.processing_dpi)
        else:
            img_proc = img_cv
            h_proc, w_proc = h_orig, w_orig
            _work_dpi = float(dpi or 0)
        # Stash the working geometry so the chain's op-log size-chain
        # column shows the actual detection resolution (otherwise the
        # middle term collapses to input).
        self.last_stats = {
            "working_wh_dpi": ((w_proc, h_proc), _work_dpi),
        }

        # Detect
        boxes = self.detector.detect(img_proc)
        
        # Scale back if needed
        if scale != 1.0 and boxes:
            inv_scale = 1.0 / scale
            scaled_boxes = []
            for (bx1, by1, bx2, by2) in boxes:
                scaled_boxes.append((
                    int(bx1 * inv_scale),
                    int(by1 * inv_scale),
                    int(bx2 * inv_scale),
                    int(by2 * inv_scale)
                ))
            boxes = scaled_boxes

        if self.debug_enabled():
            vis = to_rgb(img_cv).copy() if img_cv.ndim == 2 else img_cv.copy()
            for (bx1, by1, bx2, by2) in (boxes or []):
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
            cv2.putText(vis, f"boxes={len(boxes or [])} backend={type(self.detector).__name__}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.debug_save(vis, "0_raw_boxes", input_buf)

        if not boxes:
            return input_buf

        pages = merge_layouts(boxes)
        pages.sort(key=lambda b: b[0])

        # Contrast filter — HARD DROP at the page level (post-merge).
        # Metric: (p95 - p5) of grayscale pixels inside the page,
        # normalised by the max across all pages in the scan. Works for
        # sparse text (a tight bbox around one line still spans ink-to-paper)
        # and is polarity-agnostic. Layouts whose rel_range falls below
        # `min_contrast` are removed before smart_merge runs — kills
        # bleed-through ghosts that would otherwise survive even an
        # absorption-friendly merge pass.
        if pages and self.min_contrast > 0.0 and len(pages) > 1:
            gray = img_cv if img_cv.ndim == 2 else cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            ranges = []
            for (lx1, ly1, lx2, ly2) in pages:
                lx1c = max(0, int(lx1)); ly1c = max(0, int(ly1))
                lx2c = min(gray.shape[1], int(lx2)); ly2c = min(gray.shape[0], int(ly2))
                if lx2c <= lx1c or ly2c <= ly1c:
                    ranges.append(0.0)
                    continue
                roi = gray[ly1c:ly2c, lx1c:lx2c]
                p5, p95 = np.percentile(roi, (5, 95))
                ranges.append(float(p95 - p5))
            max_range = max(ranges) if ranges else 0.0
            rels = [(r / max_range) if max_range > 0.0 else 1.0 for r in ranges]
            kept = []
            dropped = []
            for (lay, rel) in zip(pages, rels):
                if rel >= self.min_contrast:
                    kept.append(lay)
                else:
                    dropped.append((lay, rel))
            if self.debug_enabled():
                print(f"[PageDetector] min_contrast={self.min_contrast:.2f}  "
                      f"kept={len(kept)} dropped={len(dropped)}  "
                      f"rels={[round(r, 3) for r in rels]}",
                      flush=True)
                vis = to_rgb(img_cv).copy() if img_cv.ndim == 2 else img_cv.copy()
                for lay in kept:
                    lx1, ly1, lx2, ly2 = lay
                    cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 200, 0), 3)
                for (lay, rel) in dropped:
                    lx1, ly1, lx2, ly2 = lay
                    cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 0, 220), 3)
                    cv2.putText(vis, f"{rel:.2f}", (lx1 + 6, ly1 + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
                self.debug_save(vis, "0b_contrast_filter", input_buf)
            pages = kept

        if not pages:
            return input_buf

        # Smart merge: score-based, gap + width-imbalance aware.
        # Auto-merges over-split pairs (score ≥ threshold) and forces
        # merges to hit the `max_pages` cap.
        layouts_before = len(pages)
        pages = list(smart_merge(
            pages,
            max_pages=self.max_pages,
            page_w=w_orig, page_h=h_orig,
            threshold=self.merge_threshold,
            gap_weight=self.merge_gap_weight,
            width_weight=self.merge_width_weight,
            gap_norm_cap=self.merge_gap_norm_cap,
            over_cap_strategy=getattr(self, "over_cap", "merge"),
        ))
        # Surface n_layouts, merged count, and sizes through the chain
        # op-log line. Preserve the working_wh_dpi key we stashed before
        # detection so the size chain still renders correctly.
        sizes = [f"{lx2 - lx1}×{ly2 - ly1}"
                 for (lx1, ly1, lx2, ly2) in pages]
        self.last_stats.update({
            "pages": len(pages),
            "merged": max(0, layouts_before - len(pages)),
            "sizes": sizes,
        })
        if self.debug_enabled() and layouts_before != len(pages):
            print(f"[PageDetector] smart_merge: {layouts_before} → {len(pages)} "
                  f"(threshold={self.merge_threshold:.2f}, max={self.max_pages})",
                  flush=True)

        if self.debug_enabled():
            vis = to_rgb(img_cv).copy() if img_cv.ndim == 2 else img_cv.copy()
            for (bx1, by1, bx2, by2) in boxes:
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 200, 0), 1)
            for i, (lx1, ly1, lx2, ly2) in enumerate(pages):
                cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)
                cv2.putText(vis, chr(ord('A') + i), (lx1 + 6, ly1 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(vis, f"pages={len(pages)} max={self.max_pages}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.debug_save(vis, "1_layouts", input_buf)
        
        margin_px = int((self.margin_mm / 25.4) * dpi)
        base = input_buf.filestem or "capture"

        # 2-page spread detection: exactly two pages arranged side by
        # side (x-centre separation dominates y). Children get a
        # page_side meta ("left"/"right") that PageDewarper's flat_spline
        # model uses to locate the binding (left page → bound on its
        # right edge). Decided from coordinates, NOT page order — the
        # A/B suffix follows list order, which is not guaranteed sorted.
        spread_sides = None
        if len(pages) == 2:
            (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = pages
            dx = abs((ax1 + ax2) - (bx1 + bx2)) / 2.0
            dy = abs((ay1 + ay2) - (by1 + by2)) / 2.0
            if dx > dy:
                left_first = (ax1 + ax2) <= (bx1 + bx2)
                spread_sides = (("left", "right") if left_first
                                else ("right", "left"))

        def suffix_generator():
            for i in itertools.count():
                yield chr(ord('A') + i)
        suffixes = suffix_generator()

        # Create children buffers
        for idx, (lx1, ly1, lx2, ly2) in enumerate(pages):
            fx1 = max(0, lx1 - margin_px)
            fy1 = max(0, ly1 - margin_px)
            fx2 = min(w_orig, lx2 + margin_px)
            fy2 = min(h_orig, ly2 + margin_px)

            sub_img = img_cv[fy1:fy2, fx1:fx2]
            suffix = next(suffixes)
            filestem = f"{base}_{suffix}"

            # Tighten the ROI horizontally: drop outlier detector boxes
            # (cables, hands, cup edges intruding on the X axis) by
            # clamping left/right to the 5th-95th percentile of X edges of
            # detector boxes whose centre falls in the merged page rect.
            # Keep top/bottom at the full page extent — cropping Y kills
            # the first/last text lines and they rarely have intruders.
            inner = [b for b in boxes
                     if lx1 <= (b[0] + b[2]) / 2 <= lx2
                     and ly1 <= (b[1] + b[3]) / 2 <= ly2]
            if len(inner) >= 4:
                xl = np.array([b[0] for b in inner])
                xr = np.array([b[2] for b in inner])
                tight_lx1 = int(np.percentile(xl, 5))
                tight_lx2 = int(np.percentile(xr, 95))
            else:
                tight_lx1, tight_lx2 = lx1, lx2
            # Extend the vertical extent to include a running head above / a
            # page number below the merged body rect: boxes inside the page's
            # text column (X span) but just outside its Y span. Clustering to
            # the dense body otherwise drops these isolated lines.
            tight_ly1, tight_ly2 = ly1, ly2
            xcol = [b for b in boxes
                    if lx1 <= (b[0] + b[2]) / 2 <= lx2
                    and fy1 <= (b[1] + b[3]) / 2 <= fy2]
            if xcol:
                tight_ly1 = min(tight_ly1, min(b[1] for b in xcol))
                tight_ly2 = max(tight_ly2, max(b[3] for b in xcol))

            roi_pad = int((self.roi_margin_mm / 25.4) * dpi)
            roi_x1 = max(0, tight_lx1 - fx1 - roi_pad)
            roi_y1 = max(0, tight_ly1 - fy1 - roi_pad)
            roi_x2 = min(fx2 - fx1, tight_lx2 - fx1 + roi_pad)
            roi_y2 = min(fy2 - fy1, tight_ly2 - fy1 + roi_pad)
            child_roi = [
                [float(roi_x1), float(roi_y1)],
                [float(roi_x2), float(roi_y1)],
                [float(roi_x2), float(roi_y2)],
                [float(roi_x1), float(roi_y2)],
            ]
            if parent_roi := input_buf.meta.get("roi"):
                # Intersect text-tight child_roi (in child coords) with
                # parent_roi (translated to child coords). The previous code
                # overwrote child_roi with parent ∩ crop_rect, which clobbered
                # the text-tight padding whenever parent_roi was the
                # SkewFinder-rotated full-image rect (i.e. always).
                parent_in_child = np.array(parent_roi, dtype=np.float32).copy()
                parent_in_child[:, 0] -= fx1
                parent_in_child[:, 1] -= fy1
                child_poly = np.array(child_roi, dtype=np.float32)
                try:
                    ret, intersection = cv2.intersectConvexConvex(
                        child_poly, parent_in_child
                    )
                    if ret > 0 and intersection is not None:
                        child_roi = intersection.reshape(-1, 2).astype(float).tolist()
                except Exception:
                    pass
            
            l_buf = ImageBuffer(
                sub_img,
                ImageType.COLOR,
                dpi=dpi,
                path=None,
                parent=input_buf,
                filestem=filestem,
                out_dir=input_buf.out_dir, # Propagate out_dir preference
                scan_id=input_buf.scan_id,
                pipeline_version_id=input_buf.pipeline_version_id,
                branch_label=suffix,
            )
            
            if child_roi:
                l_buf.meta["roi"] = child_roi

            if spread_sides is not None:
                l_buf.meta["page_side"] = spread_sides[idx]

            # Stamp the crop offset/size into parent coords so downstream
            # debug renderers can draw the child's bbox on the parent
            # (full deskewed) image — otherwise we lose the spatial
            # context once the buffer is cropped.
            l_buf.meta["parent_crop_xywh"] = [
                int(fx1), int(fy1),
                int(fx2 - fx1), int(fy2 - fy1),
            ]

            # NOTE: We do NOT write here; the chain owns persistence.
            input_buf.children.append(l_buf)
            
        return input_buf.children if input_buf.children else [input_buf]
