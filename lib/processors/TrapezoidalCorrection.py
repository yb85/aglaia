# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
TrapezoidalCorrection — keystone (pure perspective) correction by
text-line vanishing-point estimation.

See `docs/algorithms.md` for the column-quad / keystone design.

Phase 1+2 scope (this file):
- `line_source = "smear"` only (extract bboxes from BW via horizontal
  morphological closing + connected components).
- `mode in {"affine", "metric"}`.
- Metric upgrade uses the **line-pitch** constraint (no glyph-vertical
  detection yet — that's phase 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from lib.ImageBuffer import ImageBuffer, ImageType
from lib.Status import Status
from lib.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait
from lib.processors.geometry import (
    detect_column_quad_from_baselines, baseline_from_ink,
    zhang_he_aspect_and_focal,
)
from lib.processors.utils import binarize_fixed, to_gray


@dataclass
class TrapezoidalOption(AbstractProcessorOption):
    # --- input geometry source ---
    line_source: str = "connectivity"   # connectivity | meta
    min_line_count: int = 4
    # Horizontal MORPH_CLOSE kernel fallback (mm). Used when text-scale
    # auto-estimation gathers < 30 char-like CCs. 4.0 mm ≈ 50 px @ 300 dpi.
    line_join_mm: float = 4.0
    # Adaptive kernel width = kernel_char_mult × median char height.
    kernel_char_mult: float = 2.0
    # TEXT_MAX_THICKNESS = thickness_char_mult × median char height
    # (≈ ascender + x-height + descender).
    thickness_char_mult: float = 3.0
    # EDGE_MAX_LENGTH = edge_max_length_char_mult × median char height.
    edge_max_length_char_mult: float = 3.0

    # --- VP solving ---
    ransac_trials: int = 200
    ransac_angle_eps_deg: float = 0.3

    # --- column-edge fit (clustered margin estimator, §3.6.7) ---
    # Hard verticality prior on the L/R margin fit. Page is deskewed
    # upstream — margins can't tilt more than this from vertical.
    edge_max_tilt_deg: float = 7.0
    # Intercept-cluster split threshold as a fraction of body width:
    # gaps larger than this in de-drifted endpoint x start a new margin
    # cluster (body margin vs blockquote indent).
    edge_cluster_gap_pct: float = 0.03
    # Vertical-block split: a gap between adjacent baselines larger than
    # this × the median line pitch starts a new vertical block (body text
    # vs critical apparatus / footnotes). ~2.5 splits blocks without
    # splitting paragraphs.
    vblock_gap_mult: float = 2.5
    # Blocks are then merged into columns when their left AND right margins
    # match within this fraction of the body width (a body column split by
    # a section gap re-merges; an off-column apparatus/gloss stays out).
    # The quad fits the dominant merged column.
    vblock_edge_tol_pct: float = 0.04

    # --- analysis ---
    # Detection (binarize, CCs, morphology, spans, baselines, quad) runs
    # at this DPI; only the final warpPerspective touches the full-res
    # buffer. Same convention as PageDewarper / PageDetector.
    processing_dpi: float = 150.0

    # --- output ---
    # `dpi` removed — output buffer inherits input buf.dpi. Per-step DPI
    # forcing duplicated DPIfixer + made margin_mm sensitive to a
    # forgotten setting (margin_px = mm × dpi / 25.4).
    margin_mm: float = 3.0
    interp: str = "auto"                # auto | nearest | linear | cubic

    # --- validation ---
    min_quad_area_ratio: float = 0.10
    max_aspect_ratio: float = 5.0
    max_baseline_tilt_deg: float = 35.0
    # Skip Zhang-He metric upgrade when the quad is nearly axis-aligned
    # (skew_ratio = max edge drift / max bbox dim).
    zhang_he_min_skew: float = 0.05
    # Reject Zhang-He's recovered aspect if it drifts more than this
    # fraction from the bbox aspect — guard against degenerate solves.
    zhang_he_max_drift: float = 0.15

    # Pure UI-surfacing flag: when True, the GUI shows + pre-ticks the
    # "Normalize character width" checkbox so the user knows the
    # pipeline relies on `BlobNormalizer` for batch-wide horizontal
    # consistency. No runtime effect on TrapezoidalCorrection itself.
    norm_width: bool = False


def _interp_flag(name: str, *, force_nearest: bool) -> int:
    if force_nearest:
        return cv2.INTER_NEAREST
    return {
        "auto":    cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "linear":  cv2.INTER_LINEAR,
        "cubic":   cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }.get(name, cv2.INTER_CUBIC)


from lib.processors.option_specs import _b, _e, _f, _i


class TrapezoidalCorrection(AbstractImageProcessor):
    name: str = "TrapezoidalCorrection"
    SUMMARY = "Perspective (keystone) rectification via column-quad detection + Zhang-He."
    REPLAY_TRAIT = ReplayTrait.COORDINATE  # perspective warp
    OPTION_CLASS = TrapezoidalOption
    _ESSENTIAL_PARAMS = ("line_source", "interp", "processing_dpi")
    OPTIONS = {
        "line_source": _e("connectivity", ["connectivity", "meta"],
                          "Where to source text-line bboxes from. "
                          "connectivity = morphological analysis; meta = PageDetector bboxes.",
                          advanced=True),
        "min_line_count": _i(4, 1, 50,
                             "Below this many detected lines, fall back to passthrough."),
        "line_join_mm": _f(4.0, 0.5, 20.0, 0.5,
                           "Horizontal MORPH_CLOSE kernel for line bridging (mm at page DPI).",
                           advanced=True),
        "kernel_char_mult": _f(2.0, 0.5, 6.0, 0.1,
                               "Adaptive MORPH_CLOSE kernel width = mult × median char height."),
        "thickness_char_mult": _f(3.0, 1.0, 8.0, 0.1,
                                  "TEXT_MAX_THICKNESS = mult × median char height."),
        "edge_max_length_char_mult": _f(3.0, 0.5, 12.0, 0.1,
                                        "EDGE_MAX_LENGTH = mult × median char height. "
                                        "Max gap between adjacent cinfos the span-builder will link."),
        "processing_dpi": _f(150.0, 50.0, 600.0, 25.0,
                             "Analysis resolution for line/quad detection. "
                             "Final warp always runs at full input resolution.",
                             advanced=True),
        "ransac_trials": _i(200, 20, 2000,
                            "RANSAC trials for the column-edge + VP fits.",
                            advanced=True),
        "ransac_angle_eps_deg": _f(0.3, 0.05, 5.0, 0.05,
                                   "RANSAC inlier tolerance for the VP fit (degrees).",
                                   advanced=True),
        "edge_max_tilt_deg": _f(7.0, 1.0, 20.0, 0.5,
                                "Max tilt from vertical allowed for the "
                                "L/R column-margin fit (page is deskewed "
                                "upstream).", advanced=True),
        "edge_cluster_gap_pct": _f(0.03, 0.005, 0.2, 0.005,
                                   "Margin-cluster split threshold as a "
                                   "fraction of body width — separates the "
                                   "body margin from blockquote indents.",
                                   advanced=True),
        "vblock_gap_mult": _f(2.5, 1.5, 6.0, 0.1,
                              "Vertical-block split: baseline gap (× median "
                              "line pitch) that starts a new block (body vs "
                              "apparatus/footnotes).",
                              advanced=True),
        "vblock_edge_tol_pct": _f(0.04, 0.0, 0.2, 0.005,
                                  "Merge vertical blocks into one column "
                                  "when their L/R margins match within this "
                                  "fraction of body width; the dominant "
                                  "column feeds the quad fit.",
                                  advanced=True),
        "margin_mm": _f(3.0, 0.0, 30.0, 0.5,
                        "Extra padding around the rectified column quad.",
                        advanced=True),
        "interp": _e("auto", ["auto", "nearest", "linear", "cubic"],
                     "Remap interpolation mode.", advanced=True),
        "min_quad_area_ratio": _f(0.1, 0.0, 1.0, 0.01,
                                  "Reject if column-quad covers less of the crop than this.",
                                  advanced=True),
        "zhang_he_min_skew": _f(0.05, 0.0, 1.0, 0.005,
                                "Skip Zhang-He upgrade when quad is nearly axis-aligned.",
                                advanced=True),
        "zhang_he_max_drift": _f(0.15, 0.0, 1.0, 0.01,
                                 "Reject ZH aspect if it drifts more than this fraction from bbox aspect.",
                                 advanced=True),
        "max_baseline_tilt_deg": _f(35.0, 5.0, 60.0, 1.0,
                                    "Reject if median baseline tilt exceeds this.",
                                    advanced=True),
        "max_aspect_ratio": _f(5.0, 1.0, 20.0, 0.5,
                               "Sanity cap on recovered w/h.", advanced=True),
        "norm_width": _b(False,
                         "Surface the 'Normalize character width' checkbox in the GUI (pre-ticked).",
                         advanced=True),
    }

    def __init__(self, options: TrapezoidalOption):
        super().__init__(options)
        self.opt = options
        self.uses_gpu = False
        self._last_source_label: str = ""

    # ── line bbox detection ────────────────────────────────────────────
    def _line_bboxes_from_connectivity(self, bw: np.ndarray,
                                       dpi: float) -> list[tuple[int, int, int, int]]:
        """Return (x, y, w, h) bboxes of text lines via page-dewarp methodology.

        Strategy:
          1. Horizontal MORPH_CLOSE with a kernel sized from DPI (covers
             inter-word gaps without growing blob bounds). Same kernel used
             for dilate + erode steps, so line bboxes stay tight.
          2. ``get_contours`` filters by aspect/size/thickness.
          3. ``assemble_spans`` chains word contours into per-line spans by
             overlap + angle compatibility.
        """
        from page_dewarp.contours import get_contours
        from page_dewarp.spans import assemble_spans
        from page_dewarp.options import cfg

        if bw.ndim == 3:
            bw = to_gray(bw)
        ink = cv2.bitwise_not(bw) if bw.mean() > 127 else bw

        H_img, W_img = ink.shape
        # Char-like CC filter bounds derived from page DPI so the
        # estimator survives both very-low and very-high DPI inputs.
        cc_h_min = max(3, int(round(dpi * 0.04)))
        cc_h_max = max(cc_h_min + 1, int(round(dpi * 0.45)))
        cc_w_min = max(2, int(round(dpi * 0.02)))
        cc_w_max = max(cc_w_min + 1, int(round(dpi * 0.60)))
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
        char_h = [int(s[3]) for s in stats[1:]
                  if cc_h_min <= s[3] <= cc_h_max
                  and cc_w_min <= s[2] <= cc_w_max]
        if len(char_h) >= 30:
            h_med = float(np.median(char_h))
            kw = max(9, int(round(self.opt.kernel_char_mult * h_med)))
        else:
            h_med = 0.0
            kw = max(9, int(round(self.opt.line_join_mm * dpi / 25.4)))
        # Break thin vertical bridges between adjacent text lines BEFORE
        # the horizontal close. Without this, a single 1-2 px tall ink
        # chain between line N descender and line N+1 ascender (binariser
        # speckle or genuine touching glyphs) is widened by horizontal
        # dilation into a stripe that welds the two lines into one
        # multi-row connected component. Multi-line cinfos then trip
        # baseline_from_ink (per-column bottom-most ink jumps between
        # lines).
        vbreak = max(3, int(round(h_med / 6))) if h_med > 0 else 3
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vbreak))
        ink_clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        morphed = cv2.morphologyEx(ink_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, morphed = cv2.threshold(morphed, 127, 255, cv2.THRESH_BINARY)

        # Page-dewarp filter thresholds tied to h_med (DPI fallback).
        saved = {k: getattr(cfg, k) for k in (
            "TEXT_MAX_THICKNESS", "TEXT_MIN_WIDTH", "TEXT_MIN_HEIGHT",
            "EDGE_MAX_LENGTH", "EDGE_MAX_OVERLAP", "EDGE_MAX_ANGLE",
            "SPAN_MIN_WIDTH",
        )}
        if h_med > 0:
            cfg.TEXT_MAX_THICKNESS = max(10, int(round(
                self.opt.thickness_char_mult * h_med)))
            cfg.TEXT_MIN_WIDTH = max(8, int(round(0.5 * h_med)))
            cfg.TEXT_MIN_HEIGHT = max(2, int(round(0.5 * h_med)))
            cfg.EDGE_MAX_LENGTH = max(20, int(round(
                self.opt.edge_max_length_char_mult * h_med)))
            cfg.EDGE_MAX_OVERLAP = max(2.0, 0.1 * h_med)
            cfg.SPAN_MIN_WIDTH = max(30, int(round(10.0 * h_med)))
        else:
            cfg.TEXT_MAX_THICKNESS = max(10, int(round(dpi * 0.25)))
            cfg.TEXT_MIN_WIDTH = max(8, int(round(dpi * 0.10)))
            cfg.TEXT_MIN_HEIGHT = max(2, int(round(dpi * 0.01)))
            cfg.EDGE_MAX_LENGTH = max(100, int(round(dpi * 0.5)))
            cfg.EDGE_MAX_OVERLAP = max(2.0, dpi * 0.02)
            cfg.SPAN_MIN_WIDTH = max(30, W_img // 20)
        cfg.EDGE_MAX_ANGLE = 7.5
        try:
            rgb = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
            cinfos = get_contours("trap", rgb, morphed)
            if not cinfos:
                return []
            pagemask = np.full((H_img, W_img), 255, dtype=np.uint8)
            spans = assemble_spans("trap", rgb, pagemask, cinfos)
            if self.opt.debug:
                widths = [int(getattr(c, "local_xrng", (0, 0))[1] -
                              getattr(c, "local_xrng", (0, 0))[0])
                          for c in cinfos]
                print(f"[Trap] cinfos={len(cinfos)} spans={len(spans)} "
                      f"span_counts={[len(s) for s in spans]} "
                      f"SPAN_MIN_WIDTH={cfg.SPAN_MIN_WIDTH} "
                      f"TEXT_MAX_THICK={cfg.TEXT_MAX_THICKNESS} "
                      f"cinfo_widths sample={widths[:10]}", flush=True)
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        bboxes: list[tuple[int, int, int, int]] = []
        span_masks: list[np.ndarray] = []
        for span in spans:
            xs: list[int] = []
            ys: list[int] = []
            sm = np.zeros((H_img, W_img), dtype=np.uint8)
            for ci in span:
                x, y, w, h = ci.rect
                xs.extend([x, x + w])
                ys.extend([y, y + h])
                sub = sm[y:y + h, x:x + w]
                tm = ci.mask
                if tm.dtype != np.uint8:
                    tm = tm.astype(np.uint8) * 255
                sub |= tm
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            bboxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
            span_masks.append(sm)
        self._last_span_masks = span_masks
        return bboxes

    # ── orchestration ─────────────────────────────────────────────────
    def process(self, buf: ImageBuffer) -> ImageBuffer:
        """Column-quadrilateral + Zhang-He homography.

        The page is assumed to be a flat rectangle photographed off-axis,
        so its boundary in the image is a convex quadrilateral. We locate
        that quad from the body-text column (left/right column edges
        traced through the start/end of justified text lines; top/bottom
        from the first/last full-line baselines), recover its true aspect
        via Zhang-He (2002), and apply the single 4-point perspective
        transform.
        """
        # 1. Resolve a BW working image at analysis resolution. All
        # detection below runs on the downsampled frame; the quad scales
        # back exactly (4 corner points) and the final warpPerspective is
        # the only full-res operation.
        H_src, W_src = buf.buffer.shape[:2]
        src_dpi = float(buf.dpi or 0.0)
        ana_scale = 1.0
        analysis_dpi = src_dpi if src_dpi > 0 else self.opt.processing_dpi
        if (self.opt.processing_dpi and src_dpi > self.opt.processing_dpi):
            ana_scale = self.opt.processing_dpi / src_dpi
            analysis_dpi = self.opt.processing_dpi
        if ana_scale < 1.0:
            # NEAREST on BW preserves the 0/255 palette; AREA elsewhere.
            interp_dn = (cv2.INTER_NEAREST if buf.type == ImageType.BW
                         else cv2.INTER_AREA)
            small = cv2.resize(
                buf.buffer,
                (max(1, int(round(W_src * ana_scale))),
                 max(1, int(round(H_src * ana_scale)))),
                interpolation=interp_dn,
            )
        else:
            small = buf.buffer
        if buf.type == ImageType.BW:
            bw = small if small.ndim == 2 else to_gray(small)
        else:
            bw = binarize_fixed(small, 127)
        ink_arr = cv2.bitwise_not(bw) if bw.mean() > 127 else bw

        # 2. Line bboxes + baselines (analysis coords). Bboxes are kept
        # only as the "rectangle that contains the span"; ALL geometric
        # reasoning (full-width selection, column quad) runs on the
        # baseline endpoints (line estimates), the same way page-dewarp
        # does.
        if self.opt.line_source == "meta":
            line_bboxes = [
                tuple(int(round(v * ana_scale)) for v in bb)
                for bb in (buf.meta.get("line_boxes") or [])
            ]
            self._last_source_label = "meta"
        else:
            line_bboxes = self._line_bboxes_from_connectivity(bw, analysis_dpi)
            self._last_source_label = "connectivity"

        # Fit a baseline per bbox via robust bottom-contour scan. Drop
        # bboxes that can't be fit — they don't represent text lines.
        baselines: list[tuple[np.ndarray, np.ndarray]] = []
        baseline_bboxes: list[tuple[int, int, int, int]] = []
        span_masks_local = getattr(self, "_last_span_masks", None) or []
        for idx, bb in enumerate(line_bboxes):
            sm = (span_masks_local[idx]
                  if idx < len(span_masks_local) else None)
            bl = baseline_from_ink(ink_arr, bb, span_mask=sm)
            if bl is None:
                continue
            baselines.append(bl)
            baseline_bboxes.append(bb)
        # Masks served their purpose — release (one full analysis frame
        # per span; kept alive across images otherwise).
        self._last_span_masks = []

        if self.debug_enabled():
            self.debug_save(bw, "0_input_bw", buf)
            vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            for (x, y, w, h) in line_bboxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
            for pL, pR in baselines:
                cv2.line(vis, (int(pL[0]), int(pL[1])),
                         (int(pR[0]), int(pR[1])), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"bboxes={len(line_bboxes)} baselines={len(baselines)} src={self._last_source_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            self.debug_save(vis, "1_line_bboxes", buf)

        if len(baselines) < self.opt.min_line_count:
            return self._fallback(buf, reason="too few lines", n_lines=len(baselines))

        # 3. Column quadrilateral via the decoupled estimator: VP from
        # all baselines (angles, RANSAC + TLS), L/R + top/bot anchors
        # from the full-width subset only. See docs/algorithms.md
        # §3.6.2.  The old `min_full_width_coverage` hard gate is gone:
        # angle evidence from short lines is no longer discarded, so the
        # only structural failure mode is < 3 full-width survivors
        # (returned as None by the helper).
        result = detect_column_quad_from_baselines(
            baselines,
            ransac_trials=self.opt.ransac_trials,
            ransac_eps_deg=self.opt.ransac_angle_eps_deg,
            edge_max_tilt_deg=self.opt.edge_max_tilt_deg,
            edge_cluster_gap_pct=self.opt.edge_cluster_gap_pct,
            vblock_gap_mult=self.opt.vblock_gap_mult,
            vblock_edge_tol_pct=self.opt.vblock_edge_tol_pct,
        )
        if result is None:
            return self._fallback(
                buf,
                reason="could not isolate column quadrilateral",
                n_lines=len(baselines),
            )
        quad, quad_info = result
        full_idxs = quad_info["full_width_idxs"]
        vp_inlier_frac = quad_info["vp_inlier_frac"]

        if self.debug_enabled():
            vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            for i, (pL, pR) in enumerate(baselines):
                col = (0, 255, 0) if i in full_idxs else (0, 128, 255)
                cv2.line(vis, (int(pL[0]), int(pL[1])),
                         (int(pR[0]), int(pR[1])), col, 2, cv2.LINE_AA)
                cv2.circle(vis, (int(pL[0]), int(pL[1])), 4, col, -1)
                cv2.circle(vis, (int(pR[0]), int(pR[1])), 4, col, -1)
            cv2.putText(
                vis,
                f"full={len(full_idxs)}/{len(baselines)} "
                f"vblocks={quad_info.get('n_vblocks', 1)} "
                f"vp_inliers={quad_info['n_vp_inliers']} "
                f"({vp_inlier_frac:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.debug_save(vis, "2_full_width", buf)

        if self.debug_enabled() and quad is not None:
            vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            pts = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            for i, p in enumerate(quad.astype(int)):
                cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
                cv2.putText(vis, str(i), tuple(p + 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            self.debug_save(vis, "3_column_quad", buf)

        # Back to source coordinates — exact for the 4 corner points;
        # everything below (Zhang-He, margins, warp) is full-res.
        if ana_scale < 1.0:
            quad = quad / ana_scale

        # Sanity check: the quad must be convex and cover a meaningful fraction of the crop.
        if not _is_convex(quad):
            return self._fallback(buf, reason="column quad not convex")
        quad_area = _polygon_area(quad)
        crop_area = W_src * H_src
        if quad_area < self.opt.min_quad_area_ratio * crop_area:
            return self._fallback(buf, reason=f"column quad too small ({quad_area/crop_area:.2f})")

        # 4. Recover true aspect. Zhang-He is closed-form but very sensitive
        # to near-axis-aligned quads; sanity-check against bbox aspect and
        # fall back if the gap is implausible.
        w_bbox = (np.linalg.norm(quad[1] - quad[0])
                  + np.linalg.norm(quad[2] - quad[3])) / 2.0
        h_bbox = (np.linalg.norm(quad[3] - quad[0])
                  + np.linalg.norm(quad[2] - quad[1])) / 2.0
        bbox_aspect = float(w_bbox / max(h_bbox, 1.0))

        zh_aspect, recovered_focal = zhang_he_aspect_and_focal(
            quad, (W_src, H_src), focal_px=None,
        )
        # Skew magnitude: how far the quad departs from axis-aligned.
        skew_px = max(
            abs(quad[0, 0] - quad[3, 0]),  # left edge x-drift
            abs(quad[1, 0] - quad[2, 0]),  # right edge x-drift
            abs(quad[0, 1] - quad[1, 1]),  # top edge y-drift
            abs(quad[3, 1] - quad[2, 1]),  # bottom edge y-drift
        )
        skew_ratio = skew_px / max(w_bbox, h_bbox, 1.0)

        if (zh_aspect is not None
                and skew_ratio >= self.opt.zhang_he_min_skew
                and abs(zh_aspect - bbox_aspect) / bbox_aspect
                    <= self.opt.zhang_he_max_drift):
            recovered_aspect = float(zh_aspect)
            mode_used = "metric_zhang_he"
        else:
            recovered_aspect = bbox_aspect
            mode_used = "bbox"

        margin_px = int(round(self.opt.margin_mm * buf.dpi / 25.4))

        # 5. Map the column quad to an axis-aligned rectangle at output DPI.
        h_pix = max(1.0, h_bbox)
        col_h = max(1, int(round(h_pix)))
        col_w = max(1, int(round(col_h * recovered_aspect)))
        dst_col = np.array([
            [0.0,         0.0],
            [col_w - 1.0, 0.0],
            [col_w - 1.0, col_h - 1.0],
            [0.0,         col_h - 1.0],
        ], dtype=np.float32)
        H_col = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_col)

        # 6. Expand canvas so the full SOURCE image is preserved — nothing
        # outside the column quad gets cropped.
        src_corners = np.array(
            [[0, 0], [W_src - 1, 0], [W_src - 1, H_src - 1], [0, H_src - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(src_corners, H_col).reshape(-1, 2)
        x_min, y_min = warped_corners.min(axis=0)
        x_max, y_max = warped_corners.max(axis=0)
        tx = margin_px - x_min
        ty = margin_px - y_min
        T = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        H_total = T @ H_col
        canvas_w = int(math.ceil(x_max - x_min)) + 2 * margin_px
        canvas_h = int(math.ceil(y_max - y_min)) + 2 * margin_px

        force_nearest = (buf.type == ImageType.BW)
        flag = _interp_flag(self.opt.interp, force_nearest=force_nearest)
        # White background outside the source. BORDER_REPLICATE smeared
        # dark binarizer artefacts at the source edge into large black
        # blobs in the expanded canvas; constant white avoids that.
        if buf.type == ImageType.BW or buf.buffer.ndim == 2:
            border_val = 255
        else:
            border_val = (255, 255, 255)
        rectified = cv2.warpPerspective(
            buf.buffer, H_total, (canvas_w, canvas_h),
            flags=flag, borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )

        # 7. Validation. oob_fraction not meaningful here (canvas sized to
        # include all source); enforce only the aspect-ratio bound.
        oob = 0.0
        if not (1.0 / self.opt.max_aspect_ratio
                <= recovered_aspect
                <= self.opt.max_aspect_ratio):
            return self._fallback(buf, reason=f"recovered aspect {recovered_aspect:.2f} out of bounds")

        out = ImageBuffer(
            rectified, buf.type, dpi=buf.dpi,
            filestem=buf.filestem,
            parent=buf,
            scan_id=buf.scan_id,
            parent_node_id=buf.parent_node_id,
            pipeline_version_id=buf.pipeline_version_id,
            depth=buf.depth,
            branch_label=buf.branch_label,
        )
        # Forward the ROI polygon through H_total so a downstream
        # Binarizer can still mask off the non-page area.
        if (old_roi := buf.meta.get("roi")):
            pts = np.array(old_roi, dtype=np.float32).reshape(-1, 1, 2)
            new_pts = cv2.perspectiveTransform(pts, H_total.astype(np.float64))
            out.meta["roi"] = new_pts.reshape(-1, 2).tolist()
        else:
            # No upstream ROI — use the rectified column quad as ROI.
            out.meta["roi"] = [
                [0.0, 0.0],
                [float(canvas_w - 1), 0.0],
                [float(canvas_w - 1), float(canvas_h - 1)],
                [0.0, float(canvas_h - 1)],
            ]
        out.meta.update({
            "trapezoid_success": True,
            "status": int(Status.SUCCESS),
            "n_baselines": len(baselines),
            "n_full_width": int(quad_info["n_full_width"]),
            "n_vblocks": int(quad_info.get("n_vblocks", 1)),
            "n_vp_inliers": int(quad_info["n_vp_inliers"]),
            "vp_inlier_frac": float(quad_info["vp_inlier_frac"]),
            "column_edge_source": quad_info["column_edge_source"],
            "column_quad": quad.tolist(),
            "recovered_aspect_w_h": recovered_aspect,
            "recovered_focal_px": recovered_focal,
            "H": H_total.tolist(),
            "mode_used": mode_used,
            "line_source": self._last_source_label,
            "oob_pct": oob,
            # Replay params: single 3x3 homography on the source buffer.
            "replay_kind": "perspective",
            "replay_params": {
                "H": H_total.tolist(),
                "canvas_wh": [int(canvas_w), int(canvas_h)],
                "src_wh": [int(W_src), int(H_src)],
            },
        })
        if self.debug_enabled():
            self.debug_save(rectified, "4_rectified", buf)
            # Transform-grid overlay: sample a uniform grid on the output
            # canvas, project back through H_total^-1 to source, draw on
            # both the source (warp mesh) and the output (uniform mesh).
            try:
                H_inv = np.linalg.inv(H_total)
                step = max(40, min(canvas_w, canvas_h) // 16)
                xs = np.arange(0, canvas_w, step)
                ys = np.arange(0, canvas_h, step)
                src_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                dst_vis = rectified.copy()
                if dst_vis.ndim == 2:
                    dst_vis = cv2.cvtColor(dst_vis, cv2.COLOR_GRAY2BGR)
                # src_vis is the ANALYSIS-res bw; H_inv yields source
                # coords — scale down by ana_scale for drawing.
                for y in ys:
                    pts = np.array([[x, y] for x in xs], dtype=np.float32).reshape(-1, 1, 2)
                    src = (cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
                           * ana_scale).astype(np.int32)
                    cv2.polylines(src_vis, [src], False, (0, 220, 60), 1, cv2.LINE_AA)
                    cv2.polylines(dst_vis, [pts.astype(np.int32).reshape(-1, 2)],
                                  False, (0, 220, 60), 1, cv2.LINE_AA)
                for x in xs:
                    pts = np.array([[x, y] for y in ys], dtype=np.float32).reshape(-1, 1, 2)
                    src = (cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
                           * ana_scale).astype(np.int32)
                    cv2.polylines(src_vis, [src], False, (0, 220, 60), 1, cv2.LINE_AA)
                    cv2.polylines(dst_vis, [pts.astype(np.int32).reshape(-1, 2)],
                                  False, (0, 220, 60), 1, cv2.LINE_AA)
                pts = (quad * ana_scale).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(src_vis, [pts], True, (0, 0, 255), 2)
                self.debug_save(src_vis, "5_grid_source", buf)
                self.debug_save(dst_vis, "6_grid_rectified", buf)
            except Exception as e:
                print(f"[{self.name}] grid debug failed: {e}")
        return out

    # ── fallback path ─────────────────────────────────────────────────
    def _fallback(self, buf: ImageBuffer, *, reason: str, n_lines: int = 0) -> ImageBuffer:
        out = ImageBuffer(
            buf.buffer.copy(), buf.type, dpi=buf.dpi,
            filestem=buf.filestem,
            parent=buf,
            scan_id=buf.scan_id,
            parent_node_id=buf.parent_node_id,
            pipeline_version_id=buf.pipeline_version_id,
            depth=buf.depth,
            branch_label=buf.branch_label,
        )
        # Passthrough: image unchanged, propagate ROI so downstream
        # Binarizer can still mask off the non-page area. Default to the
        # full image rect if no upstream ROI.
        if (roi := buf.meta.get("roi")):
            out.meta["roi"] = roi
        else:
            h_img, w_img = buf.buffer.shape[:2]
            out.meta["roi"] = [
                [0.0, 0.0],
                [float(w_img - 1), 0.0],
                [float(w_img - 1), float(h_img - 1)],
                [0.0, float(h_img - 1)],
            ]
        out.meta.update({
            "trapezoid_success": False,
            "status": int(Status.REVIEW),
            "fallback_reason": reason,
            "n_baselines": n_lines,
            "line_source": self._last_source_label,
            "mode_used": "fallback_passthrough",
        })
        return out


# ───────────────────────────────────────────────────────────────────────────
# Small inline helpers
# ───────────────────────────────────────────────────────────────────────────

def _apply_h(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    pt = np.array([p[0], p[1], 1.0])
    out = H @ pt
    if abs(out[2]) < 1e-12:
        return out[:2]
    return out[:2] / out[2]


def _rotate_about(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    th = math.radians(angle_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
    T_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return T_pos @ R @ T_neg


def _line_tilt_deg(line: np.ndarray) -> float:
    """Tilt of a homogeneous line in degrees (horizontal = 0)."""
    a, b, _ = line.reshape(3)
    return math.degrees(math.atan2(-a, b))


def _is_convex(quad: np.ndarray) -> bool:
    """Sign of cross product is consistent around the quadrilateral."""
    n = quad.shape[0]
    signs = []
    for i in range(n):
        a = quad[(i + 1) % n] - quad[i]
        b = quad[(i + 2) % n] - quad[(i + 1) % n]
        signs.append(float(a[0] * b[1] - a[1] * b[0]))
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _polygon_area(quad: np.ndarray) -> float:
    """Shoelace area of an ordered 2D polygon."""
    x = quad[:, 0]
    y = quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
