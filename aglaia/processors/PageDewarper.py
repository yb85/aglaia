# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import os
import sys
import cv2
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

# CRITICAL: must be set BEFORE `import jax`. XLA reads these at backend init;
# without `platform` allocator, the worker leaks past 30 GB across distinct
# input shapes (each shape allocates its own unreleased pool buffer).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
# The slim-CUDA Linux AppImage ships only the CUDA libs the L-BFGS-B dewarp
# actually loads (cuBLAS/nvrtc/nvjitlink/cupti/nvcc/cudart) and DROPS the
# ~2.6 GB of dead weight (cuDNN/NCCL/nvshmem/cuFFT/cuSPARSE/cuSOLVER) to fit
# GitHub's 2 GiB release cap. JAX's CUDA plugin version-probes EVERY lib at
# init and, finding one absent, hard-raises and silently falls back to CPU —
# so the GPU bundle would secretly run on CPU. This bypasses that probe; the
# bundled libs are the exact pinned wheels JAX was built against, so the
# version check it skips would always have passed anyway. Source `--extra cuda`
# installs (full CUDA present) are unaffected. setdefault → user-overridable.
os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait
from aglaia.processors import utils
from aglaia.Status import Status

# Global JAX status
HAS_JAX = False
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    pass

def setup_jax():
    if not HAS_JAX:
        return

    # Persistent Cache Setup
    cache_dir = os.path.join(os.getcwd(), ".jax_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    except:
        pass  # Fallback to no-cache if dir creation fails

    # Platform allocator: buffers freed by jax.clear_caches() actually return
    # to the OS. Must be set before backend init.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
    jax.config.update("jax_enable_x64", False)
    # NVIDIA GPUs run float32 matmuls as TF32 (19-bit mantissa) by default.
    # That reduced precision makes the L-BFGS sheet optimiser converge to a
    # different — and visibly worse — dewarp than the CPU backend, so output
    # geometry mismatches across CPU/GPU. Pin highest precision for parity.
    jax.config.update("jax_default_matmul_precision", "highest")
    # JAX Metal bindings incomplete on macOS; CUDA / CPU picked automatically.

# Import page-dewarp library components
try:
    from page_dewarp.options import Config
    from page_dewarp.optimise import optimise_params
    from page_dewarp.spans import assemble_spans, keypoints_from_samples, sample_spans
    from page_dewarp.solve import get_default_params
    from page_dewarp.mask import Mask
    from page_dewarp.contours import get_contours
    from page_dewarp.options import cfg as pd_cfg
    from page_dewarp.dewarp import round_nearest_multiple, norm2pix
    from page_dewarp.keypoints import make_keypoint_index, project_keypoints
    from page_dewarp.normalisation import pix2norm
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False


from aglaia.processors.geometry import (xband_baseline_per_col,
                                     span_bottom_series, fit_span_baseline)


def _span_step_points(span, poly, step: int) -> list[tuple[float, float]]:
    """Sample `poly` (or the legacy per-contour fallback when None) at
    the step grid over every contour of `span`. Pixel coords."""
    pts: list[tuple[float, float]] = []
    for cinfo in span:
        xmin, ymin = cinfo.rect[:2]
        ncols = cinfo.mask.shape[1] if cinfo.mask is not None else 0
        if ncols == 0:
            continue
        start = int(np.floor_divide(np.mod((ncols - 1), step), 2))
        if poly is not None:
            for x in range(start, ncols, step):
                gx = x + xmin
                pts.append((gx, float(poly(gx))))
        else:
            means = xband_baseline_per_col(cinfo.mask)
            for x in range(start, ncols, step):
                m = means[x]
                if not np.isfinite(m):
                    continue  # column was outside the x-height band
                pts.append((x + xmin, float(m) + ymin))
    return pts


def _sample_spans_xband(shape: tuple[int, int],
                        spans: list,
                        baseline_source: str = "bottom") -> list[np.ndarray]:
    """Drop-in replacement for `page_dewarp.spans.sample_spans` feeding
    robust span-level baselines instead of raw per-column ink-mass means.

    Per span: gather every contour's per-column bottom-ink y, fit one
    IRLS-Tukey cubic (`fit_span_baseline`) and sample IT at the step
    grid — descender / dash / apostrophe columns no longer drop
    keypoints, so the baseline reaches the true line ends where page
    curl is strongest. Spans too small for a robust fit fall back to
    the legacy per-contour x-height-band filter
    (`xband_baseline_per_col`), which discards outlier columns.

    `baseline_source` selects which fitted curve(s) feed the model:
      - "bottom"  — baseline only (bottom-ink fit; legacy behaviour).
      - "top"     — x-height topline only (top-ink fit; ascenders /
                    capitals rejected by the robust loss). Falls back to
                    the baseline when the topline fit fails validation.
      - "average" — midline (baseline + topline) / 2: one curve per text
                    line with both descender and ascender noise averaged
                    out. Falls back to the baseline without a topline.
      - "both"    — baseline AND topline as separate model spans —
                    doubles the vertical constraints on the sheet.

    A topline is used only if its fitted curve sits 0.3–2.5 x-heights
    above the baseline along the whole span (rejects capital-heavy lines
    where the fit locks onto cap-height).

    Output: list of `(N, 1, 2)` float32 arrays in [-1, +1] normalised
    coords, ordered top-to-bottom by mean y.
    """
    step = pd_cfg.SPAN_PX_PER_STEP
    entries: list[tuple[float, list[tuple[float, float]]]] = []
    n_top = n_fit_fail = n_sep_low = n_sep_high = 0
    for span in spans:
        xs, ys, hs = span_bottom_series(span)
        bpoly = fit_span_baseline(xs, ys, hs)
        tpoly = None
        if baseline_source != "bottom" and bpoly is not None and xs.size:
            xt, yt, ht = span_bottom_series(span, side="top")
            cand = fit_span_baseline(xt, yt, ht)
            if cand is None:
                n_fit_fail += 1
            else:
                h_med = float(np.median(hs))
                x_chk = np.linspace(float(xs.min()), float(xs.max()), 32)
                sep = bpoly(x_chk) - cand(x_chk)
                if not np.all(sep > 0.3 * h_med):
                    n_sep_low += 1
                elif not np.all(sep < 2.5 * h_med):
                    n_sep_high += 1
                else:
                    tpoly = cand
                    n_top += 1
        if baseline_source == "top":
            polys = [tpoly if tpoly is not None else bpoly]
        elif baseline_source == "average":
            if tpoly is not None:
                bp, tp = bpoly, tpoly
                polys = [lambda gx, bp=bp, tp=tp: 0.5 * (bp(gx) + tp(gx))]
            else:
                polys = [bpoly]
        elif baseline_source == "both":
            polys = [bpoly] + ([tpoly] if tpoly is not None else [])
        else:  # "bottom"
            polys = [bpoly]
        for poly in polys:
            pts = _span_step_points(span, poly, step)
            if pts:
                entries.append((float(np.mean([p[1] for p in pts])), pts))
    if baseline_source != "bottom":
        print(f"[PageDewarper] toplines kept={n_top}/{len(spans)} "
              f"(fit_fail={n_fit_fail} sep_low={n_sep_low} "
              f"sep_high={n_sep_high})", flush=True)
    # page_dewarp's initial-params solve assumes top-to-bottom span order.
    entries.sort(key=lambda e: e[0])
    span_points = []
    for _, pts in entries:
        arr = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
        span_points.append(pix2norm(shape, arr))
    return span_points


def _text_mask_dpi(small_rgb: np.ndarray, pagemask: np.ndarray,
                   analysis_dpi: float, line_join_mm: float,
                   is_bw: bool = False,
                   kernel_char_mult: float = 1.5,
                   large_blob_limit: float = 0.0
                   ) -> tuple[np.ndarray, float]:
    """Build a text mask using mm-sized MORPH_CLOSE.

    Mirrors TrapezoidalCorrection's connectivity strategy: a single horizontal
    kernel for dilate + erode bridges intra-word gaps without growing blob
    bounds. Kernel width is mm × analysis DPI.

    is_bw: already binarised — just invert (text=255). Grayscale uses Otsu
    (bimodal). MEAN_C adaptive thresholding is avoided: it welds
    ascender/descender clusters into 130-px-tall blobs which TEXT_MAX_THICKNESS
    then drops wholesale.
    """
    sgray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    if is_bw:
        mask = cv2.bitwise_not(sgray)
    else:
        _, mask = cv2.threshold(sgray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.minimum(mask, pagemask)

    # Kernel width tracks observed text scale (median char-like CC height).
    # DPI alone is brittle: a wrong-DPI input grows row-wide stripes that
    # vertically weld adjacent lines. CC filter bounds derive from analysis
    # DPI to survive very-low and very-high DPI inputs.
    cc_h_min = max(3, int(round(analysis_dpi * 0.04)))
    cc_h_max = max(cc_h_min + 1, int(round(analysis_dpi * 0.45)))
    cc_w_min = max(2, int(round(analysis_dpi * 0.02)))
    cc_w_max = max(cc_w_min + 1, int(round(analysis_dpi * 0.60)))
    n, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    char_h = [int(s[3]) for s in stats[1:]
              if cc_h_min <= s[3] <= cc_h_max
              and cc_w_min <= s[2] <= cc_w_max]
    if len(char_h) >= 30:
        h_med = float(np.median(char_h))
        kw = max(9, int(round(kernel_char_mult * h_med)))
    else:
        h_med = 0.0
        kw = max(9, int(round(line_join_mm * analysis_dpi / 25.4)))
    # Wipe oversized blobs (table-grid lines / borders) before the morphology
    # + span detection, so they don't get chained into spurious spans. Uses
    # the cheap bounding-box area (w·h) vs (large_blob_limit × char h)².
    if large_blob_limit > 0 and h_med > 0:
        areas = (stats[:, cv2.CC_STAT_WIDTH].astype(np.int64)
                 * stats[:, cv2.CC_STAT_HEIGHT].astype(np.int64))
        thr = (large_blob_limit * h_med) ** 2
        large_ids = np.nonzero(areas > thr)[0]
        large_ids = large_ids[large_ids != 0]  # never the background label
        if large_ids.size:
            mask[np.isin(cc_labels, large_ids)] = 0

    # Break thin vertical bridges before the horizontal close — otherwise
    # adjacent text lines weld into multi-row blobs.
    vbreak = max(3, int(round(h_med / 6))) if h_med > 0 else 3
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vbreak))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return np.minimum(mask, pagemask), h_med

@dataclass
class DewarpOption(AbstractProcessorOption):
    max_oob: float = 400.0
    # Mask outside this margin (mm) when searching for text spans.
    page_margin_mm: float = 5.0
    dewarp_margin: float = 5.0 # margin_mm for padding
    remap_decimate: int = 4
    shear_cost: float = 40.0
    # L2 penalty λ on cubic-sheet slopes (α, β in pvec[6:8]). Default 0.0 (no
    # penalty) to match the OPTIONS spec + pipeline YAMLs — a non-zero λ
    # suppresses *real* curl on genuinely-warped pages. Phantom-curl on flat
    # inputs is handled by the divergence guard + bspline fallback instead.
    cubic_cost: float = 0.0
    focal_length: float = 1.3
    # Backend switch:
    #   auto   – MLX (Apple Metal) → padded JAX → SciPy Powell.
    #   mlx    – force MLX value+grad over JAX padded path; falls back to Powell on install error.
    #   jax    – padded JAX L-BFGS-B.
    #   powell – SciPy Powell only (safest, slowest).
    # Env overrides: AGLAIA_MLX=0 disables MLX, AGLAIA_JAX_PAD=0 disables padded JAX.
    backend: str = "auto"
    processing_dpi: Optional[float] = 150.0
    camera_matrix: Optional[np.ndarray] = None
    camera_matrix_resolution: Optional[tuple] = None
    # Horizontal MORPH_CLOSE kernel for line bridging, in mm on the page.
    # Used as a fallback when text-scale auto-estimation gathers < 30
    # char-like CCs. Pixel size = mm * dpi / 25.4.
    line_join_mm: float = 4.0
    # Adaptive kernel width = kernel_char_mult × median char height.
    # Bridges intra-word + word-gap pixels into full-line spans without
    # widening so much that horizontal dilation creates row-wide stripes
    # that vertically merge adjacent lines. 1.5 (was 2.0): the kernel is sized
    # off the BODY char height, but tight smaller-text blocks (footnotes) then
    # welded into multi-line spans → their curvature was never modelled and
    # the dewarp left them bent. 1.5 separates them; welding returns ≥~1.7.
    kernel_char_mult: float = 1.5
    # Drop connected components whose bounding-box area (w×h) exceeds
    # (large_blob_limit × median char height)² before span detection — strips
    # connected table-grid lines / borders that would otherwise chain into
    # bogus spans. 0 disables.
    large_blob_limit: float = 10.0
    # TEXT_MAX_THICKNESS = thickness_char_mult × median char height.
    # ≈ ascender + x-height + descender. Single text line column-sum
    # max stays under that bound; multi-line bridged blobs exceed it
    # and get dropped at the contour-filter stage.
    thickness_char_mult: float = 3.0
    # EDGE_MAX_LENGTH = edge_max_length_char_mult × median char height.
    # Bounds the Euclidean gap between right-endpoint of cinfo A and
    # left-endpoint of cinfo B that the span-builder will link. 3 ≈
    # one wide justified word gap + safety margin. Lower = stricter
    # spans (more fragments), higher = risk of cross-line linking.
    edge_max_length_char_mult: float = 3.0
    # Refuse to fit on a content block that is wider than it is tall.
    # Below this span count the cubic-sheet fit is under-constrained:
    # the optimizer wanders for OPT_MAX_ITER iterations on what amounts
    # to a noisy 2-3-baseline problem, ballooning MLX/JAX memory and
    # almost always producing worse output than the input. Title pages,
    # half-titles, figure-only spreads, blank versos all land here.
    # 8 spans gives the cubic fit ~5 redundant constraints — enough
    # margin to absorb the few-noisy-baselines case without admitting
    # title-page / figure-only crops that destabilise the optimizer.
    min_spans: int = 4
    # Drop spans whose total width is < this fraction of the widest span
    # in the scan. Partial baselines at page edges (footer + author line
    # + page number — none of which span the full text column) bias the
    # cubic fit toward false curvature; cutting them away leaves only
    # full-text-width lines, which carry the real warp signal.
    # Keep spans at least this fraction as wide as the widest. 0.2 keeps the
    # footnote block (short lines) so the fit's y-support reaches the bottom
    # of the page — at 0.5 footnotes were dropped, leaving them below the
    # support range where the sheet is extrapolated and fans them out
    # (athanase 132 B). The absolute SPAN_MIN_WIDTH floor still drops tiny
    # page-number / running-head fragments. 0.0 disables the filter.
    min_span_width_ratio: float = 0.2
    # Which fitted curve(s) feed the sheet model: bottom (baselines),
    # top (x-height toplines), average (midlines), both (baseline +
    # topline as separate spans — doubles vertical constraints).
    # See _sample_spans_xband. Default "bottom": the bottom-ink baseline
    # fit alone removes binding curl cleanly; "both" over-constrains and
    # leaves residual curve on post-trapezoidal pages.
    baseline_source: str = "bottom"
    # Robust (pseudo-Huber) reprojection loss on/off. When off, plain
    # L2 regardless of huber_delta.
    use_huber: bool = True
    # Pseudo-Huber transition scale for the keypoint reprojection loss,
    # in pix2norm units (2/max(h,w) per analysis px → 0.005 ≈ 3-8 px).
    # Bounds the pull of any stray span that survived the width filter
    # (footers, captions). MLX / padded-JAX backends only; Powell keeps
    # the library's L2 (cylindrical) or folds it in (twist models).
    huber_delta: float = 0.005
    # Sheet surface model:
    #   cylindrical   — stock page-dewarp cubic z(x) (every horizontal
    #                   slice has the same height profile).
    #   sine_twist    — Fourier-sine height profile (spline_modes DOF) +
    #                   linear-in-y twist gain γ. Captures non-cubic
    #                   gutter walls and curl that varies top-to-bottom.
    #                   (Legacy name "spline_twist" still accepted.)
    #   bspline_twist — clamped cubic B-spline profile (spline_modes free
    #                   control points) + twist gain γ. Local support:
    #                   a steep gutter wall doesn't ripple into the flat
    #                   field. See aglaia/processors/sheet_models.py.
    #   flat_spline   — bspline_twist specialised for post-trapezoidal
    #                   pages (flat sheet + curl at the binding only):
    #                   knots graded toward the binding edge (coarse
    #                   spline on the flat field, fine at the gutter) +
    #                   an outer-flatness penalty pulling far-from-
    #                   binding control points to z = 0. Binding side
    #                   resolved per page from the PageDetector A/B
    #                   page_side meta (binding_side: auto).
    sheet_model: str = "cylindrical"
    # flat_spline only — which page edge carries the binding/gutter.
    # auto = read the PageDetector page_side meta (left page → binding
    # right, right page → binding left); explicit left/right for single-
    # page scans. Unresolvable auto → flat features degrade to plain
    # bspline behaviour for that page (no flip, no penalty).
    binding_side: str = "auto"
    # flat_spline knot grading g ≥ 1: interior knots at 1 − (1 − u)^g
    # toward the binding. 1 = uniform (bspline_twist resolution).
    knot_grading: float = 2.5
    # flat_spline outer-flatness penalty λ: adds λ·Σ w_i·c_i² with
    # w_i = (squared) Greville distance from the binding. 0 = off.
    # Reprojection error on a typical page sums to ~0.01-0.02 (pix2norm²,
    # Huber'd), c_i ~ 0.1 → λ ≈ 1 makes a phantom outer bump cost about
    # as much as the whole data term.
    flat_outer_penalty: float = 1.0
    # Shape DOF K: sine modes (sine_twist) or free control points
    # (bspline_twist). Model params = K + 1.
    spline_modes: int = 4
    # Twist term (1 + γ·η) on/off. Off → γ baked to 0 in the objective
    # (pure generalised cylinder with the chosen basis); the γ slot stays
    # in the pvec tail so layout/replay are unchanged. Default OFF: on
    # near-flat pages a free γ rides the phantom-curl valley (span-y +
    # pose compensation under the Huber floor) and inverts the grid
    # (observed p008/p010); enable only for genuine open-book fan pages.
    twist: bool = False

    # UI-surfacing flag (no runtime effect): pre-ticks the
    # "Normalize character width" checkbox to surface BlobNormalizer reliance.
    norm_width: bool = False


from aglaia.processors.option_specs import _b, _e, _f, _i


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


@dataclass
class _DewarpCtx:
    """Everything the SOLVE + finish stages need, produced by
    _build_dewarp_problem. Lives within a single process() call (not pickled)."""
    img: Any
    small: Any
    pad_px: int
    is_bw: bool
    input_dpi: float
    char_h_frac: float
    orig_buffer: Any
    orig_roi: Any
    corners: Any
    rough_dims: Any
    span_counts: Any
    params: Any
    model_dims: Any
    support_x: Any
    support_y: Any
    support_decay: float
    flat_flip: bool
    flat_penalty_eff: float
    n_extra: int
    dstpoints: Any
    spans: Any
    span_points: Any
    # Filled in by process() after the solve, only for the spline RMS log.
    params_initial: Any = None


class PageDewarper(AbstractImageProcessor):
    name: str = "PageDewarper"
    SUMMARY = "Cubic-sheet dewarp via page-dewarp + JAX."
    REPLAY_TRAIT = ReplayTrait.COORDINATE  # nonlinear sheet remap
    OPTION_CLASS = DewarpOption
    PROVIDES_META = {
        "char_h_frac": "median glyph height as a fraction of page height "
                       "(dimensionless text scale; absent if < 30 chars found)",
        "roi": "page quad polygon [[x,y],...] in output coords",
        "success": "bool — whether the dewarp remap succeeded",
    }
    _ESSENTIAL_PARAMS = ("sheet_model", "backend", "focal_length", "twist")
    # jax.clear_caches() cadence: per-process counter, flush every N
    # dewarps. Per-image clears defeat the padded fixed-shape compile
    # cache (issue #39); N=10 keeps residual XLA pool growth well under
    # the 3 GB watchdog cap.
    JAX_CLEAR_EVERY = 10
    _dewarps_since_clear = 0
    OPTIONS = {
        "backend": _e("auto", ["auto", "mlx", "jax", "powell"],
                      "Optimizer backend. auto = MLX → padded JAX → Powell. "
                      "WARNING: jax has unresolved issue on Apple hardware "
                      "(memory cache fills up); prefer MLX."),
        "max_oob": _f(500.0, 0.0, 5000.0, 10.0,
                      "Reject the dewarp if remap goes more than this many "
                      "pixels out of bounds.", advanced=True),
        "min_spans": _i(4, 1, 50,
                        "Minimum span count to run the cubic fit. Below this, "
                        "the optimizer is under-constrained and passthrough is safer.",
                        advanced=True),
        "min_span_width_ratio": _f(0.2, 0.0, 1.0, 0.05,
                                   "Drop spans whose width is less than this fraction "
                                   "of the widest span in the scan. 0 = disabled.",
                                   advanced=True),
        "baseline_source": _e("bottom", ["bottom", "top", "average", "both"],
                              "Which fitted text-line curve(s) constrain the sheet: "
                              "bottom = baselines, top = x-height toplines, "
                              "average = midlines, both = baseline + topline as "
                              "separate spans (doubles vertical constraints)."),
        "use_huber": _b(True,
                        "Robust pseudo-Huber reprojection loss — caps the pull "
                        "of stray spans (footers, captions). Off = plain L2."),
        "huber_delta": _f(0.005, 0.001, 0.05, 0.001,
                          "Pseudo-Huber transition scale (normalized units).",
                          advanced=True,
                          visible_when={"use_huber": [True]}),
        "sheet_model": _e("cylindrical",
                          ["cylindrical", "sine_twist", "bspline_twist",
                           "flat_spline"],
                          "Page surface model. cylindrical = stock cubic z(x). "
                          "sine_twist = sine-basis profile + linear-in-y twist — "
                          "handles non-cubic gutter walls and top-to-bottom curl "
                          "variation. bspline_twist = clamped cubic B-spline "
                          "profile + twist — local support, sharp gutter wall "
                          "without ripple in the flat field. flat_spline = "
                          "B-spline for post-trapezoidal pages: flat sheet + "
                          "binding curl only — graded knots toward the binding "
                          "and an outer-flatness penalty (binding side from "
                          "the PageDetector A/B page meta)."),
        "spline_modes": _i(4, 2, 12,
                           "Shape DOF K: sine modes (sine_twist) or free "
                           "control points (bspline_twist / flat_spline); "
                           "model adds K+1 params.", advanced=True,
                           visible_when={"sheet_model": ["sine_twist",
                                                         "bspline_twist",
                                                         "flat_spline"]}),
        "twist": _b(False,
                    "Fit the linear-in-y twist gain γ (curl amplitude "
                    "varying top-to-bottom). Off = pure cylinder with the "
                    "chosen basis (γ pinned to 0). Enable only for true "
                    "open-book fan pages — on flat pages a free γ invents "
                    "phantom twist.",
                    visible_when={"sheet_model": ["sine_twist",
                                                  "bspline_twist",
                                                  "flat_spline"]}),
        "binding_side": _e("auto", ["auto", "left", "right"],
                           "flat_spline: page edge carrying the binding. "
                           "auto reads the PageDetector page_side meta "
                           "(left page of a spread → binding right, and "
                           "vice versa); set explicitly for single-page "
                           "scans. Unresolved auto disables the flat "
                           "features for that page.",
                           visible_when={"sheet_model": ["flat_spline"]}),
        "knot_grading": _f(2.5, 1.0, 6.0, 0.5,
                           "flat_spline: knot density grading toward the "
                           "binding. 1 = uniform; higher = coarser flat "
                           "field / finer gutter wall.", advanced=True,
                           visible_when={"sheet_model": ["flat_spline"]}),
        "flat_outer_penalty": _f(1.0, 0.0, 100.0, 0.5,
                                 "flat_spline: weight of the outer-flatness "
                                 "penalty λ·Σ w_i·c_i² (w = squared Greville "
                                 "distance from the binding). 0 = off; ~1 "
                                 "balances the data term on a typical page.",
                                 visible_when={"sheet_model": ["flat_spline"]}),
        "page_margin_mm": _f(5.0, 0.0, 50.0, 1.0,
                             "Mask margin in mm — defines the page-extent rectangle "
                             "for span detection."),
        "dewarp_margin": _f(5.0, 0.0, 50.0, 1.0,
                            "Pre-dewarp white padding around the crop (mm)."),
        "remap_decimate": _i(4, 1, 64,
                             "Decimation factor for the remap grid. Lower = sharper but slower.",
                             advanced=True),
        "shear_cost": _f(40.0, 0.0, 1000.0, 1.0,
                         "Penalty on rvec[0]² (camera tilt).", advanced=True),
        "cubic_cost": _f(0.0, 0.0, 100.0, 0.1,
                         "L2 penalty on cubic slopes α, β. 0 = no penalty.",
                         advanced=True),
        "focal_length": _f(1.3, 0.5, 5.0, 0.1,
                           "Normalized camera focal length. Overridden by calibration if present.",
                           advanced=True),
        "line_join_mm": _f(4.0, 0.5, 20.0, 0.5,
                           "Horizontal MORPH_CLOSE kernel fallback (mm at analysis DPI). "
                           "Used when <30 char-like CCs available.", advanced=True),
        "kernel_char_mult": _f(1.5, 0.5, 6.0, 0.1,
                               "Adaptive MORPH_CLOSE kernel width = mult × median char height. "
                               "Bridges words into full-line spans."),
        "large_blob_limit": _f(10.0, 0.0, 40.0, 1.0,
                               "Drop connected components whose bounding-box area "
                               "exceeds (mult × median char height)² before span "
                               "detection — removes table-grid lines. Use 0 to disable."),
        "thickness_char_mult": _f(3.0, 1.0, 8.0, 0.1,
                                  "TEXT_MAX_THICKNESS = mult × median char height "
                                  "(3× ≈ ascender + x-height + descender + margin)."),
        "edge_max_length_char_mult": _f(3.0, 0.5, 12.0, 0.1,
                                        "EDGE_MAX_LENGTH = mult × median char height. "
                                        "Max gap between atoms the span-builder will link."),
        "processing_dpi": _f(150.0, 36.0, 300.0, 10.0,
                             "Downsample to this DPI for span analysis; full-res for the remap."),
        "norm_width": _b(False,
                         "Surface the 'Normalize character width' checkbox in the GUI (pre-ticked).",
                         advanced=True),
    }

    @classmethod
    def inject_step_options(cls, step_opts: dict, args) -> dict:
        """Inject calibration's camera_matrix only when real. Identity /
        placeholder matrices collapse focal_length to ~0.001 → Powell wanders → leak."""
        from aglaia.workers.Initializer import _is_real_calibration  # local import
        K = args.options.get("calibration", {}).get("camera_matrix")
        res = args.options.get("calibration", {}).get("camera_matrix_resolution")
        if _is_real_calibration(K):
            step_opts["camera_matrix"] = K
            step_opts["camera_matrix_resolution"] = res
        else:
            step_opts["camera_matrix"] = None
            step_opts["camera_matrix_resolution"] = None
        return step_opts

    @classmethod
    def replay_transform(cls, params, in_wh):
        """Nonlinear sheet remap → a backward sampling map. The map is
        analytic in the fitted sheet params, so the engine folds any upstream
        affine into its source coords for a single interpolation."""
        from aglaia.processors.replay_transform import SampleMapTransform
        return SampleMapTransform(lambda in_hw: cls._replay_sample_map(in_hw, params))

    @staticmethod
    def _sample_grid(ref_h, ref_w, *, params, page_dims_w, page_dims_h,
                     model_dims, decimate, zoom, model, n_modes, focal,
                     support, support_y, support_decay, grading, flip):
        """The arc-length-uniform backward sampling grid, shared by the
        forward remap and by replay so both build pixel-identical geometry.

        Returns ``(image_points, grid_shape, target_w, target_h, w_small,
        h_small)`` — the projected sample points (still at decimated grid
        resolution) plus the output sizing. Each caller does its own
        reshape/resize tail (their float dtypes differ at the LSB). ``ref_h,
        ref_w`` is the (padded) reference size for sizing + norm→pixel."""
        from page_dewarp.dewarp import norm2pix, round_nearest_multiple

        from aglaia.processors.sheet_models import arclength_x, project_xy_model
        target_h = round_nearest_multiple(0.5 * page_dims_h * zoom * ref_h, decimate)
        # Arc-length-uniform x grid: width sized from the sheet's arc length,
        # x samples spaced uniformly in s (else the steep gutter side comes
        # out horizontally stretched by sqrt(1+z'^2)).
        arc_xs, arc_s = arclength_x(params, page_dims_w, model=model,
                                    n_modes=n_modes, model_dims=model_dims,
                                    support=support, support_decay=support_decay,
                                    grading=grading, flip=flip)
        arc_total = float(arc_s[-1])
        target_w = round_nearest_multiple(
            target_h * arc_total / page_dims_h, decimate)
        h_small, w_small = int(target_h / decimate), int(target_w / decimate)
        page_x = np.interp(np.linspace(0.0, arc_total, w_small), arc_s, arc_xs)
        page_y = np.linspace(0, page_dims_h, h_small)
        gx, gy = np.meshgrid(page_x, page_y)
        page_xy = np.hstack((gx.flatten().reshape((-1, 1)),
                             gy.flatten().reshape((-1, 1)))).astype(np.float32)
        image_points = project_xy_model(
            page_xy, params.astype(np.float32), model=model, n_modes=n_modes,
            model_dims=model_dims, focal_length=focal, support=support,
            support_y=support_y, support_decay=support_decay,
            grading=grading, flip=flip)
        image_points = norm2pix((ref_h, ref_w), image_points, False)
        return image_points, gx.shape, target_w, target_h, w_small, h_small

    @staticmethod
    def _replay_sample_map(in_hw, params):
        """Backward sampling map for a stamped dewarp step → ``(im_x, im_y,
        pad_px)``, sampling the input padded by ``pad_px`` white px on every
        side. Thin wrapper over the shared ``_sample_grid``."""
        pad_px = int(params["pad_px"])
        in_h, in_w = int(in_hw[0]), int(in_hw[1])
        page_dims = np.array(params["page_dims"], dtype=np.float32)
        image_points, shp, target_w, target_h, _, _ = PageDewarper._sample_grid(
            in_h + 2 * pad_px, in_w + 2 * pad_px,
            params=np.array(params["params"], dtype=np.float32),
            # page_dims_w float()-cast (arclength), page_dims_h kept raw —
            # mirrors the pre-extraction dtype mix exactly (byte-identical).
            page_dims_w=float(page_dims[0]), page_dims_h=page_dims[1],
            model_dims=params["model_dims"], decimate=int(params["decimate"]),
            zoom=float(params["zoom"]), model=str(params["sheet_model"]),
            n_modes=int(params["spline_modes"]), focal=float(params["focal_length"]),
            support=params["support_x"], support_y=params["support_y"],
            support_decay=params["support_decay"],
            grading=float(params["knot_grading"]), flip=bool(params["binding_flip"]))
        im_x = image_points[:, 0, 0].reshape(shp)
        im_y = image_points[:, 0, 1].reshape(shp)
        im_x = cv2.resize(im_x, (target_w, target_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        im_y = cv2.resize(im_y, (target_w, target_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return im_x, im_y, pad_px

    def __init__(self, options: DewarpOption):
        super().__init__(options)

        if not HAS_LIBRARY:
            raise ImportError("page-dewarp library not found. Install with: uv pip install 'page-dewarp[jax]'")

        # Sheet model + loss toggles — resolved before backend install so
        # the backend modules get the right baked constants. canonical_model
        # maps the legacy "spline_twist" name to "sine_twist".
        from aglaia.processors.sheet_models import (canonical_model,
                                                 SPLINE_MODELS)
        self.sheet_model = canonical_model(
            getattr(options, "sheet_model", "cylindrical"))
        if self.sheet_model not in ("cylindrical",) + SPLINE_MODELS:
            print(f"[PageDewarper] unknown sheet_model={self.sheet_model!r}; "
                  f"falling back to 'cylindrical'.", flush=True)
            self.sheet_model = "cylindrical"
        self.spline_modes = int(getattr(options, "spline_modes", 4))
        self.twist = bool(getattr(options, "twist", False))
        # flat_spline knobs — inert (grading 1, penalty 0) on other models.
        from aglaia.processors.sheet_models import MODEL_FLAT_SPLINE
        is_flat = self.sheet_model == MODEL_FLAT_SPLINE
        self.binding_side = str(getattr(options, "binding_side", "auto")
                                or "auto").lower()
        if self.binding_side not in ("auto", "left", "right"):
            print(f"[PageDewarper] unknown binding_side="
                  f"{self.binding_side!r}; falling back to 'auto'.",
                  flush=True)
            self.binding_side = "auto"
        self.knot_grading = (max(float(getattr(options, "knot_grading", 2.5)),
                                 1.0) if is_flat else 1.0)
        self.flat_outer_penalty = (max(float(getattr(
            options, "flat_outer_penalty", 1.0)), 0.0) if is_flat else 0.0)
        self.use_huber = bool(getattr(options, "use_huber", True))
        self.huber_delta = (float(getattr(options, "huber_delta", 0.005))
                            if self.use_huber else 0.0)
        self.baseline_source = str(getattr(options, "baseline_source",
                                           "bottom") or "bottom").lower()
        if self.baseline_source not in ("bottom", "top", "average", "both"):
            print(f"[PageDewarper] unknown baseline_source="
                  f"{self.baseline_source!r}; falling back to 'bottom'.",
                  flush=True)
            self.baseline_source = "bottom"

        # Resolve backend (auto / mlx / jax / powell). Env overrides:
        # AGLAIA_MLX=0 disables MLX, AGLAIA_JAX_PAD=0 disables padded JAX.
        requested = str(getattr(options, "backend", "auto") or "auto").lower()
        if requested not in ("auto", "mlx", "jax", "powell"):
            print(f"[PageDewarper] unknown backend={requested!r}; "
                  f"falling back to 'auto'.", flush=True)
            requested = "auto"

        env_mlx = os.environ.get("AGLAIA_MLX")
        env_pad = os.environ.get("AGLAIA_JAX_PAD")
        mlx_allowed = (env_mlx is None
                       or env_mlx.lower() not in ("0", "false", "no", ""))
        pad_allowed = (env_pad is None
                       or env_pad.lower() not in ("0", "false", "no", ""))

        # Resolve to one of: "mlx" | "jax" | "powell".
        active = "powell"
        if requested == "powell" or not HAS_JAX:
            active = "powell"
        else:
            # Both auto and mlx try MLX first; jax explicitly skips it.
            try_mlx = requested in ("auto", "mlx") and mlx_allowed
            setup_jax()
            if try_mlx:
                try:
                    from aglaia.processors.page_dewarp_mlx import (
                        install as _install_mlx,
                        set_cubic_cost as _set_mlx_cubic,
                        set_huber_delta as _set_mlx_huber,
                        set_knot_grading as _set_mlx_grading,
                        set_sheet_model as _set_mlx_model,
                    )
                    if _install_mlx():
                        active = "mlx"
                        _set_mlx_cubic(float(getattr(options, "cubic_cost", 0.0)))
                        _set_mlx_huber(self.huber_delta)
                        _set_mlx_model(self.sheet_model, self.spline_modes,
                                       self.twist)
                        _set_mlx_grading(self.knot_grading)
                        print("[PageDewarper] MLX backend active "
                              "(bypasses JAX-CPU host-pool leak).",
                              flush=True)
                    elif requested == "mlx":
                        print("[PageDewarper] MLX install returned False; "
                              "falling back to padded JAX.", flush=True)
                except Exception as e:
                    if requested == "mlx":
                        print(f"[PageDewarper] MLX install failed: {e}; "
                              "falling back to padded JAX.", flush=True)
            if active != "mlx":
                if pad_allowed:
                    try:
                        from aglaia.processors.page_dewarp_padded import (
                            install as _install_pad,
                            set_cubic_cost as _set_pad_cubic,
                            set_huber_delta as _set_pad_huber,
                            set_knot_grading as _set_pad_grading,
                            set_sheet_model as _set_pad_model,
                        )
                        _install_pad()
                        _set_pad_cubic(float(getattr(options, "cubic_cost", 0.0)))
                        _set_pad_huber(self.huber_delta)
                        _set_pad_model(self.sheet_model, self.spline_modes,
                                       self.twist)
                        _set_pad_grading(self.knot_grading)
                        active = "jax"
                    except Exception as e:
                        print(f"[PageDewarper] padded JAX install failed: "
                              f"{e}; falling back to Powell.", flush=True)
                        active = "powell"
                else:
                    active = "jax"  # raw, unpadded JAX
                    if self.sheet_model in SPLINE_MODELS:
                        # Library's raw JAX objective is cubic-only — the
                        # model tail would get zero gradient and the
                        # remap would then project a flat sheet.
                        print(f"[PageDewarper] {self.sheet_model} needs MLX "
                              "/ padded JAX / powell; raw JAX is cubic-only. "
                              "Falling back to cylindrical.", flush=True)
                        self.sheet_model = "cylindrical"
        self.backend = active
        self.use_jax = active in ("mlx", "jax")
        self.uses_gpu = False
        if self.use_jax:
            try:
                self.uses_gpu = jax.default_backend() != "cpu"
            except Exception:
                self.uses_gpu = False
        
        self.max_oob = options.max_oob
        self.page_margin_mm = options.page_margin_mm
        self.dewarp_margin = options.dewarp_margin
        self.processing_dpi = options.processing_dpi
        self.line_join_mm = options.line_join_mm
        self.kernel_char_mult = float(getattr(options, "kernel_char_mult", 1.5))
        self.large_blob_limit = float(getattr(options, "large_blob_limit", 10.0))
        self.thickness_char_mult = float(getattr(options, "thickness_char_mult", 3.0))
        self.edge_max_length_char_mult = float(
            getattr(options, "edge_max_length_char_mult", 3.0))
        self.min_spans = int(getattr(options, "min_spans", 4))
        self.min_span_width_ratio = float(
            getattr(options, "min_span_width_ratio", 0.0))
        self.cubic_cost = float(getattr(options, "cubic_cost", 0.0))

        # page-dewarp Config: silence library prints; output sizing is driven
        # by the per-scan remap math below, not the library canvas.
        self.cfg = Config()
        self.cfg.DEBUG_LEVEL = 0
        self.cfg.OUTPUT_DPI = 300
        self.cfg.REMAP_DECIMATE = options.remap_decimate
        self.cfg.SHEAR_COST = options.shear_cost
        
        # Powell = SciPy fallback; JAX L-BFGS-B only when use_jax.
        if self.use_jax:
            self.cfg.OPT_METHOD = "L-BFGS-B"
            try:
                self.cfg.DEVICE = jax.default_backend()
            except Exception:
                self.cfg.DEVICE = "cpu"
        else:
            self.cfg.OPT_METHOD = "Powell"
        # Cap optimizer iterations — 8-param cubic sheet converges <100 iters;
        # cap only bites on a degenerate objective. (Library default: 600_000.)
        self.cfg.OPT_MAX_ITER = 2000

        # Handle Camera calibration
        camera_matrix = options.camera_matrix
        camera_matrix_resolution = options.camera_matrix_resolution

        if camera_matrix is not None and camera_matrix_resolution is not None:
             h, w = camera_matrix_resolution
             scl = 2.0 / max(h, w)
             self.focal_length = float((camera_matrix[0,0] + camera_matrix[1,1]) / 2.0 * scl)
             self.cfg.FOCAL_LENGTH = self.focal_length
             self.K = camera_matrix.copy()
             self.K[0,0] *= scl
             self.K[1,1] *= scl
             self.K[0,2] = (self.K[0,2] - w*0.5) * scl
             self.K[1,2] = (self.K[1,2] - h*0.5) * scl
        else:
             self.focal_length = options.focal_length
             self.cfg.FOCAL_LENGTH = self.focal_length
             self.K = np.array([
                [self.focal_length, 0, 0],
                [0, self.focal_length, 0],
                [0, 0, 1]], dtype=np.float32)

    def resize_to_analysis(self, src, current_dpi, is_bw: bool = False):
        # INTER_NEAREST on BW preserves the 0/255 histogram — INTER_AREA would
        # introduce gray values that MORPH_CLOSE bridges into welded lines.
        interp = cv2.INTER_NEAREST if is_bw else cv2.INTER_AREA
        if self.processing_dpi and self.processing_dpi < current_dpi:
            scale = self.processing_dpi / current_dpi
            new_w = int(src.shape[1] * scale)
            new_h = int(src.shape[0] * scale)
            return cv2.resize(src, (new_w, new_h), interpolation=interp)
        return src

    # Page-dims sanity bound. Normalised page dims are O(1–2); a fit that
    # diverges (e.g. cubic_cost=0 cylindrical on a strongly-curved page)
    # blows this up to 1e6+ and then crashes the arc-length remap.
    _PAGE_DIM_SANE = 50.0

    def _set_backend_sheet_model(self) -> None:
        """Re-push the current sheet_model/twist to the active backend's
        module-global config (used when falling back mid-run)."""
        try:
            if self.backend == "mlx":
                from aglaia.processors.page_dewarp_mlx import set_sheet_model as _s
                _s(self.sheet_model, self.spline_modes, self.twist)
            elif self.backend == "jax":
                from aglaia.processors.page_dewarp_padded import set_sheet_model as _s
                _s(self.sheet_model, self.spline_modes, self.twist)
        except Exception:
            pass

    def _fallback_to_bspline(self, img_buf, orig_buffer, orig_roi):
        """Cylindrical fit diverged — retry this page with bspline_twist
        (no twist), which is numerically stable. Restores the pre-padding
        buffer, swaps the model for THIS call only (saved/restored so the
        next page still starts cylindrical), and re-runs process. Returns
        the recovered buffer, or None when no fallback is possible (already
        on a spline model → caller does a safe near-identity passthrough)."""
        if self.sheet_model != "cylindrical":
            return None
        saved_model, saved_twist = self.sheet_model, self.twist
        print("[PageDewarper] cylindrical fit diverged — falling back to "
              "bspline_twist (no twist) for this page.", flush=True)
        self.sheet_model = "bspline_twist"
        self.twist = False
        self._set_backend_sheet_model()
        img_buf.buffer = orig_buffer
        if orig_roi is not None:
            img_buf.meta["roi"] = orig_roi
        try:
            return self.process(img_buf)
        finally:
            self.sheet_model, self.twist = saved_model, saved_twist
            self._set_backend_sheet_model()

    def process(self, img_buf: ImageBuffer) -> ImageBuffer:
        """
        Dewarp using JAX-accelerated cubic sheet model.
        Modifies img_buf in-place.

        Structured as build -> solve (inline) -> finish. Geometry/IO is
        byte-identical to the pre-split version (golden-hash verified); only
        the code layout changed so the optimiser SOLVE is isolated.
        """
        # Unified debug dumps go through self.debug_save (see AbstractImageProcessor).
        ctx, early_buf = self._build_dewarp_problem(img_buf)
        if early_buf is not None:
            return early_buf

        from aglaia.processors import sheet_models

        small = ctx.small
        span_counts = ctx.span_counts
        model_dims = ctx.model_dims
        flat_flip = ctx.flat_flip
        flat_penalty_eff = ctx.flat_penalty_eff
        dstpoints = ctx.dstpoints
        params = ctx.params

        # DEBUG: Spans Overlay
        if self.debug_enabled():
            dbg_0 = small.copy()
            overlay = np.zeros_like(small)
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (200,100,0), (0,100,200)]
            for i, span in enumerate(ctx.spans):
                color = colors[i % len(colors)]
                for cinfo in span:
                    cv2.drawContours(overlay, [cinfo.contour], -1, color, -1)
            cv2.addWeighted(overlay, 0.4, dbg_0, 0.6, 0, dbg_0)
            # Every fitted model line (with baseline_source="both" a
            # span contributes baseline AND topline as separate
            # entries) — span_points are pix2norm'd, map back.
            for j, sp in enumerate(ctx.span_points):
                pts_px = norm2pix(small.shape, sp, False)
                pts_px = pts_px.reshape(-1, 2).astype(np.int32)
                cv2.polylines(dbg_0, [pts_px], False,
                              colors[j % len(colors)], 2)
            self.debug_save(dbg_0, "0_spans", img_buf)

            # Debug Points Helper
            def draw_points(img, pvec):
                ki = make_keypoint_index(span_counts)
                ppts = sheet_models.project_keypoints_model(
                    np.asarray(pvec, dtype=np.float64).copy(), ki,
                    model=self.sheet_model, n_modes=self.spline_modes,
                    model_dims=model_dims,
                    focal_length=self.focal_length,
                    grading=self.knot_grading, flip=flat_flip)
                res = img.copy()
                for pt in ppts:
                    cv2.circle(res, (int(pt[0,0]), int(pt[0,1])), 3, (255, 0, 0), -1)
                for pt in dstpoints:
                    cv2.circle(res, (int(pt[0,0]), int(pt[0,1])), 2, (0, 0, 255), -1)
                return res

            res_1 = np.hstack([draw_points(small, params), small])
            self.debug_save(res_1, "1_initial", img_buf)

        # 4. JAX Optimization
        params_initial = params.copy()

        try:
            with SuppressOutput():
                if (self.sheet_model in sheet_models.SPLINE_MODELS
                        and self.backend == "powell"):
                    # Library Powell objective is cubic-only — use
                    # the vendored model-aware optimiser.
                    params = sheet_models.optimise_params_spline_powell(
                        dstpoints, span_counts, params,
                        model=self.sheet_model,
                        n_modes=self.spline_modes,
                        twist=self.twist,
                        model_dims=model_dims,
                        focal_length=self.focal_length,
                        shear_cost=float(self.cfg.SHEAR_COST),
                        cubic_cost=self.cubic_cost,
                        huber_delta=self.huber_delta,
                        grading=self.knot_grading,
                        flip=flat_flip,
                        flat_penalty=flat_penalty_eff,
                        maxiter=int(self.cfg.OPT_MAX_ITER))
                else:
                    params = optimise_params("dewarp", small, dstpoints, span_counts, params, self.cfg.DEBUG_LEVEL)
        finally:
            # Drop per-shape compiled programs every N dewarps. The
            # padded backend always traces on one fixed shape, so a
            # per-image clear would force a pointless re-trace/compile
            # (~0.3-1 s) on every page; the residual XLA pool growth
            # only needs an occasional flush to stay under the 3 GB
            # watchdog cap.
            if self.use_jax:
                PageDewarper._dewarps_since_clear += 1
                if PageDewarper._dewarps_since_clear >= PageDewarper.JAX_CLEAR_EVERY:
                    PageDewarper._dewarps_since_clear = 0
                    try:
                        jax.clear_caches()
                    except Exception:
                        pass
            # MLX clings to its allocator pool across calls (unified
            # memory). After ~10 dewarps the per-worker phys_footprint
            # hits the 3 GB watchdog cap → SIGKILL → scan dropped.
            # clear_pool() frees the Metal buffers (the actual leak)
            # but keeps the compiled value_and_grad closure so the
            # next page skips re-tracing.
            if self.backend == "mlx":
                try:
                    from aglaia.processors.page_dewarp_mlx import (
                        clear_pool as _mlx_clear_pool,
                    )
                    _mlx_clear_pool()
                except Exception:
                    pass
                # Optional JAX device-memory profile dump (env-gated).
                # View with `pprof --web <file.prof>`.
                _prof_dir = os.environ.get("AGLAIA_JAX_PROFILE_DIR")
                if _prof_dir:
                    try:
                        import time as _t
                        os.makedirs(_prof_dir, exist_ok=True)
                        _path = os.path.join(
                            _prof_dir,
                            f"dewarp_pid{os.getpid()}_{int(_t.time()*1000)}.prof",
                        )
                        jax.profiler.save_device_memory_profile(_path)
                    except Exception as e:
                        print(f"[PageDewarper] jax memory profile dump failed: {e}",
                              flush=True)

        # Polish pass for Powell only — MLX/padded-JAX fold cubic_cost into
        # their main pass. For Powell we re-optimise the 8 global params
        # (rvec+tvec+α+β); per-span y/x stay frozen at lib's optimum.
        # Cylindrical only: the spline Powell path folds its own reg.
        if (self.cubic_cost > 0.0 and self.backend == "powell"
                and self.sheet_model == "cylindrical"):
            from scipy.optimize import minimize
            kpidx = make_keypoint_index(span_counts)
            target_pts = dstpoints.reshape((-1, 2))
            shear_w = float(self.cfg.SHEAR_COST)
            lam = float(self.cubic_cost)
            full = np.asarray(params, dtype=np.float64).copy()
            head0 = full[:8].copy()

            def _cubic_obj(head):
                full[:8] = head
                ppts = project_keypoints(full, kpidx).reshape((-1, 2))
                err = float(np.sum((ppts - target_pts) ** 2))
                err += shear_w * float(head[0]) ** 2
                err += lam * (float(head[6]) ** 2 + float(head[7]) ** 2)
                return err

            try:
                with SuppressOutput():
                    polish = minimize(
                        _cubic_obj, head0, method="Powell",
                        options={"maxiter": 2000,
                                 "xtol": 1e-6, "ftol": 1e-7},
                    )
                full[:8] = polish.x
                params = full.astype(np.float32)
            except Exception as e:
                print(f"[PageDewarper] cubic polish failed: {e}; "
                      "keeping lib params.", flush=True)

        if self.debug_enabled():
            res_2 = np.hstack([draw_points(small, params_initial), draw_points(small, params)])
            self.debug_save(res_2, "2_optimized", img_buf)

        ctx.params_initial = params_initial
        return self._finish_dewarp(img_buf, params, ctx)

    def _build_dewarp_problem(self, img_buf: ImageBuffer):
        """Build the dewarp optimisation problem: padding, page mask, text
        mask, span assembly/filtering, default params, support ranges,
        flat-spline state and dstpoints — plus the lib_cfg / spline-backend
        side effects. Returns ``(ctx, None)`` to proceed, or ``(None,
        early_buf)`` for the passthrough / gray-fallback short circuits that
        process() returns verbatim."""
        # Snapshot the pre-padding buffer + ROI so a diverged cylindrical fit
        # can be retried cleanly with the stable bspline model (see
        # `_fallback_to_bspline`). copyMakeBorder below returns a NEW array,
        # so this reference stays valid.
        _orig_buffer = img_buf.buffer
        _orig_roi = img_buf.meta.get("roi")

        # 0. Padding
        input_dpi = img_buf.dpi
        pad_px = int(math.ceil(input_dpi * (self.dewarp_margin / 25.4)))
        img_buf.buffer = cv2.copyMakeBorder(
            img_buf.buffer,
            pad_px, pad_px, pad_px, pad_px,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        # Shift upstream ROI into padded coords — fallback paths return the
        # padded buffer as-is and would otherwise carry a misaligned ROI.
        if (roi := img_buf.meta.get("roi")):
            shifted = [[float(x) + pad_px, float(y) + pad_px] for x, y in roi]
            img_buf.meta["roi"] = shifted

        is_bw = img_buf.type == ImageType.BW
        img = img_buf.buffer
        # Downsample first, convert the small result to RGB: to_rgb() on a
        # 12 MP gray frame would allocate a 36 MB full-res copy only to be
        # resized away (analysis-only path; output buffer untouched).
        small = self.resize_to_analysis(img, input_dpi, is_bw=is_bw)
        small = utils.to_rgb(small)
        # 1. Page Extents/Mask
        h, w = small.shape[:2]
        pagemask = np.zeros((h, w), dtype=np.uint8)

        # Calculate margin relative to active resolution
        analysis_dpi = input_dpi * (w / img.shape[1])
        mx = int(self.page_margin_mm * (analysis_dpi / 25.4) / 2)
        my = mx

        cv2.rectangle(pagemask, (mx, my), (w-mx, h-my), 255, -1)
        page_outline = np.array([[mx, my], [mx, h-my], [w-mx, h-my], [w-mx, my]])

        # 2. Extract Contours and Spans (DPI-aware text mask).
        text_mask, h_med = _text_mask_dpi(small, pagemask, analysis_dpi,
                                          self.line_join_mm, is_bw=is_bw,
                                          kernel_char_mult=self.kernel_char_mult,
                                          large_blob_limit=self.large_blob_limit)
        # Text scale relative to the analysis image (dimensionless), stamped
        # into the output meta as "char_h_frac" for downstream steps.
        char_h_frac = (h_med / float(small.shape[0])) if h_med > 0 else 0.0

        # Tie span-filter bounds to h_med (DPI fallback if char-scale unknown):
        # library defaults drop spans on warped lines or wide inter-word gaps.
        saved = {k: getattr(pd_cfg, k) for k in (
            "TEXT_MAX_THICKNESS", "TEXT_MIN_WIDTH", "TEXT_MIN_HEIGHT",
            "EDGE_MAX_LENGTH", "EDGE_MAX_OVERLAP", "EDGE_MAX_ANGLE",
            "SPAN_MIN_WIDTH",
        )}
        if h_med > 0:
            # Limits in fractions/multiples of median char height — DPI-independent.
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(
                self.thickness_char_mult * h_med)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(0.5 * h_med)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(0.5 * h_med)))
            pd_cfg.EDGE_MAX_LENGTH = max(20, int(round(
                self.edge_max_length_char_mult * h_med)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, 0.1 * h_med)
            # 10× char height ≈ 5 words — shorter spans starve the cubic fit.
            pd_cfg.SPAN_MIN_WIDTH = max(30, int(round(10.0 * h_med)))
        else:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(analysis_dpi * 0.25)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(analysis_dpi * 0.10)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(analysis_dpi * 0.01)))
            pd_cfg.EDGE_MAX_LENGTH = max(100, int(round(analysis_dpi * 0.5)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, analysis_dpi * 0.02)
            pd_cfg.SPAN_MIN_WIDTH = max(30, w // 20)
        pd_cfg.EDGE_MAX_ANGLE = 7.5
        try:
            cinfo_list = get_contours("dewarp", small, text_mask)
            spans = assemble_spans("dewarp", small, pagemask, cinfo_list)
            if self.debug_enabled():
                widths = [int(getattr(c, "local_xrng", (0, 0))[1] -
                              getattr(c, "local_xrng", (0, 0))[0])
                          for c in cinfo_list]
                print(f"[PageDewarper] dewarp_branch={img_buf.branch_label} "
                      f"cinfos={len(cinfo_list)} spans={len(spans)} "
                      f"span_counts={[len(s) for s in spans]} "
                      f"SPAN_MIN_WIDTH={pd_cfg.SPAN_MIN_WIDTH} "
                      f"cinfo_widths sample={widths[:10]}", flush=True)

            if len(spans) < 3:
                # Fall back to "line" mask (rule-detection morphology) for sparse text.
                mask_obj = Mask("dewarp", small, pagemask, "line")
                cinfo_list_line = mask_obj.contours()
                spans2 = assemble_spans("dewarp", small, pagemask, cinfo_list_line)
                if len(spans2) > len(spans):
                    spans = spans2
                    cinfo_list = cinfo_list_line

            # Drop partial-width spans (page-number / footer fragments) —
            # they bias the cubic fit toward false curvature.
            if self.min_span_width_ratio > 0.0 and spans:
                def _sp_w(sp):
                    xs: list[float] = []
                    for c in sp:
                        xs.extend(c.local_xrng)
                    return max(xs) - min(xs) if xs else 0.0
                widths = [_sp_w(s) for s in spans]
                w_max = max(widths)
                w_thr = self.min_span_width_ratio * w_max
                kept = [s for s, w in zip(spans, widths) if w >= w_thr]
                if self.debug_enabled():
                    print(f"[PageDewarper] span width filter: "
                          f"max={w_max:.0f}px thr={w_thr:.0f}px "
                          f"kept={len(kept)}/{len(spans)}", flush=True)
                spans = kept

            if self.debug_enabled():
                # (a) text_mask after threshold + line-join MORPH_CLOSE.
                self.debug_save(text_mask, "0a_text_mask", img_buf)
                # (b) contours returned by get_contours after TEXT_MIN_WIDTH /
                # TEXT_MIN_HEIGHT / TEXT_MAX_THICKNESS filters — missing body
                # lines here means filters too tight or text_mask didn't capture.
                vis_cinfo = small.copy()
                for ci in cinfo_list:
                    x, y, w_b, h_b = ci.rect
                    cv2.rectangle(vis_cinfo, (x, y), (x + w_b, y + h_b),
                                  (0, 200, 0), 1)
                cv2.putText(vis_cinfo,
                            f"cinfos={len(cinfo_list)} "
                            f"TEXT_MIN_W={pd_cfg.TEXT_MIN_WIDTH} "
                            f"H={pd_cfg.TEXT_MIN_HEIGHT} "
                            f"MAX_THICK={pd_cfg.TEXT_MAX_THICKNESS}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 200, 0), 1)
                self.debug_save(vis_cinfo, "0b_contours_kept", img_buf)
                # (c) contours that made it into spans — diff vs (b) shows
                # what assemble_spans dropped (EDGE_MAX_ANGLE / OVERLAP / SPAN_MIN_WIDTH).
                in_span_ids = set()
                for span in spans:
                    for ci in span:
                        in_span_ids.add(id(ci))
                vis_diff = small.copy()
                for ci in cinfo_list:
                    x, y, w_b, h_b = ci.rect
                    col = (0, 200, 0) if id(ci) in in_span_ids else (0, 0, 255)
                    cv2.rectangle(vis_diff, (x, y), (x + w_b, y + h_b),
                                  col, 1)
                cv2.putText(vis_diff,
                            f"in_spans={len(in_span_ids)}/{len(cinfo_list)} "
                            f"SPAN_MIN_W={pd_cfg.SPAN_MIN_WIDTH} "
                            f"EDGE_ANGLE={pd_cfg.EDGE_MAX_ANGLE} "
                            f"OVERLAP={pd_cfg.EDGE_MAX_OVERLAP}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)
                self.debug_save(vis_diff, "0c_contours_kept_vs_spanned", img_buf)
        finally:
            for k, v in saved.items():
                setattr(pd_cfg, k, v)

        success = True
        if len(spans) < 1:
            success = False

        # Coward check: skip dewarp when the cubic-sheet model has too little
        # vertical extent to fit (chapter-end pages, figures, captions).
        coward_skip = False

        # Span-count guard: under-constrained fits (title pages, figure crops)
        # let the optimiser wander OPT_MAX_ITER iters and balloon memory.
        if success and len(spans) < self.min_spans:
            coward_skip = True
            img_buf.meta["fallback_reason"] = (
                f"too few spans ({len(spans)} < {self.min_spans})"
            )
        if success and not coward_skip:
            # Stamp our config into the library's global cfg for the
            # WHOLE geometry path. This used to be scoped to the
            # optimise call and restored right after — solvePnP init
            # (get_default_params), get_page_dims and the remap
            # project_xy then silently read the library default
            # FOCAL_LENGTH=1.2 while the params had been fitted at 1.3
            # (or the calibrated value): systematic projection mismatch.
            # No restore — nothing else reads page_dewarp's cfg, and
            # every process() call re-stamps its own values.
            from page_dewarp.options import cfg as lib_cfg
            for k in self.cfg.__struct_fields__:
                setattr(lib_cfg, k, getattr(self.cfg, k))

            from aglaia.processors import sheet_models

            # 3. Sampling and Optimization
            # x-height-band sampler avoids ascender/descender bias in samples.
            span_points = _sample_spans_xband(small.shape, spans,
                                              baseline_source=self.baseline_source)
            corners, ycoords, xcoords = keypoints_from_samples("dewarp", small, pagemask, page_outline, span_points)
            rough_dims, span_counts, params = get_default_params(corners, ycoords, xcoords)
            span_counts = [len(xc) for xc in xcoords]
            params = params.astype(np.float32)
            # Model page dims: the spline parameterisation needs (W, H)
            # at FIT time — rough dims are all we have before
            # get_page_dims. Stored in replay_params so replay rebuilds
            # the identical surface.
            model_dims = (float(rough_dims[0]), float(rough_dims[1]))
            # Data support in page-x: the optimiser only constrains the
            # sheet where span keypoints exist. Outside this range the
            # spline is tangent-extended at evaluation time so the page
            # margins can't pick up phantom curl.
            if xcoords:
                _xall = np.concatenate([np.asarray(xc).ravel()
                                        for xc in xcoords])
                support_x = (float(_xall.min()), float(_xall.max()))
            else:
                support_x = (0.0, model_dims[0])
            # Same in y for the twist factor: (1 + γ·η) is linear in y and
            # keeps growing past the first/last span — amplified (or, at
            # |γ| > 2, sign-flipped) phantom curl in the top/bottom
            # margins. Clamp η to the span-y data range at evaluation.
            _yall = np.asarray(ycoords).ravel()
            if _yall.size:
                support_y = (float(_yall.min()), float(_yall.max()))
            else:
                support_y = (0.0, model_dims[1])
            # Decay length of the x tangent extension. A pure tangent
            # grows linearly into the margin — a steep gutter wall at the
            # support edge blows the margin grid up.
            support_decay = sheet_models.SUPPORT_DECAY_FRAC * model_dims[0]
            n_extra = sheet_models.n_extras(self.sheet_model,
                                            self.spline_modes)
            # flat_spline per-page state: flip puts the binding at the
            # page's left edge (basis t = 1 − x/W); flat_penalty_eff is
            # zeroed when the binding side can't be resolved.
            flat_flip = False
            flat_penalty_eff = 0.0
            if self.sheet_model == sheet_models.MODEL_FLAT_SPLINE:
                side = self.binding_side
                if side == "auto":
                    # PageDetector stamps page_side on 2-page spreads:
                    # the LEFT page of a spread is bound on its RIGHT
                    # edge, and vice versa.
                    ps = str(img_buf.meta.get("page_side") or "").lower()
                    side = {"left": "right", "right": "left"}.get(ps)
                    if side is None:
                        print("[PageDewarper] flat_spline: no page_side "
                              "meta and binding_side=auto — flat penalty "
                              "off for this page (set binding_side "
                              "explicitly for single-page scans).",
                              flush=True)
                flat_flip = side == "left"
                if side is not None:
                    flat_penalty_eff = self.flat_outer_penalty
            if n_extra:
                params = np.concatenate(
                    [params, np.zeros(n_extra, dtype=params.dtype)])
                flat_w = flat_penalty_eff * sheet_models.flat_outer_weights(
                    self.spline_modes, self.knot_grading)
                if self.backend == "mlx":
                    from aglaia.processors.page_dewarp_mlx import (
                        set_flat as _set_flat,
                        set_model_dims as _set_model_dims)
                    _set_model_dims(*model_dims)
                    _set_flat(flat_flip, flat_w)
                elif self.backend == "jax":
                    from aglaia.processors.page_dewarp_padded import (
                        set_flat as _set_flat,
                        set_model_dims as _set_model_dims)
                    _set_model_dims(*model_dims)
                    _set_flat(flat_flip, flat_w)
            dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points)).astype(np.float32)

            ctx = _DewarpCtx(
                img=img, small=small, pad_px=pad_px, is_bw=is_bw,
                input_dpi=input_dpi, char_h_frac=char_h_frac,
                orig_buffer=_orig_buffer, orig_roi=_orig_roi,
                corners=corners, rough_dims=rough_dims,
                span_counts=span_counts, params=params, model_dims=model_dims,
                support_x=support_x, support_y=support_y,
                support_decay=support_decay, flat_flip=flat_flip,
                flat_penalty_eff=flat_penalty_eff, n_extra=n_extra,
                dstpoints=dstpoints, spans=spans, span_points=span_points)
            return ctx, None

        if coward_skip:
            # Passthrough: keep input pixels as-is, mark fallback in meta.
            img_buf.meta["success"] = False
            img_buf.meta["status"] = int(Status.WARNING)
            return None, img_buf

        # len(spans) < 1 → ERROR gray fallback (the old not-success path).
        # The OOB not-success case is handled separately in _finish_dewarp.
        img_buf.buffer = img_buf.to_gray()
        img_buf.type = ImageType.GRAY
        img_buf.meta["success"] = False
        img_buf.meta["status"] = int(Status.ERROR)
        return None, img_buf

    def _finish_dewarp(self, img_buf: ImageBuffer, params, ctx) -> ImageBuffer:
        """Post-solve: build page_dims (with the divergence guard + bspline
        fallback retry), run the arc-length remap, forward the ROI, gate on
        OOB and stamp the output buffer + replay params. Reads every build
        local from ``ctx``; returns the final img_buf."""
        from aglaia.processors import sheet_models

        corners = ctx.corners
        rough_dims = ctx.rough_dims
        model_dims = ctx.model_dims
        support_x = ctx.support_x
        support_y = ctx.support_y
        support_decay = ctx.support_decay
        flat_flip = ctx.flat_flip
        flat_penalty_eff = ctx.flat_penalty_eff
        span_counts = ctx.span_counts
        dstpoints = ctx.dstpoints
        img = ctx.img
        pad_px = ctx.pad_px
        char_h_frac = ctx.char_h_frac
        _orig_buffer = ctx.orig_buffer
        _orig_roi = ctx.orig_roi
        params_initial = ctx.params_initial
        success = True

        with SuppressOutput():
            page_dims = sheet_models.get_page_dims_model(
                corners, rough_dims, params,
                model=self.sheet_model, n_modes=self.spline_modes,
                model_dims=model_dims, focal_length=self.focal_length,
                support=support_x, support_y=support_y,
                support_decay=support_decay,
                grading=self.knot_grading, flip=flat_flip)

        if np.any(page_dims < 0):
            page_dims = rough_dims
        # Divergence guard: a runaway fit explodes page_dims (1e6+) and
        # then overflows the arc-length remap below. Catch it here and
        # retry the page with the stable bspline model; if we're already
        # on a spline model, fall back to a safe near-identity instead.
        if float(np.max(np.abs(page_dims))) > self._PAGE_DIM_SANE:
            fb = self._fallback_to_bspline(img_buf, _orig_buffer, _orig_roi)
            if fb is not None:
                return fb
            print(f"[PageDewarper] {self.sheet_model} fit diverged "
                  f"(page_dims={page_dims}); passing through.", flush=True)
            page_dims = rough_dims
        if self.sheet_model in sheet_models.SPLINE_MODELS:
            _, _c, _g = sheet_models.split_extras(
                params, self.sheet_model, self.spline_modes)

            def _kp_rms(pv):
                ki = make_keypoint_index(span_counts)
                pp = sheet_models.project_keypoints_model(
                    np.asarray(pv, dtype=np.float64).copy(), ki,
                    model=self.sheet_model, n_modes=self.spline_modes,
                    model_dims=model_dims,
                    focal_length=self.focal_length,
                    grading=self.knot_grading,
                    flip=flat_flip).reshape(-1, 2)
                tgt = dstpoints.reshape(-1, 2)
                return float(np.sqrt(np.mean(
                    np.sum((tgt - pp) ** 2, axis=1))))

            _flat_note = ""
            if self.sheet_model == sheet_models.MODEL_FLAT_SPLINE:
                _flat_note = (f" binding={'left' if flat_flip else 'right'}"
                              f" g={self.knot_grading:g}"
                              f" lam={flat_penalty_eff:g}")
            print(f"[PageDewarper] {self.sheet_model} fit: c="
                  f"[{', '.join(f'{v:+.4f}' for v in _c)}] "
                  f"gamma={_g:+.4f} "
                  f"rms {_kp_rms(params_initial):.5f}->"
                  f"{_kp_rms(params):.5f}{_flat_note}", flush=True)

        # 5. Remapping — shared arc-length grid (same code path replay
        # uses, so the live remap and a replayed remap are pixel-identical).
        zoom = 1.0
        decimate = self.cfg.REMAP_DECIMATE
        image_points, grid_shape, target_w, target_h, w_small, h_small = \
            self._sample_grid(
                img.shape[0], img.shape[1], params=params,
                page_dims_w=page_dims[0], page_dims_h=page_dims[1],
                model_dims=model_dims, decimate=decimate, zoom=zoom,
                model=self.sheet_model, n_modes=self.spline_modes,
                focal=self.focal_length, support=support_x,
                support_y=support_y, support_decay=support_decay,
                grading=self.knot_grading, flip=flat_flip)

        im_x_dec = image_points[:, 0, 0].reshape(grid_shape).astype(np.float32)
        im_y_dec = image_points[:, 0, 1].reshape(grid_shape).astype(np.float32)

        im_x = cv2.resize(im_x_dec, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        im_y = cv2.resize(im_y_dec, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # BW → nearest (preserve crisp text); gray/color → cubic.
        # White border (not BORDER_REPLICATE) — replicating the dark
        # book-edge would smear inward and bias the downstream binariser.
        input_is_bw = img_buf.type == ImageType.BW
        interp = cv2.INTER_NEAREST if input_is_bw else cv2.INTER_CUBIC
        border_val = 255 if img.ndim == 2 else (255, 255, 255)
        remapped = cv2.remap(img, im_x, im_y, interp, None,
                             cv2.BORDER_CONSTANT, border_val)
        if input_is_bw:
            _, remapped = cv2.threshold(remapped, 127, 255, cv2.THRESH_BINARY)

        if self.debug_enabled():
            self.debug_save(remapped, "3_remapped", img_buf)
            # Warp mesh: where each output pixel reads from. Subsample im_x/im_y.
            try:
                src_vis = img.copy()
                if src_vis.ndim == 2:
                    src_vis = cv2.cvtColor(src_vis, cv2.COLOR_GRAY2BGR)
                th, tw = im_x.shape
                step_r = max(8, th // 32)
                step_c = max(8, tw // 32)
                for r in range(0, th, step_r):
                    pts = np.stack([im_x[r, ::step_c], im_y[r, ::step_c]],
                                   axis=-1).astype(np.int32)
                    cv2.polylines(src_vis, [pts], False, (0, 220, 60), 1, cv2.LINE_AA)
                for c in range(0, tw, step_c):
                    pts = np.stack([im_x[::step_r, c], im_y[::step_r, c]],
                                   axis=-1).astype(np.int32)
                    cv2.polylines(src_vis, [pts], False, (0, 220, 60), 1, cv2.LINE_AA)
                self.debug_save(src_vis, "4_grid_source", img_buf)

                # Uniform mesh on rectified output.
                dst_vis = remapped.copy()
                if dst_vis.ndim == 2:
                    dst_vis = cv2.cvtColor(dst_vis, cv2.COLOR_GRAY2BGR)
                oh, ow = dst_vis.shape[:2]
                for r in range(0, oh, step_r):
                    cv2.line(dst_vis, (0, r), (ow - 1, r), (0, 220, 60), 1, cv2.LINE_AA)
                for c in range(0, ow, step_c):
                    cv2.line(dst_vis, (c, 0), (c, oh - 1), (0, 220, 60), 1, cv2.LINE_AA)
                self.debug_save(dst_vis, "5_grid_rectified", img_buf)
            except Exception as e:
                print(f"[{self.name}] grid debug failed: {e}")

        # Forward the ROI polygon: rasterise it on the padded source,
        # remap with the same warp, extract the new contour. The
        # downstream Binarizer can then mask off non-page area.
        # ROI is already in padded-buffer coords (shifted at step 0).
        old_roi = img_buf.meta.get("roi")
        if old_roi:
            roi_pts_src = np.array(old_roi, dtype=np.int32).reshape(-1, 1, 2)
            roi_mask_src = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask_src, [roi_pts_src], 255)
            # Remap on the decimated grid (1/decimate² the pixels of a
            # full-res remap) — a polygon ROI doesn't need sub-pixel
            # edges; the contour scales back by ×decimate.
            roi_mask_warp = cv2.remap(
                roi_mask_src, im_x_dec, im_y_dec,
                cv2.INTER_NEAREST, None, cv2.BORDER_CONSTANT, 0,
            )
            cnts, _ = cv2.findContours(
                roi_mask_warp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if cnts:
                biggest = max(cnts, key=cv2.contourArea)
                scale_x = target_w / float(w_small)
                scale_y = target_h / float(h_small)
                roi_full = biggest.reshape(-1, 2).astype(np.float64)
                roi_full[:, 0] *= scale_x
                roi_full[:, 1] *= scale_y
                img_buf.meta["roi"] = roi_full.tolist()

        # OOB stats
        oh, ow = img.shape[:2]
        oob = {
            "x_oob": float(max(0.0, -np.min(im_x), np.max(im_x) - (ow - 1))),
            "y_oob": float(max(0.0, -np.min(im_y), np.max(im_y) - (oh - 1)))
        }
        img_buf.meta["oob"] = oob

        if oob["x_oob"] > self.max_oob or oob["y_oob"] > self.max_oob:
            success = False

        if not success:
            # Fallback to gray on padded image.
            img_buf.buffer = img_buf.to_gray()
            img_buf.type = ImageType.GRAY
            img_buf.meta["success"] = False
            img_buf.meta["status"] = int(Status.ERROR)
            return img_buf

        # Preserve original buffer type: BW input → BW output, etc.
        original_type = img_buf.type
        img_buf.buffer = remapped
        if remapped.ndim == 2:
            img_buf.type = original_type if original_type in (ImageType.BW, ImageType.GRAY) else ImageType.GRAY
        else:
            img_buf.type = ImageType.COLOR

        img_buf.meta["success"] = success
        if char_h_frac > 0:
            img_buf.meta["char_h_frac"] = char_h_frac
        # Replay params: cubic-sheet remap on the source buffer. The
        # replay engine rebuilds the grid from `params` + `page_dims`
        # against the source image shape and applies a single warp.
        img_buf.meta["replay_kind"] = "dewarp"
        img_buf.meta["replay_params"] = {
            "params": params.tolist(),
            "page_dims": [float(page_dims[0]), float(page_dims[1])],
            "src_shape": [int(img.shape[0]), int(img.shape[1])],
            "pad_px": int(pad_px),
            "zoom": float(1.0),
            "decimate": int(self.cfg.REMAP_DECIMATE),
            "sheet_model": self.sheet_model,
            "spline_modes": int(self.spline_modes),
            "model_dims": [float(model_dims[0]), float(model_dims[1])],
            "focal_length": float(self.focal_length),
            # Twist-model data support: page-x tangent-extended outside the
            # support (no phantom margin curl), page-y η-clamped at its edge.
            "support_x": [float(support_x[0]), float(support_x[1])],
            "support_y": [float(support_y[0]), float(support_y[1])],
            # Decay length λ of the page-x tangent extension.
            "support_decay": float(support_decay),
            # flat_spline surface geometry: graded knot vector + binding flip.
            "knot_grading": float(self.knot_grading),
            "binding_flip": bool(flat_flip),
        }

        # Only max_oob (above) gates status. Sub-threshold fringe OOB is
        # expected on any non-zero page-margin remap.
        img_buf.meta["status"] = int(Status.SUCCESS)
        return img_buf
