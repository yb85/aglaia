# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
import numpy as np
from dataclasses import dataclass, field, make_dataclass
from lib.ImageBuffer import ImageBuffer, ImageType
from lib.processors.utils import to_gray, to_rgb
from lib.processors.abstraction import (
    AbstractImageProcessor, AbstractProcessorOption, ReplayTrait, _fmt_value,
)
from lib.processors.option_specs import _e, _f, _i, _s

from typing import Union, Dict, Any


# Per-algorithm-family sensible defaults. Sauvola wants k≈0.2, Wolf wants
# k≈0.5, NICK/Niblack want negative k around -0.2 — sharing a single `k`
# slider across all of them was harmful. Each family ships its own
# `k_<family>`, `window_mm_<family>`, `window_px_<family>` so switching
# the method picks up the right defaults without silently inheriting
# Wolf's k into a Niblack run.
_FAMILIES: Dict[str, Dict[str, Any]] = {
    "wolf":      {"methods": ["wolf++", "wolf"],
                  "k": {"default": 0.5,  "min": -0.5, "max": 1.0},
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "sauvola":   {"methods": ["sauvola", "isauvola", "wan"],
                  "k": {"default": 0.2,  "min": 0.0,  "max": 1.0},
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "niblack":   {"methods": ["niblack"],
                  "k": {"default": -0.2, "min": -1.0, "max": 1.0},
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "nick":      {"methods": ["nick"],
                  "k": {"default": -0.2, "min": -1.0, "max": 0.0},
                  "window_mm_default": 1.6,
                  "window_px_default": 19},
    "trsingh":   {"methods": ["trsingh"],
                  "k": {"default": 0.2,  "min": -1.0, "max": 1.0},
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "bernsen":   {"methods": ["bernsen"],
                  "k": None,
                  "window_mm_default": 2.5,
                  "window_px_default": 31},
    "su":        {"methods": ["su"],
                  "k": None,
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "bataineh":  {"methods": ["bataineh"],
                  "k": None,
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "gatos":     {"methods": ["gatos"],
                  "k": None,
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
    "adotsu":    {"methods": ["adotsu"],
                  "k": None,
                  "window_mm_default": 3.2,
                  "window_px_default": 30},
}

# Method → family lookup. `otsu`, `gray`, `none` aren't here: Otsu is a
# global threshold (no window/k), and the pass-throughs ignore them.
_METHOD_TO_FAMILY: Dict[str, str] = {
    m: fam for fam, cfg in _FAMILIES.items() for m in cfg["methods"]
}

_ALL_DOXA_METHODS = list(_METHOD_TO_FAMILY) + ["otsu"]


def _build_binarizer_options() -> Dict[str, Any]:
    """Generate OPTIONS at class-def time from the family table. Each
    family contributes its own `window_mm_<fam>`, `window_px_<fam>` and
    (when applicable) `k_<fam>`, all gated on the matching methods via
    `visible_when` so the editor never shows an irrelevant knob."""
    methods = (list(_METHOD_TO_FAMILY)
               + ["otsu", "gray", "none"])
    opts: Dict[str, Any] = {
        "method": _e(
            "wolf++", methods,
            "Binarization algorithm. `wolf++` is the ROI-aware Wolf "
            "variant — at replay time it uses a mask-aware Wolf that "
            "ignores the synthetic page→canvas border, then falls back "
            "to plain doxapy Wolf on frames with no missing pixels. "
            "Other names are doxapy algorithms; `gray` keeps grayscale, "
            "`none` passes through. Each family gets its own k / window "
            "defaults — switching the method picks up the right "
            "tuning automatically.",
        ),
    }
    for fam, cfg in _FAMILIES.items():
        ms = cfg["methods"]
        nice = " / ".join(ms)
        opts[f"window_mm_{fam}"] = _f(
            cfg["window_mm_default"], 0.0, 50.0, 0.1,
            f"Sliding-window size in millimetres for {nice}. "
            f"When > 0 it overrides `window_px_{fam}`.",
            visible_when={"method": ms},
        )
        opts[f"window_px_{fam}"] = _i(
            cfg["window_px_default"], 4, 4096,
            f"Sliding-window size in pixels for {nice}. "
            f"Used only when `window_mm_{fam}` is 0.",
            visible_when={"method": ms},
        )
        if cfg["k"]:
            k_cfg = cfg["k"]
            opts[f"k_{fam}"] = _f(
                k_cfg["default"], k_cfg["min"], k_cfg["max"], 0.01,
                f"Threshold bias coefficient for {nice}.",
                visible_when={"method": ms},
            )
    opts["bernsen_contrast"] = _i(
        15, 0, 255,
        "Bernsen local-contrast threshold; pixels below it are routed "
        "to a global cut.",
        visible_when={"method": ["bernsen"]},
    )
    opts["roi_shrink"] = _i(
        0, 0, 50,
        "If meta.roi is set, erode the mask N iterations before applying.",
        advanced=True,
        visible_when={"method": _ALL_DOXA_METHODS},
    )
    opts["morpho_close"] = _i(
        0, 0, 10,
        "Post-binarisation morphological closing of text strokes. "
        "0 = off; 1..10 = ellipse kernel of diameter N px. Bridges "
        "small white pinholes inside dark glyph strokes.",
        visible_when={"method": _ALL_DOXA_METHODS},
    )
    return opts

def _build_binarizer_option_class():
    """Generate `BinarizerOption` as a dataclass with one field per
    `_FAMILIES` entry. Initializer's drop-unknown-kwargs path reads
    `dataclasses.fields()` to decide what survives, so the class must
    be a real dataclass — `make_dataclass` keeps the explosion in this
    one helper."""
    fields = [
        ("method", str, "wolf++"),
        ("bernsen_contrast", int, 15),
        ("roi_shrink", int, 0),
        ("morpho_close", int, 0),
        ("extra", Any, field(default_factory=dict)),
    ]
    for fam, cfg in _FAMILIES.items():
        fields.append((f"window_mm_{fam}", float, cfg["window_mm_default"]))
        fields.append((f"window_px_{fam}", int, cfg["window_px_default"]))
        if cfg["k"]:
            fields.append((f"k_{fam}", float, cfg["k"]["default"]))
    cls = make_dataclass(
        "BinarizerOption", fields,
        bases=(AbstractProcessorOption,),
        init=False,
    )
    cls.__module__ = __name__

    def __init__(self, **kwargs):
        # Inherited base fields.
        self.debug = kwargs.pop("debug", False)
        self.debug_dir = kwargs.pop("debug_dir", None)

        self.method = kwargs.pop("method", "wolf++")

        # Per-family fields with their literature defaults.
        for fam, cfg in _FAMILIES.items():
            setattr(self, f"window_mm_{fam}",
                    float(kwargs.pop(f"window_mm_{fam}",
                                     cfg["window_mm_default"])))
            setattr(self, f"window_px_{fam}",
                    int(kwargs.pop(f"window_px_{fam}",
                                   cfg["window_px_default"])))
            if cfg["k"]:
                setattr(self, f"k_{fam}",
                        float(kwargs.pop(f"k_{fam}",
                                         cfg["k"]["default"])))

        # Back-compat for the old shared `window` / `window_mm` /
        # `window_px` / `k` fields. Map them onto the currently-selected
        # family so existing yaml files keep working through one cycle.
        active_family = _METHOD_TO_FAMILY.get(str(self.method).lower())
        legacy_window = kwargs.pop("window", None)
        legacy_window_mm = kwargs.pop("window_mm", None)
        legacy_window_px = kwargs.pop("window_px", None)
        legacy_k = kwargs.pop("k", None)
        if active_family:
            if isinstance(legacy_window, (int, float)):
                setattr(self, f"window_px_{active_family}", int(legacy_window))
            elif isinstance(legacy_window, str):
                print(f"[Binarizer] dropping legacy template `window={legacy_window}`"
                      f" — set window_mm_{active_family} or window_px_{active_family}.")
            if legacy_window_mm is not None:
                setattr(self, f"window_mm_{active_family}", float(legacy_window_mm))
            if legacy_window_px is not None:
                setattr(self, f"window_px_{active_family}", int(legacy_window_px))
            if legacy_k is not None and _FAMILIES[active_family]["k"]:
                setattr(self, f"k_{active_family}", float(legacy_k))

        self.bernsen_contrast = int(kwargs.pop("bernsen_contrast", 15))
        self.roi_shrink = kwargs.pop("roi_shrink", 0)
        # Clamp to the spec range so a stray yaml value can't drag a
        # huge kernel through cv2 and tank a scan's elapsed_ms.
        mc = int(kwargs.pop("morpho_close", 0))
        self.morpho_close = max(0, min(10, mc))
        self.extra = kwargs

    cls.__init__ = __init__
    return cls


BinarizerOption = _build_binarizer_option_class()

class Binarizer(AbstractImageProcessor):
    name: str = "Binarizer"
    SUMMARY = "Doxapy (Wolf/Sauvola/…) or grayscale pass-through."
    REPLAY_TRAIT = ReplayTrait.PIXEL_VALUE  # windowed thresholding → applied last
    OPTION_CLASS = BinarizerOption
    OPTIONS = _build_binarizer_options()

    @classmethod
    def describe_options(cls, options, verbosity: str = "essential") -> str:
        """Method-aware: only the active family's window/k are interesting,
        not every family's. Essential → ``method · window · k``; full lists
        the active family's tuned values."""
        o = options
        method = str(getattr(o, "method", "wolf++"))
        fam = _METHOD_TO_FAMILY.get(method.lower())
        bits = [f"method {method}"]
        if fam is not None:
            wmm = getattr(o, f"window_mm_{fam}", None)
            if wmm is not None:
                bits.append(f"window {_fmt_value(wmm)} mm")
            if _FAMILIES.get(fam, {}).get("k"):
                kv = getattr(o, f"k_{fam}", None)
                if kv is not None:
                    bits.append(f"k {_fmt_value(kv)}")
        if verbosity == "essential":
            return " · ".join(bits)
        bits.append(f"bernsen_contrast: {getattr(o, 'bernsen_contrast', '')}")
        bits.append(f"morpho_close: {getattr(o, 'morpho_close', '')}")
        bits.append(f"roi_shrink: {getattr(o, 'roi_shrink', '')}")
        return "\n".join(bits)

    def __init__(self, options: BinarizerOption):
        super().__init__(options)

        self.mode = "NONE"
        # All algorithm-specific knobs are now per-family fields on
        # `options` (e.g. `k_wolf`, `window_mm_sauvola`). `_params_for_frame`
        # picks the right set based on the active method. `extra` only
        # holds yaml-passed unrecognised keys.
        self.params = options.extra or {}
        self.roi_shrink = options.roi_shrink

        algo_name = str(options.method).upper()
        # `wolf++` and `wolf` both run plain doxapy Wolf on the forward
        # pass; the `++` distinction kicks in at replay time where the
        # mask-aware variant takes over (see Replay._apply_binarize).
        if algo_name == "WOLF++":
            algo_name = "WOLF"
        if algo_name == "NONE":
            return

        if algo_name == "GRAY":
            self.mode = "GRAY"
            return

        # Doxapy Check
        try:
            import doxapy
            if hasattr(doxapy.Binarization.Algorithms, algo_name):
                algo_enum = getattr(doxapy.Binarization.Algorithms, algo_name)
                self.binarizer = doxapy.Binarization(algo_enum)
                self.mode = "DOXA"
            else:
                 print(f"Unknown binarization algo {algo_name}. Fallback to GRAY.")
                 self.mode = "GRAY"
        except Exception as e:
            print(f"Doxapy init error: {e}. Fallback to GRAY.")
            self.mode = "GRAY"

    def _active_family(self) -> Union[str, None]:
        return _METHOD_TO_FAMILY.get(str(self.options.method).lower())

    def _resolve_window_px(self, img_buf: ImageBuffer) -> int:
        """Pick the window size in pixels for this frame, reading the
        active family's `window_mm_<fam>` / `window_px_<fam>` pair.
        Falls back to 30 px when the method has no family (otsu, gray,
        none — which don't use a window anyway)."""
        fam = self._active_family()
        if fam is None:
            return 30
        window_mm = float(getattr(self.options, f"window_mm_{fam}", 0.0))
        if window_mm > 0:
            dpi = float(img_buf.dpi) or 300.0
            return max(4, int(round(window_mm / 25.4 * dpi)))
        return max(4, int(getattr(self.options, f"window_px_{fam}", 30)))

    def _params_for_frame(self, img_buf: ImageBuffer) -> dict:
        """Doxapy keyword dict for this image. Method-specific knobs
        get folded in only when the active algorithm reads them, so
        doxapy doesn't get confused by extras it doesn't recognise."""
        method = str(self.options.method).lower()
        fam = self._active_family()
        out: dict = {}
        if fam is not None:
            out["window"] = self._resolve_window_px(img_buf)
            if _FAMILIES[fam].get("k"):
                out["k"] = float(getattr(self.options, f"k_{fam}"))
        if method == "bernsen":
            out["contrast_limit"] = int(self.options.bernsen_contrast)
        for k, v in self.params.items():
            if k in out:
                continue
            out[k] = v
        return out

    def process(self, img_buf: ImageBuffer) -> ImageBuffer:
        """
        Binarize the content of the ImageBuffer associated.
        Updates the buffer in-place and returns the object.
        """
        if self.mode == "NONE":
             # Pass-through (could be color or gray)
             return img_buf
        
        if self.mode == "GRAY":
            # If it's already BW, do nothing. If Color/Gray, ensure Gray.
            if img_buf.type == ImageType.BW:
                return img_buf
            img_buf.buffer = img_buf.to_gray()
            img_buf.type = ImageType.GRAY
            return img_buf

        # For actual binarizers, we need RGB or Gray input
        # User asked: "handle the different image type (bw, gray and color => passthrough if bw)"
        if img_buf.type == ImageType.BW:
            return img_buf

        # No pre-binarisation bg-fill — interior lighting gradients still
        # produce ink rings and halo erosion clips text on tight bboxes.
        # Use the post-mask erosion path instead.

        # DOXA takes Gray
        if self.mode == "DOXA":
            gray = img_buf.to_gray()
            try:
                self.binarizer.initialize(gray)
                binary = np.empty_like(gray)

                current_params = self._params_for_frame(img_buf)
                self.binarizer.to_binary(binary, current_params)
                img_buf.buffer = binary
                img_buf.type = ImageType.BW
            except Exception as e:
                print(f"Doxa error: {e}")
                img_buf.buffer = gray
                img_buf.type = ImageType.GRAY

            if self.debug_enabled():
                self._dump_debug(img_buf, gray, current_params)

            # Replay params: re-binarize the replay buffer with the same
            # algorithm at the end of the chain (one threshold over the
            # final, cleanly-resampled pixels). Window is already in
            # pixels, k is a plain float — no resolution gymnastics.
            img_buf.meta["replay_kind"] = "binarize"
            fam = self._active_family()
            k_default = 0.0
            if fam is not None and _FAMILIES[fam].get("k"):
                k_default = float(getattr(self.options, f"k_{fam}"))
            img_buf.meta["replay_params"] = {
                "method": str(self.options.method),
                "window": int(current_params.get(
                    "window", self._resolve_window_px(img_buf))),
                "k": float(current_params.get("k", k_default)),
                "roi_shrink": int(self.options.roi_shrink),
                "morpho_close": int(self.options.morpho_close),
            }
            self._apply_roi_mask(img_buf)
            if self.morpho_close_n() > 0:
                img_buf.buffer = morpho_close(
                    img_buf.buffer, self.morpho_close_n(),
                )
            self._record_stats(img_buf)
            return img_buf

    def morpho_close_n(self) -> int:
        return int(getattr(self.options, "morpho_close", 0) or 0)

    def _record_stats(self, img_buf: "ImageBuffer") -> None:
        """Populate ``self.last_stats`` after binarisation. Used by the
        chain's unified op-log line. Includes the blob-area percentile
        distribution + the active method so a glance at the log tells
        you whether Wolf is under-binarising (p50 tiny → over-eroded
        text) or over-binarising (p100 huge → bleed-through).

        Runs on a 2x-strided subsample unless debug is enabled: full-res
        connectedComponents costs ~150-250 ms on a 12 MP frame just for a
        log line. Subsampled areas are scaled x4 so percentiles stay
        comparable across modes (p0/p10 are approximate — 1-3 px specks
        can vanish in the subsample)."""
        try:
            bw = img_buf.buffer
            if bw is None or bw.size == 0:
                return
            if self.debug_enabled():
                sample, area_scale = bw, 1
            else:
                sample = np.ascontiguousarray(bw[::2, ::2])
                area_scale = 4
            inv = cv2.bitwise_not(sample)   # blobs = foreground = 1
            n, _labels, stats, _cents = cv2.connectedComponentsWithStats(
                inv, connectivity=8
            )
            if n <= 1:
                self.last_stats = {
                    "method": str(self.options.method).lower(),
                    "blobs": 0,
                }
                return
            # stats[0] is the background — skip.
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) * area_scale
            p0, p10, p50, p90, p100 = np.percentile(
                areas, [0, 10, 50, 90, 100]
            )
            self.last_stats = {
                "method": str(self.options.method).lower(),
                "blobs": int(n - 1),
                "blob_px": f"[p0={int(p0)} p10={int(p10)} "
                            f"p50={int(p50)} p90={int(p90)} p100={int(p100)}]",
            }
        except Exception:
            self.last_stats = {"method": str(self.options.method).lower()}

        return img_buf

    def _fill_outside_roi_with_bg(self, img_buf, window_px: int = 0):
        """Replace pixels outside the ROI polygon (PLUS a halo INSIDE
        the polygon) with the estimated page-background colour. Wolf's
        adaptive window then sees uniform bg at the polygon edge
        instead of a real-page → bg discontinuity, so it doesn't
        produce a ring of black "ink" along the polygon.

        Halo width = max(roi_shrink, ceil(window_px / 2) + safety) so
        Wolf's window — even when centred on the original polygon edge —
        only touches bg-filled pixels, never the lighting gradient.
        The bg estimate is taken from the *inner* polygon (after halo
        erosion) so it represents true page paper."""
        roi = img_buf.meta.get("roi")
        if not roi:
            return
        try:
            h, w = img_buf.buffer.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            if cv2.countNonZero(mask) == 0:
                return
            halo = int(max(self.roi_shrink, (window_px // 2) + 4))
            inner = mask
            if halo > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                inner = cv2.erode(mask, kernel, iterations=halo)
                if cv2.countNonZero(inner) == 0:
                    inner = mask  # erosion ate everything; fall back
            buf = img_buf.buffer
            if buf.ndim == 2:
                bg = int(np.percentile(buf[inner > 0], 80))
            else:
                bg = np.percentile(buf[inner > 0].reshape(-1, buf.shape[2]),
                                   80, axis=0).astype(buf.dtype).tolist()
            buf[inner == 0] = bg
        except Exception as e:
            print(f"ROI bg-fill error: {e}")

    def _dump_debug(self, img_buf, gray_in, current_params):
        """Dump:
        - input gray buffer (after bg-fill) — what the binariser actually saw
        - binary output
        - window-size overlay on input (rectangle == sliding window)
        - ROI polygon overlay on input"""
        try:
            self.debug_save(gray_in, "0_input_gray", img_buf)
            self.debug_save(img_buf.buffer, "3_binary_out", img_buf)
            overlay = cv2.cvtColor(gray_in, cv2.COLOR_GRAY2BGR)
            win = current_params.get("window")
            if isinstance(win, (int, float)) and win > 0:
                wpx = int(win)
                h, w = gray_in.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.rectangle(overlay, (cx - wpx // 2, cy - wpx // 2),
                              (cx + wpx // 2, cy + wpx // 2), (0, 255, 0), 2)
                cv2.putText(overlay, f"window={wpx}px k={current_params.get('k')}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            roi = img_buf.meta.get("roi")
            if roi:
                pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)
            self.debug_save(overlay, "1_window_overlay", img_buf)
        except Exception as e:
            print(f"[{self.name}] debug dump failed: {e}")

    def _apply_roi_mask(self, img_buf):
        # Force outside-ROI pixels (and a `roi_shrink`-px ring just inside
        # the polygon) to pure white AFTER binarisation. The erosion is
        # what kills Wolf's adaptive-window border artifact: the
        # bg-fill → real-page discontinuity at the ROI boundary trips Wolf
        # into marking a thin ring as ink. Erosion trims that ring.
        roi = img_buf.meta.pop("roi", None)
        if not roi:
            return
        try:
            h, w = img_buf.buffer.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            if cv2.countNonZero(mask) == 0:
                return
            if self.roi_shrink > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.erode(mask, kernel, iterations=int(self.roi_shrink))
            mask_inv = cv2.bitwise_not(mask)
            img_buf.buffer = cv2.bitwise_or(img_buf.buffer, mask_inv)
        except Exception as e:
            print(f"ROI white-mask error: {e}")
    
    # Backward compatibility alias if needed, but we should update callers
    def binarize(self, img_buf: ImageBuffer) -> ImageBuffer:
        return self.process(img_buf)


def morpho_close(bw: np.ndarray, n: int) -> np.ndarray:
    """Fill small white pinholes / bridge tiny stroke gaps in a BW image.

    Convention: text=0, background=255. We invert so the foreground
    becomes the bright side (matching OpenCV's MORPH_CLOSE semantics),
    apply a single ellipse-kernel CLOSE, then invert back. Returns the
    input untouched when `n <= 0`.
    """
    if n <= 0 or bw is None or bw.size == 0:
        return bw
    n = max(1, min(10, int(n)))
    if bw.ndim != 2:
        gray = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    else:
        gray = bw
    # Ellipse outperforms square at N>=3: square kernels add rectangular
    # aliasing on serifs / round glyphs; the disk stays isotropic.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    inv = cv2.bitwise_not(gray)
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)


def wolf_masked(gray: np.ndarray, mask: np.ndarray, window_px: int,
                k: float = 0.25, min_valid_frac: float = 0.1) -> np.ndarray:
    """ROI-aware Wolf binarisation.

    Standard Wolf with hard bg-fill rings the polygon boundary because
    the synthetic-bg / real-page step is a giant local-window gradient.
    This variant restricts local mean / std to mask>0 pixels only via
    masked box-sums, so windows that straddle the boundary see the page
    side only — the threshold tracks real illumination instead of a
    fictitious edge. Eliminates the "doom line" artefact along the spine
    on close shade-change pages without shrinking the polygon.

    Wolf globals (M = ROI min, R = max local std) restricted to the ROI
    too, matching the algorithm's normalisation intent on the page only.

    Windows with `< min_valid_frac` of their cells inside the ROI fall
    back to a single ROI-wide Otsu threshold so the local estimate
    doesn't go noisy on a handful of pixels at sharp polygon corners.
    """
    if gray.ndim != 2:
        raise ValueError("wolf_masked expects grayscale uint8")
    w = max(int(window_px) | 1, 3)
    ksize = (w, w)
    m01 = (mask > 0).astype(np.float32)
    g = gray.astype(np.float32) * m01
    g2 = g * g

    N = cv2.boxFilter(m01, cv2.CV_32F, ksize, normalize=False,
                      borderType=cv2.BORDER_CONSTANT)
    S = cv2.boxFilter(g, cv2.CV_32F, ksize, normalize=False,
                      borderType=cv2.BORDER_CONSTANT)
    S2 = cv2.boxFilter(g2, cv2.CV_32F, ksize, normalize=False,
                       borderType=cv2.BORDER_CONSTANT)

    Nsafe = np.where(N > 0.5, N, 1.0)
    mean = S / Nsafe
    var = np.maximum(S2 / Nsafe - mean * mean, 0.0)
    std = np.sqrt(var)

    roi_pixels = gray[mask > 0]
    if roi_pixels.size == 0:
        return np.full_like(gray, 255)
    M_global = float(roi_pixels.min())

    enough = N >= (w * w * min_valid_frac)
    if enough.any():
        R = float(std[enough].max())
    else:
        R = 1.0
    R = max(R, 1e-6)

    T = mean + k * (std / R - 1.0) * (mean - M_global)
    bw = np.where(gray.astype(np.float32) > T, 255, 0).astype(np.uint8)

    fb_zone = (~enough) & (mask > 0)
    if fb_zone.any():
        otsu_thr, _ = cv2.threshold(roi_pixels.reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw[fb_zone] = np.where(gray[fb_zone] > otsu_thr, 255, 0)

    bw[mask == 0] = 255
    return bw
