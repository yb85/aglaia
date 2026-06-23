# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR engine abstraction.

`OcrEngine.recognize` takes an RGB ndarray + a list of language codes
(BCP-47 / ISO-639) and returns an `OcrResult` dict serialisable to JSON.
"""

from __future__ import annotations

import abc
import warnings
from typing import Callable, Optional, TypedDict


class OcrLine(TypedDict, total=False):
    text: str
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 in image pixels (top-left origin)
    confidence: float                  # 0..1
    quad: list[tuple[float, float]]   # optional 4-point quad if non-axis-aligned


class OcrResult(TypedDict, total=False):
    engine: str
    languages: list[str]
    page_w: int
    page_h: int
    lines: list[OcrLine]
    meta: dict


class OcrEngine(abc.ABC):
    """Base class for OCR engines (built-in and drop-in plugins).

    Contract: set `name` (the registry key — unique, non-empty), implement
    `recognize` returning an `OcrResult` (see the TypedDicts above for the
    output format), and register the class with `@register`. `display` /
    `description` feed the OCR tab's engine combo. `__init_subclass__`
    flags a subclass that forgets to set `name`."""

    name: str = "abstract"
    display: str = "Abstract"
    # One-line tagline shown in the OCR tab's engine combo. Keep ≤ ~80
    # chars so the dropdown stays compact.
    description: str = ""
    available: bool = False     # True once deps loaded successfully

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name") and cls.name in ("", "abstract"):
            warnings.warn(
                f"{cls.__name__} does not set a `name` — it cannot be "
                f"registered/resolved as an OCR engine.", stacklevel=2)

    def configure(self, params: dict[str, str]) -> None:
        """Apply per-invocation params from the standard engine spec
        (``--do-ocr name:key=value:…`` → ``{'key': 'value', …}``).

        Base implementation is a no-op. Engines (built-in or drop-in
        plugins) override this to consume their own keys; **ignore unknown
        keys** so the same spec stays forward-compatible across engines.
        Values arrive as strings — coerce as needed."""

    @abc.abstractmethod
    def recognize(self, image_rgb, languages: list[str],
                   *, src_dpi: float | None = None) -> OcrResult:
        """Recognise text in an RGB ndarray; return an `OcrResult`."""
        raise NotImplementedError

    def recognize_batch(self, images_rgb,
                         languages: list[str],
                         *, src_dpis: list[float] | None = None) -> list[OcrResult]:
        """Default fan-out — engines that can drive concurrent backends
        (Surya → llama-server's parallel slots) override this. The base
        impl falls back to serial recognize() so caller code can always
        use the batch API."""
        if src_dpis is None:
            src_dpis = [None] * len(images_rgb)
        return [self.recognize(img, languages, src_dpi=dpi)
                for img, dpi in zip(images_rgb, src_dpis)]


ENGINE_REGISTRY: dict[str, type[OcrEngine]] = {}


def register(cls: type[OcrEngine]) -> type[OcrEngine]:
    ENGINE_REGISTRY[cls.name] = cls
    return cls


def get_engine(name: str) -> OcrEngine:
    cls = ENGINE_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"OCR engine {name!r} not registered. Known: {list(ENGINE_REGISTRY)}")
    return cls()


# ── engine → GUI log bridge ───────────────────────────────────────────
#
# Engines drop diagnostic lines via ``engine_log(text, level=)``. By
# default it just prints (matches the CLI batch flow). When OcrWorker
# is running it swaps in a callback that emits the line through its
# ``log_line`` Qt signal so the GUI's Log tab gets the full story —
# previously every ``print()`` in surya.py / paddle_vl.py disappeared
# into terminal stdout, which under .app bundles == /dev/null.

from typing import Callable as _Callable

_LOG_SINK: _Callable[[str, str], None] | None = None


def set_engine_log_sink(fn: _Callable[[str, str], None] | None) -> None:
    """Install (or clear) the callback engines route diagnostics through.

    ``fn(level, text)`` — ``level`` is one of ``"info"`` / ``"warn"`` /
    ``"error"``. OcrWorker registers this at the top of ``run()`` and
    clears it on exit. Callers outside that path see plain prints.
    """
    global _LOG_SINK
    _LOG_SINK = fn


def engine_log(text: str, level: str = "info") -> None:
    sink = _LOG_SINK
    if sink is not None:
        try:
            sink(level, text)
            return
        except Exception:
            # Don't let a misbehaving sink hide the line — fall back to
            # stdout so the diagnostic still surfaces somewhere.
            pass
    # Keep prints flush-on for CLI users tailing the log.
    print(text, flush=True)


# ── unified DPI knob ──────────────────────────────────────────────────
#
# All OCR engines downsample the page image to ``OCR_DPI`` before
# inference. Lives here (not in each engine) so the UI picker, env
# override, and DB key live in exactly one place.

import os as _os


def resolve_ocr_dpi(default: int = 150) -> int:
    """Resolve the user-picked OCR target DPI.

    Lookup order: env var ``AGLAIA_OCR_DPI`` → SQLite config
    (``KEY_OCR_DPI``) → ``default``.
    """
    env = _os.environ.get("AGLAIA_OCR_DPI", "").strip()
    if env:
        try:
            v = int(env)
            if v > 0:
                return v
        except ValueError:
            pass
    try:
        from aglaia.app_data import db as _cfg
        with _cfg.session() as _conn:
            return int(_cfg.get(_conn, _cfg.KEY_OCR_DPI, default) or default)
    except Exception:
        return default


def resolve_confidence_gate(default: float = 0.7) -> float:
    """Resolve the per-line confidence gate for the apple_docs complement.

    Lookup order: env var ``AGLAIA_OCR_CONFIDENCE_GATE`` → SQLite config
    (``KEY_OCR_CONFIDENCE_GATE``) → ``default``. Clamped to (0, 1].
    """
    def _clamp(v: float) -> float:
        if v <= 0.0:
            return default
        return min(v, 1.0)

    env = _os.environ.get("AGLAIA_OCR_CONFIDENCE_GATE", "").strip()
    if env:
        try:
            return _clamp(float(env))
        except ValueError:
            pass
    try:
        from aglaia.app_data import db as _cfg
        with _cfg.session() as _conn:
            raw = _cfg.get(_conn, _cfg.KEY_OCR_CONFIDENCE_GATE, default)
            return _clamp(float(raw if raw is not None else default))
    except Exception:
        return default


def downsample_to_dpi(image_rgb, src_dpi, target_dpi: int):
    """Resize ``image_rgb`` so its effective DPI lands on ``target_dpi``.

    When ``src_dpi`` is unknown (0/None) we fall back to a longest-edge
    budget — assume a book page never exceeds ~12 inches on its long
    edge and clamp the longest pixel dimension to ``target_dpi × 12``.
    Without this fallback ``images.dpi = 0`` rows (common for older
    Aglaïa projects) silently skipped resize → engines saw the raw
    multi-megapixel capture and inflated wall-clock 5–10×.

    No-op when source already at/below the resolved target or within
    10 % of it. Uses INTER_AREA (quality-preserving shrink)."""
    try:
        src = float(src_dpi or 0)
    except (TypeError, ValueError):
        src = 0
    if target_dpi <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if src > 0:
        if src <= target_dpi * 1.1:
            return image_rgb
        scale = target_dpi / src
    else:
        # Fallback: clamp longest edge to ``target_dpi × LONGEST_INCHES``.
        # 12" covers any A4/Letter/Legal page in either orientation.
        LONGEST_INCHES = 12.0
        longest_px = max(w, h)
        longest_budget = int(target_dpi * LONGEST_INCHES)
        if longest_px <= longest_budget * 1.1:
            return image_rgb
        scale = longest_budget / longest_px
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    import cv2 as _cv2
    return _cv2.resize(image_rgb, (new_w, new_h),
                        interpolation=_cv2.INTER_AREA)


# Side-effect registration lives in aglaia/workers/ocr/__init__.py to
# avoid double-importing each engine module.
