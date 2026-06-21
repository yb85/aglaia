# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Surya OCR engine.

Wires the registry entry for the UI. The current Surya release (2.x)
runs entirely through a `llama-server` subprocess speaking the OpenAI
chat-completions protocol — the old in-process safetensors path is
gone. To run Surya in Aglaïa today the host needs:

  * the `llama-server` binary on $PATH (``brew install llama.cpp``),
  * the GGUF variant of the model
    (``datalab-to/surya-ocr-2-gguf`` on Hugging Face), which the
    backend pulls via ``huggingface_hub`` on first use.

The safetensors snapshot the in-app downloader currently fetches
(`datalab-to/surya-ocr-2`) is NOT directly usable here — it's kept on
disk so the file-presence check still passes, but ``recognize`` will
spawn `llama-server` and let it download the GGUF separately.

If the llama-server binary is missing we surface a clear actionable
error message instead of swallowing the failure as a 0-line OCR pass.
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .engine import (
    OcrEngine, OcrResult, OcrLine, register,
    resolve_ocr_dpi as _target_dpi, downsample_to_dpi as _downsample,
    engine_log as _log,
)


def _llama_server_subdir() -> Optional[str]:
    """Mirrors the build-time subdir picker in ``Aglaia.spec`` —
    keep the two in sync."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if sys_name == "darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x64"
    if sys_name == "linux":
        return "linux-arm64" if machine in ("arm64", "aarch64") else "linux-x64"
    if sys_name == "windows":
        return "windows-arm64" if machine in ("arm64", "aarch64") else "windows-x64"
    return None


def _llama_server_binary_name() -> str:
    return "llama-server.exe" if platform.system().lower() == "windows" else "llama-server"


def _resolve_llama_server() -> Optional[str]:
    """Returns the absolute path to ``llama-server`` from, in order:

      1. ``$LLAMA_CPP_BINARY`` override,
      2. PyInstaller bundle (``sys._MEIPASS``) — the .app ships its own
         copy next to the main executable,
      3. ``vendor/llama-server/<plat>/`` in a dev checkout (populated
         by ``scripts/fetch_llama_server.py``),
      4. ``$PATH`` (e.g. ``brew install llama.cpp``).
    """
    override = os.environ.get("LLAMA_CPP_BINARY", "").strip()
    if override and os.path.isfile(override):
        return override

    name = _llama_server_binary_name()

    # Frozen: PyInstaller puts bundled binaries at MEIPASS root.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / name
        if cand.is_file():
            return str(cand)

    # Dev checkout: ``vendor/llama-server/<plat>/llama-server[.exe]``
    subdir = _llama_server_subdir()
    if subdir is not None:
        repo_root = Path(__file__).resolve().parents[3]
        cand = repo_root / "vendor" / "llama-server" / subdir / name
        if cand.is_file():
            return str(cand)

    found = shutil.which("llama-server")
    return found


def _llama_server_available() -> bool:
    return _resolve_llama_server() is not None


_GGUF_MMPROJ = "surya-2-mmproj.gguf"
# Candidate weight files, ordered by *desired* speed → fall back to the
# upstream FP16 only when no quantized variant has been generated.
# Produce a Q4_K_M with:
#   ./vendor/llama-server/<plat>/llama-quantize \
#     <models>/surya-ocr-2-gguf/surya-2.gguf \
#     <models>/surya-ocr-2-gguf/surya-2-Q4_K_M.gguf Q4_K_M
_GGUF_MODEL_CANDIDATES = (
    "surya-2-Q4_K_M.gguf",   # ~390 MB · ~2-3× faster · negligible quality drop
    "surya-2-Q5_K_M.gguf",   # ~470 MB · ~1.8-2.3× faster · near-zero loss
    "surya-2-Q8_0.gguf",     # ~660 MB · ~1.4-1.7× faster · indistinguishable
    "surya-2.gguf",          # 1.27 GB · upstream FP16 baseline
)
_GGUF_MODEL = _GGUF_MODEL_CANDIDATES[-1]  # back-compat for old probes


def _surya_weights_dir() -> Optional[Path]:
    """Folder that should hold a Surya GGUF + ``surya-2-mmproj.gguf``
    after a successful model download.

    Lookup order — first non-empty hit wins:
      1. ``aglaia-models/surya-ocr-2-Q4_K_M-gguf`` (current canonical mirror)
      2. ``surya-ocr-2-gguf`` (legacy directory from the upstream
         ``datalab-to/surya-ocr-2-gguf`` snapshot — kept so existing
         installs don't re-download).
    """
    try:
        from lib.app_data import models_dir
    except Exception:
        return None
    base = models_dir()
    for name in ("surya-ocr-2-Q4_K_M-gguf", "surya-ocr-2-gguf"):
        d = base / name
        if d.is_dir():
            return d
    # Even if nothing exists yet, return the canonical (new) path so the
    # downloader knows where to plant freshly-fetched files.
    return base / "surya-ocr-2-Q4_K_M-gguf"


def _surya_model_path() -> Optional[Path]:
    """Return the fastest GGUF model variant present on disk. Iterates
    ``_GGUF_MODEL_CANDIDATES`` (quantized → FP16) so a freshly-quantized
    drop-in is preferred without code changes."""
    d = _surya_weights_dir()
    if d is None or not d.is_dir():
        return None
    for name in _GGUF_MODEL_CANDIDATES:
        f = d / name
        if f.is_file():
            return f
    return None


def _weights_present() -> bool:
    """Surya 2.x runs via llama.cpp, which needs the GGUF format. Both
    a model weight (any of ``_GGUF_MODEL_CANDIDATES``) and the mmproj
    file must be on disk before we report Surya as available."""
    d = _surya_weights_dir()
    if d is None or not d.is_dir():
        return False
    if not (d / _GGUF_MMPROJ).is_file():
        return False
    return _surya_model_path() is not None


@register
class SuryaEngine(OcrEngine):
    """Surya OCR — VLM via local llama.cpp server."""

    name = "surya"
    display = "Surya"
    description = ("VLM OCR: titles, lists, tables, handwriting, "
                   "90+ scripts. ~1.3 GB weights.")
    # Default DPI the OcrTab card seeds when the user hasn't picked one.
    # Datalab's surya-2 blog post documents 96 DPI as the sweet spot;
    # 150 stays well inside their tolerance while leaving fine glyph
    # detail untouched. 300+ wastes tokens with no quality gain.
    default_dpi: int = 150

    # Cached predictor + manager so successive scans reuse the same
    # llama-server process (spinning one up per call would dominate
    # wall-clock).
    _manager = None
    _predictor = None

    def __init__(self) -> None:
        # Surface as "available" only when both the binary and the
        # safetensors snapshot are present. The downloader UI keys off
        # this flag — false here = "Download Surya model" button shown.
        self.available = _weights_present() and _llama_server_available()

    # ── one-time setup ─────────────────────────────────────────────
    @classmethod
    def _ensure_ready(cls) -> None:
        """Spin up the llama-server-backed predictor exactly once per
        process. Idempotent — second call is a no-op. Centralising
        every settings mutation here keeps the per-page code path
        trivial and rules out the kind of drift that just shipped a
        bug where ``recognize_batch`` skipped ``src_dpi`` plumbing the
        ``recognize`` path did via copy-paste."""
        if cls._predictor is not None:
            return
        if not _weights_present():
            raise RuntimeError(
                "Surya weights not downloaded. Open the Model Downloader."
            )
        if not _llama_server_available():
            raise RuntimeError(
                "Surya needs the llama.cpp server binary. Install with\n"
                "  brew install llama.cpp\n"
                "or set LLAMA_CPP_BINARY to the binary's path."
            )

        # Lazy imports — keep startup fast when the user isn't using Surya.
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings as _surya_settings

        # ── llama.cpp tuning (bench-winning defaults) ────────────
        # Surya's defaults (8 parallel × 12288 ctx_per_slot, n_threads
        # = 4) were sized for 24+ GB Linux boxes. M-series base SKUs
        # need a tighter budget; the values below match the wall-
        # clock winner from ``bench_surya_llamacpp.py``.
        try:
            cpu_count = max(os.cpu_count() or 4, 4)
            extra_args = _surya_settings.LLAMA_CPP_EXTRA_ARGS or ""
            additions: list[str] = []
            if "-t " not in extra_args and "--threads" not in extra_args:
                additions.append(f"-t {cpu_count}")
            if "-ctk" not in extra_args:
                additions.append("-ctk q8_0")
            if "-ctv" not in extra_args:
                additions.append("-ctv q8_0")
            if "-b " not in extra_args and "--batch-size" not in extra_args:
                additions.append("-b 4096")
            if "-ub " not in extra_args and "--ubatch-size" not in extra_args:
                additions.append("-ub 2048")
            # Newer llama.cpp (>= b4500-ish) requires an explicit
            # value: ``-fa on/off/auto``. Bare ``-fa`` fails arg
            # parse → server dies at boot → RecognitionPredictor
            # hangs on health.
            if "-fa" not in extra_args and "--flash-attn" not in extra_args:
                additions.append("-fa on")
            # Bench winner: mmproj on Metal GPU = -12 % wall.
            try:
                _surya_settings.LLAMA_CPP_NO_MMPROJ_OFFLOAD = False
            except Exception:
                pass
            if additions:
                _surya_settings.LLAMA_CPP_EXTRA_ARGS = (
                    " ".join([extra_args, *additions]).strip()
                )
        except Exception:
            pass

        # Resolve weight path EARLY — the slot picker needs the actual
        # file size for quantization-aware budgeting.
        model_path = _surya_model_path()
        try:
            from lib.app_data import db as _cfg
            from .auto_workers import (
                resolve_worker_count, DEFAULT_MODEL_MB,
                DEFAULT_KV_PER_SLOT_MB,
            )
            with _cfg.session() as _conn:
                user_workers = int(
                    _cfg.get(_conn, _cfg.KEY_OCR_WORKERS, 0) or 0
                )
            quant_model_mb = DEFAULT_MODEL_MB
            kv_mb = DEFAULT_KV_PER_SLOT_MB
            extras = _surya_settings.LLAMA_CPP_EXTRA_ARGS or ""
            kv_q8 = "-ctk q8_0" in extras and "-ctv q8_0" in extras
            if model_path is not None:
                try:
                    file_mb = max(
                        1, model_path.stat().st_size // (1024 * 1024)
                    )
                    quant_model_mb = int(file_mb * 1.1) + 250  # +mmproj
                except OSError:
                    pass
                if kv_q8:
                    kv_mb = max(300, DEFAULT_KV_PER_SLOT_MB // 2)
            slots, reason = resolve_worker_count(
                user_workers,
                model_mb=quant_model_mb,
                kv_per_slot_mb=kv_mb,
            )
            if not os.environ.get("SURYA_INFERENCE_PARALLEL"):
                _surya_settings.SURYA_INFERENCE_PARALLEL = slots
            # Bench winner: 8192 per slot.
            if not os.environ.get("SURYA_INFERENCE_CTX_PER_SLOT"):
                _surya_settings.SURYA_INFERENCE_CTX_PER_SLOT = 8192
            # Cap the per-inference timeout. Surya's default is 600 s
            # (10 min); a degenerate tiny / near-blank line crop can make
            # the VLM ramble to its token cap and stall the whole OCR run
            # for minutes. 180 s comfortably covers a dense full page while
            # aborting a runaway single image ~3× sooner. Override with
            # AGLAIA_SURYA_TIMEOUT_S (or Surya's own env).
            if not os.environ.get("SURYA_INFERENCE_TIMEOUT_SECONDS"):
                _surya_settings.SURYA_INFERENCE_TIMEOUT_SECONDS = float(
                    os.environ.get("AGLAIA_SURYA_TIMEOUT_S", "180"))
            cls.workers_chosen = slots
            cls.workers_reason = reason
        except Exception as _e:
            import traceback as _tb
            _log(
                f"[surya] auto-worker tuning failed: {_e}\n"
                f"{_tb.format_exc()}",
                level="warn",
            )

        bin_path = _resolve_llama_server()
        if bin_path:
            os.environ["LLAMA_CPP_BINARY"] = bin_path
            try:
                _surya_settings.LLAMA_CPP_BINARY = bin_path
            except Exception:
                pass

        if model_path is not None:
            os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = str(model_path)
            try:
                _surya_settings.SURYA_GGUF_LOCAL_MODEL_PATH = str(model_path)
            except Exception:
                pass
            cls.weights_chosen = model_path.name

        weights_dir = _surya_weights_dir()
        if weights_dir is not None:
            mmproj_path = weights_dir / _GGUF_MMPROJ
            if mmproj_path.is_file():
                os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = str(mmproj_path)
                try:
                    _surya_settings.SURYA_GGUF_LOCAL_MMPROJ_PATH = (
                        str(mmproj_path)
                    )
                except Exception:
                    pass

        chosen = (model_path.name if model_path else "<missing>")
        extras = _surya_settings.LLAMA_CPP_EXTRA_ARGS or "<none>"
        mmproj_gpu = not bool(
            getattr(_surya_settings,
                      "LLAMA_CPP_NO_MMPROJ_OFFLOAD", False)
        )
        _log(
            f"[surya] weights={chosen}  "
            f"parallel={_surya_settings.SURYA_INFERENCE_PARALLEL}  "
            f"ctx/slot={_surya_settings.SURYA_INFERENCE_CTX_PER_SLOT}  "
            f"mmproj_on_gpu={mmproj_gpu}  "
            f"extra=[{extras}]"
        )
        # First call here blocks until the server is up. Minutes the
        # very first time (huggingface_hub fetches GGUF), seconds after.
        cls._predictor = RecognitionPredictor()

    # ── public API ─────────────────────────────────────────────────
    def recognize(self, image_rgb: np.ndarray,
                  languages: List[str],
                  *, src_dpi: float | None = None) -> OcrResult:
        """Thin wrapper — funnels through ``recognize_batch`` so the
        DPI / downsample / predictor path lives in exactly one place."""
        return self.recognize_batch(
            [image_rgb], languages,
            src_dpis=[src_dpi] if src_dpi is not None else None,
        )[0]

    def recognize_batch(self, images_rgb: list[np.ndarray],
                         languages: List[str],
                         *,
                         src_dpis: list[float] | None = None
                         ) -> list[OcrResult]:
        """Batched OCR — drives llama-server's ``--parallel`` slots.
        Single-image ``recognize()`` calls funnel through here so the
        DPI / log / predict path lives in exactly one branch."""
        if not images_rgb:
            return []
        SuryaEngine._ensure_ready()

        from PIL import Image as _Image
        target_dpi = _target_dpi()
        if src_dpis is None:
            src_dpis = [0.0] * len(images_rgb)
        scaled = [_downsample(img, dpi or 0, target_dpi)
                  for img, dpi in zip(images_rgb, src_dpis)]
        pils = [_Image.fromarray(img) for img in scaled]

        # Per-image dims log — surfaces what Surya is actually being
        # asked to crunch, separated by /-the/-arrows so an in-flight
        # batch is recognisable in the GUI log.
        dims_str = ", ".join(
            f"{img.shape[1]}x{img.shape[0]}" for img in scaled
        )
        src_str = ", ".join(f"{d or 0:.0f}" for d in src_dpis)
        import time as _t
        t0 = _t.monotonic()
        _log(
            f"[surya] → batch n={len(pils)} target_dpi={target_dpi} "
            f"src_dpi=[{src_str}] dims=[{dims_str}]"
        )
        # ``full_page=True`` makes Surya do one HIGH_ACCURACY_BBOX pass
        # over the entire page instead of running an additional
        # LayoutPredictor — halves the VLM round-trips.
        ocr_results = SuryaEngine._predictor(pils, full_page=True)
        wall = _t.monotonic() - t0
        _log(
            f"[surya] ← batch n={len(pils)} in {wall:.1f}s "
            f"({wall/max(1,len(pils)):.1f}s/img)"
        )
        return [_to_ocr_result(img, page)
                for img, page in zip(scaled, ocr_results)]


def _to_ocr_result(image_rgb: np.ndarray, page: Any) -> OcrResult:
    """Translate a ``PageOCRResult`` into the engine-agnostic
    ``OcrResult`` shape consumed by the rest of the pipeline."""
    h, w = image_rgb.shape[:2]
    lines: list[OcrLine] = []
    structure: list[dict] = []  # per-block label / html for MD export

    for block in getattr(page, "blocks", []) or []:
        polygon = getattr(block.polygon, "polygon", None) or []
        bbox = _bbox_from_polygon(polygon, w, h)
        text = _strip_html(getattr(block, "html", "") or "")
        confidence = float(getattr(block, "confidence", 0.0) or 0.0)
        if text:
            line: OcrLine = {
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
            }
            if polygon and len(polygon) == 4:
                line["quad"] = [(float(x), float(y)) for x, y in polygon]
            lines.append(line)

        structure.append({
            "label": getattr(block, "label", None),
            "raw_label": getattr(block, "raw_label", None),
            "reading_order": getattr(block, "reading_order", None),
            "html": getattr(block, "html", "") or "",
            "bbox": list(bbox),
        })

    return {
        "engine": "surya",
        "languages": [],
        "page_w": int(w),
        "page_h": int(h),
        "lines": lines,
        "meta": {
            "structure": structure,
            "ocr_dpi": int(_target_dpi() or 0),
        },
    }


def _bbox_from_polygon(polygon, w: int, h: int) -> tuple[int, int, int, int]:
    if not polygon:
        return (0, 0, 0, 0)
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x0 = max(0, int(round(min(xs))))
    y0 = max(0, int(round(min(ys))))
    x1 = min(w, int(round(max(xs))))
    y1 = min(h, int(round(max(ys))))
    return (x0, y0, x1, y1)


def _strip_html(html: str) -> str:
    """Tiny HTML→text reduction (no external dep). Surya emits
    well-formed snippets like ``<h1>Title</h1>`` and ``<li>item</li>``
    — a regex strip is sufficient for the plain-text OcrLine.text
    field; the original HTML stays in ``meta.structure`` for callers
    that want titles / lists for MD export."""
    if not html:
        return ""
    import re
    # Drop tag soup but keep visible text. Collapse runs of whitespace.
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text
