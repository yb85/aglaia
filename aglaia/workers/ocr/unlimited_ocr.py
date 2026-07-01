# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Baidu Unlimited-OCR — DeepSeek-OCR stack + R-SWA multipage attention, local.

A whole-document engine (like ``mistral_cloud``): it runs the FUSED multipage
path — all pages placed at one ``<image>`` position, one "Multi page parsing."
generation with the R-SWA ring-buffer KV cache — and returns one ``OcrResult``
per page. The fused stream has no page delimiter, so we split it back by the
per-page 0..999 coordinate resets (see ``unlimited_backend.segment_pages``).

Runs IN-PROCESS via the upstream ``mlx_vlm`` ``unlimited_ocr`` model (git-pinned
in ``pyproject.toml``) against a hybrid-precision MLX weight dir (F32 vision +
4-bit LLM) from the ``unlimited-ocr-mlx`` converter — NOT through the shared
``LocalVlmServer`` (that path is for glm/paddle). Apple-Silicon only; the CUDA
path is a later slice.
"""

from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from aglaia.app_data.downloads import (
    DownloadTarget,
    is_downloaded,
    models_dir,
    register_download,
    target_for,
)

from . import unlimited_backend as ub
from .engine import OcrEngine, OcrResult, register

# Pre-converted hybrid-precision q4 weights (F32 vision + 4-bit LLM, ~3.7 GB),
# produced by ``unlimited-ocr-mlx`` and published as an HF snapshot. The
# converter lives in ../unlimited-ocr-mlx; on-device conversion is a later
# option. NB: ``url`` is the publish target — update it to the real repo id
# once the q4 dir is uploaded.
register_download(
    DownloadTarget(
        key="unlimited_ocr_mlx",
        title="Unlimited-OCR (MLX q4)",
        filename="Unlimited-OCR-mlx-q4",
        # TODO(publish): point at the uploaded hybrid-precision q4 snapshot.
        url="aglaia/Unlimited-OCR-mlx-q4",
        approx_size_mb=3700,
        kind="hf-snapshot",
        section="other",
        purpose="OCR",
        project="baidu/Unlimited-OCR",
        platform="darwin-arm64",
        registered_by="unlimited_ocr",
    )
)


@register
class UnlimitedOcrEngine(OcrEngine):
    name = "unlimited"
    display = "Unlimited-OCR (local)"
    description = "Local doc VLM; fused multipage (R-SWA), Markdown + tables."
    mlx_target_key = "unlimited_ocr_mlx"

    # Big one-off model load → drive the OcrWorker "loading…/VLM OCR" hint.
    served_vlm = True
    # Grounded per-span bboxes → eligible to OCR a cropped block directly.
    direct_block = True

    # Output token budget PER PAGE for the fused multipage path. R-SWA
    # (RingSlidingKVCache) keeps the full image prefill but ring-buffers decode
    # KV → output length is UNBOUNDED, so we must NOT cap at a fixed 32768 (that
    # truncates docs past ~13 dense pages). Scale the cap with page count; dense
    # grounded pages run ~2000-2600 output tokens. A generous per-page budget
    # avoids clipping the last pages while still bounding a runaway.
    PER_PAGE_TOKEN_BUDGET = 3000

    # Pages per fused R-SWA window. DEFAULT 1 (per-page) after measuring the
    # multipage path: fusing 2+ pages triggers ERRATIC repetition loops (a page
    # blows up to ~30k words, penalty-resistant — raising repetition_penalty
    # made it WORSE), whereas single-page base-mode decode is clean, fast
    # (~10 s/page vs ~39 s/page fused), and accurate (0.90 word-overlap vs
    # Mistral on the athanase corpus). The fused multipage/R-SWA path is real
    # but numerically unstable in this mlx-vlm build, so it's OPT-IN: set
    # window>1 via the ``window`` spec param or AGLAIA_UNLIMITED_WINDOW. Per-page
    # also sidesteps the 32768 position-window cap on huge docs (delbrel's 333).
    WINDOW_SIZE = 1

    def __init__(self) -> None:
        self._loaded: tuple[Any, Any] | None = None
        self._max_tokens = 32768  # floor / single-page (gundam) cap
        self._window_size = self._env_window(self.WINDOW_SIZE)
        self.available = self._backend_ready()

    @staticmethod
    def _env_window(default: int) -> int:
        import os
        try:
            v = int(os.environ.get("AGLAIA_UNLIMITED_WINDOW", "") or default)
            return v if v > 0 else default
        except ValueError:
            return default

    @staticmethod
    def _backend_ready() -> bool:
        """Selectable when this is arm64 macOS, mlx_vlm is importable, AND the
        weights are downloaded (mirrors the glm/paddle ``_weights_ready`` gate)."""
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            return False
        try:
            import importlib.util
            if importlib.util.find_spec("mlx_vlm") is None:
                return False
        except Exception:
            return False
        return is_downloaded("unlimited_ocr_mlx")

    def configure(self, params: dict[str, str]) -> None:
        mt = params.get("max_tokens")
        if mt:
            try:
                self._max_tokens = int(mt)
            except ValueError:
                pass
        w = params.get("window")
        if w:
            try:
                iw = int(w)
                if iw > 0:
                    self._window_size = iw
            except ValueError:
                pass

    # ── model lifecycle ──────────────────────────────────────────────────
    def _model_path(self) -> str:
        key = self.mlx_target_key
        if not key or not is_downloaded(key):
            tgt = target_for(key)
            name = tgt.title if tgt else key
            raise RuntimeError(
                f"Unlimited-OCR weights ('{name}') are not downloaded. "
                f"Open the Model Downloader and install '{key}'."
            )
        tgt = target_for(key)
        assert tgt is not None
        return str(models_dir() / tgt.filename)

    def warmup(self, languages: list[str] | None = None) -> None:
        if self._loaded is None:
            self._loaded = ub.load_model(self._model_path())

    def _ensure(self) -> tuple[Any, Any]:
        if self._loaded is None:
            self.warmup()
        assert self._loaded is not None
        return self._loaded

    # ── whole-document (windowed fused multipage) entry point ────────────
    def recognize_rows(self, rows, languages: list[str]) -> list[OcrResult]:
        """Windowed multipage OCR: process pages in fused R-SWA windows of
        ``window_size`` (default 4), split each window back into per-page
        ``OcrResult``s by coordinate resets, concatenate in order.

        One fused call over ALL pages is impractical past a handful (per-token
        cost grows with the image prefill) and impossible past the 32768
        position window — so we window. ``doc_markdown`` (the full document) is
        stitched from every window and attached to page 0.
        """
        rows = list(rows)
        n = len(rows)
        if n == 0:
            return []
        model, processor = self._ensure()
        w = max(1, self._window_size)

        results: list[OcrResult] = []
        for start in range(0, n, w):
            chunk = rows[start:start + w]
            results.extend(self._recognize_window(model, processor, chunk,
                                                   languages))
        doc_md = "\n\n".join(r["meta"]["markdown"] for r in results
                             if r["meta"].get("markdown")).strip()
        if results:
            results[0]["meta"]["doc_markdown"] = doc_md
        return results

    def _recognize_window(self, model: Any, processor: Any, chunk,
                          languages: list[str]) -> list[OcrResult]:
        """One fused R-SWA generation over ``chunk`` pages → per-page results."""
        n = len(chunk)
        dims = [(int(r.get("width") or 0), int(r.get("height") or 0))
                for r in chunk]
        with tempfile.TemporaryDirectory(prefix="aglaia-unlimited-") as td:
            paths: list[str] = []
            for i, r in enumerate(chunk):
                p = Path(td) / f"page{i:04d}.img"
                p.write_bytes(r["blob"])
                paths.append(str(p))
            # Scale the cap with the window's page count (R-SWA → unbounded
            # output); the 32768 floor already covers a 4-page window.
            max_tokens = max(self._max_tokens, n * self.PER_PAGE_TOKEN_BUDGET)
            raw = ub.generate_text(
                model, processor, paths,
                multi_page=True, max_tokens=max_tokens,
            )
        spans = ub.parse_spans(raw)
        pages = ub.segment_pages(spans, n)
        out: list[OcrResult] = []
        for (ww, hh), grp in zip(dims, pages):
            pp = ub.build_page(grp, ww, hh)
            out.append({
                "engine": self.name, "languages": list(languages),
                "page_w": int(ww), "page_h": int(hh),
                "lines": pp.lines,
                "meta": {"source": "unlimited", "markdown": pp.markdown,
                         "spans": pp.spans},
            })
        return out

    # ── single-page entry point (abstract-method contract) ───────────────
    def recognize(self, image_rgb, languages: list[str],
                  *, src_dpi: float | None = None) -> OcrResult:
        """Single page via gundam (high-res crop) mode. The whole point of the
        engine is the fused ``recognize_rows`` path; this covers the direct /
        complement one-image case."""
        import cv2  # local: keep cv2 off the import path for headless-no-cv2

        model, processor = self._ensure()
        h, w = image_rgb.shape[:2]
        with tempfile.TemporaryDirectory(prefix="aglaia-unlimited-") as td:
            p = Path(td) / "page.png"
            cv2.imwrite(str(p), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            raw = ub.generate_text(
                model, processor, [str(p)],
                multi_page=False, max_tokens=self._max_tokens,
            )
        pp = ub.build_page(ub.parse_spans(raw), int(w), int(h))
        return {
            "engine": self.name, "languages": list(languages),
            "page_w": int(w), "page_h": int(h),
            "lines": pp.lines,
            "meta": {"source": "unlimited", "markdown": pp.markdown,
                     "spans": pp.spans},
        }
