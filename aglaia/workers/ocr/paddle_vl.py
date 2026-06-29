# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PaddleOCR-VL OCR engine.

Runs the full PaddleOCR-VL 1.5 document-parsing pipeline locally on Apple
Silicon. The stack splits in two:

* ``mlx_vlm.server`` hosts the OCR VLM weights and serves an
  OpenAI-compat ``/v1/chat/completions`` endpoint on
  ``http://127.0.0.1:<PADDLE_PORT>``. We spawn it lazily on the first
  ``recognize()`` call and keep it alive for the rest of the process so
  successive scans reuse the loaded weights.
* ``paddleocr.PaddleOCRVL`` is the orchestrator — it runs PP-DocLayoutV2
  for region detection, routes per-block crops to the VLM (text), the
  table recogniser, the formula recogniser, then assembles the result
  into Markdown.

Bench numbers (M4 base, 8 PNGs at 100 DPI): ~6.5 s/page wall-clock with
real MD output (titles, lists, formulas, tables). 2.4× faster than the
tuned Surya stack on the same images.

Weights ship via the standard model downloader as
``mlx-community/PaddleOCR-VL-1.5-4bit`` (~720 MB). Python deps (paddleocr
+ paddlepaddle CPU + mlx-vlm) are listed in ``pyproject.toml``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from aglaia.workers.vlm import LocalVlmServer, MlxBackend

from .engine import (
    OcrEngine, OcrResult, OcrLine, register,
    resolve_ocr_dpi as _target_dpi, downsample_to_dpi as _downsample,
    engine_log as _log,
)


# ── weight resolution (downloader-managed) ─────────────────────────────

def _paddle_weights_dir() -> Optional[Path]:
    """Folder where the model downloader plants the MLX snapshot.

    Mirrors Surya's resolution but points at the Paddle subdir name from the
    ``paddle_vl`` download target (see ``aglaia/app_data/downloads.py``)."""
    try:
        from aglaia.app_data import models_dir
        return models_dir() / "PaddleOCR-VL-1.5-4bit"
    except Exception:
        return None


# Files the downloader is required to land before we report the engine
# as available. Keep in sync with the ``paddle_vl`` target's ``required_files``
# in ``aglaia/app_data/downloads.py``.
_REQUIRED_FILES = (
    "model.safetensors",
    "processor_config.json",
    "config.json",
    "tokenizer.json",
)


def _weights_present() -> bool:
    d = _paddle_weights_dir()
    if d is None or not d.is_dir():
        return False
    return all((d / f).is_file() for f in _REQUIRED_FILES)


def _python_deps_present() -> bool:
    """Both ``paddleocr`` and ``mlx_vlm`` must import for the engine to
    work. We probe by module name (cheaper than ``importlib.util``)."""
    try:
        import paddleocr  # noqa: F401
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        return False


# ── engine ────────────────────────────────────────────────────────────

@register
class PaddleVlEngine(OcrEngine):
    """PaddleOCR-VL — VLM + PP-DocLayout pipeline producing native MD."""

    name = "paddle_vl"
    display = "PaddleOCR-VL"
    description = (
        "0.9B VLM, 109 langs. Markdown with tables + formulas. ~720 MB."
    )
    # We always feed the pipeline a PIL image — DPI is informational, not
    # used by the pipeline orchestrator. Surface a default anyway so the
    # OcrTab can render a unified DPI picker.
    default_dpi: int = 150

    # Class-level so all OcrEngine instances share the loaded pipeline.
    _pipeline = None
    weights_chosen: str = ""

    def __init__(self) -> None:
        self.available = _weights_present() and _python_deps_present()

    # ── private: orchestrator init ─────────────────────────────────
    @classmethod
    def _ensure_pipeline(cls) -> Any:
        if cls._pipeline is not None:
            return cls._pipeline
        if not _python_deps_present():
            raise RuntimeError(
                "Paddle Python deps missing. Install with:\n"
                "  uv pip install paddlepaddle==3.2.1 "
                "--index-url=https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
                "  uv pip install 'paddleocr[doc-parser]' 'mlx-vlm>=0.3.11'"
            )

        # Patch known incompatibilities in the downloaded model files
        # before the mlx_vlm server tries to load them.
        _patch_paddle_model_files(_paddle_weights_dir())

        # PaddleOCR-VL ships MLX-only 4-bit weights, so pin the MLX backend
        # (the generic picker would otherwise prefer vLLM on a CUDA box, which
        # can't load these weights). The shared LocalVlmServer handles spawn /
        # health / teardown; we route its log through the OCR log channel.
        model_name = str(_paddle_weights_dir())
        base_url = LocalVlmServer.ensure(
            model_name, backend=MlxBackend(), log=_log, log_stem="paddle",
        )
        from paddleocr import PaddleOCRVL
        # ``vl_rec_api_model_name`` must match the model the server PRELOADED —
        # not whatever is first in ``/v1/models`` (the server leaks every model
        # it ever cached, e.g. a stale Falcon-OCR from a prior bench run, which
        # would trigger a hot-swap → crash on Paddle-shaped vision tokens). Pin
        # to the local path the server was spawned with.
        # ``vl_rec_max_concurrency=1`` is mandatory with the
        # mlx-vlm-server backend: paddleocr fans out one HTTP request
        # per layout region, and mlx-vlm-server's continuous batcher
        # concatenates concurrent prompts on axis 0. When two region
        # crops have different image dims (the common case for full
        # pages — captions vs. paragraphs vs. tables) the concat fails
        # with ``[concatenate] All the input array dimensions must
        # match exactly`` and the whole page errors out. Serialising
        # the fan-out avoids the batch entirely. Throughput cost is
        # minor because the regions are small and the prefill cache
        # warms across them.
        cls._pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            vl_rec_backend="mlx-vlm-server",
            vl_rec_server_url=base_url,
            vl_rec_api_model_name=str(model_name),
            vl_rec_max_concurrency=1,
        )
        cls.weights_chosen = Path(str(model_name)).name
        _log(
            f"[paddle_vl] pipeline ready  url={base_url}  "
            f"model={cls.weights_chosen}"
        )
        return cls._pipeline

    # _query_served_model removed — mlx_vlm.server's ``/v1/models``
    # leaks every model the process has cached (including stale ones
    # from earlier bench runs), so its ``data[0]`` is unreliable. The
    # canonical name is the path we passed to ``--model`` on spawn.

    # ── public API ─────────────────────────────────────────────────
    def recognize(self, image_rgb: np.ndarray,
                  languages: List[str],
                  *, src_dpi: float | None = None) -> OcrResult:
        """Thin wrapper — funnels through ``recognize_batch`` so the
        DPI / downsample / timeout path lives in exactly one place."""
        return self.recognize_batch(
            [image_rgb], languages,
            src_dpis=[src_dpi] if src_dpi is not None else None,
        )[0]

    def recognize_batch(self, images_rgb: list[np.ndarray],
                         languages: List[str],
                         *,
                         src_dpis: list[float] | None = None
                         ) -> list[OcrResult]:
        """Paddle pipeline doesn't expose batched mlx-vlm-server calls
        (each region OCR is a separate HTTP request inside the
        orchestrator). We iterate per-image — same speed as a flat
        call. Single-image ``recognize()`` calls funnel through here
        so the DPI / log / predict path lives in exactly one branch."""
        if not images_rgb:
            return []
        if not _weights_present():
            raise RuntimeError(
                "PaddleOCR-VL weights not downloaded. Open the Model "
                "Downloader and install PaddleOCR-VL-1.5-4bit."
            )
        if not _python_deps_present():
            raise RuntimeError(
                "Paddle Python deps missing — install paddleocr + "
                "paddlepaddle + mlx-vlm."
            )
        pipeline = self._ensure_pipeline()
        target_dpi = _target_dpi()
        if src_dpis is None:
            src_dpis = [0.0] * len(images_rgb)
        import time as _t
        out: list[OcrResult] = []
        for img, dpi in zip(images_rgb, src_dpis):
            scaled = _downsample(img, dpi or 0, target_dpi)
            h, w = scaled.shape[:2]
            t0 = _t.monotonic()
            results = _predict_with_timeout(pipeline, scaled)
            wall = _t.monotonic() - t0
            _log(
                f"[paddle_vl] page {w}x{h} "
                f"(src_dpi={dpi or 0:.0f}, target={target_dpi}) "
                f"→ {wall:.1f}s"
            )
            if not results:
                out.append(_empty_result(scaled))
                continue
            out.append(_to_ocr_result(scaled, results[0]))
        return out




# ── result translation ────────────────────────────────────────────────

_DEFAULT_PREDICT_TIMEOUT_S = 300.0
"""Per-page wall-time budget for paddle ``pipeline.predict()``.

PaddleOCR-VL runs one VLM call **per detected text region**, so a dense
scholarly page (many lines, footnotes, Greek/Latin) can be 30-50 regions
× several seconds each on a 4-bit mlx model — minutes, not seconds. The
old 90 s budget tripped on exactly those pages and, because a timeout
**kills the server and aborts the whole OCR run**, the user got nothing.

300 s gives dense pages room while still catching a genuinely wedged
backend. Override via ``AGLAIA_PADDLE_TIMEOUT`` (seconds); lower the OCR
DPI to trade resolution for speed if pages still time out."""


def _resolve_predict_timeout() -> float:
    env = os.environ.get("AGLAIA_PADDLE_TIMEOUT", "").strip()
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_PREDICT_TIMEOUT_S


def _predict_with_timeout(pipeline: Any, image: np.ndarray) -> Any:
    """Run ``pipeline.predict(image)`` with a hard wall-time budget.

    paddleocr's OpenAI-compat client passes ``timeout=600`` to the HTTP
    layer — fine when the backend is healthy, brutal when it stalls
    (one bad page wedges the whole OCR run for 10 minutes). This wraps
    the call in a background thread + join-with-deadline so a hung
    request becomes a fast ``TimeoutError`` the OcrWorker can react to
    by tearing the whole batch down.
    """
    budget = _resolve_predict_timeout()
    result: list = [None]
    err: list = [None]

    def _runner() -> None:
        try:
            result[0] = pipeline.predict(image)
        except BaseException as e:  # noqa: BLE001 — re-raised in caller
            err[0] = e

    t = threading.Thread(target=_runner, daemon=True,
                          name="paddle_vl-predict")
    t.start()
    t.join(budget)
    if t.is_alive():
        # Best-effort: nuke the mlx_vlm.server so the orphaned HTTP
        # call inside the thread short-circuits. The thread itself
        # stays daemonic and exits on the next interpreter teardown.
        _log(
            f"[paddle_vl] predict exceeded {budget:.0f}s budget — "
            "killing mlx_vlm.server so the OCR batch can abort cleanly",
            level="error",
        )
        try:
            LocalVlmServer.stop(str(_paddle_weights_dir()))
        except Exception:
            pass
        raise TimeoutError(
            f"PaddleOCR-VL predict() exceeded {budget:.0f}s budget"
        )
    if err[0] is not None:
        raise err[0]
    return result[0]


def _patch_paddle_model_files(d: Optional[Path]) -> None:
    """One-shot fixup for known incompatibilities in the model files.

    PaddleOCR-VL 1.5's ``tokenizer_config.json`` declares
    ``"tokenizer_class": "TokenizersBackend"`` — a class that doesn't
    exist in any released ``transformers``. AutoTokenizer raises and
    the patched ``AutoProcessor`` swallows the error, ending up with
    the generic "Unrecognized processing class" failure that crashes
    mlx_vlm.server at startup.

    Stripping the bogus ``tokenizer_class`` lets AutoTokenizer fall
    back to the ``auto_map`` entry (the right path). Idempotent — only
    rewrites the file when the bad value is present.
    """
    if d is None or not d.is_dir():
        return
    p = d / "tokenizer_config.json"
    if not p.is_file():
        return
    import json as _json
    try:
        raw = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    bad = raw.get("tokenizer_class")
    if bad not in {"TokenizersBackend"}:
        return
    # ``PreTrainedTokenizerFast`` is the right replacement — the model
    # ships ``tokenizer.json`` (fast-tokenizers format) and that's the
    # generic class transformers uses when it doesn't have an arch-
    # specific tokenizer. Empirically loads cleanly with auto-map
    # disabled. We DO NOT just drop the key (tried first) because
    # AutoTokenizer then falls back to config.json's ``model_type``
    # which has no ``tokenizer_class`` registered for paddleocr_vl.
    raw["tokenizer_class"] = "PreTrainedTokenizerFast"
    try:
        p.write_text(_json.dumps(raw, indent=2), encoding="utf-8")
        _log(
            f"[paddle_vl] rewrote tokenizer_class {bad!r} → "
            f"'PreTrainedTokenizerFast' in {p.name} "
            f"(compat fix for current transformers)",
            level="warn",
        )
    except Exception:
        pass


def _empty_result(image_rgb: np.ndarray) -> OcrResult:
    h, w = image_rgb.shape[:2]
    return {
        "engine": "paddle_vl",
        "languages": [],
        "page_w": int(w),
        "page_h": int(h),
        "lines": [],
        "meta": {"structure": [], "markdown": ""},
    }


def _to_ocr_result(image_rgb: np.ndarray, page: Any) -> OcrResult:
    """Map a ``PaddleOCRResult`` to the engine-agnostic ``OcrResult``.

    The paddleocr orchestrator builds an internal JSON tree with
    ``parsing_res_list`` (per-block region + label + text/html). We
    surface:

      * ``lines`` — one entry per block, plain text + bbox.
      * ``meta.structure`` — full per-block dict (label, html/md,
        reading_order, bbox) so the MD export can rebuild the page
        without re-OCR-ing.
      * ``meta.markdown`` — the assembled MD string from
        ``save_to_markdown``'s internal renderer. Free; we hold it for
        downstream callers that just want the .md.
    """
    h, w = image_rgb.shape[:2]
    json_doc = _result_to_json(page)
    # PaddleOCR-VL nests the actual fields under a ``res`` key.
    inner = json_doc.get("res") if isinstance(json_doc.get("res"), dict) else json_doc
    blocks = inner.get("parsing_res_list") or []
    if inner.get("width"):
        w = int(inner["width"]) or w
    if inner.get("height"):
        h = int(inner["height"]) or h

    lines: list[OcrLine] = []
    structure: list[dict] = []
    for blk in blocks:
        text = (blk.get("block_content") or "").strip()
        bbox = _bbox_from_paddle(blk.get("block_bbox") or blk.get("bbox"), w, h)
        label = blk.get("block_label") or blk.get("label") or "Text"
        order = blk.get("block_order") or blk.get("order") or blk.get("reading_order")
        if text:
            lines.append({
                "text": text,
                "bbox": bbox,
                "confidence": 1.0,  # paddleocr-vl doesn't surface a per-block score
            })
        structure.append({
            "label": label,
            "reading_order": order,
            "html": blk.get("block_html") or "",
            "markdown": text,
            "bbox": list(bbox),
        })

    # Best-effort: pull the assembled MD string from the result object.
    md = _extract_markdown(page)

    return {
        "engine": "paddle_vl",
        "languages": [],
        "page_w": int(w),
        "page_h": int(h),
        "lines": lines,
        "meta": {
            "structure": structure,
            "markdown": md,
            # Stash the unified DPI the page was downsampled to so MD /
            # PDF exports can tag filenames with the actual OCR DPI.
            "ocr_dpi": int(_target_dpi() or 0),
        },
    }


def _result_to_json(page: Any) -> dict:
    """The paddleocr ``Result`` object exposes its data via ``.json``
    or ``.json_str()`` depending on version; both fall back to dict
    coercion. We're defensive because the API shape has changed across
    paddleocr 2.x → 3.x."""
    for attr in ("json", "to_dict"):
        v = getattr(page, attr, None)
        if isinstance(v, dict):
            return v
        if callable(v):
            try:
                d = v()
                if isinstance(d, dict):
                    return d
            except Exception:
                continue
    try:
        return dict(page)
    except Exception:
        return {}


def _extract_markdown(page: Any) -> str:
    """Pull the rendered MD string without writing to disk. The
    PaddleOCR-VL result object exposes ``markdown`` as a ``{markdown_texts,
    markdown_images, ...}`` dict on recent versions; older versions used
    a plain string."""
    v = getattr(page, "markdown", None)
    if isinstance(v, str) and v:
        return v
    if isinstance(v, dict):
        txt = v.get("markdown_texts") or v.get("text") or ""
        if isinstance(txt, str):
            return txt
    j = _result_to_json(page)
    inner = j.get("res") if isinstance(j.get("res"), dict) else j
    md = inner.get("markdown") or inner.get("md") or ""
    return md if isinstance(md, str) else ""


def _bbox_from_paddle(box: Any, w: int, h: int) -> tuple[int, int, int, int]:
    """Paddle returns bbox as ``[x0, y0, x1, y1]`` or a 4-point polygon.
    Normalise to (x0,y0,x1,y1) ints clamped to the image."""
    if not box:
        return (0, 0, 0, 0)
    if isinstance(box, (list, tuple)) and len(box) == 4 and not isinstance(box[0], (list, tuple)):
        x0, y0, x1, y1 = box
    else:
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        except Exception:
            return (0, 0, 0, 0)
    x0 = max(0, int(round(float(x0))))
    y0 = max(0, int(round(float(y0))))
    x1 = min(w, int(round(float(x1))))
    y1 = min(h, int(round(float(y1))))
    return (x0, y0, x1, y1)
