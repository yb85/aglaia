# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Abstract OpenAI-compatible VLM OCR engine.

A base for *end-to-end* document VLMs that emit a whole page as Markdown over an
OpenAI ``/v1/chat/completions`` endpoint (GLM-OCR, Baidu Unlimited-OCR, …). The
model runs **locally and auto-managed**: the OCR-agnostic ``LocalVlmServer``
spins up the right backend for the machine — MLX on Apple Silicon, vLLM on
CUDA — and this base does the rest (DPI downsample, the chat call, grounding-
token parsing, ``OcrResult`` assembly).

A concrete engine is almost pure configuration: declare the two download
targets (MLX / vLLM weights), the prompt, and any model-specific ``extra_body``.
Override ``parse_output`` only for a non-standard grounding grammar — the
default handles ``<|ref|>…<|/ref|><|det|>[[x,y,x,y]]<|/det|>`` (DeepSeek-OCR /
PaddleOCR-VL lineage) with a clean-Markdown fallback.

This is distinct from ``paddle_vl``: PaddleOCR-VL drives a layout orchestrator
that fans out one request per region, so it talks to ``LocalVlmServer`` directly
rather than through this whole-page chat base.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from typing import List, Optional

import cv2
import numpy as np

from aglaia.app_data import models_dir
from aglaia.app_data.downloads import is_downloaded, target_for
from aglaia.workers.vlm import LocalVlmServer, pick_backend

from .engine import (
    OcrEngine,
    OcrLine,
    OcrResult,
    downsample_to_dpi as _downsample,
    engine_log as _log,
    resolve_ocr_dpi as _target_dpi,
)

_HTTP_TIMEOUT_S = 600.0

# Grounding grammar shared across the lineage: <|ref|>TEXT<|/ref|> followed by
# <|det|>[[x0,y0,x1,y1]]<|/det|> with coords normalised to 0–(coord_scale-1).
_REF_DET = re.compile(
    r"<\|ref\|>(?P<text>.*?)<\|/ref\|>\s*<\|det\|>(?P<box>.*?)<\|/det\|>",
    re.DOTALL,
)
_ANY_TOKEN = re.compile(r"<\|/?(?:ref|det)\|>")
_BOX_NUMS = re.compile(r"-?\d+(?:\.\d+)?")


def to_data_url(image_rgb: np.ndarray) -> str:
    """RGB ndarray → base64 JPEG data URL (cv2 wants BGR)."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _det_to_pixels(
    box: str, w: int, h: int, scale: float
) -> Optional[tuple[int, int, int, int]]:
    nums = [float(n) for n in _BOX_NUMS.findall(box or "")]
    if len(nums) < 4:
        return None
    x0, y0, x1, y1 = nums[:4]

    def px(v: float, span: int) -> int:
        return max(0, min(span, int(round(v / scale * span))))

    bx0, bx1 = sorted((px(x0, w), px(x1, w)))
    by0, by1 = sorted((px(y0, h), px(y1, h)))
    return (bx0, by0, bx1, by1)


def html_to_markdown(html: str) -> str:
    """Convert a model's HTML output (Surya emits ``<p>`` / ``<table>`` / ``<h*>``)
    to clean Markdown — tables become Markdown tables. Uses ``html2text`` when
    available, falling back to a crude tag-strip that at least preserves block
    breaks so a ``.md`` never ships raw ``<div>`` soup."""
    html = (html or "").strip()
    if not html:
        return ""
    try:
        import html2text

        conv = html2text.HTML2Text()
        conv.body_width = 0  # don't hard-wrap lines
        conv.ignore_emphasis = False
        return conv.handle(html).strip()
    except Exception:
        text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li|table)>", "\n", html)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_grounded_markdown(
    raw: str, w: int, h: int, *, coord_scale: float, fallback_full_page: bool
) -> tuple[str, list[OcrLine]]:
    """Split grounded model output into (clean Markdown, per-line bboxes).

    The Markdown (tokens stripped) feeds ``md_export``; the ``<|det|>`` boxes
    become ``OcrLine``s with pixel bboxes so ``pdf_export`` can place a real
    searchable text layer. If the build emits plain Markdown (no grounding) and
    ``fallback_full_page`` is set, keep one page-spanning line so per-page search
    and a coarse PDF text layer still work."""
    lines: list[OcrLine] = []
    for m in _REF_DET.finditer(raw):
        text = (m.group("text") or "").strip()
        bbox = _det_to_pixels(m.group("box"), w, h, coord_scale)
        if text and bbox is not None:
            lines.append({"text": text, "bbox": bbox, "confidence": 1.0})
    markdown = _ANY_TOKEN.sub("", _REF_DET.sub(lambda m: m.group("text"), raw))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if not lines and markdown and fallback_full_page:
        lines = [{"text": markdown, "bbox": (0, 0, int(w), int(h)), "confidence": 1.0}]
    return markdown, lines


class OpenAiCompatVlmOcr(OcrEngine):
    """Base for whole-page VLM OCR over a locally-served OpenAI endpoint."""

    # Abstract base — concrete engines override `name`. The "abstract" sentinel
    # keeps OcrEngine.__init_subclass__ from warning about a missing name here.
    name = "abstract"

    # ── subclass configuration ───────────────────────────────────────
    mlx_target_key: str = ""  # download key for MLX weights (Apple Silicon)
    vllm_target_key: str = ""  # download key for vLLM weights (CUDA)
    prompt: str = "Convert this document image to Markdown."
    extra_body: dict = {}  # merged into the chat request (model knobs)
    coord_scale: float = 1000.0  # det-coordinate normalisation
    fallback_full_page: bool = True
    output_html: bool = False  # model emits HTML (e.g. Surya) → convert to Markdown

    # ── UI/capability traits ─────────────────────────────────────────
    default_dpi: int = 150
    supports_live = False  # a server round-trip per frame is wasteful
    cloud = False  # self-hosted (local GPU / Apple Silicon)

    def __init__(self) -> None:
        self._max_tokens = 8192
        self._backend_override: Optional[str] = None
        self.available = self._weights_ready()

    # ── backend / weights resolution ─────────────────────────────────
    def _backend(self):
        return pick_backend(self._backend_override)

    def _target_key_for(self, backend) -> str:
        if backend is None:
            return ""
        return self.mlx_target_key if backend.name == "mlx" else self.vllm_target_key

    def _weights_ready(self) -> bool:
        be = self._backend()
        key = self._target_key_for(be)
        return bool(be and key and is_downloaded(key))

    def configure(self, params: dict[str, str]) -> None:
        """Spec options: ``backend=mlx|vllm`` forces a backend (if available);
        ``max_tokens=N`` raises the per-page generation budget."""
        if "backend" in params:
            self._backend_override = params["backend"] or None
        if "max_tokens" in params:
            try:
                self._max_tokens = int(params["max_tokens"])
            except ValueError:
                pass

    # ── parsing hook ─────────────────────────────────────────────────
    def parse_output(self, raw: str, w: int, h: int) -> tuple[str, list[OcrLine]]:
        if self.output_html:
            md = html_to_markdown(raw)
            lines: list[OcrLine] = (
                [{"text": md, "bbox": (0, 0, int(w), int(h)), "confidence": 1.0}]
                if md
                else []
            )
            return md, lines
        return parse_grounded_markdown(
            raw,
            w,
            h,
            coord_scale=self.coord_scale,
            fallback_full_page=self.fallback_full_page,
        )

    # ── public API ───────────────────────────────────────────────────
    def recognize(
        self,
        image_rgb: np.ndarray,
        languages: List[str],
        *,
        src_dpi: float | None = None,
    ) -> OcrResult:
        return self.recognize_batch(
            [image_rgb],
            languages,
            src_dpis=[src_dpi] if src_dpi is not None else None,
        )[0]

    def recognize_batch(
        self,
        images_rgb: list[np.ndarray],
        languages: List[str],
        *,
        src_dpis: list[float] | None = None,
    ) -> list[OcrResult]:
        if not images_rgb:
            return []
        be = self._backend()
        if be is None:
            raise RuntimeError(
                f"{self.name}: no local VLM backend available — install one: "
                "`--extra macos` (Apple Silicon / mlx-vlm) or "
                "`--extra cuda` (Linux / vLLM)."
            )
        key = self._target_key_for(be)
        if not key or not is_downloaded(key):
            raise RuntimeError(
                f"{self.name}: weights for the {be.name} backend aren't "
                f"downloaded. Open the Model Downloader and install '{key}'."
            )
        model_path = str(models_dir() / target_for(key).filename)
        base_url = LocalVlmServer.ensure(
            model_path,
            backend=be,
            log=_log,
            log_stem=self.name,
            max_tokens=self._max_tokens,
        )
        model_name = LocalVlmServer.served_model_name(model_path, be)

        target_dpi = _target_dpi()
        if src_dpis is None:
            src_dpis = [0.0] * len(images_rgb)
        out: list[OcrResult] = []
        for img, dpi in zip(images_rgb, src_dpis):
            scaled = _downsample(img, dpi or 0, target_dpi)
            h, w = scaled.shape[:2]
            try:
                raw = self._chat_completion(base_url, model_name, scaled)
                markdown, lines = self.parse_output(raw, w, h)
            except Exception as e:  # noqa: BLE001 — surface, don't crash the run
                _log(f"[{self.name}] page {w}x{h} failed: {e}", level="error")
                out.append(self._empty(w, h))
                continue
            out.append(
                {
                    "engine": self.name,
                    "languages": [],
                    "page_w": int(w),
                    "page_h": int(h),
                    "lines": lines,
                    "meta": {
                        "markdown": markdown,
                        "ocr_dpi": int(target_dpi or 0),
                        "backend": be.name,
                    },
                }
            )
        return out

    # ── HTTP ─────────────────────────────────────────────────────────
    def _chat_completion(
        self, base_url: str, model_name: str, image_rgb: np.ndarray
    ) -> str:
        body = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(image_rgb)},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            **self.extra_body,
        }
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"] or ""

    def _empty(self, w: int, h: int) -> OcrResult:
        return {
            "engine": self.name,
            "languages": [],
            "page_w": int(w),
            "page_h": int(h),
            "lines": [],
            "meta": {"markdown": ""},
        }
