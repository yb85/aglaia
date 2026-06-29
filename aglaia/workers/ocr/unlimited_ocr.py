# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Baidu Unlimited-OCR engine — a dense 3B document VLM, run locally.

MLX on Apple Silicon (the quantized variants ship ``model_type`` pre-patched to
``deepseekocr`` so mlx-vlm recognises them), vLLM on CUDA — auto-picked by
``LocalVlmServer``. Configuration on top of ``OpenAiCompatVlmOcr``: the model
card wants the prompt to start with a literal ``<image>`` and
``skip_special_tokens=False`` so the grounding tokens reach us for the searchable
PDF text layer.
"""

from __future__ import annotations

from aglaia.app_data.downloads import DownloadTarget, register_download

from .engine import register
from .openai_compat import OpenAiCompatVlmOcr

register_download(
    DownloadTarget(
        key="unlimited_ocr_mlx",
        title="Unlimited-OCR (MLX 4-bit)",
        filename="Unlimited-OCR-mlx",
        url="sahilchachra/unlimited-ocr-4bit-mlx",
        approx_size_mb=2200,
        kind="hf-snapshot",
        section="other",
        purpose="OCR",
        project="baidu/Unlimited-OCR",
        platform="darwin-arm64",
        registered_by="unlimited_ocr",
    )
)
register_download(
    DownloadTarget(
        key="unlimited_ocr_vllm",
        title="Unlimited-OCR (vLLM)",
        filename="Unlimited-OCR",
        url="baidu/Unlimited-OCR",
        approx_size_mb=6500,
        kind="hf-snapshot",
        section="other",
        purpose="OCR",
        project="baidu/Unlimited-OCR",
        platform="cuda",
        registered_by="unlimited_ocr",
    )
)


@register
class UnlimitedOcrEngine(OpenAiCompatVlmOcr):
    name = "unlimited"
    display = "Unlimited-OCR (local)"
    description = "Local 3B doc VLM (MLX/vLLM); long docs, Markdown + tables."
    mlx_target_key = "unlimited_ocr_mlx"
    vllm_target_key = "unlimited_ocr_vllm"
    # The model card requires a literal "<image>" prefix.
    prompt = "<image>\nConvert this page to Markdown."
    # skip_special_tokens=False keeps the <|ref|>/<|det|> grounding tokens (else
    # the server strips them and we lose the bbox layer); the R-SWA xargs are the
    # recipe's recommended defaults.
    extra_body = {
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 35, "window_size": 1024},
    }
