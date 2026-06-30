# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Surya 2 OCR — served locally as a VLM (no torch / llama-server / GGUF).

Surya 2 (``datalab-to/surya-ocr-2``) is a Qwen3.5-VL document model, so it runs
through the shared ``LocalVlmServer`` like the other VLM engines: the MLX build
(``aglaia-models/surya-ocr-2-mlx``) on Apple Silicon, the HF weights via vLLM on
CUDA. This replaces the old ``surya-ocr`` + ``llama-server`` + GGUF stack, which
pinned torch / huggingface-hub<1 / openai<2 and made the install a packaging
headache (mutually exclusive with the MLX/vLLM stack).

Output is HTML-flavoured Markdown (``<p>`` / ``<table>`` …); the base engine
keeps it as ``meta.markdown`` for the exporters (HTML is valid in Markdown).
"""

from __future__ import annotations

from .engine import register
from .openai_compat import OpenAiCompatVlmOcr


@register
class SuryaEngine(OpenAiCompatVlmOcr):
    name = "surya"
    display = "Surya 2 (local)"
    description = "Local Surya 2 doc VLM (MLX/vLLM); Markdown, tables, 90+ scripts."
    mlx_target_key = "surya_mlx"
    vllm_target_key = "surya_vllm"
    # Surya 2 is trained to emit EITHER layout JSON OR full-page OCR HTML,
    # selected by the prompt. This is its exact whole-page OCR prompt
    # (HIGH_ACCURACY_BBOX_PROMPT in datalab's surya/inference/prompts.py) — a
    # generic "convert to Markdown" instead triggered the layout task (the
    # [{"label","bbox","count"}] JSON we were exporting as garbage .md). Output
    # is <div data-label=… data-bbox=…>text</div> blocks.
    prompt = (
        "OCR this image to HTML. Each block is a div with data-label and "
        "data-bbox (x0 y0 x1 y1, normalized 0-1000)."
    )
    output_html = True  # <div> blocks → Markdown via html_to_markdown
