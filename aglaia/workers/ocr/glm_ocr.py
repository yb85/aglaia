# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""GLM-OCR engine — a 0.9B GLM-V document VLM (zai-org/GLM-OCR), run locally.

MLX on Apple Silicon, vLLM on CUDA — auto-picked by ``LocalVlmServer``. Code is
Apache-2.0, the model MIT. Pure configuration on top of ``OpenAiCompatVlmOcr``:
two download targets, a prompt, and the default grounding parser.
"""

from __future__ import annotations

from aglaia.app_data.downloads import DownloadTarget, register_download

from .engine import register
from .openai_compat import OpenAiCompatVlmOcr

# Weights — MLX (Apple Silicon) and full-precision (vLLM/CUDA) variants. Both
# fetched into models_dir() so serving is offline and registry-tracked.
register_download(
    DownloadTarget(
        key="glm_ocr_mlx",
        title="GLM-OCR (MLX 8-bit)",
        filename="GLM-OCR-mlx",
        url="mlx-community/GLM-OCR-8bit",
        approx_size_mb=1000,
        kind="hf-snapshot",
        section="other",
        purpose="OCR",
        project="zai-org/GLM-OCR",
        platform="darwin-arm64",
        registered_by="glm_ocr",
    )
)
register_download(
    DownloadTarget(
        key="glm_ocr_vllm",
        title="GLM-OCR (vLLM)",
        filename="GLM-OCR",
        url="zai-org/GLM-OCR",
        approx_size_mb=1900,
        kind="hf-snapshot",
        section="other",
        purpose="OCR",
        project="zai-org/GLM-OCR",
        platform="cuda",
        registered_by="glm_ocr",
    )
)


@register
class GlmOcrEngine(OpenAiCompatVlmOcr):
    name = "glm"
    display = "GLM-OCR (local)"
    description = "Local 0.9B GLM-V doc VLM (MLX/vLLM); Markdown + tables."
    mlx_target_key = "glm_ocr_mlx"
    vllm_target_key = "glm_ocr_vllm"
    prompt = "Convert this document image to Markdown, preserving tables and layout."
