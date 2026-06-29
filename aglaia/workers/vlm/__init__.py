# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR-agnostic local VLM serving.

A small, model-neutral layer that spins up an OpenAI-compatible inference
server for a set of *local* weights and tears it down cleanly. The backend is
chosen by platform — **MLX** (`mlx_vlm.server`) on Apple Silicon, **vLLM**
(`vllm.entrypoints.openai.api_server`) on CUDA/others — so the same OCR engine
runs locally everywhere without caring which serves it.

Nothing here is OCR-specific: it manages a process and an HTTP endpoint. OCR
engines (and anything else needing a hosted VLM) call ``LocalVlmServer.ensure``
and talk OpenAI-compat to the returned base URL.
"""

from __future__ import annotations

from .backends import MlxBackend, VllmBackend, VlmBackend, pick_backend
from .server import LocalVlmServer, set_log_sink

__all__ = [
    "LocalVlmServer",
    "MlxBackend",
    "VllmBackend",
    "VlmBackend",
    "pick_backend",
    "set_log_sink",
]
