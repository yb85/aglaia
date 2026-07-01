# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""VLM serving backends — how to spawn an OpenAI-compatible server for local
weights, and which one this machine can run.

Two backends, both speaking the OpenAI ``/v1`` protocol so the client side is
identical:

* ``MlxBackend`` → ``python -m mlx_vlm.server`` (Apple Silicon, unified memory,
  no CUDA). The proven path PaddleOCR-VL already uses.
* ``VllmBackend`` → ``python -m vllm.entrypoints.openai.api_server`` (CUDA/other).

``pick_backend()`` chooses by platform + installed deps, overridable with
``AGLAIA_VLM_BACKEND=mlx|vllm``.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class VlmBackend(Protocol):
    """How to launch and reach an OpenAI-compatible server for a local model."""

    name: str

    def available(self) -> bool:
        """True iff this machine can run the backend (right OS + deps installed)."""
        ...

    def spawn_cmd(
        self, model_path: str, port: int, *, max_tokens: int, log_level: str
    ) -> list[str]:
        """Argv to launch the server bound to 127.0.0.1:``port`` serving
        ``model_path`` (a local weights dir/file)."""
        ...

    def env(self) -> dict[str, str]:
        """Extra environment for the spawned process (merged over os.environ)."""
        ...

    def served_model_name(self, model_path: str) -> str:
        """The id a client must pass as ``model`` in chat requests."""
        ...


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


class MlxBackend:
    """Apple-Silicon MLX server (``mlx_vlm.server``)."""

    name = "mlx"

    def available(self) -> bool:
        return (
            sys.platform == "darwin"
            and platform.machine() == "arm64"
            and _module_installed("mlx_vlm")
        )

    def spawn_cmd(
        self, model_path: str, port: int, *, max_tokens: int, log_level: str
    ) -> list[str]:
        # ``--trust-remote-code`` is required for models shipping a custom
        # processor/tokenizer class via transformers' Auto-class path (e.g.
        # PaddleOCR-VL); without it the server exits before serving a request.
        return [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            model_path,
            "--trust-remote-code",
            "--max-tokens",
            str(max_tokens),
            "--log-level",
            log_level,
        ]

    def env(self) -> dict[str, str]:
        # Newer mlx_vlm.server reads MLX_TRUST_REMOTE_CODE at import, racing the
        # CLI flag's side effect in at least one release. Set it unambiguously.
        return {"MLX_TRUST_REMOTE_CODE": "true"}

    def served_model_name(self, model_path: str) -> str:
        # mlx_vlm.server registers the model under the path it was given.
        return model_path


class VllmBackend:
    """CUDA/other vLLM server (``vllm.entrypoints.openai.api_server``)."""

    name = "vllm"

    def available(self) -> bool:
        return _module_installed("vllm")

    def spawn_cmd(
        self, model_path: str, port: int, *, max_tokens: int, log_level: str
    ) -> list[str]:
        # vLLM caps generation per-request (not at serve time), so max_tokens is
        # unused here. Pin the served id to the weights' basename so the client
        # has a stable, predictable name.
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            model_path,
            "--served-model-name",
            self.served_model_name(model_path),
            "--trust-remote-code",
            "--uvicorn-log-level",
            log_level.lower(),
        ]

    def env(self) -> dict[str, str]:
        return {}

    def served_model_name(self, model_path: str) -> str:
        return Path(model_path).name


# Concrete backends in platform-preference order.
_BACKENDS: dict[str, VlmBackend] = {"mlx": MlxBackend(), "vllm": VllmBackend()}


def pick_backend(prefer: str | None = None) -> VlmBackend | None:
    """The best available backend for this machine, or ``None`` if neither is
    installed. ``AGLAIA_VLM_BACKEND`` (or ``prefer``) forces a choice — but only
    if that backend is actually available."""
    forced = (prefer or os.environ.get("AGLAIA_VLM_BACKEND", "")).strip().lower()
    if forced:
        b = _BACKENDS.get(forced)
        return b if (b is not None and b.available()) else None
    # Platform preference: MLX on Apple Silicon, else vLLM.
    order = ("mlx", "vllm") if sys.platform == "darwin" else ("vllm", "mlx")
    for key in order:
        b = _BACKENDS[key]
        if b.available():
            return b
    return None
