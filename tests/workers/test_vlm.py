# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR-agnostic local VLM serving layer — backend selection, command shapes,
and the server's non-spawning surface (no mlx/vllm install needed)."""

from __future__ import annotations

import socket

import pytest

from aglaia.workers.vlm import LocalVlmServer
from aglaia.workers.vlm import backends as B


def test_pick_backend_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(B.MlxBackend, "available", lambda self: False)
    monkeypatch.setattr(B.VllmBackend, "available", lambda self: False)
    assert B.pick_backend() is None


def test_pick_backend_forced_must_be_available(monkeypatch):
    monkeypatch.setattr(B.MlxBackend, "available", lambda self: False)
    assert B.pick_backend(prefer="mlx") is None  # forced but absent → None
    monkeypatch.setattr(B.MlxBackend, "available", lambda self: True)
    assert B.pick_backend(prefer="mlx").name == "mlx"


def test_pick_backend_platform_preference(monkeypatch):
    monkeypatch.setattr(B.MlxBackend, "available", lambda self: True)
    monkeypatch.setattr(B.VllmBackend, "available", lambda self: True)
    monkeypatch.setattr(B.sys, "platform", "darwin")
    assert B.pick_backend().name == "mlx"  # Apple Silicon → MLX
    monkeypatch.setattr(B.sys, "platform", "linux")
    assert B.pick_backend().name == "vllm"  # else → vLLM


def test_env_var_forces_backend(monkeypatch):
    monkeypatch.setattr(B.MlxBackend, "available", lambda self: True)
    monkeypatch.setattr(B.VllmBackend, "available", lambda self: True)
    monkeypatch.setattr(B.sys, "platform", "darwin")  # would prefer mlx…
    monkeypatch.setenv("AGLAIA_VLM_BACKEND", "vllm")  # …but env wins
    assert B.pick_backend().name == "vllm"


def test_served_model_name_per_backend():
    assert B.MlxBackend().served_model_name("/w/PaddleOCR-VL") == "/w/PaddleOCR-VL"
    assert B.VllmBackend().served_model_name("/w/GLM-OCR") == "GLM-OCR"


def test_mlx_spawn_cmd_shape():
    cmd = B.MlxBackend().spawn_cmd("/w/m", 8200, max_tokens=2048, log_level="WARNING")
    assert "mlx_vlm.server" in cmd
    assert cmd[cmd.index("--model") + 1] == "/w/m"
    assert cmd[cmd.index("--port") + 1] == "8200"
    assert "--trust-remote-code" in cmd
    assert cmd[cmd.index("--max-tokens") + 1] == "2048"


def test_vllm_spawn_cmd_shape():
    cmd = B.VllmBackend().spawn_cmd(
        "/w/GLM-OCR", 8200, max_tokens=2048, log_level="INFO"
    )
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "GLM-OCR"
    assert cmd[cmd.index("--uvicorn-log-level") + 1] == "info"  # lowercased


def test_mlx_env_sets_trust_remote_code():
    assert B.MlxBackend().env().get("MLX_TRUST_REMOTE_CODE") == "true"
    assert B.VllmBackend().env() == {}


def test_free_port_is_bindable():
    from aglaia.workers.vlm.server import _free_port

    p = _free_port()
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p))  # must be free right after _free_port returns
    finally:
        s.close()


def test_ensure_raises_without_backend(monkeypatch):
    monkeypatch.setattr("aglaia.workers.vlm.server.pick_backend", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="No local VLM backend"):
        LocalVlmServer.ensure("/no/such/model")


def test_served_model_name_classmethod_uses_backend(monkeypatch):
    monkeypatch.setattr(
        "aglaia.workers.vlm.server.pick_backend", lambda *a, **k: B.VllmBackend()
    )
    assert LocalVlmServer.served_model_name("/w/GLM-OCR") == "GLM-OCR"


def test_log_sink_routing(monkeypatch):
    from aglaia.workers.vlm import server as S

    seen = []
    S.set_log_sink(lambda text, level="info": seen.append((level, text)))
    S._LOG("hello", level="warn")
    S.set_log_sink(None)  # restore stdout
    assert ("warn", "hello") in seen
