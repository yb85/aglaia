# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Abstract OpenAI-compatible VLM OCR engine + the glm / unlimited engines.
Server spawn and HTTP are stubbed — no mlx/vllm install or network needed."""

from __future__ import annotations

import numpy as np
import pytest

from aglaia.workers.ocr.engine import ENGINE_REGISTRY, get_engine
from aglaia.workers.ocr.openai_compat import (
    OpenAiCompatVlmOcr,
    html_to_markdown,
    parse_grounded_markdown,
)


# ── HTML → Markdown (Surya emits HTML) ───────────────────────────────
def test_html_to_markdown_paragraphs():
    real = "<div>\n<p>Line one.</p>\n<p>Line two: 1234567890.</p>\n</div>"
    md = html_to_markdown(real)
    assert "Line one." in md and "Line two: 1234567890." in md
    assert "<" not in md  # no raw HTML left


def test_html_to_markdown_table():
    md = html_to_markdown(
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    )
    assert "A" in md and "B" in md and "---" in md  # rendered as a Markdown table


def test_surya_engine_converts_html_output():
    e = get_engine("surya")
    assert e.output_html is True
    md, lines = e.parse_output("<div><p>Hello world</p></div>", 800, 300)
    assert md == "Hello world" and len(lines) == 1
    assert lines[0]["bbox"] == (0, 0, 800, 300)


# ── grounding parser ─────────────────────────────────────────────────
def test_parse_grounded_splits_md_and_boxes():
    raw = (
        "# Title\n"
        "<|ref|>Chapter One<|/ref|><|det|>[[100, 50, 700, 95]]<|/det|>\n"
        "<|ref|>Body<|/ref|><|det|>[[120, 120, 680, 160]]<|/det|>\n"
    )
    md, lines = parse_grounded_markdown(
        raw, 800, 1000, coord_scale=1000.0, fallback_full_page=True
    )
    assert md == "# Title\nChapter One\nBody"
    assert [ln["text"] for ln in lines] == ["Chapter One", "Body"]
    assert lines[0]["bbox"] == (80, 50, 560, 95)  # 0–999 normalised → pixels


def test_parse_grounded_fallback_full_page():
    md, lines = parse_grounded_markdown(
        "# plain page", 800, 1000, coord_scale=1000.0, fallback_full_page=True
    )
    assert len(lines) == 1 and lines[0]["bbox"] == (0, 0, 800, 1000)


def test_parse_grounded_no_fallback():
    _md, lines = parse_grounded_markdown(
        "# plain page", 800, 1000, coord_scale=1000.0, fallback_full_page=False
    )
    assert lines == []


# ── engines registered + configured ──────────────────────────────────
def test_engines_registered():
    assert "glm" in ENGINE_REGISTRY
    assert "unlimited" in ENGINE_REGISTRY


def test_unlimited_extra_body_carries_recipe_knobs():
    u = get_engine("unlimited")
    assert u.extra_body["skip_special_tokens"] is False
    assert u.extra_body["vllm_xargs"]["ngram_size"] == 35
    assert u.prompt.startswith("<image>")


def test_targets_registered():
    from aglaia.app_data import downloads as D

    keys = {t.key for t in D.registry()}
    assert {
        "glm_ocr_mlx",
        "glm_ocr_vllm",
        "unlimited_ocr_mlx",
        "unlimited_ocr_vllm",
    } <= keys


# ── availability + backend/target wiring ─────────────────────────────
class _FakeBackend:
    def __init__(self, name):
        self.name = name


def test_target_key_follows_backend():
    g = get_engine("glm")
    assert g._target_key_for(_FakeBackend("mlx")) == "glm_ocr_mlx"
    assert g._target_key_for(_FakeBackend("vllm")) == "glm_ocr_vllm"
    assert g._target_key_for(None) == ""


def test_available_requires_backend_and_weights(monkeypatch):
    import aglaia.workers.ocr.openai_compat as OC

    monkeypatch.setattr(OC, "pick_backend", lambda *a, **k: _FakeBackend("mlx"))
    monkeypatch.setattr(OC, "is_downloaded", lambda key: key == "glm_ocr_mlx")
    assert get_engine("glm")._weights_ready() is True  # mlx weights present
    assert get_engine("unlimited")._weights_ready() is False  # its mlx key absent


def test_configure_backend_and_max_tokens():
    g = get_engine("glm")
    g.configure({"backend": "vllm", "max_tokens": "4096"})
    assert g._backend_override == "vllm"
    assert g._max_tokens == 4096


def test_recognize_batch_raises_without_backend(monkeypatch):
    import aglaia.workers.ocr.openai_compat as OC

    monkeypatch.setattr(OC, "pick_backend", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="no local VLM backend"):
        get_engine("glm").recognize_batch([np.zeros((4, 4, 3), np.uint8)], [])


def test_recognize_batch_end_to_end_stubbed(monkeypatch):
    """Stub backend pick, weights presence, server spin-up, and the HTTP call —
    exercise downsample → parse → OcrResult assembly."""
    import aglaia.workers.ocr.openai_compat as OC

    monkeypatch.setattr(OC, "pick_backend", lambda *a, **k: _FakeBackend("mlx"))
    monkeypatch.setattr(OC, "is_downloaded", lambda key: True)
    monkeypatch.setattr(
        OC.LocalVlmServer,
        "ensure",
        classmethod(lambda cls, *a, **k: "http://127.0.0.1:9/"),
    )
    monkeypatch.setattr(
        OC.LocalVlmServer, "served_model_name", classmethod(lambda cls, *a, **k: "glm")
    )

    served = "# Doc\n<|ref|>Heading<|/ref|><|det|>[[0,0,500,40]]<|/det|>\nbody"
    g = get_engine("glm")
    monkeypatch.setattr(g, "_chat_completion", lambda url, model, img: served)

    [res] = g.recognize_batch([np.zeros((200, 160, 3), np.uint8)], [])
    assert res["engine"] == "glm"
    assert res["meta"]["markdown"] == "# Doc\nHeading\nbody"
    assert res["meta"]["backend"] == "mlx"
    assert res["lines"][0]["text"] == "Heading"


def test_recognize_batch_per_page_failure_is_isolated(monkeypatch):
    import aglaia.workers.ocr.openai_compat as OC

    monkeypatch.setattr(OC, "pick_backend", lambda *a, **k: _FakeBackend("mlx"))
    monkeypatch.setattr(OC, "is_downloaded", lambda key: True)
    monkeypatch.setattr(
        OC.LocalVlmServer,
        "ensure",
        classmethod(lambda cls, *a, **k: "http://127.0.0.1:9/"),
    )
    monkeypatch.setattr(
        OC.LocalVlmServer, "served_model_name", classmethod(lambda cls, *a, **k: "glm")
    )

    def boom(url, model, img):
        raise RuntimeError("server 500")

    g = get_engine("glm")
    monkeypatch.setattr(g, "_chat_completion", boom)
    [res] = g.recognize_batch([np.zeros((50, 50, 3), np.uint8)], [])
    assert res["lines"] == [] and res["meta"]["markdown"] == ""  # empty, not crash


def test_base_is_not_registered():
    # The abstract base must never end up resolvable as an engine.
    assert OpenAiCompatVlmOcr.name == "abstract"
    assert "abstract" not in ENGINE_REGISTRY
