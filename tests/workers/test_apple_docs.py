# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Unit tests for the Apple Document OCR engine's engine-agnostic logic.

The Vision passes (``_flat_lines`` / ``_recognize_document``) need macOS +
the framework, so they're exercised only behind an availability guard. The
pure-Python pieces — the confidence gate, the complement fan-out, the
bbox→document splice, the degenerate-crop filter, and the capability probe
— are tested here with NO live Vision / Surya, by mocking ``get_engine``.
"""
from __future__ import annotations

import numpy as np
import pytest

from lib.workers.ocr import apple_docs as ad
from lib.workers.ocr import apple_caps as caps


# ── helpers ─────────────────────────────────────────────────────────

class _StubEngine:
    """A complement engine that maps any crop to a fixed text."""

    def __init__(self, available=True, text="ΓΡΕΕΚ", raise_on_call=False):
        self.available = available
        self._text = text
        self._raise = raise_on_call
        self.calls = 0

    def recognize(self, crop, languages, **kw):
        self.calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return {"lines": [{"text": self._text, "confidence": 0.95}]}


def _img(h=400, w=600):
    return np.full((h, w, 3), 255, np.uint8)


def _line(text, conf, bbox):
    return {"text": text, "bbox": list(bbox), "confidence": conf}


# ── resolve_complement ──────────────────────────────────────────────

def test_resolve_complement_default():
    assert ad.resolve_complement(None) == "surya"


def test_resolve_complement_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("AGLAIA_OCR_COMPLEMENT", "surya")
    assert ad.resolve_complement("paddle_vl") == "paddle_vl"


def test_resolve_complement_env(monkeypatch):
    monkeypatch.delenv("AGLAIA_OCR_COMPLEMENT", raising=False)
    monkeypatch.setenv("AGLAIA_OCR_COMPLEMENT", "none")
    assert ad.resolve_complement(None) == "none"


def test_resolve_complement_rejects_garbage(monkeypatch):
    monkeypatch.delenv("AGLAIA_OCR_COMPLEMENT", raising=False)
    assert ad.resolve_complement("nope") == "surya"


# ── degenerate filter ───────────────────────────────────────────────

@pytest.mark.parametrize("text", ["", "65", "12.", "THOSTAZIE", "65 "])
def test_degenerate_true(text):
    assert ad._is_degenerate(text) is True


@pytest.mark.parametrize("text", [
    "ἀλλὰ πρὸς τὰ γενητά καὶ τὰ ἔργα",
    "a real multi word greek line here",
])
def test_degenerate_false(text):
    assert ad._is_degenerate(text) is False


# ── confidence gate → complement ────────────────────────────────────

def test_gate_offloads_only_low_conf(monkeypatch):
    stub = _StubEngine(text="CORRECTED")
    monkeypatch.setattr(ad, "get_engine", lambda name: stub)

    lines = [
        _line("bon français", 1.0, (10, 10, 500, 40)),       # keep
        _line("Greek garble here please", 0.30, (10, 60, 500, 100)),  # offload
    ]
    n, used = ad._apply_complement(_img(), lines, [], "surya", ["fr-FR"])
    assert n == 1
    assert used == "surya"
    assert stub.calls == 1
    assert lines[0]["text"] == "bon français"        # untouched
    assert lines[1]["text"] == "CORRECTED"           # corrected
    assert lines[1]["confidence"] >= 0.99
    assert lines[1]["complement"] == "surya"


def test_gate_skips_degenerate(monkeypatch):
    stub = _StubEngine()
    monkeypatch.setattr(ad, "get_engine", lambda name: stub)
    lines = [_line("65", 0.50, (10, 10, 60, 40))]   # page number
    n, used = ad._apply_complement(_img(), lines, [], "surya", [])
    assert n == 0
    assert stub.calls == 0


def test_gate_fail_open_when_unavailable(monkeypatch):
    stub = _StubEngine(available=False)
    monkeypatch.setattr(ad, "get_engine", lambda name: stub)
    lines = [_line("greek garble line text", 0.30, (10, 10, 500, 40))]
    n, used = ad._apply_complement(_img(), lines, [], "surya", [])
    assert n == 0
    assert used is None
    assert lines[0]["text"] == "greek garble line text"   # Vision text kept


def test_gate_fail_open_on_engine_error(monkeypatch):
    stub = _StubEngine(raise_on_call=True)
    monkeypatch.setattr(ad, "get_engine", lambda name: stub)
    lines = [_line("greek garble line text", 0.30, (10, 10, 500, 40))]
    n, used = ad._apply_complement(_img(), lines, [], "surya", [])
    assert n == 0
    assert lines[0]["text"] == "greek garble line text"


def test_gate_fail_open_on_missing_engine(monkeypatch):
    def _boom(name):
        raise KeyError(name)
    monkeypatch.setattr(ad, "get_engine", _boom)
    lines = [_line("greek garble line text", 0.30, (10, 10, 500, 40))]
    n, used = ad._apply_complement(_img(), lines, [], "surya", [])
    assert n == 0


# ── block grouping (thin strips defeat the VLM → crop blocks) ───────

def test_group_low_blocks_contiguous():
    # three lines stacked ~one line apart → one block.
    low = [
        _line("a", 0.3, (10, 100, 500, 140)),
        _line("b", 0.3, (10, 150, 500, 190)),
        _line("c", 0.3, (10, 200, 500, 240)),
    ]
    blocks = ad._group_low_blocks(low)
    assert len(blocks) == 1
    assert len(blocks[0]) == 3


def test_group_low_blocks_splits_on_big_gap():
    low = [
        _line("a", 0.3, (10, 100, 500, 140)),
        _line("b", 0.3, (10, 150, 500, 190)),
        # big vertical jump → new block
        _line("c", 0.3, (10, 900, 500, 940)),
    ]
    blocks = ad._group_low_blocks(low)
    assert len(blocks) == 2
    assert len(blocks[0]) == 2
    assert len(blocks[1]) == 1


def test_union_bbox():
    low = [
        _line("a", 0.3, (50, 100, 400, 140)),
        _line("b", 0.3, (10, 150, 500, 190)),
    ]
    assert ad._union_bbox(low) == [10, 100, 500, 190]


def test_split_block_text_matching_lines():
    assert ad._split_block_text("one\ntwo\nthree", 3) == ["one", "two", "three"]


def test_split_block_text_mismatch_collapses_to_first():
    out = ad._split_block_text("a single blob of text", 3)
    assert out[0] == "a single blob of text"
    assert out[1:] == ["", ""]


def test_gate_block_redistributes_multiline(monkeypatch):
    # A 2-line Greek block; the VLM returns both lines newline-joined.
    class _Multi:
        available = True

        def recognize(self, crop, languages, **kw):
            return {"lines": [
                {"text": "ΓΡΑΜΜΗ ΑΛΦΑ"}, {"text": "ΓΡΑΜΜΗ ΒΗΤΑ"},
            ]}

    monkeypatch.setattr(ad, "get_engine", lambda name: _Multi())
    lines = [
        _line("garble alpha line here", 0.30, (10, 100, 500, 140)),
        _line("garble beta line here", 0.30, (10, 150, 500, 190)),
    ]
    n, used = ad._apply_complement(_img(), lines, [], "surya", [])
    assert n == 2
    assert lines[0]["text"] == "ΓΡΑΜΜΗ ΑΛΦΑ"
    assert lines[1]["text"] == "ΓΡΑΜΜΗ ΒΗΤΑ"


# ── document splice ─────────────────────────────────────────────────

def test_replace_in_document_block_line():
    # a paragraph block with two lines; correct the 2nd.
    doc = [{
        "type": "block",
        "text": "first line second garble",
        "bbox": [10, 10, 500, 100],
        "lines": [
            {"text": "first line", "bbox": [10, 10, 500, 50]},
            {"text": "second garble", "bbox": [10, 55, 500, 95]},
        ],
    }]
    ad._replace_in_document(doc, [10, 55, 500, 95], "CORRECTED GREEK")
    assert doc[0]["lines"][1]["text"] == "CORRECTED GREEK"
    # whole-block text re-joined from corrected lines
    assert "CORRECTED GREEK" in doc[0]["text"]
    assert "first line" in doc[0]["text"]


def test_replace_in_document_list_item():
    doc = [{
        "type": "list",
        "items": [
            {"text": "item one", "bbox": [10, 10, 500, 40]},
            {"text": "item garble", "bbox": [10, 50, 500, 80]},
        ],
    }]
    ad._replace_in_document(doc, [10, 50, 500, 80], "FIXED")
    assert doc[0]["items"][1]["text"] == "FIXED"


def test_replace_in_document_no_match_is_noop():
    doc = [{"type": "block", "text": "x", "bbox": [10, 10, 50, 50], "lines": []}]
    ad._replace_in_document(doc, [900, 900, 950, 950], "Y")
    assert doc[0]["text"] == "x"


def test_gate_splices_into_document(monkeypatch):
    stub = _StubEngine(text="CORRECTED")
    monkeypatch.setattr(ad, "get_engine", lambda name: stub)
    doc = [{
        "type": "block", "text": "garble line text here",
        "bbox": [10, 60, 500, 100],
        "lines": [{"text": "garble line text here", "bbox": [10, 60, 500, 100]}],
    }]
    lines = [_line("garble line text here", 0.30, (10, 60, 500, 100))]
    n, _ = ad._apply_complement(_img(), lines, doc, "surya", [])
    assert n == 1
    assert doc[0]["lines"][0]["text"] == "CORRECTED"
    assert doc[0]["text"] == "CORRECTED"


# ── degenerate crop sizing ──────────────────────────────────────────

def test_pad_crop_upscales_thin_line():
    arr = _img(h=600, w=800)
    crop = ad._pad_crop(arr, [100, 100, 700, 130])   # 30 px tall
    assert crop is not None
    assert crop.shape[0] >= ad._MIN_CROP_H


def test_pad_crop_degenerate_none():
    arr = _img(h=600, w=800)
    assert ad._pad_crop(arr, [100, 100, 102, 101]) is None


# ── quad → bbox (top-left origin, no Y-flip) ────────────────────────

class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Quad:
    def __init__(self, x0, y0, x1, y1):
        self._tl, self._tr = _Pt(x0, y0), _Pt(x1, y0)
        self._bl, self._br = _Pt(x0, y1), _Pt(x1, y1)

    def topLeft(self):
        return self._tl

    def topRight(self):
        return self._tr

    def bottomLeft(self):
        return self._bl

    def bottomRight(self):
        return self._br


def test_quad_bbox_top_left_origin():
    # A region in the TOP of the page (small normalised y) must produce a
    # small pixel y0 — i.e. NO Y-flip. Verified empirically on macOS 26.
    q = _Quad(0.1, 0.05, 0.9, 0.10)
    bbox = ad._quad_bbox(q, 1000, 2000)
    assert bbox == [100, 100, 900, 200]


# ── capability probe ────────────────────────────────────────────────

def test_probe_apple_caps_shape():
    c = caps.probe_apple_caps()
    assert isinstance(c.is_macos, bool)
    assert isinstance(c.has_vision, bool)
    assert isinstance(c.has_documents, bool)
    # invariant: documents ⇒ vision ⇒ macos
    if c.has_documents:
        assert c.has_vision
    if c.has_vision:
        assert c.is_macos


# ── engine registration + availability ──────────────────────────────

def test_engine_registered():
    from lib.workers.ocr import ENGINE_REGISTRY
    assert "apple_docs" in ENGINE_REGISTRY
    eng = ENGINE_REGISTRY["apple_docs"]()
    assert eng.name == "apple_docs"
    # availability mirrors the capability probe's has_documents.
    assert eng.available == caps.probe_apple_caps().has_documents
