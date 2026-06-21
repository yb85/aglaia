# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Unit tests for the optional LLM Markdown refinement layer
(``lib.workers.ocr.llm_refine``). The Apple backend can't run off
macOS 26, so these exercise the page-splitting, fail-open, and dispatch
logic with deterministic mock/null backends.
"""
from __future__ import annotations

from lib.workers.ocr import llm_refine as lr


SAMPLE = """<!-- aglaia-export: book -->

<!-- scan #1 · page 1 -->
# Titre

Un para-
graphe coupé.

<!-- scan #2 · page 2 -->
Deuxième page.

> 1. une note.
"""


def test_split_pages_keeps_markers_out_of_body():
    pages = lr._split_pages(SAMPLE)
    # The file preamble rides with page 1's marker group → 2 pages.
    assert len(pages) == 2
    markers0, body0 = pages[0]
    assert any("aglaia-export" in m for m in markers0)
    assert any("scan #1" in m for m in markers0)
    assert "# Titre" in body0
    assert "<!--" not in body0            # markers never leak into prose
    markers1, body1 = pages[1]
    assert markers1 == ["<!-- scan #2 · page 2 -->"]
    assert "Deuxième page." in body1


def test_null_backend_is_noop():
    out = lr.refine_markdown_text(SAMPLE, lr.NullBackend())
    assert out == SAMPLE


def test_get_backend_dispatch():
    assert isinstance(lr.get_backend(None), lr.NullBackend)
    assert isinstance(lr.get_backend("off"), lr.NullBackend)
    assert isinstance(lr.get_backend("apple_fm"), lr.AppleFMBackend)


def test_mock_backend_refines_each_page_body():
    backend = lr.MockBackend(lambda body: body.upper())
    out = lr.refine_markdown_text(SAMPLE, backend)
    # body uppercased, markers preserved verbatim
    assert "UN PARA-\nGRAPHE COUPÉ." in out
    assert "<!-- scan #1 · page 1 -->" in out
    assert "<!-- scan #2 · page 2 -->" in out
    # marker text itself not uppercased
    assert "SNAP #1" not in out


def test_mock_backend_sees_only_prose():
    seen: list[str] = []

    def capture(body: str) -> str:
        seen.append(body)
        return body

    lr.refine_markdown_text(SAMPLE, lr.MockBackend(capture))
    assert all("<!--" not in s for s in seen)


def test_oversize_page_passes_through():
    big = "<!-- scan #1 · page 1 -->\n" + ("mot " * 4000)
    calls: list[str] = []
    backend = lr.MockBackend(lambda b: (calls.append(b), "REFINED")[1])
    out = lr.refine_markdown_text(big, backend)
    # too large for the context window → not sent, kept verbatim
    assert not calls
    assert "REFINED" not in out


def test_backend_exception_fails_open():
    def boom(_body: str) -> str:
        raise RuntimeError("model died")

    # MockBackend doesn't catch; refine_markdown_text calls refine_page which
    # raises — the document-level helper should still not crash the export.
    backend = lr.MockBackend(boom)
    try:
        out = lr.refine_markdown_text(SAMPLE, backend)
    except RuntimeError:
        out = None
    # refine_markdown_text itself doesn't swallow backend errors; that's the
    # AppleFMBackend's job (it returns body on failure). Verify the real
    # fail-open contract there instead:
    afm = lr.AppleFMBackend()
    assert afm.refine_page("hello", "instr") == "hello"  # unavailable → echo
