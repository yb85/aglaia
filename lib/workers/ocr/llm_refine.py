# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Optional LLM post-processing for the Markdown export.

The geometric heuristics in ``lib.workers.md_export`` recover *structure*
(headings, paragraphs, footnotes, lists) but cannot fix what the OCR got
wrong at the *character / semantic* level: broken accents, mis-split words,
garbled Greek, a heading the geometry missed. An on-device LLM can — without
sending the user's scans anywhere.

This module is the seam for that. It defines a tiny backend protocol and three
implementations:

* ``AppleFMBackend`` — Apple's Foundation Models framework via the official
  ``apple-fm-sdk`` Python bindings (on-device, offline, free). Requires
  **macOS 26+** with Apple Intelligence enabled; until then ``available()``
  returns ``(False, reason)`` and the export silently stays heuristic-only.
* ``NullBackend`` — always unavailable; the default.
* ``MockBackend`` — deterministic, for tests (applies a caller-supplied fn).

Design constraints baked in here:

* **Per-page chunking.** The on-device model's context window is 8192 tokens
  (WWDC26; was 4096 in the first release) for the whole session — prompt +
  instructions + response. We refine one page at a time, each in a *fresh*
  session so transcripts never accumulate, and skip pages too large for a
  round-trip. A ``PrivateCloudComputeLanguageModel`` backend (32000-token
  window) could instead refine whole chapters for better cross-page coherence.
* **Multimodal (WWDC26).** The Swift API now takes image ``Attachment``s, so a
  future backend can hand the model the page *raster* alongside the OCR text —
  image-grounded correction is far stronger than text-only for garbled Greek,
  accents, and layout. ``refine_page`` takes an optional ``image`` for exactly
  this; the current Apple backend ignores it until the Python SDK exposes it.
* **Pluggable providers (WWDC26).** Apple's own ``LanguageModel`` protocol now
  admits Claude / Gemini / MLX backends behind the same session API — the same
  shape as ``RefineBackend`` here, so a cloud backend slots in cleanly.
* **Faithful, not creative.** The instruction forbids translating,
  summarising, or inventing content — this is OCR correction, not authorship.
* **Fail open.** Any backend error leaves that page's heuristic Markdown
  untouched; a bad LLM never loses data.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Protocol


# Page chunks are split on the scan markers md_export emits.
_PAGE_MARKER_RE = re.compile(r"^<!-- scan #\d+.*?-->$")
_COMMENT_RE = re.compile(r"^<!--.*-->$")

# Context windows (WWDC26): on-device model 8192 tokens, Private Cloud Compute
# 32000. At ~4 chars/token, an 8k-token window leaves room for instructions +
# response with a single page of body well under this cap; larger pages fall
# back to heuristic output untouched. (A PCC backend could lift this far higher
# and refine whole chapters at once — see module docs.)
_MAX_INPUT_CHARS = 12000


_CLEANUP_INSTRUCTION = (
    "You are an OCR post-processor for printed scholarly books (often in "
    "French, with Greek and Latin quotations). You receive the Markdown of a "
    "single page produced by OCR, which may contain recognition errors. "
    "Return a corrected version of THE SAME page as clean Markdown:\n"
    "- Fix obvious character errors and broken/missing accents.\n"
    "- Join words split by end-of-line hyphenation.\n"
    "- Preserve the existing structure exactly: headings (#), block quotes "
    "(>), list items (-), and footnotes.\n"
    "- Preserve every language; do NOT translate.\n"
    "- Do NOT summarise, add, reorder, or delete content.\n"
    "- If a passage is garbled beyond recognition (e.g. unreadable Greek), "
    "leave it as it is.\n"
    "- Output ONLY the corrected Markdown for this page, with no preamble, "
    "explanation, or code fence."
)

_STRUCTURE_INSTRUCTION = (
    "You are a layout-aware OCR post-processor. You receive the OCR Markdown of "
    "a single printed page. Re-express it as well-structured Markdown that "
    "reads as flowing prose: merge lines into paragraphs, promote real titles "
    "to headings (#/##/###), keep footnotes as block quotes, and fix obvious "
    "OCR errors and hyphenation. Preserve all languages and content verbatim "
    "in meaning — never translate, summarise, or invent. Output ONLY the "
    "Markdown for this page."
)

_INSTRUCTIONS = {"cleanup": _CLEANUP_INSTRUCTION, "structure": _STRUCTURE_INSTRUCTION}


class RefineBackend(Protocol):
    """A page-at-a-time Markdown refiner."""

    def available(self) -> tuple[bool, str]:
        """``(usable, reason)``. ``reason`` explains a False for the user."""
        ...

    def refine_page(self, body: str, instruction: str,
                    image: object | None = None) -> str:
        """Return a corrected version of one page's Markdown ``body``.

        ``image`` (optional) is the page raster for multimodal, image-grounded
        correction (WWDC26 ``Attachment``); backends that can't use it ignore
        it and fall back to text-only."""


class NullBackend:
    """Always-unavailable backend — the default, keeps export heuristic-only."""

    name = "null"

    def available(self) -> tuple[bool, str]:
        return (False, "LLM refinement disabled")

    def refine_page(self, body: str, instruction: str,
                    image: object | None = None) -> str:
        return body


class MockBackend:
    """Deterministic backend for tests; applies ``fn`` to each page body."""

    name = "mock"

    def __init__(self, fn: Callable[[str], str]):
        self._fn = fn

    def available(self) -> tuple[bool, str]:
        return (True, "mock")

    def refine_page(self, body: str, instruction: str,
                    image: object | None = None) -> str:
        return self._fn(body)


class AppleFMBackend:
    """Apple Foundation Models (on-device) via ``apple-fm-sdk``.

    Lazily imports the SDK and probes ``SystemLanguageModel.is_available()`` so
    that importing this module never fails on machines (or CI) without the
    framework. Each page is refined in a fresh ``LanguageModelSession`` to keep
    every request inside the small context window.
    """

    name = "apple_fm"

    def __init__(self) -> None:
        self._fm = None
        self._loop = None
        self._reason = ""

    def _ensure(self) -> bool:
        if self._fm is not None:
            return True
        try:
            import apple_fm_sdk as fm  # type: ignore
        except Exception as e:  # ImportError or load failure off-macOS-26
            self._reason = f"apple-fm-sdk not installed: {e}"
            return False
        try:
            ok, reason = fm.SystemLanguageModel().is_available()
        except Exception as e:
            self._reason = f"Foundation Models probe failed: {e}"
            return False
        if not ok:
            self._reason = reason or "Apple Intelligence unavailable"
            return False
        self._fm = fm
        return True

    def available(self) -> tuple[bool, str]:
        if self._ensure():
            return (True, "Apple Foundation Models ready")
        return (False, self._reason)

    def _run(self, coro):
        import asyncio
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def refine_page(self, body: str, instruction: str,
                    image: object | None = None) -> str:
        if not self._ensure():
            return body
        fm = self._fm
        # ``image`` is accepted for the WWDC26 multimodal path but ignored
        # until apple-fm-sdk exposes image Attachments to Python; text-only
        # refinement is used in the meantime.

        async def _go() -> str:
            session = fm.LanguageModelSession(instructions=instruction)
            return await session.respond(body)

        try:
            out = self._run(_go())
        except Exception:
            return body  # fail open — keep heuristic output for this page
        text = getattr(out, "content", None) or getattr(out, "text", None) or str(out)
        return text.strip() or body


def get_backend(name: Optional[str]) -> RefineBackend:
    """Factory: ``"apple_fm"`` → on-device LLM, anything else → null."""
    if name in (None, "", "none", "null", "off"):
        return NullBackend()
    if name in ("apple", "apple_fm", "foundation_models"):
        return AppleFMBackend()
    return NullBackend()


def _split_pages(md_text: str) -> list[tuple[list[str], str]]:
    """Split exported Markdown into ``(marker_lines, body)`` per page.

    Leading HTML comments (the scan/branch markers) head each page and are kept
    out of the LLM input — the model only ever sees prose."""
    lines = md_text.splitlines()
    pages: list[tuple[list[str], str]] = []
    cur_markers: list[str] = []
    cur_body: list[str] = []

    def flush():
        if cur_markers or cur_body:
            pages.append((cur_markers[:], "\n".join(cur_body).strip("\n")))

    started = False
    for ln in lines:
        if _PAGE_MARKER_RE.match(ln):
            if started:
                flush()
                cur_markers.clear()
                cur_body.clear()
            started = True
            cur_markers.append(ln)
        elif not started and _COMMENT_RE.match(ln):
            # File-level header comment (aglaia-export) — its own preamble page.
            cur_markers.append(ln)
        else:
            cur_body.append(ln)
    flush()
    return pages


def refine_markdown_text(md_text: str, backend: RefineBackend,
                         mode: str = "cleanup") -> str:
    """Refine an exported Markdown document page-by-page with ``backend``.

    Pages with no prose, or too large for the context window, pass through
    untouched. Markers are re-attached verbatim. Returns the original text
    unchanged when the backend is unavailable."""
    ok, _ = backend.available()
    if not ok:
        return md_text
    instruction = _INSTRUCTIONS.get(mode, _CLEANUP_INSTRUCTION)

    out_chunks: list[str] = []
    for markers, body in _split_pages(md_text):
        refined = body
        stripped = body.strip()
        if stripped and len(body) <= _MAX_INPUT_CHARS:
            refined = backend.refine_page(body, instruction) or body
        block = "\n".join(markers)
        if refined:
            block = f"{block}\n{refined}" if block else refined
        out_chunks.append(block.rstrip())
    return "\n\n".join(c for c in out_chunks if c) + "\n"
