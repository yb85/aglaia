# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Unlimited-OCR backend — in-process MLX inference + output parsing.

The `unlimited` engine runs Baidu's Unlimited-OCR (DeepSeek-OCR stack + R-SWA
"multipage attention") locally via the upstream ``mlx_vlm`` ``unlimited_ocr``
model (git-pinned in ``pyproject.toml``), against a hybrid-precision MLX weight
dir (F32 vision + 4-bit LLM) produced by the ``unlimited-ocr-mlx`` converter.

This module is deliberately NOT an ``OcrEngine`` subclass (so the registry does
not pick it up) — it holds the pieces the engine composes:

* ``load_model`` / ``generate_text`` — the vendored inference path (mirrors
  ``unlimited-ocr-mlx``'s ``infer.run_ocr``);
* ``parse_spans`` — the ``<|det|>label [box]<|/det|>text`` grounding grammar;
* ``segment_pages`` — split the *fused* multipage stream (no delimiter, coords
  reset 0-999 per page) back into exactly N pages by the strongest y-resets;
* ``spans_to_markdown`` / ``scale_box`` — render + map boxes to page pixels.

Design notes live in ``docs/ocr.md`` and the port's ``HANDOFF.md``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

# Dynamic-resolution modes (mirrors unlimited-ocr-mlx config.py). BASE = the
# multi-page / R-SWA path (no crop, one 1024 square per page); GUNDAM = the
# single-image high-res crop path.
BASE: dict[str, Any] = {"cropping": False, "image_size": 1024, "base_size": 1024}
GUNDAM: dict[str, Any] = {"cropping": True, "image_size": 640, "base_size": 1024}

# Grounding coordinate space (DeepSeek-OCR): boxes are normalised 0..999
# relative to each page's *padded square*.
COORD_MAX = 999.0

# Repetition control: greedy decode + no penalty loops into a LaTeX/`1. 2. …`
# tail (same failure as aglaia #56). These are the port's validated defaults.
REPETITION_PENALTY = 1.15
REPETITION_CONTEXT = 64


# ── Inference (vendored from unlimited-ocr-mlx infer.run_ocr) ─────────────

def load_model(model_path: str) -> tuple[Any, Any]:
    """Load the hybrid-precision MLX model + processor. Expensive (~3.7 GB q4);
    the engine caches the result across a whole OCR run."""
    from mlx_vlm import load
    return load(model_path)


def generate_text(
    model: Any,
    processor: Any,
    image_paths: list[str],
    *,
    multi_page: bool,
    max_tokens: int = 32768,
    repetition_penalty: float | None = REPETITION_PENALTY,
) -> str:
    """Run Unlimited-OCR over one or more page images, return the decoded text.

    ``multi_page`` → base mode + "Multi page parsing." (all pages fused at one
    ``<image>`` position — the R-SWA continuous path); otherwise gundam mode +
    "document parsing." for a single page.
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    mode = BASE if multi_page else GUNDAM
    task = "Multi page parsing." if multi_page else "document parsing."
    prompt = apply_chat_template(
        processor, model.config, task, num_images=len(image_paths)
    )
    kwargs: dict[str, Any] = {}
    if repetition_penalty and repetition_penalty != 1.0:
        from mlx_lm.sample_utils import make_logits_processors
        kwargs["logits_processors"] = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=REPETITION_CONTEXT,
        )
    result = generate(
        model=model,
        processor=processor,
        image=list(image_paths),
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        **mode,
        **kwargs,
    )
    return str(getattr(result, "text", result))


# ── Output parsing (the <|det|> grounding grammar) ───────────────────────

@dataclass
class Span:
    """One grounded output span: a semantic ``label`` (text / header /
    ref_text / page_number / footer / equation / image / …), its normalised
    boxes (0..999, may be several), and the transcribed ``text`` that follows."""

    label: str
    boxes: list[tuple[float, float, float, float]]
    text: str

    @property
    def y0(self) -> float | None:
        """Top-most normalised y across this span's boxes (page-order key)."""
        return min((b[1] for b in self.boxes), default=None)


# Matches both emitted shapes:
#   <|det|>label [x,y,x,y]<|/det|>text
#   <|ref|>label<|/ref|><|det|>[[x,y,x,y],…]<|/det|>text
# label may come from either the ref or the det side; box is optional (some
# spans are pure text continuations); text runs to the next marker / EOS.
_SPAN_RE = re.compile(
    r"(?:<\|ref\|>\s*(?P<rlabel>[^<]*?)\s*<\|/ref\|>\s*)?"
    r"<\|det\|>\s*(?P<dlabel>[A-Za-z_][\w-]*)?\s*"
    r"(?P<box>\[[^<]*?\])?\s*<\|/det\|>"
    r"(?P<text>.*?)(?=<\|ref\|>|<\|det\|>|$)",
    re.DOTALL,
)


def _parse_boxes(box_str: str | None) -> list[tuple[float, float, float, float]]:
    if not box_str:
        return []
    try:
        val = ast.literal_eval(box_str.strip())
    except (ValueError, SyntaxError):
        return []
    if val and isinstance(val[0], (int, float)):
        val = [val]  # flat [x,y,x,y] → single box
    out: list[tuple[float, float, float, float]] = []
    for b in val:
        try:
            x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            out.append((x0, y0, x1, y1))
        except (TypeError, IndexError, ValueError):
            continue
    return out


def parse_spans(text: str) -> list[Span]:
    """Parse the raw decoded stream into ordered `Span`s. Spans with neither a
    box nor text are dropped (stray markers)."""
    spans: list[Span] = []
    for m in _SPAN_RE.finditer(text):
        label = (m.group("dlabel") or m.group("rlabel") or "").strip()
        boxes = _parse_boxes(m.group("box"))
        body = (m.group("text") or "").strip()
        if not boxes and not body:
            continue
        spans.append(Span(label=label or "text", boxes=boxes, text=body))
    return spans


# ── Per-page segmentation of the fused multipage stream ──────────────────

def segment_pages(spans: list[Span], n_pages: int) -> list[list[Span]]:
    """Split the fused span stream into exactly ``n_pages`` groups.

    The model fuses all pages at one ``<image>`` position and emits ONE stream
    with no page delimiter — but grounding coords reset to 0..999 per page, so
    the y-coordinate sawtooths (climbs down a page, then jumps back to the top
    for the next). We take the ``n_pages - 1`` largest downward y-jumps as page
    boundaries: this always yields ``n_pages`` groups and keys off the strongest
    reset signal rather than a fragile per-span threshold. Spans without a box
    (pure text continuations) attach to the current page and never split it.
    """
    if n_pages <= 1 or len(spans) <= 1:
        return [spans]

    # y0 per span (None for box-less spans — ignored as boundary candidates).
    ys = [s.y0 for s in spans]
    # Candidate boundaries: index i starts a new page if y drops vs the last
    # box-bearing span. Score = magnitude of the downward jump.
    drops: list[tuple[float, int]] = []
    last_y: float | None = None
    for i, y in enumerate(ys):
        if y is None:
            continue
        if last_y is not None and y < last_y:
            drops.append((last_y - y, i))
        last_y = y
    if not drops:
        # No reset signal (e.g. single-column monotonic) — even split fallback.
        step = max(1, len(spans) // n_pages)
        return [spans[i:i + step] for i in range(0, len(spans), step)][:n_pages] \
            or [spans]

    # Strongest (n_pages-1) downward jumps → boundary indices, in order.
    drops.sort(key=lambda d: d[0], reverse=True)
    cuts = sorted(idx for _, idx in drops[: n_pages - 1])
    groups: list[list[Span]] = []
    prev = 0
    for c in cuts:
        groups.append(spans[prev:c])
        prev = c
    groups.append(spans[prev:])
    # Guard: coalesce/pad to exactly n_pages so page↔result alignment holds.
    groups = [g for g in groups if g] or [spans]
    while len(groups) < n_pages:
        groups.append([])          # trailing blank pages (model stopped early)
    if len(groups) > n_pages:      # merge overflow tail into the last page
        groups[n_pages - 1:] = [sum(groups[n_pages - 1:], [])]
    return groups


# ── Rendering + coordinate mapping ───────────────────────────────────────

# Labels that are structural noise for the reading-order markdown body.
_SKIP_MD_LABELS = {"page_number", "footer", "image"}
_HEADING_LABELS = {"header", "title", "section"}


def spans_to_markdown(spans: list[Span]) -> str:
    """Render a page's spans to reading-order markdown. Headers become ``#``
    lines; page furniture (page numbers, running footers, figure crops) is
    dropped from the body. Faithful-but-light — refinement is a later pass."""
    lines: list[str] = []
    for s in spans:
        if not s.text or s.label in _SKIP_MD_LABELS:
            continue
        if s.label in _HEADING_LABELS:
            lines.append(f"# {s.text}")
        else:
            lines.append(s.text)
    return "\n\n".join(lines).strip()


def scale_box(
    box: tuple[float, float, float, float],
    page_w: int,
    page_h: int,
    *,
    base_size: int = 1024,
) -> tuple[int, int, int, int]:
    """Map a 0..999 grounding box to page pixels.

    Base mode letterboxes each page into a ``base_size`` square (``ImageOps.pad``
    — aspect-preserving, centred), so a coord is relative to that padded square,
    not the raw page. Undo the pad: scale = base/max(w,h); the shorter axis is
    centred with ``(base - scaled)/2`` padding.
    """
    if page_w <= 0 or page_h <= 0:
        return (0, 0, 0, 0)
    scale = base_size / max(page_w, page_h)
    new_w, new_h = page_w * scale, page_h * scale
    pad_x, pad_y = (base_size - new_w) / 2.0, (base_size - new_h) / 2.0
    xs = []
    for coord, pad, sz in (
        (box[0], pad_x, page_w), (box[1], pad_y, page_h),
        (box[2], pad_x, page_w), (box[3], pad_y, page_h),
    ):
        px = (coord / COORD_MAX * base_size - pad) / scale
        xs.append(int(max(0, min(sz, px))))
    return (xs[0], xs[1], xs[2], xs[3])


@dataclass
class PageParse:
    """One page's worth of parsed output."""

    markdown: str
    lines: list[dict] = field(default_factory=list)   # OcrLine-shaped dicts
    spans: list[dict] = field(default_factory=list)   # structured, for meta


def build_page(spans: list[Span], page_w: int, page_h: int) -> PageParse:
    """Turn a page's spans into markdown + pixel-mapped OcrLine dicts."""
    md = spans_to_markdown(spans)
    lines: list[dict] = []
    structured: list[dict] = []
    for s in spans:
        px_boxes = [scale_box(b, page_w, page_h) for b in s.boxes]
        structured.append({"label": s.label, "text": s.text,
                           "boxes": px_boxes})
        if s.text and px_boxes:
            lines.append({"text": s.text, "bbox": px_boxes[0],
                          "confidence": 1.0})
    return PageParse(markdown=md, lines=lines, spans=structured)
