# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Markdown export of OCR text.

Goal: produce a free-flowing document that reads like the original page,
not a per-scan report card. Structural markers (scan id, branch path)
are emitted as HTML comments so they're invisible in rendered MD but
still parseable when re-importing.

Three source shapes feed the exporter, in order of preference:

  1. ``meta.markdown`` — PaddleOCR-VL ships a fully-assembled, layout-aware
     MD string per page. Emitted verbatim.
  2. ``meta.structure`` (Surya) — per-block label/html, rendered by
     ``_render_structure``.
  3. Flat ``lines`` list (Apple Vision and other line-only engines) — only
     ``(text, bbox, confidence)`` per line, no font weight / italic / style.
     Structure is inferred *geometrically* by the document-level pipeline in
     ``_render_line_pages``:

     * **Dehyphenation + line joining** — consecutive lines are fused into
       paragraphs; a trailing ``-``/``¬`` is treated as a soft word split.
     * **Paragraph breaks** — vertical gap, first-line indent (alinéa), or a
       short terminated previous line. Continuation cues (lowercase start
       after an unterminated line) suppress spurious breaks.
     * **Cross-page continuation** — a sentence spanning a page break is kept
       as one paragraph; the page marker is spliced in as an inline comment so
       it stays invisible without breaking the flow.
     * **Running header / footer / page number removal** — lines repeated in
       the same band across many pages are dropped (document-level pass).
     * **Headings** — multi-cue scoring (height ratio + isolation + centering +
       caps + explicit markers), with heading *level* assigned from a
       document-wide clustering of candidate heights so the same title gets the
       same depth on every page.
     * **Bold** — short ALL-CAPS lines wrapped in ``**…**``.
     * **Lists** — bullet (`•·*…`) and numbered (`1.`, `a)`, `IV.`) markers →
       `-` items. Em-dash dialogue is intentionally *not* treated as a list.
     * **Block quotes** — paragraphs inset from both margins → `>`.
     * **Footnotes** — small lines in the bottom band → `>` blockquote.
     * **Confidence filter** — very-low-confidence lines (figure noise) dropped.
     * **Two columns** — a clear central gutter splits the page; columns are
       read left-then-right.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from pathlib import Path


# ── Surya layout-label → Markdown ──────────────────────────────────
# Surya emits per-block `label` (canonicalised) on every OCR run. We
# translate them to Markdown so the export reads as a structured doc
# rather than a flat text dump. Labels not in this map fall back to
# plain paragraph text. List sourced from surya/layout/label.py.

_SKIP_LABELS = {
    "PageHeader", "PageFooter",
    "Picture", "Figure",      # binary content; no path to embed here
}

# Heading depth per label. Anything not in this map but recognised as a
# heading-ish label maps to h2.
_HEADING_LEVEL = {
    "Title": 1,
    "SectionHeader": 2,
}

_LIST_LABELS = {"ListItem"}
_QUOTE_LABELS = {"Footnote"}
_FORMULA_LABELS = {"Formula", "Equation"}
_CODE_LABELS = {"Code", "Listing"}
_CAPTION_LABELS = {"Caption"}
_TABLE_LABELS = {"Table"}


def _strip_html(html: str) -> str:
    """Pull plain text out of Surya's HTML snippet. Block-level tags
    become spaces so adjacent words stay separated."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _render_structure(blocks: list[dict]) -> list[str]:
    """Render Surya-style blocks (one per layout region, with ``label``
    + ``html`` + ``reading_order``) into a list of Markdown chunks.

    Empty strings denote paragraph breaks. Each non-empty entry is one
    "block" that the caller writes verbatim — no extra wrapping."""
    if not blocks:
        return []

    # Honour the model's reading order. Fall back to file order when
    # the field is missing for one or more blocks.
    def _order(b: dict) -> tuple[int, int]:
        ro = b.get("reading_order")
        return (0 if ro is None else 1, ro if isinstance(ro, int) else 0)

    sorted_blocks = sorted(blocks, key=_order)

    out: list[str] = []
    in_list = False
    for b in sorted_blocks:
        label = (b.get("label") or "").strip()
        if label in _SKIP_LABELS:
            continue
        html = b.get("html") or ""
        text = _strip_html(html)
        # Even when text is empty, a Table block can still carry HTML
        # worth emitting. For everything else, empty == drop.
        if not text and label not in _TABLE_LABELS:
            continue

        # Insert a blank line between non-list and list runs so the
        # markdown renderer doesn't glue them into a single paragraph.
        is_list = label in _LIST_LABELS
        if in_list and not is_list:
            out.append("")
        in_list = is_list

        if label in _HEADING_LEVEL:
            level = _HEADING_LEVEL[label]
            out.append(f"{'#' * level} {text}")
            out.append("")
        elif is_list:
            out.append(f"- {text}")
        elif label in _QUOTE_LABELS:
            out.append(f"> {text}")
            out.append("")
        elif label in _FORMULA_LABELS:
            out.append("$$")
            out.append(text)
            out.append("$$")
            out.append("")
        elif label in _CODE_LABELS:
            out.append("```")
            out.append(text)
            out.append("```")
            out.append("")
        elif label in _CAPTION_LABELS:
            out.append(f"*{text}*")
            out.append("")
        elif label in _TABLE_LABELS:
            # Pass through the HTML — most MD renderers (incl. GitHub,
            # marked, commonmark.js) accept inline HTML.
            out.append(html.strip())
            out.append("")
        else:
            out.append(text)
            out.append("")

    # Trim trailing blank.
    while out and out[-1] == "":
        out.pop()
    return out


# ── Apple-Document structured tree → Markdown ──────────────────────
# The ``apple_docs`` engine (and the structured path of ``apple_vision``)
# stamps a reading-ordered ``meta.document`` tree of typed blocks. Each
# entry is one of:
#   {"type": "block",  "text": str, ...}            → paragraph / heading
#   {"type": "list",   "items": [{"text": str}, …]} → bullet list
#   {"type": "table",  "markdown": str}             → GFM table verbatim
# Vision already did paragraph assembly, reading order and table layout, so
# this is a direct, high-fidelity render — strictly preferred over the
# geometric line heuristics, which stay as the fallback for line-only
# engines.

# A short, isolated, non-terminated block reads as a heading. We keep the
# heuristic tiny: Vision doesn't label headings, but a centred/short line
# with no sentence punctuation at the top of the reading order is almost
# always a title. md_export's existing line-path heading scorer is richer;
# here we deliberately under-promote to avoid false headings in prose.
def _looks_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 60:
        return False
    if t.endswith((".", ",", ";", ":", "!", "?", "»")):
        return False
    words = t.split()
    if len(words) > 8:
        return False
    # ALL-CAPS or Title-ish short line.
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return True
    return False


def _render_document(blocks: list[dict]) -> list[str]:
    """Render a ``meta.document`` tree into Markdown chunks.

    Returns a flat list of lines (blank entries are paragraph breaks),
    matching the contract of ``_render_structure``."""
    out: list[str] = []
    in_list = False
    for blk in blocks:
        btype = blk.get("type")
        if btype == "table":
            md = (blk.get("markdown") or "").strip()
            if md:
                if in_list:
                    out.append("")
                in_list = False
                out.append(md)
                out.append("")
            continue
        if btype == "list":
            items = [it.get("text", "").strip()
                     for it in (blk.get("items") or [])]
            items = [t for t in items if t]
            if not items:
                continue
            if not in_list:
                out.append("")
            in_list = True
            for t in items:
                out.append(f"- {t}")
            continue
        # block (paragraph / heading)
        text = (blk.get("text") or "").strip()
        if not text:
            continue
        if in_list:
            out.append("")
        in_list = False
        if _looks_heading(text):
            out.append(f"## {text}")
            out.append("")
        else:
            out.append(text)
            out.append("")
    while out and out[-1] == "":
        out.pop()
    return out


def _project_slug(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT slug FROM project WHERE id = 1").fetchone()
    return (row["slug"] if row else None) or "project"


# Map raw engine names from ``ocr_runs.engine`` to the suffix shown in
# export filenames (Aglaïa convention: short engine tag + ``OCR``).
_ENGINE_SUFFIXES = {
    "apple_vision": "_appleOCR",
    "surya":         "_suryaOCR",
    "paddle_vl":     "_paddleOCR",
}


def ocr_engine_suffix(conn: sqlite3.Connection) -> str:
    """Return a filename suffix matching the engine + DPI that
    produced the currently visible OCR text in this project, e.g.
    ``_paddleOCR_150dpi``.

    Counts only the LATEST done run per (scan, branch) tuple — the
    earlier "dominant engine" heuristic ignored the freshness of the
    rows, so a project OCR'd 50 pages with Surya then re-OCR'd today
    with Paddle still got tagged ``_suryaOCR``. Using
    ``MAX(version) per (scan, branch_path)`` matches the same row set
    ``write_markdown`` actually exports, then picks the engine most
    used across those latest rows. Returns an empty string when no
    OCR data is present."""
    row = conn.execute("""
        SELECT engine, COUNT(*) AS n
          FROM ocr_runs o
          JOIN (
              SELECT scan_id, branch_path, MAX(version) AS v
                FROM ocr_runs
               WHERE status = 'done'
            GROUP BY scan_id, branch_path
          ) m ON m.scan_id = o.scan_id
             AND m.branch_path = o.branch_path
             AND m.v = o.version
         WHERE o.status = 'done'
           AND o.engine IS NOT NULL
           AND o.engine != ''
      GROUP BY engine
      ORDER BY n DESC
         LIMIT 1
    """).fetchone()
    if not row:
        return ""
    engine_tag = _ENGINE_SUFFIXES.get(
        row["engine"], f"_{row['engine']}OCR"
    )
    # DPI: most common ocr_dpi across the latest done runs for that
    # winning engine. Reads meta.ocr_dpi out of result_json. Empty
    # ``_NNNdpi`` segment when no run carries a DPI hint (older runs
    # written before the field landed).
    dpi_tag = _dominant_ocr_dpi_tag(conn, row["engine"])
    return f"{engine_tag}{dpi_tag}"


def _dominant_ocr_dpi_tag(conn: sqlite3.Connection, engine: str) -> str:
    """Most common ``meta.ocr_dpi`` across the latest done runs for
    ``engine``. Returns ``_NNNdpi`` or ``""`` when unknown."""
    rows = conn.execute("""
        SELECT o.result_json
          FROM ocr_runs o
          JOIN (
              SELECT scan_id, branch_path, MAX(version) AS v
                FROM ocr_runs
               WHERE status = 'done' AND engine = ?
            GROUP BY scan_id, branch_path
          ) m ON m.scan_id = o.scan_id
             AND m.branch_path = o.branch_path
             AND m.v = o.version
         WHERE o.status = 'done' AND o.engine = ?
    """, (engine, engine)).fetchall()
    counts: dict[int, int] = {}
    for r in rows:
        raw = r["result_json"]
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        v = (d.get("meta") or {}).get("ocr_dpi")
        try:
            dpi = int(v) if v is not None else 0
        except (TypeError, ValueError):
            dpi = 0
        if dpi > 0:
            counts[dpi] = counts.get(dpi, 0) + 1
    if not counts:
        return ""
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return f"_{best}dpi"


# ── Text-shape helpers ─────────────────────────────────────────────

# Trailing chars that may follow a sentence's terminal punctuation
# (closing quotes / brackets) — stripped before the terminal test.
_CLOSERS = " \t\"'”’»)]}"
_TERMINALS = (".", "!", "?", "…", ":")

# Bullet markers. Dashes (– — -) are deliberately excluded: in French
# prose a leading em/en-dash marks dialogue, not a list item.
_BULLET_RE = re.compile(r"^[•·‣◦⁃\*∙]\s+")
# Numbered / lettered markers: "1.", "12)", "a)", "IV.". Kept short to
# avoid swallowing ordinary sentences that merely start with a capital.
_NUM_RE = re.compile(r"^(\d{1,3}|[ivxlcdmIVXLCDM]{1,7}|[a-zA-Z])[.)]\s+")

_HEAD_WORD_RE = re.compile(
    r"^(chapitre|chapter|section|partie|livre|titre|article|§|prologue|"
    r"introduction|conclusion|annexe|appendice)\b",
    re.IGNORECASE,
)
_ROMAN_HEAD_RE = re.compile(r"^[IVXLC]{1,6}[.—\-]?\s")
# Numbered section head, e.g. "1. — UN THÈME" or "2.) Le plan". Distinct
# from a list item by the dash and/or an ALL-CAPS tail (handled in the
# heading scorer, which runs before the list test).
_SECNUM_RE = re.compile(r"^(\d{1,3}|[IVXLC]{1,6}|[A-Za-z])[.)]\s*[—–\-]\s*\S")


def _is_terminated(text: str) -> bool:
    """True when ``text`` looks like the end of a sentence/clause —
    used to decide whether the next line continues it."""
    s = text.rstrip(_CLOSERS)
    if not s:
        return True
    return s.endswith(_TERMINALS)


def _ends_hyphen(text: str) -> bool:
    """True when the line ends with a soft word-split hyphen."""
    s = text.rstrip()
    return len(s) >= 2 and s[-1] in "-¬" and s[-2].isalpha()


def _looks_continuation(text: str) -> bool:
    """True when ``text`` looks like the tail of a paragraph started on a
    previous line: begins lowercase or with a clause-closing mark."""
    s = text.lstrip()
    if not s:
        return False
    return s[0].islower() or s[0] in "),;»”’"


def _looks_bold(text: str) -> bool:
    """Heuristic: short ALL-CAPS line with no lowercase chars and at
    least one letter. Captures markers like "CHAPITRE I" / "I. — TEXTE"
    without false-positiving short body sentences."""
    if len(text) > 60 or not any(c.isalpha() for c in text):
        return False
    if any(c.islower() for c in text):
        return False
    # Require ≥2 words: emphatic markers are phrases ("CHAPITRE I", "LA
    # DISTINCTION DES PERSONNES"). A lone ALL-CAPS token is usually a
    # garbled-Greek OCR artefact, not deliberate emphasis — don't bold it.
    if len(text.split()) < 2:
        return False
    upper_count = sum(1 for c in text if c.isupper())
    return upper_count >= 2


def _join_lines(prev: str, nxt: str) -> str:
    """Fuse two physical lines into running text, honouring soft
    hyphenation (trailing ``-``/``¬`` → glue with no space)."""
    if _ends_hyphen(prev):
        return prev.rstrip()[:-1] + nxt.lstrip()
    if not prev:
        return nxt
    return prev.rstrip() + " " + nxt.lstrip()


def _norm_running(text: str) -> str:
    """Signature for running header/footer detection: drop digits (page
    numbers vary) and punctuation, lowercase, collapse whitespace."""
    t = re.sub(r"\d+", "", text.lower())
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_page_number(text: str) -> bool:
    """A standalone page-number-ish token: only digits / roman numerals /
    light punctuation, short."""
    s = text.strip()
    if not s or len(s) > 12:
        return False
    return bool(re.fullmatch(r"[\d IVXLCDMivxlcdm.—\-–—()\[\]]+", s)) \
        and any(c.isdigit() or c.isalpha() for c in s)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _cluster_1d(values: list[float], k: int) -> list[list[float]]:
    """Split a sorted 1-D sample into up to ``k`` clusters at its largest
    gaps. Returns clusters in ascending order of value."""
    s = sorted(values)
    if len(s) <= 1 or k <= 1:
        return [s] if s else []
    gaps = sorted(
        range(1, len(s)),
        key=lambda i: s[i] - s[i - 1],
        reverse=True,
    )
    cuts = sorted(gaps[: k - 1])
    clusters: list[list[float]] = []
    start = 0
    for c in cuts:
        clusters.append(s[start:c])
        start = c
    clusters.append(s[start:])
    return clusters


def _build_heading_levels(ratios: list[float]):
    """From the height-ratios of all heading candidates across the
    document, build a ``ratio → level`` map (1 = biggest). Clustering the
    whole corpus keeps the same physical title at the same depth on every
    page, instead of letting per-page medians wobble the level."""
    uniq = [r for r in ratios if r > 0]
    if not uniq:
        return lambda r: 2
    clusters = _cluster_1d(uniq, 3)
    # Descending by centroid → tier 0 is the biggest type.
    centroids = [statistics.mean(c) for c in clusters if c]
    centroids.sort(reverse=True)
    tiers: list[tuple[float, int]] = []
    last_level = 0
    for rank, c in enumerate(centroids):
        floor = 1 if c >= 1.7 else 2 if c >= 1.35 else 3
        level = min(max(rank + 1, floor, last_level), 3)
        last_level = level
        tiers.append((c, level))

    def level_for(r: float) -> int:
        best = min(tiers, key=lambda t: abs(t[0] - r))
        return best[1]

    return level_for


# ── Line-page parsing & geometry ───────────────────────────────────

_MIN_CONFIDENCE = 0.3   # Apple Vision figure noise lives below this.


def _line_items(lines: list[dict]) -> list[dict]:
    """Filter usable lines (text + 4-tuple bbox + confidence) and
    precompute geometry."""
    items: list[dict] = []
    for ln in lines:
        text = (ln.get("text") or "").strip()
        bbox = ln.get("bbox")
        if not text or not bbox or len(bbox) != 4:
            continue
        conf = ln.get("confidence")
        if conf is not None:
            try:
                if float(conf) < _MIN_CONFIDENCE:
                    continue
            except (TypeError, ValueError):
                pass
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        items.append({
            "text": text,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "h": max(1.0, y1 - y0),
        })
    return items


def _reading_order(items: list[dict], page_w: float) -> list[dict]:
    """Order lines for reading. Detects a two-column layout (clear central
    gutter, both sides populated, nothing spanning) and reads left column
    fully before the right; otherwise top-to-bottom, left-to-right."""
    if len(items) < 6 or page_w <= 0:
        return sorted(items, key=lambda i: (i["y0"], i["x0"]))
    for frac in (0.5, 0.45, 0.55):
        split = page_w * frac
        left = [i for i in items if (i["x0"] + i["x1"]) / 2 < split]
        right = [i for i in items if (i["x0"] + i["x1"]) / 2 >= split]
        span = sum(1 for i in items if i["x0"] < split < i["x1"])
        if len(left) >= 3 and len(right) >= 3 and span <= max(1, 0.08 * len(items)):
            return (sorted(left, key=lambda i: (i["y0"], i["x0"]))
                    + sorted(right, key=lambda i: (i["y0"], i["x0"])))
    return sorted(items, key=lambda i: (i["y0"], i["x0"]))


def _page_geometry(items: list[dict], page_w: float, page_h: float) -> dict:
    """Page-level reference metrics + per-item gap/ratio enrichment.
    Mutates ``items`` in place (adds gap_above/gap_below/ratio)."""
    heights = [i["h"] for i in items]
    median_h = statistics.median(heights) if heights else 1.0
    left = _percentile([i["x0"] for i in items], 0.10)
    right = _percentile([i["x1"] for i in items], 0.90)

    gaps: list[float] = []
    for prev, cur in zip(items, items[1:]):
        g = cur["y0"] - prev["y1"]
        if g > 0:
            gaps.append(g)
    median_gap = statistics.median(gaps) if gaps else median_h * 0.4

    for idx, it in enumerate(items):
        it["ratio"] = it["h"] / median_h if median_h > 0 else 1.0
        # Edge lines border the page, not whitespace — use a neutral gap so
        # the first/last line doesn't earn a phantom "isolated" heading cue.
        it["gap_above"] = (it["y0"] - items[idx - 1]["y1"]) if idx > 0 else median_gap
        it["gap_below"] = (items[idx + 1]["y0"] - it["y1"]) if idx + 1 < len(items) else median_gap

    return {
        "page_w": page_w or (right - left) or 1.0,
        "page_h": page_h or 1.0,
        "median_h": median_h,
        "median_gap": median_gap,
        "left": left,
        "right": right,
        "para_gap": median_gap * 1.6,
        "indent_thresh": max(median_h * 1.0, (page_w or 1000) * 0.018),
        "short_thresh": (page_w or 1000) * 0.12,
    }


def _is_heading(it: dict, ctx: dict) -> bool:
    """Multi-cue heading test. Height ratio is the gate; isolation,
    centering, brevity, caps and explicit markers add weight."""
    ratio = it["ratio"]
    text = it["text"]
    if len(text) > 80:
        return False
    letters = [c for c in text if c.isalpha()]
    all_caps = len(letters) >= 2 and all(c.isupper() for c in letters)
    # A heading keyword, or a numbered section marker with an ALL-CAPS tail,
    # is decisive on its own — even when the OCR bbox is no taller than body
    # (caps set tight, headings sometimes in the body face).
    if _HEAD_WORD_RE.match(text) or (_SECNUM_RE.match(text) and all_caps):
        return True
    # Otherwise a smaller-than-body line is footnote apparatus, never a
    # heading; a same-height line still can be on the strength of non-size
    # cues (caps, centering, isolation).
    if ratio < 0.9:
        return False
    score = 0.0
    if ratio >= 1.6:
        score += 2.0
    elif ratio >= 1.3:
        score += 1.2
    elif ratio >= 1.15:
        score += 0.6
    if it["gap_above"] > ctx["median_gap"] * 1.4:
        score += 0.6
    if it["gap_below"] > ctx["median_gap"] * 1.4:
        score += 0.6
    left_sp = it["x0"] - ctx["left"]
    right_sp = ctx["right"] - it["x1"]
    if left_sp > ctx["page_w"] * 0.06 and abs(left_sp - right_sp) < ctx["page_w"] * 0.10:
        score += 0.8
    if len(text) <= 40:
        score += 0.4
    if all_caps:
        score += 0.8
    if _HEAD_WORD_RE.match(text):
        score += 1.0
    if _ROMAN_HEAD_RE.match(text):
        score += 0.4
    if _SECNUM_RE.match(text):
        score += 1.0
    if not _is_terminated(text):
        score += 0.2
    return score >= 1.8


def _mark_footnotes(items: list[dict], ctx: dict) -> None:
    """Flag the page's footnote apparatus by **region**, not per-line size.

    In a critical edition the notes sit in a block at the foot of the page,
    cut off from the body by a gap far larger than any inter-paragraph gap
    (an invisible rule / extra leading). We find that separator as the
    single largest ``gap_above`` in the bottom band that clearly outranks
    the next-largest gap, and mark everything below it as footnote text.
    A small-and-low fallback catches pages whose separator is unclear."""
    page_h = ctx["page_h"]
    median_gap = ctx["median_gap"]
    para_gap = ctx["para_gap"]

    lower = [(items[i]["gap_above"], i)
             for i in range(1, len(items))
             if items[i]["y0"] > page_h * 0.58]
    if len(lower) >= 1:
        lower.sort(reverse=True)
        top_gap, cut = lower[0]
        second = lower[1][0] if len(lower) > 1 else 0.0
        is_outlier = (top_gap >= 3 * median_gap
                      and top_gap >= 1.8 * max(second, para_gap))
        if is_outlier and (len(items) - cut) >= 2:
            for it in items[cut:]:
                it["footnote"] = True

    # Fallback: distinctly small lines low on the page are notes too.
    for it in items:
        if not it.get("footnote") and it["ratio"] < 0.8 and it["y0"] > page_h * 0.6:
            it["footnote"] = True


def _list_marker(text: str) -> str | None:
    """Strip a leading list marker, returning the remaining text, or None
    when the line isn't a list item."""
    m = _BULLET_RE.match(text)
    if m:
        return text[m.end():].strip()
    m = _NUM_RE.match(text)
    if m:
        return text[m.end():].strip()
    return None


def _finalize_para(p: dict) -> None:
    """Decide paragraph kind (para vs block-quote) and continuation flag
    from the accumulated geometry.

    A block quote indents *every* line from the left margin and runs over
    more than one line. That distinguishes it from a first-line alinéa
    (whose continuation lines return to the margin, so ``left_min`` tracks
    the margin) and from short one-off fragments (footnote bits, captions)
    that merely fail to reach the right margin."""
    inset_left = p["left_min"] > p["_ctx_left"] + p["_ctx_indent"]
    if inset_left and p["n_lines"] >= 2:
        p["type"] = "quote"
    p["open_end"] = not _is_terminated(p["text"])
    # Internal scratch no longer needed.
    for k in ("_ctx_left", "_ctx_right", "_ctx_indent", "left_min",
              "right_max", "last_x1", "n_lines"):
        p.pop(k, None)


def _render_page_blocks(items: list[dict], ctx: dict, origin: dict,
                        level_for) -> list[dict]:
    """Turn one page's enriched, reading-ordered lines into typed blocks
    (heading / para / quote / list / footnote)."""
    body = [it for it in items if not it.get("footnote")]
    foot = [it for it in items if it.get("footnote")]

    blocks: list[dict] = []
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur is not None:
            _finalize_para(cur)
            blocks.append(cur)
            cur = None

    def new_para(it, text):
        return {
            "type": "para", "text": text, **origin,
            "open_start": _looks_continuation(text),
            "left_min": it["x0"], "right_max": it["x1"], "last_x1": it["x1"],
            "n_lines": 1,
            "_ctx_left": ctx["left"], "_ctx_right": ctx["right"],
            "_ctx_indent": ctx["indent_thresh"],
        }

    def append_para(it, text):
        cur["text"] = _join_lines(cur["text"], text)
        cur["left_min"] = min(cur["left_min"], it["x0"])
        cur["right_max"] = max(cur["right_max"], it["x1"])
        cur["last_x1"] = it["x1"]
        cur["n_lines"] += 1

    for it in body:
        text = it["text"]
        if it.get("heading"):
            flush()
            blocks.append({"type": "heading", "level": level_for(it["ratio"]),
                           "text": text, **origin})
            continue
        rest = _list_marker(text)
        if rest is not None:
            flush()
            blocks.append({"type": "list", "text": rest, **origin})
            continue

        if cur is None:
            cur = new_para(it, text)
            continue

        cont = _looks_continuation(text) and not _is_terminated(cur["text"])
        big_gap = it["gap_above"] > ctx["para_gap"]
        # Alinéa: indented from the page margin *and* from the current
        # paragraph's own left edge. The second clause lets a block quote
        # (every line sharing one indent) stay a single paragraph instead
        # of fragmenting into one paragraph per line.
        indent = ((it["x0"] - ctx["left"]) > ctx["indent_thresh"]
                  and (it["x0"] - cur["left_min"]) > ctx["indent_thresh"] * 0.6)
        prev_short = cur["last_x1"] < ctx["right"] - ctx["short_thresh"]
        if big_gap or indent:
            flush()
            cur = new_para(it, text)
        elif prev_short and _is_terminated(cur["text"]) and not cont:
            flush()
            cur = new_para(it, text)
        else:
            append_para(it, text)
    flush()

    # Footnotes: fuse the apparatus lines into one blockquote per note. A
    # leading note number (or list marker) starts a fresh note; the rest is
    # joined as running text with soft-hyphen handling, so a single note
    # broken across several physical lines reads as one sentence.
    note = ""
    for it in foot:
        text = it["text"]
        if note and re.match(r"^\d{1,3}[.)]?\s", text):
            blocks.append({"type": "footnote", "text": note, **origin})
            note = ""
        note = _join_lines(note, text) if note else text
    if note:
        blocks.append({"type": "footnote", "text": note, **origin})

    return blocks


# ── Per-row classification ─────────────────────────────────────────

def _classify_row(result_json: str | None) -> dict | None:
    """Inspect one stored OCR run and return either an assembled-MD page
    (Paddle / Surya) or a parsed line page (Apple Vision &c)."""
    if not result_json:
        return None
    try:
        data = json.loads(result_json)
    except Exception:
        return None
    meta = data.get("meta") or {}
    md = meta.get("markdown")
    if isinstance(md, str) and md.strip():
        return {"kind": "assembled", "text": md.rstrip()}
    structure = meta.get("structure") or []
    engine = (data.get("engine") or "").lower()
    if structure and engine == "surya":
        chunks = _render_structure(structure)
        return {"kind": "assembled", "text": "\n".join(chunks).rstrip()}
    # Apple-Document structured tree (apple_docs, and apple_vision's
    # structured pass). Reading-order typed blocks — render directly;
    # strictly preferred over the geometric line heuristics below.
    document = meta.get("document") or []
    if document:
        chunks = _render_document(document)
        if chunks:
            return {"kind": "assembled", "text": "\n".join(chunks).rstrip()}
    items = _line_items(data.get("lines") or [])
    if not items:
        return None
    page_w = float(data.get("page_w") or 0.0)
    page_h = float(data.get("page_h") or 0.0)
    return {"kind": "lines", "items": items, "page_w": page_w, "page_h": page_h}


def _mark_running(pages: list[dict]) -> None:
    """Document-level pass: flag lines that recur in the same band across
    many pages (running headers/footers) or are standalone page numbers,
    so the renderer can drop them."""
    line_pages = [p for p in pages if p.get("kind") == "lines" and p["items"]]
    n = len(line_pages)
    top_sig: dict[str, int] = {}
    bot_sig: dict[str, int] = {}
    for p in line_pages:
        items = p["items"]
        for it in items[:2]:
            s = _norm_running(it["text"])
            if s:
                top_sig[s] = top_sig.get(s, 0) + 1
        for it in items[-2:]:
            s = _norm_running(it["text"])
            if s:
                bot_sig[s] = bot_sig.get(s, 0) + 1
    # Running heads are often chapter-scoped (a title repeated only across
    # that chapter's pages), so a document-fraction threshold misses them.
    # A short line recurring in the same edge band on ≥3 pages is almost
    # always a running head / folio, never body — use an absolute floor.
    thresh = 3 if n >= 3 else 1 << 30
    running_top = {s for s, c in top_sig.items() if c >= thresh}
    running_bot = {s for s, c in bot_sig.items() if c >= thresh}

    # Keep the first occurrence of a recurring head/foot (it's usually the
    # real chapter title on its opening page) and drop only the repeats.
    # Page numbers carry no content, so they're always dropped.
    kept: set[str] = set()
    for p in line_pages:
        items = p["items"]
        page_h = p["geom"]["page_h"]
        for pos, it in enumerate(items):
            top = pos < 2 or it["y0"] < page_h * 0.10
            bot = pos >= len(items) - 2 or it["y0"] > page_h * 0.90
            sig = _norm_running(it["text"])
            if _is_page_number(it["text"]) and (top or bot):
                it["running"] = True
            elif top and sig in running_top:
                if sig in kept:
                    it["running"] = True
                else:
                    kept.add(sig)
            elif bot and sig in running_bot:
                if sig in kept:
                    it["running"] = True
                else:
                    kept.add(sig)


def _render_line_pages(pages: list[dict]) -> None:
    """Geometry + heading-clustering + running-line passes, then render
    each line page to typed blocks. Stores blocks under ``p["blocks"]``."""
    # Pass 1: per-page geometry + reading order + heading/footnote flags.
    heading_ratios: list[float] = []
    for p in pages:
        if p.get("kind") != "lines":
            continue
        items = _reading_order(p["items"], p["page_w"])
        ctx = _page_geometry(items, p["page_w"], p["page_h"])
        _mark_footnotes(items, ctx)
        for it in items:
            it["heading"] = (not it.get("footnote")) and _is_heading(it, ctx)
            if it["heading"]:
                heading_ratios.append(it["ratio"])
        p["items"] = items
        p["geom"] = ctx

    # Pass 2: document-level passes that need every page at once.
    _mark_running(pages)
    level_for = _build_heading_levels(heading_ratios)

    # Pass 3: render, dropping running lines.
    for p in pages:
        if p.get("kind") != "lines":
            continue
        keep = [it for it in p["items"] if not it.get("running")]
        p["blocks"] = _render_page_blocks(
            keep, p["geom"], p["origin"], level_for)


# ── Cross-page assembly + write ────────────────────────────────────

def _same_origin(a: dict, b: dict) -> bool:
    return a["scan_id"] == b["scan_id"] and a.get("branch_path") == b.get("branch_path")


def _merge_paragraphs(blocks: list[dict]) -> list[dict]:
    """Stitch a paragraph that runs across a page break into one block.
    When the join crosses a scan/branch boundary, the new page's marker is
    recorded as an inline comment so flow is preserved."""
    out: list[dict] = []
    for b in blocks:
        if (out and b["type"] == "para" and out[-1]["type"] == "para"
                and out[-1].get("open_end") and b.get("open_start")):
            prev = out[-1]
            if not _same_origin(prev, b):
                prev.setdefault("inline_marks", []).append(
                    (len(prev["text"]), _inline_marker(b)))
            prev["text"] = _join_lines(prev["text"], b["text"])
            prev["open_end"] = b.get("open_end")
            continue
        out.append(b)
    return out


def _inline_marker(b: dict) -> str:
    bp = b.get("branch_path") or ""
    if bp:
        return f"<!-- scan #{b['scan_id']} · {bp} -->"
    return f"<!-- scan #{b['scan_id']} -->"


def _apply_inline_marks(b: dict) -> str:
    marks = b.get("inline_marks")
    if not marks:
        return b["text"]
    text = b["text"]
    for pos, comment in sorted(marks, key=lambda m: m[0], reverse=True):
        pos = max(0, min(pos, len(text)))
        text = f"{text[:pos]} {comment} {text[pos:]}".replace("  ", " ")
    return text


def _render_block(b: dict) -> list[str]:
    """One typed block → MD lines (no surrounding blank lines; the writer
    spaces blocks)."""
    t = b["type"]
    if t == "raw":
        return [b["text"]]
    text = _apply_inline_marks(b)
    if t == "heading":
        return [f"{'#' * b['level']} {text}"]
    if t == "list":
        return [f"- {text}"]
    if t in ("quote", "footnote"):
        return [f"> {text}"]
    # para: surface ALL-CAPS short single-line emphasis as bold.
    if _looks_bold(text):
        return [f"**{text}**"]
    return [text]


def write_markdown(conn: sqlite3.Connection, output_path: Path, *,
                   refine: Optional[str] = None,
                   refine_mode: str = "cleanup") -> bool:
    """Dump every scan's OCR text into a Markdown file.

    Scan + branch markers go into HTML comments so the rendered text
    stays free-flowing. Returns False when no OCR data exists.

    ``refine`` optionally names an on-device LLM backend (e.g.
    ``"apple_fm"``) that post-processes the heuristic Markdown page-by-page
    to fix OCR errors — see ``aglaia.workers.ocr.llm_refine``. When the backend
    is unavailable (default, or pre-macOS-26) the heuristic output is kept
    verbatim, so this is always safe to pass."""
    output_path = Path(output_path)
    rows = conn.execute("""
        SELECT s.id AS scan_id, s.idx AS scan_idx,
               b.branch_path AS branch_path,
               o.result_json AS result_json
          FROM scans s
          JOIN branches b ON b.scan_id = s.id
          JOIN (
              SELECT o.*
                FROM ocr_runs o
                JOIN (
                    SELECT scan_id, branch_path, MAX(version) AS v
                      FROM ocr_runs WHERE status = 'done'
                  GROUP BY scan_id, branch_path
                ) m ON m.scan_id = o.scan_id
                   AND m.branch_path = o.branch_path
                   AND m.v = o.version
          ) o ON o.scan_id = b.scan_id AND o.branch_path = b.branch_path
         WHERE s.deleted_at IS NULL
           AND b.trashed_at IS NULL
         ORDER BY s.page_order ASC, s.idx ASC, b.branch_path ASC
    """).fetchall()
    if not rows:
        return False

    # Parse + classify every row into an ordered list of pages.
    pages: list[dict] = []
    for r in rows:
        cls = _classify_row(r["result_json"])
        if cls is None:
            continue
        origin = {"scan_id": r["scan_id"], "scan_idx": r["scan_idx"],
                  "branch_path": r["branch_path"] or ""}
        cls["origin"] = origin
        pages.append(cls)
    if not pages:
        return False

    # Document-level rendering of the line-based pages.
    _render_line_pages(pages)

    # Flatten to a single block stream in page order.
    blocks: list[dict] = []
    for p in pages:
        origin = p["origin"]
        if p["kind"] == "assembled":
            if p["text"]:
                blocks.append({"type": "raw", "text": p["text"], **origin})
        else:
            blocks.extend(p.get("blocks", []))
    if not blocks:
        return False

    blocks = _merge_paragraphs(blocks)

    slug = _project_slug(conn)
    out_lines: list[str] = [f"<!-- aglaia-export: {slug} -->", ""]
    cur_scan = None
    cur_branch = None
    in_list = False
    for b in blocks:
        if b["scan_id"] != cur_scan:
            out_lines.append(f"<!-- scan #{b['scan_id']} · page {b['scan_idx']} -->")
            cur_scan = b["scan_id"]
            cur_branch = None
        bp = b.get("branch_path") or ""
        if bp and bp != cur_branch:
            out_lines.append(f"<!-- branch {bp} -->")
            cur_branch = bp

        is_list = b["type"] == "list"
        if in_list and not is_list:
            out_lines.append("")
        in_list = is_list

        for line in _render_block(b):
            out_lines.append(line.rstrip())
        if not is_list:
            out_lines.append("")

    md_text = "\n".join(out_lines).rstrip() + "\n"

    # Optional on-device LLM cleanup pass — no-op when unavailable.
    if refine:
        try:
            from aglaia.workers.ocr.llm_refine import get_backend, refine_markdown_text
            backend = get_backend(refine)
            ok, reason = backend.available()
            if ok:
                md_text = refine_markdown_text(md_text, backend, refine_mode)
        except Exception:
            pass  # never let refinement break a working export

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding="utf-8")
    return True
