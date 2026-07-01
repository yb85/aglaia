# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Markdown post-processing for OCR output (Mistral first).

Cloud OCR (Mistral) returns clean per-page Markdown but keeps LaTeX-math
superscripts (``$^{170}$``), inline page furniture (running heads, page
numbers), and footnote *entries* as plain ``170. …`` lines. This module lifts
that into structured Markdown, **one page at a time** (footnote numbering
resets per page in many books):

1. **Footnotes** — a marker is treated as a footnote only when it appears BOTH
   as a superscript AND as a line-start entry (intersection disambiguation).
   That spares French ordinals (``XV$^{e}$`` with no ``e.`` entry → stays a
   ``^e^`` superscript) and numbered lists (``1.`` with no ``$^{1}$`` reference
   → stays a list). Superscript refs → GFM ``[^N]``; entries → ``[^N]: …``;
   non-footnote superscripts → pandoc ``^x^``. ``mode`` = ``numeric`` (digit
   markers) or ``alphabetic`` (letter markers — for books that letter their
   footnotes; ordinal ``e``/``er``/… are still filtered by the entry gate).

2. **Header / footer** — no engine gives us semantic blocks, so heuristically:
   a bare-number line = the page number; a short ALL-CAPS line at the page top =
   the running head. Wrapped in ``<header>``/``<footer>`` with CSS classes.

Designed to be extended (more passes) behind the same per-page entry point.
"""

from __future__ import annotations

import re

# Mistral marks footnote references TWO ways, mixed within one document:
#   • LaTeX inline math:  $^{170}$   (and \( ^{170} \))
#   • Unicode superscript glyphs:  ¹⁷⁰   (⁰¹²³⁴⁵⁶⁷⁸⁹, incl. ¹²³ from Latin-1)
# We normalise both to a plain marker before deciding footnote-hood.
_SUP_RE = re.compile(r"\$\^\{([^}]*)\}\$")
_USUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_USUP_MAP = str.maketrans(_USUP_DIGITS, "0123456789")
# Unicode superscript LETTERS (for alphabetic footnotes / ordinals): map the
# common ones back to ASCII so ᵃ/ᵇ/… and the French ordinal ᵉ are handled.
_USUP_ALPHA = {"ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f",
               "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l",
               "ᵐ": "m", "ⁿ": "n", "ᵒ": "o", "ᵖ": "p", "ʳ": "r", "ˢ": "s",
               "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z"}
_USUP_ALPHA_MAP = str.maketrans(_USUP_ALPHA)
_USUP_NUM_RE = re.compile(f"[{_USUP_DIGITS}]+")
_USUP_ALPHA_RE = re.compile(f"[{''.join(_USUP_ALPHA)}]+")
# French ordinal suffixes that ride a numeral/roman as a superscript, never a
# footnote — extra guard on top of the entry-intersection gate.
_ORDINAL = {"e", "er", "re", "ère", "nd", "d", "es", "ème", "èmes", "os"}


def _marker_class(mode: str) -> re.Pattern[str]:
    # numeric: 1+ digits; alphabetic: 1–2 letters (a … zz), case-insensitive.
    return re.compile(r"\d+" if mode == "numeric" else r"[A-Za-z]{1,2}")


def _norm_usup(md: str, mode: str) -> tuple[str, set[str]]:
    """Normalise Unicode-superscript refs (¹⁷⁰, ᵃ) to ``$^{…}$`` so the single
    LaTeX path below handles them. Returns (rewritten md, seen markers)."""
    seen: set[str] = set()
    usup_re, tr = ((_USUP_NUM_RE, _USUP_MAP) if mode == "numeric"
                   else (_USUP_ALPHA_RE, _USUP_ALPHA_MAP))

    def repl(m: re.Match[str]) -> str:
        v = m.group(0).translate(tr)
        seen.add(v)
        return f"$^{{{v}}}$"

    return usup_re.sub(repl, md), seen


def _sup_and_entries(md: str, mode: str) -> tuple[set[str], set[str]]:
    """One page's (superscript refs, line-start entries) after Unicode folding."""
    cls = _marker_class(mode)
    md, _ = _norm_usup(md, mode)
    sup = {m.group(1).strip() for m in _SUP_RE.finditer(md)
           if cls.fullmatch(m.group(1).strip())}
    entry_num = set(re.findall(rf"^\s*({cls.pattern})\.\s", md, re.M))
    entry_sup = {m.strip() for m in re.findall(
        rf"^\s*\$\^\{{({cls.pattern})\}}\$", md, re.M)}
    return sup, (entry_num | entry_sup)


def _page_text(page) -> str:
    """A page's full footnote-bearing text = body markdown + header + footer
    (Mistral puts note *definitions* in the extracted footer field)."""
    if isinstance(page, dict):
        return "\n".join([page.get("markdown", "") or "",
                          page.get("header", "") or "",
                          page.get("footer", "") or ""])
    return str(page or "")


def windowed_markers(pages, mode: str = "numeric",
                     window: int = 1) -> "list[set[str] | None]":
    """Per-page footnote markers, paired within a ±``window``-page span.

    The ref (``$^{1}$``, body) and its definition (``1.``, footer field) sit on
    the SAME page, and footnote numbers reset per page — so pairing is LOCAL, not
    document-wide (a global set would wrongly link page 2's "1" to page 500's
    "1"). ±1 page tolerates a note spilling to the next page / a ref near a page
    edge. Returns one marker set per page (``None`` when footnotes are off)."""
    n = len(pages)
    if mode not in ("numeric", "alphabetic"):
        return [None] * n
    per = [_sup_and_entries(_page_text(p), mode) for p in pages]
    out: list[set[str] | None] = []
    for i in range(n):
        sup: set[str] = set()
        ent: set[str] = set()
        for j in range(max(0, i - window), min(n, i + window + 1)):
            sup |= per[j][0]
            ent |= per[j][1]
        out.append({x for x in (sup & ent) if x.lower() not in _ORDINAL})
    return out


def document_markers(mds: "list[str]", mode: str = "numeric") -> set[str]:
    """Doc-wide footnote markers = (∪ superscript refs) ∩ (∪ line-start entries)
    − ordinals, over ALL pages. Kept for single-shot / non-paged callers;
    Mistral pages use ``windowed_markers`` (local pairing) instead."""
    allsup: set[str] = set()
    allent: set[str] = set()
    for md in mds:
        s, e = _sup_and_entries(md or "", mode)
        allsup |= s
        allent |= e
    return {x for x in (allsup & allent) if x.lower() not in _ORDINAL}


def _ordered_footnote_numbers(page, mode: str,
                              markers: "set[str]") -> "list[str]":
    """Footnote numbers on this page (ref or definition) that ARE markers, in
    order of first appearance — so anchors are assigned in reading order."""
    text, _ = _norm_usup(_page_text(page), mode)
    cls = _marker_class(mode)
    pat = re.compile(rf"\$\^\{{({cls.pattern})\}}\$|^\s*({cls.pattern})\.",
                     re.M)
    seen: list[str] = []
    s: set[str] = set()
    for m in pat.finditer(text):
        n = m.group(1) or m.group(2)
        if n in markers and n not in s:
            s.add(n)
            seen.append(n)
    return seen


def assign_page_mappings(pages, mode: str = "numeric",
                         window: int = 1) -> "list[dict[str, str] | None]":
    """Per-page ``{original_number: unique_anchor}`` maps. Footnote numbers reset
    per chapter, so the SAME number recurs; each occurrence gets a unique anchor
    that keeps the original number (``1`` first, then ``1-2``, ``1-3``, …) so GFM
    links every ref to ITS own note instead of collapsing all ``[^1]`` onto the
    first. Pages are processed in order (stateful occurrence counter)."""
    n = len(pages)
    if mode not in ("numeric", "alphabetic"):
        return [None] * n
    marker_sets = windowed_markers(pages, mode, window)
    seen: dict[str, int] = {}
    out: list[dict[str, str] | None] = []
    for page, markers in zip(pages, marker_sets):
        mapping: dict[str, str] = {}
        for num in _ordered_footnote_numbers(page, mode, markers or set()):
            k = seen.get(num, 0) + 1
            seen[num] = k
            mapping[num] = num if k == 1 else f"{num}-{k}"
        out.append(mapping)
    return out


def convert_footnotes(md: str, mode: str = "numeric", *,
                      mapping: "dict[str, str] | None" = None,
                      markers: "set[str] | None" = None) -> str:
    """Superscript refs (LaTeX ``$^{N}$`` **and** Unicode ``¹⁷⁰``) + line-start
    entries → GFM footnotes.

    ``mapping`` maps each footnote's original number → its **unique anchor**
    (e.g. ``{"1": "1", ...}`` first time, ``{"1": "1-2"}`` on the next chapter's
    reset) so repeated numbers don't collide in GFM while the anchor still shows
    the original number. ``markers`` (a set) is the back-compat shorthand for an
    identity mapping. ``None`` → per-page intersection (single-page use)."""
    cls = _marker_class(mode)
    md, _ = _norm_usup(md, mode)   # fold Unicode superscripts into $^{…}$ form
    if mapping is None:
        if markers is None:
            sup, entries = _sup_and_entries(md, mode)
            markers = {x for x in (sup & entries) if x.lower() not in _ORDINAL}
        mapping = {x: x for x in markers}

    # Pass 1: line-start ENTRIES ("N. …" or "$^{N}$ …") → "[^anchor]: …".
    # Done before inline refs so a definition led by its own superscript marker
    # is treated as a definition, not a reference.
    out: list[str] = []
    entry_re = re.compile(
        rf"^\s*(?:\$\^\{{({cls.pattern})\}}\$|({cls.pattern})\.)\s+(.*)$")
    for line in md.split("\n"):
        m = entry_re.match(line)
        if m:
            num = m.group(1) or m.group(2)
            if num in mapping:
                out.append(f"[^{mapping[num]}]: {m.group(3)}")
                continue
        out.append(line)
    md = "\n".join(out)

    # Pass 2: inline REFS. A mapped number → its anchor; else a bare exponent.
    def repl_sup(m: re.Match[str]) -> str:
        c = m.group(1).strip()
        return f"[^{mapping[c]}]" if c in mapping else f"^{c}^"

    return _SUP_RE.sub(repl_sup, md)


def postprocess_page(md: str, *, footnotes: str | None = "numeric") -> str:
    """Footnote-only per-page pass. ``footnotes`` = ``numeric`` | ``alphabetic``
    | None. Header/footer are handled by ``postprocess_mistral_page`` from the
    engine's API fields (no markdown heuristic)."""
    if footnotes:
        md = convert_footnotes(md, footnotes)
    return md


def _split_defs(text: str) -> tuple[str, str]:
    """(footnote-definition lines ``[^N]: …``, remaining-furniture lines)."""
    defs: list[str] = []
    other: list[str] = []
    for line in text.split("\n"):
        (defs if re.match(r"^\s*\[\^[^\]]+\]:", line) else other).append(line)
    return "\n".join(defs).strip(), "\n".join(other).strip()


def postprocess_mistral_page(md: str, page, *, footnotes: str = "numeric",
                             headers: bool = True,
                             mapping: "dict[str, str] | None" = None) -> str:
    """One Mistral page → post-processed markdown. THE single implementation
    shared by the sync engine (``_assemble_results``) and the batch import
    (``mistral_batch.page_to_result``) — both must produce identical output.

    * Footnote lift (LaTeX/Unicode superscript → GFM) in the body.
    * ``extract_header``/``extract_footer`` fields: Mistral lumps the footnote
      *definitions* into the footer (they sit at the page bottom), so we run the
      footnote lift there too and emit the resulting ``[^N]:`` definitions as
      real footnotes (NOT buried inside ``<footer>``, which would break GFM);
      only genuine furniture (running head, page number) is wrapped.

    ``mapping`` = this page's ``{original_number: unique_anchor}`` map (see
    ``assign_page_mappings``) — the ref and its definition are on the same page,
    numbers reset per chapter, so each occurrence gets a unique anchor that still
    carries the original number.
    """
    if footnotes in ("numeric", "alphabetic"):
        md = convert_footnotes(md, footnotes, mapping=mapping)
    if not headers or not isinstance(page, dict):
        return md

    defs: list[str] = []
    furn = {"header": "", "footer": ""}
    for key in ("header", "footer"):
        text = (page.get(key) or "").strip()
        if not text:
            continue
        if footnotes in ("numeric", "alphabetic"):
            text = convert_footnotes(text, footnotes, mapping=mapping)
        d, rest = _split_defs(text)
        if d:
            defs.append(d)
        furn[key] = rest

    parts: list[str] = []
    if furn["header"]:
        parts.append(
            f'<header class="page-header">\n\n{furn["header"]}\n\n</header>')
    parts.append(md)
    if defs:
        parts.append("\n".join(defs))   # footnote definitions → real footnotes
    if furn["footer"]:
        parts.append(
            f'<footer class="page-footer">\n\n{furn["footer"]}\n\n</footer>')
    return "\n\n".join(parts)


def mistral_settings() -> tuple[str, bool]:
    """(footnotes_mode, headers) from the config DB — the Mistral card toggles.
    Best-effort; defaults ``("numeric", True)``. Used by the batch import path,
    which has no engine instance to read the toggles from."""
    try:
        from aglaia.app_data import db as _cfg
        with _cfg.session() as _c:
            return (str(_cfg.get(_c, _cfg.KEY_MISTRAL_FOOTNOTES, "numeric")),
                    bool(_cfg.get(_c, _cfg.KEY_MISTRAL_HEADERS, True)))
    except Exception:
        return "numeric", True


def batch_mappings(pages) -> "list[dict[str, str] | None]":
    """Per-page ``{number: unique_anchor}`` maps for a batch's pages (honoring
    the card's footnote toggle). The batch import loops pages one at a time, so
    it precomputes this list once and passes ``mappings[i]`` to each
    ``page_to_result``."""
    fn, _ = mistral_settings()
    return assign_page_mappings(pages, fn)
