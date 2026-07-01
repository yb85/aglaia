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


def document_markers(mds: "list[str]", mode: str = "numeric") -> set[str]:
    """Doc-wide footnote markers = (∪ superscript refs) ∩ (∪ line-start entries)
    − ordinals, over ALL pages. A ref (``$^{1}$``) and its definition (``1.``)
    routinely land on DIFFERENT pages, so the intersection MUST be document-wide
    — a per-page intersection drops those, leaving refs as ``^1^`` exponents."""
    allsup: set[str] = set()
    allent: set[str] = set()
    for md in mds:
        s, e = _sup_and_entries(md or "", mode)
        allsup |= s
        allent |= e
    return {x for x in (allsup & allent) if x.lower() not in _ORDINAL}


def convert_footnotes(md: str, mode: str = "numeric", *,
                      markers: "set[str] | None" = None) -> str:
    """Superscript refs (LaTeX ``$^{N}$`` **and** Unicode ``¹⁷⁰``) + line-start
    entries → GFM footnotes. ``markers`` = the confirmed footnote markers; pass
    a **document-wide** set (see ``document_markers``) so refs/definitions that
    straddle pages still convert. When ``None``, falls back to a per-page
    intersection (fine for single-page use)."""
    cls = _marker_class(mode)
    md, _ = _norm_usup(md, mode)   # fold Unicode superscripts into $^{…}$ form
    if markers is None:
        sup, entries = _sup_and_entries(md, mode)
        markers = {x for x in (sup & entries) if x.lower() not in _ORDINAL}

    def repl_sup(m: re.Match[str]) -> str:
        c = m.group(1).strip()
        return f"[^{c}]" if c in markers else f"^{c}^"

    md = _SUP_RE.sub(repl_sup, md)

    out: list[str] = []
    for line in md.split("\n"):
        m = re.match(r"^\s*\[\^([^\]]+)\]\s*[.:]?\s+(.*)$", line)
        if m and m.group(1) in markers:
            out.append(f"[^{m.group(1)}]: {m.group(2)}")
            continue
        m = re.match(rf"^\s*({cls.pattern})\.\s+(.*)$", line)
        if m and m.group(1) in markers:
            out.append(f"[^{m.group(1)}]: {m.group(2)}")
            continue
        out.append(line)
    return "\n".join(out)


def postprocess_page(md: str, *, footnotes: str | None = "numeric") -> str:
    """Footnote-only per-page pass. ``footnotes`` = ``numeric`` | ``alphabetic``
    | None. Header/footer are handled by ``postprocess_mistral_page`` from the
    engine's API fields (no markdown heuristic)."""
    if footnotes:
        md = convert_footnotes(md, footnotes)
    return md


def postprocess_mistral_page(md: str, page, *, footnotes: str = "numeric",
                             headers: bool = True,
                             markers: "set[str] | None" = None) -> str:
    """One Mistral page → post-processed markdown. THE single implementation
    shared by the sync engine (``_assemble_results``) and the batch import
    (``mistral_batch.page_to_result``) — both must produce identical output.

    Footnote lift (LaTeX/Unicode superscript → GFM) + wrap the header/footer
    Mistral extracted via ``extract_header``/``extract_footer`` in HTML tags.
    Pass ``markers`` = ``document_markers(all_page_mds, footnotes)`` so footnotes
    whose ref/definition straddle pages still convert.
    """
    if footnotes in ("numeric", "alphabetic"):
        md = convert_footnotes(md, footnotes, markers=markers)
    if headers and isinstance(page, dict):
        hd = (page.get("header") or "").strip()
        ft = (page.get("footer") or "").strip()
        parts: list[str] = []
        if hd:
            parts.append(f'<header class="page-header">\n\n{hd}\n\n</header>')
        parts.append(md)
        if ft:
            parts.append(f'<footer class="page-footer">\n\n{ft}\n\n</footer>')
        md = "\n\n".join(parts)
    return md


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


def batch_markers(pages) -> "set[str] | None":
    """Document-wide footnote markers for a batch's page objects (honoring the
    card's footnote toggle). ``None`` when footnotes are off. The batch import
    loops pages one at a time, so it precomputes this once and passes it to
    every ``page_to_result`` — refs/definitions straddle pages."""
    fn, _ = mistral_settings()
    if fn not in ("numeric", "alphabetic"):
        return None
    mds = [(p.get("markdown", "") if isinstance(p, dict) else str(p or ""))
           for p in pages]
    return document_markers(mds, fn)
