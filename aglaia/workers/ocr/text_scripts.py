# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Language → script inference + unexpected-script (garbage) detection.

Coverage is the **full ISO/Unicode set**, not tailored to any engine's language
list — a plugin VLM may add other languages, and a document in an unknown
language but a known script still OCRs fine. So we work at the *script* level:

* ``langcodes`` (CLDR likely-subtags) maps any BCP-47 / ISO 639-1/2/3 language
  code — with or without an explicit script subtag — to its ISO 15924 script;
* the ``regex`` module's Unicode Script property (``\\p{sc=…}``) tests each
  character.

Why this exists: **Apple Vision, handed a script it doesn't support, falls back
to a confusable script it does — ancient Greek → Cyrillic — and reports it with
HIGH confidence.** The confidence gate alone therefore leaks those misreads. A
line carrying letters from a script none of the chosen languages cover is a
near-certain misread (measured 100% precision on a fr+el corpus) and should be
offloaded to a complement engine regardless of confidence.
"""

from __future__ import annotations

import functools

import langcodes
import regex

# ISO 15924 *aggregate* codes langcodes emits that regex's \p{sc=…} rejects
# (they name writing systems, not Unicode scripts) → their component scripts.
_AGGREGATE: dict[str, frozenset[str]] = {
    "Hans": frozenset({"Hani"}),                 # Simplified Chinese → Han
    "Hant": frozenset({"Hani"}),                 # Traditional Chinese → Han
    "Jpan": frozenset({"Hani", "Hira", "Kana"}),  # Japanese → Han+Hiragana+Katakana
    "Kore": frozenset({"Hang", "Hani"}),          # Korean → Hangul+Han
}

# CLDR likely-subtags quirks worth overriding for OCR. Ancient Greek (grc)
# maximises to the Cypriot syllabary (Cprt); real patristic/classical text is
# in the Greek script.
_LANG_SCRIPT_OVERRIDE: dict[str, str] = {"grc": "Grek"}


@functools.lru_cache(maxsize=1024)
def scripts_for_language(code: str) -> frozenset[str]:
    """ISO 15924 script codes (regex ``\\p{sc=…}``-compatible) a language uses.

    Honours an explicit BCP-47 script subtag (``sr-Cyrl`` → Cyrl), else infers
    it from CLDR likely-subtags. Aggregate writing systems (Japanese, Korean,
    Chinese) expand to their component scripts. Unknown/``auto`` → empty."""
    code = (code or "").strip()
    if not code or code.lower() in ("auto", "und", "mul"):
        return frozenset()
    try:
        lang = langcodes.Language.get(code)
    except Exception:
        return frozenset()
    sc = lang.script  # explicit script subtag, if any
    if not sc:
        sc = _LANG_SCRIPT_OVERRIDE.get((lang.language or "").lower())
    if not sc:
        try:
            sc = lang.maximize().script
        except Exception:
            sc = None
    if not sc:
        return frozenset()
    return _AGGREGATE.get(sc, frozenset({sc}))


@functools.lru_cache(maxsize=512)
def expected_scripts(languages: tuple[str, ...]) -> frozenset[str]:
    """Union of the scripts implied by the chosen OCR language codes."""
    out: set[str] = set()
    for code in languages:
        out |= scripts_for_language(code)
    return frozenset(out)


@functools.lru_cache(maxsize=512)
def _unexpected_matcher(expected: frozenset[str]) -> "regex.Pattern | None":
    """Compiled matcher for a LETTER outside every expected script.

    Uses the regex module's VERSION1 set subtraction: ``[\\p{L}--<expected>]``.
    Returns None when no scripts are expected (nothing to judge against)."""
    if not expected:
        return None
    minus = "".join(rf"\p{{sc={s}}}" for s in sorted(expected))
    return regex.compile(rf"[\p{{L}}--{minus}]", regex.V1)


def has_unexpected_script(text: str, languages, min_chars: int = 2) -> bool:
    """True if ``text`` carries ≥ ``min_chars`` letters from a script none of
    the chosen ``languages`` cover.

    Empty/``auto`` languages → False: with no expected scripts declared there is
    nothing to judge against (no hidden domain default — the caller supplies the
    language context, e.g. the OCR tab's language picker)."""
    matcher = _unexpected_matcher(expected_scripts(tuple(languages or ())))
    if matcher is None:
        return False
    n = 0
    for _ in matcher.finditer(text):
        n += 1
        if n >= min_chars:
            return True
    return False


def language_hint(languages) -> str:
    """Human-readable language list for a VLM prompt, e.g. 'French, Modern
    Greek'. Empty when no concrete languages are given."""
    names: list[str] = []
    for code in (languages or ()):
        c = (code or "").strip()
        if not c or c.lower() in ("auto", "und", "mul"):
            continue
        try:
            names.append(langcodes.Language.get(c).display_name())
        except Exception:
            names.append(c)
    # de-dup, preserve order
    seen: set[str] = set()
    uniq = [n for n in names if not (n in seen or seen.add(n))]
    return ", ".join(uniq)
