# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Footnote lift in `workers/ocr/md_postprocess.py`.

The load-bearing case is the same-line one. A footnote is recognised by the
intersection of *superscript refs in the body* and *line-start entries in the
footer*. Critical editions pack several notes onto one physical line —

    (12) premier. (13) second.

— so the second marker never sits at a line start, never enters the entry set,
and is therefore never classified as a footnote **at all**: its ref stays a
bare `(13)` in the body and its text stays glued to note 12.
"""
import pytest

from aglaia.workers.ocr import md_postprocess as mp


PACKED = "(12) premier. (13) second. (14) troisième."
BODY = "Texte (12) avec (13) trois (14) appels."


def test_a_packed_line_hides_every_marker_after_the_first():
    """Without the flag, only the line-START marker is an entry — this is the
    behaviour the same-line handling exists to fix, pinned so the fix is
    visibly a change and not a no-op."""
    _sup, entries = mp._sup_and_entries(PACKED, "numeric")
    assert entries == {"12"}


def test_same_line_finds_the_markers_packed_after_it():
    _sup, entries = mp._sup_and_entries(PACKED, "numeric", same_line=True)
    assert entries == {"12", "13", "14"}


def test_same_line_only_looks_inside_lines_that_start_with_an_entry():
    """A marker inside ordinary prose is a REFERENCE, not a definition —
    promoting it would invent footnotes out of citations."""
    prose = "Comme le note (12) l'auteur, voir aussi (13) plus bas."
    _sup, entries = mp._sup_and_entries(prose, "numeric", same_line=True)
    assert entries == set()


def test_convert_splits_a_packed_definition_line():
    md = f"{BODY}\n\n{PACKED}"
    mapping = {"12": "12", "13": "13", "14": "14"}
    out = mp.convert_footnotes(md, "numeric", mapping=mapping, same_line=True)
    assert "[^12]: premier." in out
    assert "[^13]: second." in out
    assert "[^14]: troisième." in out
    # …and the body refs became links, not literals.
    assert "Texte [^12] avec [^13] trois [^14] appels." in out


def test_without_the_flag_the_packed_line_stays_glued():
    md = f"{BODY}\n\n{PACKED}"
    mapping = {"12": "12", "13": "13", "14": "14"}
    out = mp.convert_footnotes(md, "numeric", mapping=mapping)
    assert "[^12]: premier. [^13] second. [^14] troisième." in out
    assert "[^13]: " not in out


def test_only_mapped_markers_cut_a_line():
    """A stray citation inside a note must NOT split it — `(3)` here is not a
    footnote on this page, so note 12 keeps its text intact."""
    md = "Texte (12) ici.\n\n(12) voir Migne (3) col. 44. (13) second."
    mapping = {"12": "12", "13": "13"}
    out = mp.convert_footnotes(md, "numeric", mapping=mapping, same_line=True)
    assert "[^12]: voir Migne (3) col. 44." in out
    assert "[^13]: second." in out


def test_bare_numeric_forms_never_cut_mid_line():
    """`N.` is far too ambiguous inside running text — dates, verse
    references, enumerations. Only superscript / parenthesised forms cut."""
    md = "Texte $^{7}$ ici.\n\n7. voir Jn 3. 16 et Lc 4. 18 pour le contexte."
    mapping = {"7": "7", "3": "3", "4": "4"}
    out = mp.convert_footnotes(md, "numeric", mapping=mapping, same_line=True)
    assert "[^7]: voir Jn 3. 16 et Lc 4. 18 pour le contexte." in out
    assert "[^3]:" not in out and "[^4]:" not in out


def test_same_line_is_off_by_default_everywhere():
    """Opt-in: the flag changes how a page is segmented, so it must not alter
    existing exports until it is asked for."""
    md = f"{BODY}\n\n{PACKED}"
    mapping = {"12": "12", "13": "13", "14": "14"}
    assert (mp.convert_footnotes(md, "numeric", mapping=mapping)
            == mp.convert_footnotes(md, "numeric", mapping=mapping,
                                    same_line=False))


def test_windowed_markers_threads_the_flag():
    pages = [{"markdown": BODY, "header": "", "footer": PACKED}]
    plain = mp.windowed_markers(pages, "numeric")
    same = mp.windowed_markers(pages, "numeric", same_line=True)
    assert plain[0] == {"12"}
    assert same[0] == {"12", "13", "14"}


@pytest.mark.parametrize("mode", ["numeric", "alphabetic"])
def test_superscript_packed_definitions_split_in_both_modes(mode):
    marker = ("$^{2}$", "$^{3}$") if mode == "numeric" else ("$^{b}$", "$^{c}$")
    a, b = marker
    md = f"Texte {a} et {b}.\n\n{a} premier. {b} second."
    keys = ("2", "3") if mode == "numeric" else ("b", "c")
    mapping = {k: k for k in keys}
    out = mp.convert_footnotes(md, mode, mapping=mapping, same_line=True)
    assert f"[^{keys[0]}]: premier." in out
    assert f"[^{keys[1]}]: second." in out


def test_the_setting_reaches_the_export(monkeypatch):
    """The flag has to be reachable or it is dead code. `mistral_settings`
    reads it from the config DB and `write_markdown` threads it through both
    the mapping pass and the per-page pass."""
    import inspect
    from aglaia.app_data import db as cfg
    from aglaia.workers import md_export

    assert hasattr(cfg, "KEY_MISTRAL_SAME_LINE")
    assert cfg.BUILTIN_DEFAULTS[cfg.KEY_MISTRAL_SAME_LINE] is False, "must be opt-in"

    mode, headers, same_line = mp.mistral_settings()
    assert isinstance(same_line, bool)

    src = inspect.getsource(md_export.write_markdown)
    assert "same_line=same_line" in src, "export does not thread the flag"


def test_mistral_settings_degrades_when_the_config_db_is_unreachable(monkeypatch):
    """Export must not fail because a setting can't be read."""
    import aglaia.app_data.db as cfg

    def boom(*a, **k):
        raise RuntimeError("no db")

    monkeypatch.setattr(cfg, "session", boom)
    assert mp.mistral_settings() == ("numeric", True, False)
