# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Unit tests for the line-geometry → Markdown heuristics in
``lib.workers.md_export``.

These exercise the document-level pipeline directly on synthetic line
data (no DB, no OCR engine), one test per heuristic from issue #52.
"""
from __future__ import annotations

import pytest

from lib.workers import md_export as mx


# ── Synthetic-page helpers ─────────────────────────────────────────

def _line(text, y0, *, x0=100, x1=900, h=30, conf=0.99):
    return {"text": text, "bbox": [x0, y0, x1, y0 + h], "confidence": conf}


def _page(lines, *, scan_id=1, scan_idx=0, branch="", page_w=1000, page_h=1400):
    """Build a 'lines' page dict in the shape ``_render_line_pages`` wants."""
    items = mx._line_items(lines)
    return {
        "kind": "lines", "items": items,
        "page_w": page_w, "page_h": page_h,
        "origin": {"scan_id": scan_id, "scan_idx": scan_idx, "branch_path": branch},
    }


def _blocks_of(lines, **kw):
    p = _page(lines, **kw)
    mx._render_line_pages([p])
    return p["blocks"]


def _texts(blocks):
    return [b["text"] for b in blocks]


# ── Text-shape helpers ─────────────────────────────────────────────

def test_is_terminated():
    assert mx._is_terminated("Fin de phrase.")
    assert mx._is_terminated('Il dit : « Bonjour. »')
    assert mx._is_terminated("Question ?")
    assert not mx._is_terminated("une phrase qui continue")
    assert not mx._is_terminated("mot coupé-")


def test_ends_hyphen_and_continuation():
    assert mx._ends_hyphen("conti-")
    assert mx._ends_hyphen("inté¬")
    assert not mx._ends_hyphen("fin -")          # space before: not a soft split
    assert mx._looks_continuation("nuation du mot")
    assert not mx._looks_continuation("Majuscule début")


def test_join_lines_dehyphenates():
    assert mx._join_lines("conti-", "nuation") == "continuation"
    assert mx._join_lines("inté¬", "rieur") == "intérieur"
    assert mx._join_lines("deux", "mots") == "deux mots"


def test_looks_bold():
    assert mx._looks_bold("CHAPITRE I")
    assert not mx._looks_bold("Chapitre premier")
    assert not mx._looks_bold("X" * 80)


# 1. Dehyphenation + line joining → one paragraph
def test_paragraph_join_and_dehyphenation():
    blocks = _blocks_of([
        _line("Voici une longue phrase qui se pour-", 100, x1=900),
        _line("suit naturellement sur la ligne suivante", 135, x1=600),
    ])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "para"
    assert "poursuit naturellement" in blocks[0]["text"]


# 2. Paragraph break on first-line indent (alinéa)
def test_indent_starts_new_paragraph():
    blocks = _blocks_of([
        _line("Premier paragraphe qui occupe toute la largeur de la colonne ici.", 100, x0=100, x1=900),
        _line("Deuxième paragraphe marqué par un alinéa en retrait.", 138, x0=160, x1=900),
    ])
    assert len(blocks) == 2
    assert all(b["type"] == "para" for b in blocks)


# 2b. Big vertical gap also breaks
def test_gap_starts_new_paragraph():
    blocks = _blocks_of([
        _line("Ligne un dans le premier bloc de texte.", 100),
        _line("Ligne deux collée juste dessous sans trou.", 135),
        _line("Bloc séparé par un grand espace vertical.", 260),
    ])
    assert len(blocks) == 2


# 2c. Continuation cue suppresses a spurious break after a short line
def test_continuation_suppresses_short_line_break():
    blocks = _blocks_of([
        _line("Une phrase qui se termine court", 100, x1=400),
        _line("car elle continue en minuscule juste après le saut.", 135, x1=900),
    ])
    assert len(blocks) == 1


# 5/6. Heading detection + level clustering across pages
def test_heading_detection_and_levels():
    lines = [
        _line("CHAPITRE PREMIER", 100, x0=300, x1=700, h=60),   # big, centered, caps
        _line("Le titre de section", 240, x0=100, x1=500, h=40),
    ]
    # Body-dominated page so the median height tracks ordinary lines.
    lines += [_line(f"Corps de texte normal numéro {i} bien rempli ici.", 360 + i * 35, h=30)
              for i in range(8)]
    blocks = _blocks_of(lines)
    headings = [b for b in blocks if b["type"] == "heading"]
    assert any(b["text"] == "CHAPITRE PREMIER" for b in headings)
    h1 = next(b for b in headings if b["text"] == "CHAPITRE PREMIER")
    assert h1["level"] == 1
    # body line stays a paragraph
    assert any(b["type"] == "para" for b in blocks)


def test_numbered_section_header_is_heading():
    # "1. — UN THÈME" with a caps tail is a section head even when its OCR
    # bbox is no taller than body text.
    lines = [_line("1. - UN THÈME CRUCIAL PEU EXPLORÉ", 100, x0=260, x1=540, h=19)]
    lines += [_line(f"Corps de phrase numéro {i} bien rempli sur la ligne.", 160 + i * 35, h=22)
              for i in range(8)]
    blocks = _blocks_of(lines)
    head = [b for b in blocks if b["type"] == "heading"]
    assert any("UN THÈME CRUCIAL" in b["text"] for b in head)


def test_numbered_lowercase_stays_list():
    # A lowercase numbered item is an ordinary list, not a heading.
    blocks = _blocks_of([
        _line("1. - acheter du pain et du lait au marché", 100),
        _line("2. - préparer le repas du soir tranquillement", 140),
    ])
    assert all(b["type"] != "heading" for b in blocks)


def test_body_lines_not_headings():
    lines = [_line(f"Ligne de corps numéro {i} avec assez de texte pour être body.", 100 + i * 35)
             for i in range(8)]
    blocks = _blocks_of(lines)
    assert all(b["type"] != "heading" for b in blocks)


# 8. List items (bullets + numbered), em-dash dialogue excluded
def test_list_items():
    blocks = _blocks_of([
        _line("• Premier point de la liste", 100),
        _line("• Deuxième point", 140),
        _line("1. Élément numéroté", 200),
    ])
    lists = [b for b in blocks if b["type"] == "list"]
    assert len(lists) == 3
    assert lists[0]["text"] == "Premier point de la liste"


def test_em_dash_dialogue_not_list():
    blocks = _blocks_of([
        _line("— Bonjour, dit-il en entrant dans la pièce sombre.", 100),
        _line("— Bonsoir, répondit-elle sans lever les yeux.", 140),
    ])
    assert all(b["type"] != "list" for b in blocks)


# 9. Block quote: paragraph inset from both margins
def test_block_quote_inset():
    blocks = _blocks_of([
        _line("Texte courant aligné à la marge gauche normale du document.", 100, x0=100, x1=900),
        _line("Citation en retrait des deux côtés du texte.", 160, x0=250, x1=750),
        _line("qui se poursuit toujours en retrait des deux marges.", 195, x0=250, x1=750),
    ])
    assert any(b["type"] == "quote" for b in blocks)


# 7. Footnote: small line low on the page
def test_footnote_bottom_band():
    blocks = _blocks_of([
        _line("Corps principal de la page en haut avec du texte long.", 200, h=30),
        _line("1. Note de bas de page en petits caractères tout en bas.", 1300, h=18),
    ], page_h=1400)
    assert any(b["type"] == "footnote" for b in blocks)


# 10. Confidence filter drops noise
def test_confidence_filter():
    items = mx._line_items([
        _line("Texte fiable", 100, conf=0.95),
        _line("bruit", 140, conf=0.05),
    ])
    assert len(items) == 1
    assert items[0]["text"] == "Texte fiable"


# 4. Running header/footer removal across pages
def test_running_header_footer_removed():
    pages = []
    for i in range(6):
        pages.append(_page([
            _line("Histoire de l'Église", 30, h=20),          # running header
            _line(f"Contenu unique de la page {i} avec du texte de corps.", 200),
            _line(f"{i + 12}", 1360, x0=480, x1=520, h=18),    # page number
        ], scan_id=i + 1, scan_idx=i, page_h=1400))
    mx._render_line_pages(pages)
    # The repeated head is kept once (its first page) and dropped from the
    # rest; the page number is removed everywhere.
    appearances = sum(
        "Histoire de l'Église" in " ".join(_texts(p["blocks"])) for p in pages)
    assert appearances == 1
    for p in pages:
        joined = " ".join(_texts(p["blocks"]))
        assert "12" not in joined and "17" not in joined  # folio numbers gone


# 11. Two-column reading order
def test_two_column_reading_order():
    lines = []
    # left column
    for i in range(4):
        lines.append(_line(f"gauche {i}", 100 + i * 40, x0=80, x1=460))
    # right column
    for i in range(4):
        lines.append(_line(f"droite {i}", 100 + i * 40, x0=540, x1=920))
    items = mx._line_items(lines)
    ordered = mx._reading_order(items, 1000)
    texts = [it["text"] for it in ordered]
    assert texts.index("gauche 3") < texts.index("droite 0")


# 3. Cross-page paragraph continuation (merge)
def test_cross_page_paragraph_merge():
    p1 = _page([
        _line("Une phrase commencée sur la première page qui ne se termine pas", 100, x1=900),
    ], scan_id=1, scan_idx=0)
    p2 = _page([
        _line("mais se poursuit naturellement au début de la page suivante.", 100, x1=900),
    ], scan_id=2, scan_idx=1)
    mx._render_line_pages([p1, p2])
    merged = mx._merge_paragraphs(p1["blocks"] + p2["blocks"])
    assert len(merged) == 1
    assert "se poursuit naturellement" in merged[0]["text"]
    # inline page marker spliced in, invisible HTML comment
    rendered = mx._render_block(merged[0])[0]
    assert "<!-- scan #2 -->" in rendered


def test_cross_page_no_merge_when_terminated():
    p1 = _page([_line("Phrase terminée proprement.", 100)], scan_id=1)
    p2 = _page([_line("Nouvelle phrase indépendante ensuite.", 100)], scan_id=2)
    mx._render_line_pages([p1, p2])
    merged = mx._merge_paragraphs(p1["blocks"] + p2["blocks"])
    assert len(merged) == 2


# ── Clustering / utility units ─────────────────────────────────────

def test_cluster_1d_splits_on_gaps():
    clusters = mx._cluster_1d([1.0, 1.05, 1.1, 2.0, 2.1, 3.5], 3)
    assert len(clusters) == 3
    assert clusters[0] == [1.0, 1.05, 1.1]
    assert clusters[-1] == [3.5]


def test_heading_levels_monotonic():
    level_for = mx._build_heading_levels([2.4, 2.5, 1.6, 1.55, 1.2])
    # bigger ratio → smaller (more important) level
    assert level_for(2.5) <= level_for(1.6) <= level_for(1.2)
    assert level_for(2.5) == 1


def test_is_page_number():
    assert mx._is_page_number("12")
    assert mx._is_page_number("XIV")
    assert mx._is_page_number("— 7 —")
    assert not mx._is_page_number("Chapitre 7 sur la grâce")


# ── Apple-Document structured tree → Markdown ──────────────────────

def test_render_document_paragraphs_and_list():
    blocks = [
        {"type": "block", "text": "A flowing paragraph of prose."},
        {"type": "list", "items": [
            {"text": "first item"}, {"text": "second item"},
        ]},
        {"type": "block", "text": "Another paragraph after the list."},
    ]
    out = mx._render_document(blocks)
    text = "\n".join(out)
    assert "A flowing paragraph of prose." in text
    assert "- first item" in text
    assert "- second item" in text
    assert "Another paragraph after the list." in text


def test_render_document_table_verbatim():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    out = mx._render_document([{"type": "table", "markdown": md}])
    assert md in "\n".join(out)


def test_render_document_heading_heuristic():
    out = mx._render_document([{"type": "block", "text": "INTRODUCTION"}])
    assert out[0] == "## INTRODUCTION"


def test_render_document_prose_not_heading():
    # A long sentence ending in a period is prose, never a heading.
    txt = "This is a long sentence of prose that ends with a period."
    out = mx._render_document([{"type": "block", "text": txt}])
    assert out[0] == txt


def test_classify_row_prefers_document_tree():
    import json
    data = {
        "engine": "apple_docs",
        "page_w": 1000, "page_h": 1400,
        "lines": [{"text": "ignored geometric line", "bbox": [0, 0, 1, 1],
                   "confidence": 0.99}],
        "meta": {"document": [
            {"type": "block", "text": "Structured paragraph wins."},
        ]},
    }
    cls = mx._classify_row(json.dumps(data))
    assert cls["kind"] == "assembled"
    assert "Structured paragraph wins." in cls["text"]


def test_classify_row_falls_back_to_lines_without_document():
    import json
    data = {
        "engine": "apple_docs",
        "page_w": 1000, "page_h": 1400,
        "lines": [{"text": "geometric line", "bbox": [100, 100, 900, 140],
                   "confidence": 0.99}],
        "meta": {},
    }
    cls = mx._classify_row(json.dumps(data))
    assert cls["kind"] == "lines"
