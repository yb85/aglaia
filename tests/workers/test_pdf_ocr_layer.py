# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The PDF OCR layer is searchable, or the export says it is not (#149).

Measured on a real 278-page Mistral project before this fix: 0 pages with more
than 100 extractable characters, 561 characters out of 378 697 — 0.1 % — and
`create_pdf_from_db` returned True.

Mistral stores a page as ONE line whose bbox is the whole page and whose text
is the page's whole Markdown. Drawn as a single text run, that is a font the
height of the page on a baseline at its bottom edge, one line wide, running
off the page to the right: an extractor clips everything outside the page box
and keeps a few glyphs. The per-block geometry was there all along, in
`meta.mistral_page.blocks`, in the pixel frame of the image SENT to Mistral.

The data below is shaped exactly like a stored `mistral_cloud` result.
"""
import io
import json

import pypdfium2 as pdfium
import pytest
from PIL import Image

from aglaia.storage.db import open_db
from aglaia.storage.repo import (BranchRepo, ImageRepo, NodeRepo, OcrRepo,
                                 PipelineRepo, ScanRepo)
from aglaia.workers.PDFprocessor import OcrLayerError, create_pdf_from_db

PAGE_W, PAGE_H = 1456, 2333          # the stored page frame
SENT_W, SENT_H = 636, 1019           # what Mistral saw (ocr_dpi downsample)


def _png(mode, w=PAGE_W, h=PAGE_H):
    im = Image.new("L", (w, h), 255)
    if mode == "1":
        im = im.convert("1")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _mistral_result(paragraphs):
    """A stored `mistral_cloud` result: one full-page line, the real blocks
    in the SENT frame under `meta.mistral_page`."""
    blocks, y = [], 40
    for para in paragraphs:
        blocks.append({"top_left_x": 50, "top_left_y": y,
                       "bottom_right_x": 590, "bottom_right_y": y + 90,
                       "content": para})
        y += 120
    markdown = "\n\n".join(paragraphs)
    return {
        "engine": "mistral_cloud", "languages": ["fr"],
        "page_w": PAGE_W, "page_h": PAGE_H,
        "lines": [{"text": markdown, "bbox": [0, 0, PAGE_W, PAGE_H],
                   "confidence": 1.0}],
        "meta": {"source": "mistral", "model": "mistral-ocr-latest",
                 "batch": True, "markdown": markdown,
                 "mistral_page": {"index": 0, "markdown": markdown,
                                  "blocks": blocks,
                                  "dimensions": {"dpi": 131, "width": SENT_W,
                                                 "height": SENT_H}}},
    }


def _project(tmp_path, pages):
    """`pages` = [(image_type, paragraphs_or_None), …] in page order."""
    db = open_db(str(tmp_path / "p.agl"))
    pv = PipelineRepo(db).upsert("name: t\npipeline: []\n", "t", 0)
    for idx, (itype, paras) in enumerate(pages):
        scan = ScanRepo(db).create("import", pv)
        blob = _png("1" if itype == "BW" else "L")
        img = ImageRepo(db).insert(blob, "PNG", itype, PAGE_W, PAGE_H, 300.0)
        node = NodeRepo(db).insert(
            scan_id=scan, parent_id=None, pipeline_version_id=pv, step_idx=0,
            step_name=None, processor_name=None, branch_label=None, depth=0,
            filestem=f"p{idx}", image_id=img)
        BranchRepo(db).upsert(scan, "", node)
        if paras is not None:
            run = OcrRepo(db).start(scan_id=scan, node_id=node, branch_path="",
                                    engine="mistral_cloud", languages=["fr"])
            OcrRepo(db).finish(run, _mistral_result(paras))
    db.commit()
    return db


def _page_texts(pdf_path):
    """Text an extractor sees INSIDE each page box.

    Bounded on purpose. The broken layer does put the text in the file — on
    the real project, 312 258 characters by `get_text_range` — but off the
    page, and `get_text_bounded` finds 547 of them, as PyMuPDF (561) and
    `pdftotext` (609) do: the consumers that matter clip to the page. An
    unbounded read passes on the very defect this file exists to catch."""
    doc = pdfium.PdfDocument(str(pdf_path))
    return [doc[i].get_textpage().get_text_bounded() for i in range(len(doc))]


PAGE_1 = ["La croissance de notre Foi dans ce sens demande un tel dépassement",
          "Le Christ c'est l'Église."]
PAGE_2 = ["Il n'est pas plus séparable de son corps que la tête",
          "Nous le rencontrons dans ses membres."]
PAGE_3 = ["Troisième page, pour vérifier l'alignement des pages."]


def test_every_page_carries_its_own_searchable_text(tmp_path):
    db = _project(tmp_path, [("BW", PAGE_1), ("BW", PAGE_2), ("BW", PAGE_3)])
    out = tmp_path / "o.pdf"
    assert create_pdf_from_db(db, out, compression="g4",
                              add_ocr_layer=True, engine="mistral_cloud")
    texts = _page_texts(out)
    assert len(texts) == 3
    for text, paras in zip(texts, (PAGE_1, PAGE_2, PAGE_3)):
        flat = " ".join(text.split())
        for para in paras:
            assert para in flat, f"missing {para!r} in {flat!r}"


def test_paragraphs_come_out_in_reading_order(tmp_path):
    """Block by block, top to bottom — not one run collapsing into fragments."""
    db = _project(tmp_path, [("BW", PAGE_1)])
    out = tmp_path / "o.pdf"
    create_pdf_from_db(db, out, compression="g4", add_ocr_layer=True,
                       engine="mistral_cloud")
    flat = " ".join(_page_texts(out)[0].split())
    assert flat.index(PAGE_1[0]) < flat.index(PAGE_1[1])


def test_a_skipped_row_does_not_shift_the_text_onto_the_next_page(tmp_path):
    """The bitonal builder skips a non-BW row. The layer used to be indexed by
    PDF page against a list indexed by row, so every page after the gap got
    the NEXT page's text — worse than no text."""
    db = _project(tmp_path, [("BW", PAGE_1), ("GRAY", PAGE_2), ("BW", PAGE_3)])
    out = tmp_path / "o.pdf"
    assert create_pdf_from_db(db, out, compression="g4",
                              add_ocr_layer=True, engine="mistral_cloud")
    texts = [" ".join(t.split()) for t in _page_texts(out)]
    assert len(texts) == 2
    assert PAGE_1[0] in texts[0]
    assert PAGE_3[0] in texts[1]
    assert PAGE_2[0] not in texts[1]


def test_a_blank_page_is_not_a_failure(tmp_path):
    """A page with no OCR run has nothing to embed — that is not an error."""
    db = _project(tmp_path, [("BW", PAGE_1), ("BW", None)])
    out = tmp_path / "o.pdf"
    assert create_pdf_from_db(db, out, compression="g4",
                              add_ocr_layer=True, engine="mistral_cloud")


def test_an_ocr_layer_that_cannot_be_written_is_an_error(tmp_path):
    """The silent-success path: OCR exists, nothing lands in the PDF, and the
    export still reported an OCR PDF. It must say so, and leave no file that
    could be mistaken for one."""
    db = _project(tmp_path, [("BW", PAGE_1)])
    # Break the stored geometry the way a lookup mismatch would.
    row = db.execute("SELECT id, result_json FROM ocr_runs").fetchone()
    broken = json.loads(row["result_json"])
    broken["page_w"] = broken["page_h"] = 0
    broken["meta"].pop("mistral_page")
    db.execute("UPDATE ocr_runs SET result_json = ? WHERE id = ?",
               (json.dumps(broken), row["id"]))
    db.commit()
    out = tmp_path / "o.pdf"
    with pytest.raises(OcrLayerError) as ei:
        create_pdf_from_db(db, out, compression="g4", add_ocr_layer=True,
                           engine="mistral_cloud")
    assert ei.value.expected == 1 and ei.value.written == 0
    assert not out.exists()


def test_a_block_of_run_together_paragraphs_still_lands_on_the_page(tmp_path):
    """Real page 9 of the Delbrêl project: ONE block covering most of the page
    whose content is 1 178 characters with no line break — Mistral ran the
    paragraphs together ("…tome I).1924 « Conversion…"). Split on newlines
    alone, that is one run whose font is 80 % of the block's height, which
    overflows the page however hard it is squeezed. The line count has to come
    from the block's geometry."""
    long_para = " ".join(["Rencontre de Jean Maydieu pour lequel elle a une "
                          "forte inclination, mais qui entrera chez les "
                          "dominicains en 1925."] * 12)
    db = _project(tmp_path, [("BW", None)])
    run = OcrRepo(db).start(scan_id=1, node_id=1, branch_path="",
                            engine="mistral_cloud", languages=["fr"])
    result = _mistral_result([long_para])
    result["meta"]["mistral_page"]["blocks"] = [{
        "top_left_x": 24, "top_left_y": 26, "bottom_right_x": 582,
        "bottom_right_y": 938, "content": long_para}]
    OcrRepo(db).finish(run, result)
    db.commit()
    out = tmp_path / "o.pdf"
    create_pdf_from_db(db, out, compression="g4", add_ocr_layer=True,
                       engine="mistral_cloud")
    got = len("".join(_page_texts(out)[0].split()))
    want = len("".join(long_para.split()))
    assert got >= 0.95 * want, f"only {got} of {want} characters on the page"
