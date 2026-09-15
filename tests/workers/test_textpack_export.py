# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The OCR textpack matches corpus's contract (#148).

Built from a small stubbed project — no paid API. Checks the names, one
`<!-- page N -->` per PDF page, `assets/source.pdf` present and searchable
page by page, the raw batch output byte for byte, and the `info.json` keys
corpus's reference writer (`corpus/cloud_ocr.py`) produces."""
import io
import json
import zipfile

import pypdfium2 as pdfium
import pytest
from PIL import Image

from aglaia.storage.db import open_db
from aglaia.storage.repo import (BranchRepo, ImageRepo, MistralBatchRepo,
                                 NodeRepo, OcrRepo, PipelineRepo, ScanRepo)
from aglaia.workers.PDFprocessor import OcrLayerError
from aglaia.workers.textpack_export import (TextpackError, default_name,
                                            write_textpack)

W, H = 1456, 2333
TEXTS = ["Première page du livre, en français.",
         "Deuxième page, qui continue le propos.",
         "Troisième page, la dernière du scan."]
RAW = (b'{"custom_id":"0","response":{"status_code":200,"body":{"pages":[],'
       b'"usage_info":{"pages_processed":3}}}}\n')

PROVENANCE_KEYS = {"engine", "model", "mode", "jobs", "requests", "pages",
                   "page_numbers", "pages_document", "tokens", "billed_pages",
                   "cost_usd", "cost_basis", "submitted_at", "completed_at",
                   "source_sha256", "tool", "raw_assets"}


def _result(text):
    block = {"top_left_x": 50, "top_left_y": 60, "bottom_right_x": 590,
             "bottom_right_y": 140, "content": text}
    md = f"# Chapitre\n\n{text}"
    return {"engine": "mistral_cloud", "languages": ["fr"],
            "page_w": W, "page_h": H,
            "lines": [{"text": md, "bbox": [0, 0, W, H], "confidence": 1.0}],
            "meta": {"model": "mistral-ocr-latest", "batch": True,
                     "markdown": md,
                     "mistral_page": {"index": 0, "markdown": md,
                                      "blocks": [block],
                                      "dimensions": {"dpi": 131, "width": 636,
                                                     "height": 1019}}}}


def _project(tmp_path, *, store_raw=True, source_pdf=True):
    db = open_db(str(tmp_path / "p.agl"))
    pv = PipelineRepo(db).upsert("name: t\npipeline: []\n", "t", 0)
    scan_pdf = tmp_path / "Delbrel - La femme.pdf"
    scan_pdf.write_bytes(b"%PDF-1.4 original scan")
    buf = io.BytesIO()
    Image.new("1", (W, H), 1).save(buf, "PNG")
    runs = []
    for i, text in enumerate(TEXTS):
        scan = ScanRepo(db).create(
            "pdf" if source_pdf else "capture", pv,
            source_ref=f"{scan_pdf}#{i + 1}" if source_pdf else "webcam#0")
        img = ImageRepo(db).insert(buf.getvalue(), "PNG", "BW", W, H, 300.0)
        node = NodeRepo(db).insert(
            scan_id=scan, parent_id=None, pipeline_version_id=pv, step_idx=0,
            step_name=None, processor_name=None, branch_label=None, depth=0,
            filestem=f"p{i}", image_id=img)
        BranchRepo(db).upsert(scan, "", node)
        run = OcrRepo(db).start(scan_id=scan, node_id=node, branch_path="",
                                engine="mistral_cloud", languages=["fr"])
        OcrRepo(db).finish(run, _result(text))
        runs.append(run)
    repo = MistralBatchRepo(db)
    repo.add("job-abc", page_count=3, status="SUCCESS", run_ids=runs,
             submitted_at="2026-09-15T08:00:00+00:00")
    repo.mark_imported("job-abc")
    if store_raw:
        repo.store_output("job-abc", RAW, output_file_id="file-1",
                          completed_at="2026-09-15T08:05:00+00:00")
    db.commit()
    return db


def _open(path):
    z = zipfile.ZipFile(path)
    return z, {n: z.read(n) for n in z.namelist()}


def test_the_archive_has_corpus_structure(tmp_path):
    db = _project(tmp_path)
    assert default_name(db) == "Delbrel - La femme_OCR.textpack"
    out = write_textpack(db, tmp_path / default_name(db), compression="g4")
    z, files = _open(out)
    root = "Delbrel - La femme_OCR.textbundle"
    assert set(files) == {f"{root}/info.json", f"{root}/text.md",
                          f"{root}/assets/source.pdf",
                          f"{root}/assets/mistral-job-abc.jsonl"}
    assert z.getinfo(f"{root}/assets/source.pdf").compress_type == zipfile.ZIP_STORED
    assert files[f"{root}/assets/mistral-job-abc.jsonl"] == RAW
    assert not list(tmp_path.glob("*.partiel"))


def test_text_md_marks_every_pdf_page_and_matches_it(tmp_path):
    db = _project(tmp_path)
    out = write_textpack(db, tmp_path / "x.textpack", compression="g4")
    _, files = _open(out)
    root = "Delbrel - La femme_OCR.textbundle"
    md = files[f"{root}/text.md"].decode("utf-8")
    pdf = pdfium.PdfDocument(files[f"{root}/assets/source.pdf"])
    assert len(pdf) == 3
    assert [f"<!-- page {n} -->" in md for n in (1, 2, 3)] == [True] * 3
    assert md.count("<!-- page ") == 3
    chunks = md.split("<!-- page ")[1:]
    for n, (chunk, text) in enumerate(zip(chunks, TEXTS)):
        assert text in chunk, f"page {n + 1} of text.md"
        layer = " ".join(pdf[n].get_textpage().get_text_bounded().split())
        assert text in layer, f"page {n + 1} of source.pdf is not searchable"


def test_info_json_carries_textbundle_keys_and_provenance(tmp_path):
    db = _project(tmp_path)
    out = write_textpack(db, tmp_path / "x.textpack", compression="g4",
                         zlib_id=913898)
    _, files = _open(out)
    info = json.loads(files["Delbrel - La femme_OCR.textbundle/info.json"])
    assert info["version"] == 2
    assert info["type"] == "net.daringfireball.markdown"
    assert info["transient"] is False
    assert info["creatorIdentifier"] == "cc.bibli.corpus.ocr"
    assert info["cc.bibli.corpus.ocr"]["stem"] == "Delbrel - La femme"
    corpus = info["corpus"]
    assert corpus["scan_source"] == "Delbrel - La femme.pdf"
    assert corpus["zlib_id"] == 913898
    assert corpus["double_page"] is False
    ocr = corpus["ocr"]
    assert set(ocr) == PROVENANCE_KEYS
    assert ocr["engine"] == "mistral-ocr" and ocr["mode"] == "batch"
    assert ocr["jobs"] == ["job-abc"]
    assert ocr["pages"] == 3 and ocr["billed_pages"] == 3
    assert ocr["cost_usd"] == pytest.approx(0.006)
    assert ocr["raw_assets"] == ["mistral-job-abc.jsonl"]
    assert ocr["submitted_at"] == "2026-09-15T08:00:00+00:00"
    assert ocr["completed_at"] == "2026-09-15T08:05:00+00:00"
    assert len(ocr["source_sha256"]) == 64


def test_a_batch_job_without_its_raw_output_is_refused(tmp_path):
    db = _project(tmp_path, store_raw=False)
    with pytest.raises(TextpackError, match="check-ocr"):
        write_textpack(db, tmp_path / "x.textpack", compression="g4")
    assert not (tmp_path / "x.textpack").exists()


def test_an_unsearchable_pdf_is_never_packed(tmp_path):
    db = _project(tmp_path)
    for row in db.execute("SELECT id, result_json FROM ocr_runs").fetchall():
        broken = json.loads(row["result_json"])
        broken["page_w"] = broken["page_h"] = 0
        broken["meta"].pop("mistral_page")
        db.execute("UPDATE ocr_runs SET result_json = ? WHERE id = ?",
                   (json.dumps(broken), row["id"]))
    with pytest.raises(OcrLayerError):
        write_textpack(db, tmp_path / "x.textpack", compression="g4")
    assert not (tmp_path / "x.textpack").exists()


def test_a_captured_project_is_named_after_its_slug(tmp_path):
    db = _project(tmp_path, source_pdf=False)
    db.execute("INSERT INTO project (id, name, slug, created_at, updated_at) "
               "VALUES (1, 'Delbrêl', 'delbrel-oc9', 'now', 'now')")
    assert default_name(db) == "delbrel-oc9_OCR.textpack"


def test_the_cli_writes_it_and_hands_it_to_send_to(tmp_path):
    """`--export md+textpack` through the headless driver. The file list that
    `--send-to` reads used to be appended to without ever being created, so
    a Markdown export raised and no PDF was ever sent."""
    from aglaia.workers import headless
    from aglaia.workers.cli import _parse_export_arg as parse_exports

    db = _project(tmp_path)
    db.close()
    rc = headless._run_exports(str(tmp_path / "p.agl"), tmp_path, "p",
                               parse_exports("pdf:g4+md+textpack:g4:zlib=42"),
                               ocr_layer=True)
    assert rc == 0
    names = sorted(p.name for p in headless._run_exports.written)
    assert names == ["Delbrel - La femme_OCR.textpack", "p.md", "p.pdf"]
    _, files = _open(tmp_path / "Delbrel - La femme_OCR.textpack")
    info = json.loads(files["Delbrel - La femme_OCR.textbundle/info.json"])
    assert info["corpus"]["zlib_id"] == 42
