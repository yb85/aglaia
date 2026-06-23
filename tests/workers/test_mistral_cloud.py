# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Cloud OCR (Mistral) engine — offline tests.

The Mistral round-trip (``_ocr_pdf``) is monkeypatched, so no network /
API key is touched. Covers: bitonal detection, G4 vs native PDF assembly,
whole-document batch splicing, page-count mismatch handling, and the
no-key / no-SDK guards.
"""

import numpy as np
import pikepdf
import pytest

from aglaia.workers.ocr import get_engine, ENGINE_REGISTRY
from aglaia.workers.ocr import mistral_cloud as mc


def _bw(h=120, w=90):
    a = np.zeros((h, w, 3), np.uint8)
    a[10:30, 5:80] = 255          # pure 0/255 → bitonal
    return a


def _color(h=120, w=90):
    a = np.full((h, w, 3), 128, np.uint8)
    a[..., 0] = 200               # 3 distinct values → not bitonal
    return a


def test_registered_and_whole_doc():
    assert "mistral_cloud" in ENGINE_REGISTRY
    e = get_engine("mistral_cloud")
    assert e.whole_doc is True
    assert e.name == "mistral_cloud"


def test_is_bitonal():
    assert mc._is_bitonal(_bw())
    assert not mc._is_bitonal(_color())


def test_all_bitonal_pdf_is_g4(tmp_path):
    out = tmp_path / "d.pdf"
    assert mc._images_to_pdf([_bw(), _bw()], [300, 300], out)
    pdf = pikepdf.open(str(out))
    assert len(pdf.pages) == 2
    assert pikepdf.Name.CCITTFaxDecode == pdf.pages[0].Resources.XObject.Im0.Filter


def test_mixed_pages_native_jpeg(tmp_path):
    out = tmp_path / "d.pdf"
    assert mc._images_to_pdf([_bw(), _color()], [300, 300], out)
    pdf = pikepdf.open(str(out))
    assert len(pdf.pages) == 2
    # native path embeds DCTDecode JPEGs (even the bitonal page)
    assert pikepdf.Name.DCTDecode == pdf.pages[1].Resources.XObject.Im0.Filter


def test_recognize_batch_splices_markdown(monkeypatch):
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(e, "_ocr_pdf",
                        lambda key, pdf: ["page zero md", "Θεός page one"])
    res = e.recognize_batch([_bw(), _bw()], ["el", "fr"], src_dpis=[300, 300])
    assert len(res) == 2
    assert res[0]["meta"]["markdown"] == "page zero md"
    assert res[1]["meta"]["markdown"] == "Θεός page one"
    assert res[0]["engine"] == "mistral_cloud"
    assert res[0]["meta"]["source"] == "mistral"
    # one full-page line so per-branch line counts / search still work
    assert res[1]["lines"][0]["text"] == "Θεός page one"


def test_page_count_mismatch_pads(monkeypatch):
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setattr(e, "_ocr_pdf", lambda key, pdf: ["only one"])
    res = e.recognize_batch([_bw(), _bw(), _bw()], ["fr"])
    assert len(res) == 3
    assert res[0]["meta"]["markdown"] == "only one"
    assert res[1]["meta"]["markdown"] == ""       # padded
    assert res[1]["lines"] == []


def test_positional_mapping(monkeypatch):
    """Mistral page 0,1,2 must map to the i-th SELECTED scan, by position."""
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setattr(e, "_build_capped_pdf", lambda i, d, p: (b"x", 3))
    monkeypatch.setattr(e, "_ocr_pdf",
                        lambda key, pdf: ["scan12", "scan45", "scan67"])
    res = e.recognize_batch([_bw(), _bw(), _bw()], ["fr"])
    assert [r["meta"]["markdown"] for r in res] == ["scan12", "scan45", "scan67"]


def test_truncation_flags_overflow(monkeypatch):
    """Pages past the upload cap are flagged truncated (not markdown)."""
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    # Pretend only 2 of 4 pages fit the upload.
    monkeypatch.setattr(e, "_build_capped_pdf", lambda i, d, p: (b"PDF", 2))
    monkeypatch.setattr(e, "_ocr_pdf", lambda key, pdf: ["A", "B"])
    res = e.recognize_batch([_bw()] * 4, ["fr"])
    assert len(res) == 4
    assert res[0]["meta"]["markdown"] == "A"
    assert res[1]["meta"]["markdown"] == "B"
    for i in (2, 3):
        assert res[i]["meta"]["truncated"] is True
        assert res[i]["lines"] == []
        assert "markdown" not in res[i]["meta"]


def test_build_capped_pdf_page_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "MAX_PAGES", 2)
    e = get_engine("mistral_cloud")
    _b, n = e._build_capped_pdf([_bw()] * 5, [300] * 5, tmp_path / "x.pdf")
    assert n == 2
    assert len(pikepdf.open(str(tmp_path / "x.pdf")).pages) == 2


def test_build_capped_pdf_size_cap(monkeypatch, tmp_path):
    # Absurdly small byte cap → must still send at least the first page.
    monkeypatch.setattr(mc, "MAX_UPLOAD_BYTES", 50)
    e = get_engine("mistral_cloud")
    _b, n = e._build_capped_pdf([_bw()] * 4, [300] * 4, tmp_path / "x.pdf")
    assert n == 1


def _png_bw_row(h=120, w=90, dpi=300):
    """A lightweight blob row like OcrWorker hands recognize_rows()."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(_bw(h, w)).convert("1").save(buf, format="PNG")
    return {"blob": buf.getvalue(), "dpi": dpi, "type": "BW",
            "format": "PNG", "width": w, "height": h}


def test_recognize_rows_builds_pdf_from_blobs(monkeypatch):
    """The low-memory path assembles the PDF from blobs (not RGB) and maps
    page i → row i."""
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    seen = {}

    def fake_ocr(key, pdf):
        import io
        seen["pages"] = len(pikepdf.open(io.BytesIO(pdf)).pages)
        return ["A", "B", "C"]

    monkeypatch.setattr(e, "_ocr_pdf", fake_ocr)
    res = e.recognize_rows([_png_bw_row() for _ in range(3)], ["fr"])
    assert [r["meta"]["markdown"] for r in res] == ["A", "B", "C"]
    assert seen["pages"] == 3            # PDF really built from the 3 blobs
    assert res[0]["page_w"] == 90 and res[0]["page_h"] == 120


def test_build_capped_pdf_from_rows_is_g4(tmp_path):
    e = get_engine("mistral_cloud")
    _b, n = e._build_capped_pdf_from_rows(
        [_png_bw_row(), _png_bw_row()], tmp_path / "x.pdf")
    assert n == 2
    pdf = pikepdf.open(str(tmp_path / "x.pdf"))
    assert pikepdf.Name.CCITTFaxDecode == pdf.pages[0].Resources.XObject.Im0.Filter


def test_recognize_rows_truncation(monkeypatch):
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setattr(e, "_build_capped_pdf_from_rows",
                        lambda rows, p: (b"PDF", 2))
    monkeypatch.setattr(e, "_ocr_pdf", lambda key, pdf: ["A", "B"])
    res = e.recognize_rows([_png_bw_row() for _ in range(4)], ["fr"])
    assert res[0]["meta"]["markdown"] == "A"
    assert res[2]["meta"]["truncated"] is True and res[2]["lines"] == []


def test_no_key_raises(monkeypatch):
    e = get_engine("mistral_cloud")
    e.available = True
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(mc, "get_engine", lambda *_: e, raising=False)
    # force secrets to resolve empty
    import aglaia.app_data.secrets as sec
    monkeypatch.setattr(sec, "get_mistral_api_key", lambda: "")
    with pytest.raises(RuntimeError, match="No Mistral API key"):
        e.recognize_batch([_bw()], ["fr"])


def test_unavailable_raises():
    e = get_engine("mistral_cloud")
    e.available = False
    with pytest.raises(RuntimeError, match="Cloud OCR unavailable"):
        e.recognize_batch([_bw()], ["fr"])
