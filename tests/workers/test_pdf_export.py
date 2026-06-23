# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""pdf_export — bitonal PDF assembly regressions."""

import io

import pikepdf
from PIL import Image

from aglaia.workers import pdf_export


def _bw_row(w=200, h=260, dpi=300):
    buf = io.BytesIO()
    Image.new("1", (w, h), 1).save(buf, format="PNG")
    return {"blob": buf.getvalue(), "dpi": dpi, "type": "BW",
            "format": "PNG", "width": w, "height": h}


def test_jbig2_falls_back_to_g4_when_encoder_missing(tmp_path, monkeypatch):
    """A phantom `aglaia_jbig2` namespace package (repo crate dir on path,
    wheel not built) used to make the availability probe pass while the
    encoder was absent — build then raised mid-loop and wrote NO file.
    The probe now imports the encoder symbol, so it cleanly falls back to
    G4 and still produces a PDF."""
    # Force the encoder import to fail regardless of the local environment.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "aglaia_jbig2":
            raise ImportError("simulated: wheel not built")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    out = tmp_path / "doc.pdf"
    ok = pdf_export.build_bitonal_pdf([_bw_row(), _bw_row()], out,
                                      engine="jbig2")
    assert ok is True
    assert out.exists() and out.stat().st_size > 0
    pdf = pikepdf.open(str(out))
    assert len(pdf.pages) == 2
    # fell back to CCITT G4
    assert pikepdf.Name.CCITTFaxDecode == \
        pdf.pages[0].Resources.XObject.Im0.Filter


def test_g4_engine_writes_file(tmp_path):
    out = tmp_path / "g4.pdf"
    assert pdf_export.build_bitonal_pdf([_bw_row()], out, engine="g4")
    assert out.exists()
    pdf = pikepdf.open(str(out))
    assert pikepdf.Name.CCITTFaxDecode == \
        pdf.pages[0].Resources.XObject.Im0.Filter
