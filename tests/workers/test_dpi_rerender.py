# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Fixing a PDF's input DPI has to re-render it, not relabel it (#173).

A PDF is imported by RENDERING each page at a chosen density, so the density
is in the pixels. A page rendered at 72 dpi holds 72 dpi worth of detail, and
calling it 350 only changes which way `DPIfixer` resamples — the reported
symptom was an export several times the size of the source, from a 350 dpi
PDF registered at 72.

A capture or an imported image is the opposite case: those pixels ARE the
original, and relabelling is the whole fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from aglaia.storage.db import open_db
from aglaia.storage.repo import ImageRepo, NodeRepo, ScanRepo
from aglaia.workers.ImportHelpers import rerender_pdf_sources

pdfium = pytest.importorskip("pypdfium2")


@pytest.fixture
def pdf(tmp_path):
    """A one-page PDF, written with the library the app reads it with."""
    pikepdf = pytest.importorskip("pikepdf")
    p = tmp_path / "book.pdf"
    doc = pikepdf.Pdf.new()
    doc.add_blank_page(page_size=(612, 792))      # US Letter at 72 pt/in
    doc.save(p)
    doc.close()
    return p


def _project(tmp_path, pdf_path, *, page: int, dpi: float, source: str):
    """A project holding one scan rendered from `pdf_path` at `dpi`."""
    from aglaia.storage.persister import Persister
    from aglaia.workers.pdf_extract import render_one

    db = tmp_path / "p.agl"
    conn = open_db(db)
    conn.execute(
        "INSERT INTO pipeline_versions (id, yaml_text, yaml_sha256, name, "
        "step_count, created_at, is_active) VALUES (1,'','h','p',1,'t',1)")
    arr = (render_one(pdf_path, page - 1, dpi) if source == "pdf"
           else np.full((400, 300, 3), 200, np.uint8))
    persister = Persister(conn)
    scan_id = ScanRepo(conn).create(
        source, 1, source_ref=f"{pdf_path}#{page}", capture_dpi=dpi)
    image_id = persister.persist_image(arr, "COLOR", dpi=dpi)
    root = persister.persist_node(
        scan_id=scan_id, parent_id=None, pipeline_version_id=1, step_idx=0,
        step_name=None, processor_name=None, branch_label=None, depth=0,
        filestem="p_001", image_id=image_id)
    ScanRepo(conn).set_root(scan_id, root)
    conn.commit()
    conn.close()
    return db, scan_id


def _root_image(db, scan_id):
    conn = open_db(db)
    try:
        row = ScanRepo(conn).get(scan_id)
        node = NodeRepo(conn).get(row["root_node_id"])
        return dict(ImageRepo(conn).get(node["image_id"]))
    finally:
        conn.close()


def test_a_pdf_page_gains_the_pixels_the_new_dpi_asks_for(tmp_path, pdf):
    """The regression: at 72 dpi a Letter page is 612×792 and no amount of
    relabelling makes it 300 dpi worth of detail."""
    db, scan_id = _project(tmp_path, pdf, page=1, dpi=72.0, source="pdf")
    before = _root_image(db, scan_id)
    assert (before["width"], before["height"]) == (612, 792)

    report = rerender_pdf_sources(db_path=str(db), dpi_by_scan={scan_id: 288.0})

    assert report["rendered"] == [scan_id]
    after = _root_image(db, scan_id)
    assert after["dpi"] == 288.0
    # 288/72 = 4× linear, within a pixel of rounding.
    assert abs(after["width"] - 612 * 4) <= 2
    assert abs(after["height"] - 792 * 4) <= 2


def test_an_imported_image_is_left_alone(tmp_path, pdf):
    """Its pixels are the original — re-rendering would be the bug."""
    db, scan_id = _project(tmp_path, pdf, page=1, dpi=72.0, source="import")
    before = _root_image(db, scan_id)
    report = rerender_pdf_sources(db_path=str(db), dpi_by_scan={scan_id: 300.0})
    assert report == {"rendered": [], "missing": [], "failed": []}
    assert _root_image(db, scan_id)["sha256"] == before["sha256"]


def test_a_source_that_has_moved_is_reported(tmp_path, pdf):
    db, scan_id = _project(tmp_path, pdf, page=1, dpi=72.0, source="pdf")
    pdf.unlink()
    report = rerender_pdf_sources(db_path=str(db), dpi_by_scan={scan_id: 300.0})
    assert report["rendered"] == []
    assert report["missing"] and report["missing"][0][0] == scan_id


def test_the_replaced_render_is_not_left_behind(tmp_path, pdf):
    db, scan_id = _project(tmp_path, pdf, page=1, dpi=72.0, source="pdf")
    old_id = _root_image(db, scan_id)["id"]
    rerender_pdf_sources(db_path=str(db), dpi_by_scan={scan_id: 200.0})
    conn = open_db(db)
    try:
        assert conn.execute("SELECT 1 FROM images WHERE id = ?",
                            (old_id,)).fetchone() is None
    finally:
        conn.close()


def test_nothing_to_do_is_cheap(tmp_path):
    assert rerender_pdf_sources(db_path="/nonexistent.agl", dpi_by_scan={}) == \
        {"rendered": [], "missing": [], "failed": []}


def test_an_unreadable_page_reference_is_reported(tmp_path, pdf):
    db, scan_id = _project(tmp_path, pdf, page=1, dpi=72.0, source="pdf")
    conn = open_db(db)
    conn.execute("UPDATE scans SET source_ref = ? WHERE id = ?",
                 (str(pdf), scan_id))        # no "#page"
    conn.commit()
    conn.close()
    report = rerender_pdf_sources(db_path=str(db), dpi_by_scan={scan_id: 300.0})
    assert report["failed"] and report["failed"][0][0] == scan_id
