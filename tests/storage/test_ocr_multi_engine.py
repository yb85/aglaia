# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Multi-engine OCR storage: one layer per (branch, engine), per-engine
`branches_needing_ocr`, `available_ocr_layers`, and per-engine export selection."""

from __future__ import annotations

import json

import numpy as np

from aglaia.storage.persister import Persister
from aglaia.storage.repo import BranchRepo, OcrRepo, ScanRepo


def _seed_one_branch(db, pid):
    """A scan with one branch ("") whose chosen node is the raw image."""
    p = Persister(db)
    scan_id = ScanRepo(db).create("import", pid)
    img_id = p.persist_image(np.zeros((8, 8, 3), np.uint8), "COLOR", dpi=100.0)
    node_id = p.persist_node(
        scan_id=scan_id,
        parent_id=None,
        pipeline_version_id=pid,
        step_idx=0,
        step_name=None,
        processor_name=None,
        branch_label=None,
        depth=0,
        filestem="t_000",
        image_id=img_id,
    )
    ScanRepo(db).set_root(scan_id, node_id)
    BranchRepo(db).upsert(scan_id, "", node_id)
    return scan_id, node_id


def _ocr(db, scan_id, node_id, engine, markdown):
    o = OcrRepo(db)
    rid = o.start(
        scan_id=scan_id, node_id=node_id, branch_path="", engine=engine, languages=[]
    )
    o.finish(
        rid,
        {
            "engine": engine,
            "page_w": 8,
            "page_h": 8,
            "lines": [{"text": markdown, "bbox": [0, 0, 8, 8], "confidence": 1.0}],
            "meta": {"markdown": markdown},
        },
    )
    return rid


def test_branches_needing_ocr_is_per_engine(seeded_db):
    db, pid = seeded_db
    scan_id, node_id = _seed_one_branch(db, pid)
    o = OcrRepo(db)
    _ocr(db, scan_id, node_id, "apple_vision", "A")

    # apple done → not needed for apple; surya never ran → still needed for surya.
    assert o.branches_needing_ocr(include_stale=True, engine="apple_vision") == []
    assert len(o.branches_needing_ocr(include_stale=True, engine="surya")) == 1
    # Cross-engine (engine=None) keeps the old behaviour: any done run excludes it.
    assert o.branches_needing_ocr(include_stale=True) == []


def test_engines_coexist_and_layers_latest_first(seeded_db):
    db, pid = seeded_db
    scan_id, node_id = _seed_one_branch(db, pid)
    _ocr(db, scan_id, node_id, "apple_vision", "A")
    _ocr(db, scan_id, node_id, "surya", "S")

    layers = OcrRepo(db).available_ocr_layers()
    assert [r["engine"] for r in layers] == ["surya", "apple_vision"]  # latest first
    assert all(r["n_branches"] == 1 for r in layers)
    # Both layers persist (not replaced across engines).
    n = db.execute("SELECT COUNT(*) AS n FROM ocr_runs WHERE status='done'").fetchone()[
        "n"
    ]
    assert n == 2


def test_rerun_same_engine_replaces_max_one_per_engine(seeded_db):
    db, pid = seeded_db
    scan_id, node_id = _seed_one_branch(db, pid)
    _ocr(db, scan_id, node_id, "surya", "v1")
    _ocr(db, scan_id, node_id, "surya", "v2")  # rerun same engine → replaces

    rows = db.execute(
        "SELECT COUNT(*) AS n FROM ocr_runs WHERE engine='surya' AND status='done'"
    ).fetchone()
    assert rows["n"] == 1  # max 1 per engine
    surviving = OcrRepo(db).latest_for_branch(scan_id, "")
    assert json.loads(surviving["result_json"])["meta"]["markdown"] == "v2"


def test_md_export_selects_engine_layer(seeded_db, tmp_path):
    from aglaia.workers.md_export import write_markdown

    db, pid = seeded_db
    scan_id, node_id = _seed_one_branch(db, pid)
    _ocr(db, scan_id, node_id, "apple_vision", "APPLE_TXT")
    _ocr(db, scan_id, node_id, "surya", "SURYA_TXT")  # latest

    a = tmp_path / "a.md"
    write_markdown(db, a, engine="apple_vision")
    assert "APPLE_TXT" in a.read_text() and "SURYA_TXT" not in a.read_text()

    s = tmp_path / "s.md"
    write_markdown(db, s, engine="surya")
    assert "SURYA_TXT" in s.read_text()

    d = tmp_path / "d.md"
    write_markdown(db, d)  # default = latest layer = surya
    assert "SURYA_TXT" in d.read_text()


def test_pdf_ocr_results_select_engine(seeded_db):
    from aglaia.workers.PDFprocessor import _ocr_results_for_rows

    db, pid = seeded_db
    scan_id, node_id = _seed_one_branch(db, pid)
    _ocr(db, scan_id, node_id, "apple_vision", "APPLE_TXT")
    _ocr(db, scan_id, node_id, "surya", "SURYA_TXT")

    rows = [{"scan_id": scan_id, "branch_path": ""}]
    assert (
        _ocr_results_for_rows(db, rows, "apple_vision")[0]["meta"]["markdown"]
        == "APPLE_TXT"
    )
    assert (
        _ocr_results_for_rows(db, rows, "surya")[0]["meta"]["markdown"] == "SURYA_TXT"
    )
    assert (
        _ocr_results_for_rows(db, rows)[0]["meta"]["markdown"] == "SURYA_TXT"
    )  # latest
