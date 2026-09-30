# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A batch OCR that lands after its branch was reprocessed (#159).

Real case, `delbrel - eucharistie vecue.agl`: a Mistral batch was submitted,
then branch A of scans 149 and 153 was replayed, which deleted the nodes the
runs pointed at. "Check result" then imported the pages with a 0 × 0 frame
(`_dims_for_run` found no node), `finish()` cleared the stale flag the replay
had set, the page list called both pages OCR'd — and the textpack export
refused them: "292 of 294 pages … (missing pages: 271, 279)".
"""
import json

import pytest

from aglaia.storage.repo import BranchRepo, NodeRepo, OcrRepo
from aglaia.workers.ocr import mistral_batch
from aglaia.workers.PDFprocessor import OcrLayerError, create_pdf_from_db

from .test_pdf_ocr_layer import (PAGE_1, SENT_H, SENT_W, _mistral_result,
                                 _page_texts, _project)


def _replay(db, scan_id, old_node):
    """What `reprocess_branch` leaves behind: a new chosen node, the old one gone."""
    new = NodeRepo(db).insert(
        scan_id=scan_id, parent_id=None, pipeline_version_id=1, step_idx=0,
        step_name=None, processor_name=None, branch_label=None, depth=0,
        filestem="replayed", image_id=1)
    BranchRepo(db).upsert(scan_id, "", new)
    # The real project kept the runs of the deleted nodes (271, 279): the node
    # was dropped by a connection without `foreign_keys`, so the CASCADE on
    # `ocr_runs.node_id` never fired. Reproduce THAT state, orphan run included.
    db.execute("PRAGMA foreign_keys = OFF")
    db.execute("DELETE FROM nodes WHERE id = ?", (old_node,))
    db.execute("PRAGMA foreign_keys = ON")
    return new


def test_a_run_whose_node_was_replaced_stays_stale(tmp_path):
    db = _project(tmp_path, [("BW", None)])
    run = OcrRepo(db).start(scan_id=1, node_id=1, branch_path="",
                            engine="mistral_cloud", languages=["fr"])
    _replay(db, 1, 1)
    OcrRepo(db).finish(run, _mistral_result(PAGE_1))
    stale = db.execute("SELECT is_stale FROM ocr_runs WHERE id = ?", (run,)).fetchone()[0]
    assert stale == 1
    assert OcrRepo(db).branch_status_map()[(1, "")]["state"] == "stale"


def test_a_run_on_the_chosen_node_is_fresh(tmp_path):
    db = _project(tmp_path, [("BW", PAGE_1)])
    assert OcrRepo(db).branch_status_map()[(1, "")]["state"] == "fresh"


def test_a_batch_page_without_node_dims_takes_mistral_dimensions():
    page = _mistral_result(PAGE_1)["meta"]["mistral_page"]
    r = mistral_batch.page_to_result(page, 0, 0, ["fr"])
    assert (r["page_w"], r["page_h"]) == (SENT_W, SENT_H)
    assert r["lines"][0]["bbox"] == (0, 0, SENT_W, SENT_H)


def test_a_run_stored_with_a_zero_frame_still_lands_on_the_page(tmp_path):
    """Projects already carrying such runs export without re-OCR."""
    db = _project(tmp_path, [("BW", PAGE_1)])
    row = db.execute("SELECT id, result_json FROM ocr_runs").fetchone()
    broken = json.loads(row["result_json"])
    broken["page_w"] = broken["page_h"] = 0
    broken["lines"][0]["bbox"] = [0, 0, 0, 0]
    db.execute("UPDATE ocr_runs SET result_json = ? WHERE id = ?",
               (json.dumps(broken), row["id"]))
    db.commit()
    out = tmp_path / "o.pdf"
    assert create_pdf_from_db(db, out, compression="g4", add_ocr_layer=True,
                              engine="mistral_cloud")
    assert "Le Christ c'est l'Église." in " ".join(_page_texts(out)[0].split())


def test_a_missing_page_is_named_by_scan_and_branch(tmp_path):
    """The list shows scans, the error showed PDF pages: a retry went to the
    wrong page. Name both."""
    db = _project(tmp_path, [("BW", PAGE_1)])
    row = db.execute("SELECT id, result_json FROM ocr_runs").fetchone()
    broken = json.loads(row["result_json"])
    broken["page_w"] = broken["page_h"] = 0
    broken["meta"].pop("mistral_page")
    db.execute("UPDATE ocr_runs SET result_json = ? WHERE id = ?",
               (json.dumps(broken), row["id"]))
    db.commit()
    with pytest.raises(OcrLayerError) as ei:
        create_pdf_from_db(db, tmp_path / "o.pdf", compression="g4",
                           add_ocr_layer=True, engine="mistral_cloud")
    assert "1 (scan 1)" in str(ei.value)


def test_a_stale_batch_does_not_delete_the_current_layer(tmp_path):
    """Batch submitted, branch replayed and re-OCR'd, THEN the batch lands."""
    db = _project(tmp_path, [("BW", None)])
    late = OcrRepo(db).start(scan_id=1, node_id=1, branch_path="",
                             engine="mistral_cloud", languages=["fr"])
    new = _replay(db, 1, 1)
    good = OcrRepo(db).start(scan_id=1, node_id=new, branch_path="",
                             engine="mistral_cloud", languages=["fr"])
    OcrRepo(db).finish(good, _mistral_result(PAGE_1))
    OcrRepo(db).finish(late, _mistral_result(["ancien recadrage"]))
    kept = {r[0] for r in db.execute("SELECT id FROM ocr_runs WHERE status = 'done'")}
    assert good in kept
