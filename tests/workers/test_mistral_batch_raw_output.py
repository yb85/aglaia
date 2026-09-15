# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The raw Mistral batch output is kept in the project, byte for byte (#147).

No network: `poll` and `fetch_output` are stubbed. The output bytes carry
odd spacing, key order and a CRLF on purpose — a parse/re-serialise round
trip would change them, and the test would catch it."""
import hashlib
import json
import io

import pytest
from PIL import Image

from aglaia.storage.db import open_db
from aglaia.storage.repo import (BranchRepo, ImageRepo, MistralBatchRepo,
                                 NodeRepo, OcrRepo, PipelineRepo, ScanRepo)
from aglaia.workers import headless
from aglaia.workers.ocr import mistral_batch

RAW = (b'{"custom_id":"0",  "response":{"status_code":200,"body":{"pages":'
       b'[{"index":0,"markdown":"# Titre\\n\\nPremi\\u00e8re page",'
       b'"blocks":[],"dimensions":{"dpi":131,"width":636,"height":1019}}],'
       b'"usage_info":{"pages_processed":1}}}}\r\n'
       b'{"response":{"body":{"pages":[{"markdown":"Deuxi\\u00e8me page",'
       b'"index":0}]}},"custom_id":"1"}\n')


def _project(tmp_path, n=2):
    db_path = tmp_path / "p.agl"
    db = open_db(str(db_path))
    pv = PipelineRepo(db).upsert("name: t\npipeline: []\n", "t", 0)
    buf = io.BytesIO()
    Image.new("L", (100, 160), 255).save(buf, "PNG")
    runs = []
    for idx in range(n):
        scan = ScanRepo(db).create("import", pv)
        img = ImageRepo(db).insert(buf.getvalue(), "PNG", "GRAY", 100, 160, 300.0)
        node = NodeRepo(db).insert(
            scan_id=scan, parent_id=None, pipeline_version_id=pv, step_idx=0,
            step_name=None, processor_name=None, branch_label=None, depth=0,
            filestem=f"p{idx}", image_id=img)
        BranchRepo(db).upsert(scan, "", node)
        runs.append(OcrRepo(db).start(scan_id=scan, node_id=node,
                                      branch_path="", engine="mistral_cloud",
                                      languages=["fr"]))
    MistralBatchRepo(db).add("job-1", page_count=n, status="QUEUED",
                             run_ids=runs)
    db.commit()
    db.close()
    return db_path


@pytest.fixture
def stub_mistral(monkeypatch):
    calls = {"fetch": 0}

    def fetch_output(_key, job_id):
        calls["fetch"] += 1
        return RAW, {"output_file_id": f"file-{job_id}",
                     "completed_at": "2026-09-15T10:00:00+00:00"}

    monkeypatch.setattr(mistral_batch, "poll", lambda _k, _j: ("SUCCESS", None))
    monkeypatch.setattr(mistral_batch, "fetch_output", fetch_output)
    monkeypatch.setattr("aglaia.app_data.secrets.get_mistral_api_key",
                        lambda: "test-key")
    return calls


def test_check_ocr_stores_the_output_byte_for_byte(tmp_path, stub_mistral):
    db_path = _project(tmp_path)
    assert headless._check_ocr(str(db_path)) == 0
    db = open_db(str(db_path))
    repo = MistralBatchRepo(db)
    assert repo.output("job-1") == RAW
    (meta,) = repo.outputs()
    assert meta["sha256"] == hashlib.sha256(RAW).hexdigest()
    assert meta["size"] == len(RAW)
    assert meta["output_file_id"] == "file-job-1"
    assert meta["completed_at"] == "2026-09-15T10:00:00+00:00"
    # And the pages were still imported from those same bytes.
    texts = [json.loads(r["result_json"])["meta"]["markdown"] for r in db.execute(
        "SELECT result_json FROM ocr_runs WHERE status = 'done' ORDER BY id")]
    assert len(texts) == 2 and "Première page" in texts[0]


def test_a_second_check_neither_duplicates_nor_loses_it(tmp_path, stub_mistral):
    db_path = _project(tmp_path)
    headless._check_ocr(str(db_path))
    headless._check_ocr(str(db_path))
    db = open_db(str(db_path))
    assert db.execute("SELECT count(*) FROM mistral_batch_outputs").fetchone()[0] == 1
    assert MistralBatchRepo(db).output("job-1") == RAW
    assert stub_mistral["fetch"] == 1


def test_a_job_imported_before_the_table_existed_is_backfilled(tmp_path, stub_mistral):
    """Projects OCR'd before #147 hold the job id and nothing else."""
    db_path = _project(tmp_path)
    db = open_db(str(db_path))
    MistralBatchRepo(db).set_status("job-1", "SUCCESS")
    MistralBatchRepo(db).mark_imported("job-1")
    db.commit()
    db.close()
    headless._check_ocr(str(db_path))
    db = open_db(str(db_path))
    assert MistralBatchRepo(db).output("job-1") == RAW
    assert MistralBatchRepo(db).missing_outputs() == []


def test_a_job_row_deleted_from_the_jobs_tab_keeps_its_output(tmp_path, stub_mistral):
    db_path = _project(tmp_path)
    headless._check_ocr(str(db_path))
    db = open_db(str(db_path))
    MistralBatchRepo(db).delete("job-1")
    db.commit()
    assert MistralBatchRepo(db).output("job-1") == RAW


def test_pages_from_output_matches_the_old_fetch_pages_parse():
    pages = mistral_batch.pages_from_output(RAW)
    assert [p["markdown"] for p in pages] == ["# Titre\n\nPremière page",
                                              "Deuxième page"]
