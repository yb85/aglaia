# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""apple_docs export filename folds in the complement engine. apple_docs alone,
+surya and +glm produce DIFFERENT text, so they must produce different
filenames — else they silently overwrite each other (the reported bug)."""

from __future__ import annotations

import json
import sqlite3

from aglaia.workers.md_export import ocr_engine_suffix


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE ocr_runs (
        id INTEGER PRIMARY KEY, scan_id INT, branch_path TEXT DEFAULT '',
        engine TEXT, version INT, status TEXT, result_json TEXT)""")
    return c


def _run(c, scan, branch, engine, ver, complement=None, dpi=None):
    meta = {}
    if complement is not None:
        meta["complement_used"] = complement
    if dpi is not None:
        meta["ocr_dpi"] = dpi
    c.execute("INSERT INTO ocr_runs(scan_id,branch_path,engine,version,status,result_json)"
              " VALUES(?,?,?,?, 'done', ?)",
              (scan, branch, engine, ver, json.dumps({"meta": meta, "lines": []})))


def test_apple_docs_with_surya_complement():
    c = _conn()
    for s in range(4):
        _run(c, s, "A", "apple_docs", 1, complement="surya", dpi=200)
    assert ocr_engine_suffix(c, "apple_docs") == "_apple_docs_suryaOCR_200dpi"


def test_apple_docs_with_glm_complement_distinct():
    c = _conn()
    for s in range(4):
        _run(c, s, "A", "apple_docs", 1, complement="glm", dpi=200)
    # Must NOT collide with the surya filename.
    assert ocr_engine_suffix(c, "apple_docs") == "_apple_docs_glmOCR_200dpi"


def test_apple_docs_no_complement():
    c = _conn()
    for s in range(4):
        _run(c, s, "A", "apple_docs", 1, complement="none", dpi=200)
    assert ocr_engine_suffix(c, "apple_docs") == "_apple_docsOCR_200dpi"


def test_plain_engine_unaffected():
    c = _conn()
    _run(c, 0, "A", "surya", 1, dpi=150)
    assert ocr_engine_suffix(c, "surya") == "_suryaOCR_150dpi"
    assert ocr_engine_suffix(c, "apple_vision") == "_appleOCR"   # no runs → no dpi


def test_dominant_path_picks_complement():
    c = _conn()
    for s in range(5):
        _run(c, s, "A", "apple_docs", 1, complement="surya", dpi=200)
    # No explicit engine → dominant-engine path; still folds the complement in.
    assert ocr_engine_suffix(c) == "_apple_docs_suryaOCR_200dpi"
