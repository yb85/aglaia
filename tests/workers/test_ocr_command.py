# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""`aglaia ocr` — chain-free ingest creates an OCR-able branch (= raw page),
then the shared OCR/export primitives run. OCR engine is stubbed (CI-portable)."""

from __future__ import annotations

import cv2
import numpy as np


def _write_image(path) -> None:
    cv2.imwrite(str(path), np.full((60, 90, 3), 255, np.uint8))


def test_command_registered():
    from aglaia.cli import KNOWN_COMMANDS

    assert "ocr" in KNOWN_COMMANDS


def test_ocr_only_ingests_raw_branch_and_runs_ocr(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "appdata"))
    img = tmp_path / "page.png"
    _write_image(img)

    from aglaia.cli.shared import ocr_config

    cfg = ocr_config([img], None, "auto", None, None, "ocrtest", tmp_path, None, False)
    assert cfg.source == "images"

    # Stub the real OCR so the test needs no engine; capture the DB path.
    import aglaia.workers.headless as H

    captured = {}

    def _stub_ocr(db_path, **kw):
        captured["db"] = db_path
        captured["engine"] = kw.get("engine_name")
        return 0

    monkeypatch.setattr(H, "_run_ocr", _stub_ocr)

    rc = H.run_ocr_only(cfg)
    assert rc == 0
    assert captured.get("engine") == "auto"  # bare `ocr` defaults engine to auto

    # The chain-free ingest must leave exactly one branch (branch_path "") whose
    # chosen node is the raw page — i.e. branches_needing_ocr sees it.
    from aglaia.storage.db import open_db
    from aglaia.storage.repo import OcrRepo

    conn = open_db(captured["db"])
    try:
        rows = OcrRepo(conn).branches_needing_ocr(include_stale=True)
        nodes = conn.execute("SELECT COUNT(*) AS n FROM nodes").fetchone()["n"]
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["branch_path"] == ""
    assert nodes == 1  # only the raw root node, no pipeline steps


def test_ocr_only_no_inputs_returns_2(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "appdata"))
    from aglaia.cli.shared import ocr_config
    import aglaia.workers.headless as H

    cfg = ocr_config([], None, "auto", None, None, None, tmp_path, None, False)
    assert H.run_ocr_only(cfg) == 2
