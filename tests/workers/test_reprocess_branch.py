# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Branch-level reprocess (per-page step toggle).

Toggling a per-page step on one page must rerun ONLY that page-branch — its
sibling pages (and other scans) must be left untouched — instead of the old
behaviour that reran the whole scan from raw. Uses the test_augustin fixture,
a processed 2-scan project where PageDetector split each scan into pages A & B.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from aglaia.workers.ImportHelpers import reprocess_branch

FIXTURE = (Path(__file__).resolve().parents[2]
           / "test_data" / "test_augustin"
           / "test-augustin-confessions-vii.agl")


class _MockChain:
    """Records enqueue_resume calls instead of running the pipeline."""
    def __init__(self):
        self.calls = []

    def enqueue_resume(self, **kw):
        self.calls.append(kw)


def _ids(conn, scan_id, label):
    return sorted(r["id"] for r in conn.execute(
        "SELECT id FROM nodes WHERE scan_id = ? AND branch_label = ?",
        (scan_id, label)))


def _ids_null(conn, scan_id):
    return sorted(r["id"] for r in conn.execute(
        "SELECT id FROM nodes WHERE scan_id = ? AND branch_label IS NULL",
        (scan_id,)))


@pytest.fixture()
def db_path(tmp_path):
    if not FIXTURE.exists():
        pytest.skip("test_augustin fixture not present")
    dst = tmp_path / "branch.agl"
    shutil.copy(FIXTURE, dst)
    return str(dst)


def test_reprocess_branch_isolates_sibling(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    a_before = _ids(conn, 1, "A")
    b_before = _ids(conn, 1, "B")
    null_before = _ids_null(conn, 1)
    scan2_b_before = _ids(conn, 2, "B")
    anchor = conn.execute(
        "SELECT id, step_idx FROM nodes WHERE scan_id = 1 AND branch_label = 'A' "
        "ORDER BY step_idx ASC LIMIT 1").fetchone()
    conn.close()
    assert len(a_before) > 1 and len(b_before) > 1  # really split

    chain = _MockChain()
    ret = reprocess_branch(db_path=db_path, pipeline_version_id=1, chain=chain,
                           scan_id=1, branch_label="A")

    assert ret == 1  # branch rerun, not the whole-scan fallback

    # Resume the pipeline from page A's anchor (its PageDetector child).
    assert len(chain.calls) == 1
    call = chain.calls[0]
    assert call["node_id"] == anchor["id"]
    assert call["start_idx"] == anchor["step_idx"]
    assert call["branch_path"] == "A"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Page A's downstream subtree is wiped down to the anchor; everything else
    # — sibling page B, the pre-split nodes, the other scan — is untouched.
    assert _ids(conn, 1, "A") == [anchor["id"]]
    assert _ids(conn, 1, "B") == b_before
    assert _ids_null(conn, 1) == null_before
    assert _ids(conn, 2, "B") == scan2_b_before
    conn.close()


def test_reprocess_branch_falls_back_when_unsplit(db_path):
    """An unknown / empty branch label can't be isolated → whole-scan path.
    With a mock chain (no real enqueue) the fallback reprocesses the scan via
    reprocess_active_scans, which calls chain.enqueue — assert we did NOT take
    the branch-resume path."""
    chain = _MockChain()  # has no .enqueue → fallback would raise, caught below
    # branch_label "" has no matching nodes (pre-split nodes are NULL), so this
    # must NOT do a branch resume.
    try:
        reprocess_branch(db_path=db_path, pipeline_version_id=1, chain=chain,
                         scan_id=1, branch_label="")
    except Exception:
        pass  # fallback path hits chain.enqueue which the mock lacks — fine
    assert chain.calls == []  # never took the branch-resume path
