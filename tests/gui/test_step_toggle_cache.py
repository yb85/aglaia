# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Per-page step-toggle cache freshness.

`cell_disable_states` is memoised per scan and keyed by NODE id. A (re)processed
scan gets fresh node ids, so the cache MUST be dropped on branch_ready — else
the stale map misses every new node, the round stage-toggle resolves to locked,
and (a locked button can't fire the only other invalidator) the per-page disable
stays dead. Regression: the toggle silently stopped working after any reprocess."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from aglaia.gui.MainWindow import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _mw(qapp):
    m = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(m)
    m.status_bar_widget = SimpleNamespace(
        progress=SimpleNamespace(mark_done=lambda *_: None))
    m.scan_widgets_by_scan = {}
    m._maybe_schedule_live_ocr = lambda: None
    return m


def test_branch_ready_invalidates_stale_disable_cache(qapp):
    m = _mw(qapp)
    # Pre-populate a stale per-scan map (keyed by OLD node ids).
    m.__dict__["_cell_disable_cache"] = {26: {111: (True, False)}, 99: {222: (True, False)}}

    m._on_status_branch_ready({"scan_id": 26})

    # Scan 26's stale entry is dropped → next cell_disable_states re-queries
    # fresh node ids and the toggle resolves correctly again.
    assert 26 not in m.__dict__["_cell_disable_cache"]
    # Other scans' caches are untouched.
    assert 99 in m.__dict__["_cell_disable_cache"]


def test_branch_ready_no_cache_is_safe(qapp):
    m = _mw(qapp)
    # No cache dict yet (nothing rendered) — must not raise.
    m._on_status_branch_ready({"scan_id": 7})
