# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PipelineProgressBar — OCR ("tick") progress is isolated from the pipeline's
deduped `mark_done`, so a concurrent chain branch_ready can't inflate an active
OCR pass (the "334/322" over-count)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from aglaia.gui.StatusBarWidget import PipelineProgressBar  # noqa: E402


@pytest.fixture(scope="module")
def _qapp():
    return QApplication.instance() or QApplication([])


def test_ocr_ticks_isolated_from_pipeline_mark_done(_qapp):
    bar = PipelineProgressBar()
    bar.reset()
    bar.set_imported(322)
    # The chain (still running) emits branch_ready → mark_done while OCR ticks.
    for i in range(12):
        bar.mark_done(1000 + i)   # concurrent pipeline pollution
        bar.mark_tick()           # real OCR completion
    assert bar._done_count() == 12          # NOT 24, NOT 322+12=334
    assert "12/322" in bar._build_label()


def test_new_ocr_run_restarts_from_zero(_qapp):
    bar = PipelineProgressBar()
    bar.reset()
    bar.set_imported(322)
    for _ in range(322):
        bar.mark_tick()
    assert bar._done_count() == 322
    # A second pass (e.g. re-OCR with another engine) must restart, not stack.
    bar.reset()
    bar.set_imported(322)
    bar.mark_tick()
    bar.mark_tick()
    assert bar._done_count() == 2
