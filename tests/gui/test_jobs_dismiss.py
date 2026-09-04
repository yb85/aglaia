# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Clearing finished Mistral batch jobs out of the Jobs tab (#125).

The table lists jobs from the ACCOUNT (`client.batch.jobs.list`), and the
Mistral Batch API has create / get / list / cancel and **no delete** — a
finished job is permanent account history. So "delete" here can only mean
"stop showing me this", which makes it reversible and makes saying so in the
UI part of the feature rather than a caveat.

Dismissing also drops the project's own record of the job: that row is what
makes it "pending" for the OCR card's *Check result*, and leaving it would
keep asking about a job the user has just finished with.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.MistralJobsTab import (COL_DISMISS, TERMINAL,    # noqa: E402
                                       MistralJobsTab)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _tab(monkeypatch, rows, dismissed=()):
    """A tab with the network and the config DB stubbed out."""
    import aglaia.gui.MistralJobsTab as mod
    saved = []
    monkeypatch.setattr(mod, "_load_dismissed", lambda: set(dismissed))
    monkeypatch.setattr(mod, "_save_dismissed", lambda ids: saved.append(set(ids)))
    monkeypatch.setattr(MistralJobsTab, "refresh", lambda self: None)
    t = MistralJobsTab(db_path="")
    t._populate(rows, "")
    return t, saved


def _job(jid, status, **kw):
    d = {"id": jid, "status": status, "created_at": "2026-09-04T10:00:00",
         "project": "/tmp/x.agl", "chunk": 0, "chunks_total": 1,
         "total": 3, "succeeded": 3}
    d.update(kw)
    return d


def _ids(tab):
    return [tab.table.item(r, 0) is not None for r in range(tab.table.rowCount())]


def test_a_finished_job_gets_a_button(app, monkeypatch):
    t, _ = _tab(monkeypatch, [_job("a", "SUCCESS")])
    assert t.table.cellWidget(0, COL_DISMISS) is not None


@pytest.mark.parametrize("status", sorted(TERMINAL))
def test_every_terminal_status_can_be_dismissed(app, monkeypatch, status):
    t, _ = _tab(monkeypatch, [_job("a", status)])
    assert t.table.cellWidget(0, COL_DISMISS) is not None


@pytest.mark.parametrize("status", ["RUNNING", "QUEUED",
                                    "CANCELLATION_REQUESTED"])
def test_a_live_job_gets_none(app, monkeypatch, status):
    """Nothing to clear away yet — and a trash button beside a RUNNING row
    would read as "cancel", which it is not."""
    t, _ = _tab(monkeypatch, [_job("a", status)])
    assert t.table.cellWidget(0, COL_DISMISS) is None


def test_dismissing_hides_the_row_and_persists(app, monkeypatch):
    t, saved = _tab(monkeypatch, [_job("a", "SUCCESS"), _job("b", "FAILED")])
    assert t.table.rowCount() == 2
    t._dismiss("a")
    assert t.table.rowCount() == 1
    assert saved and saved[-1] == {"a"}


def test_a_dismissed_job_stays_hidden_on_the_next_listing(app, monkeypatch):
    """The account still returns it — the API cannot delete it."""
    t, _ = _tab(monkeypatch, [_job("a", "SUCCESS"), _job("b", "FAILED")],
                dismissed={"a"})
    assert t.table.rowCount() == 1


def test_the_count_says_how_many_are_hidden(app, monkeypatch):
    t, _ = _tab(monkeypatch, [_job("a", "SUCCESS"), _job("b", "FAILED")],
                dismissed={"a"})
    assert "1 dismissed" in t._status_lbl.text()


def test_nothing_hidden_means_no_show_toggle(app, monkeypatch):
    t, _ = _tab(monkeypatch, [_job("a", "SUCCESS")])
    assert t._show_chk.isVisible() is False


def test_showing_dismissed_brings_them_back_with_an_undo(app, monkeypatch):
    t, _ = _tab(monkeypatch, [_job("a", "SUCCESS"), _job("b", "FAILED")],
                dismissed={"a"})
    t._on_show_dismissed(True)
    assert t.table.rowCount() == 2
    assert t.table.cellWidget(0, COL_DISMISS) is not None


def test_restoring_un_hides_it(app, monkeypatch):
    t, saved = _tab(monkeypatch, [_job("a", "SUCCESS")], dismissed={"a"})
    assert t.table.rowCount() == 0
    t._restore("a")
    assert t.table.rowCount() == 1
    assert saved[-1] == set()


def test_a_job_with_no_id_is_never_dismissable(app, monkeypatch):
    t, _ = _tab(monkeypatch, [_job("", "SUCCESS")])
    assert t.table.cellWidget(0, COL_DISMISS) is None
