# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A project-wide reprocess must leave nothing of the previous run on screen
(#162).

Three ways the cards lied after "Reprocess all" / "Apply pipeline + reprocess":

* the decoded-pixmap cache is keyed by `image_id`, and a reprocess deletes the
  top of `images` and inserts over the freed rowids — so a hit is the previous
  run's picture;
* `forget_layouts` (#123) ran only for the per-scan rerun, so the project-wide
  ones kept columns the new run no longer produces;
* a card keeps the pipeline it was born with, and `_register_node` appends
  unknown step names, so an edited pipeline left it walking a merged rail.

And the reconciler that rescues a dropped `branch_ready` would force the bar to
100% over a run that had genuinely lost scans.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow      # noqa: E402

from aglaia.gui.MainWindow import MainWindow                 # noqa: E402
from aglaia.gui.ScanItemWidget import ScanItemWidget         # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ── the card ───────────────────────────────────────────────────────────

def _card(steps=("01_a", "02_b", "03_c")):
    """A widget stubbed to the bookkeeping these methods touch."""
    w = ScanItemWidget.__new__(ScanItemWidget)
    w.raw_filestem = "page_001"
    w.pipeline_steps = list(steps)
    w.max_steps = len(steps)
    w.global_history = ["raw"] + list(steps)
    w.current_history_idx = 2
    w.items = {
        "page_001": {"history": ["raw"], "nodes": {"raw": {"node_id": 1}},
                     "node_to_step": {1: "raw"}, "parent": None,
                     "children": ["page_001_A"], "current_idx": 2,
                     "trashed": False},
        "page_001_A": {"history": ["01_a"],
                       "nodes": {"01_a": {"node_id": 10, "image_id": 77,
                                          "meta": {}}},
                       "node_to_step": {10: "01_a"}, "parent": "page_001",
                       "children": [], "current_idx": 2, "trashed": False},
    }
    w._stem_for_node = {1: "page_001", 10: "page_001_A"}
    w._image_ids = {77}
    w._pix_cache = OrderedDict({(77, 300, False, None): ("old-pixmap",)})
    w._pix_cache_cap = 96
    return w


def test_rerunning_a_step_drops_the_memoised_pixmap(qapp):
    """The regression: same stem, same step, same image id — new pixels.

    `images.id` is a rowid the reprocess reuses, so "the id did not change"
    is not evidence the picture did not.
    """
    w = _card()
    w._register_node("page_001_A", "01_a", node_id=10, image_id=77, meta={})
    assert w._pix_cache == {}


def test_a_first_registration_keeps_the_cache(qapp):
    """A stage arriving for the first time is not a rerun — clearing there
    would throw the cache away once per stage of every normal run."""
    w = _card()
    w._register_node("page_001_A", "02_b", node_id=11, image_id=78, meta={})
    assert (77, 300, False, None) in w._pix_cache


def test_forget_layouts_drops_the_cache_even_with_one_layout(qapp):
    """The common reprocess shape: same layout count, all-new images. The
    early return for "nothing dropped" must not skip the cache."""
    w = _card()
    w.items = {"page_001": w.items["page_001"]}
    w.items["page_001"]["children"] = []
    w.forget_layouts()
    assert w._pix_cache == {}


def test_set_pipeline_steps_rebuilds_the_rail(qapp):
    w = _card()
    w.global_history.append("99_appended_by_an_old_run")
    w.set_pipeline_steps(["01_a", "02_new", "03_b", "04_c"])
    assert w.pipeline_steps == ["01_a", "02_new", "03_b", "04_c"]
    assert w.max_steps == 4
    assert w.global_history == ["raw", "01_a", "02_new", "03_b", "04_c"]
    assert w.current_history_idx == 0
    assert all(e["current_idx"] == 0 for e in w.items.values())
    assert w._pix_cache == {}


def test_set_pipeline_steps_is_a_noop_when_unchanged(qapp):
    w = _card()
    w.set_pipeline_steps(["01_a", "02_b", "03_c"])
    assert w.current_history_idx == 2          # cursor not disturbed
    assert w._pix_cache                        # cache kept


# ── the window ─────────────────────────────────────────────────────────

class _FakeCard:
    def __init__(self):
        self.forgot = False
        self.processing = None

    def forget_layouts(self):
        self.forgot = True

    def set_processing(self, on):
        self.processing = on


def _mw(qapp, db_path=None):
    m = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(m)
    m.scan_widgets_by_scan = {}
    m.db_path = db_path
    m._logged = []
    m._toasts = []
    m._on_log_line = lambda lvl, txt: m._logged.append((lvl, txt))
    m.toast = lambda txt, *a, **k: m._toasts.append(txt)
    return m


def test_prepare_cards_for_rerun_resets_every_card(qapp):
    m = _mw(qapp)
    m.scan_widgets_by_scan = {1: _FakeCard(), 2: _FakeCard()}
    assert m._prepare_cards_for_rerun() == 2
    assert all(c.forgot and c.processing is True
               for c in m.scan_widgets_by_scan.values())


def test_one_bad_card_does_not_stop_the_rerun(qapp):
    class _Boom(_FakeCard):
        def forget_layouts(self):
            raise RuntimeError("boom")

    m = _mw(qapp)
    good = _FakeCard()
    m.scan_widgets_by_scan = {1: _Boom(), 2: good}
    assert m._prepare_cards_for_rerun() == 1
    assert good.forgot


# ── the reconciler ─────────────────────────────────────────────────────

def _project(tmp_path, *, missing: set[int]):
    """A real project DB with 4 active scans; those in `missing` have no
    branch row — what a reprocess leaves behind when it never got to them."""
    from aglaia.storage.db import open_db

    p = tmp_path / "p.agl"
    conn = open_db(p)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO pipeline_versions (id, yaml_text, yaml_sha256, name, "
        "step_count, created_at, is_active) VALUES (1,'','x','p',1,'t',1)")
    for i in range(1, 5):
        conn.execute(
            "INSERT INTO scans (id, idx, source, pipeline_version_id, "
            "created_at) VALUES (?, ?, 'capture', 1, 't')", (i, i))
        if i not in missing:
            conn.execute(
                "INSERT INTO branches (scan_id, branch_path, "
                "terminal_node_id, chosen_node_id, created_at, updated_at) "
                "VALUES (?, 'A', 1, 1, 't', 't')", (i,))
    conn.commit()
    conn.close()
    return p


def test_a_hole_is_reported_not_reconciled_away(qapp, tmp_path):
    m = _mw(qapp, _project(tmp_path, missing={2, 3}))
    m._report_scans_left_unprocessed()
    assert m._reported_unprocessed == [2, 3]
    assert any("2, 3" in t for _, t in m._logged)
    assert m._toasts


def test_a_complete_run_says_nothing(qapp, tmp_path):
    m = _mw(qapp, _project(tmp_path, missing=set()))
    m._report_scans_left_unprocessed()
    assert m._reported_unprocessed == []
    assert not m._toasts and not m._logged


def test_a_trashed_branch_is_not_missing_output(qapp, tmp_path):
    """Hiding a page is a user decision, not a lost scan — but a scan whose
    only branch is hidden has no output left to export, so it counts."""
    from aglaia.storage.db import open_db

    p = _project(tmp_path, missing=set())
    conn = open_db(p)
    conn.execute("UPDATE branches SET trashed_at = 'x' WHERE scan_id = 4")
    conn.commit()
    conn.close()
    m = _mw(qapp, p)
    m._report_scans_left_unprocessed()
    assert m._reported_unprocessed == [4]


def test_the_same_hole_is_reported_only_once(qapp, tmp_path):
    m = _mw(qapp, _project(tmp_path, missing={2}))
    m._report_scans_left_unprocessed()
    m._report_scans_left_unprocessed()
    assert len(m._toasts) == 1
