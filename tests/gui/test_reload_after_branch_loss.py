# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A debug tab whose branch was deleted must not sit busy forever (#121).

`_reprocess` puts the editor in a busy state and only `reload_for` clears it.
That method returned False on two paths that the layout set (#118) made
reachable for the first time:

* the rerun reports another branch, because ours no longer exists;
* our branch resolves to no node at all.

Deleting a layout hits both, and the editor stayed disabled on "Reprocessing…"
for the rest of the session over a page that had already finished.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.DebugViewerTab import DebugViewerWidget          # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Note:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _s):
        pass


def _viewer(branch, leaves, rebuilt):
    """A viewer stubbed down to what `reload_for` touches.

    `leaves` maps a branch label to its surviving leaf id; "" is the
    any-branch lookup `_resolve_leaf` does when it drops the filter.
    """
    v = DebugViewerWidget.__new__(DebugViewerWidget)
    v._row_keys = [(7, branch, "PageDetector")]
    v.leaf_node_id = 99
    v.busy = None
    v._overlay_note = _Note()
    v._editor_stack = type("S", (), {"setEnabled": lambda _s, _v: None})()
    v.reprocess_btn = type("B", (), {"setEnabled": lambda _s, _v: None})()
    v._clear_btn = v.reprocess_btn
    v._busy = True
    v.tr = lambda t: t
    v.strip = type("Strip", (), {"currentRow": lambda _s: 0})()
    v._resolve_leaf = lambda sid, br: leaves.get(br)
    v._rebuild = lambda **kw: rebuilt.append(kw)
    v._set_busy = lambda on: setattr(v, "_busy", bool(on))
    return v


def test_a_surviving_branch_reloads_as_before(app):
    rebuilt = []
    v = _viewer("A", {"A": 42, "": 42}, rebuilt)
    assert v.reload_for(7, "A") is True
    assert v.leaf_node_id == 42
    assert v._busy is False
    assert rebuilt


def test_a_deleted_branch_retargets_the_survivor(app):
    """The rerun reports branch A; this tab was showing B, which is gone."""
    rebuilt = []
    v = _viewer("B", {"A": 42, "": 42}, rebuilt)          # no "B"
    assert v.reload_for(7, "A", node_id=42) is True
    assert v.leaf_node_id == 42
    assert v._busy is False


def test_a_deleted_branch_does_not_adopt_the_other_branchs_leaf_blindly(app):
    """`node_id` belongs to the branch that reported, not to ours. Dropping
    it and re-resolving is what keeps the tab on a real chain."""
    rebuilt = []
    v = _viewer("B", {"A": 42, "": 77}, rebuilt)
    v.reload_for(7, "A", node_id=42)
    assert v.leaf_node_id == 77


def test_another_branch_is_still_ignored_while_ours_lives(app):
    """The ordinary case must not regress into reloading on every sibling."""
    rebuilt = []
    v = _viewer("B", {"A": 42, "B": 43, "": 43}, rebuilt)
    assert v.reload_for(7, "A") is False
    assert rebuilt == []


def test_a_scan_with_nothing_left_clears_busy_and_says_so(app):
    v = _viewer("B", {}, [])
    assert v.reload_for(7, "A") is False
    assert v._busy is False                     # the whole point
    assert "no longer exists" in v._overlay_note.text


def test_a_different_scan_is_not_ours(app):
    v = _viewer("A", {"A": 42, "": 42}, [])
    assert v.reload_for(8, "A") is False
