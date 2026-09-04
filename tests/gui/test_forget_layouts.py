# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A card must lose the layouts a rerun no longer produces (#123).

`ScanItemWidget.handle_event` only ever ADDS a stem. Nothing ever removed
one, which did not matter while the layout count came from detection and was
stable — until the layout set (#118) let the user delete one. The card then
kept showing the deleted page's thumbnail, built from nodes the rerun had
already dropped from the DB.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.ScanItemWidget import ScanItemWidget             # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _card():
    """A widget stubbed to the bookkeeping `forget_layouts` touches."""
    w = ScanItemWidget.__new__(ScanItemWidget)
    w.raw_filestem = "page_001"
    w.current_history_idx = 3
    w.items = {
        "page_001": {"history": ["raw"], "nodes": {"raw": {"node_id": 1}},
                     "node_to_step": {1: "raw"}, "parent": None,
                     "children": ["page_001_A", "page_001_B"],
                     "current_idx": 3, "trashed": False},
        "page_001_A": {"history": ["pages_2ppf"], "nodes": {}, 
                       "node_to_step": {10: "pages_2ppf"},
                       "parent": "page_001", "children": [],
                       "current_idx": 3, "trashed": False},
        "page_001_B": {"history": ["pages_2ppf"], "nodes": {},
                       "node_to_step": {20: "pages_2ppf"},
                       "parent": "page_001", "children": [],
                       "current_idx": 3, "trashed": False},
    }
    w._stem_for_node = {1: "page_001", 10: "page_001_A", 20: "page_001_B"}
    return w


def test_only_the_raw_source_survives(app):
    w = _card()
    w.forget_layouts()
    assert list(w.items) == ["page_001"]


def test_the_raw_entrys_children_are_dropped_too(app):
    """A stale child name would re-link the moment the raw stem is walked."""
    w = _card()
    w.forget_layouts()
    assert w.items["page_001"]["children"] == []


def test_the_raw_entrys_own_nodes_are_kept(app):
    """It is the source the rerun feeds FROM, not a result of it."""
    w = _card()
    w.forget_layouts()
    assert w.items["page_001"]["nodes"] == {"raw": {"node_id": 1}}


def test_node_lookups_for_dropped_layouts_are_forgotten(app):
    """`_resolve_parent_stem` reads this map; a stale entry would re-attach a
    new node to a layout that no longer exists."""
    w = _card()
    w.forget_layouts()
    assert w._stem_for_node == {1: "page_001"}


def test_a_card_with_no_layouts_yet_is_untouched(app):
    w = _card()
    w.items = {"page_001": w.items["page_001"]}
    w.items["page_001"]["children"] = []
    before = dict(w.items["page_001"])
    w.forget_layouts()
    assert w.items["page_001"] == before


def test_it_is_idempotent(app):
    w = _card()
    w.forget_layouts()
    once = dict(w.items)
    w.forget_layouts()
    assert w.items == once
