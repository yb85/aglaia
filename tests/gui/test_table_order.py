# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The list view follows DISPLAY order, not scan_id order (#105).

The grid is seeded from `ScanRepo.list_scans` (``ORDER BY page_order``) and a
drag moves the card inside that layout. The table used to enumerate the same
widgets `sorted()` by scan_id, so a reorder was invisible here — and the drop
index it handed the shared handler indexed a different sequence than the grid
layout that handler walks.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.ScansTableView import ScansTableView, _SnapBlock  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Snap:
    """The bits of `ScanItemWidget` the table reads."""

    def __init__(self, sid: int):
        self.scan_id = sid
        self.idx = sid
        self.raw_filestem = f"scan-{sid}"
        self.items: dict = {}
        self.global_history: list = []
        self._ocr_state = "none"


class _Loader:
    ready = None

    def request(self, *a, **k):
        return None


def _table(order):
    widgets = {sid: _Snap(sid) for sid in order}
    t = ScansTableView(get_snap_widgets=lambda: dict(widgets),
                       thumb_loader=_Loader())
    return t, widgets


def _rows(t):
    return [w.scan_id for w in t._snap_blocks()]


def test_refresh_keeps_the_providers_order(app):
    """7, 3, 5 is what a reorder leaves behind. Sorted, it would read 3,5,7
    and the view would silently disagree with the grid."""
    t, _ = _table([7, 3, 5])
    t.refresh()
    assert _rows(t) == [7, 3, 5]


def test_add_snap_lands_where_display_order_puts_it(app):
    t, widgets = _table([7, 3, 5])
    t.refresh()
    # A page inserted BETWEEN two others in display order — its scan_id (9)
    # would have appended it last under the old id comparison.
    widgets.clear()
    for sid in (7, 9, 3, 5):
        widgets[sid] = _Snap(sid)
    t.add_snap(9)
    assert _rows(t) == [7, 9, 3, 5]


def test_add_snap_appends_when_it_sorts_last(app):
    t, widgets = _table([7, 3, 5])
    t.refresh()
    widgets[2] = _Snap(2)          # last in display order, lowest id
    t.add_snap(2)
    assert _rows(t) == [7, 3, 5, 2]
    assert isinstance(t._snap_blocks()[-1], _SnapBlock)
