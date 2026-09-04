# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Reordering a card in the scans grid (#105).

A drag-reorder is `removeWidget` + `insertWidget` on the FlowLayout. The
removal hides the widget on purpose — reparenting a visible one flashes it
as a bare top-level window on macOS — and `hide()` sets the EXPLICIT hide
flag, which `QLayout.addChildWidget` does not clear. So the card came back
in the right slot and stayed invisible: dragging a scan to a new position
made it vanish.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QWidget      # noqa: E402

from aglaia.gui.FlowLayout import FlowLayout                     # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _host(n: int):
    host = QWidget()
    lay = FlowLayout(host)
    cards = [QLabel(f"w{i}") for i in range(n)]
    for c in cards:
        lay.insertWidget(-1, c)
    host.resize(400, 300)
    host.show()
    QApplication.processEvents()
    return host, lay, cards


def _order(lay):
    return [lay.itemAt(i).widget().text() for i in range(lay.count())]


def test_reorder_keeps_the_card_visible(app):
    host, lay, cards = _host(3)
    moved = cards[0]
    lay.removeWidget(moved)
    lay.insertWidget(2, moved)
    QApplication.processEvents()
    assert _order(lay) == ["w1", "w2", "w0"]
    assert not moved.isHidden()
    assert all(not c.isHidden() for c in cards)


def test_reorder_to_the_front_keeps_it_visible(app):
    host, lay, cards = _host(3)
    moved = cards[2]
    lay.removeWidget(moved)
    lay.insertWidget(0, moved)
    QApplication.processEvents()
    assert _order(lay) == ["w2", "w0", "w1"]
    assert not moved.isHidden()


def test_remove_still_hides_on_the_way_out(app):
    """The hide is deliberate — keep it, or the reparent flashes a bare
    window. This pins the pair: hidden while out, shown once back in."""
    host, lay, cards = _host(2)
    lay.removeWidget(cards[0])
    QApplication.processEvents()
    assert cards[0].isHidden()
    assert cards[0].parent() is None
