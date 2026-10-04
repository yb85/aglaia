# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Double-clicking a .agl must open it, not the picker (#168).

macOS delivers the document as a `QFileOpenEvent` once an event loop is
already turning — after the launcher has shown `StartupWindow`. The dialog
runs a modal loop of its own, so stashing the path in an app property reaches
nobody until that loop ends, and cancelling threw the path away.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QWidget   # noqa: E402

from aglaia.app import dismiss_launcher                        # noqa: E402
from aglaia.gui.StartupWindow import StartupWindow             # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _launcher(qapp) -> StartupWindow:
    """A StartupWindow without running its constructor: it builds the whole
    recent-projects pane, and none of that is what this is about."""
    w = StartupWindow.__new__(StartupWindow)
    QDialog.__init__(w)
    w.show()
    return w


def test_a_visible_launcher_is_closed(qapp):
    w = _launcher(qapp)
    assert w.isVisible()
    assert dismiss_launcher(qapp) is True
    assert not w.isVisible()
    # Rejected, not accepted — the launcher loop must not read `choice()`.
    assert w.result() == QDialog.DialogCode.Rejected
    w.deleteLater()


def test_nothing_to_dismiss_is_not_an_error(qapp):
    assert dismiss_launcher(qapp) is False


def test_other_windows_are_left_alone(qapp):
    """Only the launcher. A project window takes the other path in the
    filter — it is closed so the chain stops first, and reopened."""
    other = QWidget()
    other.show()
    assert dismiss_launcher(qapp) is False
    assert other.isVisible()
    other.deleteLater()


def test_a_hidden_launcher_is_not_counted(qapp):
    w = _launcher(qapp)
    w.hide()
    assert dismiss_launcher(qapp) is False
    w.deleteLater()
