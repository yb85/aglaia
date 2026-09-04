# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A finished QThread must not poison its own guard (#111).

`_track_worker` calls `deleteLater()` when the thread finishes. That frees
the C++ object while the owning Python attribute keeps pointing at the husk,
and `isRunning()` on a husk does NOT answer False — it raises
`RuntimeError: Internal C++ object already deleted`, straight out of whatever
slot re-checked it. One "Check result" click therefore left the button dead
for the rest of the session: the exception unwound before the worker was ever
constructed, so not even the toast appeared.

Two defences, both pinned here: the attribute is cleared when the thread
finishes, and the liveness check treats "already deleted" as "not running".
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread                               # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.MainWindow import MainWindow                     # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Worker(QThread):
    def run(self):
        return None


class _Host:
    """Stands in for the window — `_track_worker` only touches attributes."""

    _batch_worker = None

    _track_worker = MainWindow._track_worker
    # `_worker_alive` is a staticmethod — take it off the class so the
    # descriptor isn't re-bound into an instance method here.
    _worker_alive = staticmethod(MainWindow._worker_alive)


def _run_to_completion(app, w):
    w.start()
    w.wait()
    for _ in range(4):            # drain `finished` and the deferred delete
        QApplication.processEvents()


def test_a_finished_worker_clears_the_owning_attribute(app):
    host = _Host()
    host._batch_worker = _Worker()
    host._track_worker(host._batch_worker, attr="_batch_worker")
    _run_to_completion(app, host._batch_worker)
    assert host._batch_worker is None
    assert host._live_workers == []


def test_liveness_is_false_for_a_deleted_worker(app):
    """The raw guard raises here. This is the whole bug, isolated."""
    host = _Host()
    w = _Worker()
    host._track_worker(w)          # no `attr` — the husk stays reachable
    _run_to_completion(app, w)
    with pytest.raises(RuntimeError):
        w.isRunning()              # what the old guard did
    assert host._worker_alive(w) is False


def test_liveness_is_true_only_while_running(app):
    host = _Host()
    assert host._worker_alive(None) is False
    w = _Worker()
    assert host._worker_alive(w) is False       # constructed, not started
    host._track_worker(w, attr="_batch_worker")
    _run_to_completion(app, w)
    assert host._worker_alive(getattr(host, "_batch_worker", None)) is False


def test_tracking_without_an_attr_still_drops_the_live_list_entry(app):
    host = _Host()
    w = _Worker()
    host._track_worker(w)
    assert host._live_workers == [w]
    _run_to_completion(app, w)
    assert host._live_workers == []
