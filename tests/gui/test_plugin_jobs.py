# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A plugin call runs off the GUI thread, and always answers.

`send` and `check` reach the network, and their duration is not ours: mailing a
45 MB PDF opens an SMTP session with a 60 s timeout, and a calibre server on a
sleeping laptop answers when it answers.

The rule this file exists for is the second one. Whatever the plugin does, the
`done` signal must fire — an exception escaping `run()` leaves the window at
"Sending…" with a disabled button forever, which looks like a hang, never
resolves, and is a worse failure than the one that caused it.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer                       # noqa: E402
from PySide6.QtWidgets import QApplication                          # noqa: E402

from aglaia.gui.plugin_jobs import DestinationJob, Outcome          # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _run(call, timeout_ms=5000):
    """Run one job and return its Outcome, or fail if none ever arrives."""
    got = []
    loop = QEventLoop()
    job = DestinationJob(call)
    job.done.connect(lambda o: (got.append(o), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    job.start()
    loop.exec()
    job.wait(2000)
    assert got, "the job never emitted `done` — the GUI would hang forever"
    return got[0]


class _Result:
    def __init__(self, ok, message):
        self.ok, self.message = ok, message


def test_it_does_not_run_on_the_calling_thread(app):
    """The whole point. If it ran inline the GUI would freeze for the call's
    duration, which is the bug this class exists to fix."""
    import threading
    here = threading.get_ident()
    out = _run(lambda: _Result(True, str(threading.get_ident())))
    assert out.message != str(here)


def test_a_result_comes_back_whole(app):
    out = _run(lambda: _Result(True, "Added to calibre as #7."))
    assert out.ok is True
    assert out.message == "Added to calibre as #7."
    assert out.error == ""


def test_a_refusal_is_not_a_failure(app):
    """`ok=False` from the plugin is an answer, not a crash — it carries a
    message the user is meant to read."""
    out = _run(lambda: _Result(False, "The server rejected the password."))
    assert out.ok is False
    assert out.message == "The server rejected the password."
    assert out.error == ""


class TestTheSignalAlwaysFires:
    def test_after_an_ordinary_exception(self, app):
        def boom():
            raise TimeoutError("no route to host")
        out = _run(boom)
        assert out.ok is False
        assert out.message == "TimeoutError: no route to host"
        assert out.error == "TimeoutError: no route to host"

    def test_after_a_baseexception(self, app):
        """This is plugin code. A plugin that calls `sys.exit()` in a worker
        thread must not take the Export button down with it."""
        out = _run(lambda: (_ for _ in ()).throw(SystemExit(1)))
        assert out.ok is False and "SystemExit" in out.message

    def test_when_the_plugin_returns_nothing_at_all(self, app):
        """A `send` that forgets to return still has to leave the UI usable."""
        out = _run(lambda: None)
        assert out.ok is False
        assert out.message == ""


def test_an_outcome_with_no_result_is_not_ok():
    assert Outcome(None, "boom").ok is False
    assert Outcome(None, "boom").message == "boom"


def test_the_trust_sentence_is_one_source_of_truth(app):
    """The label and the comparison must be the same string.

    Two `tr()` calls with identical source text translate identically today,
    and become a dialog that can never be completed the day one of them is
    edited — with no error, because the button simply stays disabled."""
    from aglaia.gui.PluginsTab import trust_sentence
    a, b = trust_sentence(), trust_sentence()
    assert a == b and a.strip() == a and len(a.split()) >= 4
