# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Installing a plugin must not freeze the window.

The index fetch was threaded from the start; the install was not — and it is
the slower of the two, one HTTP request per file in the plugin. On a slow link
that was over a minute of beach ball with no way to tell a slow install from a
hung one.

These tests assert the shape that prevents it: the work runs on a QThread, the
GUI thread stays free while it does, and the button says what is happening.
"""
import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread                                  # noqa: E402
from PySide6.QtWidgets import QApplication                          # noqa: E402

from aglaia.app_data.plugin_registry import InstallResult, RegistryEntry  # noqa: E402
from aglaia.gui.PluginsTab import _InstallJob                       # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _entry(slug="a-plugin"):
    return RegistryEntry(slug=slug, kind="destinations", name="A",
                         version="1.0.0",
                         files={"a.py": "sha256:0", "b.py": "sha256:0"})


def test_the_install_runs_off_the_gui_thread(app, monkeypatch):
    import aglaia.app_data.plugin_registry as reg
    gui_thread = QThread.currentThread()
    seen = {}

    def _slow(entry, **kw):
        seen["thread"] = QThread.currentThread()
        time.sleep(0.25)
        return InstallResult(True, "installed")

    monkeypatch.setattr(reg, "install_from_registry", _slow)
    job = _InstallJob(_entry())
    out = []
    job.done.connect(out.append)
    t0 = time.time()
    job.start()
    # The GUI thread must be free while it works — this loop is what a user
    # moving the mouse would be doing.
    spins = 0
    while job.isRunning() and time.time() - t0 < 5:
        app.processEvents()
        spins += 1
    job.wait(5000)
    app.processEvents()
    assert seen["thread"] is not gui_thread, "install ran on the GUI thread"
    assert spins > 5, "the GUI thread was blocked"
    assert out and out[0].ok


def test_a_failure_comes_back_as_a_result_not_an_exception(app, monkeypatch):
    """A raise inside the thread would be lost and the button would stay
    'Installing…' forever."""
    import aglaia.app_data.plugin_registry as reg

    def _boom(entry, **kw):
        raise RuntimeError("network gone")

    monkeypatch.setattr(reg, "install_from_registry", _boom)
    job = _InstallJob(_entry())
    out = []
    job.done.connect(out.append)
    job.start()
    job.wait(5000)
    QApplication.processEvents()
    assert out and out[0].ok is False and "network gone" in out[0].message


def test_progress_names_the_file_and_the_count(app, monkeypatch):
    import aglaia.app_data.plugin_registry as reg

    def _with_progress(entry, on_progress=None, **kw):
        on_progress(1, 2, "a.py")
        on_progress(2, 2, "b.py")
        return InstallResult(True, "installed")

    monkeypatch.setattr(reg, "install_from_registry", _with_progress)
    job = _InstallJob(_entry())
    msgs = []
    job.progress.connect(msgs.append)
    job.start()
    job.wait(5000)
    QApplication.processEvents()
    assert msgs == ["A: a.py (1/2)", "A: b.py (2/2)"]


def test_the_registry_reports_progress_per_file(monkeypatch, tmp_path):
    """The count comes from the index entry, so it is right before the first
    byte is fetched."""
    import aglaia.app_data.plugin_registry as reg
    seen = []

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, **kw):
            return type("R", (), {"status_code": 404, "content": b""})()

    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **kw: _C())
    reg.install_from_registry(
        _entry(), on_progress=lambda i, n, rel: seen.append((i, n, rel)))
    assert seen and seen[0] == (1, 2, "a.py")
