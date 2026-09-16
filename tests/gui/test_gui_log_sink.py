# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A plugin's diagnostics belong in the Log tab, not on stdout.

A destination test that fails prints the server's answer — the one line that
makes the failure diagnosable. It went to stdout, which a bundled app does
not have, while the dialog told the user to look at the Log tab.
"""
import pytest

pytest.importorskip("PySide6")

from aglaia.gui import gui_log                                  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    yield
    gui_log.set_sink(None)


def test_a_line_reaches_the_installed_sink():
    seen = []
    gui_log.set_sink(lambda level, text: seen.append((level, text)))
    gui_log.log("error", "[plugins] send-to-corpus check: 302")
    assert seen == [("error", "[plugins] send-to-corpus check: 302")]


def test_without_a_sink_the_line_still_goes_somewhere(capsys):
    gui_log.log("info", "no window here")
    assert "no window here" in capsys.readouterr().out


def test_a_broken_sink_never_takes_the_caller_down(capsys):
    def _boom(level, text):
        raise RuntimeError("the log tab is gone")
    gui_log.set_sink(_boom)
    gui_log.log("error", "still said")
    assert "still said" in capsys.readouterr().err


def test_the_settings_dialog_logs_a_failed_test(monkeypatch, tmp_path):
    """The whole point: the detail of a refused connection is in the log."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from aglaia.gui.PluginsTab import PluginSettingsDialog
    from aglaia.plugin_api import CheckResult, Destination, Field

    class D(Destination):
        name = "probe"
        CONFIG_FIELDS = (Field("base_url", "URL", "str", ""),)

        def check(self):
            return CheckResult(False, "Refused.", {"status": 302},
                               kind=CheckResult.AUTH)

    class _Store(dict):
        available = True

        def get(self, key, default=None):
            return dict.get(self, key, default)

        def set(self, key, value):
            self[key] = value

    class _Ctx:
        def __init__(self):
            self.config = _Store()
            self.secrets = _Store()

    QApplication.instance() or QApplication([])
    seen = []
    gui_log.set_sink(lambda level, text: seen.append((level, text)))
    d = D()
    d.ctx = _Ctx()
    dlg = PluginSettingsDialog(d)
    dlg._on_test()
    dlg._test_job.wait(5000)
    QApplication.processEvents()
    assert any("[plugins] probe check:" in t for _l, t in seen), seen
    assert any("302" in t for _l, t in seen), seen
