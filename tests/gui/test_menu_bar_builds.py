# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The menu bar builds, and every action it wires actually exists.

This exists because it didn't. A `self._build_plugin_menu(...)` call shipped
without the method: importing the module was fine, and no test constructed a
MainWindow, so the suite stayed green while the app died on launch with
`AttributeError: 'MainWindow' object has no attribute '_build_plugin_menu'`.

An import check proves a module parses. It proves nothing about whether the
names a method calls at RUNTIME are there. `_build_menu_bar` is the one method
that runs unconditionally on every launch and touches a dozen others, so it is
worth the cost of a real instance.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow             # noqa: E402

from aglaia.gui.MainWindow import MainWindow                        # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, tmp_path, monkeypatch):
    """A MainWindow shell with just enough for `_build_menu_bar` to run.

    Constructing the real thing needs a camera, a chain and a project; the
    menu bar needs none of that, and the bug this guards against was in the
    menu bar."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    w = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(w)
    w.tr = lambda t, *a: t
    return w


def test_the_menu_bar_builds(window):
    """The regression: every method `_build_menu_bar` reaches for must exist."""
    window._build_menu_bar()
    titles = [m.title() for m in window.menuBar().findChildren(type(
        window.menuBar().addMenu("x")))]
    assert "Help" in titles
    assert "View" in titles


def test_every_menu_method_it_calls_is_defined(window):
    """Named individually so a failure says WHICH one is missing, rather than
    only that the menu bar blew up."""
    for name in ("_build_plugin_menu", "_open_plugin_window",
                 "open_plugins_tab", "open_mistral_jobs_tab",
                 "_close_current_tab", "_open_model_downloader"):
        assert callable(getattr(window, name, None)), f"missing {name}"


def _menu_titles(window):
    return [m.title() for m in window.menuBar().findChildren(type(
        window.menuBar().addMenu("x")))]


def test_the_plugins_menu_exists_even_with_no_plugin_windows(window):
    """It always carries "Manage plugins…", and a menu that appears only once
    a plugin happens to contribute a window is a menu nobody can learn."""
    from aglaia.plugin_api import WINDOW_REGISTRY
    WINDOW_REGISTRY.clear()
    window._build_menu_bar()
    assert "Plugins" in _menu_titles(window)


def test_there_is_exactly_one_way_into_the_plugins_tab(window):
    """It was in View as well, so the menu bar carried two identical entries
    opening the same tab."""
    from aglaia.plugin_api import WINDOW_REGISTRY
    WINDOW_REGISTRY.clear()
    window._build_menu_bar()
    hits = [a.text() for m in window.menuBar().findChildren(type(
                window.menuBar().addMenu("x")))
            for a in m.actions()
            if "plugin" in a.text().lower()]
    assert len(hits) == 1, f"several ways in: {hits}"


def test_a_contributed_window_gets_a_menu_entry(window, monkeypatch):
    from aglaia.plugin_api import (PluginWindow, WINDOW_REGISTRY,
                                   register_window)
    from aglaia.workers import plugin_windows
    WINDOW_REGISTRY.clear()
    monkeypatch.setattr(plugin_windows, "load_all", lambda **kw: [])
    register_window("stamp-remover",
                    PluginWindow("lib", "Stamp library", factory=lambda c: c))
    window._build_menu_bar()
    titles = [m.title() for m in window.menuBar().findChildren(type(
        window.menuBar().addMenu("x")))]
    assert "Plugins" in titles
    assert "stamp-remover" in titles          # grouped under the slug
    WINDOW_REGISTRY.clear()


def test_a_factory_that_raises_does_not_take_the_app_with_it(window,
                                                             monkeypatch):
    """A plugin's factory is someone else's code."""
    import sys
    from aglaia.plugin_api import PluginWindow
    mw = sys.modules["aglaia.gui.MainWindow"]

    shown = []
    monkeypatch.setattr(mw.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: shown.append(a)))

    def _boom(ctx):
        raise RuntimeError("no")

    window._open_plugin_window(
        "a-plugin", PluginWindow("w", "W", factory=_boom))
    assert shown, "a broken factory must report itself, not vanish"
