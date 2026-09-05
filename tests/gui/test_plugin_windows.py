# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A plugin that contributes a window (#130).

Some plugins need a workspace, not a settings form — a stamp library wants to
show snippets and let you trace a polygon on one, which `Field` cannot express.
So `PluginWindow.factory(ctx) -> QWidget` is a real Qt widget, gated on a
declared `ui` capability.

That gate does not narrow what a plugin COULD do: it already runs in-process
and could import Qt regardless. What it changes is what the install dialog says
and what a reviewer looks for — a plugin that can draw can draw something that
looks like Aglaïa asking for a password.
"""
import importlib

import pytest

from aglaia.app_data import plugin_manifest as pm

UI_TOML = """\
[plugin]
slug = "ui-plugin"
name = "UI plugin"
version = "1.0.0"
entry = "ui_plugin.py"
license = "MIT"
[requires]
api = 1
imports = ["PySide6"]
[capabilities]
config = true
ui = true
"""


def test_pyside_needs_the_ui_capability(tmp_path):
    d = tmp_path / "ui-plugin"
    d.mkdir()
    (d / "aglaia-plugin.toml").write_text(
        UI_TOML.replace("ui = true", "ui = false"), encoding="utf-8")
    with pytest.raises(pm.ManifestError, match="capabilities.ui = true"):
        pm.parse_manifest(d / "aglaia-plugin.toml", kind="processors",
                          expect_slug="ui-plugin")


def test_with_the_capability_it_parses_and_is_declared(tmp_path):
    d = tmp_path / "ui-plugin"
    d.mkdir()
    (d / "aglaia-plugin.toml").write_text(UI_TOML, encoding="utf-8")
    man = pm.parse_manifest(d / "aglaia-plugin.toml", kind="processors",
                            expect_slug="ui-plugin")
    assert man.ui is True
    assert "adds a window to the Plugins menu" in man.declared()


def test_the_scan_allows_qt_only_for_a_ui_plugin():
    man = pm.Manifest(slug="ui-plugin", kind="processors", name="x",
                      version="1", ui=True)
    assert pm.scan_source("from PySide6.QtWidgets import QWidget", man).clean
    man.ui = False
    r = pm.scan_source("from PySide6.QtWidgets import QWidget", man)
    assert r.undeclared and not r.refused


def test_registering_a_window_groups_it_by_slug():
    from aglaia.plugin_api import (PluginWindow, WINDOW_REGISTRY,
                                   register_window)
    WINDOW_REGISTRY.clear()
    register_window("a-plugin", PluginWindow("w1", "One", factory=lambda c: c))
    register_window("a-plugin", PluginWindow("w2", "Two", factory=lambda c: c))
    register_window("b-plugin", PluginWindow("w1", "One", factory=lambda c: c))
    assert [w.title for w in WINDOW_REGISTRY["a-plugin"]] == ["One", "Two"]
    # Same key in two plugins must not collide — the slug is the namespace.
    assert len(WINDOW_REGISTRY["b-plugin"]) == 1
    WINDOW_REGISTRY.clear()


def test_a_disabled_plugin_contributes_no_window(tmp_path, monkeypatch):
    """Import is code execution, so a disabled plugin must not be imported —
    the same rule everything else obeys."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as r
    import aglaia.workers.plugin_windows as pw
    for m in (ad, pc, r, pw):
        importlib.reload(m)
    d = r.installed_root("processors") / "ui-plugin"
    d.mkdir(parents=True)
    (d / "aglaia-plugin.toml").write_text(UI_TOML, encoding="utf-8")
    (d / "ui_plugin.py").write_text(
        "raise RuntimeError('must not be imported')", encoding="utf-8")
    r.set_disabled("ui-plugin", True)
    pw.reset_for_tests()
    assert pw.load_all() == []


def test_a_plugin_that_raises_on_import_is_skipped_not_fatal(tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as r
    import aglaia.workers.plugin_windows as pw
    for m in (ad, pc, r, pw):
        importlib.reload(m)
    d = r.installed_root("processors") / "ui-plugin"
    d.mkdir(parents=True)
    (d / "aglaia-plugin.toml").write_text(UI_TOML, encoding="utf-8")
    (d / "ui_plugin.py").write_text("raise RuntimeError('boom')",
                                    encoding="utf-8")
    pw.reset_for_tests()
    assert pw.load_all() == []          # skipped, and no exception escaped


def test_a_plugin_without_ui_is_not_imported_for_the_menu(tmp_path,
                                                          monkeypatch):
    """The menu pass imports only what declares a window — it must not drag
    every installed processor into the GUI process at startup."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as r
    import aglaia.workers.plugin_windows as pw
    for m in (ad, pc, r, pw):
        importlib.reload(m)
    d = r.installed_root("processors") / "ui-plugin"
    d.mkdir(parents=True)
    (d / "aglaia-plugin.toml").write_text(
        UI_TOML.replace("imports = [\"PySide6\"]", "imports = []")
               .replace("ui = true", "ui = false"), encoding="utf-8")
    (d / "ui_plugin.py").write_text("raise RuntimeError('must not import')",
                                    encoding="utf-8")
    pw.reset_for_tests()
    assert pw.load_all() == []
