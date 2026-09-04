# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Export plugins appear where the export happens.

An installed export plugin is only useful if it is offered in the Export tab.
Two rules make the section honest: it lists only destinations that accept the
format currently selected — being told "not accepted" *after* the export ran is
a thing to prevent, not to report — and a destination that is not configured
says what it needs and offers setup instead of a Send button that would fail.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (QApplication, QLabel,                 # noqa: E402
                               QPushButton)

from aglaia.gui.sidebar.tabs.ExportTab import ExportTab              # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import importlib
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    for m in (ad, pc):
        importlib.reload(m)
    from aglaia.workers import destinations as d
    d.reset_for_tests()
    tab = ExportTab()
    yield tab
    # This test file registers a destination in a process-global registry.
    # Leaving it there made the bundled-destination tests see four where they
    # expect three — which is how `forget()` came to exist.
    d.forget("ready-dest")
    d.reset_for_tests()


def _cards(tab):
    return [tab._send_layout.itemAt(i).widget()
            for i in range(tab._send_layout.count())]


def _texts(card):
    return [w.text() for w in card.findChildren(QLabel)]


def _button(card):
    return card.findChildren(QPushButton)[0]


def test_pdf_offers_the_destinations_that_take_a_pdf(tab):
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    names = [_texts(c)[0] for c in _cards(tab)]
    assert "Export to Calibre server" in names
    assert "Export to Kindle by email" in names


def test_a_format_nobody_accepts_hides_the_whole_section(tab):
    """An empty heading with a promise under it is worse than no heading."""
    tab.format_group.set_current_key("slim")     # .agl — nothing takes it
    tab.refresh_destinations()
    assert _cards(tab) == []
    assert tab._send_label.isVisibleTo(tab) is False
    assert tab._send_box.isVisibleTo(tab) is False


def test_an_unconfigured_destination_says_what_it_needs(tab):
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    card = next(c for c in _cards(tab)
                if _texts(c)[0] == "Export to Calibre server")
    assert "Needs:" in _texts(card)[1]
    assert _button(card).text() == "Set up…"


def _install_ready_destination(tmp_path):
    """A destination that needs nothing — so "Ready" is reachable in a test.

    Written into the tmp APP_DATA rather than skipping when a registry plugin
    happens not to be installed: a skip that always skips proves nothing."""
    import importlib
    from aglaia.app_data import plugin_registry as reg
    from aglaia.workers import destinations as d
    slug = "ready-dest"
    p = reg.installed_root("destinations") / slug
    p.mkdir(parents=True, exist_ok=True)
    (p / "aglaia-plugin.toml").write_text(
        '[plugin]\nslug = "ready-dest"\nname = "Ready dest"\n'
        'version = "1.0.0"\nentry = "ready_dest.py"\nlicense = "MIT"\n'
        '[requires]\napi = 1\n[capabilities]\nconfig = true\n',
        encoding="utf-8")
    (p / "ready_dest.py").write_text(
        "from aglaia.plugin_api import Destination, register_destination\n"
        "@register_destination\n"
        "class R(Destination):\n"
        "    name = 'ready-dest'\n"
        "    display = 'Ready dest'\n"
        "    accepts = ('pdf',)\n",
        encoding="utf-8")
    d.reset_for_tests()
    return slug


def test_a_configured_destination_offers_send(tab, tmp_path):
    slug = _install_ready_destination(tmp_path)
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    card = next(c for c in _cards(tab) if _texts(c)[0] == "Ready dest")
    assert _texts(card)[1] == "Ready"
    assert _button(card).text() == "Send"


def test_pressing_send_asks_the_host_by_name(tab, tmp_path):
    """The tab knows WHICH destination, not how to reach it."""
    slug = _install_ready_destination(tmp_path)
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    asked = []
    tab.send_to_requested.connect(asked.append)
    card = next(c for c in _cards(tab) if _texts(c)[0] == "Ready dest")
    _button(card).click()
    assert asked == [slug]


def test_pressing_set_up_asks_for_settings_not_a_send(tab):
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    sends, settings = [], []
    tab.send_to_requested.connect(sends.append)
    tab.destination_settings_requested.connect(settings.append)
    card = next(c for c in _cards(tab)
                if _texts(c)[0] == "Export to Calibre server")
    _button(card).click()
    assert settings == ["send-to-calibre"] and sends == []


def test_changing_the_format_rebuilds_the_section(tab):
    tab.format_group.set_current_key("slim")
    tab.refresh_destinations()
    assert _cards(tab) == []
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    assert _cards(tab)


def test_forget_removes_a_destination_from_this_process(tab, tmp_path):
    """Uninstalling deletes the files; the class stays in the registry and the
    module in sys.modules unless something undoes the import. Without that, a
    removed destination went on being listed and offered until restart."""
    import sys
    from aglaia.plugin_api import DESTINATION_REGISTRY
    from aglaia.workers import destinations as d
    _install_ready_destination(tmp_path)
    d.load_all()
    assert "ready-dest" in DESTINATION_REGISTRY
    assert "aglaia_plugin_ready_dest" in sys.modules
    # The real pairing: uninstall removes the files, forget removes the
    # registration. Either alone leaves the destination half-present — files
    # gone but still offered, or unregistered and re-imported on the next
    # discovery.
    from aglaia.app_data import plugin_registry as reg
    reg.uninstall("ready-dest")
    d.forget("ready-dest")
    assert "ready-dest" not in DESTINATION_REGISTRY
    assert "aglaia_plugin_ready_dest" not in sys.modules
    d.reset_for_tests()
    tab.format_group.set_current_key("pdf")
    tab.refresh_destinations()
    assert "Ready dest" not in [_texts(c)[0] for c in _cards(tab)]


def test_forget_alone_does_not_survive_the_next_discovery(tab, tmp_path):
    """Because the files are still there. This is why uninstall does both,
    and why `forget` is not offered as an "unload" button."""
    from aglaia.workers import destinations as d
    _install_ready_destination(tmp_path)
    d.load_all()
    d.forget("ready-dest")
    d.reset_for_tests()
    assert "ready-dest" in d.load_all()
