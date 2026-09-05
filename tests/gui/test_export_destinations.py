# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Export plugins appear where the export happens, as ordinary formats.

An exporter is an exporter. A plugin that puts a finished export somewhere gets
the SAME card as PDF and Markdown, in the same list, selected the same way, run
by the same Export button — not a second control below it with its own send
buttons, which was two ways to start an export for one idea.

What is left for this file to pin: the cards appear and disappear with the
plugins; a destination that is not configured says so on the card instead of
failing after the export has already run; and the card resolves to a format the
app can actually produce.
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
    """An Export tab against an APP_DATA with NO plugins installed.

    Which is the honest starting point: nothing ships inside the app, so a
    fresh install has no exporters beyond PDF, Markdown and the slim project.
    Tests that need one install it (`_install_ready_destination`)."""
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
    # Leaving it there let it leak into every later test — which is how
    # `forget()` came to exist.
    d.forget("ready-dest")
    d.forget("needy-dest")
    d.reset_for_tests()


P = ExportTab.SEND_PREFIX


def _send_keys(tab):
    return [k for k in tab.format_group.keys() if k.startswith(P)]


def _frame(tab, key):
    return tab.format_group._cards[key].frame


def _texts(card):
    return [w.text() for w in card.findChildren(QLabel)]


def _button(card):
    return card.findChildren(QPushButton)[0]


def test_a_fresh_install_offers_no_exporters(tab):
    """Nothing ships inside the app, so nothing appears until it is installed.

    The defect this replaced: three destinations lived in `aglaia/plugins/` and
    loaded unconditionally, so "Export to Calibre server" was in the Export tab
    of every install whether or not anyone had asked for it."""
    tab.refresh_destinations()
    assert _send_keys(tab) == []
    # …and the built-in formats are untouched.
    assert {"pdf", "markdown", "slim"} <= set(tab.format_group.keys())


def test_an_exporter_is_a_format_card_like_any_other(tab, tmp_path):
    """Same group as PDF and Markdown — that is the whole point."""
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    keys = tab.format_group.keys()
    assert "pdf" in keys and "markdown" in keys
    assert f"{P}ready-dest" in keys


def test_selecting_one_is_selecting_a_format(tab, tmp_path):
    """No separate send button: the ordinary Export button runs it, so the
    destination has to be reachable as the current format."""
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    assert tab.format_group.set_current_key(f"{P}ready-dest")
    assert tab.current_format() == f"{P}ready-dest"


def test_an_unconfigured_destination_says_so_on_the_card(tab, tmp_path):
    """Before it is used, not after the export has already run."""
    _install_needy_destination(tmp_path)
    tab.refresh_destinations()
    card = _frame(tab, f"{P}needy-dest")
    assert any("Not set up yet" in t for t in _texts(card))


def test_the_card_resolves_to_a_format_aglaia_can_produce(tab, tmp_path):
    """A destination that also accepts txt and epub must not offer them:
    Aglaïa writes pdf and md, and nothing else."""
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    assert tab.destination_format("ready-dest") in ("pdf", "md")


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
        # txt and epub are here on purpose: the card must offer only the two
        # formats Aglaïa can actually write.
        "    accepts = ('pdf', 'md', 'txt', 'epub')\n",
        encoding="utf-8")
    d.reset_for_tests()
    return slug


def _install_needy_destination(tmp_path):
    """One with a required setting, so "Not set up yet" is reachable."""
    from aglaia.app_data import plugin_registry as reg
    from aglaia.workers import destinations as d
    slug = "needy-dest"
    p = reg.installed_root("destinations") / slug
    p.mkdir(parents=True, exist_ok=True)
    (p / "aglaia-plugin.toml").write_text(
        '[plugin]\nslug = "needy-dest"\nname = "Needy dest"\n'
        'version = "1.0.0"\nentry = "needy_dest.py"\nlicense = "MIT"\n'
        '[requires]\napi = 1\n[capabilities]\nconfig = true\n',
        encoding="utf-8")
    (p / "needy_dest.py").write_text(
        "from aglaia.plugin_api import (Destination, Field,\n"
        "                               register_destination)\n"
        "@register_destination\n"
        "class N(Destination):\n"
        "    name = 'needy-dest'\n"
        "    display = 'Needy dest'\n"
        "    accepts = ('pdf',)\n"
        "    CONFIG_FIELDS = (Field('base_url', 'Server URL', 'str', '',\n"
        "                           required=True),)\n",
        encoding="utf-8")
    d.reset_for_tests()
    return slug


def test_a_configured_destination_says_nothing_alarming(tab, tmp_path):
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    card = _frame(tab, f"{P}ready-dest")
    assert not any("Not set up yet" in t for t in _texts(card))


def test_the_settings_button_asks_the_host_by_name(tab, tmp_path):
    """The tab knows WHICH destination, not how to reach it."""
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    settings = []
    tab.destination_settings_requested.connect(settings.append)
    _button(_frame(tab, f"{P}ready-dest")).click()
    assert settings == ["ready-dest"]


def test_refreshing_keeps_the_selection(tab, tmp_path):
    """Installing a plugin rebuilds the list; it must not silently move the
    user off the format they had chosen."""
    tab.format_group.set_current_key("markdown")
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    assert tab.current_format() == "markdown"


def _uninstall_ready(tmp_path):
    from aglaia.app_data import plugin_registry as reg
    from aglaia.workers import destinations as d
    reg.uninstall("ready-dest")
    d.forget("ready-dest")
    d.reset_for_tests()


def test_a_removed_destination_loses_its_card(tab, tmp_path):
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    assert f"{P}ready-dest" in _send_keys(tab)
    _uninstall_ready(tmp_path)
    tab.refresh_destinations()
    assert f"{P}ready-dest" not in _send_keys(tab)


def test_removing_the_selected_destination_falls_back_to_pdf(tab, tmp_path):
    """A card that goes away while selected would otherwise leave the Export
    button pointed at nothing."""
    _install_ready_destination(tmp_path)
    tab.refresh_destinations()
    tab.format_group.set_current_key(f"{P}ready-dest")
    _uninstall_ready(tmp_path)
    tab.refresh_destinations()
    assert tab.current_format() == "pdf"


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
    assert f"{P}ready-dest" not in _send_keys(tab)


def test_forget_alone_does_not_survive_the_next_discovery(tab, tmp_path):
    """Because the files are still there. This is why uninstall does both,
    and why `forget` is not offered as an "unload" button."""
    from aglaia.workers import destinations as d
    _install_ready_destination(tmp_path)
    d.load_all()
    d.forget("ready-dest")
    d.reset_for_tests()
    assert "ready-dest" in d.load_all()
