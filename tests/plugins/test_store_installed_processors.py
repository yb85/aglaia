# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A plugin installed from the store must actually be usable (#140).

A processor gets to `<APP_DATA>/plugins/processors/` two ways, and until now
only one of them was looked at. `glob("*.py")` finds a file dropped in by hand;
it does not descend, so a plugin installed from the store — which lands in
`processors/<slug>/` with a manifest — was invisible to the processor registry.

The symptom had no error in it, which is why it survived: StampRemover
installed, reported success, appeared in the Plugins tab as installed, and
simply never showed up in the pipeline editor's element list.

The second half is consent. It was given in the install dialog; asking again on
the next launch asks about a decision the user has just made, and a warning
that fires when nothing is wrong is one people learn to dismiss. So an install
writes to the same acceptance ledger the drop-in gate reads — not around it,
so a file that changes on disk afterwards still reverts to pending.
"""
import importlib
import zipfile
from pathlib import Path

import pytest

MANIFEST = ('[plugin]\nslug = "{slug}"\nname = "Demo"\nversion = "1.0.0"\n'
            'entry = "{entry}"\nlicense = "MIT"\nauthor = "A <a@e.org>"\n'
            '[requires]\napi = 1\n[capabilities]\n')

SOURCE = '''
from aglaia.processors.abstraction import AbstractImageProcessor


class {cls}(AbstractImageProcessor):
    SUMMARY = "a demo processor"
    OPTIONS = {{}}

    def process(self, image_buffer):
        return image_buffer
'''


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A clean APP_DATA plus the store's own installer."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugins as pl
    import aglaia.app_data.plugin_registry as reg
    for m in (ad, pl, reg):
        importlib.reload(m)
    return reg, pl, tmp_path


def _archive(tmp_path, slug, cls, entry="demo_proc.py"):
    d = tmp_path / "src" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "aglaia-plugin.toml").write_text(
        MANIFEST.format(slug=slug, entry=entry), encoding="utf-8")
    (d / entry).write_text(SOURCE.format(cls=cls), encoding="utf-8")
    z = tmp_path / f"{slug}.aglplugin"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                zf.write(f, f"{slug}/{f.relative_to(d)}")
    return z


def _names(reg, pl):
    """Processor names visible after a fresh discovery pass."""
    from aglaia.processors import registry as R
    importlib.reload(R)
    return set(R.all_processors())


def test_a_store_installed_processor_reaches_the_element_list(store, tmp_path):
    """The bug, exactly: installed, reported installed, and invisible."""
    reg, pl, _ = store
    before = _names(reg, pl)
    assert "DemoProc" not in before

    res = reg.install_from_archive(
        _archive(tmp_path, "demo-proc", "DemoProc"), "processors")
    assert res.ok, res.message
    assert "DemoProc" in _names(reg, pl)


def test_installing_is_the_consent_no_second_prompt(store, tmp_path):
    """The trust gate must not re-ask about a decision just made — a warning
    that fires when nothing is wrong is one people learn to click through."""
    reg, pl, _ = store
    reg.install_from_archive(
        _archive(tmp_path, "demo-proc", "DemoProc"), "processors")
    assert pl.scan_pending() == []


def test_editing_an_installed_plugin_reverts_it_to_pending(store, tmp_path):
    """Consent is recorded in the ledger, not bypassed. So the sha still
    guards it: code that changes on disk after install has never been
    acknowledged, whoever changed it."""
    reg, pl, app_data = store
    reg.install_from_archive(
        _archive(tmp_path, "demo-proc", "DemoProc"), "processors")
    entry = app_data / "plugins" / "processors" / "demo-proc" / "demo_proc.py"
    entry.write_text(entry.read_text() + "\n# tampered\n", encoding="utf-8")
    pending = pl.scan_pending()
    assert [c.reason for c in pending] == ["changed"]
    assert pending[0].path == entry.resolve()


def test_uninstalling_drops_the_consent(store, tmp_path):
    """Otherwise a later plugin installed at the same path inherits consent
    the user gave to a different one."""
    reg, pl, _ = store
    reg.install_from_archive(
        _archive(tmp_path, "demo-proc", "DemoProc"), "processors")
    assert pl.scan_pending() == []
    reg.uninstall("demo-proc")

    # A DIFFERENT plugin, same slug and same path.
    reg.install_from_archive(
        _archive(tmp_path, "demo-proc", "Impostor"), "processors")
    entry = pl.plugins_dir("processors") / "demo-proc" / "demo_proc.py"
    assert "Impostor" in entry.read_text()
    # It was consented to on ITS install, not on the previous one's.
    assert pl.scan_pending() == []


def test_a_hand_dropped_file_still_works(store, tmp_path):
    """The older shape did not go away — and it still faces the gate, because
    nobody consented to a file that simply appeared."""
    reg, pl, _ = store
    d = pl.plugins_dir("processors")
    d.mkdir(parents=True, exist_ok=True)
    (d / "dropped_proc.py").write_text(SOURCE.format(cls="DroppedProc"),
                                       encoding="utf-8")
    pending = pl.scan_pending()
    assert [c.path.name for c in pending] == ["dropped_proc.py"]
    assert pending[0].reason == "new"


def test_a_directory_without_a_manifest_is_not_a_plugin(store, tmp_path):
    """Support files, caches and half-extracted archives live down there too.
    The manifest is what makes a directory a plugin."""
    reg, pl, _ = store
    d = pl.plugins_dir("processors") / "just-a-folder"
    d.mkdir(parents=True, exist_ok=True)
    (d / "helper.py").write_text("x = 1\n", encoding="utf-8")
    assert pl.scan_pending() == []


def test_a_manifest_cannot_point_outside_its_own_directory(store, tmp_path):
    """`entry` names one module beside the manifest. A path that escapes would
    make an arbitrary file on disk into an accepted plugin."""
    reg, pl, _ = store
    d = pl.plugins_dir("processors") / "escapee"
    d.mkdir(parents=True, exist_ok=True)
    (d / "aglaia-plugin.toml").write_text(
        MANIFEST.format(slug="escapee", entry="../../../evil.py"),
        encoding="utf-8")
    assert pl._manifest_entry(d) is None
    assert pl.scan_pending() == []
