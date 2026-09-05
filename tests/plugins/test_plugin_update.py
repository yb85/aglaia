# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Updating a plugin when the registry has a newer version.

The rule that matters: an update is NOT uninstall-then-install. Uninstall
deletes the plugin's data directory, its settings and its secrets — which for
the stamp remover is every hand-traced stamp and for an export plugin is the
stored SMTP password. Nobody expects an update to cost them that, and the
person it bit first was the author, who reinstalled StampRemover and lost the
library.
"""
import importlib
import zipfile
from pathlib import Path

import pytest

MANIFEST = ('[plugin]\nslug = "up-probe"\nname = "Up probe"\n'
            'version = "{v}"\nentry = "up_probe.py"\nlicense = "MIT"\n'
            'author = "A <a@e.org>"\n[requires]\napi = 1\n[capabilities]\n{caps}')

SOURCE = ('from aglaia.processors.abstraction import AbstractImageProcessor\n\n\n'
          'class UpProbe(AbstractImageProcessor):\n'
          '    SUMMARY = "v{v}"\n    OPTIONS = {{}}\n\n'
          '    def process(self, image_buffer):\n        return image_buffer\n')


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugins as pl
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as r
    for m in (ad, pl, pc, r):
        importlib.reload(m)
    return r


def _archive(tmp_path, version, caps=""):
    d = tmp_path / f"src{version}" / "up-probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / "aglaia-plugin.toml").write_text(
        MANIFEST.format(v=version, caps=caps), encoding="utf-8")
    (d / "up_probe.py").write_text(SOURCE.format(v=version), encoding="utf-8")
    z = tmp_path / f"up-probe-{version}.aglplugin"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                zf.write(f, f"up-probe/{f.relative_to(d)}")
    return z


def _entry(reg, version, caps=None):
    return reg.RegistryEntry(slug="up-probe", kind="processors",
                             name="Up probe", version=version,
                             capabilities=caps or {})


class TestIsThereAnUpdate:
    @pytest.mark.parametrize("installed,offered,expected", [
        ("1.0.0", "1.1.0", True),
        ("1.0.0", "1.0.1", True),
        ("1.9.0", "1.10.0", True),      # not a string comparison
        ("1.0.0", "1.0.0", False),      # same version is not an update
        ("1.1.0", "1.0.0", False),      # a registry that went backwards
        ("2.0.0", "1.99.99", False),
    ])
    def test_only_a_strictly_newer_version_counts(self, reg, tmp_path,
                                                  installed, offered, expected):
        """Re-offering the same version forever is how an update badge becomes
        wallpaper; offering a downgrade is worse."""
        reg.install_from_archive(_archive(tmp_path, installed), "processors")
        assert reg.update_available("up-probe", offered) is expected

    def test_a_plugin_that_is_not_installed_has_no_update(self, reg):
        assert reg.update_available("up-probe", "9.9.9") is False

    def test_an_unparseable_version_does_not_raise(self, reg, tmp_path):
        """One odd version string must not break the check for every other
        plugin in the list."""
        reg.install_from_archive(_archive(tmp_path, "1.0.0"), "processors")
        assert reg.update_available("up-probe", "banana") is False


class TestUpdatingKeepsWhatThePluginOwns:
    def test_the_data_directory_survives(self, reg, tmp_path, monkeypatch):
        """The stamp library lives here. This is the whole reason update is
        not uninstall-then-install."""
        reg.install_from_archive(_archive(tmp_path, "1.0.0"), "processors")
        from aglaia.app_data.plugin_ctx import build_context
        ctx = build_context("up-probe", wants_secrets=False)
        keep = ctx.data_dir / "library" / "traced.txt"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_text("hand-traced", encoding="utf-8")

        monkeypatch.setattr(reg, "install_from_registry",
                            lambda e, **k: reg.install_from_archive(
                                _archive(tmp_path, "1.1.0"), "processors"))
        res = reg.update_from_registry(_entry(reg, "1.1.0"))
        assert res.ok, res.message
        assert keep.read_text(encoding="utf-8") == "hand-traced"

    def test_the_settings_survive(self, reg, tmp_path, monkeypatch):
        reg.install_from_archive(_archive(tmp_path, "1.0.0"), "processors")
        from aglaia.app_data.plugin_ctx import build_context
        build_context("up-probe", wants_secrets=False).config.set("host", "x")

        monkeypatch.setattr(reg, "install_from_registry",
                            lambda e, **k: reg.install_from_archive(
                                _archive(tmp_path, "1.1.0"), "processors"))
        assert reg.update_from_registry(_entry(reg, "1.1.0")).ok
        assert build_context("up-probe",
                             wants_secrets=False).config.get("host") == "x"

    def test_the_code_is_actually_replaced(self, reg, tmp_path, monkeypatch):
        reg.install_from_archive(_archive(tmp_path, "1.0.0"), "processors")
        monkeypatch.setattr(reg, "install_from_registry",
                            lambda e, **k: reg.install_from_archive(
                                _archive(tmp_path, "1.1.0"), "processors"))
        reg.update_from_registry(_entry(reg, "1.1.0"))
        assert reg.installed_record("up-probe")["version"] == "1.1.0"
        entry = (reg.installed_root("processors") / "up-probe" / "up_probe.py")
        assert 'SUMMARY = "v1.1.0"' in entry.read_text(encoding="utf-8")


def test_a_version_asking_for_more_is_refused(reg, tmp_path):
    """"I trusted it when it only read files" is not consent to a version
    that has learned to use the network. The update is refused and says so;
    installing it deliberately is still possible."""
    reg.install_from_archive(_archive(tmp_path, "1.0.0"), "processors")
    res = reg.update_from_registry(
        _entry(reg, "2.0.0", caps={"network": True, "secrets": True}))
    assert res.ok is False
    assert "network" in res.message and "secrets" in res.message
    # …and nothing was touched.
    assert reg.installed_record("up-probe")["version"] == "1.0.0"


def test_the_same_capabilities_are_not_treated_as_new(reg, tmp_path):
    reg.install_from_archive(
        _archive(tmp_path, "1.0.0", caps="config = true\n"), "processors")
    res = reg.update_from_registry(
        _entry(reg, "1.1.0", caps={"config": True}))
    # It gets as far as trying to download, which is past the capability gate.
    assert "asks for more" not in res.message
