# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""`aglaia plugins …` — the Plugins tab from a terminal.

Everything goes through the same functions the GUI uses, so a plugin set up on
a headless box and one set up in the window are the same plugin. The tests
install a stub export plugin from a local archive and drive every subcommand
through typer's runner; the interactive view is exercised through its
non-interactive twin, `--set`.
"""
import importlib
import zipfile

import pytest
from typer.testing import CliRunner

MANIFEST = ('[plugin]\nslug = "demo-dest"\nname = "Demo destination"\n'
            'version = "1.0.0"\nentry = "demo_dest.py"\nlicense = "MIT"\n'
            'author = "A <a@e.org>"\n[requires]\napi = 1\n'
            '[capabilities]\nconfig = true\nsecrets = true\n')
SOURCE = '''
from aglaia.plugin_api import (CheckResult, Destination, Field, SendResult,
                               register_destination)


@register_destination
class Demo(Destination):
    name = "demo-dest"
    display = "Demo destination"
    accepts = ("pdf",)
    CONFIG_FIELDS = (
        Field("host", "Server", "str", "", required=True),
        Field("port", "Port", "int", 587),
        Field("mode", "Mode", "choice", "a", choices=("a", "b")),
        Field("dry", "Dry run", "bool", False),
    )
    SECRET_FIELDS = (Field("token", "Token", "secret", "", required=True),)

    def check(self):
        return CheckResult(True, f"ok {self.conf('host')}:{self.conf('port')}")

    def send(self, path, meta):
        return SendResult(True, "sent")
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "appdata"))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as reg
    for m in (ad, pc, reg):
        importlib.reload(m)
    monkeypatch.setattr(pc.PluginSecrets, "_keyring", lambda self: None)
    from aglaia.workers import destinations as d
    d.reset_for_tests()
    # no network: the registry index is empty
    monkeypatch.setattr(reg, "fetch_index", lambda **k: reg.IndexResult())
    d_ = tmp_path / "src" / "demo-dest"; d_.mkdir(parents=True)
    (d_ / "aglaia-plugin.toml").write_text(MANIFEST, encoding="utf-8")
    (d_ / "demo_dest.py").write_text(SOURCE, encoding="utf-8")
    z = tmp_path / "demo-dest.aglplugin"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(d_.rglob("*")):
            if f.is_file():
                zf.write(f, f"demo-dest/{f.relative_to(d_)}")
    from aglaia.cli import app
    yield CliRunner(), app, z
    d.forget("demo-dest"); d.reset_for_tests()


def test_plugins_is_a_real_subcommand_not_a_project_path(env):
    """`aglaia plugins` used to fall through to `gui plugins` — the default
    command guard did not know the word."""
    runner, app, _ = env
    r = runner.invoke(app, ["plugins", "--help"])
    assert r.exit_code == 0 and "install" in r.output and "config" in r.output


def test_an_archive_needs_trust(env):
    runner, app, z = env
    r = runner.invoke(app, ["plugins", "install", str(z)])
    assert r.exit_code == 2
    assert "nobody has reviewed it" in r.output.lower() or "--trust" in r.output


def test_install_list_config_toggle_remove(env):
    runner, app, z = env
    r = runner.invoke(app, ["plugins", "install", str(z), "--trust"])
    assert r.exit_code == 0, r.output
    assert "installed" in r.output.lower()
    assert "still needs" in r.output and "aglaia plugins config demo-dest" in r.output

    r = runner.invoke(app, ["plugins", "list"])
    assert r.exit_code == 0 and "Demo destination" in r.output
    assert "Server" in r.output and "Token" in r.output      # what it needs

    # scripted config, with type coercion
    r = runner.invoke(app, ["plugins", "config", "demo-dest",
                            "--set", "host=books.example.org", "--set", "port=465",
                            "--set", "mode=b", "--set", "dry=yes", "--set", "token=s3cret",
                            "--test"])
    assert r.exit_code == 0, r.output
    assert "ok books.example.org:465" in r.output
    from aglaia.workers import destinations as d
    d.reset_for_tests()
    dest = d.load_all()["demo-dest"]
    assert dest.conf("port") == 465 and dest.conf("dry") is True and dest.conf("mode") == "b"
    assert dest.secret("token") == "s3cret"
    assert dest.missing_settings() == []

    r = runner.invoke(app, ["plugins", "list"])
    assert "ready" in r.output

    # a wrong key and a wrong type are refused with the fix named
    r = runner.invoke(app, ["plugins", "config", "demo-dest", "--set", "nope=1"])
    assert r.exit_code == 2 and "has no setting" in r.output
    r = runner.invoke(app, ["plugins", "config", "demo-dest", "--set", "port=lots"])
    assert r.exit_code == 2

    r = runner.invoke(app, ["plugins", "toggle", "demo-dest"])
    assert r.exit_code == 0 and "disabled" in r.output
    r = runner.invoke(app, ["plugins", "toggle", "demo-dest"])
    assert "enabled" in r.output

    r = runner.invoke(app, ["plugins", "remove", "demo-dest"])   # no tty, no --yes
    assert r.exit_code == 2 and "--yes" in r.output
    r = runner.invoke(app, ["plugins", "remove", "demo-dest", "--yes"])
    assert r.exit_code == 0 and "removed" in r.output.lower()
    r = runner.invoke(app, ["plugins", "list"])
    assert "No plugins installed" in r.output


def test_unknown_plugin_messages(env):
    runner, app, _ = env
    assert runner.invoke(app, ["plugins", "toggle", "ghost"]).exit_code == 1
    assert runner.invoke(app, ["plugins", "config", "ghost"]).exit_code == 1
    r = runner.invoke(app, ["plugins", "install", "ghost"])
    assert r.exit_code == 1 and "aglaia plugins search" in r.output
