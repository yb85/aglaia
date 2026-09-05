# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Manifests, the import scan, and installing (#128-#130).

The scan is a lint, not a sandbox — it reads `import` statements and cannot see
a runtime-built one (`docs/plugin-store.md` §1). What it IS good for is pinned
here: `keyring`, `sqlite3` and `os` are refused outright, so "this plugin went
around the context API" stops being something a reviewer has to notice and
becomes something the tooling says out loud.
"""
import hashlib
import importlib
import json
import zipfile

import pytest

from aglaia.app_data import plugin_manifest as pm

GOOD = """\
[plugin]
slug = "a-plugin"
name = "A plugin"
version = "1.0.0"
summary = "Does a thing."
author = "Jane Doe <jane@example.org>"
license = "MIT"
entry = "a_plugin.py"

[requires]
aglaia = ">=0.1.0rc5,<0.2"
python = ">=3.12"
api = 1
imports = []

[capabilities]
config = true
secrets = false
network = false
files = false
"""


def _write(tmp_path, toml=GOOD, code="from aglaia.plugin_api import Destination\n",
           slug="a-plugin", entry="a_plugin.py"):
    d = tmp_path / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "aglaia-plugin.toml").write_text(toml, encoding="utf-8")
    (d / entry).write_text(code, encoding="utf-8")
    return d


# ── manifests ────────────────────────────────────────────────────────

def test_a_good_manifest_parses(tmp_path):
    d = _write(tmp_path)
    man = pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations",
                            expect_slug="a-plugin")
    assert man.slug == "a-plugin" and man.version == "1.0.0"
    assert man.config is True and man.secrets is False


def test_the_directory_decides_the_slug(tmp_path):
    """The slug names the keychain namespace and the settings file. A manifest
    that disagrees with its own directory is ambiguous about both."""
    d = _write(tmp_path, slug="somewhere-else")
    with pytest.raises(pm.ManifestError, match="directory"):
        pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations",
                          expect_slug="somewhere-else")


@pytest.mark.parametrize("bad,msg", [
    ('slug = "a-plugin"', "slug"),                # removed below
    ('version = "1.0.0"', "version"),
    ('entry = "a_plugin.py"', "entry"),
])
def test_a_manifest_missing_a_required_field_says_which(tmp_path, bad, msg):
    d = _write(tmp_path, toml=GOOD.replace(bad + "\n", ""))
    with pytest.raises(pm.ManifestError, match=msg):
        pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations")


def test_an_entry_that_is_a_path_is_refused(tmp_path):
    d = _write(tmp_path, toml=GOOD.replace('entry = "a_plugin.py"',
                                           'entry = "../../evil.py"'))
    with pytest.raises(pm.ManifestError, match="single .py"):
        pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations")


def test_a_dependency_the_host_does_not_ship_is_refused(tmp_path):
    """No plugin ever installs a dependency — which is why this list can be
    closed rather than open."""
    d = _write(tmp_path, toml=GOOD.replace("imports = []",
                                           'imports = ["requests"]'))
    with pytest.raises(pm.ManifestError, match="ever installs a dependency"):
        pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations")


def test_a_network_library_needs_the_network_capability(tmp_path):
    d = _write(tmp_path, toml=GOOD.replace("imports = []",
                                           'imports = ["httpx"]'))
    with pytest.raises(pm.ManifestError, match="network = true"):
        pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations")


def test_a_foreign_api_major_is_refused_by_name(tmp_path):
    d = _write(tmp_path, toml=GOOD.replace("api = 1", "api = 99"))
    man = pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations")
    ok, why = pm.api_compatible(man)
    assert ok is False and "99" in why and str(pm.API_VERSION) in why


# ── the import scan ──────────────────────────────────────────────────

@pytest.mark.parametrize("src", [
    "import os", "import sys", "import subprocess", "import socket",
    "import keyring", "import sqlite3", "import shutil", "import ctypes",
    "import pickle", "import importlib",
])
def test_the_modules_the_context_replaces_are_refused(src):
    assert pm.scan_source(src).refused


def test_only_the_facade_is_reachable_under_aglaia():
    assert pm.scan_source("from aglaia.plugin_api import Destination").clean
    assert pm.scan_source("from aglaia.workers.ocr import x").refused
    assert pm.scan_source("import aglaia.storage.db").refused


def test_urllib_parse_is_allowed_but_urllib_request_is_not():
    """Building a URL is string work; fetching one is not."""
    assert pm.scan_source("from urllib.parse import quote").clean
    assert pm.scan_source("import urllib.request").refused


def test_future_annotations_do_not_trip_the_scan():
    assert pm.scan_source("from __future__ import annotations").clean


def test_a_network_stdlib_module_needs_the_capability(tmp_path):
    d = _write(tmp_path)
    man = pm.parse_manifest(d / "aglaia-plugin.toml", kind="destinations")
    assert "smtplib" in " ".join(pm.scan_source("import smtplib", man).undeclared)
    man.network = True
    assert pm.scan_source("import smtplib", man).clean


def test_an_undeclared_shipped_library_is_reported_not_refused():
    """The dialog shows what a plugin does beyond what it admitted to."""
    r = pm.scan_source("import numpy")
    assert r.undeclared == ["numpy"] and not r.refused


def test_re_compile_is_not_flagged_for_review():
    """An attribute call is ordinary. Flagging it would fire on nearly every
    plugin and teach a reviewer to skim past the flag."""
    assert pm.scan_source("import re\nx = re.compile('a')").review == []


def test_a_bare_dangerous_builtin_is_flagged():
    assert "eval" in pm.scan_source("eval('2+2')").review
    assert "__import__" in pm.scan_source("__import__('os')").review
    assert "getattr" in pm.scan_source("getattr(x, 'y')").review


def test_a_module_that_will_not_parse_says_so():
    assert "will not parse" in pm.scan_source("def (:").error


# ── installing ───────────────────────────────────────────────────────

@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "appdata"))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as r
    importlib.reload(ad)
    importlib.reload(pc)
    importlib.reload(r)
    return r


def _archive(tmp_path, name="p.aglplugin", extra=None, toml=GOOD,
             code="from aglaia.plugin_api import Destination\n",
             root="a-plugin", entry="a_plugin.py"):
    z = tmp_path / name
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(f"{root}/aglaia-plugin.toml", toml)
        zf.writestr(f"{root}/{entry}", code)
        for rel, data in (extra or {}).items():
            zf.writestr(rel, data)
    return z


def test_an_archive_stages_and_scans(reg, tmp_path):
    man, files, scan, err = reg.stage_archive(_archive(tmp_path))
    assert err == "" and man.slug == "a-plugin"
    assert scan.clean


def test_a_path_escape_is_refused_before_extraction(reg, tmp_path):
    z = _archive(tmp_path, extra={"../evil.py": "x = 1"})
    _man, _f, _s, err = reg.stage_archive(z)
    assert "refused" in err


def test_a_compiled_extension_is_refused(reg, tmp_path):
    """It cannot be reviewed by reading it, which is exactly what this system
    cannot honestly accept."""
    z = _archive(tmp_path, extra={"a-plugin/fast.so": b"\x7fELF"})
    _m, _f, _s, err = reg.stage_archive(z)
    assert "compiled" in err


def test_two_top_level_modules_are_refused(reg, tmp_path):
    z = _archive(tmp_path, extra={"a-plugin/second.py": "x = 1"})
    _m, _f, _s, err = reg.stage_archive(z)
    assert "ONE top-level module" in err


def test_an_entry_not_in_the_archive_is_refused(reg, tmp_path):
    z = _archive(tmp_path, entry="different.py")
    _m, _f, _s, err = reg.stage_archive(z)
    assert "not in the archive" in err


def test_installing_from_an_archive_writes_it(reg, tmp_path):
    res = reg.install_from_archive(_archive(tmp_path), "destinations")
    assert res.ok, res.message
    assert (res.path / "a_plugin.py").is_file()
    assert reg.installed_record("a-plugin")["source"] == "zip"


def test_a_registry_download_that_does_not_match_its_hash_is_refused(
        reg, tmp_path, monkeypatch):
    """This is what the hashes are for."""
    entry = reg.RegistryEntry(
        slug="a-plugin", kind="destinations", name="A", version="1.0.0",
        files={"aglaia-plugin.toml": "sha256:" + "0" * 64})

    class _R:
        status_code = 200
        content = b"not what the index promised"

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, **kw): return _R()

    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **kw: _C())
    res = reg.install_from_registry(entry)
    assert res.ok is False and "hash" in res.message


def test_a_registry_install_records_where_it_came_from(reg, tmp_path,
                                                       monkeypatch):
    payload = {"aglaia-plugin.toml": GOOD.encode(),
               "a_plugin.py": b"from aglaia.plugin_api import Destination\n"}
    files = {k: "sha256:" + hashlib.sha256(v).hexdigest()
             for k, v in payload.items()}
    entry = reg.RegistryEntry(slug="a-plugin", kind="destinations", name="A",
                              version="1.0.0", files=files)

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, **kw):
            rel = url.rsplit("/", 1)[-1]
            return type("R", (), {"status_code": 200,
                                  "content": payload[rel]})()

    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **kw: _C())
    res = reg.install_from_registry(entry)
    assert res.ok, res.message
    assert reg.installed_record("a-plugin")["source"] == "registry"


def test_uninstall_takes_the_settings_and_the_secrets_too(reg, tmp_path,
                                                          monkeypatch):
    import aglaia.app_data.plugin_ctx as pc
    monkeypatch.setattr(pc.PluginSecrets, "_keyring", lambda self: None)
    reg.install_from_archive(_archive(tmp_path), "destinations")
    ctx = pc.build_context("a-plugin", wants_secrets=True)
    ctx.config.set("k", "v")
    ctx.secrets.set("token", "s3cret")
    data_dir = pc.plugin_data_dir("a-plugin")
    assert data_dir.is_dir()
    res = reg.uninstall("a-plugin")
    assert res.ok
    assert not (reg.installed_root("destinations") / "a-plugin").exists()
    assert not data_dir.exists()
    assert reg.installed_record("a-plugin") == {}


def test_disable_is_remembered(reg, tmp_path):
    reg.install_from_archive(_archive(tmp_path), "destinations")
    assert reg.is_disabled("a-plugin") is False
    reg.set_disabled("a-plugin", True)
    assert reg.is_disabled("a-plugin") is True


def test_list_installed_reports_a_broken_manifest_rather_than_hiding_it(
        reg, tmp_path):
    d = reg.installed_root("destinations") / "a-plugin"
    d.mkdir(parents=True)
    (d / "aglaia-plugin.toml").write_text("not toml {{{", encoding="utf-8")
    items = reg.list_installed()
    assert items and items[0]["error"]


def test_the_index_falls_back_to_its_cache_when_the_network_is_gone(
        reg, monkeypatch):
    cache = reg._cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"plugins": [
        {"slug": "cached-one", "kind": "destinations", "name": "Cached",
         "version": "1.0.0"}]}), encoding="utf-8")

    import httpx

    def _boom(**kw):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(httpx, "Client", _boom)
    idx = reg.fetch_index()
    assert [e.slug for e in idx.entries] == ["cached-one"]
    assert "last copy" in idx.error


def test_a_settings_connection_does_not_outlive_its_use(tmp_path):
    """`with sqlite3.connect(...)` commits; it does not close.

    Every get and set therefore leaked a handle onto the plugin's own
    settings file. POSIX unlinks an open file happily, so the leak was
    invisible here and removed a plugin's settings on Windows only in theory
    — the real uninstall left them on disk."""
    import sqlite3

    import aglaia.app_data.plugin_ctx as pc
    cfg = pc.PluginConfig("a-plugin", path=tmp_path / "config.db")
    with cfg._connect() as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_uninstall_says_so_when_the_files_survive(reg, tmp_path, monkeypatch):
    """A removal that could not remove must not answer "removed".

    `shutil.rmtree(..., ignore_errors=True)` is what lets one locked file not
    abort the rest; on its own it also let uninstall report success over a
    directory that is still there."""
    import shutil

    import aglaia.app_data.plugin_ctx as pc
    monkeypatch.setattr(pc.PluginSecrets, "_keyring", lambda self: None)
    reg.install_from_archive(_archive(tmp_path), "destinations")
    pc.build_context("a-plugin", wants_secrets=True).config.set("k", "v")
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: None)
    res = reg.uninstall("a-plugin")
    assert not res.ok
    assert "still in use" in res.message
