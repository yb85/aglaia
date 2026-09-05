# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Install the first-party plugins the way a user does.

No plugin ships inside the app any more. Three used to, and they loaded
unconditionally — "Export to Calibre server" appeared in every install whether
or not anyone had asked for it. Code that ships in the application and always
runs is a feature; a plugin is something the user chose.

So the tests install them, through `install_from_archive` — the same function
the "Install from file…" button calls, with the same manifest parsing, the same
import scan and the same on-disk layout. A test that copied the directories
into place instead would pass on a build where installation itself is broken.

The plugin sources live in the registry repo. `AGLAIA_PLUGINS_REPO` points at a
checkout; without one these tests skip, because a green run that silently
tested nothing is worse than a skip that says so.
"""
import os
import zipfile
from pathlib import Path

import pytest

FIRST_PARTY = ("send-to-calibre", "send-to-kindle", "send-to-corpus")


def _repo() -> Path | None:
    """A checkout of github.com/yb85/aglaia-plugins, if there is one."""
    env = os.environ.get("AGLAIA_PLUGINS_REPO")
    candidates = [Path(env)] if env else []
    candidates += [Path("/tmp/aglaia-plugins"),
                   Path.home() / "Documents/projects/aglaia-plugins",
                   Path(__file__).resolve().parents[2].parent / "aglaia-plugins"]
    for c in candidates:
        if (c / "destinations").is_dir():
            return c
    return None


def _zip_plugin(src: Path, out: Path) -> Path:
    """Pack one plugin directory the way a submission would be packed."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f"{src.name}/{f.relative_to(src)}")
    return out


@pytest.fixture()
def dests(tmp_path, monkeypatch):
    """A fresh APP_DATA with the three first-party destinations installed."""
    repo = _repo()
    if repo is None:
        pytest.skip("no aglaia-plugins checkout; set AGLAIA_PLUGINS_REPO")

    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import importlib
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as reg
    for m in (ad, pc, reg):
        importlib.reload(m)

    from aglaia.workers import destinations as D
    D.reset_for_tests()
    for slug in FIRST_PARTY:
        src = repo / "destinations" / slug
        assert src.is_dir(), f"{src} missing from the plugins repo"
        res = reg.install_from_archive(
            _zip_plugin(src, tmp_path / f"{slug}.aglplugin"), "destinations")
        assert res.ok, f"{slug} would not install: {res.message}"
    # Secrets go to a dict, not the developer's real keychain.
    monkeypatch.setattr(pc.PluginSecrets, "_keyring", lambda self: None)
    D.reset_for_tests()
    yield D
    for slug in FIRST_PARTY:
        D.forget(slug)
    D.reset_for_tests()
