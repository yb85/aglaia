# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The headless secret store governs WRITING, not reading (#166).

`use_plaintext_store()` promises "nothing already in a keychain becomes
unreachable". `PluginSecrets` routed reads through the same helper as writes,
so with the headless store on, a plugin configured in the GUI reported every
secret as missing — `aglaia plugins config` printed "not set" over a stored
key, and `--send-to` refused a configured destination.
"""

from __future__ import annotations

import pytest

from aglaia.app_data import secrets as appsec
from aglaia.app_data.plugin_ctx import PluginConfig, PluginSecrets


class _FakeKeyring:
    """Stands in for the `keyring` module: a dict with its two calls."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, value):
        self.store[(service, username)] = value

    def delete_password(self, service, username):
        self.store.pop((service, username), None)


@pytest.fixture
def plaintext_off():
    before = appsec.plaintext_store()
    appsec.use_plaintext_store(False)
    yield
    appsec.use_plaintext_store(before)


@pytest.fixture
def secrets(tmp_path, monkeypatch, plaintext_off):
    kr = _FakeKeyring()
    cfg = PluginConfig("send-to-corpus", tmp_path / "c.db")
    s = PluginSecrets("send-to-corpus", cfg)
    monkeypatch.setattr(s, "_keyring",
                        lambda for_write=False: (
                            None if (for_write and appsec.plaintext_store())
                            else kr))
    # An empty .env, so the first lookup step finds nothing.
    monkeypatch.setattr(appsec, "_read_env_file", dict)
    monkeypatch.setattr(appsec, "_write_env_file", lambda *_: None)
    return s, kr


def test_a_keychain_secret_stays_readable_in_headless_mode(secrets):
    s, _ = secrets
    s.set("api_key", "k-123")
    assert s.get("api_key") == "k-123"
    appsec.use_plaintext_store(True)          # what every CLI command does
    assert s.get("api_key") == "k-123"


def test_headless_mode_still_writes_away_from_the_keychain(secrets):
    s, kr = secrets
    appsec.use_plaintext_store(True)
    assert s._keyring(for_write=True) is None
    assert s._keyring() is not None
    assert kr.store == {}
