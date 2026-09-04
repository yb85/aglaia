# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""What a plugin is handed instead of the host's internals (#127).

The namespacing is not a security boundary — an in-process plugin can
`import keyring` and read anything (docs/plugin-store.md §1). What these tests
pin is that it is impossible to cross namespaces *through the API*, which is
what makes a reviewer able to say "this plugin only touches its own things"
by reading the imports.
"""
import importlib

import pytest


@pytest.fixture()
def ctxmod(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    importlib.reload(ad)
    importlib.reload(pc)
    return pc


# ── slugs and keys are validated once, at the door ───────────────────

@pytest.mark.parametrize("slug", ["ab", "-lead", "trail-", "UPPER", "has_us",
                                  "has/slash", "has..dots", "a" * 41, "",
                                  "x\x1fy"])
def test_a_bad_slug_is_refused(ctxmod, slug):
    with pytest.raises(ValueError):
        ctxmod.validate_slug(slug)


@pytest.mark.parametrize("slug", ["send-to-kindle", "abc", "a1-b2-c3"])
def test_a_good_slug_passes(ctxmod, slug):
    assert ctxmod.validate_slug(slug) == slug


@pytest.mark.parametrize("key", ["", "a" * 65, "sep\x1fkey", "sp ace",
                                 "/etc/passwd", "-lead"])
def test_a_bad_key_is_refused(ctxmod, key):
    with pytest.raises(ValueError):
        ctxmod.PluginConfig("a-plugin").get(key)


# ── config: one file per plugin ──────────────────────────────────────

def test_settings_round_trip_with_their_type(ctxmod):
    c = ctxmod.PluginConfig("a-plugin")
    c.set("port", 587)
    c.set("tls", True)
    c.set("host", "smtp.example.org")
    c.set("tags", ["a", "b"])
    assert c.get("port") == 587 and isinstance(c.get("port"), int)
    assert c.get("tls") is True
    assert c.get("host") == "smtp.example.org"
    assert c.get("tags") == ["a", "b"]


def test_a_missing_setting_gives_the_default(ctxmod):
    assert ctxmod.PluginConfig("a-plugin").get("nope", 42) == 42


def test_each_plugin_gets_its_own_file(ctxmod):
    a, b = ctxmod.PluginConfig("plugin-a"), ctxmod.PluginConfig("plugin-b")
    a.set("shared", "from-a")
    b.set("shared", "from-b")
    assert a.get("shared") == "from-a"
    assert b.get("shared") == "from-b"
    assert a.path != b.path


def test_delete_removes_it(ctxmod):
    c = ctxmod.PluginConfig("a-plugin")
    c.set("k", 1)
    c.delete("k")
    assert c.get("k") is None and "k" not in c.all()


# ── secrets: bound to the slug by the host ───────────────────────────

def _fake_keyring(monkeypatch, store):
    """A keyring that records exactly what service/username it was given."""
    import keyring
    monkeypatch.setattr(keyring, "get_keyring", lambda: object())
    monkeypatch.setattr(keyring, "set_password",
                        lambda s, u, v: store.__setitem__((s, u), v))
    monkeypatch.setattr(keyring, "get_password",
                        lambda s, u: store.get((s, u)))
    monkeypatch.setattr(keyring, "delete_password",
                        lambda s, u: store.pop((s, u), None))
    return store


def test_secrets_round_trip(ctxmod, monkeypatch):
    pytest.importorskip("keyring")
    _fake_keyring(monkeypatch, {})
    s = ctxmod.PluginSecrets("a-plugin")
    s.set("api_key", "sk-123")
    assert s.get("api_key") == "sk-123"
    s.delete("api_key")
    assert s.get("api_key") is None


def test_the_username_carries_the_slug_namespace(ctxmod, monkeypatch):
    pytest.importorskip("keyring")
    store = _fake_keyring(monkeypatch, {})
    ctxmod.PluginSecrets("send-to-kindle").set("password", "hunter2")
    (service, username), value = next(iter(store.items()))
    assert service == ctxmod.SECRET_SERVICE
    assert username == f"send-to-kindle{ctxmod.NS_SEP}password"
    assert value == "hunter2"


def test_two_plugins_cannot_read_each_other(ctxmod, monkeypatch):
    pytest.importorskip("keyring")
    _fake_keyring(monkeypatch, {})
    a, b = ctxmod.PluginSecrets("plugin-a"), ctxmod.PluginSecrets("plugin-b")
    a.set("token", "secret-a")
    b.set("token", "secret-b")
    assert a.get("token") == "secret-a"
    assert b.get("token") == "secret-b"


def test_a_key_cannot_escape_into_another_namespace(ctxmod, monkeypatch):
    """The separator is the whole defence: if a key could contain it, a
    plugin could name its way into a neighbour's entry."""
    pytest.importorskip("keyring")
    _fake_keyring(monkeypatch, {})
    s = ctxmod.PluginSecrets("plugin-a")
    with pytest.raises(ValueError):
        s.get(f"plugin-b{ctxmod.NS_SEP}token")


def test_keys_lists_names_never_values(ctxmod, monkeypatch):
    pytest.importorskip("keyring")
    _fake_keyring(monkeypatch, {})
    s = ctxmod.PluginSecrets("a-plugin")
    s.set("user", "yann")
    s.set("password", "hunter2")
    assert s.keys() == ["password", "user"]
    assert "hunter2" not in str(s.keys())


def test_purge_empties_the_namespace(ctxmod, monkeypatch):
    pytest.importorskip("keyring")
    _fake_keyring(monkeypatch, {})
    s = ctxmod.PluginSecrets("a-plugin")
    s.set("a", "1")
    s.set("b", "2")
    s.purge()
    assert s.keys() == [] and s.get("a") is None


# ── no keychain: fall back, but say so ───────────────────────────────

def test_without_a_keychain_it_still_works_and_admits_it(ctxmod, monkeypatch):
    monkeypatch.setattr(ctxmod.PluginSecrets, "_keyring", lambda self: None)
    s = ctxmod.PluginSecrets("a-plugin")
    assert s.available is False
    s.set("api_key", "sk-plain")
    assert s.get("api_key") == "sk-plain"
    assert s.keys() == ["api_key"]


def test_the_plaintext_copy_is_dropped_once_a_keychain_takes_it(
        ctxmod, monkeypatch):
    """A stale plaintext copy is exactly what a keychain was meant to avoid."""
    pytest.importorskip("keyring")
    monkeypatch.setattr(ctxmod.PluginSecrets, "_keyring", lambda self: None)
    cfg = ctxmod.PluginConfig("a-plugin")
    s = ctxmod.PluginSecrets("a-plugin", cfg)
    s.set("api_key", "sk-plain")
    assert cfg._raw_get("__secret__.api_key") == "sk-plain"
    _fake_keyring(monkeypatch, {})
    import keyring
    monkeypatch.setattr(ctxmod.PluginSecrets, "_keyring",
                        lambda self: keyring)
    s.set("api_key", "sk-kept")
    assert cfg._raw_get("__secret__.api_key") is None
    assert s.get("api_key") == "sk-kept"


# ── the context the host builds ──────────────────────────────────────

def test_a_plugin_that_did_not_ask_gets_no_secrets_object(ctxmod):
    assert ctxmod.build_context("a-plugin").secrets is None
    assert ctxmod.build_context("a-plugin", wants_secrets=True).secrets


def test_the_data_dir_is_the_plugins_own(ctxmod):
    ctx = ctxmod.build_context("a-plugin")
    assert ctx.data_dir.is_dir()
    assert ctx.data_dir.parent.name == "a-plugin"
    other = ctxmod.build_context("plugin-b")
    assert other.data_dir != ctx.data_dir


def test_the_context_refuses_a_bad_slug(ctxmod):
    with pytest.raises(ValueError):
        ctxmod.build_context("../escape")
