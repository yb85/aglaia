# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Carrying one plugin's settings to another machine (#165).

The bundle is a file the user keeps and moves, so the things that matter are:
it says whose settings it holds, it does not leak a password past the switch
that governs passwords, and it is not world-readable when it holds one.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from aglaia.app_data import plugin_transfer as xfer
from aglaia.app_data.plugin_ctx import PluginConfig


class _Secrets:
    """A keychain stand-in: the real one needs an OS backend the CI box has
    no business unlocking.

    `unreadable` names secrets the index knows about but whose value cannot be
    fetched from here — the real shape of a GUI-stored key seen from a
    headless process.
    """

    def __init__(self, store=None, refuse=(), unreadable=()):
        self._v = dict(store or {})
        self._refuse = set(refuse)
        self._unreadable = set(unreadable)

    def keys(self):
        return sorted(set(self._v) | self._unreadable)

    def get(self, key):
        return None if key in self._unreadable else self._v.get(key)

    def set(self, key, value):
        if key in self._refuse:
            raise RuntimeError("keychain declined")
        self._v[key] = value


class _Ctx:
    def __init__(self, slug, config, secrets=None, version="1.0.0"):
        self.slug = slug
        self.config = config
        self.secrets = secrets
        self.version = version


@pytest.fixture
def ctx(tmp_path):
    cfg = PluginConfig("send-to-corpus", tmp_path / "config.db")
    cfg.set("base_url", "https://corpus.example.org")
    cfg.set("timeout_s", 300)
    cfg.set("extra_headers", json.dumps(["CF-Access-Client-Id"]))
    sec = _Secrets({"api_key": "k-123",
                    "extra_headers.CF-Access-Client-Id": "cf-abc"})
    return _Ctx("send-to-corpus", cfg, sec)


# ── what goes in the file ──────────────────────────────────────────────

def test_settings_only_by_default(ctx):
    b = xfer.build(ctx)
    assert b["settings"]["base_url"] == "https://corpus.example.org"
    assert b["secrets"] == {}
    assert b["contains_secrets"] is False


def test_secrets_are_opt_in(ctx):
    b = xfer.build(ctx, include_secrets=True)
    assert b["secrets"] == {"api_key": "k-123",
                            "extra_headers.CF-Access-Client-Id": "cf-abc"}
    assert b["contains_secrets"] is True


def test_the_plaintext_fallback_never_rides_out_as_a_setting(tmp_path):
    """With no keychain, `PluginSecrets` stores the value in the config DB
    under a reserved prefix. If `settings` were read raw, a settings-only
    export would carry the API key anyway — past the switch meant to govern
    it."""
    cfg = PluginConfig("send-to-corpus", tmp_path / "c.db")
    cfg.set("base_url", "https://corpus.example.org")
    cfg._raw_set("__secret__.api_key", "leaked")
    cfg._raw_set("__secret_keys__", ["api_key"])
    b = xfer.build(_Ctx("send-to-corpus", cfg, None))
    assert "leaked" not in json.dumps(b)
    assert list(b["settings"]) == ["base_url"]


# ── the file on disk ───────────────────────────────────────────────────

def test_a_bundle_with_secrets_is_not_world_readable(ctx, tmp_path):
    p = xfer.write(ctx, tmp_path / "x.json", include_secrets=True)
    mode = stat.S_IMODE(os.stat(p).st_mode)
    assert mode == 0o600, oct(mode)


def test_no_partial_file_is_left_behind(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(xfer.json, "dump",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
    out = tmp_path / "out"
    with pytest.raises(OSError):
        xfer.write(ctx, out / "x.json")
    assert list(out.iterdir()) == []


# ── reading one back ───────────────────────────────────────────────────

def test_round_trip(ctx, tmp_path):
    xfer.write(ctx, tmp_path / "x.json", include_secrets=True)
    target = _Ctx("send-to-corpus",
                  PluginConfig("send-to-corpus", tmp_path / "other.db"),
                  _Secrets())
    report = xfer.apply(target, xfer.read(tmp_path / "x.json"))
    assert target.config.get("base_url") == "https://corpus.example.org"
    assert target.config.get("timeout_s") == 300
    assert target.secrets.get("api_key") == "k-123"
    assert target.secrets.get("extra_headers.CF-Access-Client-Id") == "cf-abc"
    assert report.skipped == []
    assert report.total == 5


def test_another_plugins_bundle_is_refused(ctx, tmp_path):
    xfer.write(ctx, tmp_path / "x.json")
    other = _Ctx("send-to-kindle",
                 PluginConfig("send-to-kindle", tmp_path / "k.db"), _Secrets())
    with pytest.raises(xfer.TransferError) as e:
        xfer.apply(other, xfer.read(tmp_path / "x.json"))
    assert "send-to-corpus" in str(e.value)


def test_a_file_that_is_not_a_bundle_says_so(tmp_path):
    p = tmp_path / "notes.json"
    p.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(xfer.TransferError, match="not an Aglaïa settings"):
        xfer.read(p)
    p.write_text("not json at all", encoding="utf-8")
    with pytest.raises(xfer.TransferError, match="not an Aglaïa settings"):
        xfer.read(p)


def test_a_newer_format_is_refused_rather_than_guessed(ctx, tmp_path):
    p = tmp_path / "x.json"
    xfer.write(ctx, p)
    bundle = json.loads(p.read_text(encoding="utf-8"))
    bundle["version"] = xfer.VERSION + 1
    p.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(xfer.TransferError, match="newer version"):
        xfer.read(p)


def test_one_refused_key_does_not_lose_the_rest(ctx, tmp_path):
    xfer.write(ctx, tmp_path / "x.json", include_secrets=True)
    target = _Ctx("send-to-corpus",
                  PluginConfig("send-to-corpus", tmp_path / "o.db"),
                  _Secrets(refuse={"api_key"}))
    report = xfer.apply(target, xfer.read(tmp_path / "x.json"))
    assert report.skipped == ["api_key"]
    assert target.config.get("base_url") == "https://corpus.example.org"
    assert target.secrets.get("extra_headers.CF-Access-Client-Id") == "cf-abc"


def test_importing_settings_only_leaves_the_stored_password_alone(ctx, tmp_path):
    xfer.write(ctx, tmp_path / "x.json", include_secrets=True)
    target = _Ctx("send-to-corpus",
                  PluginConfig("send-to-corpus", tmp_path / "o.db"),
                  _Secrets({"api_key": "already-here"}))
    xfer.apply(target, xfer.read(tmp_path / "x.json"), include_secrets=False)
    assert target.secrets.get("api_key") == "already-here"


def test_counts_are_numbers_not_a_sentence(ctx, tmp_path):
    """The confirmation wording belongs to the front-end. A phrase built here
    would be interpolated into a translated string as an English fragment."""
    xfer.write(ctx, tmp_path / "x.json", include_secrets=True)
    assert xfer.counts(xfer.read(tmp_path / "x.json")) == (3, 2)
    xfer.write(ctx, tmp_path / "y.json")
    assert xfer.counts(xfer.read(tmp_path / "y.json")) == (3, 0)


# ── naming only what will really be written ────────────────────────────

def test_a_name_whose_value_is_out_of_reach_is_not_promised(tmp_path):
    """The index of names and the values live in different stores, so a name
    can be listed and its value unreachable. Warning about a password that is
    not going to be written teaches the user to ignore the warning."""
    cfg = PluginConfig("send-to-corpus", tmp_path / "c.db")
    ctx = _Ctx("send-to-corpus", cfg,
               _Secrets({"api_key": "k"}, unreadable={"proxy_token"}))
    assert xfer.exported_secret_names(ctx) == ["api_key"]
    assert xfer.unreadable_secret_names(ctx) == ["proxy_token"]
    assert xfer.build(ctx, include_secrets=True)["secrets"] == {"api_key": "k"}
