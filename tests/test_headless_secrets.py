# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A headless run keeps its secrets in `APP_DATA/.env`, not in a keychain.

A keychain is a session service. On Linux it needs a logged-in desktop to
unlock, so a key a terminal (or the GUI) put there is unreadable to the cron
job, the ssh session or the systemd unit that has to use it — and the failure
looks like a wrong key, which sends the user to rotate a key that was fine.

Reading is unchanged and still checks every store, so nothing already in a
keychain becomes unreachable.
"""
import importlib

import pytest


@pytest.fixture()
def app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("AGLAIA_SECRETS_PLAINTEXT", raising=False)
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.secrets as sec
    for m in (ad, sec, pc):
        importlib.reload(m)
    yield tmp_path, sec, pc
    sec.use_plaintext_store(False)


def _env_text(tmp_path):
    p = tmp_path / ".env"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_the_cli_turns_the_plaintext_store_on_for_every_headless_command(app_data):
    _tmp, sec, _pc = app_data
    from aglaia.cli import run
    assert sec.plaintext_store() is False
    run(["version"])
    assert sec.plaintext_store() is True


def test_the_gui_command_leaves_the_keychain_in_charge(app_data):
    _tmp, sec, _pc = app_data
    from aglaia.cli import _prepare_args
    # `aglaia book.agl` is the gui command; it must not flip the store.
    assert _prepare_args(["book.agl"])[0] == "gui"
    assert sec.plaintext_store() is False


def test_an_env_var_overrides_the_mode_either_way(app_data, monkeypatch):
    _tmp, sec, _pc = app_data
    sec.use_plaintext_store(True)
    monkeypatch.setenv("AGLAIA_SECRETS_PLAINTEXT", "0")
    assert sec.plaintext_store() is False
    sec.use_plaintext_store(False)
    monkeypatch.setenv("AGLAIA_SECRETS_PLAINTEXT", "1")
    assert sec.plaintext_store() is True


def test_a_plugin_secret_lands_in_the_env_file_and_reads_back(app_data):
    tmp_path, sec, pc = app_data
    sec.use_plaintext_store(True)
    store = pc.PluginSecrets("send-to-corpus")
    store.set("api_key", "k-1")
    store.set("extra_headers.CF-Access-Client-Id", "id-1")

    text = _env_text(tmp_path)
    assert "AGLAIA_PLUGIN_SEND_TO_CORPUS_API_KEY=k-1" in text
    assert ("AGLAIA_PLUGIN_SEND_TO_CORPUS_EXTRA_HEADERS_CF_ACCESS_CLIENT_ID=id-1"
            in text)
    assert store.get("api_key") == "k-1"
    assert store.get("extra_headers.CF-Access-Client-Id") == "id-1"
    # The names are still listed (that index is what the settings view reads).
    assert "api_key" in store.keys()


def test_deleting_a_secret_removes_its_line(app_data):
    tmp_path, sec, pc = app_data
    sec.use_plaintext_store(True)
    store = pc.PluginSecrets("send-to-corpus")
    store.set("api_key", "k-1")
    store.delete("api_key")
    assert "AGLAIA_PLUGIN_SEND_TO_CORPUS_API_KEY" not in _env_text(tmp_path)
    assert store.get("api_key") is None


def test_the_env_file_stays_private(app_data):
    tmp_path, sec, pc = app_data
    sec.use_plaintext_store(True)
    pc.PluginSecrets("send-to-corpus").set("api_key", "k-1")
    mode = (tmp_path / ".env").stat().st_mode & 0o777
    assert mode == 0o600, f"the .env holds secrets; found mode {mode:o}"


def test_the_mistral_key_goes_to_the_same_file(app_data):
    tmp_path, sec, _pc = app_data
    sec.use_plaintext_store(True)
    assert sec.set_mistral_api_key("m-1") == "env_file"
    assert f"{sec.ENV_MISTRAL}=m-1" in _env_text(tmp_path)
    assert sec.get_mistral_api_key() == "m-1"
    assert sec.set_mistral_api_key("") == ""
    assert sec.ENV_MISTRAL not in _env_text(tmp_path)


def test_a_secret_written_by_an_older_build_is_still_read(app_data):
    """Before this, a keychain-less install wrote into the config DB."""
    _tmp, sec, pc = app_data
    store = pc.PluginSecrets("send-to-corpus")
    store._config._raw_set(store._FALLBACK_PREFIX + "api_key", "legacy")
    sec.use_plaintext_store(True)
    assert store.get("api_key") == "legacy"
