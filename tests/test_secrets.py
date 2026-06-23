# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Secret storage (aglaia/app_data/secrets) — offline tests.

APP_DATA is redirected to a tmp dir; the OS keychain is forced to fail so
the `.env` fallback path is exercised hermetically (no real Keychain
writes on the dev machine).
"""

import importlib

import pytest

# `keyring` ships with the `cloud` extra (Mistral key storage). CI syncs only
# `dev`, so skip this module's keychain-fallback tests when it's absent —
# secrets.py itself imports keyring lazily and degrades to the .env path.
pytest.importorskip("keyring")


@pytest.fixture()
def sec(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    import aglaia.app_data as ad
    import aglaia.app_data.secrets as secrets
    importlib.reload(ad)
    importlib.reload(secrets)
    # Force keychain unavailable so set()/get() use the .env fallback.
    import keyring
    def _boom(*a, **k):
        raise RuntimeError("no backend")
    monkeypatch.setattr(keyring, "set_password", _boom)
    monkeypatch.setattr(keyring, "get_password", _boom)
    monkeypatch.setattr(keyring, "delete_password", _boom)
    return secrets


def test_set_falls_back_to_env_file(sec, tmp_path):
    where = sec.set_mistral_api_key("sk-abc")
    assert where == "env_file"
    assert (tmp_path / ".env").exists()
    assert sec.get_mistral_api_key() == "sk-abc"
    assert sec.mistral_key_location() == "env_file"
    # 0600 perms on the cleartext fallback
    import stat
    mode = stat.S_IMODE((tmp_path / ".env").stat().st_mode)
    assert mode == 0o600


def test_env_var_overrides_file(sec, monkeypatch):
    sec.set_mistral_api_key("in-file")
    monkeypatch.setenv("MISTRAL_API_KEY", "in-env")
    assert sec.get_mistral_api_key() == "in-env"
    assert sec.mistral_key_location() == "env"


def test_clear_removes_key(sec):
    sec.set_mistral_api_key("sk-xyz")
    assert sec.get_mistral_api_key() == "sk-xyz"
    assert sec.set_mistral_api_key("") == ""
    assert sec.get_mistral_api_key() == ""
    assert sec.mistral_key_location() == ""


def test_env_file_ignores_comments_and_blanks(sec, tmp_path):
    (tmp_path / ".env").write_text(
        "# a comment\n\nMISTRAL_API_KEY = \"sk-quoted\"\nOTHER=1\n")
    assert sec.get_mistral_api_key() == "sk-quoted"
