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
import os

import pytest

# `keyring` ships with the `cloud` extra (Mistral key storage). CI syncs only
# `dev`, so the fixture below skips when it's absent — secrets.py itself
# imports keyring lazily and degrades to the .env path. The
# `keychain_backend` tests do NOT take that fixture: reporting a missing
# PACKAGE as a missing BACKEND is exactly the bug (#107), and the case worth
# covering everywhere is the one where keyring is not installed.


@pytest.fixture()
def sec(tmp_path, monkeypatch):
    pytest.importorskip("keyring")
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
    # 0600 perms on the cleartext fallback — POSIX only. Windows can't
    # represent owner-only mode bits (os.chmod toggles just the read-only
    # bit), so st_mode never equals 0o600 there; on Windows the secure store
    # is the Credential Manager via keyring, and .env is a last-resort fallback.
    if os.name != "nt":
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


# ── keychain probe policy ─────────────────────────────────────────────

def test_keychain_read_prompts_only_on_macos(sec, monkeypatch):
    """Only the macOS Keychain asks the user to authorise a reading app. The
    Linux Secret Service and the Windows Credential Locker answer silently
    once the session is unlocked, so a UI that defers the probe there hides a
    key that is in fact stored — which is how a stored key looked missing on
    Linux for a whole session."""
    for plat, prompts in (("darwin", True), ("linux", False), ("win32", False)):
        monkeypatch.setattr(sec.sys, "platform", plat)
        assert sec.keychain_read_prompts() is prompts


# ── keychain availability, and WHY not ────────────────────────────────

def test_keychain_backend_reports_a_missing_package_as_such(monkeypatch):
    """`import keyring` failing is a missing dependency, not a broken OS
    keychain. Saying "No OS keychain was available" for it sent a user
    hunting a Keychain fault on a Mac whose Keychain was fine (#107)."""
    import builtins
    import aglaia.app_data.secrets as secrets
    real_import = builtins.__import__

    def _no_keyring(name, *a, **k):
        if name == "keyring" or name.startswith("keyring."):
            raise ModuleNotFoundError("No module named 'keyring'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_keyring)
    assert secrets.keychain_backend() == (False, "not_installed")


def test_keychain_backend_reports_the_fail_backend_as_no_backend(monkeypatch):
    """keyring installed but nothing to store into — a headless Linux with
    no Secret Service — resolves to `keyring.backends.fail.Keyring`."""
    pytest.importorskip("keyring")
    import keyring
    from keyring.backends.fail import Keyring as FailKeyring
    import aglaia.app_data.secrets as secrets
    monkeypatch.setattr(keyring, "get_keyring", lambda: FailKeyring())
    assert secrets.keychain_backend() == (False, "no_backend")


def test_keychain_backend_accepts_a_real_backend(monkeypatch):
    pytest.importorskip("keyring")
    import keyring
    from keyring.backend import KeyringBackend
    import aglaia.app_data.secrets as secrets

    class _Store(KeyringBackend):
        priority = 1  # type: ignore[assignment]

        def get_password(self, service, username):
            return None

        def set_password(self, service, username, password):
            pass

        def delete_password(self, service, username):
            pass

    monkeypatch.setattr(keyring, "get_keyring", lambda: _Store())
    assert secrets.keychain_backend() == (True, "")
