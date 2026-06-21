# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Secret storage for Aglaïa (API keys).

The only secret today is the **Mistral Cloud OCR** API key.

**Read** order (first hit wins):

1. **Env var** ``MISTRAL_API_KEY`` — highest priority. Headless / CI
   inject without touching disk; a power user overrides what's persisted.
2. **Plaintext ``APP_DATA/.env``** (``MISTRAL_API_KEY=…``) — checked
   *before* the keychain so a dev who prefers the dotenv way never
   triggers an OS keychain probe (which can pop an unlock prompt).
3. **OS keychain** via `keyring` — macOS Keychain, Windows Credential
   Locker, Linux Secret Service.

**Write** prefers the most secure store: the OS keychain, falling back to
the plaintext ``.env`` (0600) only when no keychain backend is available
(headless Linux / bare Windows). Whichever store wins, the other's copy is
cleared so there's a single source of truth.

``set_mistral_api_key`` returns where the value landed (``"keychain"`` /
``"env_file"`` / ``""`` when cleared) so the UI can tell the user.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import app_data_dir

# keyring service / account namespacing.
_SERVICE = "aglaia"
_ACCOUNT_MISTRAL = "mistral_api_key"

# Env var read at the top of the lookup chain (also the .env file's key).
ENV_MISTRAL = "MISTRAL_API_KEY"


def _env_file() -> Path:
    return app_data_dir() / ".env"


# ── .env (plaintext fallback) ─────────────────────────────────────────

def _read_env_file() -> dict[str, str]:
    """Parse ``APP_DATA/.env`` into a dict. Tolerant: blank lines and
    ``#`` comments are ignored; values keep their raw text (optionally
    surrounded by quotes, which are stripped)."""
    path = _env_file()
    out: dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    except Exception:
        return {}
    return out


def _write_env_file(values: dict[str, str]) -> None:
    """Rewrite ``APP_DATA/.env`` from ``values`` (0600 perms). Keys with
    empty values are dropped so clearing a secret removes the line."""
    path = _env_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Aglaïa secrets — plaintext fallback when no OS keychain is",
        "# available. Prefer the OS keychain (set via the GUI). Do not commit.",
    ]
    for k, v in values.items():
        if v:
            lines.append(f"{k}={v}")
    body = "\n".join(lines) + "\n"
    path.write_text(body, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ── public API ────────────────────────────────────────────────────────

def get_mistral_api_key() -> str:
    """Resolve the Mistral key. Lookup order:

    1. process env ``MISTRAL_API_KEY``
    2. ``APP_DATA/.env`` file — checked **before** the keychain so a dev
       who prefers the dotenv way never triggers an OS keychain probe
       (which can pop an unlock prompt).
    3. OS keychain.

    ``""`` when unset everywhere."""
    env = os.environ.get(ENV_MISTRAL, "").strip()
    if env:
        return env
    from_file = _read_env_file().get(ENV_MISTRAL, "").strip()
    if from_file:
        return from_file
    try:
        import keyring
        kv = keyring.get_password(_SERVICE, _ACCOUNT_MISTRAL)
        if kv:
            return kv.strip()
    except Exception:
        pass
    return ""


def set_mistral_api_key(value: str) -> str:
    """Persist (or clear) the Mistral key. Returns the store it landed in:
    ``"keychain"``, ``"env_file"``, or ``""`` when cleared.

    Tries the OS keychain first; on any keychain failure (no backend on a
    headless box) falls back to the plaintext ``.env`` file. Whichever
    store wins, the *other* store's copy is cleared so there's a single
    source of truth and no stale secret left behind."""
    value = (value or "").strip()

    # 1. Try the OS keychain.
    try:
        import keyring
        if value:
            keyring.set_password(_SERVICE, _ACCOUNT_MISTRAL, value)
        else:
            try:
                keyring.delete_password(_SERVICE, _ACCOUNT_MISTRAL)
            except Exception:
                pass
        _env_file_clear(ENV_MISTRAL)          # drop any plaintext copy
        return "keychain" if value else ""
    except Exception:
        pass

    # 2. Fallback: plaintext .env in APP_DATA.
    values = _read_env_file()
    if value:
        values[ENV_MISTRAL] = value
    else:
        values.pop(ENV_MISTRAL, None)
    _write_env_file(values)
    return "env_file" if value else ""


def _env_file_clear(key: str) -> None:
    values = _read_env_file()
    if key in values:
        values.pop(key, None)
        _write_env_file(values)


def mistral_key_location() -> str:
    """Where the key currently resolves from — for UI status. One of
    ``"env"`` (process env), ``"keychain"``, ``"env_file"``, or ``""``."""
    if os.environ.get(ENV_MISTRAL, "").strip():
        return "env"
    if _read_env_file().get(ENV_MISTRAL, "").strip():
        return "env_file"
    try:
        import keyring
        if keyring.get_password(_SERVICE, _ACCOUNT_MISTRAL):
            return "keychain"
    except Exception:
        pass
    return ""
