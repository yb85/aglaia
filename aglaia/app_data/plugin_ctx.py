# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""What a plugin is handed instead of the host's internals (#127).

A plugin needs three things the app already has: somewhere to keep settings,
somewhere to keep a credential, and somewhere to write a file. Left to itself
it would reach for `sqlite3`, `keyring` and `pathlib` — and then it would be
holding the app's own config DB, the app's own keychain namespace, and a path
to anywhere. `PluginContext` hands it a bounded version of each, and those
modules stay off the import allow-list so reaching around it is a *visible*
violation a reviewer can act on.

Read `docs/plugin-store.md` §1 before assuming this is a security boundary. It
is not: an in-process plugin can `import keyring` and read anything the user
can. What this buys is that accidents are impossible, intent is declared, and
misuse is legible. The boundary is the review.

Layout, all under ``<APP_DATA>/plugins/data/<slug>/``:

    config.db   one `kv` table, this plugin's settings
    files/      scratch — a cached model, a temp render

One file per plugin, so a plugin that corrupts or bloats its store damages
nothing but itself, and uninstalling is a directory removal.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import app_data_dir

#: A slug names a directory, a keychain namespace and a config file, so it is
#: validated once, here, and nowhere else has to wonder. Lowercase only: two
#: plugins differing by case would collide on a case-insensitive filesystem.
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")

#: Keychain service shared by every plugin. The plugin never supplies it.
SECRET_SERVICE = "aglaia.plugin"

#: Separator between slug and key inside the keychain username. U+001F is a
#: control character: `SLUG_RE` cannot produce it and `_check_key` refuses it,
#: so no plugin can build a username that lands in another's namespace.
NS_SEP = "\x1f"

#: Keys are names the user may see in a settings form, not free-form blobs.
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        raise ValueError(
            f"invalid plugin slug {slug!r}: 3-40 chars, lowercase letters, "
            f"digits and hyphens, not starting or ending with a hyphen")
    return slug


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise ValueError(
            f"invalid key {key!r}: letters, digits, '_', '.', '-'; "
            f"1-64 chars, starting alphanumeric")
    return key


def plugin_data_dir(slug: str, *, create: bool = True) -> Path:
    """``<APP_DATA>/plugins/data/<slug>/`` — created on access.

    `create=False` for the one caller that is about to DELETE the directory:
    creating it there means uninstall removes a folder it just made, and on a
    failure leaves a fresh empty one behind."""
    d = app_data_dir() / "plugins" / "data" / validate_slug(slug)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ── settings ──────────────────────────────────────────────────────────

class PluginConfig:
    """A plugin's own key/value store, in its own SQLite file.

    Values round-trip through JSON, so a plugin gets back the type it put in
    rather than a string it has to parse — the app's own config DB works the
    same way, and a plugin author should not have to learn a second rule."""

    def __init__(self, slug: str, path: Optional[Path] = None) -> None:
        self.slug = validate_slug(slug)
        self.path = Path(path) if path is not None else (
            plugin_data_dir(self.slug) / "config.db")
        self._ensure()

    @contextlib.contextmanager
    def _connect(self):
        """A connection that is committed AND closed.

        `with sqlite3.connect(...) as conn` commits the transaction and leaves
        the connection open — so every get and set leaked a file handle, and
        the handles kept the file alive until the garbage collector felt like
        it. POSIX deletes an open file anyway; Windows does not, which is how
        uninstalling a plugin came to leave its settings on disk."""
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                " key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    # Host-reserved rows. `KEY_RE` refuses a leading underscore, so a plugin
    # calling `get`/`set` can never name one of these — the reserved namespace
    # is unreachable through the plugin-facing API, by construction.
    def _raw_get(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def _raw_set(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, ensure_ascii=False)))
            conn.commit()

    def _raw_delete(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        _check_key(key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        _check_key(key)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, ensure_ascii=False)))
            conn.commit()

    def delete(self, key: str) -> None:
        _check_key(key)
        with self._connect() as conn:
            conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            conn.commit()

    def all(self) -> dict[str, Any]:
        """The plugin's own settings. Host-reserved rows (the secret index,
        the no-keychain fallback) are not settings and are not listed —
        `all()` feeds a settings form, and a password does not belong in one."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM kv").fetchall()
        out: dict[str, Any] = {}
        for r in rows:
            if str(r["key"]).startswith("_"):
                continue
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                continue
        return out


# ── secrets ───────────────────────────────────────────────────────────

class PluginSecrets:
    """A plugin's own corner of the OS keychain.

    Constructed BY THE HOST with the slug already bound — there is no argument
    a plugin can pass to reach another's namespace, because the namespace is
    not an argument. `keys()` returns names, never values: a settings form
    needs to know whether a password is set, not what it is.

    Falls back to the plugin's own config DB, under a reserved key prefix, when
    no OS keychain is reachable — the same trade the host makes for the Mistral
    key (see `app_data/secrets.py`). `available` says which one is in use so a
    UI can tell the user his credential is on disk in the clear."""

    _FALLBACK_PREFIX = "__secret__."

    def __init__(self, slug: str,
                 config: Optional["PluginConfig"] = None) -> None:
        self.slug = validate_slug(slug)
        self._config = config if config is not None else PluginConfig(slug)

    # -- backend -------------------------------------------------------
    def _keyring(self):
        try:
            import keyring
            from keyring.backends.fail import Keyring as _Fail
            backend = keyring.get_keyring()
            if isinstance(backend, _Fail):
                return None
            return keyring
        except Exception:
            return None

    @property
    def available(self) -> bool:
        """True when a real OS keychain is holding these, False when they are
        falling back to the plugin's own (plaintext) config file."""
        return self._keyring() is not None

    def _username(self, key: str) -> str:
        return f"{self.slug}{NS_SEP}{_check_key(key)}"

    # -- api -----------------------------------------------------------
    def get(self, key: str) -> Optional[str]:
        kr = self._keyring()
        if kr is not None:
            try:
                v = kr.get_password(SECRET_SERVICE, self._username(key))
                if v is not None:
                    return v
            except Exception:
                pass
        v = self._config._raw_get(self._FALLBACK_PREFIX + _check_key(key))
        return str(v) if isinstance(v, str) else None

    def set(self, key: str, value: str) -> None:
        _check_key(key)
        if value is None:
            self.delete(key)
            return
        kr = self._keyring()
        if kr is not None:
            try:
                kr.set_password(SECRET_SERVICE, self._username(key),
                                str(value))
                # One home only: a stale plaintext copy is exactly the thing
                # a keychain was meant to prevent.
                self._config._raw_delete(self._FALLBACK_PREFIX + key)
                self._remember(key)
                return
            except Exception:
                pass
        self._config._raw_set(self._FALLBACK_PREFIX + key, str(value))
        self._remember(key)

    def delete(self, key: str) -> None:
        _check_key(key)
        kr = self._keyring()
        if kr is not None:
            try:
                kr.delete_password(SECRET_SERVICE, self._username(key))
            except Exception:
                pass
        self._config._raw_delete(self._FALLBACK_PREFIX + key)
        self._forget(key)

    def keys(self) -> list[str]:
        """Names of the secrets this plugin has stored. Never values.

        Kept as an index in the plugin's config DB because `keyring` has no
        portable "list what I stored" call — the Secret Service can enumerate,
        the macOS backend cannot."""
        idx = self._config._raw_get("__secret_keys__", [])
        return sorted(idx) if isinstance(idx, list) else []

    def _remember(self, key: str) -> None:
        idx = set(self.keys()) | {key}
        self._config._raw_set("__secret_keys__", sorted(idx))

    def _forget(self, key: str) -> None:
        idx = set(self.keys()) - {key}
        self._config._raw_set("__secret_keys__", sorted(idx))

    def purge(self) -> None:
        """Drop every secret this plugin stored — used at uninstall."""
        for key in self.keys():
            self.delete(key)


# ── the context ───────────────────────────────────────────────────────

@dataclass
class PluginContext:
    """Everything a plugin is allowed to reach in the host.

    The host builds it; the plugin receives it. Deliberately small: each field
    replaces one import that is not on the allow-list, and nothing here hands
    out a path, a connection or a credential belonging to anyone else."""

    slug: str
    version: str = "0.0.0"
    config: PluginConfig = None          # type: ignore[assignment]
    secrets: Optional[PluginSecrets] = None
    log: Callable[[str], None] = None    # type: ignore[assignment]

    @property
    def data_dir(self) -> Path:
        """Scratch space for this plugin — created on access."""
        d = plugin_data_dir(self.slug) / "files"
        d.mkdir(parents=True, exist_ok=True)
        return d


def build_context(slug: str, *, version: str = "0.0.0",
                  wants_secrets: bool = False,
                  log: Optional[Callable[[str], None]] = None
                  ) -> PluginContext:
    """The host's constructor. `wants_secrets` comes from the manifest, so a
    plugin that never declared secrets has no object to misuse — and the user
    saw "stores secrets" in the install dialog for one that did."""
    slug = validate_slug(slug)
    config = PluginConfig(slug)
    return PluginContext(
        slug=slug,
        version=str(version),
        config=config,
        secrets=PluginSecrets(slug, config) if wants_secrets else None,
        log=log or (lambda msg: print(f"[plugin:{slug}] {msg}", flush=True)),
    )
