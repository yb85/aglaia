# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Move one plugin's settings between machines (#165).

Configuring a destination means a URL, an API key, and sometimes the headers
of whatever proxy stands in front of it. None of that is reachable by hand:
the settings live in the plugin's own SQLite file and the secrets in the OS
keychain. So a second machine means typing it all again.

The bundle is JSON and names the plugin it belongs to. Nothing here is
plugin-specific — it reads whatever the plugin declared, through the same
`PluginContext` the settings dialog writes with, so a plugin written next year
exports correctly without knowing this module exists.

**Secrets are written in clear.** A keychain's whole point is that nothing
else can read it, and a file next to it is not a keychain. So carrying them is
opt-in per export (`include_secrets`), the file is written `0600`, and the
caller is expected to have said so in words the user read. `contains_secrets`
in the bundle lets an importer say what it is about to install.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Optional

#: What `format` must say. A bundle is a file the user may keep for a year and
#: carry to a build that has moved on, so it identifies itself.
FORMAT = "aglaia-plugin-settings"
VERSION = 1

#: The extension offered in the file dialogs. Double-barrelled so a bundle is
#: recognisable in a downloads folder, and still a plain `.json` to anything
#: that reads it.
SUFFIX = ".aglplugin.json"


class TransferError(RuntimeError):
    """The bundle cannot be read, or does not belong to this plugin."""


class ImportReport(NamedTuple):
    """What an import actually did. Returned rather than logged: the dialog
    tells the user, and a test asserts on it."""

    settings: list[str]
    secrets: list[str]
    skipped: list[str]

    @property
    def total(self) -> int:
        return len(self.settings) + len(self.secrets)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build(ctx, *, include_secrets: bool = False,
          plugin_version: str = "") -> dict[str, Any]:
    """The bundle for `ctx`'s plugin, as a dict.

    `config.all()` already drops the host-reserved rows (the secret index, the
    no-keychain fallback), which is exactly right here: they are not settings,
    and the fallback rows ARE the secrets — exporting them under `settings`
    would leak a password past the `include_secrets` switch that is supposed
    to govern it.
    """
    settings = dict(ctx.config.all()) if ctx.config is not None else {}
    secrets: dict[str, str] = {}
    if include_secrets and getattr(ctx, "secrets", None) is not None:
        for key in ctx.secrets.keys():
            value = ctx.secrets.get(key)
            if value is not None:
                secrets[key] = value
    return {
        "format": FORMAT,
        "version": VERSION,
        "slug": ctx.slug,
        "plugin_version": str(plugin_version or getattr(ctx, "version", "")),
        "exported_at": _now(),
        "contains_secrets": bool(secrets),
        "settings": settings,
        "secrets": secrets,
    }


def write(ctx, path: Path, *, include_secrets: bool = False,
          plugin_version: str = "") -> Path:
    """Write the bundle to `path`. Returns the path actually written.

    `0600` whenever it carries secrets, and the mode is set on the temporary
    file BEFORE the content goes in — a chmod after the write leaves a window
    in which the key is on disk world-readable. Written through a temp file in
    the same directory and renamed, so a failure half-way leaves no partial
    bundle for someone to import.

    `os.fchmod` is POSIX-only, and calling it unguarded made this raise
    `AttributeError` on Windows — so exporting with secrets did not merely
    skip the permission bit, it failed outright (#177). Where the platform has
    no POSIX mode the file is written anyway, inheriting the directory's ACL:
    the sentence the user agreed to ("anyone who opens the file can read
    them") is the true one on every platform, and refusing to write the file
    at all would be a worse answer than writing it.
    """
    bundle = build(ctx, include_secrets=include_secrets,
                   plugin_version=plugin_version)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        if bundle["contains_secrets"] and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read(path: Path) -> dict[str, Any]:
    """Parse and sanity-check a bundle. Raises `TransferError` with something
    the user can act on — "expecting value: line 1 column 1" is not that."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise TransferError(f"The file could not be read: {e}") from e
    try:
        bundle = json.loads(raw)
    except ValueError as e:
        raise TransferError("This is not an Aglaïa settings file.") from e
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT:
        raise TransferError("This is not an Aglaïa settings file.")
    try:
        version = int(bundle.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version > VERSION:
        raise TransferError(
            "This file was written by a newer version of Aglaïa.")
    if not isinstance(bundle.get("settings"), dict):
        bundle["settings"] = {}
    if not isinstance(bundle.get("secrets"), dict):
        bundle["secrets"] = {}
    return bundle


def apply(ctx, bundle: dict[str, Any], *,
          include_secrets: bool = True) -> ImportReport:
    """Write `bundle` into this plugin's stores.

    Refuses a bundle for another plugin. Two plugins can hold a key called
    `base_url` meaning different servers, and the one thing worse than no
    settings is someone else's — so the slug is checked, not coerced.

    A key the stores refuse (a name this build's `KEY_RE` will not take, a
    keychain that declines) is collected into `skipped` rather than aborting:
    a bundle is usually mostly good, and an import that stops at the first bad
    row leaves the plugin half-configured with no account of what landed.
    """
    slug = str(bundle.get("slug") or "")
    if slug != ctx.slug:
        raise TransferError(
            f"These settings are for {slug or 'another plugin'}, "
            f"not for {ctx.slug}.")
    settings_done: list[str] = []
    secrets_done: list[str] = []
    skipped: list[str] = []
    for key, value in (bundle.get("settings") or {}).items():
        try:
            ctx.config.set(str(key), value)
            settings_done.append(str(key))
        except Exception:  # noqa: BLE001 — see docstring
            skipped.append(str(key))
    if include_secrets:
        for key, value in (bundle.get("secrets") or {}).items():
            if ctx.secrets is None:
                skipped.append(str(key))
                continue
            try:
                ctx.secrets.set(str(key), str(value))
                secrets_done.append(str(key))
            except Exception:  # noqa: BLE001
                skipped.append(str(key))
    return ImportReport(sorted(settings_done), sorted(secrets_done),
                        sorted(skipped))


def default_filename(slug: str) -> str:
    return f"{slug}{SUFFIX}"


def counts(bundle: dict[str, Any]) -> tuple[int, int]:
    """How many settings and how many secrets are in this bundle.

    Numbers, not a sentence. The sentence belongs to whichever front-end is
    asking — the dialog wraps it in `tr()`, and a phrase built here would be
    interpolated into a translated string as an English fragment, which is how
    a French dialog ends up reading "… par 3 setting(s) and 2 password(s) ?".
    """
    return (len(bundle.get("settings") or {}),
            len(bundle.get("secrets") or {}))


def exported_secret_names(ctx) -> list[str]:
    """Which secrets an export WOULD carry. The warning names them, so the
    user is agreeing to something specific rather than to the word
    'passwords'.

    Only the ones that can actually be READ here. The index of names and the
    values live in different places — the index in the plugin's config DB, the
    values in the keychain or the .env — so a name can be listed and its value
    be out of reach. Warning about a password that is not going to be written
    is a warning the user will learn to ignore.
    """
    return [k for k, v in _readable(ctx).items() if v is not None]


def unreadable_secret_names(ctx) -> list[str]:
    """Secrets this plugin has stored that cannot be read from here, so an
    export would silently leave them out. The caller says so rather than
    reporting a clean export of nothing."""
    return [k for k, v in _readable(ctx).items() if v is None]


def _readable(ctx) -> dict[str, Optional[str]]:
    secrets: Optional[Any] = getattr(ctx, "secrets", None)
    if secrets is None:
        return {}
    try:
        names = list(secrets.keys())
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, Optional[str]] = {}
    for key in names:
        try:
            out[key] = secrets.get(key)
        except Exception:  # noqa: BLE001
            out[key] = None
    return out
