# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Discovery and wiring for `destinations` plugins (#133).

A destination is somewhere a finished export goes: a calibre library, a Kindle
mailbox, a corpus. Three ship with the app, under
``aglaia/plugins/destinations/<slug>/``, in exactly the layout the plugin
registry will use — same manifest, same one-module rule — so they are both
working destinations and the reference every submitted plugin is measured
against. A user's own destinations live in
``<APP_DATA>/plugins/destinations/<slug>/`` and load the same way.

The one asymmetry is trust, and it is deliberate: a bundled destination is
first-party code that shipped inside the signed application, so it does not
face the acknowledgement gate a dropped-in file does. The gate exists to stop
a user running a file *he* was handed; it has nothing to say about the app's
own contents.

Each loaded destination gets a `PluginContext` — settings, secrets, a scratch
dir, a log line — built by the host with the slug baked in, so a destination
reads its own configuration and nobody else's.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aglaia.app_data import app_data_dir
from aglaia.app_data.plugin_ctx import SLUG_RE, build_context
from aglaia.plugin_api import DESTINATION_REGISTRY, API_VERSION, Destination

KIND = "destinations"

#: The three that ship with the app.
BUNDLED_DIR = Path(__file__).resolve().parents[1] / "plugins" / KIND


def user_dir() -> Path:
    d = app_data_dir() / "plugins" / KIND
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class Found:
    slug: str
    dir: Path
    entry: Path
    manifest: dict
    bundled: bool


def _read_manifest(path: Path) -> dict:
    try:
        import tomllib
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


def discover(*, include_user: bool = True) -> list[Found]:
    """Every destination plugin directory that looks well-formed.

    Malformed ones are skipped rather than raised on: a broken plugin must not
    be able to stop the app from starting, and the ones that ARE well-formed
    should still load. What "well-formed" means is checked here and not later,
    so a bad manifest is a plugin that never appears rather than one that
    appears and then fails at the moment it is used."""
    out: list[Found] = []
    seen: set[str] = set()
    roots = [(BUNDLED_DIR, True)]
    if include_user:
        roots.append((user_dir(), False))
    for root, bundled in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "_")):
                continue
            slug = d.name
            if not SLUG_RE.match(slug) or slug in seen:
                continue
            man = _read_manifest(d / "aglaia-plugin.toml")
            plugin = man.get("plugin") or {}
            entry_name = str(plugin.get("entry") or "")
            if not entry_name or str(plugin.get("slug") or "") != slug:
                # A manifest that disagrees with its own directory is the one
                # thing that must never be guessed at: the directory decides
                # the slug, which decides the keychain namespace.
                continue
            entry = d / entry_name
            if not entry.is_file():
                continue
            want_api = (man.get("requires") or {}).get("api", 1)
            try:
                if int(want_api) != API_VERSION:
                    print(f"[destinations] {slug} wants plugin API "
                          f"{want_api}; this build implements {API_VERSION} "
                          f"— not loaded")
                    continue
            except (TypeError, ValueError):
                continue
            seen.add(slug)
            out.append(Found(slug, d, entry, man, bundled))
    return out


def _import_entry(found: Found):
    """Import the entry module under a name that cannot collide.

    Prefixed with the slug because two plugins may both call their module
    `main.py`, and the second import would otherwise silently return the
    first."""
    mod_name = f"aglaia_plugin_{found.slug.replace('-', '_')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, found.entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {found.entry}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_loaded: dict[str, Destination] = {}
#: slug -> why it did not load. A plugin that fails must be able to SAY so:
#: "check the Log tab" is not a diagnosis when nothing routes there, and the
#: user is left with a plugin that is installed, listed, and inert.
_errors: dict[str, str] = {}


def load_error(slug: str) -> str:
    """Why `slug` is not in `load_all()`, or "" if it is (or was never seen)."""
    load_all()
    return _errors.get(slug, "")


def load_all(*, log: Optional[Callable[[str, str], None]] = None
             ) -> dict[str, Destination]:
    """Import every discovered destination and return live, context-bound
    instances keyed by name. Idempotent — safe to call from the GUI and from
    a worker."""
    if _loaded:
        return dict(_loaded)
    _errors.clear()
    found_slugs: list[str] = []
    for found in discover():
        found_slugs.append(found.slug)
        try:
            _import_entry(found)
        except Exception as e:  # noqa: BLE001 — one bad plugin, not a dead app
            msg = f"{type(e).__name__}: {e}"
            _errors[found.slug] = f"it would not import — {msg}"
            print(f"[destinations] failed to import {found.slug}: {msg}")
            if log:
                log("WARNING", f"[destinations] {found.slug}: {msg}")
            continue
    caps_by_slug = {f.slug: (f.manifest.get("capabilities") or {})
                    for f in discover()}
    vers_by_slug = {f.slug: str((f.manifest.get("plugin") or {}).get(
        "version", "0.0.0")) for f in discover()}
    for name, cls in DESTINATION_REGISTRY.items():
        try:
            inst = cls()
        except Exception as e:  # noqa: BLE001
            _errors[name] = (f"it imported but would not construct — "
                             f"{type(e).__name__}: {e}")
            print(f"[destinations] {name} would not construct: {e}")
            continue
        caps = caps_by_slug.get(name, {})
        # A destination that declares secret fields needs a secrets object;
        # for the bundled three the manifest says so, and for anything else
        # the manifest is what the user consented to.
        wants = bool(caps.get("secrets")) or bool(cls.SECRET_FIELDS)
        try:
            inst.ctx = build_context(name, version=vers_by_slug.get(name, "0"),
                                     wants_secrets=wants)
        except ValueError as e:
            _errors[name] = f"its slug is unusable — {e}"
            print(f"[destinations] {name} has an unusable slug: {e}")
            continue
        _loaded[name] = inst
    # A plugin directory that was discovered but registered nothing: the
    # module imported and simply never called `@register_destination`, which
    # is the one failure that leaves no exception behind to report.
    for slug in found_slugs:
        if slug not in _loaded and slug not in _errors:
            _errors[slug] = ("it imported but registered no destination — is "
                             "the class decorated with @register_destination, "
                             "and does its `name` match the plugin slug?")
    return dict(_loaded)


def get(name: str) -> Optional[Destination]:
    return load_all().get(name)


def for_format(ext: str) -> list[Destination]:
    """Destinations that accept this export format, in display order.

    The GUI asks this instead of listing everything and failing later: a
    Markdown export should not offer a destination that only takes PDFs."""
    ext = str(ext or "").lower().lstrip(".")
    out = [d for d in load_all().values() if ext in d.accepts]
    return sorted(out, key=lambda d: (d.display or d.name).lower())


def forget(slug: str) -> None:
    """Drop every trace of one plugin from this process.

    Uninstalling removes the files, but the class stays in
    `DESTINATION_REGISTRY` and the module in `sys.modules` — so a removed
    destination went on being listed and offered until the app restarted.
    Registration happens at import, so undoing it means undoing the import
    too, or the next `load_all` would not re-run the module even if the
    plugin came back."""
    slug = str(slug)
    _loaded.pop(slug, None)
    _errors.pop(slug, None)
    DESTINATION_REGISTRY.pop(slug, None)
    sys.modules.pop(f"aglaia_plugin_{slug.replace('-', '_')}", None)


def reset_for_tests() -> None:
    _loaded.clear()
    _errors.clear()
