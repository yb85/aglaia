# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Drop-in plugin trust gate + discovery.

Users extend Aglaïa by dropping a ``*.py`` file into

    <APP_DATA>/plugins/processors/   — pipeline processors
    <APP_DATA>/plugins/ocr/          — OCR engines

Threat model: guard **only** against a user blindly running a file he
dropped (or that something dropped for him) — make him *consciously
acknowledge before it runs*. The data dir itself is assumed secure, so
there is no signing/keyring; the gate is a UX speed-bump.

Flow:
  * `scan_pending()` lists files that are new or whose content changed
    since they were accepted (sha256 mismatch). The GUI shows a popup;
    `acknowledge()` / `reject()` resolve each.
  * `import_accepted(kind)` puts the plugin dir on ``sys.path`` and
    imports every accepted, sha-matching module. It NEVER imports a
    pending file — import == code execution, so unacknowledged code
    must not run. Called by the processor registry and the OCR registry
    (and re-run inside spawned workers, which is why it's idempotent and
    DB-driven rather than popup-driven).
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from . import db as _db
from . import plugins_dir

KIND_PROCESSORS = "processors"
KIND_OCR = "ocr"
KINDS = (KIND_PROCESSORS, KIND_OCR)


@dataclass(frozen=True)
class PluginCandidate:
    kind: str
    path: Path
    sha256: str
    reason: str  # "new" | "changed"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iter_files() -> list[tuple[str, Path]]:
    """All importable ``.py`` files across the typed plugin dirs.

    Skips dunder/private files (``__init__``, ``_helpers`` …) so a plugin
    can ship private support modules without each one tripping the gate."""
    out: list[tuple[str, Path]] = []
    for kind in KINDS:
        d = plugins_dir(kind)
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_"):
                continue
            out.append((kind, p.resolve()))
    return out


def scan_pending() -> list[PluginCandidate]:
    """Files that need user acknowledgement: never accepted (``new``) or
    accepted-but-content-changed (``changed``)."""
    with _db.session() as conn:
        accepted = _db.accepted_plugins(conn)
    pending: list[PluginCandidate] = []
    for kind, path in _iter_files():
        key = str(path)
        cur = sha256_file(path)
        rec = accepted.get(key)
        if rec is None:
            pending.append(PluginCandidate(kind, path, cur, "new"))
        elif rec["sha256"] != cur:
            pending.append(PluginCandidate(kind, path, cur, "changed"))
    return pending


def acknowledge(candidate: PluginCandidate) -> None:
    """Mark a candidate trusted — its current content becomes accepted."""
    with _db.session() as conn:
        _db.acknowledge_plugin(conn, candidate.kind, candidate.path,
                               candidate.sha256)


def reject(candidate: PluginCandidate, *, delete_file: bool = True) -> None:
    """Decline a candidate. Drops any stale DB row and (default) deletes
    the file so it doesn't re-prompt every startup."""
    with _db.session() as conn:
        _db.forget_plugin(conn, candidate.path)
    if delete_file:
        try:
            Path(candidate.path).unlink()
        except FileNotFoundError:
            pass


def accepted_for_load(kind: str) -> list[Path]:
    """Paths of accepted plugins of ``kind`` whose on-disk content still
    matches the acknowledged hash. A changed file is excluded (it reverts
    to pending and the popup re-asks)."""
    with _db.session() as conn:
        accepted = _db.accepted_plugins(conn)
    out: list[Path] = []
    for key, rec in accepted.items():
        if rec["kind"] != kind:
            continue
        p = Path(key)
        if not p.is_file():
            continue
        if sha256_file(p) != rec["sha256"]:
            continue
        out.append(p)
    return out


def _ensure_on_path(kind: str) -> None:
    d = str(plugins_dir(kind))
    if d not in sys.path:
        sys.path.insert(0, d)


def import_accepted(kind: str) -> list[str]:
    """Import every accepted, sha-matching plugin module of ``kind``.

    Returns the module names imported. Modules are imported by file stem
    off a ``sys.path`` entry so the name resolves identically in spawned
    workers (spawn re-imports by name — a path-only import would break
    unpickling of plugin classes). Import failures are logged, not fatal.
    """
    _ensure_on_path(kind)
    imported: list[str] = []
    for path in accepted_for_load(kind):
        mod_name = path.stem
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:  # noqa: BLE001 — one bad plugin must not kill startup
            print(f"[plugins] failed to import {kind} plugin {path}: {e}")
    return imported
