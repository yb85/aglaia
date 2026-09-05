# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Importing plugins so their windows register (#130).

A plugin contributes a window by calling `register_window` at import time. The
`destinations` loader already imports its own kind; this does the same for
`processors` and `ocr` plugin directories, which are otherwise imported by the
processor registry only when a pipeline actually uses them — too late for a
menu built at startup.

Import is code execution, so this obeys the same rule everything else does: a
disabled plugin is not imported, and a plugin that raises is logged and
skipped rather than taking the menu with it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional

from aglaia.app_data.plugin_manifest import KINDS, ManifestError, parse_manifest
from aglaia.app_data.plugin_registry import installed_root, is_disabled

_done = False


def load_all(*, log: Optional[Callable[[str], None]] = None) -> list[str]:
    """Import every installed plugin that declares `ui`. Idempotent."""
    global _done
    if _done:
        return []
    _done = True
    imported: list[str] = []
    for kind in KINDS:
        root = installed_root(kind)
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "_")):
                continue
            if is_disabled(d.name):
                continue
            try:
                man = parse_manifest(d / "aglaia-plugin.toml", kind=kind,
                                     expect_slug=d.name)
            except ManifestError:
                continue
            if not man.ui:
                continue
            mod_name = f"aglaia_plugin_{d.name.replace('-', '_')}"
            if mod_name in sys.modules:
                imported.append(d.name)
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    mod_name, d / man.entry)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                imported.append(d.name)
            except Exception as e:  # noqa: BLE001
                msg = f"[plugin-windows] {d.name} would not import: {e}"
                print(msg)
                if log:
                    log(msg)
                sys.modules.pop(mod_name, None)
    return imported


def reset_for_tests() -> None:
    global _done
    _done = False
