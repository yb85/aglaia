# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Aglaïa per-user APP_DATA directory.

Cross-platform locations resolved through `platformdirs` (tox-dev). The
project's vendor domain is **bibli.cc**; appauthor is set accordingly so
that platforms which scope per vendor (Windows) nest the app dir
correctly.

Resolved paths (defaults shown):

| Platform | `app_data_dir()`                                  | `log_dir()`                                   | `cache_dir()`                              |
|----------|---------------------------------------------------|-----------------------------------------------|--------------------------------------------|
| macOS    | `~/Library/Application Support/Aglaia`            | `~/Library/Logs/Aglaia`                        | `~/Library/Caches/Aglaia`                   |
| Linux    | `$XDG_DATA_HOME/Aglaia` (~/.local/share/Aglaia)    | `$XDG_STATE_HOME/Aglaia/log` (~/.local/state/...) | `$XDG_CACHE_HOME/Aglaia` (~/.cache/Aglaia)  |
| Windows  | `%APPDATA%\\bibli.cc\\Aglaia`                       | `%LOCALAPPDATA%\\bibli.cc\\Aglaia\\Logs`        | `%LOCALAPPDATA%\\bibli.cc\\Aglaia\\Cache`    |

Layout:

```
APP_DATA/
  aglaia-config.db        SQLite — config KV + recent projects
  pipelines/              user pipelines (mirrors repo `config/pipelines/`)

CACHE/
  models/                 downloaded ML weights (Surya, EAST, DBNet …).
                          Lives in cache_dir on purpose — safe to delete,
                          re-downloads on demand.

LOG/                      app run logs (one rotated file per session).
```

Override via env vars for tests / portable installs:
- `AGLAIA_APP_DATA_DIR` — overrides `app_data_dir()`
- `AGLAIA_LOG_DIR`      — overrides `log_dir()`
- `AGLAIA_CACHE_DIR`    — overrides `cache_dir()`
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from platformdirs import (
    user_cache_dir, user_data_dir, user_documents_dir, user_log_dir,
)

APP_NAME = "Aglaia"
# Reverse-DNS vendor identifier — matches the download point
# aglaia.bibli.cc and groups per-vendor dirs on Windows.
APP_AUTHOR = "bibli.cc"

_ENV_DATA = "AGLAIA_APP_DATA_DIR"
_ENV_LOG = "AGLAIA_LOG_DIR"
_ENV_CACHE = "AGLAIA_CACHE_DIR"


def _resolve(env_var: str, platformdir_fn) -> Path:
    override = os.environ.get(env_var, "").strip()
    if override:
        path = Path(override).expanduser()
    else:
        path = Path(platformdir_fn(APP_NAME, APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_data_dir() -> Path:
    return _resolve(_ENV_DATA, user_data_dir)


def log_dir() -> Path:
    return _resolve(_ENV_LOG, user_log_dir)


def cache_dir() -> Path:
    return _resolve(_ENV_CACHE, user_cache_dir)


# Bundled read-only pipeline defs live INSIDE the package (aglaia/config/…),
# both in source and in the PyInstaller bundle (spec ships them at the same
# package-relative path). `parents[1]` is the `aglaia/` package dir; using
# `parents[2]` (the old repo-root `config/`) silently finds nothing on a fresh
# install — config moved into the package during the pip-packaging refactor.
_BUNDLED_PIPELINES = Path(__file__).resolve().parents[1] / "config" / "pipelines"
_pipelines_seeded = False


def bundled_pipelines_dir() -> Path:
    """Read-only repo/bundle pipeline definitions, seeded into APP_DATA."""
    return _BUNDLED_PIPELINES


def seed_pipelines(*, force: bool = False) -> Path:
    """Copy the bundled pipelines into `<APP_DATA>/pipelines` so they are
    user-editable. Existing files are kept (a user edit wins) unless
    `force` — Settings → "Restore original pipelines" passes force=True.
    Returns the user pipelines dir."""
    dst = app_data_dir() / "pipelines"
    dst.mkdir(parents=True, exist_ok=True)
    if _BUNDLED_PIPELINES.is_dir():
        for p in _BUNDLED_PIPELINES.glob("*.yaml"):
            target = dst / p.name
            if force or not target.exists():
                try:
                    shutil.copy2(p, target)
                except OSError:
                    pass
    return dst


def pipelines_dir() -> Path:
    """User pipelines dir (`<APP_DATA>/pipelines`), seeded from the bundle
    on first access so the shipped pipelines are present + editable."""
    global _pipelines_seeded
    if not _pipelines_seeded:
        d = seed_pipelines()
        _pipelines_seeded = True
        return d
    d = app_data_dir() / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plugins_dir(kind: str | None = None) -> Path:
    """User drop-in plugin directory.

    ``<APP_DATA>/plugins`` (kind=None), or a typed subdir
    ``<APP_DATA>/plugins/<kind>`` for ``kind in {"processors", "ocr"}``.
    Created on access. See `aglaia/app_data/plugins.py` for the trust gate
    and discovery that read these dirs.
    """
    base = app_data_dir() / "plugins"
    if kind:
        base = base / kind
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    """Downloaded ML weights directory.

    User-configurable via the `models_dir` config key (Settings → Models):
      * empty / unset → `<APP_DATA>/models`
      * relative path → resolved against APP_DATA
      * absolute path → used as-is

    Built-in default is APP_DATA-rooted (not cache) so the user's
    explicitly-downloaded weights don't get wiped by a cache purge.
    """
    base = _models_dir_override()
    if base is None:
        base = app_data_dir() / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _models_dir_override() -> Path | None:
    """Read the user-configured models_dir, if any. Lazy import on the
    config DB to keep `aglaia.app_data` import-free at process start."""
    try:
        from aglaia.app_data import db as _db  # noqa: WPS433
        with _db.session() as conn:
            raw = (_db.get(conn, _db.KEY_MODELS_DIR, "") or "").strip()
    except Exception:
        return None
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = app_data_dir() / p
    return p


def config_db_path() -> Path:
    return app_data_dir() / "aglaia-config.db"


def default_documents_dir() -> Path:
    """User's Documents folder — fallback when `cwd_project` is unset.

    `platformdirs.user_documents_dir` already handles localisation on
    Windows; on Linux it falls back to `$XDG_DOCUMENTS_DIR` or `~/Documents`.
    """
    try:
        return Path(user_documents_dir())
    except Exception:
        return Path.home() / "Documents"
