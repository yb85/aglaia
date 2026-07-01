# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Back-compat shim over the download registry (`aglaia.app_data.downloads`).

The model catalogue used to live in a static ``model-list.json`` loaded here.
It now lives in Python: core assets register themselves in ``downloads.py`` and
plugins register theirs at import. This module keeps the old public names
(``ModelSpec`` / ``MODEL_SPECS`` / ``is_model_installed`` / ``download_model`` /
``spec_for`` / ``_load_model_specs``) pointing at the registry so the GUI
(``ModelDownloaderTab``), the onboarding wizard, and ``aglaia setup`` need no
change. New code should import from ``aglaia.app_data.downloads`` directly.
"""

from __future__ import annotations

from typing import Optional

from aglaia.app_data.downloads import (
    DownloadTarget,
    ProgressCb,
    download_model,
    is_downloaded,
    registry,
    target_for,
)

# Legacy aliases. ``ModelSpec`` was the old descriptor dataclass — now the same
# thing as ``DownloadTarget`` (identical field names).
ModelSpec = DownloadTarget

__all__ = [
    "ModelSpec",
    "MODEL_SPECS",
    "ProgressCb",
    "download_model",
    "is_model_installed",
    "spec_for",
    "_load_model_specs",
]


def _load_model_specs() -> list[DownloadTarget]:
    """Legacy name for the catalogue. The registry is rebuilt at import, so
    this just returns the current set (core + any registered plugins)."""
    return registry()


# Module-level snapshot kept for callers that read it directly. The GUI also
# calls ``_load_model_specs()`` per open, so plugin-registered targets appear
# without a restart.
MODEL_SPECS: list[DownloadTarget] = registry()


def spec_for(key: str) -> Optional[DownloadTarget]:
    return target_for(key)


def is_model_installed(key: str) -> bool:
    """Lightweight on-disk presence check (no Qt, no DB)."""
    return is_downloaded(key)
