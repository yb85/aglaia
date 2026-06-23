# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Resolver for the top-level ``assets/`` directory (icons, brand images, app
icns, mode previews, calibration boards).

The assets live OUTSIDE the ``lib`` package, so they can't be found by a
package-relative ``__file__`` walk the way they could when they sat under
``lib/``. This resolver handles both layouts:

* **From source** — ``<repo>/assets`` (this module is ``lib/assets.py``, so the
  repo root is ``parents[1]``).
* **Frozen (PyInstaller)** — ``<sys._MEIPASS>/assets``. The bundle MUST ship
  the directory there; ``Aglaia.spec`` adds ``(REPO/'assets', 'assets')`` to
  ``datas``. If that mapping ever drifts, icon/logo loading silently fails in
  the .app while working fine from source — verify on the next DMG build.
"""

from __future__ import annotations

import sys
from pathlib import Path


def assets_root() -> Path:
    """Absolute path to the ``assets/`` directory (source or bundle)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "assets"
    return Path(__file__).resolve().parents[1] / "assets"


def asset_path(*parts: str) -> Path:
    """``assets_root()`` joined with ``parts``, e.g. ``asset_path('icons',
    'ruler.svg')`` or ``asset_path('brand', 'aglaia-light.png')``."""
    return assets_root().joinpath(*parts)
