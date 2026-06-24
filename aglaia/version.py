# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Single source of truth for the app version, working frozen and from source.

Resolution order (first hit wins):

1. ``AGLAIA_VERSION`` env var — explicit override (CLI / dev).
2. ``aglaia/_version.py`` — baked at PyInstaller build time by ``Aglaia.spec``
   from the same env var (this is what makes the frozen .app/.exe/.AppImage
   report the release tag, since the env var is NOT set at runtime).
3. ``importlib.metadata`` — for ``pip install aglaia``.
4. ``pyproject.toml`` — for a source checkout.
5. ``"0.0.0-dev"`` fallback.

Every version readout (About, Diagnostics, Bug report, the Mistral UA, …) MUST
go through :func:`get_version` — no hardcoded numbers.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_version() -> str:
    # 1. explicit env override
    env = (os.environ.get("AGLAIA_VERSION") or "").lstrip("v").strip()
    if env:
        return env

    # 2. build-time baked module (frozen app)
    try:
        from aglaia._version import __version__ as baked  # type: ignore
        if baked:
            return str(baked).lstrip("v").strip()
    except Exception:
        pass

    # 3. source checkout pyproject (fresh — beats possibly-stale install metadata)
    try:
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except Exception:
        pass

    # 4. installed package metadata (pip install — no pyproject alongside)
    try:
        from importlib.metadata import version
        return version("aglaia")
    except Exception:
        return "0.0.0-dev"
