# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia setup` — interactive first-run setup (CLI-only installs)."""

from __future__ import annotations

import typer


def setup() -> None:
    """Interactive first-run setup: language, models, defaults."""
    from aglaia.workers.setup_cli import run_setup
    raise typer.Exit(run_setup())
