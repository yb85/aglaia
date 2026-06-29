# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia version` — print the version (also available as `--version`)."""

from __future__ import annotations

import typer


def version() -> None:
    """Print the Aglaïa version and exit."""
    from aglaia.version import get_version
    typer.echo(f"Aglaïa {get_version()}")
    raise typer.Exit(0)
