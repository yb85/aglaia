# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""`aglaia skill` — print the agent skill to stdout.

The skill (`aglaia/assets/SKILL.md`) is the document an AI agent reads to drive
this CLI well: what each command is for, which to use in which circumstance,
the questions to ask a user before choosing, and the mistakes that cost the
most. Printing it from the installed binary means an agent on any machine gets
the version that matches the commands it will run.
"""

from __future__ import annotations

import sys

import typer


def skill() -> None:
    """Print the agent skill (how to drive this CLI) to stdout."""
    from aglaia.assets import asset_path
    p = asset_path("SKILL.md")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        typer.echo(f"The skill file is missing from this install ({e}).", err=True)
        raise typer.Exit(1)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
