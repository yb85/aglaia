# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia list pipelines|ocr|exports` — introspection."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer


class ListKind(str, Enum):
    pipelines = "pipelines"
    ocr = "ocr"
    exports = "exports"


def list_(kind: Annotated[ListKind, typer.Argument(help="What to list.")]) -> None:
    """List available pipelines, OCR engines, or export formats."""
    from aglaia.workers.cli import CliConfig, run_list_commands

    cfg = CliConfig(
        list_pipelines=kind is ListKind.pipelines,
        list_ocr=kind is ListKind.ocr,
        list_exports=kind is ListKind.exports,
    )
    run_list_commands(cfg)
    raise typer.Exit(0)
