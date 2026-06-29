# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Shared CLI options and the CliConfig builders.

The annotated option aliases here are reused across commands that legitimately
share a flag (`--pipeline`/`--workers`/`--force-proc` on `gui` and `run`) — the
"identical behaviour, shared parsing code" the subcommand design allows
(docs/subcommand-cli.md). The builders translate a command's parsed params into
the existing :class:`aglaia.workers.cli.CliConfig`, reusing its spec parsers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

# Shared across `gui` and `run`.
PipelineOpt = Annotated[
    Optional[str],
    typer.Option("--pipeline", "-p", help="Pipeline name (e.g. 'book_curved_x2') or a .yaml path."),
]
WorkersOpt = Annotated[
    Optional[int],
    typer.Option("--workers", help="Pipeline worker processes (overrides config). 0 = auto."),
]
ForceProcOpt = Annotated[
    bool,
    typer.Option("--force-proc", help="Reprocess every active scan on open (wipe branches/intermediates)."),
]


def gui_config(
    project: Optional[Path],
    pipeline: Optional[str],
    workers: Optional[int],
    force_proc: bool,
    camera_id: Optional[int],
    diagnose_memory: bool,
) -> "object":
    from aglaia.workers.cli import CliConfig, classify_inputs

    cfg = CliConfig(
        paths=[Path(project).expanduser()] if project else [],
        pipeline=pipeline,
        workers=workers,
        force_proc=force_proc,
        camera_id=camera_id,
        diagnose_memory=diagnose_memory,
    )
    classify_inputs(cfg)
    return cfg


def run_config(
    paths: list[Path],
    pipeline: Optional[str],
    workers: Optional[int],
    force_proc: bool,
    ocr: Optional[str],
    ocr_lang: str,
    export: Optional[str],
    md_refine: Optional[str],
    project_name: Optional[str],
    parent_dir: Optional[Path],
    input_dpi: Optional[str],
    check_ocr: bool,
) -> "object":
    from aglaia.workers.cli import (
        CliConfig, build_ocr_fields, classify_inputs,
        _parse_export_arg, _parse_input_dpi,
    )

    dpi, dpi_force = _parse_input_dpi(input_dpi)
    cfg = CliConfig(
        paths=[Path(p).expanduser() for p in paths],
        pipeline=pipeline,
        workers=workers,
        force_proc=force_proc,
        input_dpi=dpi,
        input_dpi_force=dpi_force,
        exports=_parse_export_arg(export),
        md_refine=md_refine,
        project_name=project_name,
        parent_dir=Path(parent_dir).expanduser() if parent_dir else None,
        check_ocr=check_ocr,
        headless=True,
        **build_ocr_fields(ocr, ocr_lang),
    )
    classify_inputs(cfg)
    return cfg
