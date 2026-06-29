# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia run PATHS…` — headless batch: ingest → pipeline → (ocr) → (export)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from aglaia.cli.shared import ForceProcOpt, PipelineOpt, WorkersOpt, run_config


def run(
    paths: Annotated[list[Path], typer.Argument(help="Images, PDFs, or one .agl project to (re)process.")],
    pipeline: PipelineOpt = None,
    workers: WorkersOpt = None,
    force_proc: ForceProcOpt = False,
    ocr: Annotated[Optional[str], typer.Option("--ocr", help="Run OCR: ENGINE[:opt…]. Use '--ocr auto' for the default (Apple Vision → Surya).")] = None,
    ocr_lang: Annotated[str, typer.Option("--ocr-lang", help="'+'-joined BCP-47 codes (e.g. fr-FR+en-US) or 'auto'.")] = "auto",
    export: Annotated[Optional[str], typer.Option("--export", help="'+'-joined export specs, e.g. 'pdf:g4+md'.")] = None,
    md_refine: Annotated[Optional[str], typer.Option("--md-refine", help="On-device LLM backend for Markdown cleanup, e.g. 'apple_fm'.")] = None,
    project_name: Annotated[Optional[str], typer.Option("--project-name", help="Name for a new project (default: from the input filename).")] = None,
    parent_dir: Annotated[Optional[Path], typer.Option("--parent-dir", help="Parent folder for a new project.")] = None,
    input_dpi: Annotated[Optional[str], typer.Option("--input-dpi", metavar="[force:]N", help="Input DPI for imported images; 'force:N' overrides every input.")] = None,
    check_ocr: Annotated[bool, typer.Option("--check-ocr", help="Poll + import pending Mistral batch OCR jobs for the project, then exit.")] = False,
) -> None:
    """Run the full pipeline headlessly (no Qt). `run` is always headless — there
    is no `--headless` flag."""
    cfg = run_config(
        paths, pipeline, workers, force_proc, ocr, ocr_lang, export,
        md_refine, project_name, parent_dir, input_dpi, check_ocr,
    )
    if not cfg.has_inputs():
        typer.echo("run: needs one .agl, or one or more PDFs / images.", err=True)
        raise typer.Exit(2)
    from aglaia.app import _run_headless
    raise typer.Exit(_run_headless(cfg))
