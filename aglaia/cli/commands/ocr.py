# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia ocr PATHS…` — OCR PDFs/images directly, no geometric processing.

For already-clean documents (born-digital PDFs, flat scans) that don't need the
dewarp/binarize/page-split pipeline. Ingests each page as-is, runs OCR, and
exports — no IntegratedProcessingChain. Pass a `.agl` to re-OCR an existing
project (or with `--check-ocr`, poll its pending Mistral batch jobs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from aglaia.cli.shared import ocr_config


def ocr(
    paths: Annotated[
        list[Path], typer.Argument(help="PDFs, images, or one .agl project to OCR.")
    ],
    ocr: Annotated[
        Optional[str],
        typer.Option(
            "--ocr",
            help="OCR engine: ENGINE[:opt…]. Default 'auto' (Apple Vision → Surya).",
        ),
    ] = None,
    ocr_lang: Annotated[
        str,
        typer.Option(
            "--ocr-lang", help="'+'-joined BCP-47 codes (e.g. fr-FR+en-US) or 'auto'."
        ),
    ] = "auto",
    ocr_dpi: Annotated[
        Optional[int],
        typer.Option(
            "--ocr-dpi",
            help="OCR target DPI — pages are downsampled to this before OCR. "
            "Default 200 (matches the GUI; see docs/ocr-benchmark.md).",
        ),
    ] = None,
    export: Annotated[
        Optional[str],
        typer.Option("--export", help="'+'-joined export specs, e.g. 'pdf:g4+md'."),
    ] = None,
    md_refine: Annotated[
        Optional[str],
        typer.Option(
            "--md-refine",
            help="On-device LLM backend for Markdown cleanup, e.g. 'apple_fm'.",
        ),
    ] = None,
    project_name: Annotated[
        Optional[str],
        typer.Option(
            "--project-name",
            help="Name for a new project (default: from the input filename).",
        ),
    ] = None,
    parent_dir: Annotated[
        Optional[Path],
        typer.Option("--parent-dir", help="Parent folder for a new project."),
    ] = None,
    input_dpi: Annotated[
        Optional[str],
        typer.Option(
            "--input-dpi",
            metavar="[force:]N",
            help="Input DPI for imported images; 'force:N' overrides every input.",
        ),
    ] = None,
    check_ocr: Annotated[
        bool,
        typer.Option(
            "--check-ocr",
            help="Poll + import pending Mistral batch OCR jobs for the project, then exit.",
        ),
    ] = False,
) -> None:
    """OCR documents that don't need processing. Headless (no Qt), no pipeline."""
    if ocr_dpi is not None:
        import os
        os.environ["AGLAIA_OCR_DPI"] = str(int(ocr_dpi))
    cfg = ocr_config(
        paths,
        ocr,
        ocr_lang,
        export,
        md_refine,
        project_name,
        parent_dir,
        input_dpi,
        check_ocr,
    )
    if not cfg.has_inputs():
        typer.echo("ocr: needs one .agl, or one or more PDFs / images.", err=True)
        raise typer.Exit(2)
    from aglaia.app import _run_ocr_only

    raise typer.Exit(_run_ocr_only(cfg))
