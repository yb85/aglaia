# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia gui [PROJECT]` — launch the capture GUI (the default command)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from aglaia.cli.shared import ForceProcOpt, PipelineOpt, WorkersOpt, gui_config


def gui(
    project: Annotated[Optional[Path], typer.Argument(help="A .agl project to open (optional).")] = None,
    pipeline: PipelineOpt = None,
    workers: WorkersOpt = None,
    force_proc: ForceProcOpt = False,
    camera_id: Annotated[Optional[int], typer.Option("--camera-id", help="Capture camera index.")] = None,
    diagnose_memory: Annotated[bool, typer.Option("--diagnose-memory", help="tracemalloc snapshots in the GUI process.")] = False,
) -> None:
    """Launch the capture GUI. With no PROJECT it opens the start window; with a
    `.agl`/PDF/image it opens or ingests that. Falls back to headless if Qt isn't
    installed."""
    cfg = gui_config(project, pipeline, workers, force_proc, camera_id, diagnose_memory)
    from aglaia.app import launch_gui
    raise typer.Exit(launch_gui(cfg))
