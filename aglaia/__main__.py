# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Process entry point for `python -m aglaia`, the `aglaia` console script, and
the PyInstaller-frozen app — all route through :func:`run`.

``run`` does the multiprocessing wiring BEFORE anything spawns a worker:
``freeze_support`` (so a frozen worker re-running this module executes the
worker task and exits instead of popping a second GUI — the fork-bomb symptom)
and ``set_start_method('spawn')``."""

import multiprocessing
import os
import sys


def _ensure_std_streams() -> None:
    """In a windowed PyInstaller build there is no attached console, so
    ``sys.stdout``/``sys.stderr`` are ``None``. ``rich.Console()`` (built at
    import time in the GUI) and the app's many ``print(..., file=sys.stderr)``
    calls then crash on ``None.isatty()``. Point the missing streams at the
    null device so writes are harmless no-ops. Runs on every spawned worker
    too, since ``spawn`` re-imports this module before running its task."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


_ensure_std_streams()

from aglaia.app import main


def run() -> int:
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    return main()


if __name__ == "__main__":
    sys.exit(run())
