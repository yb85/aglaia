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
import sys

from aglaia.app import main


def run() -> int:
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    return main()


if __name__ == "__main__":
    sys.exit(run())
