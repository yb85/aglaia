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
    """Make ``sys.stdout``/``sys.stderr`` safe to write to on Windows, in two
    cases the app's many ``print(...)`` / ``rich.Console()`` calls otherwise
    crash on:

    1. **Windowed PyInstaller build** — no attached console, so the streams are
       ``None``; ``rich.Console()`` (built at GUI import) hits
       ``None.isatty()``. Point them at the null device instead.
    2. **Real console under a non-UTF-8 code page** (the default ``cp1252`` on
       Windows) — startup banners print glyphs like ``✓``/``ï`` that ``cp1252``
       can't encode, raising ``UnicodeEncodeError``. Reconfigure the existing
       stream to UTF-8 (``errors='replace'`` so it can never crash on an odd
       glyph).

    Runs on every spawned worker too, since ``spawn`` re-imports this module
    before running its task."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
        else:
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except (ValueError, OSError):
                    pass


_ensure_std_streams()

from aglaia.app import main


def run() -> int:
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    return main()


if __name__ == "__main__":
    sys.exit(run())
