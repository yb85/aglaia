# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""One way for any GUI widget to reach the Log tab.

`MainWindow._on_log_line` is the sink, but a dialog several layers down has
no handle on the window — so the diagnostics that matter most (a plugin's
refusal, with the server's answer in it) were written with `print()`. In a
bundled app stdout is /dev/null, and even from a terminal the user is told
to "see the Log tab" and finds nothing there.

Same shape as `ocr.engine.set_engine_log_sink`: the host installs a sink at
startup, everyone else calls `log()`. With no sink — tests, a headless
import — the line still goes to stdout rather than vanishing.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

_sink: Optional[Callable[[str, str], None]] = None


def set_sink(fn: Optional[Callable[[str, str], None]]) -> None:
    """Route every `log()` call to `fn(level, text)`. `None` restores stdout."""
    global _sink
    _sink = fn


def log(level: str, text: str) -> None:
    """One line for the Log tab. Never raises: its callers are error paths,
    and a logging call that throws replaces a handled failure with an
    unhandled one."""
    fn = _sink
    if fn is not None:
        try:
            fn(level, text)
            return
        except Exception:
            pass
    stream = sys.stderr if level in ("error", "warning") else sys.stdout
    try:
        print(text, file=stream)
    except Exception:
        pass
