# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Run one export-plugin call off the GUI thread.

`Destination.send` and `Destination.check` are network calls, and unlike the
registry fetch their duration is not merely slow — it is unbounded and not
ours. Mailing a 45 MB PDF to a Kindle opens an SMTP session with a 60 s
timeout and then uploads over the user's link; a calibre server on a laptop
that has gone to sleep answers when it answers. Running that on the GUI thread
is a beach ball for however long the other end takes, with no way to tell a
slow send from a hung one.

The thread is deliberately thin: it calls, and it reports. Everything that
touches widgets, files or state stays on the GUI thread in the `done` handler,
because the caller is what knows whether its window still exists.

Two rules matter more than they look:

* **The signal must fire whatever happens.** An exception escaping `run()`
  leaves the GUI sitting at "Sending…" forever with a disabled button — a
  worse failure than the one that caused it, because it looks like a hang and
  it never resolves. So `run` catches everything, including `BaseException`:
  this is plugin code, and a plugin that calls `sys.exit()` in a worker thread
  must not take the button down with it.
* **The caller must survive the answer.** A dialog can be closed while its
  test is in flight; the handler then runs against a deleted C++ object.
  Callers guard with a liveness flag — see `PluginSettingsDialog`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class Outcome:
    """What the call produced, or why it produced nothing.

    Both halves are carried rather than raised, because the receiving end is a
    Qt slot: there is nothing there to catch an exception."""

    result: Optional[Any] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and getattr(self.result, "ok", False)

    @property
    def message(self) -> str:
        """One line to show the user, whichever way it went."""
        if self.error:
            return self.error
        return str(getattr(self.result, "message", "") or "")


class DestinationJob(QThread):
    """Call one plugin method in the background and emit its `Outcome`."""

    done = Signal(object)

    def __init__(self, call: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._call = call

    def run(self) -> None:
        try:
            self.done.emit(Outcome(self._call()))
        except BaseException as e:  # noqa: BLE001 — see the module docstring
            self.done.emit(Outcome(None, f"{type(e).__name__}: {e}"))
