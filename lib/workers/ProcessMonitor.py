# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import re
from collections import deque
from queue import Empty
from PySide6.QtCore import QThread, Signal
from rich.console import Console

console = Console()


# `[RSS-poll] gui_pid=97338=774MB/25t | Worker-Integrated-0_pid=97883=281MB/38t | ...`
# Capture worker name (gui or Worker-Integrated-N) + MB. Threads count
# discarded (UI doesn't surface it for now).
_RSS_POLL_RE = re.compile(
    r"(gui|Worker-Integrated-\d+)(?:_pid=\d+)?=(-?\d+(?:\.\d+)?)MB"
)

# Strip the per-stage RSS suffix — the status bar already displays it
# live per worker, the value bolted onto every timing log was redundant
# noise. Matches " RSS=123 MB" at end of line.
_STAGE_RSS_SUFFIX_RE = re.compile(r"\s+RSS=\s*-?\d+(?:\.\d+)?\s*MB\s*$")


class ProcessMonitor(QThread):
    # Dict payload: scan_id, node_id, parent_node_id, image_id, event_type, filestem, depth, meta, ...
    image_event_signal = Signal(dict)
    branch_ready_signal = Signal(dict)
    snap_imported_signal = Signal(dict)
    worker_started_signal = Signal()
    # Parsed `[RSS-poll]` lines, fed to the status bar's `WorkerRssStrip`.
    rss_signal = Signal(dict)  # {"gui": float_mb, "Worker-Integrated-0": float_mb, ...}
    # Every log line (level, text). MainWindow shows the latest in the
    # status bar's `LogStrip` and buffers full history for the log tab.
    log_signal = Signal(str, str)
    # Per-step wall-clock timing: (proc_name, elapsed_ms, success).
    # Sidebar's PipelineTimingView consumes it to maintain p5/median/p95.
    timing_signal = Signal(str, float, bool)

    # Cap on the rolling log buffer fed to the log tab. Older lines drop.
    LOG_BUFFER_MAX = 5000

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.running = True
        # Plain in-monitor buffer — MainWindow asks for a snapshot when
        # the user clicks the status-bar log strip. Survives across tab
        # close/reopen.
        self.log_buffer: deque[str] = deque(maxlen=self.LOG_BUFFER_MAX)
        
    def run(self):
        while self.running:
            try:
                # Use blocking get with timeout to prevent busy-waiting
                msg = self.log_queue.get(block=True, timeout=0.1)
                
                if msg is None: break
                
                if msg[0] == 'worker_started':
                    self.worker_started_signal.emit()
                elif msg[0] == 'log_info':
                    text = str(msg[1])
                    # Re-route [RSS-poll] noise into a structured signal
                    # the status bar consumes — don't spam the terminal
                    # or the log tab with one line every 5 s per worker.
                    if text.startswith("[RSS-poll]"):
                        values: dict[str, float] = {}
                        for name, mb in _RSS_POLL_RE.findall(text):
                            try:
                                values[name] = float(mb)
                            except ValueError:
                                continue
                        if values:
                            self.rss_signal.emit(values)
                        # Skip console + log_buffer for these.
                    else:
                        # Drop the trailing " RSS=… MB" — UI already
                        # displays per-worker memory live.
                        text = _STAGE_RSS_SUFFIX_RE.sub("", text)
                        from lib.workers.oplog import strip_markup
                        # Colour goes to the terminal; Qt status-bar
                        # labels render plain text only.
                        console.print(rf"[cyan]\[INFO][/] {text}")
                        plain = strip_markup(text)
                        line = f"[INFO] {plain}"
                        self.log_buffer.append(line)
                        self.log_signal.emit("info", plain)
                elif msg[0] == 'log_warning':
                    text = str(msg[1])
                    from lib.workers.oplog import strip_markup
                    console.print(rf"[yellow]\[WARN][/] {text}")
                    plain = strip_markup(text)
                    line = f"[WARN] {plain}"
                    self.log_buffer.append(line)
                    self.log_signal.emit("warn", plain)
                elif msg[0] == 'error':
                    text = str(msg[1])
                    from lib.workers.oplog import strip_markup
                    console.print(rf"[red]\[ERROR][/] {text}")
                    plain = strip_markup(text)
                    line = f"[ERROR] {plain}"
                    self.log_buffer.append(line)
                    self.log_signal.emit("error", plain)
                elif msg[0] == 'progress':
                    pass # Ignore granular progress for now
                elif msg[0] == 'image_event':
                    # msg format (M0): ('image_event', payload_dict)
                    payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
                    self.image_event_signal.emit(payload)
                elif msg[0] == 'branch_ready':
                    payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
                    self.branch_ready_signal.emit(payload)
                elif msg[0] == 'scan_imported':
                    payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
                    self.snap_imported_signal.emit(payload)
                elif msg[0] == 'timing':
                    # msg format: ('timing', stem, dims, dpi, proc_name, ms, success)
                    stem, dims, dpi, proc, ms, success = msg[1], msg[2], msg[3], msg[4], msg[5], msg[6]
                    self.timing_signal.emit(str(proc), float(ms), bool(success))
                    emoji = "[green]●[/]" if success else "[red]●[/]"
                    
                    # Pretty tabulation using fixed-width fields or rich columns
                    # format: Emoji Stem (WxH@dpi) Proc MS
                    line = f"{emoji} [bold white]{stem:<15}[/] "
                    line += f"[dim]({dims}@{dpi:.0f})[/] "
                    line += f"[cyan]{proc:<15}[/] "
                    line += f"[bold yellow]{ms:6.1f}ms[/]"
                    console.print(line)
            except Empty:
                continue
            except Exception as e:
                console.print(rf"[red]\[ERROR][/] ProcessMonitor Error: {e}")
                break

    def stop(self):
        self.running = False
        self.wait()
