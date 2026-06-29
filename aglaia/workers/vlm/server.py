# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Local OpenAI-compatible VLM server lifecycle.

Generalised from PaddleOCR-VL's private ``_MlxVlmServer``: lazy spawn, free-port
pick, health-wait, log-tail into the app's log channel, and — critically — a
**process-group SIGKILL** teardown. Spawn children are in their own session
(``start_new_session=True``); a plain ``terminate()`` can leave a native-hung
worker (the backend's XLA/Metal threads) orphaned, so we ``killpg`` the group.

One server per local ``model_path`` (keyed); concurrent engines that share a
model reuse the same process, distinct models get distinct servers (backends
don't reliably hot-swap weights).
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .backends import VlmBackend, pick_backend

# Log sink — defaults to stdout; the OCR layer routes it into the GUI Log tab.
# Signature matches aglaia.workers.ocr.engine.engine_log: ``log(text, level)``.
LogFn = Callable[..., None]


def _default_log(text: str, level: str = "info") -> None:
    print(text, flush=True)


_LOG: LogFn = _default_log


def set_log_sink(fn: Optional[LogFn]) -> None:
    """Route server diagnostics through ``fn`` (e.g. the OCR engine_log) so the
    GUI Log tab sees mlx/vllm output. ``None`` restores stdout."""
    global _LOG
    _LOG = fn or _default_log


def _free_port(start: int = 8111) -> int:
    p = start
    while p < start + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    raise RuntimeError(f"no free port in {start}..{start + 100}")


def _log_dir() -> Path:
    try:
        from aglaia.app_data import app_data_dir

        d = app_data_dir() / "logs"
    except Exception:
        d = Path.home() / ".cache" / "aglaia" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


class LocalVlmServer:
    """A spawned OpenAI-compatible server for one local model. Use the classmethod
    ``ensure`` — instances are cached per ``model_path``."""

    _servers: dict[str, "LocalVlmServer"] = {}

    def __init__(self, model_path: str, backend: VlmBackend) -> None:
        self.model_path = model_path
        self.backend = backend
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.log_path: Optional[Path] = None
        self._tail_thread: Optional[threading.Thread] = None
        self._tail_stop: Optional[threading.Event] = None

    # ── public, process-wide API ─────────────────────────────────────
    @classmethod
    def ensure(
        cls,
        model_path: str | Path,
        *,
        backend: Optional[VlmBackend] = None,
        log: Optional[LogFn] = None,
        max_tokens: int = 4096,
        log_stem: str = "vlm",
        health_timeout: float = 240.0,
    ) -> str:
        """Return the base URL (``http://127.0.0.1:<port>/``) of a running server
        for ``model_path``, spawning it on first use. Picks a backend if none is
        given; raises if no backend is available or the server fails to come up."""
        if log is not None:
            set_log_sink(log)
        key = str(model_path)
        srv = cls._servers.get(key)
        if srv is not None and srv.proc is not None and srv.proc.poll() is None:
            return srv._base_url()

        be = backend or pick_backend()
        if be is None:
            raise RuntimeError(
                "No local VLM backend available — install `mlx-vlm` (Apple "
                "Silicon) or `vllm` (CUDA), or set AGLAIA_VLM_BACKEND."
            )
        srv = cls(key, be)
        srv._spawn(
            max_tokens=max_tokens, log_stem=log_stem, health_timeout=health_timeout
        )
        cls._servers[key] = srv
        return srv._base_url()

    @classmethod
    def served_model_name(
        cls, model_path: str | Path, backend: Optional[VlmBackend] = None
    ) -> str:
        """The id a client passes as ``model`` for ``model_path`` under the
        active (or given) backend."""
        be = backend or pick_backend()
        if be is None:
            return str(model_path)
        return be.served_model_name(str(model_path))

    @classmethod
    def stop(cls, model_path: str | Path) -> None:
        srv = cls._servers.pop(str(model_path), None)
        if srv is not None:
            srv._terminate()

    @classmethod
    def stop_all(cls) -> None:
        for srv in list(cls._servers.values()):
            srv._terminate()
        cls._servers.clear()

    # ── lifecycle internals ──────────────────────────────────────────
    def _base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def _spawn(self, *, max_tokens: int, log_stem: str, health_timeout: float) -> None:
        d = Path(self.model_path)
        if not d.exists():
            raise RuntimeError(
                f"weights path missing: {self.model_path} — download it first."
            )
        self.port = _free_port()
        self.log_path = _log_dir() / f"{log_stem}_{self.backend.name}_server.log"

        # WARNING is the right default — DEBUG floods a line per token and stalls
        # the server event loop. AGLAIA_MLX_VLM_LOG_LEVEL overrides for one-shot
        # debugging (kept that env name for back-compat with the paddle path).
        log_level = os.environ.get("AGLAIA_MLX_VLM_LOG_LEVEL", "WARNING").upper()
        cmd = self.backend.spawn_cmd(
            self.model_path, self.port, max_tokens=max_tokens, log_level=log_level
        )
        proc_env = os.environ.copy()
        proc_env.update(self.backend.env())
        _LOG(f"[vlm/{self.backend.name}] spawn: {' '.join(cmd)}")
        log_fp = open(self.log_path, "wb")
        # start_new_session=True → own process group, so teardown can killpg the
        # whole tree (the backend's native threads don't reparent-and-survive).
        self.proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=proc_env,
        )
        self._start_log_tail()
        atexit.register(self._terminate)

        if not self._wait_health(health_timeout):
            self._dump_tail()
            self._terminate()
            raise RuntimeError(
                f"{self.backend.name} server failed to come up — see {self.log_path}"
            )

    def _wait_health(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
                    if r.status == 200:
                        return True
            except Exception:
                pass
            if self.proc is not None and self.proc.poll() is not None:
                return False  # died during startup — stop waiting
            time.sleep(1)
        return False

    def _dump_tail(self) -> None:
        """Forward the last lines of the server log to the app log channel so a
        startup crash is visible even when stdout is /dev/null (.app bundle)."""
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "rb") as fp:
                fp.seek(0, 2)
                fp.seek(max(0, fp.tell() - 16384))
                tail = fp.read().decode("utf-8", errors="replace")
            for line in tail.splitlines()[-40:]:
                _LOG(f"[vlm/{self.backend.name}] {line}", level="error")
        except Exception:
            pass

    def _terminate(self) -> None:
        if self._tail_stop is not None:
            self._tail_stop.set()
        self._tail_thread = None
        self._tail_stop = None
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM the whole group
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()  # SIGKILL fallback
            except Exception:
                pass

    def _start_log_tail(self) -> None:
        """Daemon thread that forwards new server-log lines through the sink.
        Polls at 200 ms (no `tail -f` dep); exits when the proc dies or we stop."""
        if self.log_path is None:
            return
        stop_evt = threading.Event()
        log_path = self.log_path
        tag = f"[{self.backend.name}]"

        def _tail() -> None:
            try:
                while not log_path.exists():
                    if stop_evt.wait(0.1):
                        return
                with open(log_path, "rb") as fp:
                    buf = b""
                    while not stop_evt.is_set():
                        chunk = fp.read(65536)
                        if not chunk:
                            if self.proc is not None and self.proc.poll() is not None:
                                chunk = fp.read(65536)
                                if chunk:
                                    buf += chunk
                                break
                            time.sleep(0.2)
                            continue
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            txt = line.decode("utf-8", errors="replace").rstrip()
                            if not txt:
                                continue
                            lvl = (
                                "error"
                                if (
                                    "ERROR" in txt
                                    or "Traceback" in txt
                                    or "Exception" in txt
                                )
                                else "warn"
                                if ("WARNING" in txt or "Warning" in txt)
                                else "info"
                            )
                            _LOG(f"{tag} {txt}", level=lvl)
                    if buf:
                        txt = buf.decode("utf-8", errors="replace").rstrip()
                        if txt:
                            _LOG(f"{tag} {txt}")
            except Exception:
                pass  # the tailer must never take down inference

        t = threading.Thread(
            target=_tail, daemon=True, name=f"vlm-{self.backend.name}-tail"
        )
        t.start()
        self._tail_thread = t
        self._tail_stop = stop_evt
