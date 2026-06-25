# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Integrated processing chain backed by SQLite project storage.

Each worker owns its SQLite connection (journal_mode=DELETE — single
project file, no WAL sidecars); writes serialize behind SQLite's global
write lock with busy_timeout retries. Workers persist directly via
`Persister`, one transaction per step.

Per-step semantics:
  - Each step emits N ImageBuffers (1 for linear, >1 for PageDetector splits).
  - Each output is a `nodes` row whose parent_id is the previous step's node
    (or the scan's root node for the first step).
  - >1 outputs mark the parent as `is_branch_point=1`; each child re-enters at
    `step_idx + 1`.
  - On pipeline completion (or processor returns None), the terminal node is
    registered in `branches` for that scan+branch_path.

Image dedup via sha256 bounds storage cost across no-op intermediate steps.
"""

import json
import multiprocessing
import queue
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Type

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.abstraction import AbstractImageProcessor, ReplayTrait
from aglaia.workers.chain_abstraction import SimpleChainElement

# Auto-discovered registry — adding a processor only needs its file in `aglaia/processors/`.
from aglaia.processors import registry as _proc_registry


def processor_registry() -> Dict[str, Type[AbstractImageProcessor]]:
    """Module-level shortcut to the auto-discovered registry. The
    underlying discovery is cached, so this is a cheap dict lookup."""
    return _proc_registry.processor_classes()


def _clean_meta(meta: Optional[dict]) -> Optional[dict]:
    """Make meta JSON-serializable (drop numpy types, tuple→list)."""
    if not meta:
        return None
    out = {}
    for k, v in meta.items():
        out[k] = _jsonable(v)
    return out


def _jsonable(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            pass
    return str(v)


def _gui_meta(meta: Optional[dict]) -> dict:
    """JSON-cleaned meta for GUI/SSE consumption."""
    return _clean_meta(meta) or {}


class IntegratedProcessingChain:
    """
    Single worker pool, each worker holds the full pipeline.

    DB-backed (no WriterProcess). Workers open their own SQLite connection
    (journal_mode=DELETE; single writer at a time, busy_timeout retries).
    """

    def __init__(self, elements: List[SimpleChainElement], num_workers: int,
                 log_queue: multiprocessing.Queue, db_path: str,
                 queue_factory=multiprocessing.Queue, paths: Optional[dict] = None,
                 replay_enabled: bool = True):
        self.elements = elements
        self.num_workers = max(1, int(num_workers))
        self.log_queue = log_queue
        self.db_path = str(db_path)
        self.queue_factory = queue_factory
        self.paths = paths or {}
        # Pipeline-level switch: when False, the replay pass at end-of-leaf
        # is skipped entirely (no node persisted, no progress slot). When
        # True the replay still self-suppresses if no element produced a
        # `replay_kind` stamp on the buffer trail.
        self.replay_enabled = bool(replay_enabled)
        self.input_queue = None       # external, bounded — initial scans
        self.routed_queue = None      # internal, unbounded — branch routes
        self.workers: list[multiprocessing.Process] = []
        self.running = False
        # Bound initial-scan backlog so mass imports stay finite. Routed
        # branches use an unbounded queue so workers never block each other.
        self.max_input_pending = max(4, 4 * self.num_workers)

        # In-flight tracker — worker stores its current item before processing.
        # When the watchdog SIGKILLs for OOM, the in-flight entry is requeued.
        self._mgr = multiprocessing.Manager()
        self._inflight = self._mgr.dict()
        # Retry count per (scan_id, start_node_idx) caps the kill→requeue loop.
        self._retries = self._mgr.dict()

        registry = processor_registry()
        for el in elements:
            if el.processor_name not in registry:
                raise ValueError(f"Unknown processor: {el.processor_name}")

        self._build_chain()

    def _build_chain(self):
        # Bounded input queue — put() blocks naturally where qsize() is N/A.
        try:
            self.input_queue = self.queue_factory(maxsize=self.max_input_pending)
        except TypeError:
            self.input_queue = self.queue_factory()
        # Routed queue is Manager.Queue (not raw mp.Queue): a SIGKILLed worker
        # would leak the POSIX semaphore backing mp.Queue, jamming every
        # sibling's get_nowait() forever. Manager.Queue lives in the manager
        # process — worker death only drops the RPC connection.
        self.routed_queue = self._mgr.Queue()
        for w_idx in range(self.num_workers):
            p = multiprocessing.Process(
                target=IntegratedProcessingChain._worker_loop,
                args=(self.elements, self.input_queue, self.routed_queue,
                      self.log_queue, self.db_path, self._inflight,
                      self.replay_enabled),
                name=f"Worker-Integrated-{w_idx}",
                daemon=True,
            )
            self.workers.append(p)

    def start(self):
        self.running = True
        for p in self.workers:
            p.start()
        if self.log_queue is not None:
            import os as _os
            pid_list = ", ".join(f"{p.name}={p.pid}" for p in self.workers)
            self.log_queue.put(("log_info",
                f"IntegratedProcessingChain (DB-backed) started with "
                f"{len(self.workers)} workers. gui_pid={_os.getpid()} | {pid_list}"))

        # Respawn workers that self-exited on RSS budget. Without it the
        # chain silently bleeds capacity over long sessions.
        import threading
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="ChainWatchdog",
        )
        self._watchdog_thread.start()

        # Sampler observes growth even when a worker is hung mid-call.
        self._rss_sampler_thread = threading.Thread(
            target=self._rss_sampler_loop, daemon=True, name="ChainRssSampler",
        )
        self._rss_sampler_thread.start()

    def _rss_sampler_loop(self):
        """Sample 'phys_footprint' (Activity Monitor 'Memory') via vmmap.
        psutil RSS excludes compressed + IOKit-mapped memory and can underreport
        by 30×; we shell out to `vmmap -summary` and parse 'Physical footprint'."""
        try:
            import psutil
        except Exception:
            psutil = None
        import os as _os

        from aglaia.workers.macos_mem import phys_footprint_mb, thread_count

        while not self._watchdog_stop.wait(5.0):
            if not self.running:
                return
            gui_pid = _os.getpid()
            gui_mb = phys_footprint_mb(gui_pid)
            gui_th = thread_count(gui_pid)
            parts = [f"gui_pid={gui_pid}={gui_mb:.0f}MB/{gui_th}t"]
            for p in list(self.workers):
                if not p.is_alive():
                    parts.append(f"{p.name}=dead")
                    continue
                mb = phys_footprint_mb(p.pid)
                th = thread_count(p.pid)
                parts.append(f"{p.name}_pid={p.pid}={mb:.0f}MB/{th}t")
            line = "[RSS-poll] " + " | ".join(parts)
            if self.log_queue is not None:
                self.log_queue.put(("log_info", line))
            # In GUI mode the log_queue is drained into Qt signals, never
            # stdout — so the bench harness can't see memory. Under
            # AGLAIA_TEST, echo straight to stdout so both GUI and headless
            # expose the same parseable [RSS-poll] stream.
            if _os.environ.get("AGLAIA_TEST"):
                print(line, flush=True)

    def _watchdog_loop(self):
        """Parent-side supervision:
          1. Respawn exited workers.
          2. SIGKILL workers whose phys_footprint (vmmap) exceeds the budget.
        Uses vmmap because psutil RSS underreports actual footprint on Apple Silicon."""
        import os as _os

        from aglaia.workers.macos_mem import phys_footprint_mb

        cap_mb = float(_os.environ.get("AGLAIA_WORKER_CAP_MB", "3072"))
        while not self._watchdog_stop.wait(2.0):
            if not self.running:
                return
            for idx, p in enumerate(list(self.workers)):
                if not p.is_alive():
                    exitcode = p.exitcode
                    # Re-enqueue the worker's in-flight item if any. Covers
                    # both clean exits with a scan mid-flight (rare) and
                    # SIGKILL paths where the watchdog killed for OOM.
                    self._reenqueue_inflight(p.name)
                    if self.log_queue is not None:
                        self.log_queue.put((
                            "log_info",
                            f"Worker {p.name} exited (code={exitcode}); respawning.",
                        ))
                    # Shutdown race: stop() may have terminated this worker
                    # after our `running` check above — don't respawn into
                    # closed queue handles.
                    if self._watchdog_stop.is_set() or not self.running:
                        return
                    new_p = multiprocessing.Process(
                        target=IntegratedProcessingChain._worker_loop,
                        args=(self.elements, self.input_queue,
                              self.routed_queue, self.log_queue, self.db_path,
                              self._inflight, self.replay_enabled),
                        name=p.name,
                        daemon=True,
                    )
                    try:
                        new_p.start()
                    except (OSError, ValueError):
                        # Queue fds already closed mid-shutdown.
                        return
                    self.workers[idx] = new_p
                    continue
                mb = phys_footprint_mb(p.pid)
                if mb < 0:
                    continue
                if mb > cap_mb:
                    if self.log_queue is not None:
                        self.log_queue.put((
                            "log_warning",
                            f"Worker {p.name} pid={p.pid} phys_footprint="
                            f"{mb:.0f} MB > {cap_mb:.0f} MB cap — SIGKILL.",
                        ))
                    # Re-enqueue before kill to survive lifecycle races on respawn.
                    self._reenqueue_inflight(p.name)
                    try:
                        p.kill()
                    except Exception:
                        try:
                            p.terminate()
                        except Exception:
                            pass

    def _reenqueue_inflight(self, worker_name: str) -> None:
        """Push the killed worker's in-flight item back so no scan is lost.
        The in-flight entry is a small DB reference (node_id + start_idx);
        the consuming worker re-decodes the image from the project DB.
        Caps retries to break infinite kill/retry on a chronically over-budget scan."""
        try:
            entry = self._inflight.pop(worker_name, None)
        except Exception:
            entry = None
        if entry is None:
            return
        try:
            import pickle as _pickle
            ref = _pickle.loads(entry)
            start_idx = int(ref["start_idx"])
        except Exception as e:
            if self.log_queue is not None:
                self.log_queue.put(("log_warning",
                    f"reenqueue: failed to unpickle in-flight item for "
                    f"{worker_name}: {e}"))
            return
        key = (ref.get("scan_id"), start_idx)
        try:
            n = int(self._retries.get(key, 0))
        except Exception:
            n = 0
        if n >= 3:
            if self.log_queue is not None:
                self.log_queue.put(("log_warning",
                    f"reenqueue: dropping scan_id={key[0]} "
                    f"start_idx={key[1]} after {n} retries (chronically "
                    f"over budget)."))
            return
        self._retries[key] = n + 1
        try:
            # Re-inject at the same start_idx — restarts the killed step on a
            # clean worker without re-running upstream.
            self.routed_queue.put(("ref", ref))
            if self.log_queue is not None:
                self.log_queue.put(("log_warning",
                    f"reenqueue: scan_id={key[0]} start_idx={key[1]} "
                    f"retry #{n + 1} (was on {worker_name})."))
        except Exception as e:
            if self.log_queue is not None:
                self.log_queue.put(("log_warning",
                    f"reenqueue: put() failed: {e}"))

    def stop(self):
        self.running = False
        try:
            if getattr(self, "_watchdog_stop", None) is not None:
                self._watchdog_stop.set()
        except Exception:
            pass
        # Watchdog must be dead before workers are terminated, else it
        # respawns a "crashed" worker into closing queue handles.
        try:
            wt = getattr(self, "_watchdog_thread", None)
            if wt is not None and wt.is_alive():
                wt.join(timeout=3.0)
        except Exception:
            pass
        self._reap_workers()

    def _reap_workers(self, term_wait: float = 2.0) -> None:
        """Terminate every worker and GUARANTEE it's gone. A worker wedged in
        a native call (a JAX/XLA hang, or a `vmmap` shell-out on a stuck pid)
        ignores SIGTERM, so a plain terminate()+join leaves it alive — and
        once the parent exits it's orphaned (reparented to launchd), the
        "leaked worker after a clean run" symptom. So we escalate: SIGTERM →
        join(term_wait) → SIGKILL the stragglers → join. After this no worker
        survives teardown."""
        alive = [p for p in self.workers if p.is_alive()]
        if alive and self.log_queue is not None:
            self.log_queue.put(("log_info",
                f"[chain] reaping {len(alive)} worker(s): SIGTERM "
                + ", ".join(f"{p.name}({p.pid})" for p in alive)))
        for p in alive:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self.workers:
            try:
                p.join(timeout=term_wait)
            except Exception:
                pass
        # Escalate to SIGKILL for anything still breathing.
        for p in self.workers:
            if p.is_alive():
                if self.log_queue is not None:
                    self.log_queue.put(("log_warning",
                        f"[chain] worker {p.name}({p.pid}) ignored SIGTERM "
                        f"→ SIGKILL"))
                try:
                    p.kill()              # SIGKILL — uninterruptible reap
                except Exception:
                    pass
        for p in self.workers:
            try:
                p.join(timeout=2)
            except Exception:
                pass
            if p.is_alive() and self.log_queue is not None:
                self.log_queue.put(("log_warning",
                    f"worker {p.name} (pid={p.pid}) survived SIGKILL join"))

    def hard_stop(self) -> int:
        """Cancel all in-flight work: stop watchdog, terminate workers,
        drain both queues. Returns the number of items discarded across
        the input + routed queues. The chain is left in a `running=False`
        state; the caller is expected to drop it and build a fresh one."""
        self.running = False
        try:
            if getattr(self, "_watchdog_stop", None) is not None:
                self._watchdog_stop.set()
        except Exception:
            pass
        try:
            wt = getattr(self, "_watchdog_thread", None)
            if wt is not None and wt.is_alive():
                wt.join(timeout=3.0)
        except Exception:
            pass
        # Terminate first so workers stop pulling new items while we drain;
        # SIGKILL escalation guarantees no straggler is orphaned.
        self._reap_workers()
        drained = 0
        # input_queue: bounded mp.Queue. routed_queue: Manager.Queue.
        for q in (self.input_queue, self.routed_queue):
            if q is None:
                continue
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
                except Exception:
                    # Manager.Queue raises EOFError when the manager is
                    # dying; either way we're done draining.
                    break
        try:
            self._inflight.clear()
        except Exception:
            pass
        return drained

    def enqueue(self, item: ImageBuffer):
        """Push a fresh scan. Blocks at `max_input_pending` for import backpressure.

        When the buffer's raw image is already persisted (scan_id +
        parent_node_id set — true for every import/capture path), only a
        DB reference crosses the process boundary; the worker re-decodes
        from the project DB. Avoids pickling ~36 MB frames into the queue."""
        if self.input_queue is None:
            return
        if (isinstance(item, ImageBuffer)
                and item.parent_node_id is not None
                and item.scan_id is not None):
            self.input_queue.put(("ref", {
                "node_id": int(item.parent_node_id),
                "start_idx": 0,
                "branch_path": item.branch_path or "",
                "scan_id": int(item.scan_id),
                "parent_stem": item.parent_stem,
            }))
            return
        self.input_queue.put(item)

    def enqueue_resume(self, *, node_id: int, start_idx: int, branch_path: str,
                       scan_id: int, parent_stem: Optional[str] = None):
        """Resume the pipeline mid-chain from an already-persisted node.

        Used for branch-level reprocess: re-run only the steps from
        ``start_idx`` onward for a single page-branch (``branch_path``),
        re-decoding ``node_id``'s image. The worker applies that branch's
        per-page disable overrides as it goes (see ``run_pipeline``). Unlike
        :meth:`enqueue` (which always starts a fresh scan at step 0), this
        leaves sibling branches untouched."""
        if self.input_queue is None:
            return
        self.input_queue.put(("ref", {
            "node_id": int(node_id),
            "start_idx": int(start_idx),
            "branch_path": branch_path or "",
            "scan_id": int(scan_id),
            "parent_stem": parent_stem,
        }))

    def is_idle(self) -> bool:
        """True when nothing is in flight and both queues are drained.

        Best-effort: ``multiprocessing.Queue.empty()`` is only approximate, so
        callers should debounce across a few checks before treating the
        pipeline as finished. ``_inflight`` (a Manager dict the workers update
        as they pick up / finish items) is the authoritative "a worker is busy"
        signal; the queue checks catch work that's queued but not yet picked up."""
        try:
            if len(self._inflight) > 0:
                return False
            iq = self.input_queue
            if iq is not None and not iq.empty():
                return False
            rq = self.routed_queue
            if rq is not None and not rq.empty():
                return False
            return True
        except Exception:
            return False

    def get_input_queue(self):
        return self.input_queue

    # ───────────────────────────────────────────────────────────────────
    # Worker
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _worker_loop(elements_config: List[SimpleChainElement],
                     input_queue: multiprocessing.Queue,
                     routed_queue: multiprocessing.Queue,
                     log_queue: multiprocessing.Queue,
                     db_path: str,
                     inflight=None,
                     replay_enabled: bool = True):
        # Lazy imports inside worker (spawn-safe)
        import os as _os
        import pickle as _pickle
        import io as _io
        import numpy as _np
        from PIL import Image as _PILImage
        # Spawn-worker lifecycle: hard-exit if orphaned (issue #23) + optional
        # QoS (issue #24). Must run before the heavy imports / DB open so an
        # orphaned worker can't get stuck holding resources.
        from aglaia.workers.worker_lifecycle import (
            install_worker_lifecycle, maybe_start_memray, stop_memray,
        )
        install_worker_lifecycle()
        # Dev memory profiling: trace the first AGLAIA_MEMRAY_PAGES top-level
        # scans, then flush (a worker runs until killed, which wouldn't flush).
        _memray = maybe_start_memray("worker")
        _memray_target = int(_os.environ.get("AGLAIA_MEMRAY_PAGES", "3"))
        from aglaia.storage.db import open_db, in_transaction
        from aglaia.storage.persister import Persister
        from aglaia.storage.repo import NodeRepo, BranchRepo, ImageRepo, StepOverrideRepo

        conn = open_db(db_path)
        persister = Persister(conn)
        nodes_repo = NodeRepo(conn)
        branches_repo = BranchRepo(conn)
        images_repo = ImageRepo(conn)

        def load_ref(ref: dict) -> Optional[ImageBuffer]:
            """Rebuild an ImageBuffer from a queue reference. Queue items
            carry (node_id, start_idx, branch_path) instead of pixels —
            the image was persisted before enqueue, so a ~100-byte tuple
            replaces a multi-MB pickle through the manager process."""
            node = nodes_repo.get(int(ref["node_id"]))
            if node is None:
                return None
            img_row = images_repo.get(node["image_id"])
            if img_row is None:
                return None
            # PIL decode keeps the stored RGB channel order (cv2.imdecode
            # would return BGR and silently swap channels downstream).
            pil = _PILImage.open(_io.BytesIO(bytes(img_row["blob"])))
            itype = str(img_row["type"])
            if itype in ("BW", "GRAY"):
                arr = _np.asarray(pil.convert("L"))
            else:
                arr = _np.asarray(pil.convert("RGB"))
            buf = ImageBuffer(
                _np.ascontiguousarray(arr), ImageType(itype),
                dpi=float(img_row["dpi"] or 300.0),
                filestem=node["filestem"],
                # parent_stem distinguishes layout children from top-level
                # scans in the GUI — must survive the ref round-trip.
                parent_stem=ref.get("parent_stem"),
                scan_id=node["scan_id"],
                parent_node_id=int(ref["node_id"]),
                pipeline_version_id=node["pipeline_version_id"],
                branch_path=str(ref.get("branch_path") or ""),
                branch_label=node["branch_label"],
                depth=int(node["depth"] or 0),
            )
            if node["meta_json"]:
                try:
                    buf.meta = json.loads(node["meta_json"]) or {}
                except Exception:
                    buf.meta = {}
            return buf

        # Instantiate processors (may load models).
        registry = processor_registry()
        processors: list[AbstractImageProcessor] = []
        for el in elements_config:
            cls = registry[el.processor_name]
            processors.append(cls(el.options))

        N = len(processors)

        if log_queue is not None:
            log_queue.put(("worker_started", "Integrated"))

        # Deterministic worker recycle every N top-level inputs. Floor below
        # the vmmap-based check, which can't always read its own spawned process.
        _scans_processed = 0
        _recycle_scans = int(_os.environ.get("AGLAIA_WORKER_RECYCLE_SNAPS", "0"))
        # Set after each scan, cleared when the worker next goes idle — gates
        # the one-shot idle memory release so it runs once per work burst.
        _dirty_since_idle = False
        # Deferred idle recycle: malloc_trim can't free the JAX/CUDA resident
        # stack (loaded native libs + CUDA context, ~1.5 GB on Linux GPU).
        # After a work burst we've been idle this many seconds, exit so the
        # watchdog respawns a fresh light worker — releasing it fully (verified
        # 1.6 GB → 0.75 GB per worker). OPT-IN (default 0/off): the recycle's
        # respawn-during-shutdown traffic currently stalls the headless
        # grace-drain / clean exit, so it's behind an env flag until the
        # shutdown path is made recycle-aware. Next job pays a cold JAX import.
        _idle_recycle_s = float(_os.environ.get("AGLAIA_WORKER_IDLE_RECYCLE_S", "0"))
        _recycle_armed = False
        _idle_at = 0.0

        # Log RSS at +50 MB delta or every 30s. Catches leaks on long sessions.
        try:
            import psutil, os as _os
            _proc = psutil.Process(_os.getpid())
        except Exception:
            _proc = None
        _last_rss_mb = 0.0
        _last_rss_ts = 0.0

        # Recycle threshold reads the same vmmap metric as the parent watchdog
        # (psutil RSS misses XLA's compressed pool on Apple Silicon).
        from aglaia.workers.macos_mem import phys_footprint_mb as _phys_footprint_for_pid

        def _phys_footprint_mb() -> float:
            return _phys_footprint_for_pid(_os.getpid())

        def _release_idle_memory() -> None:
            """Return freed C-heap memory to the OS when the worker goes idle.

            glibc on Linux keeps freed heap in the malloc arena (RSS stays
            high) until ``malloc_trim``; macOS frees eagerly, which is why
            idle workers only looked leaky on Linux. (We deliberately do NOT
            call ``jax.clear_caches()`` here — it leaves the XLA/CUDA runtime
            unable to exit cleanly, hanging chain shutdown, and freed little.
            The big resident JAX/CUDA stack is reclaimed by the idle recycle
            below, not by trimming.)"""
            import gc as _gc
            _gc.collect()
            import sys as _sys
            if _sys.platform.startswith("linux"):
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

        def maybe_log_mem(tag: str = ""):
            """No-op stub. Memory reporting lives in `_rss_sampler_loop` /
            `_watchdog_loop` (vmmap phys_footprint), the single source of truth."""
            return

        def emit_event(buf: ImageBuffer, node_id: int, branch_id: Optional[int], event_type: str,
                       image_id: Optional[int] = None, parent_node_id: Optional[int] = None):
            if log_queue is None:
                return
            log_queue.put(("image_event", {
                "scan_id": buf.scan_id,
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "image_id": image_id,
                "branch_id": branch_id,
                "event_type": event_type,
                "filestem": buf.filestem,
                "depth": buf.depth,
                "branch_path": buf.branch_path,
                "branch_label": buf.branch_label,
                "meta": _gui_meta(buf.meta),
            }))

        def persist_step(out_buf: ImageBuffer, config: SimpleChainElement, step_idx: int,
                         parent_node_id: Optional[int], elapsed_ms: float, status_int: int,
                         store_image: Optional[bool] = None) -> tuple[int, Optional[int]]:
            # One transaction per step: image + node land together instead
            # of N autocommit transactions each paying a journal fsync.
            #
            # Every step stores its image. The old `persist:false` path
            # dropped the image of replay-kind steps (skew/trap/dewarp) as a
            # storage optimisation, leaving node.image_id NULL — which made
            # the debug overlay draw a step's quad on the WRONG (pre-transform)
            # ancestor image, and was a steady source of "image None" bugs.
            # Storage is cheap; an always-materialised tree is not surprising.
            if store_image is None:
                store_image = True
            with in_transaction(conn):
                return _persist_step_inner(out_buf, config, step_idx,
                                           parent_node_id, elapsed_ms, status_int,
                                           store_image)

        def _persist_step_inner(out_buf: ImageBuffer, config: SimpleChainElement, step_idx: int,
                                parent_node_id: Optional[int], elapsed_ms: float, status_int: int,
                                store_image: bool) -> tuple[int, Optional[int]]:
            image_id = (persister.persist_image(out_buf.buffer, out_buf.type.value, out_buf.dpi)
                        if store_image else None)
            node_id = persister.persist_node(
                scan_id=out_buf.scan_id,
                parent_id=parent_node_id,
                pipeline_version_id=out_buf.pipeline_version_id,
                step_idx=step_idx,
                step_name=config.instance_name,
                processor_name=config.processor_name,
                branch_label=out_buf.branch_label,
                depth=out_buf.depth,
                filestem=out_buf.filestem,
                image_id=image_id,
                status_int=status_int,
                elapsed_ms=elapsed_ms,
                meta=_clean_meta(out_buf.meta),
            )
            return node_id, image_id

        def run_pipeline(buf: ImageBuffer, start_idx: int):
            current = buf
            # Per-page processor disable: the layout's skip set, keyed
            # (branch_path, step_idx). A disabled step is bypassed with a
            # passthrough node instead of running the processor (see below).
            # Loaded once per run; resumed branch workers re-enter here and
            # reload their own (branch-specific) set.
            disabled_steps: set[tuple[str, int]] = (
                StepOverrideRepo(conn).map_for_scan(int(current.scan_id))
                if current.scan_id is not None else set()
            )
            for i in range(start_idx, N):
                config = elements_config[i]
                processor = processors[i]
                # Per-page disable: skip this processor for this layout. Emit
                # a passthrough node (same image — dedups to the parent's
                # image_id — no process(), no replay stamp) so the node tree
                # stays contiguous and downstream runs on the un-transformed
                # image; Replay auto-excludes it (nothing stamped). Only
                # COORDINATE/PIXEL_VALUE processors are skippable: ROI /
                # branch-emitting steps (PageDetector) would restructure the
                # branch tree, so they're locked in the UI and ignored here.
                trait = getattr(type(processor), "REPLAY_TRAIT", None)
                skippable = trait in (ReplayTrait.COORDINATE, ReplayTrait.PIXEL_VALUE)
                if skippable and (current.branch_path or "", i + 1) in disabled_steps:
                    pass_buf = ImageBuffer(
                        current.buffer, current.type, dpi=current.dpi,
                        filestem=current.filestem, scan_id=current.scan_id,
                        pipeline_version_id=current.pipeline_version_id,
                        depth=current.depth + 1,
                        branch_label=current.branch_label,
                        branch_path=current.branch_path,
                        parent_node_id=current.parent_node_id,
                    )
                    pass_buf.meta["disabled"] = True
                    pass_buf.meta["status"] = 1
                    nid, iid = persist_step(pass_buf, config, i + 1,
                                            current.parent_node_id, 0.0, 1)
                    pass_buf.parent_node_id = nid
                    emit_event(pass_buf, nid, None, config.instance_name,
                               image_id=iid, parent_node_id=current.parent_node_id)
                    if log_queue is not None:
                        log_queue.put(("log_info",
                            f"[{current.filestem}] {config.processor_name}: "
                            "disabled (passthrough)"))
                    # `disabled` marks THIS node only — don't let it ride the
                    # buffer onto downstream steps' meta (they'd render as
                    # disabled too). The node row already has it (persisted).
                    pass_buf.meta.pop("disabled", None)
                    current = pass_buf
                    continue
                start_t = time.time()
                old_type = current.type
                # Snapshot replay stamp BEFORE in-place mutation; used to clear
                # stale stamps that would leak across non-replay steps.
                old_replay_kind = (current.meta or {}).get("replay_kind")
                # Snapshot pre-process buffer geometry for the op-log
                # size chain. Reading after `process()` would only
                # capture the output side.
                try:
                    in_h, in_w = current.buffer.shape[:2]
                except Exception:
                    in_h, in_w = 0, 0
                in_dpi = float(getattr(current, "dpi", 0) or 0)
                # No per-call timeout: parent watchdog SIGKILLs over-budget
                # workers instead (a per-call thread leak hits ~500 MB per hang).
                try:
                    result = processor.run(current)
                except Exception as e:
                    if log_queue is not None:
                        log_queue.put(("error",
                            f"[{current.filestem}] {config.processor_name}: {e}\n{traceback.format_exc()}"))
                    return
                elapsed_ms = (time.time() - start_t) * 1000.0

                if result is None:
                    if log_queue is not None:
                        log_queue.put(("log_warning",
                            f"[{current.filestem}] {config.processor_name} returned None; stopping branch."))
                    return

                # Determine outputs (single or branched)
                if isinstance(result, list):
                    outputs = result
                elif getattr(result, "children", None):
                    outputs = result.children
                else:
                    outputs = [result]

                # BW palette preservation across non-binarizing steps
                if old_type == ImageType.BW:
                    from aglaia.processors.utils import binarize_fixed
                    for t in outputs:
                        if t.type != ImageType.BW or not t.check_binary():
                            t.buffer = binarize_fixed(t.buffer, 127)
                            t.type = ImageType.BW

                # Stamp GPU usage onto outputs — UI renders a rocket on the timing bar.
                proc_uses_gpu = bool(getattr(processor, "uses_gpu", False))
                if proc_uses_gpu:
                    for t in outputs:
                        if t.meta is None:
                            t.meta = {}
                        t.meta["gpu"] = True

                # Clear inherited replay_kind/params when the processor didn't
                # stamp fresh ones — meta must reflect "this step contributed".
                for t in outputs:
                    if t.meta is None:
                        t.meta = {}
                    cur_kind = t.meta.get("replay_kind")
                    if cur_kind is not None and cur_kind == old_replay_kind:
                        t.meta.pop("replay_kind", None)
                        t.meta.pop("replay_params", None)

                branched = len(outputs) > 1

                if log_queue is not None:
                    h, w = current.buffer.shape[:2]
                    log_queue.put(("timing", current.filestem, f"{w}x{h}",
                                   current.dpi, config.processor_name, elapsed_ms, True))
                    # Unified op-log line — drains per-processor stats
                    # populated during `process()` so every subsystem
                    # logs the same shape (see aglaia/workers/oplog.py).
                    if not _os.environ.get("AGLAIA_QUIET_STAGE"):
                        from aglaia.workers.oplog import format_op, fmt_size_chain
                        stats = dict(getattr(processor, "last_stats", {}) or {})
                        scope: dict = {}
                        if getattr(current, "scan_id", None) is not None:
                            scope["scan"] = current.scan_id
                        bl = getattr(current, "branch_label", None)
                        if bl:
                            scope["layout"] = bl
                        # Build the (input — working -> output) size
                        # chain. Input comes from the pre-process buffer
                        # we snapshotted before `processor.process(...)`
                        # — see `in_h, in_w, in_dpi` below.
                        first_out = outputs[0] if outputs else None
                        if first_out is not None:
                            oh, ow = first_out.buffer.shape[:2]
                            out_dpi = float(first_out.dpi or 0)
                        else:
                            oh, ow, out_dpi = h, w, float(current.dpi or 0)
                        working = stats.pop("working_wh_dpi", None)
                        size_chain = fmt_size_chain(
                            ((in_w, in_h), in_dpi),
                            working,
                            ((ow, oh), out_dpi),
                        )
                        line = format_op(
                            f"pipeline.{config.processor_name}",
                            elapsed_ms=elapsed_ms,
                            size_chain=size_chain,
                            **scope,
                            **stats,
                        )
                        log_queue.put(("log_info", line))

                if branched:
                    # Mark previous step's node as a branch point.
                    if current.parent_node_id is not None:
                        nodes_repo.mark_branch_point(current.parent_node_id)
                    for idx_b, out_buf in enumerate(outputs):
                        # Inherit + extend tree context
                        out_buf.scan_id = current.scan_id
                        out_buf.pipeline_version_id = current.pipeline_version_id
                        out_buf.depth = current.depth + 1
                        label = out_buf.branch_label or chr(ord("A") + idx_b)
                        out_buf.branch_label = label
                        if current.branch_path:
                            out_buf.branch_path = f"{current.branch_path}.{label}"
                        else:
                            out_buf.branch_path = label
                        status_int = int(out_buf.meta.get("status", 1)) if out_buf.meta else 1
                        nid, iid = persist_step(out_buf, config, i + 1,
                                                current.parent_node_id, elapsed_ms, status_int,
                                                store_image=True)
                        out_buf.parent_node_id = nid
                        emit_event(out_buf, nid, None, config.instance_name,
                                   image_id=iid, parent_node_id=current.parent_node_id)
                        # Re-inject onto routed queue (unbounded — no
                        # deadlock) as a DB reference: the child was just
                        # persisted, so ~100 bytes replace a multi-MB
                        # pixel pickle through the manager process.
                        routed_queue.put(("ref", {
                            "node_id": int(nid),
                            "start_idx": i + 1,
                            "branch_path": out_buf.branch_path or "",
                            "scan_id": out_buf.scan_id,
                            "parent_stem": out_buf.parent_stem,
                        }))
                    return  # this worker stops; children continue in another worker
                else:
                    out_buf = outputs[0]
                    out_buf.scan_id = current.scan_id
                    out_buf.pipeline_version_id = current.pipeline_version_id
                    out_buf.depth = current.depth + 1
                    out_buf.branch_path = current.branch_path
                    out_buf.branch_label = current.branch_label
                    status_int = int(out_buf.meta.get("status", 1)) if out_buf.meta else 1
                    nid, iid = persist_step(out_buf, config, i + 1,
                                            current.parent_node_id, elapsed_ms, status_int)
                    parent_for_event = current.parent_node_id
                    out_buf.parent_node_id = nid
                    emit_event(out_buf, nid, None, config.instance_name,
                               image_id=iid, parent_node_id=parent_for_event)
                    current = out_buf

            # Pipeline finished cleanly for this branch — register terminal.
            if current.parent_node_id is not None and current.scan_id is not None:
                bid = branches_repo.upsert(current.scan_id, current.branch_path or "",
                                           current.parent_node_id)
                if log_queue is not None:
                    log_queue.put(("branch_ready", {
                        "scan_id": current.scan_id,
                        "branch_id": bid,
                        "branch_path": current.branch_path or "",
                        "chosen_node_id": current.parent_node_id,
                    }))

                # Replay: reapply stamped transforms to post-Layout colour source
                # in one composite warp + single binarisation. Final pixels go
                # through one interpolation. Two off-switches:
                #   * `replay_enabled=False` (top-level pipeline `replay: false`)
                #   * `AGLAIA_NO_REPLAY=1` env var (debug)
                if not replay_enabled or _os.environ.get("AGLAIA_NO_REPLAY"):
                    return
                replay_start_t = time.time()
                replay_ok = False
                try:
                    from aglaia.workers.Replay import replay_branch
                    replay_img, replay_meta = replay_branch(
                        conn, current.scan_id,
                        branch_path=current.branch_path or "",
                        dpi=float(current.dpi or 300.0),
                    )
                    replay_elapsed_ms = (time.time() - replay_start_t) * 1000.0
                    replay_buf = ImageBuffer(
                        replay_img,
                        ImageType.BW if replay_img.ndim == 2 else ImageType.COLOR,
                        dpi=current.dpi,
                        filestem=current.filestem,
                        scan_id=current.scan_id,
                        pipeline_version_id=current.pipeline_version_id,
                        depth=current.depth + 1,
                        branch_label=current.branch_label,
                        branch_path=current.branch_path,
                    )
                    replay_buf.meta.update(replay_meta)
                    replay_buf.meta["status"] = 1
                    replay_buf.meta["elapsed_ms"] = replay_elapsed_ms
                    replay_inst = f"{N + 1:02d}_replay"
                    replay_config = SimpleChainElement(
                        "Replay", None, instance_name=replay_inst
                    )
                    rnid, riid = persist_step(
                        replay_buf, replay_config, N + 1,
                        current.parent_node_id, replay_elapsed_ms, 1,
                    )
                    branches_repo.upsert(current.scan_id,
                                         current.branch_path or "", rnid)
                    emit_event(replay_buf, rnid, bid, replay_inst,
                               image_id=riid,
                               parent_node_id=current.parent_node_id)
                    replay_ok = True
                    if log_queue is not None:
                        rh, rw = replay_img.shape[:2]
                        log_queue.put(("timing", current.filestem,
                                       f"{rw}x{rh}", current.dpi,
                                       "Replay", replay_elapsed_ms, True))
                except Exception as e:
                    # "No replay-participating nodes" = pipeline simply
                    # had no replay-stamping step (Skew/Dewarp/Margin/
                    # Binarizer). That's the documented behaviour of
                    # `replay: true` over a non-participating pipeline,
                    # not an error — skip silently.
                    msg = str(e)
                    silent = "No replay-participating nodes" in msg
                    if log_queue is not None and not silent:
                        replay_elapsed_ms = (time.time() - replay_start_t) * 1000.0
                        log_queue.put(("timing", current.filestem, "?",
                                       current.dpi, "Replay",
                                       replay_elapsed_ms, False))
                        log_queue.put(("log_warning",
                            f"[{current.filestem}] replay failed: {e}"))

        # Drain routed (branch) queue first; clearing children before pulling
        # another raw input keeps memory bounded.
        while True:
            try:
                item = None
                try:
                    item = routed_queue.get_nowait()
                except queue.Empty:
                    pass
                if item is None:
                    try:
                        item = input_queue.get(timeout=0.5)
                    except queue.Empty:
                        if _dirty_since_idle:
                            # First idle tick after a work burst: cheap release
                            # now, then arm the deferred full recycle. Stay
                            # silent — extra log traffic perturbs the headless
                            # grace-drain's queue-silence exit detection.
                            _dirty_since_idle = False
                            _release_idle_memory()
                            _idle_at = time.time()
                            _recycle_armed = True
                        elif (_recycle_armed and _idle_recycle_s > 0
                                and (time.time() - _idle_at) > _idle_recycle_s):
                            if log_queue is not None:
                                log_queue.put(("log_info",
                                    f"[worker pid={_os.getpid()}] idle "
                                    f"{_idle_recycle_s:.0f}s; recycling to "
                                    "release memory."))
                            conn.close()
                            _os._exit(0)
                        continue
                if item is None:
                    break

                start_node_idx = 0
                buf_to_process = None
                if isinstance(item, tuple) and item and item[0] == "ref":
                    ref = dict(item[1])
                    start_node_idx = int(ref.get("start_idx", 0))
                    buf_to_process = load_ref(ref)
                    if buf_to_process is None:
                        if log_queue is not None:
                            log_queue.put(("log_warning",
                                f"ref item dropped: node_id="
                                f"{ref.get('node_id')} no longer in DB."))
                        continue
                else:
                    buf_to_process = item

                if isinstance(buf_to_process, ImageBuffer):
                    # In-flight registration so the watchdog can requeue on
                    # SIGKILL. Keyed by process name; watchdog already knows
                    # worker names. Stores a DB reference, NOT the pixels —
                    # the old full-buffer pickle cost ~70 MB of memcpy +
                    # manager RPC per item just for crash bookkeeping.
                    _worker_name = multiprocessing.current_process().name
                    if (inflight is not None
                            and buf_to_process.parent_node_id is not None):
                        try:
                            inflight[_worker_name] = _pickle.dumps({
                                "node_id": int(buf_to_process.parent_node_id),
                                "start_idx": int(start_node_idx),
                                "branch_path": buf_to_process.branch_path or "",
                                "scan_id": buf_to_process.scan_id,
                                "parent_stem": buf_to_process.parent_stem,
                            }, protocol=_pickle.HIGHEST_PROTOCOL)
                        except Exception:
                            pass
                    run_pipeline(buf_to_process, start_node_idx)
                    if inflight is not None:
                        try:
                            inflight.pop(_worker_name, None)
                        except Exception:
                            pass
                    # Drop refs + force GC to break JAX-side reference cycles.
                    buf_to_process = None
                    item = None
                    import gc as _gc
                    _gc.collect()
                    # Top-level + routed children both count: a worker draining
                    # only routed items would otherwise underestimate pressure.
                    _scans_processed += 1
                    _dirty_since_idle = True
                    maybe_log_mem(tag=f"after scan")
                    if _memray is not None and _scans_processed >= _memray_target:
                        stop_memray(_memray)   # flush after N pages
                        _memray = None
                    # Deterministic scan-count recycle (vmmap can be flaky in spawn).
                    if (_recycle_scans > 0
                            and _scans_processed >= _recycle_scans):
                        if log_queue is not None:
                            log_queue.put((
                                "log_warning",
                                f"[worker pid={_os.getpid()}] "
                                f"scans={_scans_processed} >= "
                                f"{_recycle_scans} cap; recycling.",
                            ))
                        conn.close()
                        _os._exit(0)
                    # XLA-CPU's host pool grows per distinct input shape and
                    # rarely releases. Once vmmap phys_footprint crosses the cap,
                    # exit and let the watchdog respawn — only reliable release
                    # path on macOS. psutil RSS underreports 5-10× on Apple Silicon.
                    try:
                        recycle_mb = float(
                            _os.environ.get("AGLAIA_WORKER_RECYCLE_MB", "3072")
                        )
                        phys_mb = _phys_footprint_mb()
                        if phys_mb > recycle_mb:
                            if log_queue is not None:
                                log_queue.put((
                                    "log_warning",
                                    f"[worker pid={_os.getpid()}] "
                                    f"phys_footprint={phys_mb:.0f} MB > "
                                    f"{recycle_mb:.0f} MB cap; recycling.",
                                ))
                            conn.close()
                            _os._exit(0)
                    except Exception:
                        pass
            except (ConnectionError, EOFError, BrokenPipeError):
                # Parent (and its Manager process) is gone — Ctrl-C, crash, or
                # shutdown. The routed_queue (Manager.Queue) connection is dead,
                # so there is no work source left. Exit quietly + immediately
                # instead of spamming the error in a tight retry loop (which
                # flooded the terminal with ConnectionRefusedError on shutdown).
                try:
                    import sys as _sys
                    _sys.stderr.write(
                        f"[worker {_os.getpid()}] parent/manager gone — exiting\n")
                    _sys.stderr.flush()
                except Exception:
                    pass
                try:
                    stop_memray(_memray)
                except Exception:
                    pass
                _os._exit(0)
            except Exception as e:
                if log_queue is not None:
                    log_queue.put(("error",
                        f"Integrated Worker Loop Error: {e}\n{traceback.format_exc()}"))
