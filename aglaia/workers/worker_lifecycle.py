# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Spawn-worker lifecycle helpers, called once at the top of every
`IntegratedProcessingChain._worker_loop` (initial + respawned workers).

Two concerns, both spawn-specific:

* **Parent-death cleanup (issue #23).** A `multiprocessing` spawn child does
  NOT die when its parent does — it reparents to launchd/init and keeps
  running (each carries ~40 threads + the chain's 5 s `vmmap` RSS sampler).
  So a crash, OOM-kill, GUI force-quit, `kill -9 <main>`, or a `timeout`
  wrapper orphans the whole worker pool, which then piles up and swamps the
  machine. The graceful path is covered by `chain._reap_workers`
  (SIGTERM→SIGKILL); this covers the *ungraceful* one — a tiny daemon thread
  watches `getppid()` and hard-exits the worker the moment it's reparented.

* **Worker QoS (issue #24).** Apple Silicon schedules un-annotated sustained
  compute on the E-cores. Setting the worker's QoS class lets macOS place it
  on the P-cores. Gated by ``AGLAIA_WORKER_QOS`` (unset = no change = current
  behaviour) until a controlled A/B confirms it actually helps — see #24.
"""

from __future__ import annotations

import os
import sys
import threading
import time


def install_parent_death_watch(poll_s: float = 1.0) -> None:
    """Hard-exit this process if its parent dies (it gets reparented).

    Two layers:

    * **Linux** — ``prctl(PR_SET_PDEATHSIG, SIGKILL)``: the kernel SIGKILLs us
      the instant the parent dies, regardless of what we're doing (even deep
      in a native call). Fully robust.
    * **All platforms** — a daemon thread polling ``getppid()``; when it
      returns ``1`` the parent has died and we've been reparented to
      launchd/init, so we ``os._exit``. This is the only option on macOS (no
      PDEATHSIG), and it has a known limitation: a worker stuck in a
      **GIL-holding native call** (JAX/XLA) can't run the watch thread until
      that call returns, so its exit is bounded by the in-flight op rather
      than instant. Still bounds an orphan's life to ~one op instead of
      forever. See issue #23.

    IMPORTANT — the test is ``getppid() == 1``, NOT "ppid changed from the
    one captured at start". A spawn worker captures its ppid mid-bootstrap,
    when the parent is a *transient* launcher that then exits and the worker
    reparents to the real main → ppid legitimately CHANGES with the main
    fully alive. The old "!= orig" check fired on exactly that, silently
    ``os._exit``-ing healthy workers a beat after they started → processing
    ground to a halt (~first batch). ``== 1`` only fires on a true orphan.

    ``os._exit`` (not ``sys.exit``) skips finalizers that could hang a wedged
    worker — the OS reclaims the DB handle etc."""
    # Linux kernel-level guarantee (only fires on a real parent death).
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal
            PR_SET_PDEATHSIG = 1
            ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except Exception:
            pass

    def _watch() -> None:
        while True:
            time.sleep(poll_s)
            try:
                ppid = os.getppid()
            except Exception:
                continue
            if ppid == 1:                # reparented to launchd/init = orphan
                sys.stderr.write(
                    f"[worker {os.getpid()}] parent died (reparented to pid 1) "
                    f"— self-exiting, orphan cleanup (issue #23)\n")
                sys.stderr.flush()
                os._exit(1)

    threading.Thread(target=_watch, daemon=True,
                     name="parent-death-watch").start()


# qos_class_t enum values (sys/qos.h).
_QOS_CLASSES = {
    "user_interactive": 0x21,
    "user_initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
    "background": 0x09,
}


def apply_worker_qos() -> None:
    """Set this worker thread's macOS QoS class.

    Defaults to ``user_initiated`` — the semantically-correct class for an
    interactive app's heavy compute, so macOS schedules it on the P-cores
    instead of demoting un-annotated sustained work to the E-cores (#24).
    Override / opt out with ``AGLAIA_WORKER_QOS`` (e.g. ``default``,
    ``utility``, or ``none``/``off`` to leave scheduling untouched).

    Sets the CURRENT thread via ``pthread_set_qos_class_self_np``. GCD-backed
    work (Apple Accelerate / vecLib BLAS) inherits the QoS and moves with it;
    libraries with their own pthread pools (XLA, OpenCV) may not — so the #24
    A/B is still needed to quantify the real effect."""
    if sys.platform != "darwin":
        return
    want = os.environ.get("AGLAIA_WORKER_QOS", "user_initiated").strip().lower()
    if want in ("", "none", "off"):
        return
    cls = _QOS_CLASSES.get(want)
    if cls is None:
        return
    try:
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
        # int pthread_set_qos_class_self_np(qos_class_t, int relative_priority)
        lib.pthread_set_qos_class_self_np(ctypes.c_int(cls), ctypes.c_int(0))
    except Exception:
        pass


def install_worker_lifecycle() -> None:
    """One call for a spawned worker: parent-death watch + optional QoS."""
    install_parent_death_watch()
    apply_worker_qos()
