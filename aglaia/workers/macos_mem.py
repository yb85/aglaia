# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""macOS phys_footprint / thread-count helpers.

`Physical footprint` from `vmmap -summary` is the value Activity Monitor
reports as "Memory" and is the only one that reflects XLA's compressed
pool. psutil RSS on Apple Silicon under-reports by 30× when JAX is hot,
so the watchdog, sampler, and worker self-recycle all need this metric.
"""

import ctypes
import ctypes.util
import subprocess
import sys


_UNIT_MULTIPLIER = {
    "": 1.0 / 1024,   # vmmap sometimes omits the suffix; default to KB.
    "K": 1.0 / 1024,
    "M": 1.0,
    "G": 1024.0,
    "T": 1024.0 * 1024.0,
}


# ── proc_pid_rusage (libproc) — same ri_phys_footprint metric as vmmap, but
# in microseconds and WITHOUT suspending the target task. `vmmap` walks the
# target's VM regions while it's suspended (hundreds of ms on a JAX-fat
# worker), so the watchdog/sampler itself periodically stalled the pipeline.
class _RUsageInfoV0(ctypes.Structure):
    # struct rusage_info_v0 from <sys/resource.h>. ri_phys_footprint is the
    # 8th uint64 after the 16-byte uuid (byte offset 72).
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


def _load_proc_pid_rusage():
    try:
        path = ctypes.util.find_library("proc") or "/usr/aglaia/libproc.dylib"
        lib = ctypes.CDLL(path, use_errno=True)
        fn = aglaia.proc_pid_rusage
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        return fn
    except Exception:
        return None


# Resolved once per process (workers each import the module fresh).
_PROC_PID_RUSAGE = _load_proc_pid_rusage()
_RUSAGE_INFO_V0 = 0


def _rusage_footprint_mb(pid: int) -> float:
    """phys_footprint (MB) via proc_pid_rusage, or -1.0 if unavailable."""
    if _PROC_PID_RUSAGE is None:
        return -1.0
    info = _RUsageInfoV0()
    rc = _PROC_PID_RUSAGE(int(pid), _RUSAGE_INFO_V0, ctypes.byref(info))
    if rc != 0:
        return -1.0
    return info.ri_phys_footprint / (1024.0 * 1024.0)


def _psutil_rss_mb(pid: int) -> float:
    """RSS (MB) via psutil, or -1.0 if unavailable. The cross-platform
    fallback — good enough on Linux/Windows where the macOS phys_footprint
    metric doesn't exist."""
    try:
        import psutil
        return psutil.Process(int(pid)).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return -1.0


def phys_footprint_mb(pid: int) -> float:
    """Return phys_footprint (MB) for `pid`, or -1.0 on failure.

    On macOS prefers proc_pid_rusage (fast, no task suspension), then
    `vmmap -summary` parsing — the metric Activity Monitor shows. Off
    macOS (Linux/Windows) those tools don't exist, so fall back to
    psutil RSS. Pure — safe to import in spawned workers.
    """
    if sys.platform == "darwin":
        mb = _rusage_footprint_mb(pid)
        if mb >= 0.0:
            return mb
        try:
            out = subprocess.run(
                ["vmmap", "-summary", str(pid)],
                capture_output=True, text=True, timeout=3.0,
            ).stdout
            for line in out.splitlines():
                if not line.startswith("Physical footprint:"):
                    continue
                raw = line.split(":", 1)[1].strip()
                digits = raw.rstrip("KMGT")
                suffix = raw[len(digits):]
                return float(digits) * _UNIT_MULTIPLIER.get(suffix, 1.0)
        except Exception:
            pass
    return _psutil_rss_mb(pid)


def thread_count(pid: int) -> int:
    """psutil.num_threads(), or -1 if psutil isn't installed / pid gone."""
    try:
        import psutil
        return psutil.Process(pid).num_threads()
    except Exception:
        return -1
