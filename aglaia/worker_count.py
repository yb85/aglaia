# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Single source of truth for the pipeline worker count.

A configured value of 0 (or any non-positive / unset value, in the config DB
or the `--workers` CLI flag) means AUTO: pick a count from the CPU budget.
Used by the chain (to spawn), the CLI, and the GUI (to display the resolved
count + mode in the pipeline sidebar) so all three agree."""

import os
import platform

# Measured peak resident per worker (loaded native libs + XLA/CUDA stack +
# per-step buffers). The auto count keeps the fleet under 0.9 of available RAM
# at this rate, and a manual count above that gets a swap-risk warning.
PER_WORKER_GB = 1.5
RAM_HEADROOM = 0.9


def _cpu_workers() -> int:
    """CPU-budget worker count, platform-aware (before the RAM cap).

      * Apple Silicon — no SMT, P/E split; dewarp runs on the MLX backend (on
        the workers), so use the PERFORMANCE core count
        (`sysctl hw.perflevel0.physicalcpu`). M4 (4P+6E) → 4, Pro → 8, Max → 12.
      * x86 (and the rest) — symmetric cores with SMT; `os.cpu_count()` is
        *logical* (2x physical). With batched GPU dewarp the solve runs in a
        separate process, so workers do only light CPU steps and more of them
        help — logical/2 − 1 (≈ physical − 1). Ryzen 7900X (12c/24t) → 11
        (measured: 6w 2'10", 10-12w 1'44").
    """
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, timeout=2)
            p = int(out.stdout.strip())
            if p > 0:
                return max(p, 1)
        except Exception:
            pass
        return max((os.cpu_count() or 4) // 2, 1)
    return max((os.cpu_count() or 4) // 2 - 1, 1)


def ram_worker_cap() -> int | None:
    """Max workers that fit in RAM_HEADROOM of *available* RAM at PER_WORKER_GB
    each, or None if it can't be measured (don't cap then)."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        return max(int(RAM_HEADROOM * avail_gb / PER_WORKER_GB), 1)
    except Exception:
        return None


def auto_workers() -> int:
    """Auto pipeline-worker count: the CPU-budget count, capped so the fleet
    fits in available RAM (floored at 1)."""
    n = _cpu_workers()
    cap = ram_worker_cap()
    if cap is not None:
        n = min(n, cap)
    return max(n, 1)


def ram_warning(count: int) -> str | None:
    """A swap-risk message if a MANUAL `count` exceeds what RAM can hold, else
    None. Used by both the Settings dialog and the headless CLI."""
    cap = ram_worker_cap()
    if cap is None or count <= cap:
        return None
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        avail_gb = 0.0
    return (f"{count} workers may not fit in RAM (~{PER_WORKER_GB:.1f} GB each, "
            f"~{avail_gb:.0f} GB available → ~{cap} fit safely) — risk of "
            f"swapping and slowdowns.")


def resolve_workers(raw) -> tuple[int, bool]:
    """Resolve a configured worker value to (count, is_auto).

    `raw` <= 0, None, or non-numeric → auto. Otherwise the manual count."""
    try:
        v = int(raw)
    except (TypeError, ValueError):
        v = 0
    if v <= 0:
        return auto_workers(), True
    return v, False
