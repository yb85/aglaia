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


def auto_workers() -> int:
    """Auto pipeline-worker count, platform-aware (floored at 1).

    The dewarp/optimise workload is performance-core-bound (CLAUDE.md) and
    each worker spawns its own XLA/BLAS thread pool, so over-subscribing hurts:

      * Apple Silicon — no SMT, with a P/E core split. Use the PERFORMANCE
        core count (`sysctl hw.perflevel0.physicalcpu`); E-cores only slow
        this workload. M4 (4P+6E) → 4, M4 Pro → 8, M4 Max → 12.
      * x86 (and the rest) — symmetric cores with SMT. `os.cpu_count()`
        reports *logical* CPUs (2x physical under SMT); use logical/4
        (≈ half the physical cores) to leave headroom for the per-worker
        thread pools, the GUI/OS, and GPU feeding. Ryzen 7900X (12c/24t) → 6.
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
        # Fallback (e.g. older sysctl): half the cores ~ P-core estimate.
        return max((os.cpu_count() or 4) // 2, 1)

    return max((os.cpu_count() or 4) // 4, 1)


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
