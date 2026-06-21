# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Pick a safe ``llama-server --parallel`` slot count for the host.

The OCR engine ships through llama.cpp, which holds N independent KV
caches (one per slot) plus the model weights in (V)RAM. Oversizing N
crashes the server with OOM or, on Apple unified memory, silently
spills layers to CPU and kills throughput.

This module probes the host's available GPU/VRAM/unified-memory budget,
subtracts the model weights, and divides the remainder by an empirical
per-slot KV-cache cost. Falls back to ``1`` when nothing useful is
detectable (CPU-only).

Supported backends:
  * Apple Silicon (macOS) — unified memory via ``sysctl hw.memsize``.
    Metal sees ~75% of the unified pool. We budget 60% to leave room
    for the rest of the OS + the Aglaïa process.
  * NVIDIA (Linux / Windows) — VRAM via ``nvidia-smi`` query.
  * AMD ROCm (Linux) — VRAM via ``rocm-smi`` query.
  * Anything else — return 1.

Caller passes the model size + per-slot cost in MB and gets back an
int slot count clamped to ``[1, max_slots]`` (default max=8).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Optional


# Empirical defaults — keep in sync with what Surya actually loads.
# kv_per_slot includes ~750 MB raw KV cache at 12K ctx + ~750 MB worth
# of compute / batch / mmproj buffers that llama.cpp allocates per
# slot. Slightly pessimistic so a fresh install on a 16 GB Mac picks
# 3-4 slots instead of biting off 8 and seeing layers spill to CPU.
DEFAULT_MODEL_MB = 1700
DEFAULT_KV_PER_SLOT_MB = 1500
DEFAULT_MAX_SLOTS = 8

# Fraction of detected memory we're willing to spend on the OCR stack.
# Apple unified memory: Metal sees ~75 % of the pool; we further halve
# that so the OS + Aglaïa's pipeline workers (BG decode, dewarp, etc.)
# don't get evicted while Surya runs.
_MAC_BUDGET_FRACTION = 0.45
_CUDA_BUDGET_FRACTION = 0.85   # discrete GPU = we own the VRAM
_ROCM_BUDGET_FRACTION = 0.85


def _macos_unified_memory_mb() -> Optional[int]:
    """Returns total unified memory in MB or None when unavailable."""
    if platform.system().lower() != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=2
        ).decode().strip()
        return int(out) // (1024 * 1024)
    except Exception:
        return None


def _nvidia_vram_mb() -> Optional[int]:
    """Returns largest GPU's free + used VRAM in MB, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        sizes = [int(line.strip()) for line in out.splitlines() if line.strip()]
        return max(sizes) if sizes else None
    except Exception:
        return None


def _rocm_vram_mb() -> Optional[int]:
    """Best-effort AMD ROCm probe via ``rocm-smi``."""
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            timeout=5,
        ).decode()
        # CSV columns vary by version — pick the first plausible int.
        best = 0
        for line in out.splitlines():
            for tok in line.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    best = max(best, int(tok))
        # rocm-smi returns bytes — convert to MB.
        return best // (1024 * 1024) if best > 0 else None
    except Exception:
        return None


def _detect_budget_mb() -> tuple[Optional[int], float, str]:
    """Returns ``(memory_mb, budget_fraction, backend_label)``.

    ``memory_mb`` is ``None`` when no GPU/VRAM/unified memory was found.
    """
    # NVIDIA wins on any platform.
    cuda = _nvidia_vram_mb()
    if cuda is not None:
        return cuda, _CUDA_BUDGET_FRACTION, "cuda"
    rocm = _rocm_vram_mb()
    if rocm is not None:
        return rocm, _ROCM_BUDGET_FRACTION, "rocm"
    mac = _macos_unified_memory_mb()
    if mac is not None:
        return mac, _MAC_BUDGET_FRACTION, "apple"
    return None, 0.0, "cpu"


def _apple_compute_cap(mem_mb: int) -> int:
    """Compute-bound cap for Apple Silicon.

    Updated cap from the M4-base sweep in ``bench_surya_llamacpp.py``:
    parallel=4 (15.6 s/img) was the wall-clock winner across the
    8-image bench corpus, beating parallel=8 (no measurable lift, +RAM
    pressure) and parallel=2 (~19 % slower from under-utilized GPU).
    The unified-memory model on M-series saturates at p4 because Metal
    GQA + KV-q8 shares one engine across slots; spinning more slots
    burns context without speeding up decode.
    """
    return 4


def auto_worker_count(
    *,
    model_mb: int = DEFAULT_MODEL_MB,
    kv_per_slot_mb: int = DEFAULT_KV_PER_SLOT_MB,
    max_slots: int = DEFAULT_MAX_SLOTS,
) -> tuple[int, str]:
    """Pick a slot count that fits the detected GPU budget AND the
    backend's actual parallel-compute capacity.

    Returns ``(slots, reason)`` where ``reason`` is a one-line string
    suitable for logging / surfacing in the Backends footer.
    """
    mem_mb, frac, backend = _detect_budget_mb()
    if mem_mb is None or mem_mb <= 0:
        return 1, f"backend={backend} (no GPU memory detected → 1 slot)"

    # Apple Silicon caps hard on compute, NOT memory — Metal slots
    # multiplex one engine. Apply a per-SKU cap before the memory
    # math.
    compute_cap = _apple_compute_cap(mem_mb) if backend == "apple" else max_slots
    effective_max = min(max_slots, compute_cap)

    budget = int(mem_mb * frac)
    available_for_kv = budget - model_mb
    if available_for_kv <= 0:
        return 1, (f"backend={backend} {mem_mb} MB total, "
                    f"budget {budget} MB < model {model_mb} MB → 1 slot")
    slots = max(1, min(effective_max, available_for_kv // kv_per_slot_mb))
    return int(slots), (
        f"backend={backend} {mem_mb} MB · budget {budget} MB · "
        f"{slots} slot(s) × {kv_per_slot_mb} MB KV + model {model_mb} MB"
        + (f" · apple_cap={compute_cap}" if backend == "apple" else "")
    )


def resolve_worker_count(
    config_value: int,
    *,
    model_mb: int = DEFAULT_MODEL_MB,
    kv_per_slot_mb: int = DEFAULT_KV_PER_SLOT_MB,
) -> tuple[int, str]:
    """Translate the user-facing ``ocr_workers`` config (0 = auto,
    >=1 = explicit) into a concrete slot count + a human-readable
    reason string. ``model_mb`` / ``kv_per_slot_mb`` let the caller
    pass quantization-aware budgets — a Q4_K_M weight + Q8 KV cache
    costs ~3× less than the FP16 defaults, so the auto-picker can
    open way more slots when those are in use."""
    try:
        n = int(config_value)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return auto_worker_count(
            model_mb=model_mb, kv_per_slot_mb=kv_per_slot_mb,
        )
    return n, f"user-set ({n} slot(s))"
