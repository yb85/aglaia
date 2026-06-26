# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Pluggable GPU batching for pipeline processors.

A processor whose per-item work is a small GPU optimisation (page dewarp is
the first) wins big by solving MANY items in one batched GPU call — but a
single small problem loses to CPU (measured: dewarp crossover ≈ 5 items;
1 item → CPU ~3.6x faster). This module is the GENERIC backbone so any such
processor opts in without the chain knowing about it:

  * `BatchableTrait`  — mixin a processor adds: declares its `Batcher`, and
    how to split its work into a request + a result-application.
  * `Batcher`         — the op-specific solver (one instance per `batcher_key`,
    lives in the single GPU-owner process): `solve_batch` (GPU, vmap) and
    `solve_one` (CPU fallback).
  * `BatchEngine`     — op-agnostic runtime: accumulates requests per
    `bucket_key`, flushes at `max_batch` OR `flush_ms`, routes large groups to
    the GPU and small/tail groups to the CPU. This is the load-bearing,
    unit-tested logic; the chain just feeds it submit()/poll().

The chain integration (park a batched item, submit, resume on result) is built
ON this — see IntegratedProcessingChain. This module has NO chain or JAX
dependency so it stays import-safe everywhere; `Batcher` subclasses pull in
their heavy deps lazily inside `solve_batch`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BatchItem:
    """One unit of batchable work. `payload` is opaque to the engine — only
    the owning Batcher interprets it. `bucket_key` groups items the Batcher
    can solve together in one call (e.g. same padded array shape for vmap)."""
    bucket_key: Any
    payload: Any


class Batcher(ABC):
    """Op-specific batched solver. One instance per `batcher_key`, owned by the
    GPU process. Subclass for each batchable processor (DewarpBatcher, …)."""

    #: Distinct key → one Batcher/engine. Same-key processors share a batcher.
    batcher_key: str = "batcher"
    #: Activation hint for the worker: only batch when the pending-queue depth
    #: exceeds this (below it, batching can't fill, so solve inline). The chain
    #: reads this; the engine does not.
    min_queue: int = 5
    #: Flush a bucket once it reaches this many items.
    max_batch: int = 32
    #: Flush a partial bucket this long after its FIRST item arrived (the tail
    #: safety-net — without it, stragglers never form a full batch and hang).
    flush_ms: int = 40
    #: Groups smaller than this go to `solve_one` (CPU); the GPU loses below
    #: its crossover, so a 2-item flush should not pay GPU launch overhead.
    gpu_min_batch: int = 5

    @abstractmethod
    def solve_batch(self, bucket_key: Any, payloads: list) -> list:
        """Solve all `payloads` (same bucket) in ONE batched GPU call.
        Returns one result per payload, order-aligned."""

    @abstractmethod
    def solve_one(self, payload: Any) -> Any:
        """Solve a single payload on the CPU (fallback for tiny/tail groups)."""


@dataclass
class _Group:
    items: list = field(default_factory=list)   # (request_id, payload)
    opened_ms: float = 0.0


class BatchEngine:
    """Accumulate → flush → dispatch. Op-agnostic; deterministic & clock-
    injected so it unit-tests without real time or a GPU.

    Usage (in the solver process):
        eng = BatchEngine(DewarpBatcher())
        eng.submit(req_id, BatchItem(bucket, payload), now_ms)
        for req_id, result in eng.poll(now_ms): ...     # ready results
        for req_id, result in eng.flush_all(now_ms): ... # on shutdown
    """

    def __init__(self, batcher: Batcher,
                 clock: Optional[Callable[[], float]] = None):
        self.batcher = batcher
        self._groups: dict[Any, _Group] = {}
        if clock is None:
            import time
            clock = lambda: time.monotonic() * 1000.0
        self._clock = clock

    def submit(self, request_id: Any, item: BatchItem,
               now_ms: Optional[float] = None) -> None:
        now = self._clock() if now_ms is None else now_ms
        g = self._groups.get(item.bucket_key)
        if g is None:
            g = self._groups[item.bucket_key] = _Group(opened_ms=now)
        g.items.append((request_id, item.payload))

    def _solve_group(self, bucket_key: Any, items: list) -> list:
        """Dispatch one ready group: GPU batch if large enough, else per-item CPU."""
        req_ids = [r for r, _ in items]
        payloads = [p for _, p in items]
        if len(payloads) >= self.batcher.gpu_min_batch:
            results = self.batcher.solve_batch(bucket_key, payloads)
        else:
            results = [self.batcher.solve_one(p) for p in payloads]
        return list(zip(req_ids, results))

    def _ready(self, key: Any, g: _Group, now: float, force: bool) -> bool:
        if not g.items:
            return False
        if force:
            return True
        if len(g.items) >= self.batcher.max_batch:
            return True
        return (now - g.opened_ms) >= self.batcher.flush_ms

    def poll(self, now_ms: Optional[float] = None) -> list:
        """Return (request_id, result) for every bucket that is full or timed
        out. Call regularly (e.g. each loop tick) to drain ready work."""
        now = self._clock() if now_ms is None else now_ms
        out: list = []
        for key in list(self._groups.keys()):
            g = self._groups[key]
            if self._ready(key, g, now, force=False):
                out.extend(self._solve_group(key, g.items))
                del self._groups[key]
        return out

    def flush_all(self, now_ms: Optional[float] = None) -> list:
        """Drain EVERY pending group regardless of size/age (shutdown / final
        page). Tiny groups still take the CPU path."""
        now = self._clock() if now_ms is None else now_ms
        out: list = []
        for key in list(self._groups.keys()):
            g = self._groups[key]
            if g.items:
                out.extend(self._solve_group(key, g.items))
            del self._groups[key]
        return out

    @property
    def pending(self) -> int:
        return sum(len(g.items) for g in self._groups.values())


class BatchableTrait:
    """Mixin a processor adds to opt into GPU batching. The chain checks
    `isinstance(proc, BatchableTrait)`; if so it MAY (by queue depth) split the
    work: `to_request` → submit to the Batcher → `apply_result` on the result.
    A processor's normal inline path is just
    `apply_result(buf, batcher.solve_one(to_request(buf).payload))`."""

    #: Must match the Batcher's `batcher_key`.
    batcher_key: str = "batcher"

    def make_batcher(self) -> Batcher:
        """Construct THIS processor's Batcher (called once at pipeline setup,
        in the GPU-owner process)."""
        raise NotImplementedError

    def to_request(self, buffer) -> BatchItem:
        """Extract the batchable problem from the buffer."""
        raise NotImplementedError

    def apply_result(self, buffer, result):
        """Splice a solved result back and finish the step (e.g. remap),
        returning the output ImageBuffer — exactly as inline `run` would."""
        raise NotImplementedError
