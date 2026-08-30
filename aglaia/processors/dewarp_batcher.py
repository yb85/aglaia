# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""DewarpBatcher — the first `Batcher`: page-dewarp on the GPU, batched.

The dewarp is L-BFGS over an ~8-DOF sheet model. A single page is tiny and
loses to CPU (the GPU's fixed per-iteration launch overhead dominates;
measured crossover ≈ 5 pages). Solving MANY pages as one `vmap`'d, on-device
optax L-BFGS amortises that overhead — validated 8–28× over per-page scipy.

  * `solve_batch` — pad each same-bucket page, stack, `vmap` an optax L-BFGS
    whose whole loop is one `lax.scan` (no per-iteration host round-trip), run
    on the GPU, unpad. One result pvec per page.
  * `solve_one`   — the existing scipy path (`page_dewarp_padded`), used by the
    engine for tiny/tail groups below the GPU crossover.

A page's batchable `payload` is the raw problem; the BAKED config (cubic /
huber) is constant across a pipeline run and set on the batcher at
construction. The per-page runtime config the twist/spline models needed
(`model_dims`, `flat_*`) went with them — the cylindrical surface depends only
on the pvec.
`bucket_key` is the padded shape `(max_nspans, max_npts)` — `vmap` needs
identical shapes, so the existing dewarp buckets are the grouping.

JAX/optax are imported lazily inside `solve_batch` so this stays import-safe on
CPU-only / no-JAX installs (where `solve_one` still works)."""
from __future__ import annotations

from typing import Any

import numpy as np

from aglaia.processors.batching import Batcher


class DewarpBatcher(Batcher):
    batcher_key = "dewarp"
    # Dewarp crossover (cuLBFGSB size-sweep + optax prototype): GPU wins from
    # ~5 pages; below that the CPU path is faster.
    gpu_min_batch = 5
    max_batch = 48
    flush_ms = 40

    def __init__(self, *, cubic_cost: float = 0.0, huber: float = 0.0,
                 iters: int = 1500, grad_tol: float = 1e-3):
        # Baked (compile-constant) config — identical for every page of a run.
        self.cfg = dict(cubic_cost=float(cubic_cost), huber=float(huber))
        # `iters` is the MAX cap; the batched solve stops early per page once
        # its gradient norm drops below `grad_tol` (convergence). Capping high +
        # stopping early gives ~scipy quality without always paying the cap.
        self.iters = int(iters)
        self.grad_tol = float(grad_tol)

    # ── helpers ──────────────────────────────────────────────────────
    def _apply_baked(self):
        """Set the compile-constant config into page_dewarp_padded's globals so
        `_get_compiled()` bakes this batcher's objective."""
        import aglaia.processors.page_dewarp_padded as P
        P.set_cubic_cost(self.cfg["cubic_cost"])
        P.set_huber_delta(self.cfg["huber"])

    def _sizes(self, payload) -> tuple[int, int]:
        npts = int(np.asarray(payload["dstpoints"]).reshape(-1, 2).shape[0]) - 1
        nspans = int(np.asarray(payload["params"]).shape[0]) - 8 - npts
        return nspans, npts

    def bucket_key_for(self, payload) -> Any:
        """The padded shape this page batches under, or None if it exceeds the
        largest bucket (the caller solves those via `solve_one`, which prunes)."""
        import aglaia.processors.page_dewarp_padded as P
        nspans, npts = self._sizes(payload)
        return P._pick_bucket(nspans, npts)

    # ── Batcher API ──────────────────────────────────────────────────
    def solve_one(self, payload) -> np.ndarray:
        import aglaia.processors.page_dewarp_padded as P
        self._apply_baked()
        r = P._run_jax_lbfgsb_padded(
            np.asarray(payload["dstpoints"]),
            np.asarray(payload["keypoint_index"]),
            np.asarray(payload["params"]))
        return np.asarray(r.x)

    def solve_batch(self, bucket_key, payloads) -> list:
        import jax
        import jax.numpy as jnp
        import optax
        from jax import lax
        import aglaia.processors.page_dewarp_padded as P

        self._apply_baked()
        mns, mnp = bucket_key
        vag = P._get_compiled()

        x0s, dsts, kis, masks, metas = [], [], [], [], []
        for pl in payloads:
            x0, pdst, pki, mask, rns, rnp = P._pad(
                np.asarray(pl["dstpoints"]), np.asarray(pl["keypoint_index"]),
                np.asarray(pl["params"]), mns, mnp)
            x0s.append(np.asarray(x0, np.float64))
            dsts.append(pdst)
            kis.append(pki.astype(np.int32))
            masks.append(mask)
            metas.append((rns, rnp))

        maxit = self.iters
        tol = self.grad_tol

        def solve_dev(x0, dst, ki, mask):
            def fun(p):
                return vag(p, dst, ki, mask)[0]
            opt = optax.lbfgs()
            st = opt.init(x0)
            vg = optax.value_and_grad_from_state(fun)

            # Per-page convergence: iterate until the gradient norm is below tol
            # or the cap. Under vmap this runs to the batch's slowest page;
            # already-converged pages keep iterating but with ~zero gradient, so
            # their params don't drift (optax update ≈ no-op).
            def cond(carry):
                _, _, it, gnorm = carry
                return (it < maxit) & (gnorm > tol)

            def body(carry):
                p, s, it, _ = carry
                v, g = vg(p, state=s)
                upd, s = opt.update(g, s, p, value=v, grad=g, value_fn=fun)
                p = optax.apply_updates(p, upd)
                return (p, s, it + 1, jnp.linalg.norm(g))

            p, _, _, _ = lax.while_loop(
                cond, body, (x0, st, jnp.int32(0), jnp.asarray(jnp.inf)))
            return p

        batched = jax.jit(jax.vmap(solve_dev))
        res = np.asarray(batched(
            jnp.asarray(np.stack(x0s)), jnp.asarray(np.stack(dsts)),
            jnp.asarray(np.stack(kis)), jnp.asarray(np.stack(masks))))
        return [P._unpad(res[i], metas[i][0], metas[i][1], mns)
                for i in range(len(payloads))]
