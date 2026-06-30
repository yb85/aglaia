# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PageDewarper warm-start curl cache.

The cold seed sets cubic_slopes = [0, 0]; the optimiser can early-exit there
(a flat local optimum → page left undewarped). The worker-local cache seeds
the curl DOFs from the last same-side fit so the next page starts in the right
basin. Only the global shape transfers (cubic slopes pvec[6:8] + spline tail);
pose and per-span keypoints stay page-specific."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aglaia.processors.PageDewarper import DewarpOption, PageDewarper


def _d():
    return PageDewarper(DewarpOption(backend="powell", sheet_model="cylindrical"))


def _buf(side):
    return SimpleNamespace(meta={"page_side": side} if side else {})


def test_seed_is_noop_until_a_side_has_solved():
    d = _d()
    p = np.zeros(20, dtype=np.float32)
    d._seed_warm_curl(p, 0, _buf("left"))
    assert np.array_equal(p[6:8], [0.0, 0.0])   # empty cache → cold stays cold


def test_remember_then_seed_same_side():
    d = _d()
    solved = np.zeros(20, dtype=np.float32)
    solved[6:8] = [0.123, -0.456]
    d._remember_warm_curl(solved, 0, _buf("left"))

    cold = np.zeros(20, dtype=np.float32)
    d._seed_warm_curl(cold, 0, _buf("left"))
    assert np.allclose(cold[6:8], [0.123, -0.456])     # curl transferred
    assert np.array_equal(cold[:6], np.zeros(6))        # pose untouched


def test_median_of_recent_ignores_a_single_runaway():
    # Two stable fits + one runaway. The robust median seed must track the
    # stable pages, NOT the runaway — else over-curl compounds page-to-page.
    d = _d()
    for cubic in ([0.10, 0.20], [0.10, 0.20], [5.0, 8.0]):
        p = np.zeros(20, dtype=np.float32)
        p[6:8] = cubic
        d._remember_warm_curl(p, 0, _buf("left"))
    cold = np.zeros(20, dtype=np.float32)
    d._seed_warm_curl(cold, 0, _buf("left"))
    assert np.allclose(cold[6:8], [0.10, 0.20], atol=1e-3)


def test_seed_magnitude_is_capped():
    # A run of large fits: even the median is big, so the cap must bound the
    # seed magnitude (the hard runaway guard).
    d = _d()
    for _ in range(3):
        p = np.zeros(20, dtype=np.float32)
        p[6:8] = [1.0, 1.0]            # |curl| = √2 ≫ cap
        d._remember_warm_curl(p, 0, _buf("left"))
    cold = np.zeros(20, dtype=np.float32)
    d._seed_warm_curl(cold, 0, _buf("left"))
    assert np.linalg.norm(cold[6:8]) <= PageDewarper._WARM_CURL_MAX + 1e-5
    assert np.linalg.norm(cold[6:8]) == pytest.approx(
        PageDewarper._WARM_CURL_MAX, abs=1e-5)


def test_other_side_not_seeded():
    d = _d()
    solved = np.zeros(20, dtype=np.float32)
    solved[6:8] = [0.2, 0.3]
    d._remember_warm_curl(solved, 0, _buf("left"))
    cold = np.zeros(20, dtype=np.float32)
    d._seed_warm_curl(cold, 0, _buf("right"))           # different side
    assert np.array_equal(cold[6:8], [0.0, 0.0])


def test_spline_extras_transfer_when_length_matches():
    d = _d()
    n_extra = 3
    solved = np.zeros(8 + 5 + n_extra, dtype=np.float32)
    solved[6:8] = [0.1, 0.1]
    solved[-n_extra:] = [1.0, 2.0, 3.0]                 # coeffs + gamma tail
    d._remember_warm_curl(solved, n_extra, _buf("single"))

    cold = np.zeros(8 + 5 + n_extra, dtype=np.float32)
    d._seed_warm_curl(cold, n_extra, _buf("single"))
    assert np.allclose(cold[-n_extra:], [1.0, 2.0, 3.0])
    # A mismatching n_extra must NOT corrupt the tail.
    cold2 = np.zeros(8 + 5 + 2, dtype=np.float32)
    d._seed_warm_curl(cold2, 2, _buf("single"))
    assert np.array_equal(cold2[-2:], [0.0, 0.0])


def test_non_finite_curl_is_ignored():
    d = _d()
    bad = np.zeros(20, dtype=np.float32)
    bad[6:8] = [np.nan, np.inf]
    d._remember_warm_curl(bad, 0, _buf("left"))
    cold = np.zeros(20, dtype=np.float32)
    d._seed_warm_curl(cold, 0, _buf("left"))
    assert np.array_equal(cold[6:8], [0.0, 0.0])        # NaN/inf never seeded
