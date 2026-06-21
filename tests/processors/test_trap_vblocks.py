# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0

"""Trap vertical-block segmentation + margin clustering. The column quad
must merge same-column blocks split by a section gap (keep the whole body)
and leave out off-column blocks (a differently-set apparatus / gloss) —
without discarding short body lines."""

import numpy as np

from lib.processors.geometry import (
    segment_baselines_vblocks,
    cluster_blocks_by_margins,
    detect_column_quad_from_baselines,
)


def _bl(y, x0=100.0, x1=600.0):
    """One horizontal baseline at height y spanning [x0, x1]."""
    return (np.array([x0, y], dtype=np.float64),
            np.array([x1, y], dtype=np.float64))


# ── segment_baselines_vblocks ─────────────────────────────────────────

def test_section_gap_splits_into_blocks():
    top = [_bl(100.0 + 20.0 * i) for i in range(12)]       # y 100..320
    bot = [_bl(560.0 + 20.0 * i) for i in range(12)]       # y 560..780 (big gap)
    blocks = segment_baselines_vblocks(top + bot, gap_mult=2.5)
    assert len(blocks) == 2


def test_paragraph_gap_does_not_split():
    ys = [100.0 + 20.0 * i for i in range(10)]
    ys += [ys[-1] + 40.0 + 20.0 * i for i in range(10)]    # one blank line
    assert len(segment_baselines_vblocks([_bl(y) for y in ys], gap_mult=2.5)) == 1


# ── cluster_blocks_by_margins ─────────────────────────────────────────

def test_same_margin_blocks_merge():
    # Two blocks, identical L/R margins → one column.
    b0 = list(range(0, 12))
    b1 = list(range(12, 24))
    baselines = ([_bl(100.0 + 20.0 * i) for i in range(12)]
                 + [_bl(560.0 + 20.0 * i) for i in range(12)])
    clusters = cluster_blocks_by_margins(baselines, [b0, b1], x_tol=20.0)
    assert len(clusters) == 1 and len(clusters[0]) == 24


def test_different_margin_block_stays_separate():
    body = [_bl(100.0 + 20.0 * i, 100.0, 600.0) for i in range(12)]
    narrow = [_bl(560.0 + 20.0 * i, 200.0, 500.0) for i in range(8)]  # inset
    baselines = body + narrow
    clusters = cluster_blocks_by_margins(
        baselines, [list(range(12)), list(range(12, 20))], x_tol=20.0)
    assert len(clusters) == 2
    assert len(clusters[0]) == 12   # body is dominant (largest first)


# ── end-to-end: dominant column, merged, off-column excluded ──────────

def test_quad_merges_body_split_by_gap():
    # Body column split by a section gap → quad must span BOTH halves.
    top = [_bl(100.0 + 20.0 * i, 100.0, 600.0) for i in range(12)]
    bot = [_bl(620.0 + 20.0 * i, 100.0, 600.0) for i in range(12)]
    result = detect_column_quad_from_baselines(top + bot)
    assert result is not None
    quad, info = result
    assert info["n_all"] == 24                       # nothing discarded
    assert quad[:, 1].max() > 800.0                  # reaches the bottom half


def test_quad_excludes_offcolumn_apparatus():
    body = [_bl(100.0 + 20.0 * i, 100.0, 600.0) for i in range(24)]   # y 100..560
    appar = [_bl(760.0 + 12.0 * i, 220.0, 470.0) for i in range(12)]  # inset block
    result = detect_column_quad_from_baselines(body + appar)
    assert result is not None
    quad, info = result
    # The endpoint-clustering estimator keeps every line in `baselines` but
    # fits the column on the body only: the inset apparatus (left 220 / right
    # 470) sits inside the body's outermost margins (100 / 600) so it never
    # enters the fitted column.
    assert info["n_full_width"] == 24                # apparatus not in column
    assert quad[:, 1].max() < 640.0                  # quad stays on the body


def test_vblock_disabled_via_env(monkeypatch):
    monkeypatch.setenv("AGLAIA_TRAP_VBLOCK", "0")
    body = [_bl(100.0 + 20.0 * i, 100.0, 600.0) for i in range(24)]
    appar = [_bl(760.0 + 12.0 * i, 220.0, 470.0) for i in range(12)]
    _, info = detect_column_quad_from_baselines(body + appar)
    assert info["n_all"] == 36                       # merged again when off
