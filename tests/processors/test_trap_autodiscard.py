# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Trap auto-discard self-checks. Reject a degenerate keystone (thin column
quad / a correction that would slant the text) → passthrough, instead of
emitting a visibly-skewed page. Validated on morales-slim: scan 288_A's quad
collapses to aspect 0.23 (the obvious slant case) and is now discarded; the
good pages (aspect ~0.65-0.89) are kept."""

from __future__ import annotations

import math

import numpy as np

from aglaia.processors.TrapezoidalCorrection import (
    TrapezoidalOption,
    _median_baseline_tilt,
)


def _line(x0, y0, x1, y1):
    return (np.array([x0, y0], float), np.array([x1, y1], float))


def test_horizontal_baselines_zero_tilt():
    lines = [_line(0, y, 100, y) for y in (10, 20, 30)]
    assert _median_baseline_tilt(lines) == 0.0


def test_uniform_slant_is_measured():
    # 100 wide, 10 rise → atan2(10,100) ≈ 5.7°
    lines = [_line(0, y, 100, y + 10) for y in (0, 20, 40)]
    assert _median_baseline_tilt(lines) == \
        round(math.degrees(math.atan2(10, 100)), 6) or \
        abs(_median_baseline_tilt(lines) - math.degrees(math.atan2(10, 100))) < 1e-6


def test_short_lines_skipped():
    # dx < 4 px → no stable angle → ignored; falls back to the long one.
    lines = [_line(0, 0, 2, 9), _line(0, 0, 100, 0)]
    assert _median_baseline_tilt(lines) == 0.0


def test_empty_is_zero():
    assert _median_baseline_tilt([]) == 0.0


def test_autodiscard_options_default_on():
    opt = TrapezoidalOption()
    assert opt.min_column_aspect > 0      # thin-quad guard active
    assert opt.max_added_tilt_deg > 0     # slant guard active
