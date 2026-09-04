# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The debug view keeps its composites across a rerun (#106).

Committing a dewarp slider reruns the branch, and the rebuild used to clear
`_overlay_bytes` for as long as the background render took. With no overlay
the row falls back to the bare stage image plus the light Qt overlay — so the
source | output picture with the fitted green grid was replaced by a single
frame and a red sheet contour, precisely while the user was comparing the
live slider preview against it.

Carrying the previous composites over is only safe row-by-row: a rerun that
changed the chain's shape must not pair a row with another step's picture.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from aglaia.gui.DebugViewerTab import DebugViewerWidget          # noqa: E402


class _Strip:
    def __init__(self, n: int):
        self._n = n

    def count(self) -> int:
        return self._n


def _viewer(procs, n=None):
    """A viewer with just the fields `_adopt_stale_overlays` touches."""
    v = DebugViewerWidget.__new__(DebugViewerWidget)
    v._row_keys = [(1, "", p) for p in procs]
    v.strip = _Strip(len(procs) if n is None else n)
    v._overlay_bytes = []
    v._overlay_geom = []
    return v


def test_same_chain_keeps_every_composite():
    procs = ["SkewFinder", "PageDetector", "PageDewarper"]
    v = _viewer(procs)
    v._adopt_stale_overlays([b"a", b"b", b"c"],
                            [{"scale": 1}, {"scale": 2}, {"curl": {}}],
                            list(procs))
    assert v._overlay_bytes == [b"a", b"b", b"c"]
    assert v._overlay_geom[2] == {"curl": {}}


def test_a_changed_chain_drops_the_rows_that_moved():
    """A step disabled between runs shifts everything after it. Those rows
    fall back to the bare image rather than show another step's picture."""
    v = _viewer(["SkewFinder", "PageDewarper", "Binarizer"])
    v._adopt_stale_overlays([b"a", b"b", b"c"],
                            [{"i": 0}, {"i": 1}, {"i": 2}],
                            ["SkewFinder", "PageDetector", "PageDewarper"])
    assert v._overlay_bytes == [b"a", None, None]
    assert v._overlay_geom == [{"i": 0}, {}, {}]


def test_a_longer_new_chain_pads_rather_than_indexes_off_the_end():
    v = _viewer(["SkewFinder", "PageDetector", "PageDewarper", "Binarizer"])
    v._adopt_stale_overlays([b"a", b"b"], [{"i": 0}, {"i": 1}],
                            ["SkewFinder", "PageDetector"])
    assert v._overlay_bytes == [b"a", b"b", None, None]
    assert len(v._overlay_geom) == 4


def test_a_first_open_has_nothing_to_carry():
    v = _viewer(["SkewFinder"])
    v._adopt_stale_overlays([], [], [])
    assert v._overlay_bytes == []
    assert v._overlay_geom == []
