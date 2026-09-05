# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Erase regions reach the replay in the frame they were measured in (#142).

The stamp came back at the end of the replay. `replay_erase: 1` said the
polygon had been punched into the keep-mask, and it had — into the wrong place.

The replay anchors on the ROI barrier (the layout split). A stamp finder runs
after the DPI normalise that follows it, so its polygons are in a frame that
can be twice the anchor's size. There WAS a guard for this, and it was dead: it
looked for `meta["frame_wh"]` or a node key `image_wh`, neither of which is
ever written, so `if wh:` was always false and the polygons were punched
unmapped.

A dead guard is worse than no guard. It reported success, so the failure looked
like a matching problem in the plugin rather than a coordinate problem in the
host.

The fix maps rather than guards: every coordinate step in between already
describes itself through `replay_transform`, so composing and inverting takes
the polygons back exactly. Skipping is reserved for the case that genuinely
cannot be mapped — a producer sitting after the dewarp's nonlinear map.
"""
import numpy as np
import pytest

from aglaia.workers.Replay import _anchor_erase


def _node(step_idx, name, processor, meta):
    return {"id": step_idx, "step_idx": step_idx, "step_name": name,
            "processor_name": processor, "image_id": step_idx, "meta": meta}


def _square(x, y, s):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


def _resample(step_idx, in_wh, out_wh):
    """A DPI-normalise node: pure scale, and the step that actually sat
    between the anchor and StampRemover in the project this came from."""
    return _node(step_idx, "04_dpi_normalize_output", "DPIfixer",
                 {"replay_kind": "resample",
                  "replay_params": {"in_wh": list(in_wh), "out_wh": list(out_wh)}})


def _producer(step_idx, polys):
    return _node(step_idx, "05_stampremover", "StampRemover",
                 {"erase": polys})


class TestMappedBackToTheAnchor:
    def test_a_polygon_measured_after_a_resample_is_scaled_back(self):
        """The exact shape of the bug: anchor 624x1040, producer 1342x2237."""
        polys, note = _anchor_erase(
            {"meta": {}},
            [_resample(4, (624, 1040), (1342, 2237)),
             _producer(5, [_square(556, 1532, 240)])],
            (2237, 1342))          # anchor_shape is (h, w)… of the ANCHOR
        assert note == ""
        assert len(polys) == 1

    def test_the_mapping_is_the_inverse_of_the_forward_scale(self):
        """624/1342 ≈ 0.465. A polygon at x=1342 in the producer frame is at
        x=624 in the anchor's — not left where it was."""
        polys, note = _anchor_erase(
            {"meta": {}},
            [_resample(4, (1000, 1000), (2000, 2000)),
             _producer(5, [_square(1000, 1000, 200)])],
            (1000, 1000))
        assert note == ""
        p = np.array(polys[0], dtype=float)
        assert np.allclose(p.min(0), [500, 500], atol=1.0)
        assert np.allclose(p.max(0), [600, 600], atol=1.0)

    def test_the_dead_guard_stays_dead(self):
        """`frame_wh` and `image_wh` are never written by anything. If a future
        version starts writing one, this test says so rather than letting a
        second dead check accumulate."""
        polys, _ = _anchor_erase(
            {"meta": {}},
            [_resample(4, (500, 500), (1000, 1000)),
             _producer(5, [_square(100, 100, 50)])],
            (500, 500))
        assert polys, "a mismatched frame must be mapped, not dropped"


class TestWhenItGenuinelyCannotBeMapped:
    def test_a_producer_after_the_dewarp_is_skipped_with_a_note(self):
        """A dewarp is a nonlinear sample map. There is no inverse to compose,
        so the honest answer is to skip — and to SAY so, because a silently
        dropped erase looks exactly like a plugin that found nothing."""
        polys, note = _anchor_erase(
            {"meta": {}},
            [_node(9, "09_pages_dewarp", "PageDewarper",
                   {"replay_kind": "dewarp", "replay_params": {}}),
             _producer(10, [_square(100, 100, 50)])],
            (1000, 1000))
        assert polys == []
        assert "dewarp" in note and "before" in note

    def test_the_note_travels_rather_than_vanishing(self):
        _, note = _anchor_erase(
            {"meta": {}},
            [_node(9, "09_pages_dewarp", "PageDewarper",
                   {"replay_kind": "dewarp", "replay_params": {}}),
             _producer(10, [_square(1, 1, 5)])],
            (100, 100))
        assert note and "05_stampremover" in note


def test_a_producer_already_in_the_anchor_frame_is_untouched():
    """Nothing moved the pixels, so nothing should move the polygon."""
    poly = _square(10, 20, 30)
    polys, note = _anchor_erase({"meta": {}}, [_producer(5, [poly])],
                                (1000, 1000))
    assert note == ""
    assert np.allclose(np.array(polys[0], dtype=float),
                       np.array(poly, dtype=float))


def test_erase_on_the_anchor_itself_wins_immediately():
    poly = _square(1, 2, 3)
    polys, note = _anchor_erase({"meta": {"erase": [poly]}},
                                [_producer(5, [_square(9, 9, 9)])],
                                (100, 100))
    assert note == "" and np.allclose(np.array(polys[0], dtype=float),
                                      np.array(poly, dtype=float))


def test_no_erase_anywhere_is_not_an_error():
    assert _anchor_erase({"meta": {}}, [_resample(4, (10, 10), (20, 20))],
                         (10, 10)) == ([], "")
