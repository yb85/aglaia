# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
import numpy as np
import pytest

from aglaia.processors.layout_backends import get_backend
from aglaia.processors.layout_backends.heuristic import HeuristicBackend


def _dbnet_model_present() -> bool:
    from aglaia.processors.layout_backends.dbnet import _resolve_model_path
    try:
        _resolve_model_path()
        return True
    except FileNotFoundError:
        return False


def _east_model_present() -> bool:
    from aglaia.processors.layout_backends.east import _resolve_model_path
    try:
        _resolve_model_path()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(_dbnet_model_present() or _east_model_present(),
                    reason="a neural backend model is installed; auto picks it")
def test_factory_auto_returns_heuristic_without_models():
    backend = get_backend("auto")
    if backend.name != "apple_vision":
        assert isinstance(backend, HeuristicBackend)


def test_factory_explicit_heuristic():
    assert isinstance(get_backend("heuristic"), HeuristicBackend)


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("doesnotexist")


@pytest.mark.skipif(_east_model_present(),
                    reason="east model installed; missing-model path n/a")
def test_factory_east_missing_model_raises():
    """`east` without a model file must raise FileNotFoundError."""
    import os
    saved = os.environ.pop("AGLAIA_EAST_MODEL", None)
    try:
        with pytest.raises(FileNotFoundError):
            get_backend("east")
    finally:
        if saved:
            os.environ["AGLAIA_EAST_MODEL"] = saved


@pytest.mark.skipif(_dbnet_model_present(),
                    reason="dbnet model installed; missing-model path n/a")
def test_factory_dbnet_missing_model_raises():
    """`dbnet` without a model file must raise FileNotFoundError."""
    import os
    saved = os.environ.pop("AGLAIA_DBNET_MODEL", None)
    try:
        with pytest.raises(FileNotFoundError):
            get_backend("dbnet")
    finally:
        if saved:
            os.environ["AGLAIA_DBNET_MODEL"] = saved


def _real_scan(name: str):
    """A raw two-page book capture from the bundled test documents."""
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "test_data" / name


@pytest.mark.skipif(not _dbnet_model_present(),
                    reason="no dbnet model installed")
def test_dbnet_runs_on_real_scan():
    """If model is present, instantiate + run on a real scan to verify the
    OpenCV dnn ONNX load + detect path."""
    sample = _real_scan("test_athanase/athanase_150.jpg")
    img = cv2.imread(str(sample))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    backend = get_backend("dbnet")
    boxes = backend.detect(rgb)
    # Real book scan should produce many word-level boxes; assert a sane floor.
    assert len(boxes) > 5, f"only {len(boxes)} boxes — dbnet may be misconfigured"


def _two_col_page(w: int = 800, h: int = 600) -> np.ndarray:
    """White page with two text-like ink columns.

    Real text via cv2.putText so glyphs have realistic shape + internal gaps —
    column-wise per-row ink density stays well below the heuristic's
    solid-bar threshold (0.85).
    """
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(120, h - 80, 40):
        cv2.putText(img, "Lorem ipsum dolor",
                    (70, y), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Lorem ipsum dolor",
                    (470, y), font, 0.7, (0, 0, 0), 2)
    return img


def test_heuristic_detects_two_columns():
    img = _two_col_page()
    boxes = HeuristicBackend().detect(img)
    assert len(boxes) >= 2
    # boxes should be sorted/distinguishable on x
    xs = sorted(b[0] for b in boxes)
    # left column starts well below right column
    assert xs[0] < 200 < xs[-1]


def test_heuristic_blank_page_no_boxes():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    boxes = HeuristicBackend().detect(img)
    assert boxes == []


def test_heuristic_rejects_solid_blob_alongside_text_column():
    """
    Simulates a hand covering the left half of the page: a uniform dark blob
    on the left, dashed text on the right. Heuristic must keep only the text
    column.
    """
    w, h = 1600, 800
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # solid dark blob (hand-like)
    cv2.rectangle(img, (50, 100), (650, 700), (40, 40, 40), -1)
    # real-shaped text on the right
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(120, h - 80, 40):
        cv2.putText(img, "Lorem ipsum text",
                    (920, y), font, 0.8, (0, 0, 0), 2)
    boxes = HeuristicBackend().detect(img)
    assert len(boxes) == 1, f"expected 1 box, got {len(boxes)}: {boxes}"
    x0, _, x1, _ = boxes[0]
    # bbox is on the right half, not the blob
    assert x0 > 700, f"box x0={x0} should be > 700 (blob side rejected)"


def test_heuristic_excludes_solid_top_bar_from_extent():
    """
    Page with a solid black bar at the top and dashed text rows below.
    Expected y0 must start at the text, not the bar.
    """
    w, h = 800, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # solid bar at top (book edge artifact)
    cv2.rectangle(img, (0, 0), (w, 40), (10, 10, 10), -1)
    # real-shaped text below
    text_top = 200
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(text_top, h - 80, 40):
        cv2.putText(img, "Lorem ipsum dolor sit amet",
                    (100, y), font, 0.7, (0, 0, 0), 2)
    boxes = HeuristicBackend().detect(img)
    assert boxes, "expected one column"
    _, y0, _, y1 = boxes[0]
    # y0 should be at the text, not 0 (the bar). Allow modest slack.
    assert y0 > 100, f"y0={y0} indicates solid bar leaked into extent"
    assert y1 > text_top, f"y1={y1} too low to be text"


def test_heuristic_hand_in_real_scan_is_rejected():
    """Regression: athanase_150 has a hand holding the book open in the
    bottom-left; the heuristic must not return a bbox in that x-range
    (the bare projection profile saw it as a full-width page → x0=0)."""
    sample = _real_scan("test_athanase/athanase_150.jpg")
    img = cv2.imread(str(sample))
    assert img is not None, f"fixture missing: {sample}"
    w = img.shape[1]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = HeuristicBackend().detect(rgb)
    assert boxes, "heuristic returned no boxes on athanase_150"
    # The hand sits in the left ~15 % of the frame; text-line verification
    # must keep every page box clear of it.
    hand_x = int(0.13 * w)
    for (x0, y0, x1, y1) in boxes:
        assert x0 >= hand_x, f"hand-area box leaked through: {(x0,y0,x1,y1)}"
