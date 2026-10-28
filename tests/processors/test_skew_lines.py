"""SkewFinder `method: lines` (#184).

A camera capture where the book is a small, low-contrast part of the frame:
the Postl estimator on a globally thresholded 400 px frame scored the table,
the hands and the shadows, and returned +0.2/+0.3° whatever the page did.

The synthetic test builds that frame. The real-data test reruns the five
scans of `delbrel oc 14` that needed a manual angle — read from the user's
project, never committed: they are pages of a copyrighted book.
"""
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.SkewFinder import SkewFinder, SkewFinderOption


def _finder(**over):
    opts = dict(max_angle=30, min_angle=0.1, accuracy=0.1,
                apply_rotation=False, k_cluster=0)
    opts.update(over)
    return SkewFinder(SkewFinderOption(**opts))


def _capture(skew: float, seed: int = 0) -> np.ndarray:
    """A 3840x2160 frame: a dark, unevenly lit table, a bright patch, a dark
    blob for a hand, and an open book of faint text lines covering about
    half the frame, rotated so that `meta["skew"]` should read `skew`."""
    rng = np.random.default_rng(seed)
    h, w = 2160, 3840
    xx = np.mgrid[0:h, 0:w][1]
    table = 120 + 40 * (xx / w) + rng.normal(0, 6, (h, w))
    frame = np.clip(table, 0, 255).astype(np.uint8)
    cv2.ellipse(frame, (3200, 300), (900, 300), 30, 0, 360, 235, -1)   # sun
    cv2.ellipse(frame, (3000, 1300), (500, 220), -10, 0, 360, 70, -1)  # hand

    book = np.full((1700, 2100), 200, np.uint8)
    for x0 in (130, 1150):                      # two pages
        for y in range(120, 1600, 34):         # faint text lines
            x = x0
            while x < x0 + 820:
                wl = int(rng.integers(20, 90))
                cv2.rectangle(book, (x, y), (min(x + wl, x0 + 820), y + 12),
                              150, -1)
                x += wl + 14
    pad = np.full((h, w), 0, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    oy, ox = 230, 300
    pad[oy:oy + 1700, ox:ox + 2100] = book
    mask[oy:oy + 1700, ox:ox + 2100] = 255
    c = (ox + 1050, oy + 850)
    # `SkewFinder` straightens with a rotation of -skew, so a page seen at
    # `skew` is the upright page rotated by +skew.
    M = cv2.getRotationMatrix2D(c, skew, 1.0)
    pad = cv2.warpAffine(pad, M, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    frame[mask > 0] = pad[mask > 0]
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def _estimate(img: np.ndarray, **over) -> ImageBuffer:
    buf = ImageBuffer(img, ImageType.COLOR, dpi=196, filestem="t")
    return _finder(**over).process(buf)


@pytest.mark.parametrize("skew", [-2.4, -0.8, 0.0, 1.4, 3.1])
def test_lines_finds_a_small_book_in_a_cluttered_frame(skew):
    out = _estimate(_capture(skew, seed=int(abs(skew) * 10)))
    assert out.meta["skew_method"] == "lines"
    assert abs(out.meta["skew"] - skew) <= 0.3


def test_lines_is_the_default():
    assert SkewFinderOption().method == "lines"


def test_postl_stays_available_and_says_so():
    out = _estimate(_capture(1.4), method="postl")
    assert out.meta["skew_method"] == "postl"


def test_a_frame_without_text_falls_back_and_says_so():
    img = np.full((1200, 1600, 3), 120, np.uint8)
    out = _estimate(img)
    assert out.meta["skew_method"] in ("outline", "postl", "none")
    assert abs(out.meta["skew"]) <= 30


# ── real data: delbrel oc 14 (opt-in) ───────────────────────────────────

DB = Path.home() / "Desktop" / "delbrel oc 14.agl"
#: scan id → the angle the user accepted: set by eye on the first five,
#: just enough for the split — a sanity bound, not ground truth.
EXPECTED = {105: 1.38, 104: 1.23, 103: 1.05, 100: 1.71, 86: 1.72,
            50: 0.0, 20: -2.0}


def _input_of(conn, scan_id: int) -> np.ndarray:
    row = conn.execute(
        "SELECT i.blob FROM nodes n JOIN images i ON i.id = n.image_id "
        "WHERE n.scan_id = ? AND n.step_name = '01_dpi_clamp_input'",
        (scan_id,)).fetchone()
    return cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)


def _split(img: np.ndarray, skew: float):
    """Pages `PageDetector` finds once the frame is turned by `skew`, with
    the options of the project's `pages_2ppf` step. None without a model."""
    from aglaia.processors import registry
    registry._discover_once()
    pd = registry.processor_classes()["PageDetector"](
        registry.option_classes()["PageDetector"](
            margin_mm=2, max_pages=2, rescale_threshold=0.01,
            processing_dpi=100.0, backend="auto", min_contrast=0.5))
    buf = ImageBuffer(img, ImageType.COLOR, dpi=196, filestem="t")
    buf.meta = {"manual_overrides": {"skew_deg": skew}}
    buf = _finder(apply_rotation=True).process(buf)
    try:
        pd.process(buf)
    except Exception:            # no layout model on this machine
        return None
    return (pd.last_stats or {}).get("pages")


@pytest.mark.skipif(not DB.is_file(), reason="no delbrel oc 14 on ~/Desktop")
def test_delbrel_oc14_capture_deskew():
    """The bar is the split. The user's angles are no ground truth: they
    were set by eye, just far enough for the spread to split, and the two
    pages of a spread differ by up to 1° anyway (the left one curves). So
    every spread must split into its two pages, and the angle is only held
    within 1° of the user's — enough to catch a gross error. Precision is
    the synthetic test's job, where the true angle is known."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    frames = {s: _input_of(conn, s) for s in EXPECTED}
    conn.close()
    got = {s: _estimate(f).meta["skew"] for s, f in frames.items()}
    assert all(abs(got[s] - EXPECTED[s]) <= 1.0 for s in EXPECTED), got
    pages = {s: _split(frames[s], got[s]) for s in EXPECTED}
    if None not in pages.values():
        assert all(n == 2 for n in pages.values()), pages
