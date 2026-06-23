# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Round-trip + integration tests for `TrapezoidalCorrection`.

Strategy: synthesize a BW text page, apply a known homography to warp it,
run the processor on the warped buffer, then assert that after
rectification baselines are horizontal again.
"""
import math

import cv2
import numpy as np
import pytest

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors.TrapezoidalCorrection import (
    TrapezoidalCorrection, TrapezoidalOption,
)


# ───────────────────────── fixture helpers ────────────────────────────

def _make_text_page(*, w: int = 1200, h: int = 1600,
                    line_h: int = 32, line_pitch: int = 56,
                    margin: int = 80, lines: int = 22) -> np.ndarray:
    """A clean BW image with text-like "word" rectangles per line.

    Each "line" is a sequence of short horizontal bars at the line baseline,
    sized to mimic words. Thickness per column stays low so the page-dewarp
    connectivity pipeline doesn't reject them as non-text blobs.
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    word_w = 70
    word_h = max(6, min(10, line_h // 3))  # keep thickness low
    gap = 22
    for i in range(lines):
        y0 = margin + i * line_pitch
        if y0 + line_h >= h - margin:
            break
        # Random shorter line every 5th row to simulate paragraph endings.
        line_w = w - 2 * margin
        if i % 5 == 4:
            line_w = int(line_w * 0.6)
        x = margin
        # Bars sit at the baseline (bottom of the line slot).
        by = y0 + line_h - word_h
        while x + word_w <= margin + line_w:
            cv2.rectangle(img, (x, by), (x + word_w, by + word_h),
                          color=0, thickness=cv2.FILLED)
            x += word_w + gap
    return img


def _perspective_warp(img: np.ndarray, tilt_deg: float = 18.0,
                      direction: str = "horizontal") -> tuple[np.ndarray, np.ndarray]:
    """Warp `img` by a known homography simulating an off-axis camera.

    Returns `(warped, H)` where `H` maps source → warped.
    """
    h, w = img.shape[:2]
    # Build a homography: bring far corners inward proportionally to tilt.
    shrink = math.tan(math.radians(tilt_deg)) * 0.35
    if direction == "horizontal":
        # right side appears farther → shrinks vertically.
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dx = int(w * shrink * 0.5)
        dst = np.array([
            [0, 0],
            [w - 1, dx],
            [w - 1, h - 1 - dx],
            [0, h - 1],
        ], dtype=np.float32)
    else:  # vertical
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dy = int(h * shrink * 0.5)
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1 - dy, h - 1],
            [dy, h - 1],
        ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h),
                                 borderValue=255, flags=cv2.INTER_CUBIC)
    return warped, H


def _wrap_bw(arr: np.ndarray) -> ImageBuffer:
    return ImageBuffer(arr, ImageType.BW, dpi=300.0,
                       filestem="test", path=None, parent=None)


def _measure_baseline_tilts(bw: np.ndarray) -> np.ndarray:
    """Run the same smear-then-CC pass as the processor to measure baseline tilts."""
    if bw.mean() > 127:
        ink = cv2.bitwise_not(bw)
    else:
        ink = bw
    line_h = max(8, ink.shape[0] // 60)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(8, int(line_h * 6.0)), max(1, int(line_h * 0.4))),
    )
    smeared = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
    _, smeared = cv2.threshold(smeared, 127, 255, cv2.THRESH_BINARY)
    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(smeared, connectivity=8)
    tilts = []
    for i in range(1, nlabels):
        x, y, w, h, _ = stats[i]
        if w < 5 * line_h or h < line_h * 0.4:
            continue
        # Best-fit line through the bottom contour points.
        ys = np.full(w, y + h - 1, dtype=np.float32)
        xs = np.arange(x, x + w, dtype=np.float32)
        coef = np.polyfit(xs, ys, 1)
        tilts.append(math.degrees(math.atan(coef[0])))
    return np.array(tilts) if tilts else np.array([0.0])


# ───────────────────────── tests ─────────────────────────────────────

def test_processor_runs_on_clean_page_passthrough():
    """Sanity: a perfectly axis-aligned page should produce a near-identity warp."""
    img = _make_text_page()
    buf = _wrap_bw(img)
    out = TrapezoidalCorrection(TrapezoidalOption()).process(buf)
    assert out.meta.get("trapezoid_success") is True
    tilts = _measure_baseline_tilts(out.buffer)
    assert abs(np.median(tilts)) < 1.0


def test_processor_recovers_horizontal_baselines_under_horizontal_perspective():
    img = _make_text_page()
    warped, _ = _perspective_warp(img, tilt_deg=18.0, direction="horizontal")
    buf = _wrap_bw(warped)
    out = TrapezoidalCorrection(TrapezoidalOption()).process(buf)
    assert out.meta.get("trapezoid_success") is True
    tilts = _measure_baseline_tilts(out.buffer)
    median_tilt = float(np.median(tilts))
    tilt_std = float(np.std(tilts))
    # Both median and per-line spread must drop sharply versus input.
    assert abs(median_tilt) < 0.8, f"median tilt {median_tilt:.2f}° too large"
    assert tilt_std < 0.8, f"baseline tilt std {tilt_std:.2f}° too large"


def test_metric_mode_emits_recovered_aspect():
    img = _make_text_page()
    warped, _ = _perspective_warp(img, tilt_deg=15.0, direction="vertical")
    buf = _wrap_bw(warped)
    out = TrapezoidalCorrection(TrapezoidalOption()).process(buf)
    assert out.meta.get("trapezoid_success") is True
    assert out.meta.get("mode_used") in {"metric_zhang_he", "bbox"}
    assert out.meta.get("recovered_aspect_w_h") is not None


def test_passthrough_when_too_few_lines():
    """With min_line_count=50 the smear pass returns < 50 lines and we fall back."""
    img = _make_text_page(lines=8)
    buf = _wrap_bw(img)
    opt = TrapezoidalOption(min_line_count=50)
    out = TrapezoidalCorrection(opt).process(buf)
    assert out.meta.get("trapezoid_success") is False
    assert out.meta.get("mode_used") == "fallback_passthrough"
    # Passthrough should preserve buffer dimensions.
    assert out.buffer.shape == buf.buffer.shape


def test_chain_registry_includes_processor():
    from aglaia.workers.IntegratedProcessingChain import processor_registry
    assert processor_registry()["TrapezoidalCorrection"] is TrapezoidalCorrection
