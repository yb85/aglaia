# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Opt-in real-hardware round-trip for the local VLM OCR engines.

The unit tests stub the server + HTTP, so the actual spawn → health → chat path
is never exercised in CI. This module IS that exercise — it spins up a real
backend (MLX on Apple Silicon, vLLM on CUDA), runs a page through the engine,
and checks the text comes back.

It is **opt-in and slow** (downloads/loads multi-GB weights, first run is
minutes), so it is skipped unless you ask for it:

    # macOS (mlx-vlm installed, weights downloaded via the Model Downloader):
    AGLAIA_VLM_SMOKE=1 uv run pytest tests/workers/test_vlm_integration.py -s

    # CUDA box (vllm installed, weights present):
    AGLAIA_VLM_SMOKE=1 uv run pytest tests/workers/test_vlm_integration.py -s

Each engine self-skips when its backend or weights aren't ready, so on a fresh
machine you'll see skips telling you exactly what's missing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AGLAIA_VLM_SMOKE") != "1",
    reason="opt-in: set AGLAIA_VLM_SMOKE=1 (needs a VLM backend + downloaded weights)",
)


def _text_image(text: str, w: int = 1000, h: int = 300) -> np.ndarray:
    """A white page with one line of large black text — enough for a doc VLM."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ):
        try:
            font = ImageFont.truetype(path, 64)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=64)  # Pillow ≥ 10
        except Exception:
            font = ImageFont.load_default()
    draw.text((50, 110), text, fill="black", font=font)
    return np.array(img, dtype=np.uint8)


@pytest.mark.parametrize("engine_name", ["glm", "unlimited", "paddle_vl"])
def test_local_vlm_ocr_roundtrip(engine_name):
    from aglaia.workers.ocr import get_engine

    eng = get_engine(engine_name)
    if not eng.available:
        pytest.skip(f"{engine_name}: backend or weights not ready on this machine")

    phrase = "The quick brown fox"
    result = eng.recognize(_text_image(phrase), [])

    md = (result.get("meta") or {}).get("markdown", "")
    blob = (
        md + " " + " ".join(ln.get("text", "") for ln in result.get("lines", []))
    ).lower()
    # A distinctive word should survive the round-trip (don't demand a perfect
    # transcription — just prove the server served and the page was read).
    assert "quick" in blob or "brown" in blob, (
        f"{engine_name} returned no recognisable text; got markdown={md!r}"
    )
