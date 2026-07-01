# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Whole-page complement fallback: when >30% of a page's lines need the
complement, one clean whole-page VLM pass replaces the per-block splice."""

from __future__ import annotations

import numpy as np

import aglaia.workers.ocr.apple_docs as ad


class _FakeVLM:
    available = True

    def recognize(self, arr, languages, *, src_dpi=None):
        return {"lines": [{"text": "clean whole page", "bbox": [0, 0, 100, 100],
                           "confidence": 1.0}], "meta": {"markdown": "clean"}}


def _line(text, conf, y):
    return {"text": text, "confidence": conf, "bbox": [0, y, 100, y + 10]}


def test_whole_page_when_majority_flagged(monkeypatch):
    monkeypatch.setattr(ad, "get_engine", lambda c: _FakeVLM())
    arr = np.zeros((120, 120, 3), np.uint8)
    # 6/10 lines are Cyrillic garble in a fr+el doc → 60% > 30% → whole page.
    lines = [_line("propre substance", 0.99, i * 10) for i in range(4)]
    lines += [_line("Ібо той Лохои", 0.99, (4 + i) * 10) for i in range(6)]
    document = [{"type": "para", "text": "x"}]
    n, comp = ad._apply_complement(arr, lines, document, "glm", ["fr-FR", "el-GR"])
    assert comp == "glm"
    assert len(lines) == 1 and lines[0]["text"] == "clean whole page"
    assert document == []          # doc tree dropped → md falls back to lines


def test_per_block_when_minority_flagged(monkeypatch):
    # 1/10 flagged (10% < 30%) → NOT whole-page (fake VLM would return the
    # sentinel line; instead the block path runs and finds nothing to splice
    # from this tiny synthetic image → keeps Vision text).
    monkeypatch.setattr(ad, "get_engine", lambda c: _FakeVLM())
    arr = np.zeros((120, 120, 3), np.uint8)
    lines = [_line("propre substance", 0.99, i * 10) for i in range(9)]
    lines += [_line("Ібо", 0.99, 90)]
    document = [{"type": "para", "text": "x"}]
    ad._apply_complement(arr, lines, document, "glm", ["fr-FR", "el-GR"])
    # not replaced by the single whole-page sentinel line
    assert len(lines) == 10
