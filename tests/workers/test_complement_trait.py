# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""DirectBlockOCR trait + dynamic apple_docs complement resolution: any
registered direct-block engine is an eligible complement (no hardcoded list)."""

from __future__ import annotations

import aglaia.workers.ocr  # noqa: F401 — registers the built-in engines
from aglaia.workers.ocr.apple_docs import resolve_complement
from aglaia.workers.ocr.engine import direct_block_engines


def test_direct_block_membership():
    dbe = set(direct_block_engines())
    # The local recognisers (VLMs + Surya) qualify…
    assert {"surya", "glm", "unlimited"} <= dbe
    # …the Vision engines (what apple_docs complements) and cloud do not.
    assert "apple_docs" not in dbe
    assert "apple_vision" not in dbe
    assert "mistral_cloud" not in dbe


def test_resolve_complement_accepts_any_direct_block():
    assert resolve_complement("glm") == "glm"          # not just surya/paddle now
    assert resolve_complement("unlimited") == "unlimited"
    assert resolve_complement("surya") == "surya"
    assert resolve_complement("none") == "none"


def test_resolve_complement_default_and_invalid():
    assert resolve_complement("bogus") == "surya"      # invalid → default
    assert resolve_complement(None) == "surya"


def test_resolve_complement_env(monkeypatch):
    monkeypatch.setenv("AGLAIA_OCR_COMPLEMENT", "glm")
    assert resolve_complement() == "glm"
    monkeypatch.setenv("AGLAIA_OCR_COMPLEMENT", "nope")
    assert resolve_complement() == "surya"             # invalid env → default
