# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Debug overlay renderer resolves a configured step name to its processor
class. A pipeline naming its PageDetector step "LayoutDetector" must still get
the layout overlay — not the "no debug renderer" default (the layout page
bboxes silently stopped showing in the debug viewer)."""

from __future__ import annotations

from aglaia.storage.debug_renderers import (
    _default_renderer,
    _page_renderer,
    _resolve_renderer,
    _skew_renderer,
)


def test_alias_resolves_to_page_renderer():
    # The configured name in real pipelines (config/pipelines/*.yaml).
    assert _resolve_renderer("LayoutDetector") is _page_renderer


def test_class_name_still_resolves():
    assert _resolve_renderer("PageDetector") is _page_renderer
    assert _resolve_renderer("SkewFinder") is _skew_renderer


def test_unknown_falls_back_to_default():
    assert _resolve_renderer("Binarizer") is _default_renderer   # no renderer
    assert _resolve_renderer("NotAProcessor") is _default_renderer
    assert _resolve_renderer("") is _default_renderer
