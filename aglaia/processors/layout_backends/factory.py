# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from typing import Optional

from aglaia.processors.layout_backends.base import LayoutBackend
from aglaia.processors.layout_backends.heuristic import HeuristicBackend


def probe_active_backend(name: str = "auto") -> str:
    """Return the name of the backend `get_backend(name)` would actually
    pick — without instantiating heavy models if possible. For "auto",
    walks the same fallback chain checking file presence / module import
    cheaply; falls back to actually loading on uncertainty."""
    import sys, os
    name = (name or "auto").lower()
    if name in ("dbnet", "east", "apple_vision", "heuristic"):
        try:
            b = get_backend(name)
        except Exception:
            return "heuristic"
        return getattr(b, "name", name)

    # auto
    is_mac = sys.platform == "darwin"
    if is_mac:
        try:
            from aglaia.processors.layout_backends import apple_vision as _av
            if _av.AppleVisionBackend.is_available():  # type: ignore[attr-defined]
                return "apple_vision"
        except Exception:
            try:
                from aglaia.processors.layout_backends.apple_vision import AppleVisionBackend
                AppleVisionBackend()
                return "apple_vision"
            except Exception:
                pass
    try:
        from aglaia.processors.layout_backends.east import _resolve_model_path as _east_path
        _east_path()
        return "east"
    except Exception:
        pass
    try:
        from aglaia.processors.layout_backends.dbnet import _resolve_model_path as _db_path
        _db_path()
        return "dbnet"
    except Exception:
        pass
    return "heuristic"


def get_backend(name: str = "auto") -> LayoutBackend:
    """
    Resolve a backend by name.

    - "dbnet" — PP-OCR mobile det (~5 MB ONNX). Modern, lightweight, accurate.
                Default on non-Apple hardware.
    - "apple_vision" — Apple Vision (macOS only). Default on Apple Silicon /
                       Intel macOS.
    - "east"  — OpenCV dnn EAST text detector (~95 MB pb). Older, dated.
    - "heuristic" — projection-profile fallback. No ML deps. Cross-platform.
    - "auto" — macOS: apple_vision → east → dbnet → heuristic.
               other: east → dbnet → heuristic.
    """
    import sys
    name = (name or "auto").lower()
    if name == "dbnet":
        from aglaia.processors.layout_backends.dbnet import DbnetBackend
        return DbnetBackend()
    if name == "east":
        from aglaia.processors.layout_backends.east import EastBackend
        return EastBackend()
    if name == "apple_vision":
        from aglaia.processors.layout_backends.apple_vision import AppleVisionBackend
        return AppleVisionBackend()
    if name == "heuristic":
        return HeuristicBackend()
    if name == "auto":
        is_mac = sys.platform == "darwin"
        loaders = []
        if is_mac:
            loaders.append(lambda: __import__(
                "aglaia.processors.layout_backends.apple_vision",
                fromlist=["AppleVisionBackend"]).AppleVisionBackend())
        # EAST first (proven on existing scans) then dbnet (smaller/newer
        # but trips on dense gutter blobs).
        loaders.append(lambda: __import__(
            "aglaia.processors.layout_backends.east",
            fromlist=["EastBackend"]).EastBackend())
        loaders.append(lambda: __import__(
            "aglaia.processors.layout_backends.dbnet",
            fromlist=["DbnetBackend"]).DbnetBackend())
        for loader in loaders:
            try:
                return loader()
            except Exception:
                continue
        return HeuristicBackend()
    raise ValueError(f"Unknown layout backend: {name}")
