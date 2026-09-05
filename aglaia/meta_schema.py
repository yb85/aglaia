# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""What every `ImageBuffer.meta` key IS, so a warp knows what to do with it.

`meta` is a free-form dict that rides a page through the pipeline. Some of its
values are coordinates in the page's frame — the ROI polygon, the erase
regions — and every processor that moves pixels must move them too. Each one
did so by hand, per key, and each forgot a different one: `DPIfixer` scaled
`roi` and not `erase` (155ab37); `TrapezoidalCorrection` kept an allow-list;
`PageDewarper` did it in two separate steps. A new key, or a new processor,
and one of the N×M hand-written carries is missing again.

The fix is not a naming convention — `roi` and `erase` are indexed by the DB,
the GUI and the manual-override code, and a convention cannot be enforced —
but a **declared schema**: every key has a `MetaKind`, geometric kinds are
transformed by the warp machinery (#139) and everything else is copied. An
undeclared key cannot be carried through a warp: it is logged and dropped,
because an undeclared coordinate list passing through untransformed is exactly
the bug this exists to end, and "pass it along, it's probably fine" is the
failure mode rather than the safe default.

Plugins declare their keys through `plugin_api.declare_meta`. A test scans the
source for every `meta["…"]` literal and refuses an undeclared one, so the
schema cannot drift from the code; another checks it against the table in
`docs/imagebuffer.md`, so the documentation cannot drift from the schema.

Keys that start with an underscore are a processor's private scratch —
`PageDewarper` keeps solver state there — and are OPAQUE by rule.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np


class MetaKind(Enum):
    # ── geometric: in the page's frame, TRANSFORMED by every warp ──
    POLYGON = "polygon"        # [[x, y], …]
    POLYGONS = "polygons"      # [[[x, y], …], …]
    POINTS = "points"          # [[x, y], …], unordered
    RECT_XYWH = "rect_xywh"    # [x, y, w, h]; through an affine as its corners' bbox,
                               # refused through a nonlinear map (no such rectangle)
    # ── geometric RECORD: coordinates in the PRODUCING step's own input frame,
    #    kept for that step's debug view and never carried downstream ──
    RECORD_GEOM = "record_geom"
    # ── carried untouched ──
    SCALAR = "scalar"          # numbers, bools
    LABEL = "label"            # strings, enums, lists of strings
    OPAQUE = "opaque"          # structured payloads nobody interprets in transit

    @property
    def geometric(self) -> bool:
        return self in (MetaKind.POLYGON, MetaKind.POLYGONS,
                        MetaKind.POINTS, MetaKind.RECT_XYWH)


#: Every key the app writes to `ImageBuffer.meta`, and what it is.
META_SCHEMA: dict[str, MetaKind] = {
    # geometry that rides the page
    "roi": MetaKind.POLYGON,             # page outline (PageDetector → everyone)
    "erase": MetaKind.POLYGONS,          # regions to remove (erase.py)
    "parent_crop_xywh": MetaKind.RECT_XYWH,  # where this child sits in its parent
    # records for the producer's debug view — in ITS input frame
    "column_quad": MetaKind.RECORD_GEOM,
    "line_boxes": MetaKind.RECORD_GEOM,
    "H": MetaKind.RECORD_GEOM,           # the keystone homography itself
    "page_nums": MetaKind.RECORD_GEOM,
    # labels
    "page_side": MetaKind.LABEL,
    "erase_sources": MetaKind.LABEL,
    "manual": MetaKind.LABEL,            # which fields were hand-edited
    "manual_dropped": MetaKind.LABEL,
    "replay_kind": MetaKind.LABEL,
    "fallback_reason": MetaKind.LABEL,
    "column_edge_source": MetaKind.LABEL,
    # scalars
    "skew_angle": MetaKind.SCALAR,
    "skew": MetaKind.SCALAR,             # SkewFinder writes the GUI's name directly
    "success": MetaKind.SCALAR,          # PageDewarper, likewise
    "oob": MetaKind.SCALAR,              # PageDewarper: out-of-bounds fraction
    "char_h_frac": MetaKind.SCALAR,
    "recovered_aspect_w_h": MetaKind.SCALAR,
    "dewarp_success": MetaKind.SCALAR,
    "trapezoid_success": MetaKind.SCALAR,
    "oob_forced": MetaKind.SCALAR,
    "oob_pct": MetaKind.SCALAR,
    "status": MetaKind.SCALAR,
    "elapsed_ms": MetaKind.SCALAR,
    "disabled": MetaKind.SCALAR,
    "gpu": MetaKind.SCALAR,
    "n_baselines": MetaKind.SCALAR,
    "n_vblocks": MetaKind.SCALAR,
    "n_vp_inliers": MetaKind.SCALAR,
    "vp_inlier_frac": MetaKind.SCALAR,
    "recovered_focal_px": MetaKind.SCALAR,
    "line_source": MetaKind.LABEL,
    "mode_used": MetaKind.LABEL,
    "n_full_width": MetaKind.SCALAR,
    "stamps_found": MetaKind.SCALAR,
    # opaque payloads
    "replay_params": MetaKind.OPAQUE,    # per-step; the replay engine owns its shape
    "oob_stats": MetaKind.OPAQUE,
    "manual_overrides": MetaKind.OPAQUE,
    "manual_overrides_all": MetaKind.OPAQUE,
    "frame_wh": MetaKind.OPAQUE,         # a size, not a position — never transformed
    "erase_frame_wh": MetaKind.OPAQUE,
    "layouts_frame_wh": MetaKind.OPAQUE,
}


#: The app's own keys, frozen at import. `declare_meta` adds plugin keys to
#: `META_SCHEMA` at runtime; the docs table covers only these.
APP_KEYS: frozenset[str] = frozenset(META_SCHEMA)


def declare_meta(key: str, kind: MetaKind) -> None:
    """Register a key. Plugins call this for every key they write; a second
    declaration with a DIFFERENT kind is an error, since two producers that
    disagree about what a key is cannot both be carried correctly."""
    key = str(key)
    prev = META_SCHEMA.get(key)
    if prev is not None and prev is not kind:
        raise ValueError(f"meta key {key!r} already declared as {prev.value}, "
                         f"not {kind.value}")
    META_SCHEMA[key] = kind


def kind_of(key: str) -> Optional[MetaKind]:
    """The declared kind, OPAQUE for a private `_key`, None if undeclared."""
    if str(key).startswith("_"):
        return MetaKind.OPAQUE
    return META_SCHEMA.get(str(key))


def geometric_keys() -> frozenset[str]:
    return frozenset(k for k, v in META_SCHEMA.items() if v.geometric)


def strict() -> bool:
    """Whether an undeclared key is reported at write time. Off in
    production — a plugin with an undeclared key must not crash a run — on
    under `AGLAIA_META_STRICT=1`, which the tests set."""
    return os.environ.get("AGLAIA_META_STRICT", "") not in ("", "0")


_warned: set[str] = set()


def check_key(key: str, *, where: str = "") -> None:
    """Report an undeclared key once per process. Called by `Meta`."""
    if kind_of(key) is not None:
        return
    if key in _warned:
        return
    _warned.add(key)
    msg = (f"[meta] undeclared key {key!r}{' in ' + where if where else ''} — "
           f"declare it in aglaia/meta_schema.py (or plugin_api.declare_meta) "
           f"so a warp knows whether to transform it")
    if strict():
        raise KeyError(msg)
    print(msg)


# ── carrying meta through a warp ──────────────────────────────────────

PointMap = Callable[[np.ndarray], np.ndarray]     # (N,2) float32 → (N,2)


def transform_geometry(meta: dict, map_points: PointMap, *,
                       affine: bool = True, where: str = "") -> dict:
    """The meta a coordinate processor hands downstream.

    Geometric keys go through `map_points`; RECORD_GEOM keys are DROPPED
    (they describe the producer's input frame and are read from that node);
    SCALAR / LABEL / OPAQUE are copied; an undeclared key is logged and
    dropped. `affine=False` (a nonlinear sample map) additionally drops
    RECT_XYWH, since an axis-aligned rectangle does not survive it.

    This is the one place the rule lives. #139 makes the base class call it
    after every COORDINATE processor, using the same `replay_transform` the
    replay engine fuses warps with — one geometry, forward and replay.
    """
    out: dict = {}
    for key, value in (meta or {}).items():
        kind = kind_of(key)
        if kind is None:
            check_key(key, where=where)
            continue
        if kind is MetaKind.RECORD_GEOM:
            continue
        if not kind.geometric:
            out[key] = value
            continue
        try:
            if kind is MetaKind.POLYGON or kind is MetaKind.POINTS:
                out[key] = _map_poly(value, map_points)
            elif kind is MetaKind.POLYGONS:
                polys = [_map_poly(p, map_points) for p in (value or [])]
                out[key] = [p for p in polys if len(p) >= 3]
            elif kind is MetaKind.RECT_XYWH:
                if not affine:
                    continue
                x, y, w, h = (float(v) for v in value)
                corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                m = map_points(corners)
                x0, y0 = m.min(axis=0)
                x1, y1 = m.max(axis=0)
                out[key] = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
        except Exception as e:  # noqa: BLE001 — a bad value must not kill the page
            print(f"[meta] could not transform {key!r}: {type(e).__name__}: {e}")
    return out


def _map_poly(poly: Iterable, map_points: PointMap) -> list:
    pts = np.float32([[float(x), float(y)] for x, y in poly])
    if len(pts) == 0:
        return []
    return [[float(x), float(y)] for x, y in map_points(pts)]
