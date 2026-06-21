# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Replay engine: recompose every transform recorded by the forward pass
and apply it to the *original* source buffer with one composite pass.

Each transforming processor stamps `replay_kind` + `replay_params` in its
output's meta AND owns how to re-apply itself: COORDINATE processors expose
`replay_transform()` (returning a composable AffineTransform / SampleMapTransform),
PIXEL_VALUE / ROI processors expose `apply_replay()`. This module is the
generic engine — it walks a branch's node trail, orders steps by REPLAY_TRAIT,
fuses a contiguous run of COORDINATE transforms into a single interpolation,
and dispatches PIXEL_VALUE / ROI steps to the processor's own `apply_replay`.
It never special-cases a processor by name, so a plugin joins replay just by
implementing the contract on its class (see lib.processors.replay_transform
and AbstractImageProcessor).

Public entry point:
    `replay_branch(conn, scan_id, branch_path) -> (np.ndarray, dict)`

Returns the replayed image + a meta blob describing what was applied.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

import cv2
import numpy as np

from lib.processors.replay_transform import (
    AffineTransform, ReplayContext, interp_for, remap_by_coords,
)


def _decode_image(blob: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)



_TRAIT_RANK = {"coordinate": 0, "pixel_value": 1, "roi": 2}


def _node_trait(processor_name: Optional[str]) -> Optional[str]:
    """A processor's REPLAY_TRAIT value ("coordinate"/"pixel_value"/"roi") or
    None. See memory project_replay_trait_algebra.md."""
    if not processor_name:
        return None
    try:
        from lib.processors.registry import get_processor
        info = get_processor(processor_name)
        trait = getattr(info.processor_cls, "REPLAY_TRAIT", None) if info else None
        return getattr(trait, "value", None)
    except Exception:
        return None


def _trait_rank(processor_name: Optional[str]) -> int:
    """Replay-order rank from a processor's REPLAY_TRAIT: COORDINATE (0) →
    PIXEL_VALUE (1) → ROI (2). Unknown/untagged → 0 (ordered by step_idx)."""
    return _TRAIT_RANK.get(_node_trait(processor_name), 0)


def _ordered_replay_steps(nodes: list[dict]) -> list[dict]:
    """Return replay-participating nodes in trait order.

    Order is derived from each step's processor TRAIT, not a manual
    ordinal: COORDINATE maps first (pipeline order, so contiguous warps
    compose), PIXEL_VALUE ops last (a single quantisation on the final
    geometry), ROI terminal after. Ties break by pipeline `step_idx`.
    Replaces the old `replay_last` knob. Nodes without a stamped
    `replay_kind` are skipped — no transform to reapply.
    """
    sortable = []
    for n in nodes:
        meta = n.get("meta", {})
        if meta.get("replay_kind") is None:
            continue
        rank = _trait_rank(n.get("processor_name"))
        sortable.append((rank, int(n["step_idx"]), n))
    sortable.sort(key=lambda t: (t[0], t[1]))
    return [n for _, _, n in sortable]


# ── replay engine: trait dispatch + coordinate fusion ─────────────────
# Coordinate-transform replay_kind strings — used ONLY to locate the work
# region when anchoring a segment's source (see replay_branch). Application
# is dispatched by REPLAY_TRAIT, not by these strings.
_COORD_KINDS = frozenset({"resample", "rotate", "perspective", "dewarp"})


def _node_cls(processor_name):
    """The processor CLASS for a node's processor_name (via registry), or None."""
    if not processor_name:
        return None
    try:
        from lib.processors.registry import get_processor
        info = get_processor(processor_name)
        return info.processor_cls if info else None
    except Exception:
        return None


def _run_coordinate_group(buf, mask, steps, fuse=True):
    """Apply a run of COORDINATE steps in as few interpolations as possible.

    ``steps = [(processor_cls, params), …]`` in pipeline order. Each class'
    ``replay_transform(params, in_wh)`` returns an ``AffineTransform`` (a
    composable forward 3×3) or a ``SampleMapTransform`` (a nonlinear backward
    map). Contiguous affines accumulate into one pending homography; a
    sample-map folds that pending homography into its source coords for a
    single ``cv2.remap``; a trailing affine is flushed with one
    ``warpPerspective``. ``fuse=False`` flushes after every affine (one
    interpolation per step — identical geometry, the NOFUSE debug path).

    The engine is processor-agnostic: it only ever sees the two primitive
    types, so a plugin COORDINATE processor fuses with the built-ins for free.
    """
    pending = np.eye(3, dtype=np.float64)
    cur_w, cur_h = buf.shape[1], buf.shape[0]

    def _flush(buf, mask):
        nonlocal pending
        if np.allclose(pending, np.eye(3)):
            return buf, mask
        border = (255, 255, 255) if buf.ndim == 3 else 255
        out = cv2.warpPerspective(buf, pending, (cur_w, cur_h),
                                  flags=interp_for(buf),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        out_mask = cv2.warpPerspective(mask, pending, (cur_w, cur_h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pending = np.eye(3, dtype=np.float64)
        return out, out_mask

    for proc_cls, params in steps:
        warp = proc_cls.replay_transform(params, (cur_w, cur_h))
        if isinstance(warp, AffineTransform):
            pending = warp.H @ pending
            cur_w, cur_h = warp.out_wh
            if not fuse:
                buf, mask = _flush(buf, mask)
        else:  # SampleMapTransform — nonlinear; fold the pending affine in.
            im_x, im_y, pad = warp.make_map((cur_h, cur_w))
            bx = im_x - pad
            by = im_y - pad
            Hinv = np.linalg.inv(pending)
            wq = Hinv[2, 0] * bx + Hinv[2, 1] * by + Hinv[2, 2]
            sx = ((Hinv[0, 0] * bx + Hinv[0, 1] * by + Hinv[0, 2]) / wq).astype(np.float32)
            sy = ((Hinv[1, 0] * bx + Hinv[1, 1] * by + Hinv[1, 2]) / wq).astype(np.float32)
            buf, mask = remap_by_coords(buf, mask, sx, sy)
            pending = np.eye(3, dtype=np.float64)
            cur_w, cur_h = im_x.shape[1], im_x.shape[0]
    buf, mask = _flush(buf, mask)
    return buf, mask


def replay_branch(conn: sqlite3.Connection, scan_id: int,
                  branch_path: Optional[str] = None,
                  *, dpi: float = 300.0) -> tuple[np.ndarray, dict]:
    """Re-apply every transform from a branch onto the original source.

    The root image is taken from the scan's first (depth=0) node. Each
    subsequent node's `replay_params` are applied in trait order
    (COORDINATE → PIXEL_VALUE → ROI; see `_ordered_replay_steps`).

    Returns `(replayed_image, summary_meta)`.
    """
    conn.row_factory = sqlite3.Row
    # Pick the terminal node for the branch_path (default: first branch).
    if branch_path:
        terminal = conn.execute(
            "SELECT n.id, n.image_id FROM branches b "
            "JOIN nodes n ON n.id = b.terminal_node_id "
            "WHERE b.scan_id = ? AND b.branch_path = ?",
            (scan_id, branch_path),
        ).fetchone()
    else:
        terminal = conn.execute(
            "SELECT n.id, n.image_id FROM branches b "
            "JOIN nodes n ON n.id = b.terminal_node_id "
            "WHERE b.scan_id = ? LIMIT 1",
            (scan_id,),
        ).fetchone()
    if terminal is None:
        raise ValueError(f"No terminal node for scan_id={scan_id} branch_path={branch_path!r}")

    # Walk up to the root, gather nodes in pipeline order.
    nodes: list[dict] = []
    cur_id = terminal["id"]
    while cur_id is not None:
        row = conn.execute(
            "SELECT id, parent_id, step_idx, step_name, processor_name, "
            "image_id, meta_json FROM nodes WHERE id=?", (cur_id,),
        ).fetchone()
        if row is None:
            break
        meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
        nodes.append({
            "id": row["id"],
            "parent_id": row["parent_id"],
            "step_idx": row["step_idx"],
            "step_name": row["step_name"],
            "processor_name": row["processor_name"],
            "image_id": row["image_id"],
            "meta": meta,
        })
        cur_id = row["parent_id"]
    nodes.reverse()
    if not nodes:
        raise ValueError("Empty node trail")

    # Find the source. Replay starts from the opening ROI barrier of this
    # segment — typically the PageDetector crop, which branches + crops
    # and cannot be replayed as a warp; everything after it (resample,
    # deskew, trap, dewarp, binarize) is reapplied. Pre-barrier steps are
    # already baked into the crop. Anchoring on the ROI trait (not "latest
    # node with no replay_kind") means a no-op step that didn't stamp a
    # kind can't be mistaken for the source.
    xform_kinds = _COORD_KINDS | {"binarize"}
    xform_idxs = [i for i, n in enumerate(nodes)
                  if n["meta"].get("replay_kind") in xform_kinds]
    last_xform = xform_idxs[-1] if xform_idxs else len(nodes)
    roi_idxs = [i for i, n in enumerate(nodes)
                if i < last_xform and n["image_id"] is not None
                and _node_trait(n["processor_name"]) == "roi"]
    if roi_idxs:
        source_idx = roi_idxs[-1]
    else:
        # No ROI barrier (e.g. a pipeline without PageDetector): fall back
        # to the latest image-bearing non-transform node before the work.
        source_idx = 0
        for i, n in enumerate(nodes):
            if i >= last_xform:
                break
            if (n["meta"].get("replay_kind") not in xform_kinds
                    and n["image_id"] is not None):
                source_idx = i
    candidate_nodes = nodes[source_idx + 1:]

    source_node = nodes[source_idx]
    source_image_id = source_node["image_id"]
    img_row = conn.execute("SELECT blob FROM images WHERE id=?", (source_image_id,)).fetchone()
    if img_row is None:
        raise ValueError(f"No image blob for id={source_image_id}")
    buf = _decode_image(bytes(img_row["blob"]))

    # Build initial ROI mask: from the source node's `roi` meta if
    # present (set by PageDetector), else the full image. The mask
    # follows every geometric transform so the final binarize only sees
    # pixels that were *inside* the original page polygon.
    h_src, w_src = buf.shape[:2]
    mask = np.zeros((h_src, w_src), dtype=np.uint8)
    src_roi = source_node["meta"].get("roi")
    if src_roi:
        pts = np.array(src_roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    else:
        mask[:] = 255

    steps = _ordered_replay_steps(candidate_nodes)
    if not steps:
        raise ValueError("No replay-participating nodes")
    fuse = not os.environ.get("AGLAIA_REPLAY_NOFUSE")
    debug_dir = os.environ.get("AGLAIA_REPLAY_DEBUG")
    applied = []
    i = 0
    while i < len(steps):
        n = steps[i]

        def _record(node):
            applied.append({"node_id": node["id"],
                            "kind": node["meta"]["replay_kind"],
                            "step_name": node["step_name"]})

        # Fuse a contiguous run of COORDINATE steps into one interpolation.
        # Each processor describes its own transform via replay_transform();
        # the engine never special-cases a kind.
        if _node_trait(n["processor_name"]) == "coordinate":
            group = []
            while (i < len(steps)
                   and _node_trait(steps[i]["processor_name"]) == "coordinate"):
                s = steps[i]
                cls = _node_cls(s["processor_name"])
                if cls is not None:
                    group.append((cls, s["meta"]["replay_params"]))
                    _record(s)
                i += 1
            if group:
                buf, mask = _run_coordinate_group(buf, mask, group, fuse=fuse)
            continue

        # PIXEL_VALUE / ROI → the processor re-applies itself to the pixels.
        cls = _node_cls(n["processor_name"])
        if cls is None:
            i += 1
            continue
        ctx = ReplayContext(
            dpi=dpi, debug_dir=debug_dir,
            debug_tag=f"scan{scan_id:02d}_br{branch_path or '-'}")
        buf, mask = cls.apply_replay(buf, mask, n["meta"]["replay_params"], ctx)
        _record(n)
        i += 1

    return buf, {
        "replay_applied": applied,
        "n_steps_total": len(nodes),
        "n_replay_steps": len(steps),
    }
