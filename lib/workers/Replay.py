# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Replay engine: recompose every transform recorded by the forward pass
and apply it to the *original* source buffer with one composite pass.

Each transforming processor stamps `replay_kind` and `replay_params` in
its output's meta (see SkewFinder, TrapezoidalCorrection, PageDewarper,
Binarizer). This module walks a branch's node trail, collects those
stamps in order, fuses the geometric ones into a single warp, then
binarizes once at the end. The goal is to limit the number of
interpolation passes the final pixels go through.

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


def _decode_image(blob: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)


def _interp_for(buf: np.ndarray) -> int:
    """NN for binary buffers (no smudging across the 0/255 jump); cubic
    for grayscale / colour where smooth resampling beats stair-stepping."""
    if buf.ndim == 2:
        uniq = np.unique(buf[::8, ::8])
        if uniq.size <= 2:
            return cv2.INTER_NEAREST
    return cv2.INTER_CUBIC


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


def _apply_rotate(buf: np.ndarray, mask: np.ndarray,
                  params: dict) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = params["center_xy"]
    sw, sh = params["wh"]
    h, w = buf.shape[:2]
    # If the source we are replaying onto is larger/smaller than what the
    # forward pass saw, scale the rotation centre with it.
    cx = cx * (w / sw)
    cy = cy * (h / sh)
    M = cv2.getRotationMatrix2D((cx, cy), -float(params["angle_deg"]), 1.0)
    border_val = (255, 255, 255) if buf.ndim == 3 else 255
    out = cv2.warpAffine(buf, M, (w, h),
                         flags=_interp_for(buf),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=border_val)
    out_mask = cv2.warpAffine(mask, M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    return out, out_mask


def _apply_perspective(buf: np.ndarray, mask: np.ndarray,
                       params: dict) -> tuple[np.ndarray, np.ndarray]:
    H = np.array(params["H"], dtype=np.float64)
    canvas_w, canvas_h = params["canvas_wh"]
    src_w, src_h = params["src_wh"]
    h, w = buf.shape[:2]
    # Scale the homography if we are replaying on a different source size.
    sx, sy = w / src_w, h / src_h
    S_in = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    S_out = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    H_scaled = S_out @ H @ S_in
    cw = int(round(canvas_w * sx))
    ch = int(round(canvas_h * sy))
    out = cv2.warpPerspective(
        buf, H_scaled, (cw, ch),
        flags=_interp_for(buf),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if buf.ndim == 3 else 255,
    )
    out_mask = cv2.warpPerspective(
        mask, H_scaled, (cw, ch),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return out, out_mask


def _dewarp_sample_map(in_hw: tuple[int, int],
                       params: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Backward sampling map for a dewarp step.

    Given the dewarp INPUT image size ``in_hw = (h, w)``, return
    ``(im_x, im_y, pad_px)`` — for each output-grid pixel, the float pixel
    coordinate to sample in the *padded* input (`pad_px` white border on
    every side). The map is analytic (depends only on the fitted sheet
    params + sizes, not on pixel values), so the coordinate-fusion path can
    compose an upstream warp into it without materialising the dewarp input.
    """
    from page_dewarp.dewarp import norm2pix, round_nearest_multiple
    from lib.processors.sheet_models import arclength_x, project_xy_model
    p = np.array(params["params"], dtype=np.float32)
    page_dims = np.array(params["page_dims"], dtype=np.float32)
    decimate = int(params["decimate"])
    zoom = float(params["zoom"])
    # Sheet-model versioning: absent on old nodes → cylindrical, and
    # library-default focal 1.2 (what the old remap actually used —
    # the forward pass read the restored global cfg).
    model = str(params.get("sheet_model", "cylindrical"))
    n_modes = int(params.get("spline_modes", 0))
    model_dims = params.get("model_dims") or [float(page_dims[0]),
                                              float(page_dims[1])]
    focal = float(params.get("focal_length", 1.2))
    # Twist-model data support — absent on old nodes → unclamped (legacy).
    support = params.get("support_x")
    support_y = params.get("support_y")
    support_decay = params.get("support_decay")
    # flat_spline geometry — absent on old nodes → uniform knots, no flip.
    grading = float(params.get("knot_grading", 1.0))
    flip = bool(params.get("binding_flip", False))

    pad_px = int(params["pad_px"])
    in_h, in_w = int(in_hw[0]), int(in_hw[1])
    padded_h = in_h + 2 * pad_px
    padded_w = in_w + 2 * pad_px

    target_h = 0.5 * page_dims[1] * zoom * padded_h
    target_h = round_nearest_multiple(target_h, decimate)
    if params.get("arc_len"):
        # Mirror PageDewarper's arc-length-uniform x grid (same width
        # sizing + sample spacing) — pre-arclen nodes replay legacy.
        arc_xs, arc_s = arclength_x(p, float(page_dims[0]), model=model,
                                    n_modes=n_modes, model_dims=model_dims,
                                    support=support,
                                    support_decay=support_decay,
                                    grading=grading, flip=flip)
        arc_total = float(arc_s[-1])
        target_w = round_nearest_multiple(
            target_h * arc_total / page_dims[1], decimate)
        h_small, w_small = int(target_h / decimate), int(target_w / decimate)
        page_x = np.interp(np.linspace(0.0, arc_total, w_small),
                           arc_s, arc_xs)
    else:
        target_w = round_nearest_multiple(
            target_h * page_dims[0] / page_dims[1], decimate)
        h_small, w_small = int(target_h / decimate), int(target_w / decimate)
        page_x = np.linspace(0, page_dims[0], w_small)
    page_y = np.linspace(0, page_dims[1], h_small)
    gx, gy = np.meshgrid(page_x, page_y)
    page_xy = np.hstack((
        gx.flatten().reshape((-1, 1)),
        gy.flatten().reshape((-1, 1))
    )).astype(np.float32)

    image_points = project_xy_model(page_xy, p, model=model,
                                    n_modes=n_modes, model_dims=model_dims,
                                    focal_length=focal, support=support,
                                    support_y=support_y,
                                    support_decay=support_decay,
                                    grading=grading, flip=flip)
    image_points = norm2pix((padded_h, padded_w), image_points, False)
    im_x = image_points[:, 0, 0].reshape(gx.shape)
    im_y = image_points[:, 0, 1].reshape(gx.shape)
    im_x = cv2.resize(im_x, (target_w, target_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    im_y = cv2.resize(im_y, (target_w, target_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return im_x, im_y, pad_px


def _apply_dewarp(buf: np.ndarray, mask: np.ndarray,
                  params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sequential dewarp: build the sampling map, remap the padded input."""
    im_x, im_y, pad_px = _dewarp_sample_map(buf.shape[:2], params)
    border_val = (255, 255, 255) if buf.ndim == 3 else 255
    padded = cv2.copyMakeBorder(buf, pad_px, pad_px, pad_px, pad_px,
                                cv2.BORDER_CONSTANT, value=border_val)
    padded_mask = cv2.copyMakeBorder(mask, pad_px, pad_px, pad_px, pad_px,
                                     cv2.BORDER_CONSTANT, value=0)
    out = cv2.remap(padded, im_x, im_y, _interp_for(padded), None,
                    cv2.BORDER_CONSTANT, border_val)
    out_mask = cv2.remap(padded_mask, im_x, im_y, cv2.INTER_NEAREST, None,
                         cv2.BORDER_CONSTANT, 0)
    return out, out_mask


def _apply_margin(buf: np.ndarray, mask: np.ndarray,
                  params: dict) -> tuple[np.ndarray, np.ndarray]:
    from lib.processors.MarginSetter import _crop_to_content, _enforce_width_floor
    in_w = buf.shape[1]
    cropped, bbox = _crop_to_content(buf)
    x0, y0, w0, h0 = bbox
    cropped_mask = mask[y0:y0 + h0, x0:x0 + w0]
    l, r, t, b = params["ltrb_px"]
    if (l, r, t, b) == (0, 0, 0, 0):
        out = cropped
        out_mask = cropped_mask
    else:
        border_val = 255 if cropped.ndim == 2 else (255, 255, 255)
        out = cv2.copyMakeBorder(cropped, t, b, l, r,
                                 cv2.BORDER_CONSTANT, value=border_val)
        out_mask = cv2.copyMakeBorder(cropped_mask, t, b, l, r,
                                      cv2.BORDER_CONSTANT, value=0)
    # Width floor: dewarping a curved page can only widen it; cropping
    # whitespace plus a tight pad must not shrink below the dewarp
    # output width. Match forward MarginSetter invariant.
    min_w = int(params.get("min_width_px", in_w))
    if out.shape[1] < min_w:
        out = _enforce_width_floor(out, min_w, fill=255)
        out_mask = _enforce_width_floor(out_mask, min_w, fill=0)
    return out, out_mask


def _apply_binarize(buf: np.ndarray, mask: np.ndarray, params: dict,
                    dpi: float = 300.0,
                    debug_tag: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Last-pass binarization, ROI-aware.

    `method` selects:
      * `wolf++` — mask-aware Wolf (`lib.processors.Binarizer.wolf_masked`)
        when the ROI mask has missing values; falls back to plain
        doxapy Wolf when the full frame is in-bounds (no mask gain).
      * `wolf`   — always plain doxapy Wolf, with a post-mask white
        wipe outside ROI. No missing-data correction.
      * anything else (sauvola, niblack, …) — doxapy with the post-mask
        white wipe.

    `debug_tag`: when set, dumps under `$AGLAIA_REPLAY_DEBUG/`:
        <tag>_0_input.png       — gray fed to the binariser
        <tag>_1_input_roi.png   — same with ROI polygon outlined
        <tag>_2_bw.png          — binariser output
    """
    import os
    debug_dir = os.environ.get("AGLAIA_REPLAY_DEBUG")
    if debug_dir and debug_tag:
        os.makedirs(debug_dir, exist_ok=True)
    else:
        debug_dir = None

    gray = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY) if buf.ndim == 3 else buf
    effective_mask = mask if (mask is not None and mask.any()) else \
        np.full(gray.shape, 255, dtype=np.uint8)

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{debug_tag}_0_input.png"), gray)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(effective_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)
        cv2.putText(overlay,
                    f"window={params.get('window')} k={params.get('k')} dpi={dpi}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_tag}_1_input_roi.png"), overlay)

    method = str(params.get("method", "wolf++")).lower()
    window = int(params.get("window", 30))
    k = float(params.get("k", 0.25))

    # `wolf++` opts into the mask-aware path when the mask actually has
    # missing values; on full-coverage frames it has no edge to protect,
    # so we let plain doxapy Wolf do the work (same as `wolf`).
    has_missing = (mask is not None and mask.any()
                   and int(np.count_nonzero(mask)) < mask.size)
    if method == "wolf++" and has_missing:
        from lib.processors.Binarizer import wolf_masked
        bw = wolf_masked(gray, effective_mask, window, k=k)
    else:
        doxa_method = "wolf" if method in ("wolf", "wolf++") else method
        try:
            import doxapy as dx  # noqa: F401
        except Exception:
            _, bw = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            from lib.processors.Binarizer import Binarizer, BinarizerOption
            from lib.ImageBuffer import ImageBuffer, ImageType
            opt = BinarizerOption(
                method=doxa_method, window=window, k=k, roi_shrink=0,
            )
            img = ImageBuffer(gray, ImageType.GRAY, dpi=dpi, filestem="replay",
                              path=None, parent=None)
            bw = Binarizer(opt).process(img).buffer
        bw = np.where(effective_mask > 0, bw, 255).astype(bw.dtype)

    # Same post-binarize closing the forward pass applied. Reads from
    # `replay_params["morpho_close"]` so a yaml change re-takes effect
    # on the next replay pass without a chain restart.
    morpho_n = int(params.get("morpho_close", 0) or 0)
    if morpho_n > 0:
        from lib.processors.Binarizer import morpho_close
        bw = morpho_close(bw, morpho_n)

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{debug_tag}_2_bw.png"), bw)
    return bw, mask


# ── coordinate-transform fusion ───────────────────────────────────────
#
# Contiguous COORDINATE steps compose into ONE backward remap so the output
# takes a single interpolation instead of one per warp. On by default;
# AGLAIA_REPLAY_NOFUSE=1 forces the sequential per-step path (identical
# geometry, more interpolations). The analytic maps (resample = uniform
# scale, rotate = affine, perspective = homography) compose into a single
# 3×3; a trailing dewarp folds that homography into its sampling grid.
# Anything else falls back to sequential.

_ANALYTIC_KINDS = frozenset({"resample", "rotate", "perspective"})
_COORD_KINDS = _ANALYTIC_KINDS | frozenset({"dewarp"})


def _resample_forward_H(params: dict, w: int, h: int):
    """Forward (src→dst) 3×3 + output canvas for a uniform DPI resample."""
    in_w, in_h = params["in_wh"]
    out_w, out_h = params["out_wh"]
    sx, sy = out_w / in_w, out_h / in_h
    H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    return H, (int(round(w * sx)), int(round(h * sy)))


def _rotate_forward_H(params: dict, w: int, h: int):
    """Forward (src→dst) 3×3 + output canvas for a rotate step at input size
    (w, h); the canvas is unchanged."""
    cx, cy = params["center_xy"]
    sw, sh = params["wh"]
    cx *= w / sw
    cy *= h / sh
    M = cv2.getRotationMatrix2D((cx, cy), -float(params["angle_deg"]), 1.0)
    return np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64), (w, h)


def _perspective_forward_H(params: dict, w: int, h: int):
    """Forward (src→dst) 3×3 + output canvas (w', h') for a perspective step
    replayed onto an input of size (w, h)."""
    H = np.array(params["H"], dtype=np.float64)
    canvas_w, canvas_h = params["canvas_wh"]
    src_w, src_h = params["src_wh"]
    sx, sy = w / src_w, h / src_h
    S_in = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    S_out = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    H_scaled = S_out @ H @ S_in
    return H_scaled, (int(round(canvas_w * sx)), int(round(canvas_h * sy)))


_ANALYTIC_FWD = {
    "resample": _resample_forward_H,
    "rotate": _rotate_forward_H,
    "perspective": _perspective_forward_H,
}


def _compose_analytic(steps, w: int, h: int):
    """Compose analytic forward maps into one 3×3 (src→last canvas) + the
    final canvas size, walking `steps = [(kind, params), …]` in order."""
    H = np.eye(3, dtype=np.float64)
    for k, p in steps:
        Hs, (w, h) = _ANALYTIC_FWD[k](p, w, h)
        H = Hs @ H
    return H, (w, h)


def _remap_by_coords(buf, mask, sx, sy):
    """Sample buf/mask at per-output-pixel source coords (sx, sy)."""
    border_val = (255, 255, 255) if buf.ndim == 3 else 255
    out = cv2.remap(buf, sx, sy, _interp_for(buf), None,
                    cv2.BORDER_CONSTANT, border_val)
    out_mask = cv2.remap(mask, sx, sy, cv2.INTER_NEAREST, None,
                         cv2.BORDER_CONSTANT, 0)
    return out, out_mask


def _apply_resample(buf: np.ndarray, mask: np.ndarray,
                    params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sequential uniform-scale resample (replays a DPIfixer step)."""
    _, (ow, oh) = _resample_forward_H(params, buf.shape[1], buf.shape[0])
    up = ow >= buf.shape[1]
    out = cv2.resize(buf, (ow, oh),
                     interpolation=_interp_for(buf) if up else cv2.INTER_AREA)
    out_mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return out, out_mask


_COORD_SEQ = {
    "resample": _apply_resample,
    "rotate": _apply_rotate,
    "perspective": _apply_perspective,
    "dewarp": _apply_dewarp,
}


def _apply_coordinate_group(buf: np.ndarray, mask: np.ndarray,
                            steps: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Apply a run of COORDINATE steps `[(kind, params), …]` (pipeline order)
    as a single interpolation when the shape allows, else sequentially."""
    kinds = [k for k, _ in steps]
    # Fast path: zero or more analytic maps then exactly one dewarp (last).
    if (len(steps) >= 2 and kinds[-1] == "dewarp"
            and all(k in _ANALYTIC_KINDS for k in kinds[:-1])):
        H, (w, h) = _compose_analytic(steps[:-1], buf.shape[1], buf.shape[0])
        im_x, im_y, pad_px = _dewarp_sample_map((h, w), steps[-1][1])
        bx = im_x - pad_px            # padded-B → B (last analytic output) coords
        by = im_y - pad_px
        Hinv = np.linalg.inv(H)
        wq = Hinv[2, 0] * bx + Hinv[2, 1] * by + Hinv[2, 2]
        sx = ((Hinv[0, 0] * bx + Hinv[0, 1] * by + Hinv[0, 2]) / wq).astype(np.float32)
        sy = ((Hinv[1, 0] * bx + Hinv[1, 1] * by + Hinv[1, 2]) / wq).astype(np.float32)
        return _remap_by_coords(buf, mask, sx, sy)
    # Fast path: all analytic → compose, single warpPerspective.
    if len(steps) >= 2 and all(k in _ANALYTIC_KINDS for k in kinds):
        H, out_wh = _compose_analytic(steps, buf.shape[1], buf.shape[0])
        border_val = (255, 255, 255) if buf.ndim == 3 else 255
        out = cv2.warpPerspective(buf, H, out_wh, flags=_interp_for(buf),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
        out_mask = cv2.warpPerspective(mask, H, out_wh, flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, out_mask
    # Fallback: sequential (single step, or an unsupported ordering).
    for kind, p in steps:
        buf, mask = _COORD_SEQ[kind](buf, mask, p)
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
    applied = []
    i = 0
    while i < len(steps):
        n = steps[i]
        kind = n["meta"]["replay_kind"]
        # Fuse a contiguous run of COORDINATE steps into one interpolation.
        if fuse and kind in _COORD_KINDS:
            group = []
            while i < len(steps) and steps[i]["meta"]["replay_kind"] in _COORD_KINDS:
                s = steps[i]
                group.append((s["meta"]["replay_kind"], s["meta"]["replay_params"]))
                applied.append({"node_id": s["id"],
                                "kind": s["meta"]["replay_kind"],
                                "step_name": s["step_name"]})
                i += 1
            buf, mask = _apply_coordinate_group(buf, mask, group)
            continue
        params = n["meta"]["replay_params"]
        if kind == "resample":
            buf, mask = _apply_resample(buf, mask, params)
        elif kind == "rotate":
            buf, mask = _apply_rotate(buf, mask, params)
        elif kind == "perspective":
            buf, mask = _apply_perspective(buf, mask, params)
        elif kind == "dewarp":
            buf, mask = _apply_dewarp(buf, mask, params)
        elif kind == "margin":
            buf, mask = _apply_margin(buf, mask, params)
        elif kind == "binarize":
            tag = f"scan{scan_id:02d}_br{branch_path or '-'}"
            buf, mask = _apply_binarize(buf, mask, params, dpi=dpi,
                                        debug_tag=tag)
        else:
            i += 1
            continue
        applied.append({
            "node_id": n["id"],
            "kind": kind,
            "step_name": n["step_name"],
        })
        i += 1

    return buf, {
        "replay_applied": applied,
        "n_steps_total": len(nodes),
        "n_replay_steps": len(steps),
    }
