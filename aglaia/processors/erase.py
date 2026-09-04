# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Erase masks — regions the pipeline must remove, carried as metadata.

``meta["erase"]`` is a list of polygons, each ``[[x, y], …]``, in the
coordinates of the buffer that carries them. Any processor may append one; the
`Binarizer` is what finally consumes them. A library stamp across a page is the
motivating case (`StampRemover`), but nothing here knows what a stamp is — a
processor that can find *anything* it should not have found says so by adding a
polygon.

## Why a polygon and not a raster mask

Because it has to survive the geometry. Deskew rotates, keystone applies a
homography, dewarp remaps a curved sheet — and the erase region has to arrive
at the binarizer sitting exactly where the stamp is, several transforms later.
`meta["roi"]` already makes that trip; a polygon is what makes it possible, and
it costs a few dozen floats in metadata instead of a full-page array per step.

So an erase mask is the **dual of the ROI**: the ROI is the region to keep, an
erase mask is a region to drop. They ride the same rails and they are consumed
by the same two-phase treatment in the binarizer, with the polarity flipped.

## Why two phases, and why the fill is not white

Painting the stamp white *before* binarizing is the obvious move and it is
wrong twice over.

* **A hard white patch is a strong local edge.** Wolf's threshold comes from a
  local window's mean and variance. A window straddling the patch boundary sees
  a big step, and marks a black ring along it — a spurious blob exactly where
  the user asked for nothing. So the fill is the **paper level**, not white,
  with a halo at least half a Wolf window wide, so a window centred on the
  polygon edge sees only paper on the erased side.

  The `Binarizer` tried a version of this for the ROI and gave it up —
  ``"interior lighting gradients still produce ink rings"``
  (`_fill_outside_roi_with_bg`, now dead code). The lesson is not that the
  idea is wrong but that *one* paper level cannot serve a whole page: a fill
  that does not match its surroundings is exactly the step it was meant to
  remove. Here each polygon is filled from a ring measured just outside
  itself, where that value really is the paper.
* **Wolf is globally sensitive to a dark region.** Its threshold formula uses
  the image's global minimum grey. A big black stamp drags that minimum down
  and shifts thresholds across the *whole page*. Removing it before
  binarization is therefore a real statistics exclusion, not a cosmetic one —
  which is why "exclude from the statistics" and "paint it out" are not two
  competing options but two halves of one operation.

Then, *after* binarization, the polygon is filled pure white with
`LINE_8` — no anti-aliasing, because an anti-aliased edge produces mid-greys,
and a mid-grey next to a threshold is a black pixel. It is filled slightly
grown (`erase_grow`), the mirror of the ROI's `roi_shrink`, so that any
one-pixel rim the binarizer still managed to produce at the boundary goes with
it.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import cv2
import numpy as np

META_KEY = "erase"

#: How far outside the polygon the pre-binarize paper fill extends, when no
#: window size is known. Half a Wolf window is the real requirement; this is
#: the floor for callers that have no window to ask about.
DEFAULT_HALO_PX = 6


def get(meta: Optional[dict]) -> list[list[list[float]]]:
    """The erase polygons on a meta dict — always a list, never None."""
    raw = (meta or {}).get(META_KEY) or []
    out: list[list[list[float]]] = []
    for poly in raw:
        try:
            pts = [[float(x), float(y)] for x, y in poly]
        except (TypeError, ValueError):
            continue
        if len(pts) >= 3:
            out.append(pts)
    return out


def add(meta: dict, polygon: Iterable, *, source: str = "") -> None:
    """Append one polygon. Fewer than three points is not a region and is
    dropped rather than carried to a consumer that would have to guess."""
    try:
        pts = [[float(x), float(y)] for x, y in polygon]
    except (TypeError, ValueError):
        return
    if len(pts) < 3:
        return
    meta.setdefault(META_KEY, []).append(pts)
    if source:
        meta.setdefault("erase_sources", []).append(str(source))


def clear(meta: dict) -> None:
    meta.pop(META_KEY, None)
    meta.pop("erase_sources", None)


def as_mask(polys, shape, *, grow: int = 0) -> Optional[np.ndarray]:
    """Rasterise the polygons into a uint8 mask (255 = erase).

    `LINE_8`, never `LINE_AA`: an anti-aliased edge is a row of mid-greys, and
    a mid-grey beside a threshold is a black pixel — the exact spurious rim
    this whole module exists to avoid."""
    if not polys:
        return None
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    drawn = 0
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) < 3:
            continue
        cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_8)
        drawn += 1
    if not drawn or cv2.countNonZero(mask) == 0:
        return None
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=int(grow))
    return mask


# ── transport: an erase mask must arrive where the stamp is ───────────

def transform_affine(polys, matrix) -> list:
    """Carry the polygons through a 2×3 affine (deskew's rotation)."""
    if not polys:
        return []
    out = []
    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        out.append(cv2.transform(pts, matrix).reshape(-1, 2).tolist())
    return out


def transform_perspective(polys, homography) -> list:
    """Carry the polygons through a 3×3 homography (keystone)."""
    if not polys:
        return []
    H = np.array(homography, dtype=np.float64)
    out = []
    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        out.append(
            cv2.perspectiveTransform(pts, H).reshape(-1, 2).tolist())
    return out


def transform_translate(polys, dx: float, dy: float) -> list:
    if not polys:
        return []
    return [[[float(x) + dx, float(y) + dy] for x, y in poly]
            for poly in polys]


def transform_remap(polys, src_shape, map_x, map_y, *,
                    scale_x: float = 1.0, scale_y: float = 1.0) -> list:
    """Carry the polygons through a non-affine remap (the dewarp).

    There is no closed form to push points through a sampling grid, so this
    does what the ROI does: rasterise on the source, remap the raster with the
    same grid, and read the contours back. Nearest-neighbour and a decimated
    grid are fine — an erase region does not need sub-pixel edges, and it is
    grown again before use."""
    if not polys:
        return []
    out: list = []
    h, w = src_shape[:2]
    for poly in polys:
        src = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(src, [pts], 255, lineType=cv2.LINE_8)
        warped = cv2.remap(src, map_x, map_y, cv2.INTER_NEAREST, None,
                           cv2.BORDER_CONSTANT, 0)
        cnts, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 3 or cv2.contourArea(c) < 4:
                continue
            pts_f = c.reshape(-1, 2).astype(np.float64)
            pts_f[:, 0] *= scale_x
            pts_f[:, 1] *= scale_y
            out.append(pts_f.tolist())
    return out


def carry(src_meta: dict, dst_meta: dict, transform=None) -> None:
    """Move the erase polygons from one buffer's meta to the next.

    `transform` is a callable taking and returning a polygon list; without one
    the polygons are copied unchanged, which is right for a step that does not
    move pixels. A processor that DOES move pixels and forgets to pass one
    leaves the masks behind at the old coordinates — so the rule is: if you
    transform `roi`, transform `erase` on the same line."""
    polys = get(src_meta)
    if not polys:
        return
    if transform is not None:
        polys = transform(polys)
    polys = [p for p in polys if len(p) >= 3]
    if polys:
        dst_meta[META_KEY] = polys
        if src_meta.get("erase_sources"):
            dst_meta["erase_sources"] = list(src_meta["erase_sources"])


# ── consumption: the two phases ───────────────────────────────────────

#: Width of the ring, just outside each polygon, that the paper level is
#: measured from. Wide enough to average over paper texture, narrow enough
#: that the page's lighting has not changed across it.
RING_PX = 12


def _paper_level(buffer: np.ndarray, region: np.ndarray,
                 ring: np.ndarray) -> Optional[Any]:
    """The paper tone just outside one region, or None if nothing to sample.

    The 80th percentile rather than the mean: the ring may clip a line of
    text, and a mean would then be pulled dark and fill the stamp's footprint
    with grey — which binarizes to a solid black rectangle, the worst possible
    outcome. A high percentile reads the paper and ignores the ink."""
    sel = (ring > 0) & (region == 0)
    if not np.any(sel):
        return None
    if buffer.ndim == 2:
        return int(np.percentile(buffer[sel], 80))
    return np.percentile(buffer[sel].reshape(-1, buffer.shape[2]), 80,
                         axis=0).astype(buffer.dtype).tolist()


def fill_with_paper(buffer: np.ndarray, polys, *, halo_px: int = 0,
                    roi_polygon=None) -> int:
    """Phase one — replace the erased regions with the paper tone measured
    AROUND EACH ONE, before binarizing. Returns the pixels replaced.

    Not white: a hard white patch is a strong local edge, and Wolf marks a
    ring along it (see the module docstring).

    And not one value for the whole page either. The `Binarizer` once had a
    global bg-fill for the ROI and abandoned it —
    ``"interior lighting gradients still produce ink rings"`` — because a
    single paper level cannot match a page that is brighter at one edge than
    the other, and a fill that does not match its surroundings IS the step it
    was meant to avoid. So each polygon is filled from a ring measured just
    outside itself: locally, that value is the paper, and a patch filled with
    the tone of its own neighbourhood is not an edge at all.

    The halo extends the fill outside the polygon, because Wolf's window must
    be able to sit centred on the polygon edge and still see only paper on the
    erased side."""
    if buffer is None or not polys:
        return 0
    halo = max(int(halo_px), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_mask = as_mask([roi_polygon], buffer.shape) if roi_polygon else None

    # A global fallback, for a polygon whose ring falls off the page or lands
    # entirely on another erase region.
    all_mask = as_mask(polys, buffer.shape, grow=halo)
    if all_mask is None:
        return 0
    outside = all_mask == 0
    if roi_mask is not None and np.any(outside & (roi_mask > 0)):
        outside = outside & (roi_mask > 0)
    if buffer.ndim == 2:
        fallback: Any = int(np.percentile(buffer[outside], 80))
    else:
        fallback = np.percentile(
            buffer[outside].reshape(-1, buffer.shape[2]), 80,
            axis=0).astype(buffer.dtype).tolist()

    filled = 0
    for poly in polys:
        region = as_mask([poly], buffer.shape, grow=halo)
        if region is None:
            continue
        ring = cv2.dilate(region, kernel, iterations=RING_PX)
        # Never measure through another erase region — that would sample the
        # neighbouring stamp's ink.
        ring[all_mask > 0] = 0
        if roi_mask is not None:
            ring[roi_mask == 0] = 0
        paper = _paper_level(buffer, region, ring)
        buffer[region > 0] = fallback if paper is None else paper
        filled += int(cv2.countNonZero(region))
    return filled


def punch(mask: Optional[np.ndarray], polys, shape, *,
          grow: int = 2) -> Optional[np.ndarray]:
    """Phase one and two at once — for REPLAY, which has a better instrument.

    Replay carries a keep-mask alongside the image and transforms it through
    every geometric step, then binarises once at the end with `wolf_masked`,
    which skips masked-out pixels when computing its local statistics and
    whitens them afterwards.

    That is exactly what an erase region needs, and it is *better* than the
    forward pass's paper fill: a true exclusion from the statistics rather
    than a substitution that approximates one. So in replay an erase region
    is not painted at all — it is simply subtracted from the keep-mask, at
    the point in the chain where it was found, and it rides the geometry to
    the final binarize for free.

    `grow` matches `whiten`'s, so a page looks the same whether it came
    through the forward pass or a replay."""
    if not polys:
        return mask
    if mask is None:
        mask = np.full(shape[:2], 255, dtype=np.uint8)
    holes = as_mask(polys, mask.shape, grow=max(int(grow), 0))
    if holes is None:
        return mask
    out = mask.copy()
    out[holes > 0] = 0
    return out


def whiten(buffer: np.ndarray, polys, *, grow: int = 2) -> int:
    """Phase two — force the erased regions to pure white AFTER binarizing.

    `grow` is the mirror of the ROI's `roi_shrink`: it takes with it any
    one-pixel rim the binarizer still produced at the boundary. The user
    traced the polygon around the stamp with margin, so a couple of pixels
    outward costs nothing and buys a clean edge."""
    if buffer is None or not polys:
        return 0
    mask = as_mask(polys, buffer.shape, grow=max(int(grow), 0))
    if mask is None:
        return 0
    white: Any = 255 if buffer.ndim == 2 else (255,) * buffer.shape[2]
    buffer[mask > 0] = white
    return int(cv2.countNonZero(mask))
