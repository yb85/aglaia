# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
SIFT-based pattern tracker for freehand capture.

The user prints / draws a small (2 cm × 2 cm) high-contrast pattern,
places it next to the book under the camera, and registers it once
through the freehand dialog. After that, briefly covering the pattern
with a finger triggers a capture — no need to reach for keyboard or
voice. Long-duration occlusion (> ~1 s) is treated as accidental and
ignored, so the user can rearrange the page without firing shots.

Why SIFT: the registered patch is a small, planar, texture-rich region;
SIFT's scale + rotation invariance handles a casual hand-held setup,
and the descriptor matching cost stays under a few ms for the ~50–200
features a 2 cm patch produces.

Public surface:

  * `SiftTracker.register(frame_bgr, roi_xywh)` — store reference
    keypoints + descriptors taken from a patch inside `frame_bgr`.
  * `SiftTracker.update(frame_bgr)` — match against the live frame,
    return `(found, fraction_visible, matched_points)`.
  * `ClickGate.feed(fraction)` — finite-state machine that turns the
    raw "fraction visible" stream into discrete click events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackResult:
    found: bool             # True when at least `min_fraction_found` of the reference is visible
    fraction: float         # matched / registered, in [0, 1]
    points: Optional[np.ndarray]   # Nx2 float32 of matched keypoint locations in the live frame
    quad: Optional[np.ndarray] = None  # 4x2 float32 ROI corners projected via homography


class SiftTracker:
    """Holds the registered descriptor set and matches it per frame.

    `nfeatures` caps the SIFT detector output per frame. The matcher
    only ever sees features the detector returned, so a too-aggressive
    cap silently starves the per-frame match (registered descriptors
    end up matching nothing). 600 keeps the keypoint pool dense enough
    to cover the registered patch even when the live frame has lots
    of competing texture elsewhere.

    `ratio` is Lowe's nearest-vs-second-nearest descriptor ratio
    (lower = stricter).

    `max_dim` is the cap (longest-side, pixels) the tracker resizes the
    input frame down to before running SIFT. The full-frame coords are
    preserved end-to-end: ROIs / output quad come back in original
    pixels so callers don't have to think about the rescale. Default
    960 px makes detect+match roughly 4× cheaper than at 1080p with
    minimal accuracy cost for the freehand 2 cm patch."""

    def __init__(self, nfeatures: int = 600, ratio: float = 0.72,
                 min_fraction_found: float = 0.30,
                 max_dim: int = 960):
        self._detector = cv2.SIFT_create(nfeatures=nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._ratio = float(ratio)
        self._min_found = float(min_fraction_found)
        self._max_dim = int(max_dim)
        self._ref_kp: tuple = ()
        self._ref_desc: Optional[np.ndarray] = None
        self._ref_count: int = 0
        # ROI corners in original frame coords. Live keypoints are
        # rescaled up from the down-sampled detect pass in update() so
        # ref + live live in the same coordinate system end-to-end.
        self._ref_corners: Optional[np.ndarray] = None

    def _downsample(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize ``frame_bgr`` so its longest side ≤ ``max_dim``. Returns
        ``(resized, scale)`` with ``scale = down_size / orig_size``. A
        no-op when the frame is already small enough."""
        h, w = frame_bgr.shape[:2]
        m = max(h, w)
        if m <= self._max_dim:
            return frame_bgr, 1.0
        s = self._max_dim / float(m)
        new_w = int(round(w * s))
        new_h = int(round(h * s))
        return cv2.resize(frame_bgr, (new_w, new_h),
                          interpolation=cv2.INTER_AREA), s

    @property
    def registered(self) -> bool:
        return self._ref_desc is not None and self._ref_count > 0

    @property
    def ref_count(self) -> int:
        return self._ref_count

    def clear(self) -> None:
        self._ref_kp = ()
        self._ref_desc = None
        self._ref_count = 0
        self._ref_corners = None

    def register(self, frame_bgr: np.ndarray, roi_xywh: tuple[int, int, int, int]) -> int:
        """Extract SIFT keypoints inside `roi_xywh` and store them as the
        reference. Returns the number of keypoints captured (0 → registration failed).

        Both register and update operate on the **down-sampled frame**
        so SIFT keypoints live in a single scale-space. Cross-scale
        matching (register at full-res, update at down-sampled) is too
        unreliable in practice — descriptors near the edge of SIFT's
        scale-invariance range stop matching their twins.

        ROI corners are kept in original-frame coords for the caller's
        overlay + IoU; the homography is computed in down-sampled coords
        and its output is scaled back up before returning."""
        small, scale = self._downsample(frame_bgr)
        self._scale = scale
        # Clamp ROI to original-frame bounds first, then map to small.
        x, y, w, h = roi_xywh
        h_full, w_full = frame_bgr.shape[:2]
        x = max(0, min(int(x), w_full - 1))
        y = max(0, min(int(y), h_full - 1))
        w = max(8, min(int(w), w_full - x))
        h = max(8, min(int(h), h_full - y))
        xs = int(round(x * scale))
        ys = int(round(y * scale))
        ws = max(8, int(round(w * scale)))
        hs = max(8, int(round(h * scale)))
        sh, sw = small.shape[:2]
        xs = max(0, min(xs, sw - 1))
        ys = max(0, min(ys, sh - 1))
        ws = max(8, min(ws, sw - xs))
        hs = max(8, min(hs, sh - ys))
        patch = small[ys:ys + hs, xs:xs + ws]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        kp, desc = self._detector.detectAndCompute(gray, None)
        if desc is None or len(kp) < 12:
            self.clear()
            return 0
        # Shift kp from patch-crop coords into down-sampled frame coords.
        shifted: list[cv2.KeyPoint] = []
        for k in kp:
            kp_new = cv2.KeyPoint(
                k.pt[0] + xs, k.pt[1] + ys,
                k.size, k.angle, k.response, k.octave, k.class_id,
            )
            shifted.append(kp_new)
        self._ref_kp = tuple(shifted)
        self._ref_desc = desc
        self._ref_count = len(kp)
        # Original-frame corners for overlay + IoU.
        self._ref_corners = np.float32([
            [x,     y],
            [x + w, y],
            [x + w, y + h],
            [x,     y + h],
        ])
        return self._ref_count

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        """Match the live frame against the registered descriptor set."""
        if not self.registered:
            return TrackResult(False, 0.0, None, None)
        small, scale = self._downsample(frame_bgr)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        kp, desc = self._detector.detectAndCompute(gray, None)
        if desc is None or len(kp) < 4:
            return TrackResult(False, 0.0, None, None)
        try:
            matches = self._matcher.knnMatch(self._ref_desc, desc, k=2)
        except cv2.error:
            return TrackResult(False, 0.0, None, None)
        good_ref: list[int] = []
        good_live: list[int] = []
        for i, pair in enumerate(matches):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self._ratio * n.distance:
                good_ref.append(m.queryIdx)
                good_live.append(m.trainIdx)
        if not good_live:
            return TrackResult(False, 0.0, None, None)
        inv = 1.0 / scale if scale > 0 else 1.0
        # Live kp in down-sampled coords first (for homography src/dst
        # alignment with ref_kp), then scaled up for caller-facing
        # returns (overlay, IoU).
        pts_live_small = np.float32([kp[i].pt for i in good_live])
        pts_live_full = pts_live_small * inv
        fraction = len(good_live) / float(self._ref_count)
        quad: Optional[np.ndarray] = None
        if len(good_live) >= 4 and self._ref_corners is not None:
            src = np.float32([self._ref_kp[i].pt for i in good_ref]).reshape(-1, 1, 2)
            dst = pts_live_small.reshape(-1, 1, 2)
            H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                # Project ref corners through homography. We have the
                # ref corners in *original* coords, so multiply by scale
                # to enter the homography's coord space, project, and
                # scale back up to return them in original coords.
                corners_in = (self._ref_corners * scale).reshape(-1, 1, 2)
                projected_small = cv2.perspectiveTransform(corners_in, H)
                quad = projected_small.reshape(-1, 2) * inv
        return TrackResult(fraction >= self._min_found, fraction,
                            pts_live_full, quad)


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two convex quads (4×2 float arrays,
    any winding). Used by ``ClickGate`` to confirm that the pattern is
    back roughly where the user left it before they covered it."""
    a32 = np.asarray(a, dtype=np.float32).reshape(-1, 1, 2)
    b32 = np.asarray(b, dtype=np.float32).reshape(-1, 1, 2)
    try:
        inter_area, _ = cv2.intersectConvexConvex(a32, b32, handleNested=True)
    except cv2.error:
        return 0.0
    if inter_area <= 0:
        return 0.0
    a_area = abs(cv2.contourArea(a32))
    b_area = abs(cv2.contourArea(b32))
    union = a_area + b_area - inter_area
    return float(inter_area / union) if union > 0 else 0.0


class ClickGate:
    """Finite-state machine over a stream of (fraction, quad) samples
    with **adaptive thresholds** and a **geometric recovery check**.

    Why adaptive: the matching baseline depends on lighting, pattern
    texture, camera resolution, and the per-frame ``nfeatures`` cap.
    Real-world baselines we've seen range from 0.30 to 0.80, so a fixed
    `visible_thresh = 0.50` would either reject good captures (when
    baseline is low) or never go OBSCURED (when baseline is very high).

    The gate tracks a running baseline of the visible-state fraction
    (EMA on samples while ``STATE_VISIBLE``), and derives both
    thresholds from it:

      * ``visible`` re-entry: ``baseline × visible_factor``
      * ``obscured`` entry:   ``baseline × obscured_factor``

    Click rule:

      1. Pattern visible → covered (``fraction ≤ obscured``) for
         **≥ min_occlusion_s** (default 0.5 s). No upper bound — the
         user can hold the cover as long as they want.
      2. Pattern recovered (``fraction ≥ visible``).
      3. The recovered quad has IoU **≥ iou_thresh** (default 0.70)
         with the last visible quad before occlusion.

    Step 3 rejects "covered → moved pattern → uncovered" sequences:
    if the user rearranged papers under the cover, the recovered quad
    won't match the pre-cover quad and the gate stays silent. If the
    pattern is right where they left it, the click fires.

    There used to be a third ``LOST`` state that disarmed firing past
    a fixed timeout. It got dropped: long covers + same-location
    recovery is now a legitimate trigger (the IoU check already filters
    the move-the-pattern case, which is the only thing the timeout was
    really guarding against).
    """

    STATE_VISIBLE = "visible"
    STATE_OBSCURED = "obscured"

    def __init__(self,
                 visible_factor: float = 0.65,
                 obscured_factor: float = 0.35,
                 floor_visible: float = 0.20,
                 floor_obscured: float = 0.08,
                 min_occlusion_s: float = 0.5,
                 iou_thresh: float = 0.70,
                 cooldown_s: float = 0.6,
                 baseline_alpha: float = 0.12):
        self._vis_factor = float(visible_factor)
        self._obs_factor = float(obscured_factor)
        self._floor_vis = float(floor_visible)
        self._floor_obs = float(floor_obscured)
        self._min_occ = float(min_occlusion_s)
        self._iou_thresh = float(iou_thresh)
        self._cooldown = float(cooldown_s)
        self._alpha = float(baseline_alpha)
        self._state = self.STATE_VISIBLE
        self._t_enter_obscured: float = 0.0
        self._last_click_at: float = 0.0
        self._baseline: float = 0.5
        self._last_visible_quad: Optional[np.ndarray] = None

    # ── effective thresholds ─────────────────────────────────────────
    @property
    def _vis(self) -> float:
        return max(self._floor_vis, self._baseline * self._vis_factor)

    @property
    def _obs(self) -> float:
        return max(self._floor_obs, self._baseline * self._obs_factor)

    def is_lost(self) -> bool:
        # Kept for API compatibility — the LOST state was removed.
        return False

    def reset(self) -> None:
        self._state = self.STATE_VISIBLE
        self._t_enter_obscured = 0.0
        self._last_click_at = 0.0
        self._baseline = 0.5
        self._last_visible_quad = None

    def feed(self, fraction: float, quad: Optional[np.ndarray] = None) -> bool:
        now = time.monotonic()
        f = float(fraction)
        prev_state = self._state
        vis = self._vis
        obs = self._obs
        if self._state == self.STATE_VISIBLE:
            # Update baseline only while visible — obscured frames
            # would drag it down and break the next recovery.
            self._baseline = (1.0 - self._alpha) * self._baseline + self._alpha * f
            if quad is not None:
                # Remember the latest stable location so the IoU recovery
                # check has something to compare against.
                self._last_visible_quad = quad
            if f <= obs:
                self._state = self.STATE_OBSCURED
                self._t_enter_obscured = now
        else:  # STATE_OBSCURED
            dt = now - self._t_enter_obscured
            if f >= vis:
                # Recovered — decide whether the dip was click-shaped.
                self._state = self.STATE_VISIBLE
                iou = 0.0
                if (quad is not None
                        and self._last_visible_quad is not None):
                    iou = quad_iou(self._last_visible_quad, quad)
                if (dt >= self._min_occ
                        and iou >= self._iou_thresh
                        and (now - self._last_click_at) > self._cooldown):
                    self._last_click_at = now
                    self._last_visible_quad = quad
                    return True
                if quad is not None:
                    self._last_visible_quad = quad
        return False
