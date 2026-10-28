# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import cv2
import numpy as np
import math
from dataclasses import dataclass
from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors import erase as _erase
from aglaia.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait
from aglaia.processors.manual import manual_value, stamp_manual
from aglaia.processors.option_specs import _b, _e, _f, _i
from aglaia.processors.utils import to_gray, is_binary

# ── the `lines` estimate (#184) ──────────────────────────────────────

#: Postl's coarse step. Below the width of its score peak — see `_find_skew_angle`.
_COARSE_STEP = 0.25
#: Height the book's crop is analysed at. The 400 px of the whole frame left
#: 2-3 px per text line on a capture where the book is half the frame.
_ANALYSIS_H = 1000
#: Line-blob filters and confidence, all at `_ANALYSIS_H`.
_MIN_LINES = 8
_MAX_SPREAD_DEG = 1.5


def _book_outline(gray):
    """``((x, y, w, h), outline_angle)``: the box the book is searched in,
    and the angle of its edges — or the whole frame and None.

    The book is the largest bright region at 1/8 scale. A region under a
    tenth of the frame is no book, and one whose box fills the frame is a
    page already cropped, whose "outline" is the frame itself: both keep the
    whole frame, with no outline angle."""
    H, W = gray.shape[:2]
    k = 8
    small = cv2.resize(gray, (max(1, W // k), max(1, H // k)),
                       interpolation=cv2.INTER_AREA)
    _, m = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return (0, 0, W, H), None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    frac = st[i, cv2.CC_STAT_AREA] / float(small.size)
    if frac < 0.1:
        return (0, 0, W, H), None
    x, y, w, h = (int(v) * k for v in st[i, :4])
    if w * h >= 0.9 * W * H:
        return (0, 0, W, H), None
    ys, xs = np.nonzero(lab == i)
    _, _, a = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
    a = float(a)
    while a > 45:
        a -= 90
    while a <= -45:
        a += 90
    return (x, y, min(w, W - x), min(h, H - y)), -a


def _text_mask(gray):
    """Ink as white on black: strokes darker than their surroundings.

    A black-hat subtracts the page from itself closed by a kernel wider than
    a stroke, so a slow gradient, a shadow or a sun patch drops out and only
    the strokes are left for Otsu."""
    k = max(3, round(15 * gray.shape[0] / _ANALYSIS_H) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    _, b = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _line_angle(mask) -> tuple[float, bool]:
    """``(skew, confident)`` from the text lines in `mask`.

    Glyphs are closed into line blobs and a line is fitted through each. The
    skew is their length-weighted median, and it is trusted when there are
    enough lines and the middle half of them agree within `_MAX_SPREAD_DEG`."""
    f = mask.shape[0] / _ANALYSIS_H
    kx, ky = max(3, int(25 * f)), max(1, int(3 * f))
    blobs = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blobs)
    angles, weights = [], []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if w < 80 * f or h > 40 * f or w < 4 * h:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        vx, vy, _, _ = cv2.fitLine(np.c_[xs, ys].astype(np.float32),
                                   cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if vx < 0:
            vx, vy = -vx, -vy
        angles.append(math.degrees(math.atan2(vy, vx)))
        weights.append(w)
    if len(angles) < _MIN_LINES:
        return 0.0, False
    a = np.asarray(angles)
    o = np.argsort(a)
    c = np.cumsum(np.asarray(weights, np.float64)[o])
    q = lambda p: float(a[o][min(len(a) - 1, np.searchsorted(c, c[-1] * p))])
    # Image y runs down: a line falling to the right is a positive atan2 and a
    # page turned clockwise, which `meta["skew"]` counts as negative.
    return -q(0.5), (q(0.75) - q(0.25)) <= _MAX_SPREAD_DEG



@dataclass
class SkewFinderOption(AbstractProcessorOption):
    max_angle: float = 30.0
    min_angle: float = 0.1
    accuracy: float = 0.1
    # If False, only finding skew angle and storing in meta, no rotation.
    # Defaulting to True to act as a processor.
    apply_rotation: bool = True
    k_cluster: int = 0
    # "lines" (#184): fit the text lines inside the book's outline. "postl":
    # the projection-profile search over the whole frame.
    method: str = "lines"

class SkewFinder(AbstractImageProcessor):
    name: str = "SkewFinder"
    SUMMARY = "Projection-profile skew estimation + rotation."
    REPLAY_TRAIT = ReplayTrait.COORDINATE  # rigid rotation (affine)
    OPTION_CLASS = SkewFinderOption
    PROVIDES_META = {
        "skew": "estimated page skew angle in degrees",
        "skew_method": "which estimate gave meta.skew: lines | outline | postl | none",
        "roi": "page quad polygon [[x,y],...] rotated into output coords",
    }
    _ESSENTIAL_PARAMS = ("max_angle", "apply_rotation", "k_cluster", "method")
    OPTIONS = {
        "max_angle": _f(30.0, 0.1, 45.0, 0.5,
                        "Search range in degrees (±)."),
        "min_angle": _f(0.1, 0.0, 5.0, 0.05,
                        "Apply rotation only when detected skew exceeds this (degrees)."),
        "accuracy": _f(0.1, 0.01, 1.0, 0.01,
                       "Fine-search angle step in degrees."),
        "apply_rotation": _b(True,
                             "If false, only records meta.skew without rotating."),
        "method": _e("lines", ["lines", "postl"],
                     "lines: fit the text lines inside the book's outline — holds "
                     "on a small, low-contrast book in a cluttered frame. "
                     "postl: projection profile over the whole frame."),
        "k_cluster": _i(0, 0, 8,
                        "0 = white border. >1 = k-means cluster count for background colour detection.",
                        advanced=True),
    }
    
    DEFAULT_MAX_ANGLE = 30.0
    DEFAULT_MIN_ANGLE = 0.1
    DEFAULT_ACCURACY = 0.1

    @classmethod
    def replay_transform(cls, params, in_wh):
        """Rigid rotation about the (size-scaled) centre → affine 3×3."""
        import cv2
        import numpy as np

        from aglaia.processors.replay_transform import AffineTransform
        w, h = in_wh
        cx, cy = params["center_xy"]
        sw, sh = params["wh"]
        cx *= w / sw
        cy *= h / sh
        M = cv2.getRotationMatrix2D((cx, cy), -float(params["angle_deg"]), 1.0)
        H = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
        return AffineTransform(H, (w, h))

    def __init__(self, options: SkewFinderOption):
        super().__init__(options)
        self.max_angle = options.max_angle
        self.min_angle = options.min_angle
        self.accuracy = options.accuracy
        self.apply_rotation = options.apply_rotation
        self.k_cluster = options.k_cluster
        self.method = getattr(options, "method", "lines")

    def process(self, img_buf: ImageBuffer) -> ImageBuffer:
        """
        Calculates skew and optionally applies rotation.
        """
        image = img_buf.buffer
        current_skew = 0.0

        # Manual angle (M9 #93): the user corrected what the estimator found,
        # so the estimator must not run at all. Run-then-discard would be
        # cheaper to write and wrong to live with — the next rerun would
        # re-derive a different angle and the correction would drift.
        manual_deg = manual_value(img_buf, "skew_deg")
        if manual_deg is not None:
            try:
                current_skew = float(manual_deg)
            except (TypeError, ValueError):
                manual_deg = None
        if manual_deg is not None:
            if not img_buf.meta:
                img_buf.meta = {}
            img_buf.meta["skew"] = current_skew
            stamp_manual(img_buf, "skew_deg")
            self._rotate(img_buf, current_skew)
            return self._finish(img_buf, current_skew)

        try:
            gray = to_gray(image)
            if self.method == "postl":
                current_skew, used = self._postl(gray), "postl"
            else:
                current_skew, used = self._estimate_lines(gray)

            if not img_buf.meta: img_buf.meta = {}
            img_buf.meta["skew"] = current_skew
            img_buf.meta["skew_method"] = used
            self._rotate(img_buf, current_skew)

        except Exception:
            # Estimation failed; leave the buffer as it came in.
            pass

        return self._finish(img_buf, current_skew)

    def _finish(self, img_buf, current_skew: float):
        """Shared tail: the op-log stats both the estimated and the manual
        path report."""
        self.last_stats = {
            "angle": f"{current_skew:+.2f}°",
            "rotated": bool(self.apply_rotation
                            and abs(current_skew) >= self.min_angle),
        }
        return img_buf

    def _rotate(self, img_buf, current_skew: float) -> None:
        """Rotate the buffer by `current_skew` and carry the ROI with it.

        Shared by the estimated and the manual path (M9 #93) — a manual angle
        must land through exactly the same warp, border and ROI transform, or
        a corrected page would differ from an estimated one by more than the
        angle."""
        if self.apply_rotation and abs(current_skew) >= self.min_angle:
            (sh, sw) = img_buf.buffer.shape[:2]
            center = (sw // 2, sh // 2)
            M = cv2.getRotationMatrix2D(center, -current_skew, 1.0)
            # Replay params: stored so the replay pass can re-apply
            # this rotation on the original colour buffer with a single
            # composite interpolation.
            img_buf.meta["replay_kind"] = "rotate"
            img_buf.meta["replay_params"] = {
                "angle_deg": float(current_skew),
                "center_xy": [float(center[0]), float(center[1])],
                "wh": [int(sw), int(sh)],
            }
            
            # Determine Border Value
            border_val = (255, 255, 255) if len(img_buf.buffer.shape) == 3 else 255
            if self.k_cluster > 1:
                 bg_color = self._detect_background_color(img_buf.buffer, self.k_cluster)
                 if bg_color is not None:
                     border_val = bg_color

            # NN on BW (no smudging across the 0/255 jump), cubic on
            # gray/colour where smooth resampling avoids stair-stepping.
            input_is_bw = img_buf.type == ImageType.BW or is_binary(img_buf.buffer)
            interp_flag = cv2.INTER_NEAREST if input_is_bw else cv2.INTER_CUBIC
            img_buf.buffer = cv2.warpAffine(img_buf.buffer, M, (sw, sh), flags=interp_flag, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
            
            # Update ROI
            # ROI is stored as list of points (x,y)
            if not img_buf.meta.get("roi"):
                # Init to full image rect if missing
                img_buf.meta["roi"] = [(0,0), (sw, 0), (sw, sh), (0, sh)]
            
            old_roi = np.array(img_buf.meta["roi"], dtype=np.float32)
            # cv2.transform expects shape (N, 1, 2) for points or (N, 2) depending on call
            # But transform needs 3x2, warpAffine does 2x3. M is 2x3.
            # cv2.transform(src, m) -> dst
            # src: array of elements to transform
            
            # Reshape for transform: (N, 1, 2)
            pts = old_roi.reshape((-1, 1, 2))
            new_pts = cv2.transform(pts, M)
            img_buf.meta["roi"] = new_pts.reshape((-1, 2)).tolist()
            # Erase masks ride the same rotation. If you transform `roi`,
            # transform `erase` on the same line — a mask left at the old
            # coordinates erases the wrong part of the page, silently.
            _polys = _erase.get(img_buf.meta)
            if _polys:
                img_buf.meta[_erase.META_KEY] = _erase.transform_affine(
                    _polys, M)

            # Check binary integrity. NEAREST warp preserves binarity,
            # CUBIC never yields a binary result — input_is_bw already
            # answers this without rescanning the rotated frame.
            if input_is_bw:
                # Re-binarize to ensure sharp edges after rotation interpolation
                _, img_buf.buffer = cv2.threshold(img_buf.to_gray(), 127, 255, cv2.THRESH_BINARY)
                img_buf.type = ImageType.BW

    def _postl(self, gray) -> float:
        """Projection profile over the whole frame, 400 px tall, global Otsu."""
        h, w = gray.shape[:2]
        if h > 400:
            gray = cv2.resize(gray, (int(w * 400 / h), 400),
                              interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return float(self._find_skew_angle(binary))

    def _estimate_lines(self, gray) -> tuple[float, str]:
        """The `lines` estimate (#184) → (angle, which estimate was used).

        The book is found first, so the table, the hands and the light around
        it never reach the estimate. Inside it a black-hat keeps the strokes
        against their local surroundings, whatever the contrast of the page.
        The text lines give the angle; when they are too few or disagree, the
        book's own outline does, then Postl."""
        box, outline = _book_outline(gray)
        x, y, w, h = box
        crop = gray[y:y + h, x:x + w]
        s = min(1.0, _ANALYSIS_H / max(1, crop.shape[0]))
        if s < 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        angle, ok = _line_angle(_text_mask(crop))
        if ok and abs(angle) <= self.max_angle:
            return angle, "lines"
        if outline is not None and abs(outline) <= self.max_angle:
            return outline, "outline"
        try:
            return self._postl(gray), "postl"
        except Exception:
            return 0.0, "none"

    def _find_skew_angle(self, binary):
        # 3. Coarse Search. The step must stay below the width of the score
        # peak (~0.2-0.3° on a 400 px frame): in 1° steps the grid landed
        # off the peak and the refinement settled on noise (#184).
        best_angle = 0
        max_score = -1
        
        for angle in np.arange(-self.max_angle, self.max_angle + _COARSE_STEP,
                               _COARSE_STEP):
            score = self._calc_score(binary, angle)
            if score > max_score:
                max_score = score
                best_angle = angle
        
        # 4. Fine Search
        refined_angle = best_angle
        # optimize range
        start = best_angle - _COARSE_STEP
        end = best_angle + _COARSE_STEP
        if start < -self.max_angle: start = -self.max_angle
        if end > self.max_angle: end = self.max_angle
        
        for angle in np.arange(start, end, self.accuracy):
            score = self._calc_score(binary, angle)
            if score > max_score:
                max_score = score
                refined_angle = angle
                
        return refined_angle

    def _calc_score(self, binary, angle):
        h, w = binary.shape
        center_x = w / 2.0
        tg = math.tan(math.radians(angle))
        
        M = np.float32([
            [1, 0, 0],
            [tg, 1, -center_x * tg]
        ])
        
        # NEAREST: a bilinear shear blurs the rows at every angle but 0°, which
        # then out-scored the true angle — Postl returned exactly 0.00 (#184).
        sheared = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        projections = np.sum(sheared, axis=1, dtype=np.int32) // 255
        diffs = np.diff(projections)
        score = np.sum(diffs**2)
        return score

    def _detect_background_color(self, image, k):
        try:
            # Downscale for performance
            h, w = image.shape[:2]
            target_dim = 100
            scale = min(1.0, target_dim / h, target_dim / w)
            if scale < 1.0:
                 small_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                 small_img = image
            
            # Reshape
            if len(small_img.shape) == 3:
                 data = small_img.reshape((-1, 3))
                 is_color = True
            else:
                 data = small_img.reshape((-1, 1))
                 is_color = False
                 
            data = np.float32(data)
            
            # K-Means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # attempts=3 for better stability
            ret, label, center = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            
            # Find lightest cluster (background assumption)
            # Sum of channels is rough brightness
            sums = np.sum(center, axis=1)
            idx = np.argmax(sums)
            
            bg = center[idx]
            
            if is_color:
                return tuple([int(c) for c in bg])
            else:
                return int(bg[0])
                
        except Exception as e:
             # print(f"Background detection failed: {e}")
             return None
