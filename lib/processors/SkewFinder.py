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
from lib.ImageBuffer import ImageBuffer, ImageType
from lib.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption, ReplayTrait
from lib.processors.option_specs import _b, _f, _i
from lib.processors.utils import to_gray, is_binary

@dataclass
class SkewFinderOption(AbstractProcessorOption):
    max_angle: float = 30.0
    min_angle: float = 0.1
    accuracy: float = 0.1
    # If False, only finding skew angle and storing in meta, no rotation.
    # Defaulting to True to act as a processor.
    apply_rotation: bool = True
    k_cluster: int = 0

class SkewFinder(AbstractImageProcessor):
    name: str = "SkewFinder"
    SUMMARY = "Projection-profile skew estimation + rotation."
    REPLAY_TRAIT = ReplayTrait.COORDINATE  # rigid rotation (affine)
    OPTION_CLASS = SkewFinderOption
    _ESSENTIAL_PARAMS = ("max_angle", "apply_rotation", "k_cluster")
    OPTIONS = {
        "max_angle": _f(30.0, 0.1, 45.0, 0.5,
                        "Search range in degrees (±). Coarse search runs in 1° steps then refines."),
        "min_angle": _f(0.1, 0.0, 5.0, 0.05,
                        "Apply rotation only when detected skew exceeds this (degrees)."),
        "accuracy": _f(0.1, 0.01, 1.0, 0.01,
                       "Fine-search angle step in degrees."),
        "apply_rotation": _b(True,
                             "If false, only records meta.skew without rotating."),
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

        from lib.processors.replay_transform import AffineTransform
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

    def process(self, img_buf: ImageBuffer) -> ImageBuffer:
        """
        Calculates skew and optionally applies rotation.
        """
        image = img_buf.buffer
        current_skew = 0.0
        
        try:
            # 1. Estimation — downscale FIRST so the gray conversion and
            # Otsu run on the 400-px analysis image instead of the full
            # frame (estimation only; the output buffer is untouched).
            h, w = image.shape[:2]
            target_h = 400
            small = image
            if h > target_h:
                scale = target_h / h
                small = cv2.resize(image, (int(w * scale), target_h),
                                   interpolation=cv2.INTER_AREA)
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small

            # Threshold to binary for detection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            angle = self._find_skew_angle(binary)
            current_skew = angle
            
            # Update Meta
            if not img_buf.meta: img_buf.meta = {}
            img_buf.meta["skew"] = current_skew

            # 2. Application
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

                # Check binary integrity. NEAREST warp preserves binarity,
                # CUBIC never yields a binary result — input_is_bw already
                # answers this without rescanning the rotated frame.
                if input_is_bw:
                    # Re-binarize to ensure sharp edges after rotation interpolation
                    _, img_buf.buffer = cv2.threshold(img_buf.to_gray(), 127, 255, cv2.THRESH_BINARY)
                    img_buf.type = ImageType.BW

        except Exception as e:
            # print(f"SkewFinder Error: {e}")
            pass

        self.last_stats = {
            "angle": f"{current_skew:+.2f}°",
            "rotated": bool(self.apply_rotation
                             and abs(current_skew) >= self.min_angle),
        }
        return img_buf

    def _find_skew_angle(self, binary):
        # 3. Coarse Search
        best_angle = 0
        max_score = -1
        
        for angle in np.arange(-self.max_angle, self.max_angle + 1.0, 1.0):
            score = self._calc_score(binary, angle)
            if score > max_score:
                max_score = score
                best_angle = angle
        
        # 4. Fine Search
        refined_angle = best_angle
        # optimize range
        start = best_angle - 1.0
        end = best_angle + 1.0
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
        
        sheared = cv2.warpAffine(binary, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
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
