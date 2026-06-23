# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/


import cv2
import numpy as np
import os
import json

class Calibrator:
    def __init__(self, board_size=(6, 9), square_size_mm=25.0):
        """
        board_size: (columns, rows) of inner corners
        square_size_mm: size of a square in millimeters

        Defaults match the board we ship + generate
        (`scripts/gen_calibration_board.py`): a 7×10 SQUARE chessboard → a
        (6, 9) inner-corner grid, 25 mm squares. KEEP IN SYNC with that script
        and the `calibration:` config defaults if the printed board changes.
        """
        self.board_size = board_size
        self.square_size_mm = square_size_mm
        
        # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm

        # Storage for multi-image calibration
        self.reset()

    def reset(self):
        """Clear accumulated samples."""
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane
        self.last_img_size = None
        self.sample_dpis = []

    def collect_sample(self, img_bgr):
        """
        Try to find chessboard corners and add them to the sample list.
        Returns: (success, msg)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        self.last_img_size = (w, h)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if not ret:
            return False, "Chessboard corners not found"
            
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Add points
        self.objpoints.append(self.objp)
        self.imgpoints.append(corners2)

        # Estimate local DPI for this sample (for averaging later)
        cpts = corners2.reshape((self.board_size[1], self.board_size[0], 2))
        h_dists = []
        for r in range(self.board_size[1]):
            for c in range(self.board_size[0] - 1):
                h_dists.append(np.linalg.norm(cpts[r, c+1] - cpts[r, c]))
        v_dists = []
        for c in range(self.board_size[0]):
            for r in range(self.board_size[1] - 1):
                v_dists.append(np.linalg.norm(cpts[r+1, c] - cpts[r, c]))
        
        avg_px_per_square = (np.mean(h_dists) + np.mean(v_dists)) / 2.0
        dpi = (avg_px_per_square / self.square_size_mm) * 25.4
        self.sample_dpis.append(dpi)
        
        return True, "Sample added"

    def finalize_calibration(self):
        """
        Run calibration on all collected points.
        Returns: (success, camera_matrix, dist_coeffs, dpi, msg)
        """
        if not self.imgpoints:
            return False, None, None, None, "No samples collected"
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.last_img_size, None, None
        )
        
        if not ret:
            return False, None, None, None, "Calibration failed"
            
        # The user requested that DPI is set ONLY from the last calibration image
        # which should have been captured with the board flat at book distance.
        final_dpi = float(self.sample_dpis[-1])
        
        return True, mtx, dist, final_dpi, "Success"

def save_calibration(mtx, dist, dpi, resolution, new_mtx=None,
                     path="config/camera_params.json", base_dpi=None,
                     zoom_at_capture=None):
    """`base_dpi` (DPI normalised to zoom=1.0) and `zoom_at_capture` let
    the runtime scale DPI by the current camera zoom factor."""
    data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "dpi": dpi,
        "resolution": resolution,
        "new_camera_matrix": new_mtx.tolist() if new_mtx is not None else None,
        "base_dpi": base_dpi,
        "zoom_at_capture": zoom_at_capture,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_calibration(path="config/camera_params.json"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                "mtx": np.array(data["camera_matrix"]),
                "dist": np.array(data["dist_coeffs"]),
                "dpi": data["dpi"],
                "resolution": data.get("resolution"),
                "new_mtx": np.array(data["new_camera_matrix"]) if data.get("new_camera_matrix") is not None else None,
                "base_dpi": data.get("base_dpi"),
                "zoom_at_capture": data.get("zoom_at_capture"),
            }
    except:
        return None
