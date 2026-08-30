# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Sheet-surface model for PageDewarper: the cylindrical sheet.

    z(x) = (α+β)x³ − (2α+β)x² + αx        (+ an optional spine-curl term)

A generalised cylinder — every horizontal slice shares one height profile,
roots pinned at x = 0 and x = 1 — as in page-dewarp, projected through an
explicit focal length and an optional principal point (`n_cam` = 10).

The `spine` term (`lm_solver.SpineCurl`, `z += γ·exp(−|x−x₀|/s)`) puts extra
curvature exactly where a book fold puts it, with one grid-searched parameter.
It is fitted by the LM solver and must be reconstructed identically here, by
the remap and by replay.

## Retired: the twist/spline family

`sine_twist`, `bspline_twist` and `flat_spline` — Fourier-sine and clamped
cubic B-spline height profiles modulated by a linear-in-y twist gain — were
removed 2026-08. They were strictly more expressive on parameter count and
strictly worse on results. Measured over the fixture corpus (3 books, 12
pages), mean deviation of the output baselines from straight:

    cylindrical + spine curl   0.923 px      <- wins on all 12 pages
    flat_spline                1.679 px
    bspline_twist              1.648 px
    sine_twist                 1.645 px

That is consistent with their design: a fold concentrates curvature at the
spine, which one localized term captures robustly, while the global spline
families spread degrees of freedom where there is no curvature, overfit noisy
spans more easily, and are harder to optimise. Do not reintroduce them without
beating the number above on the same corpus.

This is a HARD break: a `.agl` node stamped with one of those models can no
longer be replayed (`replay_transform` raises); re-process it from source.
"""
from __future__ import annotations

import numpy as np

MODEL_CYLINDRICAL = "cylindrical"

#: Names of the retired twist/spline models, kept ONLY so that replaying an
#: old node fails with an explanation instead of silently rebuilding the page
#: against the wrong surface.
RETIRED_MODELS = ("sine_twist", "bspline_twist", "flat_spline", "spline_twist",
                  "flat-spline")

# page-dewarp's runaway-stretch guard on the cubic slopes.
CURL_CLIP = 0.5


def canonical_model(model: str | None) -> str:
    """Normalise a sheet-model name, rejecting the retired ones."""
    name = str(model or MODEL_CYLINDRICAL).lower()
    if name in RETIRED_MODELS:
        raise ValueError(
            f"sheet model {name!r} was retired in 2026-08 (it lost to the "
            f"cylindrical + spine-curl model on every page of the fixture "
            f"corpus — see the module docstring). A node stamped with it "
            f"cannot be replayed; re-process the page from source.")
    if name != MODEL_CYLINDRICAL:
        raise ValueError(f"unknown sheet model {name!r}")
    return name


def cylindrical_z(x, alpha: float, beta: float, spine=None):
    """Sheet height z(x) — the cubic, plus the spine-curl term when fitted."""
    x = np.asarray(x, dtype=np.float64)
    alpha = float(np.clip(alpha, -CURL_CLIP, CURL_CLIP))
    beta = float(np.clip(beta, -CURL_CLIP, CURL_CLIP))
    z = ((alpha + beta) * x ** 3
         + (-2.0 * alpha - beta) * x ** 2 + alpha * x)
    if spine is not None:
        z = z + spine.z(x)
    return z


def project_xy_model(xy_coords: np.ndarray, pvec: np.ndarray, *,
                     model: str = MODEL_CYLINDRICAL,
                     focal_length: float = 1.2,
                     n_cam: int = 8, spine=None, **_legacy) -> np.ndarray:
    """Model-aware replacement for page_dewarp.projection.project_xy.

    Returns an (N, 1, 2) array of projected image points (pix2norm units).
    Focal length is explicit; cx = cy = 0 as in the library K matrix unless
    `n_cam` = 10, which reads the principal point from pvec[8:10] (#60 — it
    absorbs the projective camera the keystone homography composes).
    `spine` is the fitted `lm_solver.SpineCurl`, or None.

    `**_legacy` swallows the twist-model keywords (`support`, `grading`, …)
    that pre-retirement replay stamps still carry; they described a surface
    this model does not have, and `canonical_model` has already rejected the
    stamps where they meant anything.
    """
    from cv2 import projectPoints

    canonical_model(model)
    xy_coords = np.asarray(xy_coords, dtype=np.float64).reshape((-1, 2))
    z_coords = cylindrical_z(xy_coords[:, 0], pvec[6], pvec[7], spine)

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))
    cx = float(pvec[8]) if int(n_cam) > 8 else 0.0
    cy = float(pvec[9]) if int(n_cam) > 9 else 0.0
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float32)
    rvec = np.asarray(pvec[0:3], dtype=np.float64)
    tvec = np.asarray(pvec[3:6], dtype=np.float64)
    image_points, _ = projectPoints(objpoints, rvec, tvec, K, np.zeros(5))
    return image_points


def project_keypoints_model(pvec: np.ndarray, keypoint_index: np.ndarray, *,
                            model: str = MODEL_CYLINDRICAL,
                            focal_length: float = 1.2,
                            n_cam: int = 8, spine=None, **_legacy
                            ) -> np.ndarray:
    """Model-aware replacement for page_dewarp.keypoints.project_keypoints.

    `keypoint_index` must be built for the same `n_cam` (see
    `lm_solver.keypoint_index` — the library's helper hardcodes 8)."""
    xy_coords = np.asarray(pvec)[np.asarray(keypoint_index)]
    xy_coords[0, :] = 0
    return project_xy_model(xy_coords, pvec, model=model,
                            focal_length=focal_length, n_cam=n_cam,
                            spine=spine)


def arclength_x(params: np.ndarray, page_w: float, *,
                model: str = MODEL_CYLINDRICAL, n: int = 4096,
                spine=None, **_legacy) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative mid-row arc length of the sheet over x ∈ [0, page_w].

    Used to build an arc-length-uniform output x grid (paper is inextensible —
    a uniform-x remap stretches text by √(1+z′²) where the sheet is steep).
    """
    canonical_model(model)
    from aglaia.processors.geometry import dewarp_arclength_x
    return dewarp_arclength_x(params, page_w, n=n, spine=spine)


def get_page_dims_model(corners: np.ndarray, rough_dims, params: np.ndarray, *,
                        model: str = MODEL_CYLINDRICAL,
                        focal_length: float = 1.2,
                        n_cam: int = 8, spine=None, **_legacy) -> np.ndarray:
    """Model-aware replacement for page_dewarp.image.get_page_dims.

    The library version projects the bottom-right corner through its own
    project_xy, which reads the global cfg focal and knows nothing about the
    principal point or the spine term — it would fit the output dims against
    a different surface than the one that was fitted."""
    from scipy.optimize import minimize

    dst_br = np.asarray(corners[2]).flatten()

    def objective(dims):
        proj = project_xy_model(
            np.asarray(dims, dtype=np.float64).reshape(1, 2), params,
            model=model, focal_length=focal_length, n_cam=n_cam, spine=spine)
        return float(np.sum((dst_br - proj.flatten()) ** 2))

    res = minimize(objective, np.array(rough_dims, dtype=np.float64),
                   method="Powell")
    return res.x
