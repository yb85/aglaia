# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The dewarp debug overlay must draw the surface that was actually fitted.

`dewarp_grid_lattice` projects the remap's page lattice back onto the step's
input image. It reads the same replay stamp `PageDewarper._replay_sample_map`
reads, and it has to honour every term of it. It used to drop two — the
10-param camera's principal point (#60) and the fitted spine curl — so on a
curled page the overlay sat several hundred px off a correct output, which
reads as a broken dewarp.
"""
import numpy as np

from aglaia.processors.PageDewarper import PageDewarper
from aglaia.storage.debug_renderers import dewarp_grid_lattice


def _stamp(**over):
    params = np.zeros(20, dtype=np.float64)
    params[0:3] = [0.02, -0.03, 0.01]        # rvec
    params[3:6] = [-0.6, -0.45, 1.9]         # tvec
    params[6], params[7] = 0.25, -0.10       # curl
    params[8], params[9] = 0.08, -0.05       # principal point
    rp = {
        "params": params.tolist(),
        "page_dims": [1.30, 2.05],
        "src_shape": [2400, 1600],
        "pad_px": 60,
        "focal_length": 1.3,
        "decimate": 4,
        "zoom": 1.0,
        "camera_np": 10,
        "spine": {"gamma": 0.10, "s_x": 0.16, "x0": 1.07},
        "slope_emphasis": 1.0,
    }
    rp.update(over)
    return rp


def _corners(rp):
    """The four page corners of the overlay lattice, as pixel positions."""
    lat = dewarp_grid_lattice(rp, 18, 28)
    return np.array([lat[0, 0], lat[0, -1], lat[-1, -1], lat[-1, 0]])


def _remap_corners(rp):
    """The same four corners, straight off the remap's own sampling grid."""
    im_x, im_y, pad = PageDewarper._replay_sample_map(
        (rp["src_shape"][0] - 2 * rp["pad_px"],
         rp["src_shape"][1] - 2 * rp["pad_px"]), rp)
    return np.array([[im_x[0, 0], im_y[0, 0]],
                     [im_x[0, -1], im_y[0, -1]],
                     [im_x[-1, -1], im_y[-1, -1]],
                     [im_x[-1, 0], im_y[-1, 0]]]) - pad


def test_overlay_corners_match_the_remap_grid():
    """The contract: the overlay traces the surface the remap samples. A few
    px of slack for the coarse 18x28 lattice and the resize of the decimated
    grid — not the hundreds the dropped terms cost."""
    assert np.abs(_corners(_stamp()) - _remap_corners(_stamp())).max() < 6.0


def _shift(**over):
    a = dewarp_grid_lattice(_stamp(), 18, 28)
    b = dewarp_grid_lattice(_stamp(**over), 18, 28)
    return float(np.abs(a - b).max())


def test_principal_point_moves_the_grid():
    """Guards the #60 term specifically: with n_cam back at 8 the same stamp
    projects somewhere else, which is the bug this test was written for."""
    assert _shift(camera_np=8) > 20.0


def test_spine_curl_moves_the_grid():
    """Same guard for the spine term. Its peak sits inside the page, so read
    the whole lattice, not just the corners."""
    assert _shift(spine=None) > 20.0


def test_pre_lm_stamp_still_projects():
    """A stamp from before #60/#70 carries none of these keys; it must fall
    back to the 8-param pinhole it was fitted on rather than raise."""
    rp = _stamp()
    for k in ("camera_np", "spine", "slope_emphasis"):
        rp.pop(k)
    lat = dewarp_grid_lattice(rp, 18, 28)
    assert lat.shape == (28, 18, 2)
    assert np.isfinite(lat).all()
