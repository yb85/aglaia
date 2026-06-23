"""Real-data validation of the clustered margin fit (§3.6.7).

Opens the athanase workspace DB READ-ONLY (no -journal file created),
extracts the TrapezoidalCorrection *input* node for the known-bad pages
(070 B, 256 A, 198 B, 191 B, 161 A), reruns baseline extraction, and
compares the legacy anchor-box + RANSAC quad against the clustered fit.

Skipped entirely when no athanase *.agl exists on ~/Desktop. Overlay
images land in debug/margin_validation/ for visual inspection.
"""
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from aglaia.processors import geometry as g
from aglaia.processors.TrapezoidalCorrection import (TrapezoidalCorrection,
                                                  TrapezoidalOption)
from aglaia.processors.geometry import baseline_from_ink
from aglaia.processors.utils import binarize_fixed

PAGES = ["070_B", "256_A", "198_B", "191_B", "161_A"]
OUT_DIR = Path("debug/margin_validation")


def _find_db() -> Path | None:
    for p in sorted(Path.home().glob("Desktop/*athanase*")):
        if p.is_file() and p.suffix == ".agl":
            return p
        if p.is_dir():
            hits = sorted(p.glob("*.agl")) + sorted(p.glob("*.db"))
            if hits:
                return hits[0]
    return None


DB = _find_db()
pytestmark = pytest.mark.skipif(DB is None,
                                reason="no athanase workspace on ~/Desktop")


def _trap_input_image(conn, page: str):
    row = conn.execute(
        """SELECT n.parent_id FROM nodes n
           WHERE n.processor_name = 'TrapezoidalCorrection'
             AND n.filestem LIKE ?
           ORDER BY n.id DESC LIMIT 1""",
        (f"%{page}%",),
    ).fetchone()
    if row is None or row[0] is None:
        return None, None
    img_row = conn.execute(
        """SELECT i.blob, i.dpi FROM nodes n
           JOIN images i ON i.id = n.image_id WHERE n.id = ?""",
        (row[0],),
    ).fetchone()
    if img_row is None:
        return None, None
    arr = cv2.imdecode(np.frombuffer(img_row[0], np.uint8),
                       cv2.IMREAD_UNCHANGED)
    return arr, float(img_row[1])


def _legacy_quad(baselines):
    """Pre-§3.6.7 edge path: anchor box + max-cardinality RANSAC."""
    from aglaia.processors.geometry import (_fit_line_lstsq, _fit_line_ransac,
                                         select_per_side_anchors)
    left_idxs, right_idxs, _ = select_per_side_anchors(baselines)
    if len(left_idxs) < 3 or len(right_idxs) < 3:
        return None
    lp = np.array([baselines[i][0] for i in left_idxs])
    rp = np.array([baselines[i][1] for i in right_idxs])
    ll = _fit_line_ransac(lp)
    ll = ll if ll is not None else _fit_line_lstsq(lp)
    rl = _fit_line_ransac(rp)
    rl = rl if rl is not None else _fit_line_lstsq(rp)
    return ll, rl


def _draw(vis, line, color):
    h = vis.shape[0]
    a, b, c = line
    if abs(a) < 1e-9:
        return
    x0 = int((-b * 0 - c) / a)
    x1 = int((-b * (h - 1) - c) / a)
    cv2.line(vis, (x0, 0), (x1, h - 1), color, 2, cv2.LINE_AA)


@pytest.mark.parametrize("page", PAGES)
def test_clustered_fit_on_athanase_page(page):
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        img, dpi = _trap_input_image(conn, page)
    finally:
        conn.close()
    if img is None:
        pytest.skip(f"page {page} not found in {DB.name}")

    opt = TrapezoidalOption()
    proc = TrapezoidalCorrection(opt)
    scale = min(1.0, opt.processing_dpi / max(dpi, 1.0))
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    bw = small if small.ndim == 2 else binarize_fixed(small, 127)
    if bw.ndim == 3:
        bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    ink = cv2.bitwise_not(bw) if bw.mean() > 127 else bw
    bboxes = proc._line_bboxes_from_connectivity(bw, dpi * scale)
    masks = getattr(proc, "_last_span_masks", None) or []
    baselines = []
    for i, bb in enumerate(bboxes):
        sm = masks[i] if i < len(masks) else None
        bl = baseline_from_ink(ink, bb, span_mask=sm)
        if bl is not None:
            baselines.append(bl)
    assert len(baselines) >= 4, f"{page}: only {len(baselines)} baselines"

    res = g.detect_column_quad_from_baselines(baselines)
    assert res is not None, f"{page}: clustered estimator returned None"
    quad, info = res
    legacy = _legacy_quad(baselines)

    tl, tr, br, bl_ = quad
    tilt_l = np.degrees(np.arctan2(abs(bl_[0] - tl[0]), abs(bl_[1] - tl[1])))
    tilt_r = np.degrees(np.arctan2(abs(br[0] - tr[0]), abs(br[1] - tr[1])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for pL, pR in baselines:
        cv2.line(vis, tuple(np.int32(pL)), tuple(np.int32(pR)),
                 (200, 200, 0), 1, cv2.LINE_AA)
    if legacy is not None:
        for ln in legacy:
            _draw(vis, ln, (0, 0, 255))        # legacy = red
    cv2.polylines(vis, [quad.astype(np.int32).reshape(-1, 1, 2)],
                  True, (0, 200, 0), 2, cv2.LINE_AA)  # clustered = green
    cv2.putText(vis, f"{page} src={info['column_edge_source']} "
                     f"tiltL={tilt_l:.2f} tiltR={tilt_r:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.imwrite(str(OUT_DIR / f"athanase_{page}.png"), vis)
    print(f"{page}: edges={info['column_edge_source']} "
          f"tiltL={tilt_l:.2f} tiltR={tilt_r:.2f} "
          f"n={info['n_all']} fw={info['n_full_width']}")

    assert info["column_edge_source"] == "endpoint_clustering"
    assert abs(tilt_l - tilt_r) < 2.5
    assert tilt_l < 8.0 and tilt_r < 8.0
