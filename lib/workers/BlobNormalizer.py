# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
BlobNormalizer — batch-level horizontal-width normalisation.

Per-scan geometry cannot recover the true page width when the upstream
PageDetector returns a text-tight bbox that varies in horizontal
extent from page to page (scan 18 A vs 17 B in the theobald corpus).
The remaining signal is *physical*: glyph stroke widths within a
single book are invariant (same font, same scan resolution). A
squished page has its blob-width histogram shifted left compared to
a non-squished one, and the ratio of the medians is the missing
horizontal scale factor.

This module solves a constrained alignment problem:

    s_i >= 1          (flattening only widens; never shrink)
    s_i * w_i ~= w*   (per-scan median blob width matches reference)

where `w_i` is the median glyph blob width on scan i and `w*` is the
batch maximum. Reference = the *least* squished scan; everything
else gets scaled up to match.

Public entry points:
    `compute_scales(db_path) -> dict[(scan_id, branch_label), float]`
    `apply_scales(db_path, scales, out_dir)`

Both work directly on the project SQLite. `compute_scales` reads the
terminal-node images (replay step), measures blob widths, returns a
mapping. `apply_scales` resizes the terminal images by the per-scan
factor and writes them to `<out_dir>/<filestem>.png`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Blob-width bounds (px). Excludes:
#   - speckle / dust          (width < min)
#   - full-line bars / rules  (width > max)
# Tuned for typical body text at ~300 dpi.
_MIN_BLOB_W = 5
_MAX_BLOB_W = 80
_MIN_BLOB_H = 5
_MAX_BLOB_H = 70
_EROSION_PX = 2


def compute_blob_aspect_ratios(img: np.ndarray) -> np.ndarray:
    """Return per-blob `width / height` ratios for inked components.

    Aspect ratio is **font-size invariant** within a typeface: at any
    point size, a Times Roman 'a' has the same w/h. So footnote and
    body-text blobs contribute the *same* ratio distribution despite
    their different absolute pixel sizes — which is exactly what we
    need when a scan mixes the two (a previous width-based approach
    skewed correction factors too high on footnote-heavy pages).

    A horizontal squish, by contrast, multiplies width by `s < 1`
    while leaving height untouched, so median(ratio) shrinks
    proportionally — the signal we want.

    Steps:
      1. Gray + Otsu binarise; flip so ink = 255.
      2. Erode by `_EROSION_PX` px to split joined glyphs.
      3. Connected components, keep blobs whose absolute size fits
         a glyph window (drops noise specks AND full-line rules).
      4. Return `w/h` as float for each surviving blob.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = cv2.bitwise_not(bw)
    if _EROSION_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (_EROSION_PX, _EROSION_PX))
        bw = cv2.erode(bw, k, iterations=1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    keep = ((ws >= _MIN_BLOB_W) & (ws <= _MAX_BLOB_W) &
            (hs >= _MIN_BLOB_H) & (hs <= _MAX_BLOB_H))
    ws = ws[keep]
    hs = hs[keep]
    if ws.size == 0:
        return np.empty(0, dtype=np.float64)
    return ws / hs


def _terminal_node_rows(con: sqlite3.Connection) -> list[tuple]:
    """Return (scan_idx, branch, filestem, image_blob, image_format).

    Terminal = the last node per (scan, branch) path. Matches what
    the bench export writes to `results/`.
    """
    return con.execute("""
        SELECT s.idx AS scan_idx,
               n.branch_label AS branch,
               n.filestem AS filestem,
               i.blob AS blob,
               i.format AS fmt
        FROM nodes n
        JOIN scans s ON s.id = n.scan_id
        JOIN images i ON i.id = n.image_id
        WHERE n.is_leaf = 1
          AND s.deleted_at IS NULL
        ORDER BY s.idx, n.branch_label
    """).fetchall()


def compute_scales(
    db_path: Path,
    *, max_scale: float = 1.20,
    ref_percentile: float = 90.0,
    min_blob_count: int = 500,
    skip_first: int = 2,
    skip_last: int = 2,
) -> dict[tuple[int, Optional[str]], float]:
    """Return per-(scan, branch) horizontal scale factor.

    Reference = p{ref_percentile} of per-scan medians of the
    *reference set*, which **excludes**:

      - scans with fewer than `min_blob_count` qualifying blobs
        (degenerate fallback / passthrough — see theobald scan 1 A
        / 108 A);
      - the first `skip_first` and last `skip_last` scans (title
        pages, end matter, ISBN / barcode covers — these often have
        atypical font sizes and shouldn't anchor the reference).

    Scans in those exclusion classes still receive a scale, but
    derived from the same reference — they just don't influence it.

    `max_scale` caps stretches. With p90 reference + same-book
    pages, scales typically fall in [1.00, 1.15]; the 1.20 cap
    protects against measurement noise. Stronger values usually
    indicate a deeper pipeline issue (dewarp blow-up, layout
    mis-crop) that won't be fixed by stretching.
    """
    con = sqlite3.connect(str(db_path))
    rows = _terminal_node_rows(con)
    con.close()

    medians: dict[tuple[int, Optional[str]], float] = {}
    insufficient: set[tuple[int, Optional[str]]] = set()
    scan_ids = sorted({r[0] for r in rows})
    first_set = set(scan_ids[:skip_first]) if skip_first > 0 else set()
    last_set = set(scan_ids[-skip_last:]) if skip_last > 0 else set()
    title_set = first_set | last_set

    for scan_idx, branch, filestem, blob, fmt in rows:
        img = cv2.imdecode(np.frombuffer(blob, np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        ratios = compute_blob_aspect_ratios(img)
        if ratios.size < min_blob_count:
            insufficient.add((scan_idx, branch))
            continue
        medians[(scan_idx, branch)] = float(np.median(ratios))

    # Reference pool = body scans (not title/end matter).
    ref_pool = [v for (idx, _), v in medians.items() if idx not in title_set]
    if not ref_pool:
        ref_pool = list(medians.values())
    if not ref_pool:
        return {}
    # Reference = high quantile of body w/h ratios (least squished).
    ref_r = float(np.percentile(ref_pool, ref_percentile))

    scales: dict[tuple[int, Optional[str]], float] = {}
    for key, r_i in medians.items():
        s = ref_r / max(r_i, 1e-6)
        s = max(1.0, min(s, max_scale))
        scales[key] = s
    for key in insufficient:
        scales[key] = 1.0
    return scales


def apply_scales(
    db_path: Path,
    scales: dict[tuple[int, Optional[str]], float],
    out_dir: Path,
) -> int:
    """Resize each terminal image horizontally by its scale and write
    to `<out_dir>/<filestem>.<ext>`.

    Returns the number of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    rows = _terminal_node_rows(con)
    con.close()

    n_written = 0
    for scan_idx, branch, filestem, blob, fmt in rows:
        s = scales.get((scan_idx, branch), 1.0)
        img = cv2.imdecode(np.frombuffer(blob, np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if s > 1.0001:
            h, w = img.shape[:2]
            new_w = int(round(w * s))
            interp = cv2.INTER_NEAREST if (img.ndim == 2 and
                                           np.unique(img[::8, ::8]).size <= 2) \
                else cv2.INTER_CUBIC
            img = cv2.resize(img, (new_w, h), interpolation=interp)
        ext = fmt.lower()
        out_path = out_dir / f"{filestem}.{ext}"
        ok = cv2.imwrite(str(out_path), img)
        if ok:
            n_written += 1
    return n_written
