# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PDF helpers for project export.

Wraps the DB-query layer over the assembly helpers in
`lib/workers/pdf_export.py`. Bitonal rows go to `build_bitonal_pdf`
(JBIG2 / CCITT G4), everything else to `build_native_pdf` (JPEG via
DCTDecode), and the optional invisible text layer is injected via
`inject_ocr_layer`.
"""

from __future__ import annotations

import io
import json
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


# ── small image utilities (kept for tests + legacy callers) ──────────

def is_monochrome(pil_img: Image.Image) -> bool:
    if pil_img.mode == "1":
        return True
    gray = pil_img.convert("L")
    hist = gray.histogram()
    black_pixels = sum(hist[:30])
    white_pixels = sum(hist[225:])
    total_pixels = pil_img.width * pil_img.height
    if total_pixels == 0:
        return False
    return (black_pixels + white_pixels) / total_pixels > 0.98


def determine_type(pil_img: Image.Image) -> str:
    if is_monochrome(pil_img):
        return "monochrome"
    if pil_img.mode == "L":
        return "grayscale"
    if pil_img.mode == "RGB":
        arr = np.array(pil_img)
        if (np.allclose(arr[:, :, 0], arr[:, :, 1], atol=5)
                and np.allclose(arr[:, :, 1], arr[:, :, 2], atol=5)):
            return "grayscale"
    return "color"


def save_image(pil_img: Image.Image, path, img_type: str, dpi: float) -> str:
    path = Path(path)
    dpi_val = int(round(dpi))
    if img_type == "monochrome":
        bw = pil_img.convert("L").point(lambda x: 0 if x < 128 else 255, "1")
        final_path = path.with_suffix(".png")
        bw.save(final_path, "PNG", dpi=(dpi_val, dpi_val))
    elif img_type == "grayscale":
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")
        final_path = path.with_suffix(".jpg")
        pil_img.save(final_path, "JPEG", quality=85, dpi=(dpi_val, dpi_val))
    else:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        final_path = path.with_suffix(".jpg")
        pil_img.save(final_path, "JPEG", quality=85, dpi=(dpi_val, dpi_val))
    return str(final_path)


# ── DB → ordered export rows ────────────────────────────────────────

def _select_export_rows(conn, step_name: str | None):
    """Pull rows ordered by `page_order` (drag-reorder aware), then `idx`.

    Always carries `scan_id` + `branch_path` so the OCR-layer pass can
    look up the matching OCR run per page without re-querying the order.
    """
    if step_name:
        q = """
            SELECT i.format, i.type, i.width, i.height, i.dpi, i.blob,
                   n.scan_id AS scan_id,
                   COALESCE(n.branch_label, '') AS branch_path
              FROM nodes n
              JOIN images i ON i.id = n.image_id
              JOIN scans s  ON s.id = n.scan_id
             WHERE n.step_name = ?
               AND s.deleted_at IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM branches b
                    WHERE b.scan_id = n.scan_id
                      AND b.branch_path = COALESCE(n.branch_label, '')
                      AND b.trashed_at IS NOT NULL)
             ORDER BY s.page_order ASC, s.idx ASC, n.branch_label ASC
        """
        return conn.execute(q, (step_name,)).fetchall()
    q = """
        SELECT i.format, i.type, i.width, i.height, i.dpi, i.blob,
               b.scan_id AS scan_id, b.branch_path AS branch_path
          FROM branches b
          JOIN nodes n  ON n.id = b.chosen_node_id
          JOIN images i ON i.id = n.image_id
          JOIN scans s  ON s.id = b.scan_id
         WHERE s.deleted_at IS NULL
           AND b.trashed_at IS NULL
         ORDER BY s.page_order ASC, s.idx ASC, b.branch_path ASC
    """
    return conn.execute(q).fetchall()


def _ocr_results_for_rows(conn, rows):
    """For each row, return the latest OCR result_json (parsed) for the
    matching (scan_id, branch_path). Entries are `None` when no completed
    run exists."""
    out: list[dict | None] = []
    for r in rows:
        try:
            scan_id = int(r["scan_id"])
            branch_path = r["branch_path"] or ""
        except (KeyError, IndexError, TypeError):
            out.append(None)
            continue
        row = conn.execute(
            "SELECT result_json FROM ocr_runs "
            "WHERE scan_id = ? AND branch_path = ? AND status = 'done' "
            "ORDER BY version DESC LIMIT 1",
            (scan_id, branch_path),
        ).fetchone()
        if row is None or row["result_json"] is None:
            out.append(None)
            continue
        try:
            out.append(json.loads(row["result_json"]))
        except Exception:
            out.append(None)
    return out


# ── public export entry point ────────────────────────────────────────

def create_pdf_from_db(
    conn, output_path, *, step_name: str | None = None,
    compression: str = "auto", add_ocr_layer: bool = False,
) -> bool:
    """Build a PDF from project SQLite rows.

    `compression`:
      - "auto"   → JBIG2 (if installed) for every BW row; otherwise native.
      - "jbig2"  → JBIG2 lossless. Non-BW rows are skipped.
      - "g4"     → CCITT G4. Non-BW rows are skipped.
      - "native" → pikepdf JPEG embedding for every row (colour or gray).

    `step_name` filters by node step; `None` exports each branch's chosen
    leaf. Order: `scans.page_order` (drag-reorder aware), then `idx`.

    `add_ocr_layer`: when True and a matching OCR run exists for the
    export set, an invisible text layer (Helvetica/WinAnsi, render mode
    3) is added on top of each page so the PDF stays selectable.
    """
    output_path = Path(output_path)
    rows = _select_export_rows(conn, step_name)
    if not rows:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_bw = all(r["type"] == "BW" for r in rows)

    ok: bool
    if compression in ("jbig2", "g4") or (compression == "auto" and all_bw):
        from lib.workers.pdf_export import build_bitonal_pdf
        engine = "jbig2" if compression in ("auto", "jbig2") else "g4"
        ok = build_bitonal_pdf(rows, output_path, engine=engine)
    else:
        from lib.workers.pdf_export import build_native_pdf
        ok = build_native_pdf(rows, output_path)

    if ok and add_ocr_layer:
        from lib.workers.pdf_export import inject_ocr_layer
        inject_ocr_layer(output_path, _ocr_results_for_rows(conn, rows))
    return ok


def create_pdf_from_images(image_dir, output_path) -> bool:
    """Bundle every image in `image_dir` into one PDF, sorted by filename.

    Used by the legacy capture flow. Reads each image, converts to a
    project row shape, and reuses :func:`create_pdf_from_db`'s native
    builder without going through the DB."""
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    if not image_dir.exists():
        return False
    images = sorted(
        [f for f in image_dir.iterdir()
         if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=lambda x: x.name,
    )
    if not images:
        return False

    rows = []
    for f in images:
        try:
            blob = f.read_bytes()
            pil = Image.open(io.BytesIO(blob))
            rows.append({
                "format": "JPG" if f.suffix.lower() in (".jpg", ".jpeg") else "PNG",
                "type": "COLOR" if pil.mode == "RGB" else (
                    "GRAY" if pil.mode == "L" else "COLOR"),
                "width": pil.width,
                "height": pil.height,
                "dpi": float(pil.info.get("dpi", (300.0, 300.0))[0]),
                "blob": blob,
            })
        except Exception:
            continue
    if not rows:
        return False
    from lib.workers.pdf_export import build_native_pdf
    return build_native_pdf(rows, output_path)
