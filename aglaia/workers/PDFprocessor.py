# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PDF helpers for project export.

Wraps the DB-query layer over the assembly helpers in
`aglaia/workers/pdf_export.py`. Bitonal rows go to `build_bitonal_pdf`
(JBIG2 / CCITT G4), everything else to `build_native_pdf` (JPEG via
DCTDecode), and the optional invisible text layer is injected via
`inject_ocr_layer`.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

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


def _ocr_results_for_rows(conn, rows, engine: str | None = None,
                          run_ids: list | None = None):
    """For each row, return the latest OCR result_json (parsed) for the
    matching (scan_id, branch_path). With ``engine`` given, the latest run of
    that engine; otherwise the latest run regardless of engine. Entries are
    `None` when no completed run exists. ``run_ids``, when given, receives
    the matching `ocr_runs.id` per row (None alongside a None result)."""
    eng_clause = " AND engine = ?" if engine else ""
    out: list[dict | None] = []
    ids = run_ids if run_ids is not None else []
    for r in rows:
        try:
            scan_id = int(r["scan_id"])
            branch_path = r["branch_path"] or ""
        except (KeyError, IndexError, TypeError):
            out.append(None)
            ids.append(None)
            continue
        row = conn.execute(
            "SELECT id, result_json FROM ocr_runs "
            f"WHERE scan_id = ? AND branch_path = ? AND status = 'done'{eng_clause} "
            "ORDER BY version DESC LIMIT 1",
            (scan_id, branch_path, engine) if engine else (scan_id, branch_path),
        ).fetchone()
        if row is None or row["result_json"] is None:
            out.append(None)
            ids.append(None)
            continue
        try:
            out.append(json.loads(row["result_json"]))
            ids.append(int(row["id"]))
        except Exception:
            out.append(None)
            ids.append(None)
    return out


class OcrLayerError(RuntimeError):
    """An OCR layer was requested and could not be written (#149).

    Raised instead of returning an OCR PDF whose layer is empty or partial:
    such a file was accepted downstream as searchable, and corpus found three
    of them with 0 readable pages. The incomplete file is removed before this
    is raised. `missing` holds 1-based PDF page numbers."""

    def __init__(self, expected: int, written: int, missing: list[int],
                 reason: str = ""):
        self.expected = int(expected)
        self.written = int(written)
        self.missing = list(missing)
        shown = ", ".join(str(p) for p in missing[:20])
        more = "…" if len(missing) > 20 else ""
        super().__init__(
            reason or f"OCR layer incomplete: {written} of {expected} pages "
                      f"with OCR text were embedded (missing pages: "
                      f"{shown}{more})")


# ── public export entry point ────────────────────────────────────────

def create_pdf_from_db(
    conn, output_path, *, step_name: str | None = None,
    compression: str = "auto", add_ocr_layer: bool = False,
    engine: str | None = None, layer: dict | None = None,
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
    3) is added on top of each page so the PDF stays selectable. `engine`
    selects which OCR layer (default: the latest run regardless of engine).

    `layer`, when given with `add_ocr_layer`, receives what was laid on the
    pages, in PDF page order: ``results`` (parsed OCR result or None) and
    ``run_ids`` — so a caller writing the same pages' text elsewhere (the
    OCR textpack) cannot drift from the PDF.
    """
    output_path = Path(output_path)
    rows = _select_export_rows(conn, step_name)
    if not rows:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_bw = all(r["type"] == "BW" for r in rows)

    # Which rows actually became pages. Both builders skip rows (the bitonal
    # one skips non-BW rows, the native one a row it cannot convert), and the
    # OCR layer is laid page by page — so a list indexed by ROW put every page
    # after a skipped one under the NEXT page's text (#149).
    kept: list[int] = []
    ok: bool
    if compression in ("jbig2", "g4") or (compression == "auto" and all_bw):
        from aglaia.workers.pdf_export import build_bitonal_pdf
        bw_engine = "jbig2" if compression in ("auto", "jbig2") else "g4"
        ok = build_bitonal_pdf(rows, output_path, engine=bw_engine, pages=kept)
    else:
        from aglaia.workers.pdf_export import build_native_pdf
        ok = build_native_pdf(rows, output_path, pages=kept)

    if ok and add_ocr_layer:
        from aglaia.workers.pdf_export import inject_ocr_layer
        run_ids: list[int | None] = []
        ocr = _ocr_results_for_rows(conn, [rows[i] for i in kept], engine,
                                    run_ids=run_ids)
        if layer is not None:
            layer.update(results=ocr, run_ids=run_ids)
        if not any(ocr):
            # An OCR PDF was asked for and no page matches a completed run of
            # that engine — the lookup-mismatch case. Refuse rather than ship a
            # file named and trusted as searchable.
            output_path.unlink(missing_ok=True)
            raise OcrLayerError(len(kept), 0, list(range(1, len(kept) + 1)),
                                reason=f"no completed OCR run"
                                       f"{' for ' + engine if engine else ''} "
                                       f"matches the exported pages")
        stats = inject_ocr_layer(output_path, ocr)
        if stats["written"] < stats["expected"]:
            output_path.unlink(missing_ok=True)
            raise OcrLayerError(stats["expected"], stats["written"],
                                stats["missing"])
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
                "dpi": float(pil.info.get("dpi", (120.0, 120.0))[0]),
                "blob": blob,
            })
        except Exception:
            continue
    if not rows:
        return False
    from aglaia.workers.pdf_export import build_native_pdf
    return build_native_pdf(rows, output_path)
