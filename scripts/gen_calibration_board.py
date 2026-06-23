#!/usr/bin/env python
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OpenCV camera-calibration board generator & guidelines.

Generates the printable A4 + US-Letter chessboard PDFs Aglaïa's **Full
Calibration** uses. Because we generate the board ourselves at exact mm
dimensions, the grid geometry is fixed and hard-coded in the calibrator
(`lib/workers/Calibrator.py` / the `calibration:` defaults): a 7×10 SQUARE board
→ a **(6, 9)** internal-corner grid, **25 mm** squares.

Run (dev extra installs reportlab):

    uv run --extra dev python scripts/gen_calibration_board.py

Outputs `A4_chessboard.pdf` and `letter_chessboard.pdf` at the repo root
(the print targets the GUI points the user at).

═══════════════════════════════════════════════════════════════════════════
PHYSICAL BOARD PREPARATION
═══════════════════════════════════════════════════════════════════════════
1. Rigidity is non-negotiable. A millimetre of wave or warp distorts the
   calibration matrix. Mount the printed sheet on a perfectly flat, rigid
   backing (clipboard, thick foam core, MDF, or glass).
2. Adhesive: spray adhesive or a glue stick. NOT tape — it causes localised
   bubbling and uneven tension.
3. Beware glare. Laser toner is often glossy; if lights reflect off the black
   squares OpenCV won't find the corners. Diffuse the lighting / angle lamps
   so the camera sees a matte surface.

═══════════════════════════════════════════════════════════════════════════
IMAGE CAPTURE STRATEGY
═══════════════════════════════════════════════════════════════════════════
1. Quantity: 15–25 high-quality frames.
2. Angles: pitch the board forward / back / left / right up to ~45°.
3. Map the edges. Lens distortion is worst at the sensor extremities — push
   the board right to the edges and corners of the field of view in several
   frames.
4. Focus & blur: keep camera and board still. Motion blur / out-of-focus
   corners wreck sub-pixel accuracy.

═══════════════════════════════════════════════════════════════════════════
SOFTWARE BEST PRACTICES (OpenCV)
═══════════════════════════════════════════════════════════════════════════
1. Grid size: this board is 7×10 squares → a (6, 9) internal corner grid.
   Pass `(6, 9)` to `cv2.findChessboardCorners()`.
2. Refinement: always `cv2.cornerSubPix()` after the initial find for
   sub-pixel corner locations.
3. Validation: check the RMS reprojection error from `cv2.calibrateCamera()`.
   For document scanning target well under 1.0 (ideally 0.1–0.2).

═══════════════════════════════════════════════════════════════════════════
PRINTING
═══════════════════════════════════════════════════════════════════════════
Set the printer dialog to "Actual Size" / "Custom Scale: 100%". NEVER use
"Fit to Page" / "Scale to Fit" — it alters the 25 mm squares and the whole
calibration with it.
"""

import os

from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# Board geometry — KEEP IN SYNC with lib/workers/Calibrator.py defaults and the
# `calibration:` block in docs/configuration. 7×10 squares → (6, 9) inner
# corners. Changing these means re-printing AND updating the calibrator.
SQUARE_SIZE_MM = 25
COLS = 7   # squares wide → 6 inner corners
ROWS = 10  # squares tall → 9 inner corners


def generate_calibration_board(filename, paper_size, square_size_mm=SQUARE_SIZE_MM,
                               cols=COLS, rows=ROWS):
    """Generate a PDF with a perfectly centred chessboard pattern.

    :param filename: output PDF path
    :param paper_size: (width, height) in points (e.g. reportlab A4 / LETTER)
    :param square_size_mm: square size in millimetres
    :param cols: squares horizontally
    :param rows: squares vertically
    """
    page_width, page_height = paper_size
    square_size_pt = square_size_mm * mm
    board_width_pt = cols * square_size_pt
    board_height_pt = rows * square_size_pt

    # Centre the board on the page (ReportLab origin is bottom-left).
    start_x = (page_width - board_width_pt) / 2.0
    start_y = (page_height - board_height_pt) / 2.0

    c = canvas.Canvas(filename, pagesize=paper_size)
    c.setTitle(f"Aglaia Calibration Board - {cols}x{rows} squares "
               f"({square_size_mm}mm)")

    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:                     # black square
                c.setFillColorRGB(0, 0, 0)
                x = start_x + (col * square_size_pt)
                y = start_y + (row * square_size_pt)
                c.rect(x, y, square_size_pt, square_size_pt, stroke=0, fill=1)
    c.save()

    margin_x_mm = (page_width - board_width_pt) / 2.0 / mm
    margin_y_mm = (page_height - board_height_pt) / 2.0 / mm
    print(f"Successfully generated: {filename}")
    print(f"  -> Left/Right margins: {margin_x_mm:.2f} mm")
    print(f"  -> Top/Bottom margins: {margin_y_mm:.2f} mm")
    if margin_x_mm < 0 or margin_y_mm < 0:
        print("  !! WARNING: board larger than the page — it will be clipped.")
    print()


if __name__ == "__main__":
    # Output at the repo root — the print targets the GUI references.
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Generating A4 board…")
    generate_calibration_board(
        os.path.join(output_dir, "A4_chessboard.pdf"), A4)

    print("Generating US-Letter board…")
    generate_calibration_board(
        os.path.join(output_dir, "letter_chessboard.pdf"), LETTER)
