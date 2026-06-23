# Camera calibration

Capture mode benefits from a calibrated camera matrix to:

1. Undistort frames before saving (removes barrel/pincushion).
2. Provide an accurate `focal_length` to `PageDewarper`.
3. Establish a known **DPI** (pixels per real-world inch) so the rest of the pipeline (DPIfixer, page margin, dewarp page margin) operates in real units.

## Target

`A4_chessboard.pdf` / `letter_chessboard.pdf` (repo root). We **generate**
these ourselves — `scripts/gen_calibration_board.py` (needs the `dev` extra for
`reportlab`) — so the grid geometry is fixed and hard-coded:

| Parameter | Value | Config key |
|---|---|---|
| Board | 7 × 10 squares | (geometry of the generated PDF) |
| Inner corners | 6 columns × 9 rows | `calibration.board_cols_inner` / `board_rows_inner` |
| Square size | 25 mm | `calibration.square_size_mm` |
| Sample count | 10 | `calibration.calnum` |

Regenerate with `uv run --extra dev python scripts/gen_calibration_board.py`.
The defaults are hard-coded in `Calibrator` + `MainWindow.__init__` (read from
`args.config.get("calibration", {})`); the bundled `config/default.yml` has no
`calibration:` block, so they apply as-is. **If you change the board, update the
script, the `Calibrator` defaults, and these keys together.**

> **Print at 100%.** Set the printer dialog to "Actual Size" / "Custom Scale:
> 100%" — never "Fit to Page", which alters the 25 mm squares (and thus every
> measurement derived from them).

### Preparing the physical board

1. **Rigidity is non-negotiable.** A millimetre of warp distorts the matrix.
   Mount the sheet on a flat, rigid backing (clipboard, foam core, MDF, glass).
2. **Adhesive:** spray adhesive or a glue stick — *not* tape (it bubbles and
   tensions unevenly).
3. **Beware glare.** Glossy laser toner reflects lights → corners aren't found.
   Diffuse the lighting or angle lamps so the surface reads matte.

### Capturing good samples

1. **15–25 frames** of decent quality.
2. **Tilt it:** pitch the board forward / back / left / right up to ~45°.
3. **Map the edges:** distortion is worst at the sensor extremities — push the
   board into the corners/edges of the field of view in several frames.
4. **Hold still:** motion blur / out-of-focus corners wreck sub-pixel accuracy.
5. A good run has an RMS reprojection error well under 1.0 (ideally 0.1–0.2).

## Workflow (Full Calibration)

In the GUI, click **Full Calibration 🏁** with the chessboard visible to the camera.

1. The button text updates per sample. Move the board between captures (rotate/tilt) to cover the FOV.
2. For sample `N - 1` (penultimate), the button text says **"Last one: put the board flat, at book distance"** — the **last sample's measured DPI** is the one persisted. Make sure the board is at the same height/distance you'll be scanning books at.
3. After `calnum` samples, `cv2.calibrateCamera` runs and `cv2.getOptimalNewCameraMatrix` produces the undistortion matrix.
4. Result is written to `config/camera_params.json`:

```json
{
  "camera_matrix":   [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs":     [[k1, k2, p1, p2, k3]],
  "dpi":             237.4,
  "resolution":      [1080, 1920],
  "new_camera_matrix": [[...]]
}
```

5. Restart capture to pick up the new calibration.

## Workflow (DPI-only)

Click **Calibrate DPI 📏** with the chessboard visible at book distance for a single sample. Updates **only** the DPI field in `config/camera_params.json`, keeping existing matrix/dist coeffs.

Useful when you've moved the camera vertically (changing scale) but not the lens.

## How DPI is computed

Per sample, after corner refinement (`cv2.cornerSubPix`):

```python
avg_px_per_square = (mean(horizontal_dists) + mean(vertical_dists)) / 2
dpi = (avg_px_per_square / square_size_mm) * 25.4
```

`finalize_calibration` uses **only the last sample's DPI** (the last sample is captured at scanning distance).

## At capture time

If `config/camera_params.json` exists, `MainWindow.capture()`:

1. `cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))`.
2. `cv2.undistort(frame, mtx, dist, None, newcameramtx)`.
3. Writes the raw frame as `.jpg`.
4. Stamps DPI EXIF via PIL `.save(dpi=(dpi, dpi))`.
5. Builds `ImageBuffer(rgb_frame, dpi=current_dpi or args.input_dpi or 100.0, ...)`.

The calibration DPI overrides `--input-dpi` when both are present.

## Camera matrix into PageDewarper

`PageDewarper.inject_step_options` pulls calibration from `args.options["calibration"]` into each `PageDewarper` step's options — but **only when the matrix is real** (`Initializer._is_real_calibration`); an identity / placeholder matrix is rejected (it underflows `focal_length` to ~0.001 and sends Powell wandering):

```python
K   = args.options.get("calibration", {}).get("camera_matrix")
res = args.options.get("calibration", {}).get("camera_matrix_resolution")
if _is_real_calibration(K):
    step_opts["camera_matrix"] = K
    step_opts["camera_matrix_resolution"] = res
else:
    step_opts["camera_matrix"] = None
    step_opts["camera_matrix_resolution"] = None
```

`PageDewarper.__init__` then normalizes the matrix to the `[-1, 1]` cube the page-dewarp library expects:

```python
scl = 2.0 / max(h, w)
focal_length = (fx + fy) / 2 * scl
K[0,0] *= scl; K[1,1] *= scl
K[0,2] = (cx - w*0.5) * scl
K[1,2] = (cy - h*0.5) * scl
```

If no calibration is available, the default `focal_length: 1.3` (`DewarpOption.focal_length`, overridable per step in the pipeline YAML) is used.

## Manual edit

`config/camera_params.json` is a plain JSON file. Safe to edit by hand if you have measurements from another source. Required keys: `camera_matrix`, `dist_coeffs`, `dpi`, `resolution`. Optional: `new_camera_matrix`.

Delete the file to force the GUI to ignore calibration on next start.

## Troubleshooting

- **"Chessboard corners not found"** — bad lighting, too much glare, or board partially out of frame. Move closer / tilt to avoid reflections.
- **DPI looks wrong by a constant factor** — wrong `square_size_mm` in config, or the PDF was printed with scaling. Measure printed squares with a ruler and update the config.
- **Undistorted frames look worse than raw** — likely too few samples or too little FOV coverage. Recapture with more variety in board position/orientation.
