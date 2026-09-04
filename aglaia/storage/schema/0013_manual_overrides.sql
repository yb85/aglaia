-- Aglaïa: per-page MANUAL parameter overrides (added 2026-09-04, M9 #91)
--
-- `step_overrides` (0010) says "skip this step for this layout". This table
-- says "run it, but with THIS value" — the user correcting what the pipeline
-- estimated. Ported from the iOS `ManualOverrides` model
-- (AglaiaCore/Sources/AglaiaCore/Pipeline/ManualOverrides.swift).
--
-- Keying, deliberately identical to `step_overrides` so both load the same way:
--   scan_id      owning capture
--   branch_path  ""      = pre-split trunk (applies to every layout)
--               "A"/"B" = one PageDetector layout
--
-- `payload_json` is one JSON object, every field optional:
--   skew_deg  REAL     deskew angle in degrees; replaces the estimate
--   roi       [[x,y]]  ROI polygon, in the branch's own (cropped) coords
--   curl      {alpha, beta, gamma}  cylindrical curl + spine term; the sheet
--                                   is frozen at these and only the pose is
--                                   re-optimised
--   frame_wh  [w, h]   pixel size of the stage frame the SPATIAL edits above
--                      were made on
--
-- `frame_wh` is the iOS `roiFrameWH` / `quadFrameWH` lesson: a polygon is only
-- meaningful against the frame it was drawn on. A consumer validates it and
-- DROPS a mismatching override with a note, rather than applying coordinates
-- that were silently rescaled or flipped.
--
-- One row per (scan, branch): the payload is edited as a whole, and a partial
-- write would leave the other fields' provenance ambiguous.

CREATE TABLE IF NOT EXISTS manual_overrides (
    id           INTEGER PRIMARY KEY,
    scan_id      INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    branch_path  TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE (scan_id, branch_path)
);

CREATE INDEX IF NOT EXISTS idx_manual_overrides_scan ON manual_overrides(scan_id);
