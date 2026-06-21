-- Aglaïa one-time `is_stale` backfill (added 2026-06-08)
--
-- Migration 0004 added the column with DEFAULT 0 — every existing row
-- therefore looks "fresh" regardless of whether `node_id` still matches
-- the branch's current `chosen_node_id`. This file's UPDATE recomputes
-- the flag against the real chosen target so the badges + bottom-bar
-- count reflect reality on first launch after the migration.
--
-- Idempotent: re-running it just re-derives the same value.

UPDATE ocr_runs
SET is_stale = CASE
    WHEN ocr_runs.node_id = (
        SELECT b.chosen_node_id FROM branches b
        WHERE b.snap_id = ocr_runs.snap_id
          AND b.branch_path = ocr_runs.branch_path
        LIMIT 1
    ) THEN 0 ELSE 1 END;
