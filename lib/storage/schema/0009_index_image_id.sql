-- Aglaïa: index the image_id foreign keys (added 2026-06).
--
-- `reprocess_active_scans` / `catchup_active_scans` garbage-collect the images
-- a wiped subtree leaves orphaned, guarding each delete with
--   SELECT 1 FROM nodes          WHERE image_id = ?
--   SELECT 1 FROM debug_artifacts WHERE image_id = ?
-- Neither column was indexed, so every guard was a full table scan. On a
-- 311-scan force rerun that is O(images × rows) — minutes of CPU on the
-- reprocess thread, holding the write lock long enough to wedge the GUI.
-- These two indexes turn each guard into an O(log n) lookup.
CREATE INDEX IF NOT EXISTS idx_nodes_image           ON nodes(image_id);
CREATE INDEX IF NOT EXISTS idx_debug_artifacts_image ON debug_artifacts(image_id);
