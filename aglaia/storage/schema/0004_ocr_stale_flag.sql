-- Aglaïa OCR stale flag (added 2026-06-08)
--
-- `is_stale` makes "this OCR result no longer matches the branch's
-- chosen output" an explicit, queryable column instead of a computed
-- comparison scattered through every consumer. Authoritative writers
-- (chosen_node_id change, fresh OCR run) maintain the value; consumers
-- (badges, bottom-bar count, branches_needing_ocr) just read it.

ALTER TABLE ocr_runs ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_ocr_runs_stale ON ocr_runs(is_stale);
