-- Aglaïa: add mistral_batch_jobs.run_ids (added 2026-06-23)
--
-- 0011 originally shipped without `run_ids`; projects opened before it was
-- added have the table without the column (and the migration ledger marks
-- 0011 applied, so it won't re-run). Add the column here. Fresh DBs created
-- from the updated 0011 already have it — the ALTER then raises "duplicate
-- column name", which ensure_schema tolerates for additive migrations.
--
-- run_ids: JSON array of OCR run ids (ocr_runs) a batch job covers, in
-- submission order — page i of the batch output maps to run_ids[i].
-- Idempotent (additive).

ALTER TABLE mistral_batch_jobs ADD COLUMN run_ids TEXT;
