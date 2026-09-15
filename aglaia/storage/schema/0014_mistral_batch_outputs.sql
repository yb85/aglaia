-- Aglaïa: raw Mistral batch output, kept in the project (added 2026-09-15, #147)
--
-- `mistral_batch_jobs` (0011) keeps the job ids; the per-page results land in
-- `ocr_runs`. The output file itself — the JSONL Mistral returns, with the full
-- OCR structure and the usage/billing fields — used to be parsed and dropped.
-- A job id is not a durable record of a paid OCR: Mistral does not promise to
-- keep output files downloadable, and downstream (corpus, the OCR textpack)
-- ships the raw output as `assets/mistral-<job>.jsonl`.
--
-- One row per job, written when the output is downloaded at import time.
--   job_id          Mistral batch job id (same key as mistral_batch_jobs)
--   output_file_id  Mistral file id the bytes came from
--   raw             the JSONL EXACTLY as `client.files.download` returned it —
--                   bytes, never re-serialised
--   sha256          hex digest of `raw`
--   size            len(raw)
--   completed_at    job completion time reported by Mistral (ISO 8601), if any
--   fetched_at      ISO 8601 download time
--
-- No foreign key to mistral_batch_jobs on purpose: deleting a job from the
-- Mistral Jobs tab must not delete the only copy of a paid result.
-- Re-importing replaces the row with the same bytes (idempotent).

CREATE TABLE IF NOT EXISTS mistral_batch_outputs (
    job_id         TEXT PRIMARY KEY,
    output_file_id TEXT,
    raw            BLOB NOT NULL,
    sha256         TEXT NOT NULL,
    size           INTEGER NOT NULL,
    completed_at   TEXT,
    fetched_at     TEXT NOT NULL
);
