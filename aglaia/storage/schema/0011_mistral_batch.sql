-- Aglaïa: Mistral batch OCR jobs (added 2026-06-23)
--
-- When the Cloud OCR (Mistral) card's "batch" toggle is on, an OCR run
-- submits one or more Mistral Batch API jobs (cheaper, async) instead of a
-- synchronous OCR call. We record each submitted job here so the OCR card
-- can show its pending state and a "Check result" / "Cancel" affordance,
-- and so results can be pulled back and applied even after a relaunch.
--
-- One row per submitted batch job. A project is "OCR-pending on Mistral"
-- while it has any row with imported_at IS NULL and status not in a
-- terminal failure state.
--
-- Columns:
--   job_id        Mistral batch job id (client.batch.jobs.* key)
--   input_file_id uploaded JSONL batch-input file id (for cleanup/debug)
--   chunk         0-based chunk index when the page/size cap forced a split
--   chunks_total  number of chunks the run was split into (1 = no split)
--   page_count    pages submitted in this chunk
--   status        last polled status: QUEUED/RUNNING/SUCCESS/FAILED/
--                 TIMEOUT_EXCEEDED/CANCELLATION_REQUESTED/CANCELLED
--   submitted_at  ISO8601 submit time (drives the "Submitted … ago" label)
--   imported_at   ISO8601 when results were applied to the project (NULL =
--                 still pending / not yet pulled)
--   run_ids       JSON array of OCR run ids (ocr_runs) this job covers, in
--                 submission order — page i of the batch output maps to
--                 run_ids[i] so deferred results land on the right branch
--   error_text    failure detail, if any
-- Idempotent.

CREATE TABLE IF NOT EXISTS mistral_batch_jobs (
    job_id        TEXT PRIMARY KEY,
    input_file_id TEXT,
    chunk         INTEGER NOT NULL DEFAULT 0,
    chunks_total  INTEGER NOT NULL DEFAULT 1,
    page_count    INTEGER,
    status        TEXT,
    submitted_at  TEXT NOT NULL,
    imported_at   TEXT,
    run_ids       TEXT,
    error_text    TEXT
);

CREATE INDEX IF NOT EXISTS idx_mistral_batch_submitted
    ON mistral_batch_jobs(submitted_at DESC);
