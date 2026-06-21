-- Aglaïa: per-page processor disable (added 2026-06-21)
--
-- Replaces exit-stage navigation (branches.chosen_node_id step-back/forward)
-- with per-page-layout processor toggles. A row here means "skip this
-- pipeline step for this layout" — the chain emits a passthrough node
-- (same image, no process(), no replay stamp) at that step instead of
-- running the processor, and the page is re-run on every toggle.
--
-- Keying:
--   scan_id      owning capture
--   branch_path  ""        = pre-split trunk (applies to every layout)
--                "A"/"B"   = one PageDetector layout
--   step_idx     node step_idx the processor's output would occupy (i+1 in
--                run_pipeline) — matches nodes.step_idx so views map 1:1
-- A present row with disabled=1 disables; disabled=0 (or no row) enables.
--
-- chosen_node_id now always tracks the rerun terminal (export queries are
-- unchanged) — collapse any prior user step-backs so existing projects'
-- exports match the new model. Idempotent.

CREATE TABLE IF NOT EXISTS step_overrides (
    id          INTEGER PRIMARY KEY,
    scan_id     INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    branch_path TEXT NOT NULL,
    step_idx    INTEGER NOT NULL,
    disabled    INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    UNIQUE (scan_id, branch_path, step_idx)
);

CREATE INDEX IF NOT EXISTS idx_step_overrides_scan ON step_overrides(scan_id);

UPDATE branches SET chosen_node_id = terminal_node_id
 WHERE chosen_node_id != terminal_node_id;
