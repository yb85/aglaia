-- Aglaïa OCR runs (added 2026-06-06)
--
-- One row per OCR pass against a chosen branch output (node).
-- Versioning: one row per run, version monotonic per (snap_id, branch_path).
-- Staleness is *computed*, not stored — a run is stale when its node_id no
-- longer matches the branch's current chosen_node_id. That keeps the DB
-- update-free when the user steps a branch back/forward.

CREATE TABLE IF NOT EXISTS ocr_runs (
    id              INTEGER PRIMARY KEY,
    snap_id         INTEGER NOT NULL REFERENCES snaps(id) ON DELETE CASCADE,
    node_id         INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    branch_path     TEXT NOT NULL DEFAULT '',
    engine          TEXT NOT NULL,                -- 'apple_vision' | 'surya'
    languages_json  TEXT NOT NULL DEFAULT '[]',   -- JSON array of BCP-47/ISO codes
    version         INTEGER NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending','running','done','error')),
    result_json     TEXT,                         -- {lines:[{text,bbox,confidence,...}], page_w, page_h}
    error_text      TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (snap_id, branch_path, version)
);

CREATE INDEX IF NOT EXISTS idx_ocr_runs_snap        ON ocr_runs(snap_id);
CREATE INDEX IF NOT EXISTS idx_ocr_runs_node        ON ocr_runs(node_id);
CREATE INDEX IF NOT EXISTS idx_ocr_runs_status      ON ocr_runs(status);
CREATE INDEX IF NOT EXISTS idx_ocr_runs_snap_branch ON ocr_runs(snap_id, branch_path, version DESC);
