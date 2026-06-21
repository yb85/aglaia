-- Aglaïa: rename scans→scans, scan_id→scan_id (added 2026-06-19)
--
-- Product rename: a "scan" is a "scan". This migration renames the table and
-- the three FK columns that point at it (nodes, branches, ocr_runs). SQLite
-- >= 3.25 (we run 3.53) auto-rewrites the *bodies* of dependent foreign keys,
-- indexes and triggers when a table/column is renamed — so idx_nodes_snap,
-- idx_ocr_runs_snap, idx_ocr_runs_snap_branch and the three ON DELETE CASCADE
-- foreign-key clauses all follow the rename without manual DROP/CREATE.
-- (Index/trigger *names* are intentionally left unchanged — their bodies now
-- say scan_id regardless; renaming the objects would be cosmetic only.)
--
-- DO NOT enable PRAGMA legacy_alter_table — the default (OFF) is what makes
-- the auto-rewrite happen.
--
-- Non-idempotent (RENAME on an already-renamed DB errors with "no such table:
-- scans"). The applied-migration ledger in ensure_schema() guarantees this
-- runs exactly once per DB; never re-run by hand.

PRAGMA foreign_keys=OFF;

BEGIN;

ALTER TABLE snaps RENAME TO scans;

ALTER TABLE nodes     RENAME COLUMN snap_id TO scan_id;
ALTER TABLE branches  RENAME COLUMN snap_id TO scan_id;
ALTER TABLE ocr_runs  RENAME COLUMN snap_id TO scan_id;

-- The index BODIES (idx_nodes_snap, idx_ocr_runs_snap,
-- idx_ocr_runs_snap_branch, idx_snaps_page_order) and the three foreign keys
-- are auto-rewritten to scan_id/scans by the RENAMEs above. Their NAMES still
-- say "scan" though — rename the index objects too so the schema reads clean.
DROP INDEX idx_nodes_snap;
CREATE INDEX idx_nodes_scan ON nodes(scan_id);
DROP INDEX idx_ocr_runs_snap;
CREATE INDEX idx_ocr_runs_scan ON ocr_runs(scan_id);
DROP INDEX idx_ocr_runs_snap_branch;
CREATE INDEX idx_ocr_runs_scan_branch ON ocr_runs(scan_id, branch_path, version DESC);
DROP INDEX idx_snaps_page_order;
CREATE INDEX idx_scans_page_order ON scans(page_order);

COMMIT;

PRAGMA foreign_keys=ON;
