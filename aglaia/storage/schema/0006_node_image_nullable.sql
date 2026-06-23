-- Aglaïa: make nodes.image_id nullable (added 2026-06-18)
--
-- A pipeline step with `persist: false` (the eye-off toggle / yaml flag)
-- applies its transform — its `replay_params` are still stamped in
-- `meta_json` so replay reconstructs the geometry — but its heavy output
-- image is NOT stored. The node row stays (lineage + replay trail intact);
-- only `image_id` becomes NULL. SQLite cannot drop a NOT NULL constraint
-- in place, so rebuild the table. The applied-migration ledger in
-- ensure_schema() guarantees this rebuild runs exactly once per DB.

PRAGMA foreign_keys=OFF;

BEGIN;

CREATE TABLE nodes_new (
    id                  INTEGER PRIMARY KEY,
    snap_id             INTEGER NOT NULL REFERENCES snaps(id) ON DELETE CASCADE,
    parent_id           INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    pipeline_version_id INTEGER NOT NULL REFERENCES pipeline_versions(id),
    step_idx            INTEGER NOT NULL,
    step_name           TEXT,
    processor_name      TEXT,
    branch_label        TEXT,
    depth               INTEGER NOT NULL,
    filestem            TEXT NOT NULL,
    is_leaf             INTEGER NOT NULL DEFAULT 1,
    is_branch_point     INTEGER NOT NULL DEFAULT 0,
    image_id            INTEGER REFERENCES images(id),
    status_int          INTEGER NOT NULL DEFAULT 1,
    elapsed_ms          REAL,
    meta_json           TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE (snap_id, filestem, step_idx)
);

INSERT INTO nodes_new
    SELECT id, snap_id, parent_id, pipeline_version_id, step_idx, step_name,
           processor_name, branch_label, depth, filestem, is_leaf,
           is_branch_point, image_id, status_int, elapsed_ms, meta_json,
           created_at
    FROM nodes;

DROP TABLE nodes;
ALTER TABLE nodes_new RENAME TO nodes;

CREATE INDEX IF NOT EXISTS idx_nodes_parent     ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_snap       ON nodes(snap_id);
CREATE INDEX IF NOT EXISTS idx_nodes_step_name  ON nodes(step_name);
CREATE INDEX IF NOT EXISTS idx_nodes_depth_leaf ON nodes(depth, is_leaf) WHERE is_leaf = 1;
CREATE INDEX IF NOT EXISTS idx_nodes_processor  ON nodes(processor_name);

-- DROP TABLE nodes also dropped this trigger (0001); recreate it on the
-- rebuilt table so a child insert keeps marking its parent non-leaf.
CREATE TRIGGER IF NOT EXISTS nodes_set_parent_not_leaf
AFTER INSERT ON nodes
WHEN NEW.parent_id IS NOT NULL
BEGIN
    UPDATE nodes SET is_leaf = 0 WHERE id = NEW.parent_id;
END;

COMMIT;

PRAGMA foreign_keys=ON;
