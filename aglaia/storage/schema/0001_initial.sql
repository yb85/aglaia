-- aglaia SQLite project storage — initial schema (M0)

CREATE TABLE IF NOT EXISTS project (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_versions (
    id              INTEGER PRIMARY KEY,
    yaml_text       TEXT NOT NULL,
    yaml_sha256     TEXT NOT NULL UNIQUE,
    name            TEXT,
    step_count      INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS calibrations (
    id                     INTEGER PRIMARY KEY,
    created_at             TEXT NOT NULL,
    camera_matrix_json     TEXT NOT NULL,
    dist_coeffs_json       TEXT NOT NULL,
    new_camera_matrix_json TEXT,
    dpi                    REAL NOT NULL,
    resolution_w           INTEGER,
    resolution_h           INTEGER,
    sample_count           INTEGER,
    is_active              INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY,
    sha256          TEXT NOT NULL UNIQUE,
    format          TEXT NOT NULL CHECK (format IN ('JPG', 'PNG')),
    type            TEXT NOT NULL CHECK (type IN ('BW', 'GRAY', 'COLOR')),
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    dpi             REAL NOT NULL,
    bytes           INTEGER NOT NULL,
    blob            BLOB NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thumbs (
    id              INTEGER PRIMARY KEY,
    image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    max_dim         INTEGER NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    blob            BLOB NOT NULL,
    UNIQUE (image_id, max_dim)
);

CREATE TABLE IF NOT EXISTS snaps (
    id                  INTEGER PRIMARY KEY,
    idx                 INTEGER NOT NULL UNIQUE,
    source              TEXT NOT NULL CHECK (source IN ('capture', 'pdf', 'import')),
    source_ref          TEXT,
    transform           TEXT,
    capture_dpi         REAL,
    calibration_id      INTEGER REFERENCES calibrations(id),
    pipeline_version_id INTEGER NOT NULL REFERENCES pipeline_versions(id),
    created_at          TEXT NOT NULL,
    deleted_at          TEXT,
    root_node_id        INTEGER,
    page_order          REAL
);

CREATE INDEX IF NOT EXISTS idx_snaps_page_order ON snaps(page_order);

CREATE TABLE IF NOT EXISTS nodes (
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
    image_id            INTEGER NOT NULL REFERENCES images(id),
    status_int          INTEGER NOT NULL DEFAULT 1,
    elapsed_ms          REAL,
    meta_json           TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE (snap_id, filestem, step_idx)
);

CREATE INDEX IF NOT EXISTS idx_nodes_parent     ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_snap       ON nodes(snap_id);
CREATE INDEX IF NOT EXISTS idx_nodes_step_name  ON nodes(step_name);
CREATE INDEX IF NOT EXISTS idx_nodes_depth_leaf ON nodes(depth, is_leaf) WHERE is_leaf = 1;
CREATE INDEX IF NOT EXISTS idx_nodes_processor  ON nodes(processor_name);

CREATE TABLE IF NOT EXISTS branches (
    id                  INTEGER PRIMARY KEY,
    snap_id             INTEGER NOT NULL REFERENCES snaps(id) ON DELETE CASCADE,
    branch_path         TEXT NOT NULL,
    terminal_node_id    INTEGER NOT NULL REFERENCES nodes(id),
    chosen_node_id      INTEGER NOT NULL REFERENCES nodes(id),
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    UNIQUE (snap_id, branch_path)
);

CREATE INDEX IF NOT EXISTS idx_branches_chosen ON branches(chosen_node_id);

CREATE TABLE IF NOT EXISTS debug_artifacts (
    id              INTEGER PRIMARY KEY,
    node_id         INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,
    image_id        INTEGER NOT NULL REFERENCES images(id),
    created_at      TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS nodes_set_parent_not_leaf
AFTER INSERT ON nodes
WHEN NEW.parent_id IS NOT NULL
BEGIN
    UPDATE nodes SET is_leaf = 0 WHERE id = NEW.parent_id;
END;
