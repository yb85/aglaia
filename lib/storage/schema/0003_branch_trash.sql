-- Aglaïa per-layout trash (added 2026-06-08)
--
-- Soft-delete pattern mirroring `snaps.deleted_at`. A non-NULL
-- `trashed_at` means the gallery / table / grid view should render this
-- branch as dimmed-with-trash-overlay. Restore = NULL the column.
--
-- Snap-level deletion still uses `snaps.deleted_at`; this is per-layout
-- within a surviving snap.

ALTER TABLE branches ADD COLUMN trashed_at TEXT;

CREATE INDEX IF NOT EXISTS idx_branches_trashed ON branches(trashed_at);
