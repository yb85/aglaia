---
title: The .AGL project file
description: What an Aglaïa project file contains, the slimmed variant, and how to inspect it — it is just SQLite.
---

An Aglaïa project is a **single file**: `<slug>.agl`. It is an ordinary
**SQLite database** — no sidecar files, no proprietary format. Everything a
project knows lives in it: the raw scans, every pipeline step, the branch
choices, the OCR text, and the calibration + pipeline snapshots used to
produce them.

## What it contains

| Table | Holds |
|---|---|
| `project` | name, slug, schema version, notes (one row) |
| `pipeline_versions` | frozen pipeline-YAML snapshots; the active row is current |
| `calibrations` | frozen camera-calibration snapshots; the active row is current |
| `images` | encoded JPG/PNG blobs (width/height/dpi/type), de-duplicated by sha256 |
| `thumbs` | per-image thumbnails for the gallery |
| `scans` | one row per imported page / captured frame (soft-deleted via `deleted_at`) |
| `nodes` | one row per pipeline-step output; a self-referencing tree per scan |
| `branches` | per-branch chosen output (e.g. the A / B halves of a spread) |
| `ocr_runs` | OCR results attached to a node |
| `debug_artifacts` | optional debug overlays attached to a node |

Because images are content-hashed in `images`, the same pixels are stored
once however many nodes point at them, and re-importing a file does not
duplicate it.

## The slimmed version

A full project keeps **every** intermediate step (so you can branch, step
back, and replay) — which makes it large. A **slim** copy keeps only what a
delivery needs:

- every active scan's **raw root** image, and
- every branch's **chosen output** (the image an export would use) and its
  **OCR**.

All other images, thumbnails, intermediate nodes, debug artifacts, and
orphaned OCR runs are dropped, then the file is `VACUUM`ed so its on-disk
size reflects the reduced content.

Two ways to produce one (GUI Export tab / project menu):

- **Export slimmed project** — writes a separate slim copy, leaving the working file intact;
- **Slim-down current project** — rewrites the live file in place (after the project window closes), then reopens it.

A slim project opens normally; it simply has no intermediate stages to
step back to.

## How to inspect it

It is plain SQLite, so any SQLite tool works — no Aglaïa needed:

```bash
sqlite3 my-book.agl '.tables'
sqlite3 my-book.agl 'SELECT id, idx, deleted_at FROM scans;'
sqlite3 my-book.agl 'SELECT id, step_idx, step_name FROM nodes ORDER BY id;'

# pull one stored image out to a file
sqlite3 my-book.agl \
  "SELECT writefile('page.jpg', data) FROM images WHERE id = 1;"
```

For a GUI, [DB Browser for SQLite](https://sqlitebrowser.org/) opens it
directly. The `images.data` column is the raw JPG/PNG bytes.

## Related resources

- [How Aglaïa works](/docs/concepts/workflow) — the file ties all four stages together
- [Storage layer](/docs/reference/storage) — the full schema reference
- [Export](/docs/concepts/export) — including slim-project export
