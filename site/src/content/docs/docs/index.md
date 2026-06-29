---
title: Overview
description: What Aglaïa is and how the pipeline fits together.
---

Aglaïa turns a webcam pointed at a book into clean, deskewed, dewarped,
binarized, searchable PDFs — locally, on your Mac.

A single processing chain runs in both the capture GUI and the headless
CLI: **capture → DPI fix → deskew → page detect → dewarp → binarize →
OCR → export**. Every step is a pluggable processor defined in a YAML
pipeline.

## Where to go next

- **[Install](/docs/install)** — download the app or build from source.
- **[CLI](/docs/reference/cli)** — the `aglaia` subcommands (`run`, `setup`,
  `list`, `server`).
- **[Server](/docs/concepts/server)** — run Aglaïa as an HTTP job API.
- **[Architecture](/docs/reference/architecture)** — how the chain runs.
- **[Pipeline](/docs/reference/pipeline)** — the YAML step schema.
- **[Processors](/docs/reference/processors)** — the built-in steps and how
  to add your own (including drop-in plugins).
- **[GUI](/docs/reference/gui)** — the capture window, sidebar, and export.

> The full reference in the sidebar is generated from the project's
> `docs/` directory, so it always matches the shipped code.
