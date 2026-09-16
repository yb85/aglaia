---
title: Overview
description: What Aglaïa is and how the pipeline fits together.
---

Aglaïa turns a webcam pointed at a book into clean, deskewed, dewarped,
binarized, searchable PDFs — locally, on your own machine (macOS, Windows or
Linux).

A single processing chain runs in both the capture GUI and the headless
CLI: **capture → DPI fix → deskew → page detect → dewarp → binarize →
OCR → export**. Every step is a pluggable processor defined in a YAML
pipeline.

## Where to go next

- **[Install](/docs/install)** — download the app or build from source.
- **[CLI](/docs/reference/cli)** — the `aglaia` subcommands (`gui`, `run`,
  `ocr`, `setup`, `list`, `plugins`, `server`, `version`, `skill`).
- **[Server](/docs/concepts/server)** — run Aglaïa as an HTTP job API.
- **[Architecture](/docs/reference/architecture)** — how the chain runs.
- **[Pipeline](/docs/reference/pipeline)** — the YAML step schema.
- **[Processors](/docs/reference/processors)** — the built-in steps and how
  to add your own (including drop-in plugins).
- **[GUI](/docs/reference/gui)** — the capture window, sidebar, and export.
- **[Export](/docs/concepts/export)** — searchable PDF, Markdown, and the OCR
  textpack that carries both plus the OCR's provenance.
- **[Plugin store](/docs/reference/plugin-store)** and
  **[Destinations](/docs/reference/destinations)** — install a processing step,
  an OCR engine or somewhere to send a finished export.

> The reference pages in the sidebar are generated from the project's
> `docs/` directory, so their text always matches the shipped code.
