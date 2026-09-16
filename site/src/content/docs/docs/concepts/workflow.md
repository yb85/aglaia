---
title: How Aglaïa works
description: The end-to-end path a page takes — import → pipeline processing → OCR → export — and the project file that holds it all.
---

Aglaïa turns a stack of photographed or scanned book pages into a clean,
searchable document. Every page travels the same four-stage path, and the
whole journey is recorded in a single project file you can reopen, branch,
and replay.

```mermaid
flowchart LR
    Import["IMPORT<br/>images / PDF / webcam frame"]
    Pipeline["PIPELINE PROCESSING<br/>DPI · deskew · page · dewarp · binarize …"]
    OCR["OCR<br/>recognise text lines"]
    Export["EXPORT<br/>searchable PDF / Markdown / OCR textpack"]
    AGL["one &lt;slug&gt;.agl SQLite project file<br/>(raw scans · every step · branches · OCR)"]
    Import --> Pipeline --> OCR --> Export
    Import --> AGL
    Pipeline --> AGL
    OCR --> AGL
    Export --> AGL
```

## The four stages

1. **[Import](/docs/concepts/import)** — image files, PDF pages, or live
   webcam frames are ingested. Each one becomes a *raw scan* (the
   untouched source image) stored in the project.
2. **[Pipeline processing](/docs/concepts/pipeline-processing)** — every
   raw scan runs the ordered chain of *processors* defined by a pipeline
   YAML: DPI normalisation, deskew, page detection, page dewarp,
   binarisation, and so on. Each step's output is persisted, so the chain
   can branch (e.g. a two-page spread splits into two pages) and replay.
3. **[OCR](/docs/concepts/ocr-engines)** — an OCR engine reads the chosen
   output of each page into text lines (and, for some engines, structure
   such as headings and tables). This runs off the pipeline, on the image
   the user selected for each page.
4. **[Export](/docs/concepts/export)** — the chosen page outputs are
   assembled into a searchable PDF (image + invisible OCR text layer),
   a Markdown document, or an **OCR textpack** carrying both plus the raw
   OCR response. An installed destination can take the finished file
   straight to a Kindle, a Calibre library, a folder or a corpus
   (`--send-to`). You can also export a *slim* copy of the project itself.

A page that needs no cleaning can skip stages 2 and 3 of the pipeline
entirely: `aglaia ocr` ingests each page as it is and OCRs it directly.

## One file holds everything

All four stages read and write a single **[`.agl` project
file](/docs/concepts/agl-project-file)** — a plain SQLite database holding
the raw scans, every pipeline step, the branch choices, and the OCR
results. The same file (and the same pipeline) drives both the capture GUI
and headless CLI batches, so an interactive scan and a scripted run produce
identical results.

## Comparison: GUI vs headless

| | Capture GUI | Headless CLI |
|---|---|---|
| Entry | `aglaia [PROJECT]` (the default command) | `aglaia run PATHS…` |
| Import | live webcam + import panel | image / PDF / `.agl` arguments |
| Processing | identical `IntegratedProcessingChain` | identical |
| OCR / export | tabs + buttons | `--ocr`, `--export`, `--send-to`, `--check-ocr` |
| Correcting one page | drag its crop, corners, rotation or dewarp sliders | `step_overrides` / `manual_overrides` already stored in the project are honoured |
| Use when | scanning interactively | batching, automation |

> The CLI is organised into subcommands — `aglaia run` (batch), `aglaia ocr`
> (OCR with no pipeline, for pages that need no cleaning), `aglaia setup`,
> `aglaia list`, `aglaia plugins`, `aglaia server`, `aglaia version`,
> `aglaia skill`. Running `aglaia` (or `aglaia ~/book.agl`) with no
> subcommand opens the GUI.

## Related resources

- [Import](/docs/concepts/import)
- [Pipeline processing](/docs/concepts/pipeline-processing)
- [OCR engines](/docs/concepts/ocr-engines)
- [Export](/docs/concepts/export)
- [The .AGL project file](/docs/concepts/agl-project-file)
