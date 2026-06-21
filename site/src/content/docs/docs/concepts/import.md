---
title: Import
description: How external inputs — image files, PDF pages, and webcam frames — enter an Aglaïa project as raw scans.
---

**Import** is the first stage: it turns whatever you feed Aglaïa into *raw
scans* inside the project. A raw scan is the untouched source image plus
its DPI; it becomes the root of a page's processing tree, and everything
downstream (pipeline, OCR, export) refers back to it.

## What you can import

| Source | How it enters | Notes |
|---|---|---|
| **Image files** | `.jpg` / `.png` / … given to the CLI, or dropped in the GUI import panel | one raw scan per file, sorted by filename |
| **PDF pages** | a `.pdf` given to the CLI or the import panel | each page is rendered to an image (via pypdfium2) at `render_dpi`, default 200 |
| **Live webcam** | the capture GUI's shutter (or voice command) | each captured frame becomes a raw scan |

> PDF import **renders** each page to a raster image — Aglaïa is a
> page-image pipeline, not a text extractor. A born-digital PDF that
> already has selectable text is better read directly; import is for
> *scanned* or photographed pages.

## What import does

Each input is persisted as a `scans` row (one per page) with a raw root
`nodes` row pointing at the decoded `COLOR` image. From there the page is
enqueued into the processing chain. Images are content-hashed, so
re-importing an identical file does not duplicate the blob.

In the GUI, imported pages appear in the scans column immediately and
begin processing; in headless mode the positional arguments to
`aglaia.py` are the import set.

## Use cases

- **Digitise a phone-photo of a book** — import the JPEGs, let the chain
  dewarp and clean them.
- **Re-process an existing PDF scan** — import the PDF; each page is
  rendered and run through the pipeline.
- **Reopen a project** — passing an existing `.agl` file is not an
  import; it loads the scans already stored.

## Related resources

- [How Aglaïa works](/docs/concepts/workflow) — the full import → export path
- [Pipeline processing](/docs/concepts/pipeline-processing) — what happens to a raw scan next
- [The .AGL project file](/docs/concepts/agl-project-file) — where raw scans live
