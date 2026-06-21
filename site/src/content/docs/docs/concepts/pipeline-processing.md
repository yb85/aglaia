---
title: Pipeline processing
description: How a raw scan flows through Aglaïa's ordered chain of image processors.
---

**Pipeline processing** is the second stage of the chain (after
[import](/docs/concepts/import), before [OCR](/docs/concepts/ocr-engines)
and [export](/docs/concepts/export)). It is the heart of Aglaïa: an ordered
list of *processors*, each transforming an image and passing it on, until a
raw scan becomes a clean, straight, readable page. The same pipeline runs in
the capture GUI and in headless CLI batches, so what you see on screen is
exactly what a script produces.

## Background

A scanned page needs several independent corrections — exposure, skew,
curvature, thresholding — and the right *order* matters: you deskew before
you dewarp, dewarp before you binarize, binarize before you OCR. Aglaïa
models this as a chain of small, single-purpose processors rather than one
monolithic function, so each step can be tuned, reordered, profiled, or
swapped on its own.

A pipeline is just a YAML file: an ordered list of steps, each naming a
processor and its options. The chain runs across multiple worker processes
and persists every intermediate result to the SQLite-backed
[project file](/docs/concepts/agl-project-file) so you can inspect, branch,
and replay.

## Use cases

- **Tune one stage** — raise the binarizer's window size or disable twist
  in the dewarper without touching any other step.
- **Branch the output** — a page-detection step can emit several child
  crops that each re-enter the pipeline independently (e.g. the two halves
  of an open spread).
- **Reorder or swap** — drop in a different binarizer, or move deskew
  before page detection, by editing the YAML.
- **Batch a shelf** — point the headless CLI at a folder and let the same
  pipeline run unattended.

## Persistence and branching

Each step's output is stored as a node in the project's node tree; a
branch-emitting step (page detection) forks the tree, and the user picks
the *chosen output* per branch — that choice is what OCR and export consume.
A step's heavy image can be dropped (`persist: false`) and rebuilt on demand
by the **replay pass**, which recomposes the page from the nearest stored
image with the fewest interpolations — so a page can be replayed from any
point without re-capturing it.

## Related resources

- [How Aglaïa works](/docs/concepts/workflow) — the full import → export path
- [Pipeline](/docs/reference/pipeline) — the YAML step schema
- [Processors](/docs/reference/processors) — the built-in steps
- [Architecture](/docs/reference/architecture) — how the chain runs across workers
- [ImageBuffer](/docs/reference/imagebuffer) — the envelope passed between steps
