---
title: Export
description: Turning the processed, OCR'd pages into a searchable PDF, a Markdown document, or a slim project copy.
---

**Export** is the last stage: it assembles the pages you have chosen into a
deliverable. The source of every export is each page's *chosen output* —
the pipeline result selected for that branch — never the raw scan or an
intermediate step.

## Export targets

| Target | What you get |
|---|---|
| **PDF** | a searchable PDF: each page's chosen image with an *invisible* OCR text layer on top, so the document looks like the scan but is selectable and indexable |
| **Markdown** | a free-flowing `.md` document reconstructed from the OCR — headings, paragraphs, lists, footnotes (from structure where the engine provides it, else inferred from line geometry) |
| **Slim project** | a pruned copy of the `.agl` file itself — see [The .AGL project file](/docs/concepts/agl-project-file) |

## PDF compression profiles

The PDF exporter picks how each page image is encoded:

| Profile | Encoding |
|---|---|
| `auto` | JBIG2 when every page is bitonal, else keep the originals |
| `g4` | CCITT Group 4 (universal bitonal fallback) |
| `jbig2` | JBIG2 — ~25–37 % smaller than G4 (needs the `aglaia_jbig2` binding) |
| `native` | keep each image's original colour/grey encoding verbatim |

An OCR text layer is added whenever OCR results exist for the page.

## Markdown refinement

Markdown export can optionally post-process the text with an on-device LLM
(`--md-refine apple_fm`) to repair line breaks and coherence — a no-op when
the backend is unavailable.

## How to run it

| | Capture GUI | Headless CLI |
|---|---|---|
| PDF | Export tab → PDF | `--export pdf` (or `pdf:g4`, `pdf:jbig2`, …) |
| Markdown | Export tab → Markdown | `--export md` |
| Both | — | `--export pdf:g4+md` |

Exports are written next to the project as `<slug>.pdf` / `<slug>.md`.

Each `--export` entry uses Aglaïa's standard option-spec format —
`name[:token|key=value]` — so `pdf:g4` (token) and `pdf:profile=g4` (param)
are equivalent, and Markdown refinement is `md:refine=apple_fm`. The same
format drives `--do-ocr` (e.g. `apple:lang=fr-FR`); `:` and `=` are
reserved, quote a value to use them literally.

## Related resources

- [How Aglaïa works](/docs/concepts/workflow) — where export sits in the chain
- [OCR engines](/docs/concepts/ocr-engines) — the text layer's source
- [Export](/docs/reference/export) — PDF export reference
- [Export to Markdown](/docs/reference/markdown_export) — Markdown reference
- [The .AGL project file](/docs/concepts/agl-project-file) — slim-project export
