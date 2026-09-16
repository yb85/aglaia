---
title: Export
description: Turning the processed, OCR'd pages into a searchable PDF, a Markdown document, an OCR textpack, or a slim project copy.
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
| **OCR textpack** | one archive holding all three of the above: the searchable PDF, its Markdown page by page, and the raw response of the OCR engine — see below |
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

## A searchable PDF is searchable, or the export fails

The text layer is invisible, so nobody sees it go wrong. Aglaïa therefore
checks it: if a page carries OCR text and no text lands inside the page box,
the export **fails and deletes the file**, and says which pages are missing.
An unsearchable PDF that looks finished is worse than no PDF — the library
that receives it finds out months later.

A page with no OCR run is owed no text, and is not a failure.

## OCR textpack

For a library that wants the scan, the text and the evidence in one file, the
**OCR textpack** (`<name>_OCR.textpack`) is a zipped
[TextBundle](https://textbundle.org/spec/):

```
<name>_OCR.textbundle/
├── info.json     what this is, and where the OCR came from
├── text.md       the Markdown, each page preceded by <!-- page N -->
└── assets/
    ├── source.pdf            the searchable PDF
    └── mistral-<job>.jsonl   the OCR engine's raw answer
```

- **`text.md` keeps the page boundaries**, so a quotation can be cited by
  page.
- **The raw answer is kept**, so the Markdown or the text layer can be rebuilt
  without paying for the OCR again. Aglaïa stores it in the `.agl` when the
  batch result is imported; for a project OCR'd before that, `aglaia run
  <file>.agl --check-ocr` fetches it back (a download, not a new OCR).
- **`info.json` records the provenance**: engine, model, job ids, pages,
  billed pages, cost, and the times.

The format is the one the [corpus](https://github.com/yb85/corpus) library
reads. With the **send-to-corpus** plugin installed, the textpack goes
straight there.

## Markdown refinement

Markdown export can optionally post-process the text with an on-device LLM
(`--md-refine apple_fm`) to repair line breaks and coherence — a no-op when
the backend is unavailable.

## How to run it

| | Capture GUI | Headless CLI (`aglaia run`) |
|---|---|---|
| PDF | Export tab → PDF | `--export pdf` (or `pdf:g4`, `pdf:jbig2`, …) |
| Markdown | Export tab → Markdown | `--export md` |
| OCR textpack | Export tab → OCR textpack | `--export textpack` (or `textpack:g4`, `textpack:zlib=<id>`) |
| Both | — | `--export pdf:g4+md` |

Exports are written next to the project as `<slug>.pdf` / `<slug>.md`; the
textpack takes the original scan's name, `<name>_OCR.textpack`.

Each `--export` entry uses Aglaïa's standard option-spec format —
`name[:token|key=value]` — so `pdf:g4` (token) and `pdf:profile=g4` (param)
are equivalent, and Markdown refinement is `md:refine=apple_fm`. The same
format drives `--ocr` (e.g. `--ocr apple:lang=fr-FR`; use `--ocr auto` for
the default engine); `:` and `=` are reserved, quote a value to use them
literally.

## Related resources

- [How Aglaïa works](/docs/concepts/workflow) — where export sits in the chain
- [OCR engines](/docs/concepts/ocr-engines) — the text layer's source
- [Export](/docs/reference/export) — PDF, textpack and OCR-layer reference
- [Destinations](/docs/reference/destinations) — sending an export to Kindle,
  Calibre, a folder or a corpus
- [Export to Markdown](/docs/reference/markdown_export) — Markdown reference
- [The .AGL project file](/docs/concepts/agl-project-file) — slim-project export
