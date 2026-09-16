# Export

Aglaïa exports a project to a **searchable PDF** or **Markdown**. Both
read the chosen stage of each visible page from the project DB.

> **Not yet pluggable.** Unlike processors and OCR engines (drop-in
> autodiscovery, no central map), export targets are hardcoded: the allowed
> names in `cli.py` `_parse_export_arg`, the dispatch in
> `headless.py:_run_exports`, and the GUI imports in `MainWindow`. Adding a
> target means editing all three. Making exporters drop-in pluggable
> (`Exporter` base + `@register` + `<APP_DATA>/plugins/exporters/`) is
> tracked in [#1](https://github.com/yb85/aglaia/issues/1).

## PDF (`aglaia/workers/pdf_export.py`)

A page is encoded by colour type:

- **Bitonal (BW)** → `build_bitonal_pdf(rows, out, engine=...)`:
  - `"jbig2"` *(default, smallest)* — lossless JBIG2 via the in-tree
    `aglaia_jbig2` PyO3 wrapper (`encode_page_lossless`), embedded as a
    `/JBIG2Decode` image XObject.
  - `"g4"` — CCITT Group 4 fallback. Used when JBIG2 is unavailable; the
    encoder probe (`from aglaia_jbig2 import encode_page_lossless`)
    degrades gracefully to G4 if the native extension isn't built.
- **Colour / gray** → JPEG (`/DCTDecode`).

Compression mode (`PDFprocessor.create_pdf_from_db`): `"jbig2"` / `"g4"` /
`"native"` / `"auto"`. `auto` uses the bitonal path (JBIG2 if installed,
else G4) when **every** page is BW, and otherwise falls back to `native`
(all pages as JPEG). `jbig2` / `g4` skip non-BW rows; `native` embeds
every row as a JPEG regardless of type.

An **invisible OCR text layer** (Helvetica, WinAnsi, render mode 3) is
overlaid per page when OCR results exist, so the PDF is selectable /
searchable while showing the scanned image. PDF object assembly + the
text layer go through `pikepdf` (qpdf); page rendering for previews uses
`pypdfium2` (PDFium).

How the layer is laid out (`pdf_export.ocr_text_lines` / `inject_ocr_layer`):

- **Line engines** (Apple Vision, Surya, …) — one run per recognised line,
  font height 0.8 × the line box, squeezed horizontally (`Tz`) to its width.
- **Mistral** stores a page as one full-page line holding the whole
  Markdown. Its real geometry is `meta.mistral_page.blocks`, in the pixel
  frame of the image *sent* to Mistral (`mistral_page.dimensions`), so each
  block is rescaled to the page frame. Inside a block, the text is split
  into visual lines: at least the Markdown lines, and at least
  `√(chars · 0.4 · h / w)` lines — Mistral often runs paragraphs together
  with no break, and one run per block would overflow the page. Markdown
  markers (`#`, `**`, list bullets) are stripped.
- The OCR list is aligned to the **pages actually built**: the bitonal
  builder skips non-BW rows, the native one skips rows it cannot convert.

**The export fails rather than ship a layer it could not write** (#149).
With the layer requested, `create_pdf_from_db` raises `OcrLayerError`
(`expected`, `written`, `missing` 1-based page numbers) and deletes the
file when no page matches a completed OCR run of the selected engine, or
when a page with OCR text got no text on it. A page with no OCR run is not
owed a layer and is not an error. The GUI shows the message in the status
bar; `--headless` prints `! PDF export failed: …` and counts a failure.

Verification: `tests/workers/test_pdf_ocr_layer.py` reads the text with
`get_text_bounded()` (clipped to the page box, like PyMuPDF and
`pdftotext`). An unbounded read passes on text drawn off the page.

> JBIG2 ships only when the build env was synced with `--extra jbig2`
> (it compiles the Rust crate). See [distribution.md](https://github.com/yb85/aglaia/blob/main/docs/distribution.md);
> the encoder is credited in [../ABOUT.md](../ABOUT.md).

## Markdown

`write_markdown` turns OCR text into free-flowing Markdown (headings,
paragraphs, dehyphenation, footnotes, lists, cross-page merge). Full
heuristics in [markdown_export.md](./markdown_export.md).

## OCR textpack

`aglaia/workers/textpack_export.py` (#148). One archive for corpus: the
searchable PDF, its text page by page, and the raw OCR response — the format
corpus reads (`docs/OCR-TEXTPACK.md` in yb85/corpus; reference writer
`corpus/cloud_ocr.py`). Names and keys are corpus's contract: do not vary them.

```
<stem>_OCR.textpack                (zip, written to .partiel then renamed)
└── <stem>_OCR.textbundle/
    ├── info.json                  TextBundle v2 + "corpus" block
    ├── text.md                    <!-- page N --> before each PDF page
    └── assets/
        ├── source.pdf             searchable PDF — fixed name, ZIP_STORED
        └── mistral-<job>.jsonl    raw batch output, byte for byte, per job
```

- **`<stem>`** — the original PDF's stem when every scan comes from one
  imported PDF (`scans.source_ref = <file>#<page>`); otherwise the project
  slug. `corpus.scan_source` is that file name (or the slug).
- **One pass, no drift.** `create_pdf_from_db(..., layer=…)` reports the OCR
  result it laid on each PDF page; `text.md` is rendered from that exact list
  (`md_export.markdown_pages`: same Mistral post-processing and line
  renderers as the Markdown export, but no cross-page paragraph merge, so page
  N of `text.md` is page N of `source.pdf`). A page without OCR still gets its
  marker, with no text.
- **Refusals.** An OCR layer that cannot be written raises `OcrLayerError`
  (#149) — corpus must not receive an unsearchable `source.pdf`. A Mistral
  batch job whose raw output is not stored raises `TextpackError`; run
  `aglaia run <file>.agl --check-ocr` to backfill it (#147).
- **`info.json`** — `version: 2`, `type: net.daringfireball.markdown`,
  `transient: false`, `creatorIdentifier: cc.bibli.corpus.ocr`,
  `cc.bibli.corpus.ocr: {version, stem}`, and `corpus`:
  `scan_source`, `zlib_id` (when given), `double_page` (a scan split into
  several layouts), `date`, and `ocr` — the provenance keys of the reference
  writer: `engine` (`mistral-ocr` for Mistral, else the engine name), `model`,
  `mode` (`batch` / `sync` / `local`), `jobs`, `requests` (`[]`), `pages`,
  `page_numbers` (`null`: every page of `source.pdf`), `pages_document`,
  `tokens` (`null`), `billed_pages` (sum of `usage_info.pages_processed` in the
  raw outputs), `cost_usd` + `cost_basis` (batch $2/1000 on billed pages),
  `submitted_at` / `completed_at` (ISO 8601), `source_sha256` (of the
  original PDF when still on disk), `tool` (`aglaia <version>`), `raw_assets`.

Surfaces: the Export tab's **OCR textpack** card (enabled with OCR, uses the
JBIG2 toggle and the OCR-layer selector); a destination plugin that accepts
`textpack` offers it in its *Export as* picker (send-to-corpus); CLI
`--export textpack[:g4|jbig2|native|auto][:ocr=ENGINE][:zlib=ID]`, with
`--send-to` handing the archive on.

## What gets exported

Only **visible** pages of **non-deleted** scans: queries filter
`scans.deleted_at IS NULL AND branches.trashed_at IS NULL`. Per-page
visibility is the eye toggle (see [gui.md](./gui.md)); each page exports
its currently-chosen stage.
