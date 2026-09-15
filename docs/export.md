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
> (it compiles the Rust crate). See [distribution.md](./distribution.md);
> the encoder is credited in [../ABOUT.md](../ABOUT.md).

## Markdown

`write_markdown` turns OCR text into free-flowing Markdown (headings,
paragraphs, dehyphenation, footnotes, lists, cross-page merge). Full
heuristics in [markdown_export.md](./markdown_export.md).

## What gets exported

Only **visible** pages of **non-deleted** scans: queries filter
`scans.deleted_at IS NULL AND branches.trashed_at IS NULL`. Per-page
visibility is the eye toggle (see [gui.md](./gui.md)); each page exports
its currently-chosen stage.
