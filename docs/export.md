# Export

Aglaïa exports a project to a **searchable PDF** or **Markdown**. Both
read the chosen stage of each visible page from the project DB.

## PDF (`lib/workers/pdf_export.py`)

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
