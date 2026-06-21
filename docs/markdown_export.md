# Markdown export

`lib/workers/md_export.py` turns the OCR text stored in `ocr_runs` into a single
free-flowing Markdown file. Entry point: `write_markdown(conn, output_path)` —
used by the GUI export menu (`MainWindow`) and the headless CLI
(`lib/workers/headless.py`, `--export … md`).

Structural markers (`<!-- scan #N · page P -->`, `<!-- branch X -->`) are emitted
as HTML comments so they vanish in the rendered document but survive a future
re-import.

## Source shapes

Per `(scan, branch)` the latest `done` run is selected, then classified:

| Source | Origin | Handling |
|---|---|---|
| `meta.markdown` | PaddleOCR-VL — fully assembled, page-aware MD per page | emitted verbatim |
| `meta.structure` | Surya — per-block `label`/`html`/`reading_order` | `_render_structure` → headings / lists / quotes / tables / formulas |
| `meta.document` | Apple Document engine (`apple_docs`) — reading-ordered typed-block tree (paragraph / list / table) | `_render_document` → titles → `#`/`##`, lists → `-`, tables → MD tables, paragraphs → prose |
| `lines` | Apple Vision &c — only `(text, bbox, confidence)` per line | geometric inference (below) |

The engine-assembled paths are trusted as-is; only the line-only path is
reconstructed. Classification order is `markdown` → Surya `structure` →
`document` → `lines`, so an `apple_docs` run with a structured tree renders
through `_render_document` and only falls back to the geometric line path
when the structured pass is empty.

## Line-geometry pipeline

Apple Vision returns no font weight, italic flag, or block structure — just a
text string, an integer bbox, and a confidence per line. Structure is inferred
from geometry by a **document-level** pass over all line pages
(`_render_line_pages`):

1. **Confidence filter** — lines below `_MIN_CONFIDENCE` (0.3) are dropped as
   figure / margin noise (`_line_items`).
2. **Reading order** (`_reading_order`) — a clear central gutter with both sides
   populated and nothing spanning it is read as two columns (left fully, then
   right); otherwise top-to-bottom, left-to-right.
3. **Per-page geometry** (`_page_geometry`) — median line height, median
   inter-line gap, left margin (10th pct of `x0`), right margin (90th pct of
   `x1`), plus per-line height ratio and gap-above/gap-below.
4. **Running head / footer / folio removal** (`_mark_running`) — a short line
   recurring in the same top/bottom band on ≥3 pages is a running head; the
   **first** occurrence is kept (it's usually the real chapter title) and the
   repeats dropped. Standalone page numbers (`_is_page_number`) are dropped
   everywhere.
5. **Heading detection + level** (`_is_heading`, `_build_heading_levels`) —
   multi-cue score (height ratio, isolation, centering, brevity, ALL-CAPS,
   explicit `Chapitre/Section/…` keywords, numbered section markers like
   `1. — TITRE`). A keyword or a caps numbered-section marker is decisive even
   when the bbox is no taller than body text. Heading *level* (`#`/`##`/`###`)
   comes from clustering the candidate heights across the **whole document**, so
   the same physical title gets the same depth on every page.
6. **Paragraph assembly** (`_render_page_blocks`) — consecutive lines fuse into
   one paragraph. A trailing `-`/`¬` is treated as a soft hyphen and joined with
   no space (`_join_lines`). A new paragraph starts on a large vertical gap, a
   first-line indent (alinéa), or a short terminated previous line — but a
   continuation cue (lowercase start after an unterminated line) suppresses a
   spurious break.
7. **Block quotes** — a paragraph whose every line is indented from the left
   margin and which runs ≥2 lines becomes a `>` block (distinct from a one-line
   alinéa).
8. **Lists** (`_list_marker`) — bullet (`•·*…`) and numbered/lettered (`1.`,
   `a)`, `IV.`) markers → `-` items. A leading **em/en-dash is intentionally not
   a bullet** (French dialogue), so `— Bonjour …` stays prose.
9. **Footnotes** (`_mark_footnotes`) — detected by **region**, not per-line
   size: in a critical edition the notes sit in a block at the foot of the
   page, cut off from the body by a gap far larger than any inter-paragraph
   gap. The separator is found as the single largest `gap_above` in the bottom
   band that clearly outranks the next-largest (≥1.8×); everything below it is
   footnote text (with a small-and-low fallback). Apparatus lines fuse into one
   `>` block per note (split on the leading note number), soft-hyphenation
   joined — so a note broken across physical lines reads as one sentence.
10. **Cross-page continuation** (`_merge_paragraphs`) — a paragraph left
    unterminated at a page break and continued (lowercase start) on the next
    page is stitched into one paragraph; the new page's marker is spliced in as
    an **inline** HTML comment so flow is preserved.
11. **Bold** — a short single-line ALL-CAPS paragraph that isn't a heading is
    wrapped in `**…**` (`_looks_bold`).

## Tuning & limitations

- Thresholds live in `_page_geometry` (`para_gap = 1.6 × median_gap`,
  `indent_thresh`, `short_thresh`) and the cue weights in `_is_heading`
  (fires at score ≥ 1.8).
- Critical editions with dense footnote apparatus produce many `>` blocks —
  expected; the apparatus is genuinely hard to separate from body by geometry
  alone, and `>` visually sets it apart.
- Soft-hyphen joining occasionally fuses a real compound (`peut-être` → loses
  the hyphen when the OCR split it across lines) — accepted trade-off for
  dehyphenating ordinary words.
- Italics are unrecoverable from line-level bboxes and are left unstyled.

Tests: `tests/workers/test_md_export.py` — one case per heuristic on synthetic
line data, plus the clustering/utility units.

## Optional LLM refinement (`lib/workers/ocr/llm_refine.py`)

The geometry recovers *structure* but can't fix what OCR got wrong at the
character/semantic level: broken accents, mis-split words, garbled Greek, a
heading the cues missed. An **on-device LLM** can — privately, offline. This is
a post-pass over the exported Markdown, off by default.

`write_markdown(conn, path, refine="apple_fm", refine_mode="cleanup")` runs each
page through the backend after the heuristics; `refine=None` (default) keeps the
pure-heuristic output. Exposed as:

- **GUI** — a "Polish with Apple Intelligence (on-device)" toggle on the
  Markdown card. It probes availability on build and disables itself with an
  explanatory tooltip until the framework is present.
- **CLI** — `--md-refine apple_fm` (headless export).

### Backends

| Backend | When | Notes |
|---|---|---|
| `AppleFMBackend` | `apple-fm-sdk` present, **macOS 26+**, Apple Intelligence on | On-device Apple Foundation Models. Lazy import + `SystemLanguageModel.is_available()` gate. |
| `NullBackend` | default / unavailable | Always `(False, reason)`; export stays heuristic-only. |
| `MockBackend` | tests | Applies a caller fn per page. |

### Design constraints

- **Per-page chunking.** Each page is refined in a *fresh*
  `LanguageModelSession` (no transcript accumulation); pages over
  `_MAX_INPUT_CHARS` (~12 kB) pass through untouched. Page markers are
  stripped from the model input and re-attached verbatim.
- **Faithful, not creative.** The instruction forbids translating, summarising,
  reordering, or inventing; it's OCR correction, not authorship. Garbled passages
  are left as-is.
- **Fail open.** Any backend error returns the page's heuristic Markdown
  unchanged — a bad LLM never loses data.

### Status

- Requires macOS 26 (Tahoe) + `pip install apple-fm-sdk`; a no-op on earlier
  macOS. The `structure` mode (full semantic re-segmentation from raw lines)
  is scaffolded but unused — the default is the conservative `cleanup` mode.

Tests: `tests/workers/test_llm_refine.py` (page splitting, null no-op, mock
refinement, oversize pass-through, fail-open).
