---
name: aglaia-cli
description: Drive the Aglaïa book-scanner CLI — turn photos, scans or PDFs of book pages into clean, OCR'd, searchable PDF and Markdown; choose between gui/run/ocr, pick a pipeline, set the DPI, run OCR, export, send to plugins, manage plugins.
---

# Aglaïa CLI — agent skill

Aglaïa turns photographs of book pages (webcam, phone, flatbed, PDF) into clean,
straight, black-and-white, OCR'd pages, exported as searchable PDF or Markdown.
This document is what an agent needs to use the `aglaia` command well: what each
command is for, which one fits which situation, what to ask the user before
running anything, and the mistakes that cost the most. `aglaia skill` prints it
from the installed binary, so it always matches the commands on the machine.

## 1. Mental model

- A **project** is one file, `NAME.agl` (SQLite). It holds the raw captures
  (**scans**), every intermediate step, the final **pages**, the OCR text, and
  the settings. Everything the CLI does either creates a project from inputs or
  works on an existing one.
- One **scan** = one photo. A photo of an open book holds two pages; the
  pipeline splits it (`_x2` pipelines). A photo of a single sheet holds one
  (`_x1`).
- The **pipeline** is a YAML-defined chain of processors (DPI clamp → deskew →
  page split → DPI normalise → binarise → deskew → keystone / dewarp → margins).
  It runs in worker processes; the same chain runs in the GUI and headless.
- **OCR** is a separate pass over the finished pages, per engine. **Exports**
  (PDF, Markdown) are produced from pages + OCR. **Export plugins**
  ("destinations") then hand the exported files somewhere: a Kindle mailbox, a
  Calibre library, a corpus server.
- **DPI is the master parameter.** Every cleanup step measures in millimetres
  and scales by DPI. A wrong DPI (most phone photos carry none) breaks strokes,
  smears the B&W, over-curls the dewarp. When output looks bad, check the DPI
  before anything else (§7).

## 2. Invoking

```bash
aglaia --version              # installed as a console script (pip / DMG)
uv run aglaia …               # from a source checkout
aglaia skill                  # this document, to stdout
```

- `aglaia` with no command opens the GUI (`gui` is the default command). Any
  first token that is not a command is treated as a project/file for `gui`:
  `aglaia book.agl` == `aglaia gui book.agl`.
- `run`, `ocr`, `list`, `plugins`, `server`, `version`, `skill` never need Qt.
  On a headless box install without the `gui` extra.
- macOS-only: webcam capture, the Apple Vision / `apple_docs` OCR engines, the
  `apple_fm` Markdown refiner. Everything else (pipeline, Surya/GLM local OCR,
  Mistral cloud OCR, exports, plugins) is cross-platform.
- Exit code is non-zero on failure. Progress and warnings go to stderr; keep
  stdout for what the command produces.

## 3. Which command?

| The user has… | …and wants | Use |
|---|---|---|
| Photos of book pages (phone, webcam, camera), or a PDF of such photos — pages are tilted, curved, two-per-shot, grey/colour | clean pages, searchable PDF/Markdown | **`aglaia run`** |
| An already clean document (born-digital PDF, flatbed scan, screenshots) | just the text: searchable PDF / Markdown | **`aglaia ocr`** |
| A book in front of a webcam / phone, or wants to inspect and hand-tune results per page | interactive capture, per-page fixes, live preview | **`aglaia gui`** |
| An existing `.agl` project | re-process, OCR, export, send | `run` (with pipeline) or `ocr` (no pipeline) on the `.agl` |
| A pending Mistral batch OCR job | fetch the results | `aglaia run PROJECT.agl --check-ocr` (or `ocr … --check-ocr`) |
| Many users / a machine on the LAN | a job API | `aglaia server` |
| Nothing yet | first-run setup (language, models) | `aglaia setup` |

Rule of thumb from the docs: **`run` when the photos need straightening,
page-splitting or binarising; `ocr` when the input is already a clean page and
you only want text out.** `ocr` skips the whole chain — running it on curved
photos gives poor OCR and no cleanup.

## 4. Questions to ask before running

Ask only what changes the command. Typical minimum: (1) what the input is,
(2) pages per photo + flat or curved, (3) DPI if photos, (4) language(s),
(5) what outputs. Everything else has a sane default.

1. **Input** — photos (which folder / formats), a PDF (photos inside, or
   born-digital?), or an existing `.agl`? A `run`/`ocr` call takes *one* `.agl`
   **or** one or more PDFs **or** one or more images — not a mix.
2. **Page geometry** — one page per photo or an open book (two)? Do the pages
   bend (thick book, spine curl) or lie flat (pressed, flatbed, loose sheets)?
   → pipeline (§6).
3. **DPI** — does the user know the capture resolution? If not, get the page
   size (A4/A5/…) and the image width in pixels, or a lowercase letter's height
   in pixels, and compute it (§7). Pass `--input-dpi N` (or `force:N`).
4. **Language(s)** — for OCR accuracy: `--ocr-lang fr-FR+en-US`. `auto` works
   for common Latin-script text. Greek/Hebrew/Arabic/CJK → Surya, GLM, or
   Mistral cloud, not Apple Vision.
5. **Outputs** — PDF (which profile: `auto` default, `g4` smallest B&W, `jbig2`
   smaller still if available, `native` keeps grey/colour), Markdown, both?
   Should Markdown be LLM-cleaned (`--md-refine apple_fm`, on-device, macOS)?
6. **Where it goes** — keep files next to the project (default), or hand them
   to an installed export plugin (`--send-to send-to-kindle`)? Cloud OCR
   (Mistral) sends page images to a third party and costs money — confirm
   before using it; local engines never leave the machine.
7. **Project** — name (`--project-name`, default: input filename, kept exactly
   as typed) and folder (`--parent-dir`). Re-running on an existing `.agl`
   reuses its work; `--force-proc` throws every intermediate away first —
   confirm before using it on a project the user hand-edited in the GUI.
8. **Machine** — macOS? (Apple Vision available.) GPU? How many cores?
   (`--workers 0` = auto, ≈ number of performance cores; leave it.)

## 5. Command reference

### `aglaia gui [PROJECT]` (default)

Launch the capture GUI. No `PROJECT` → start window; a `.agl` → opens it; a
PDF/image → ingests it into a new project. Falls back to headless if Qt is
missing. Options: `-p/--pipeline`, `--workers`, `--force-proc`, `--camera-id N`
(capture camera index), `--diagnose-memory`.

### `aglaia run PATHS… [options]` — full pipeline, headless

`PATHS`: images, PDFs, or one `.agl`. Never needs Qt; there is no `--headless`
flag because `run` is always headless.

| Option | Meaning |
|---|---|
| `-p, --pipeline NAME\|PATH` | `book_curved_x2` (default), `book_flat_x2`, `book_flat_x1`, `sheet_flat_x1`, or a `.yaml` path. |
| `--workers N` | Worker processes; `0` = auto. |
| `--force-proc` | Reprocess every active scan from the raw capture (wipes intermediates and per-page manual edits). |
| `--ocr ENGINE[:opt…]` | Run OCR. **Needs a value**: `--ocr auto` = Apple Vision → Surya. Engines: `aglaia list ocr`. |
| `--ocr-lang CODES` | `+`-joined BCP-47, e.g. `fr-FR+en-US`; default `auto`. |
| `--export SPECS` | `+`-joined: `pdf`, `pdf:g4`, `pdf:jbig2`, `pdf:native`, `md`, `md:refine=apple_fm`. e.g. `pdf:g4+md`. |
| `--md-refine BACKEND` | On-device LLM cleanup of the Markdown (`apple_fm`). Same as `md:refine=…`. |
| `--send-to SLUGS` | After exporting, hand the files to these export plugins, `+`-joined (`send-to-kindle+send-to-corpus`). Installed ones: `aglaia list destinations`. |
| `--project-name NAME` | New project's name (default: from the input filename). |
| `--parent-dir DIR` | Folder in which the new `NAME.agl` is created. |
| `--input-dpi [force:]N` | DPI for imported images that carry none; `force:N` overrides every input, even ones with metadata. |
| `--check-ocr` | Poll + import pending Mistral batch OCR jobs for the project, then exit. |

### `aglaia ocr PATHS… [options]` — OCR only, no pipeline

Same `PATHS` rule. Same options as `run` **minus** `--pipeline`, `--workers`,
`--force-proc` (nothing is processed), **plus** `--ocr-dpi N` (pages are
downsampled to this before OCR; default 200, which is what the GUI uses). `--ocr`
defaults to `auto` here — OCR is the point. Each page is ingested as-is
(colour kept) and OCR'd.

### Option-spec format (`--ocr`, `--export`)

`name[:token|key=value][:…]`. Positional tokens are flags (`pdf:g4` = profile,
`mistral_cloud:batch` = batch mode); `key=value` are parameters
(`apple_vision:lang=fr-FR`, `md:refine=apple_fm`). `:` and `=` are reserved —
quote a value to use them literally. Several specs join with `+`.

### `aglaia list pipelines|ocr|exports|destinations`

What this install can do. `ocr` marks engines `(unavailable)` when their
dependency/model/platform is missing — check it before promising an engine.
`destinations` lists installed export plugins and whether each is set up.

### `aglaia plugins …`

Plugins are the only way features beyond the core reach the app: **nothing
ships by default**; each is installed from the registry
(github.com/yb85/aglaia-plugins) or a local archive into
`<APP_DATA>/plugins/<kind>/<slug>/`. Kinds: `processors` (pipeline steps, e.g.
a stamp remover), `ocr` (engines), `destinations` (export plugins).

| Command | Does |
|---|---|
| `aglaia plugins list` | Installed plugins: version, enabled/disabled, and what each still needs (settings, secrets, dependency). |
| `aglaia plugins search [TERM]` | What the registry offers (filter by slug, name, summary). |
| `aglaia plugins install SLUG` | From the registry. `aglaia plugins install FILE.aglplugin --trust [--kind processors\|ocr\|destinations]` for a local archive — `--trust` is mandatory because nobody reviewed it. |
| `aglaia plugins update SLUG` / `--all` | To the registry's newer version; keeps settings, files and secrets. |
| `aglaia plugins toggle SLUG` | Disable ↔ enable. A pipeline that references a disabled/uninstalled processor **fails with an error** — it does not silently skip the step. |
| `aglaia plugins remove SLUG [-y]` | Uninstall, including its settings and any password it stored in the keychain. |
| `aglaia plugins config SLUG` | Interactive view (select/text/password prompts) of an export plugin's settings. `--set key=value` (repeatable) for scripts; `--test` checks the connection afterwards. |

Secrets (passwords, API keys, SMTP credentials) go to the OS keychain,
namespaced per plugin; other settings go to the config database. Only
`destinations` plugins have `config`; processor plugins are configured through
their pipeline options.

### `aglaia setup`

Interactive first run: UI language, models to download, defaults. Run it once on
a fresh machine, or when `list ocr` shows every engine unavailable.

### `aglaia server [--host 127.0.0.1] [--port 4674] [--public-url URL]`

Long-running HTTP job API (`server` extra). `POST /run` submits a job (API-key
form field), `GET /list`, `/check/{id}`, `/get/{id}` (status + download URLs),
`GET /download/{id}/{pdf|md}` (capability URL), `POST /delete/{id}`,
`/admin?secret=` (keys, stats), `/health`. Jobs are stored in
`APP_DATA/aglaia-server.db`. Use `--host 0.0.0.0` to accept LAN clients and
`--public-url` so emailed links resolve.

### `aglaia version` · `aglaia --version` · `aglaia skill`

## 6. Pipelines

| Name | Use when | Chain |
|---|---|---|
| `book_curved_x2` (default) | Open book, two pages per photo, pages bend toward the spine (most hand-held/webcam book captures) | clamp DPI → deskew capture → split 2 pages → normalise DPI → B&W → deskew → keystone → **dewarp (cubic sheet)** → margins |
| `book_flat_x2` | Open book, two pages per photo, pages pressed flat (glass, heavy press) — no dewarp, faster | same, keystone only |
| `book_flat_x1` | One book page per photo, flat; text from the facing page is discarded | 1-page layout, keystone |
| `sheet_flat_x1` | Loose sheets, printouts, one page per photo | 1-page layout, keystone |

Custom pipelines: pass a `.yaml` path; user pipelines also live in
`<APP_DATA>/pipelines/`. Dewarp is the slowest step — pick a `_flat_` pipeline
when the pages really are flat. If a book is photographed one page at a time
but bends, there is no `book_curved_x1`: use `book_curved_x2` and expect the
empty half to be dropped, or ask the user for two-page shots.

## 7. DPI — get it right first

Symptoms of wrong DPI: broken/merged letters, smeared B&W, over-curled dewarp,
noise surviving. Within ~15 % is fine; 3× off is fatal.

- **From page size**: `DPI = image_width_px × 25.4 / page_width_mm`
  (A5 148 mm, A4 210 mm, Letter 216 mm). A 1700 px-wide A5 photo → ≈ 290.
  For a two-page spread use the spread width (2 × page width + gutter).
- **From letter size**: `DPI ≈ x_height_px × 13` for ~11 pt body text (a
  lowercase letter ~8 px tall → ≈ 104).
- Pass it: `--input-dpi 290` (only where metadata is missing) or
  `--input-dpi force:290` (override everything, e.g. a PDF that claims 72).
- Scanners and PDFs usually carry a correct DPI; phone photos never do.
- In the GUI: Capture panel DPI readout / "Calibrate DPI" with a credit card;
  Pipeline panel → "Fix input DPI" for imported pages.

## 8. OCR engines

| Engine | Where | Good for | Notes |
|---|---|---|---|
| `auto` | — | default | Apple Vision on macOS, else Surya. |
| `apple_vision` | macOS, local, no download | printed Latin scripts, fast | Cannot read faint small-caps running heads or non-Latin scripts. |
| `apple_docs` | macOS 26, local | structured OCR (tables); offloads scripts Vision can't read to a complement engine | |
| `surya` | local VLM (MLX on macOS / vLLM on CUDA) | Markdown, tables, 90+ scripts | Needs the model downloaded (`aglaia setup`); slow on CPU. |
| `glm` | local 0.9B VLM | Markdown + tables | Same runtime as Surya. |
| `unlimited` | local VLM | fused multipage, Markdown + tables | |
| `mistral_cloud` | cloud | any script, clean Markdown | Needs `MISTRAL_API_KEY` (keychain or `APP_DATA/.env`). `mistral_cloud:batch` submits asynchronously; fetch later with `--check-ocr`. Sends images off-machine; billed per page. |

`--ocr-lang` matters most for Apple Vision; VLM engines detect the script.
Verify availability with `aglaia list ocr` before choosing.

## 9. Exports and destinations

- `pdf` profiles: `auto` (picks per page), `g4` (CCITT G4, tiny B&W — the
  usual choice for text), `jbig2` (smaller, needs the encoder), `native` (keeps
  grey/colour pages as they are). The PDF carries the OCR text layer.
- `md`: one Markdown file for the project; `md:refine=apple_fm` (or
  `--md-refine apple_fm`) runs an on-device LLM cleanup pass (macOS).
- Files are written next to the project, named after it, exactly as the user
  typed the project name (no slugifying).
- `--send-to SLUG[+SLUG]` runs after export and hands the files to installed
  export plugins. A destination that is not set up fails before anything is
  sent: run `aglaia plugins config SLUG --test` first. Slow destinations (mail
  to Kindle) are normal.

## 10. Where things live

| What | macOS | Linux | Windows |
|---|---|---|---|
| `APP_DATA` (config DB, `pipelines/`, `models/`, `plugins/`, `.env`, server DB) | `~/Library/Application Support/Aglaia` | `~/.local/share/Aglaia` | `%APPDATA%\bibli.cc\Aglaia` |
| Logs (one rotated file per session) | `~/Library/Logs/Aglaia` | `~/.local/state/Aglaia/log` | `%LOCALAPPDATA%\bibli.cc\Aglaia\Logs` |
| Cache (safe to delete) | `~/Library/Caches/Aglaia` | `~/.cache/Aglaia` | `%LOCALAPPDATA%\bibli.cc\Aglaia\Cache` |

`AGLAIA_APP_DATA_DIR=/path` relocates `APP_DATA` (useful for tests and
sandboxes). The project `.agl` lives wherever the user put it (`--parent-dir`).

## 11. Recipes

```bash
# Phone photos of an open, curved book → project + searchable PDF + Markdown (FR/EN)
aglaia run ~/shots/*.jpg --project-name "Les Confessions" --parent-dir ~/scans \
  --input-dpi 290 --ocr auto --ocr-lang fr-FR+en-US --export pdf:g4+md

# A PDF made of photographed pages, flat, two per page
aglaia run ~/in/book.pdf -p book_flat_x2 --ocr auto --export pdf:g4

# Loose A4 printouts photographed one at a time
aglaia run ~/sheets/*.png -p sheet_flat_x1 --input-dpi 250 --ocr auto --export pdf

# Re-process an existing project with another pipeline, from scratch
aglaia run ~/scans/book.agl -p book_flat_x2 --force-proc

# Already-clean PDF: just OCR it, straight to PDF + Markdown, no pipeline
aglaia ocr ~/in/clean.pdf --export pdf:g4+md

# Greek text with a local VLM, Markdown only
aglaia ocr ~/pages/*.png --ocr surya --ocr-lang el-GR --export md

# Cloud batch OCR (cheap, asynchronous): submit, then fetch later
aglaia run ~/scans/book.agl --ocr mistral_cloud:batch
aglaia run ~/scans/book.agl --check-ocr --export pdf:g4+md

# Export and send to a Kindle
aglaia plugins install send-to-kindle
aglaia plugins config send-to-kindle          # interactive; or --set smtp_host=… --test
aglaia run ~/scans/book.agl --export pdf:g4 --send-to send-to-kindle

# What can this machine do?
aglaia list pipelines; aglaia list ocr; aglaia list exports; aglaia list destinations
aglaia plugins search; aglaia plugins list
```

## 12. Gotchas and safety

- `--ocr` **requires a value**: `--ocr auto`, not bare `--ocr`.
- `PATHS` is one `.agl` **or** PDFs **or** images. Don't mix kinds.
- `aglaia something-misspelled` opens the GUI with that token as a project path
  (because `gui` is the default) — check the command name when the GUI pops up
  unexpectedly.
- `--force-proc` destroys per-page manual edits made in the GUI. Confirm.
- `plugins remove` also deletes the plugin's stored password/API key.
  `plugins toggle` keeps everything and is reversible — prefer it when unsure.
- `plugins install FILE --trust` runs unreviewed code with the user's
  privileges; only for archives the user knows.
- A pipeline step provided by a plugin that is disabled or uninstalled makes
  the run **fail** with a clear error rather than skipping the step. Fix with
  `plugins toggle`/`plugins install`, or choose a pipeline without it.
- Cloud OCR (`mistral_cloud`) uploads page images and bills per page. Local
  engines never leave the machine. Ask before choosing cloud.
- Workers are separate processes. Killing the parent (`kill -9`, `timeout`)
  orphans them; if you must kill a run from a script, kill the process group.
  Ctrl-C on the foreground command is fine.
- Measure quality on the output, not on the log: open the PDF / look at the
  page images. A run that finishes is not a run that worked — re-check DPI
  and pipeline choice when pages look wrong.
- Long runs: dewarp is CPU-bound, ≈ 1–4 s per page on a fast laptop; a
  300-page book with OCR is minutes, not seconds. Say so to the user.
