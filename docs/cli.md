# CLI reference

Aglaïa is a subcommand CLI:

```
aglaia [--version] <command> [options] [arguments]
```

`gui` is the **default command** — running `aglaia` with no command (or with a
project path as the first argument) opens the GUI:

```bash
aglaia                 # → aglaia gui            (start window)
aglaia ~/book.agl      # → aglaia gui ~/book.agl (open that project)
```

Commands: [`gui`](#gui), [`run`](#run), [`ocr`](#ocr), [`setup`](#setup),
[`list`](#list), [`server`](#server), [`version`](#version). `aglaia --help` and
`aglaia <command> --help` print the live usage.

> `aglaia` is a console script (`pip install aglaia`); from source use
> `uv run aglaia …`. Entry path: `aglaia/__main__.py:run` → `aglaia/cli:run`
> (a Typer app); the commands live in `aglaia/cli/commands/`. The internal
> config layer they build (`CliConfig`, the `--ocr`/`--export` spec parsers) is
> documented in [configuration.md](configuration.md); the implementation plan is
> [subcommand-cli.md](subcommand-cli.md).

## Shared options

These apply to both `gui` and `run`:

| Option | Meaning |
|---|---|
| `-p`, `--pipeline NAME\|PATH` | Pipeline name (e.g. `book_curved_x2`) or a `.yaml` path. |
| `--workers N` | Pipeline worker processes (overrides config). `0` = auto. |
| `--force-proc` | Reprocess every active scan on open (wipe branches/intermediates). |

## `gui`

```
aglaia gui [PROJECT] [options]
```

Launch the capture GUI (the default command). With no `PROJECT` it opens the
start window; with a `.agl` / PDF / image it opens or ingests that. Falls back to
headless if Qt isn't installed.

| Option | Meaning |
|---|---|
| `PROJECT` | A `.agl` project to open (optional). |
| `--camera-id N` | Capture camera index. |
| `--diagnose-memory` | `tracemalloc` snapshots in the GUI process. |
| shared | `-p/--pipeline`, `--workers`, `--force-proc`. |

```bash
aglaia gui ~/scans/my-book.agl
aglaia gui --camera-id 1 -p book_curved_x2
```

## `run`

```
aglaia run PATHS… [options]
```

Headless batch: ingest → pipeline → (OCR) → (export), no Qt. `run` is **always
headless** — there is no `--headless` flag. `PATHS` is one `.agl` project to
(re)process, **or** one or more PDFs, **or** one or more images to ingest into a
new project.

| Option | Meaning |
|---|---|
| `--ocr ENGINE[:opt…]` | Run OCR. **Requires a value** — use `--ocr auto` for the default (Apple Vision → Surya). e.g. `--ocr surya:lang=fr-FR`, `--ocr mistral:batch`. |
| `--ocr-lang CODES` | `+`-joined BCP-47 codes (e.g. `fr-FR+en-US`) or `auto`. |
| `--export SPECS` | `+`-joined export specs, e.g. `pdf:g4+md`. |
| `--md-refine BACKEND` | On-device LLM backend for Markdown cleanup, e.g. `apple_fm`. |
| `--project-name NAME` | Name for a new project (default: from the input filename). |
| `--parent-dir DIR` | Parent folder for a new project. |
| `--input-dpi [force:]N` | Input DPI for imported images; `force:N` overrides every input. |
| `--check-ocr` | Poll + import pending Mistral batch OCR jobs for the project, then exit. |
| shared | `-p/--pipeline`, `--workers`, `--force-proc`. |

```bash
# Open an existing project, run the full pipeline, OCR (FR+EN), export both
aglaia run ~/scans/book.agl -p full \
  --ocr auto --ocr-lang fr-FR+en-US --export pdf:g4+md

# Ingest a PDF into a new project and export a searchable PDF
aglaia run ~/scans/book.pdf --project-name book --ocr auto --export pdf:g4

# Poll a previously-submitted Mistral batch and import results
aglaia run ~/scans/book.agl --check-ocr
```

### Option-spec format

`--ocr` and `--export` entries share one **option-spec format** (`parse_spec` in
`aglaia/workers/cli.py`): `name[:token|key=value][:…]`. `:` and `=` are reserved
(quote a value to use them literally). Positional `token`s are flags (e.g.
`pdf:g4` selects the G4 profile, `mistral:batch` selects batch mode); `key=value`
pairs are params (e.g. `apple:lang=fr-FR`). OCR engines receive params via
`OcrEngine.configure(params)`; for PDF the profile is the first token (or
`profile=`); `md:refine=apple_fm` mirrors `--md-refine`. See [ocr.md](ocr.md) and
[export.md](export.md).

## `ocr`

```
aglaia ocr PATHS… [options]
```

OCR documents that **don't need processing** — born-digital PDFs, flat scans —
without the geometric pipeline (no dewarp / binarize / page-split). Each page is
ingested as the raw colour image and OCR'd directly, then exported. Headless, no
Qt, no processing chain. `PATHS` is one or more PDFs/images to OCR into a new
project, **or** one `.agl` to re-OCR an existing project (or, with `--check-ocr`,
poll its pending Mistral batch jobs).

Same options as `run` **minus** `-p/--pipeline`, `--workers`, `--force-proc`
(there is nothing to process): `--ocr` (defaults to `auto` if omitted — OCR is
the point), `--ocr-lang`, `--export`, `--md-refine`, `--project-name`,
`--parent-dir`, `--input-dpi`, `--check-ocr`.

```bash
# OCR a clean PDF straight to a searchable PDF + Markdown
aglaia ocr ~/scans/clean.pdf --project-name clean --export pdf:g4+md

# OCR a folder of page images with a specific engine + language
aglaia ocr ~/pages/*.png --ocr surya --ocr-lang fr-FR --export md
```

When to use `ocr` vs `run`: reach for `run` when the photos need straightening,
page-splitting, or binarizing; reach for `ocr` when the input is already a clean
page and you only want text out.

## `setup`

```
aglaia setup
```

Interactive first-run setup (CLI-only installs): language, models, defaults.

## `list`

```
aglaia list {pipelines|ocr|exports}
```

List available pipelines, OCR engines, or export formats.

```bash
aglaia list pipelines
aglaia list ocr
aglaia list exports
```

## `server`

```
aglaia server [--host HOST] [--port 4674] [--public-url URL]
```

Run the long-running HTTP job server (needs the `server` extra:
`pip install "aglaia[server]"`). Submit an `.aglbundle` (from aglaia-bridge) or a
PDF and get back a searchable PDF (+ Markdown when OCR is on). Full reference:
[server.md](server.md).

| Option | Meaning |
|---|---|
| `--host HOST` | Bind address (default `127.0.0.1`; use `0.0.0.0` to accept LAN/remote clients). |
| `--port PORT` | Port to listen on (default `4674`). |
| `--public-url URL` | Public base URL for download links in emails, e.g. `https://scan.example.com`. |

On start it prints the bound URL and the admin-panel URL (with the secret).

## `version`

```
aglaia version       # or: aglaia --version
```

Print the Aglaïa version and exit.

## What changed from the old flat CLI

The old single-command form (`aglaia <workspace_dir> --headless …`) is gone.
Mapping:

| Old | New |
|---|---|
| `aglaia <dir>` (capture GUI) | `aglaia gui <dir>` (or just `aglaia <dir>`) |
| `aglaia <path> --headless …` | `aglaia run <path> …` |
| `aglaia --setup` | `aglaia setup` |
| `aglaia --pipeline-list` | `aglaia list pipelines` |
| `aglaia --ocr-list` | `aglaia list ocr` |
| `aglaia --export-list` | `aglaia list exports` |
| bare `--ocr` (default engine) | `--ocr auto` (the flag now requires a value) |
