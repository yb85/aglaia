# Quickstart

Get from a stack of book photos (or a PDF) to a searchable PDF / Markdown in
a few minutes. This is the guided happy path; the reference docs in the
[table](./README.md) go deeper on every step.

Aglaïa has **one entry point**, `aglaia`, in two modes:

- **GUI** (macOS) — webcam capture + live processing: `aglaia <workspace>`
- **Headless CLI** (cross-platform) — same pipeline, no Qt:
  `aglaia <workspace> --headless …`

A *workspace* is a project directory; its state lives in a single
`<name>.scanproj.sqlite` (a.k.a. `.agl`) file — see [storage.md](./storage.md).

---

## 1. Install

Managed with [`uv`](https://docs.astral.sh/uv/). Pick the extras for what you
need (base deps are GUI-free):

```bash
# Headless / CLI / CI — no Qt
uv sync --extra dev

# GUI (cross-platform Qt)
uv sync --extra dev --extra gui

# macOS capture GUI (Qt + Vision/AVFoundation + MLX dewarp)
uv sync --extra dev --extra gui --extra macos

# Linux + NVIDIA GPU (CUDA 12 JAX wheels — faster dewarp)
uv sync --extra dev --extra cuda
```

> OCR via **Apple Vision** is macOS-only; **Surya** is cross-platform. Voice
> control (Vosk) is the `voice` extra. Cloud OCR (Mistral) is `cloud`.

---

## 2a. Capture from a webcam (GUI, macOS)

```bash
uv run aglaia ~/scans/my-book
```

1. **Pick a camera** in the capture sidebar (Continuity Camera works).
2. **Set the scale.** The DPI readout shows `DPI: N (uncalibrated)`. Either:
   - **Click the readout** to type a value manually (shows `(manual)`), or
   - **Calibrate DPI** — hold an ISO ID-1 credit-card-sized object flat in
     frame; the dialog auto-detects it, holds steady, and auto-captures a
     measurement. (One-off camera distortion correction lives behind **Full
     Calibration** with the A4 chessboard target — see
     [calibration.md](./calibration.md).)
3. **Capture** each page (button, keyboard, or say "photo" with voice
   control). Each shot runs the pipeline live; the scans column shows
   raw → stages → output per page.
4. When done, **export** (see step 4).

## 2b. Import images or a PDF (GUI or headless)

The import panel accepts multiple images and PDFs (per-page extract or
render). From the CLI, just pass the files:

```bash
# Ingest images into a new/!existing project
uv run aglaia ~/scans/my-book.agl --headless page-01.jpg page-02.jpg

# Extract a PDF's pages and process them
uv run aglaia ~/scans/my-book.agl --headless scan.pdf --input-dpi 300
```

`--input-dpi` sets the assumed resolution for imported **images** (PDFs
estimate per-page DPI from paper size).

---

## 3. Process

Both modes run the same YAML-defined pipeline (default
`config/pipelines/book_curved_x2.yaml` — DPI clamp → skew → page detect →
dewarp → binarize). Override with `-p`:

```bash
uv run aglaia ~/scans/my-book.agl --headless \
    -p book_curved_x2 --workers 4
```

`-p` takes a bundled name (`book_curved_x2`) or a path to a `.yaml`. See
[pipeline.md](./pipeline.md) for the schema and
[processors.md](./processors.md) for what each step does (and how to add your
own — built-in or drop-in plugin).

---

## 4. OCR + export

OCR and export are off-chain passes you can run in the same headless
invocation:

```bash
uv run aglaia ~/scans/my-book.agl --headless \
    --do-ocr apple:lang=fr-FR \
    --export "pdf:g4+md"
```

- `--do-ocr [ENGINE[:opt…]]` — `auto` | `apple` | `surya` | any registered
  engine; e.g. `apple:lang=fr-FR`, or `--ocr-lang fr-FR+en-US`. Surya
  autodetects language. See [ocr.md](./ocr.md).
- `--export` — `+`-joined specs. `pdf:<profile>` (`jbig2` default / `g4` /
  `native`) makes a searchable PDF (image + invisible OCR text layer); `md`
  makes structured Markdown. `md:refine=apple_fm` post-refines the Markdown.
  See [export.md](./export.md) and [markdown_export.md](./markdown_export.md).

In the **GUI**, the same actions live in the export panel.

---

## Full headless one-liner

Ingest a PDF, process, OCR, and emit both a G4 PDF and Markdown:

```bash
uv run aglaia ~/scans/my-book.agl --headless scan.pdf \
    -p book_curved_x2 --workers 4 \
    --do-ocr apple --ocr-lang fr-FR+en-US \
    --export "pdf:g4+md"
```

Outputs land next to the `.agl` project file.

---

## Where next

- Something looks wrong in the output → [lessons.md](./lessons.md) (DPI
  pitfalls, dewarp quirks) and per-page step toggles in [gui.md](./gui.md).
- Tune or write a pipeline → [pipeline.md](./pipeline.md),
  [processors.md](./processors.md).
- Extend without touching the repo → drop-in
  [processor / OCR plugins](./processors.md#drop-in-user-plugins-no-repo-edit).
