---
title: Install
description: Download the macOS app or build Aglaïa from source.
---

## Download (macOS, Apple Silicon)

Grab the latest signed, notarized DMG from the
[GitHub Releases page](https://github.com/yb85/aglaia/releases/latest),
open it, and drag **Aglaïa** to Applications.

:::note[macOS only]
Aglaïa depends on Apple Vision (page + OCR) and Speech (voice control),
and on Apple Silicon for the MLX-accelerated page dewarper. There is no
Windows or Linux build of the capture app.
:::

Each release ships a `SHA256SUMS.txt`; verify with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

## Install via Homebrew

The tap carries a Cask (the GUI app) and a source formula (the CLI):

```bash
brew tap yb85/aglaia https://github.com/yb85/aglaia
brew trust yb85/aglaia              # Homebrew 6.x: trust the third-party tap

brew install --cask aglaia          # the GUI app (notarized DMG) — recommended
brew install aglaia-cli             # same full app, launched from a terminal
brew install aglaia-cli --without-gui   # lighter: CLI-only, no Qt/GUI
```

`aglaia-cli` builds from source with `uv`; "cli" means *run from a
terminal* (`aglaia ~/scans/book`), not GUI-less — it's the same app as the
Cask. Add `--without-gui` for the lean headless-only build (no PySide6).

## Install via pip

Aglaïa is a pip-installable package exposing an `aglaia` command built from
subcommands: `aglaia` (or `aglaia ~/book.agl`) opens the GUI, while
`aglaia run PATHS…` batches headlessly. The base install is lean and
GUI-free:

```bash
pip install aglaia                  # lean base: headless batch pipeline, no Qt
pip install "aglaia[gui,macos]"     # macOS capture GUI: Vision, Speech, MLX dewarp
pip install "aglaia[server]"        # the HTTP job API — `aglaia server`
aglaia run ~/scans/*.jpg --ocr auto --export pdf:g4+md   # headless batch
```

See the [CLI reference](/docs/reference/cli) for every subcommand and flag,
and the [Server](/docs/concepts/server) page for the job API.

OCR engines: **`pip` can't install both** — `surya-ocr` pins
`huggingface-hub<1`, `mlx-vlm` (paddle) needs `>=1.5`, so a loose pip
resolve is unsatisfiable. Pick one. This limit is **pip-only**: `uv` (and
the shipped `.app`) reconcile both via a resolver override.

```bash
pip install "aglaia[surya]"         # Surya OCR (cross-platform)
pip install "aglaia[paddle]"        # PaddleOCR-VL (MLX)
```

Apple Vision OCR needs no extra; it ships with `[macos]`.

## Build from source

```bash
git clone https://github.com/yb85/aglaia
cd aglaia
uv sync --extra gui --extra macos
uv run aglaia ~/scans/my-book
```

To build the `.app` bundle yourself, see
[Distribution](/docs/reference/distribution).
