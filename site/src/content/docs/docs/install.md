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

## Install the CLI (pip / Homebrew)

Aglaïa is also a pip-installable package that exposes an `aglaia` command.
The base install is lean and GUI-free (the headless pipeline, no Qt):

```bash
pip install aglaia                  # `aglaia --headless …` batch pipeline
pip install "aglaia[gui,macos]"     # macOS capture GUI: Vision, Speech, MLX dewarp
aglaia ~/scans/my-book              # launch the GUI
```

OCR engines are heavy and **mutually exclusive** (their `huggingface-hub`
pins conflict) — pick at most one:

```bash
pip install "aglaia[surya]"         # Surya OCR (cross-platform)
pip install "aglaia[paddle]"        # PaddleOCR-VL (MLX)
```

Apple Vision OCR needs no extra; it ships with `[macos]`. Or via Homebrew
(builds from source with `uv`):

```bash
brew tap yb85/aglaia https://github.com/yb85/aglaia
brew install aglaia
```

## Build from source

```bash
git clone https://github.com/yb85/aglaia
cd aglaia
uv sync --extra gui --extra macos
uv run aglaia ~/scans/my-book
```

To build the `.app` bundle yourself, see
[Distribution](/docs/reference/distribution).
