---
title: Install
description: Download Aglaïa for macOS, Windows or Linux, or build it from source.
---

## Download

Every release publishes a build per platform, under a fixed name — these
links always point at the newest one:

| Platform | Download | Notes |
|---|---|---|
| **macOS** (Apple Silicon) | [`Aglaia-macos-arm64.dmg`](https://github.com/yb85/aglaia/releases/latest/download/Aglaia-macos-arm64.dmg) | Signed and notarized. Open it, drag **Aglaïa** to Applications. |
| **Windows** (x64) | [`Aglaia-windows-x64-setup.exe`](https://github.com/yb85/aglaia/releases/latest/download/Aglaia-windows-x64-setup.exe) | Installer; it registers the `.agl` file type. Not code-signed, so SmartScreen warns on first run: **More info → Run anyway**. |
| **Linux** (x86_64) | [`Aglaia-x86_64.AppImage`](https://github.com/yb85/aglaia/releases/latest/download/Aglaia-x86_64.AppImage) | `chmod +x`, then run. Needs FUSE (`fuse2`). |

All three carry the capture GUI and the full pipeline. What is macOS-only
is what Apple supplies: Vision (page detection and on-device OCR) and the
MLX-accelerated dewarp on Apple Silicon. Elsewhere, page detection uses
DBnet or EAST, OCR uses a cross-platform engine (Surya, GLM-OCR, Mistral),
the dewarp runs on CPU, and voice control uses Vosk, which is
cross-platform.

Each release ships its checksums — `SHA256SUMS.txt` (macOS),
`SHA256SUMS-windows.txt`, `SHA256SUMS-linux.txt`:

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
pip install "aglaia[gui]"           # Windows / Linux capture GUI (Qt)
pip install "aglaia[gui,macos]"     # macOS capture GUI: Vision, Speech, MLX dewarp
pip install "aglaia[server]"        # the HTTP job API — `aglaia server`
aglaia run ~/scans/*.jpg --ocr auto --export pdf:g4+md   # headless batch
```

See the [CLI reference](/docs/reference/cli) for every subcommand and flag,
and the [Server](/docs/concepts/server) page for the job API.

OCR engines need no extras of their own any more. Surya 2 is a Qwen3.5-VL
model served through the shared local-VLM backend — `mlx-vlm` on Apple
Silicon (`[macos]`), vLLM on CUDA — exactly like GLM-OCR, so the old
`surya-ocr` / torch / `llama-server` stack and its dependency conflict are
both gone. PaddleOCR-VL was dropped in 2026-07: weak on Greek,
and it pulled in paddleocr + paddlepaddle + opencv-contrib for little gain.

```bash
pip install "aglaia[macos]"         # Apple Vision + Apple Document + MLX VLMs
```

Apple Vision and Apple Document OCR need no extra; they ship with `[macos]`.
Mistral cloud OCR needs none either: the SDK and the OS-keychain storage are
base dependencies, so the Cloud OCR card works in every build. `[cloud]`
survives as an empty alias, so older commands keep working.

## Build from source

```bash
git clone https://github.com/yb85/aglaia
cd aglaia
uv sync --extra gui --extra macos   # macOS
uv sync --extra gui                 # Windows / Linux
uv sync                             # headless: pipeline only, no Qt
uv run aglaia ~/scans/my-book
```

To build the installers yourself, see `docs/distribution.md` in the
repository.
