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

## Build from source

```bash
git clone https://github.com/yb85/aglaia
cd aglaia
uv sync --extra gui --extra macos
uv run python aglaia.py ~/scans/my-book
```

To build the `.app` bundle yourself, see
[Distribution](/docs/reference/distribution).
