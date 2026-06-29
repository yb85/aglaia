---
title: OCR engines
description: The text-recognition backends Aglaïa can drive, and when to pick each.
---

Aglaïa runs OCR through a pluggable **engine** abstraction: every backend
implements the same `recognize(image, languages)` contract, so you can
switch engines per document without changing anything else. Five ship in
the box, and you can [drop in your own](/docs/reference/processors).

## Background

No single OCR engine wins everywhere. Apple Vision is fast and on-device
but weaker on non-Latin scripts; vision-language models (Surya,
PaddleOCR-VL) are far more accurate on hard pages but slower and
heavier; a cloud service reads anything but sends your page over the
wire. Aglaïa exposes all of them behind one interface and lets you
choose, even mixing a fast primary with a slow *complement* that only
re-reads the low-confidence lines.

## Use cases

- **Clean Latin text** → Apple Vision (or Apple Document), near-instant.
- **Mixed or non-Latin scripts** → a VLM engine for accuracy.
- **Tricky historical type** → Mistral Document AI in the cloud.
- **Bulk, offline** → keep everything on-device; nothing leaves the Mac.

## Comparison of OCR engines

| Engine | Where | Speed | Accuracy | Notes |
|---|---|---|---|---|
| **Apple Document** | on-device | fast | gold (mixed script) | recovers page (headings, blocks, reading order) — the choice for **Markdown** |
| **Apple Vision** | on-device | fast | good | line-based, Latin-first, **no page** — for the searchable-**PDF** text layer, not Markdown; **default** |
| **Surya** | on-device (llama.cpp) | slow | gold | VLM via bundled `llama-server` |
| **PaddleOCR-VL** | on-device | slow | high | VLM alternative |
| **Mistral Document AI** | cloud | network-bound | gold | reads any script; key in the OS keychain |

A unified **OCR DPI** knob downsamples the page to a sweet spot
(≈150 dpi) before inference, regardless of engine.

## Choosing an engine

In the GUI, pick the engine in the OCR tab. From the headless CLI, pass
`--ocr` to `aglaia run` — `--ocr auto` for the default (Apple Vision →
Surya), or a named engine with options, e.g. `--ocr apple:lang=fr-FR` or
`--ocr surya`. Add `--ocr-lang fr-FR+en-US` to set languages. List what's
available with `aglaia list ocr`.

## Related resources

- [Processors](/docs/reference/processors) — add a drop-in OCR engine plugin
- [Markdown export](/docs/reference/markdown_export) — structured output from OCR
- [Configuration](/docs/reference/configuration) — the OCR DPI and confidence-gate keys
