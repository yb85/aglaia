# OCR engine benchmark — Greek + French corpus

A comparison of every OCR engine Aglaïa can drive, across input DPI, on a
hard bilingual corpus. Goal: **find the lowest DPI that still reads accurately**,
per engine.

## Corpus

`athanase-ocr-test.agl` — 24 pages (a 12-scan two-page-spread extract, morales
scans 62–73). Content stress-tests OCR: **French prose + ancient polytonic
Greek**, titles, footnotes, block quotes, and reference apparatus. Each page is
a processed chosen-layout image, native **300 dpi** (~1700×2408).

## Method

Per engine × DPI ∈ {100, 150, 200, 300}:

```bash
AGLAIA_OCR_DPI=<dpi> AGLAIA_OCR_COMPLEMENT=<none|surya|glm> \
  uv run python -m aglaia ocr <copy>.agl --ocr <engine> \
  [--ocr-lang fr-FR+el-GR] --export pdf:g4+md
```

- **Timing** is split into **model-load** (one-off, VLM server spin-up) and
  **page processing** (steady-state), reported separately by the CLI.
- **Mistral** is DPI-independent — it uploads the native G4 PDF in **one**
  whole-document request — so it runs once and serves as the accuracy grounding
  (`mistral@300`).
- **Accuracy** is an order-insensitive **word-overlap (Dice on word multisets)**
  of the plain-text-normalised Markdown:
  1. vs each engine's **own native-300** output → *DPI degradation*;
  2. vs **mistral@300** → *coarse cross-engine accuracy* (Mistral is a reference,
     **not** ground truth — it misses some punctuation / diacritics).

## Timing — page processing (s/page)

| engine | 100 | 150 | 200 | 300 | notes |
|---|---|---|---|---|---|
| apple_vision | 0.48 | 0.60 | 0.62 | 0.70 | fastest; ~0 load |
| apple_docs | 0.88 | 0.96 | 1.01 | 1.00 | Vision + structured doc |
| **mistral** | — | — | — | **6.64** | native only; **1 request**, 159 s total, ~0 load, ~$0.02 |
| apple_docs + glm | 12.3 | 5.3 | 5.6 | 6.4 | complement re-OCRs Greek blocks |
| apple_docs + surya | 11.0 | 15.0 | 16.4 | 12.6 | |
| glm | 15.3 | 14.8 | 21.0 | 46.1 | input-size sensitive |
| surya | 43.0 | 39.0 | 40.7 | 46.2 | slowest; output-token-bound (DPI-flat) |
| **unlimited** | — | — | — | **10.2** | **DPI-independent** (whole-doc, raw blobs — like Mistral); **per-page** (window=1); ~2.6 s load |

VLM model load ≈ 2–3 s (surya/glm/unlimited); 0 for Apple, Mistral.

> **unlimited is a whole-document engine** (`recognize_rows`), so — like Mistral
> — it OCRs the raw stored page blobs and is **DPI-independent**: `AGLAIA_OCR_DPI`
> never resamples its input, and 100/150/200/300 produce byte-identical output.
> It runs **per-page by default** (`window=1`). Its fused **multipage R-SWA** path
> (`window>1`, `AGLAIA_UNLIMITED_WINDOW`) is numerically **unstable in this
> mlx-vlm build** — fusing 2+ pages triggers erratic repetition loops (a page
> balloons to ~30k words; higher `repetition_penalty` makes it *worse*), so it's
> opt-in until fixed. At `window=1`: **10.2 s/page, 0.899 vs Mistral** — fast and
> accurate, in the surya/glm band but ~4× faster than surya.

## Accuracy — word-overlap vs mistral@300

| engine | 100 | 150 | 200 | 300 |
|---|---|---|---|---|
| apple_vision | 0.876 | 0.881 | 0.871 | 0.873 |
| apple_docs | 0.878 | 0.883 | 0.872 | 0.875 |
| surya | 0.838 | 0.910 | 0.925 | **0.928** |
| glm | 0.812 | 0.891 | 0.902 | 0.909 |
| apple_docs + surya | 0.887 | 0.875 | 0.873 | 0.871 |
| apple_docs + glm | 0.752 | 0.863 | 0.858 | 0.868 |
| **unlimited** (native, window=1) | 0.899 | 0.899 | 0.899 | 0.899 |

*unlimited is DPI-independent (one native value repeated); 0.899 puts it in the
top band with surya/glm — clean per-page transcription, no Cyrillic hallucination.*

*Even Apple agrees only ~0.87 with Mistral (same content, different
formatting/footnote handling) — so ~0.87 is the "as good as Apple" band, ~0.90+
is closer to Mistral.*

## DPI degradation — word-overlap vs the engine's own native-300

| engine | 100 | 150 | 200 |
|---|---|---|---|
| apple_vision | 0.921 | 0.930 | 0.933 |
| apple_docs | 0.921 | 0.927 | 0.930 |
| surya | 0.864 | 0.948 | 0.967 |
| glm | 0.821 | 0.922 | 0.941 |
| apple_docs + glm | 0.778 | 0.905 | 0.913 |

## Recommendation — a single global default: **200 dpi**

For now Aglaïa uses **one global OCR DPI of 200** — the value already used in
the GUI, and now the CLI default too (override with `--ocr-dpi`). Why a single
default rather than per-engine tuning:

- **200 is the accuracy ceiling** — 300 buys essentially nothing (surya
  0.925→0.928, glm 0.902→0.909) at ~2× the cost, so there's no reason to go
  higher.
- **200 is safe for the local VLMs**, which clearly degrade below 150 and are
  at/near their peak by 200.
- **Apple loses nothing meaningful at 200** — 100→200 is only +0.14 s/page
  (0.48→0.62), so a lower Apple-only default isn't worth the CLI/GUI split.

The per-engine "knee" (below, from the data) is where you *could* go lower to
save time on a specific engine, but the small Apple delta and VLM floor make a
uniform 200 the pragmatic choice:

| engine class | knee (lowest accurate) | note |
|---|---|---|
| Apple (vision, docs) | 100 | accuracy flat 100→300 (but only 0.14 s/page saved) |
| Local VLMs (surya, glm) | 150 | 100 clearly degrades |
| apple_docs + complement | 150 | 100 collapses (0.78) |

Cross-engine finding: **Mistral is both the accuracy reference and faster than
every local VLM** (6.6 s/page vs surya's 40+, glm's 15–46) — only Apple beats it
on speed. Trade-off: cloud/paid vs the local engines' free/offline.

## Engine notes

- **apple_vision / apple_docs** — fast, DPI-robust, solid on both scripts. Best
  default when macOS Vision is available.
- **surya** — tracks Mistral best (0.928) but slowest; output-token-bound, so DPI
  barely changes its time.
- **glm** — good from 150 up; time grows sharply with DPI.
- **mistral** — best accuracy/speed of the non-Apple engines; one billed request
  per document.
- **unlimited** — now working via our in-process MLX port (../unlimited-ocr-mlx).
  Whole-doc, **DPI-independent**, **per-page** (window=1): **10.2 s/page, 0.899
  vs Mistral** — top-band accuracy at ~4× surya's speed, no Cyrillic
  hallucination. The fused multipage R-SWA path is unstable (repetition loops)
  and opt-in. A strong local option once the q4 model is published.

## Caveats

- Mistral is a reference, not ground truth — scores are *relative* fidelity;
  final judgement on Greek/French still wants a human spot-read.
- One book's typography and one page style; other corpora may shift the knees.
- Repetition-loop degeneration on low-context crops (Surya/GLM complement) is
  suppressed by `repetition_penalty=1.15` on the served VLMs.
