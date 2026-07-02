# Changelog

All notable changes to Aglaïa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0rc2] — 2026-07-02

Second release candidate. A large body of work landed since rc1: a subcommand
CLI, a phone-handoff bridge, a warm-pool job server, and a full OCR-engine
overhaul (local VLMs + cloud Mistral post-processing), plus many capture/GUI
stability fixes.

### Added

- **Subcommand CLI (Typer).** `aglaia [gui] [PROJECT]` (default), `run`, `ocr`,
  `server`, `setup`, `list`, `version`. `aglaia ocr` OCRs PDFs/images (or
  re-OCRs a `.agl`) with **no** processing chain — for already-clean docs.
- **Receive from phone.** A TLS receiver (QR-pinned, token-gated `/import`)
  plus an `.aglbundle` reader for the iOS handoff — capture on the phone, finish
  on the desktop.
- **Job server** (`server` extra). Warm-pool HTTP job API: run/list/check/get/
  delete/admin, processing, Mistral-batch backoff, downloads, email + admin.
- **Local VLM OCR.** An OCR-agnostic local VLM server layer with two engines —
  GLM-OCR and Baidu Unlimited-OCR (in-process MLX on macOS, vLLM on CUDA) — and
  a `DirectBlockOCR` trait so any block recogniser can complement `apple_docs`.
- **Per-engine OCR layers + export layer selection.** Keep multiple engines'
  results per page; pick which layer to export (PDF text layer + Markdown).
- **Mistral markdown post-processing.** Footnote conversion (LaTeX/Unicode
  superscripts and `(N)` → GFM `[^N]`, unique anchors that keep the original
  number) and header/footer extraction — with toggles on the Markdown export
  card, applied at export time (re-export reflects changes without re-OCR).
- **Central download registry** with resumable CLI downloads (retires
  `model-list.json`).
- **Dewarp warm-start** (curl seeded from recent same-side fits) and automatic
  discard of a degenerate trapezoid keystone.

### Changed

- **OCR post-processing is tied to markdown export, not OCR.** The raw engine
  output is stored; footnote/header-footer transforms run at export.
- **Surya** moved off its torch/GGUF stack onto the local VLM server layer.
- **Dropped the PaddleOCR-VL (`paddle_vl`) engine** — weak on Greek, heavy deps.
- Local VLM backend is **bundled by platform** (MLX / vLLM).
- Default `ocr_dpi` is **200** (matches the GUI); added `--ocr-dpi`.
- Cloud whole-document engines route through **one request** (fixes Mistral
  per-page billing); markdown export scan/branch markers are now
  `<!-- scan #N -->` + `<!-- branch N.A -->`.

### Fixed

- **QThread teardown crashes** ("Destroyed while thread is still running") on
  Mistral batch-check and on **deactivating voice control** — workers are now
  retained until they actually finish.
- **DPI-calibration "Trace manually" froze the window** (runaway width) and blew
  GUI RAM into the GBs — the trace canvas now paints directly instead of
  driving the layout via `setPixmap`.
- A fresh **capture session opens on the Capture tab** instead of the last-used
  sidebar tab.
- **Served-VLM degeneration loops** (repetition penalty), a **Cyrillic
  block-splice leak**, and script-anomaly garbage detection for complements.
- OCR **progress/ETA**: over-count (334/322), jumpy ETA, `s/page` labelling, and
  an idle watchdog that snapped the bar to 100%.
- **XY-cut page splitter** + absorb-smallest merge for DBnet 2-up scans; layout
  overlay renderer; per-page step toggle dead after reprocess; CLI now passes
  the page DPI so OCR honours the configured `ocr_dpi`.

## [0.1.0rc1] — 2026-06-27

First release candidate. Linux and Windows builds confirmed working; macOS
re-verified (full test suite + end-to-end headless chain on the MLX backend).

### Fixed

- **cv2 collision broke installs non-deterministically.** `page-dewarp` pulls a
  bare `opencv-python` (GUI build) while Aglaïa pins `opencv-python-headless`;
  both write the same `cv2/` directory. With `numpy<2.1` holding headless at
  4.11 and the GUI build floating to 4.13, a reinstall could leave a
  half-written `cv2/` (`cv2 has no attribute 'imdecode'`), and the bundle picked
  up whichever payload won. Pinned `opencv-python` to 4.11.0.86 so the shared
  `cv2/` is always one consistent payload.
- **Stuck per-card spinner.** A card could stay dimmed + spinning after the chain
  went idle and the progress bar read 100% (a dropped `branch_ready`). Idle
  reconciliation now sweeps any card still marked processing, even once the bar
  has finished.
- **Lost page-visibility toggle.** A hide/show whose `(scan, label)` matched no
  branch row was silently dropped and reappeared on reload; it now logs a loud
  `[visibility]` warning so the offending label can be diagnosed.

## [0.1.0a6] — 2026-06-26

Sixth alpha. Same as a5 plus a Windows build fix (a5's Windows installer
failed its release gate, so no a5 `.exe` shipped).

### Fixed

- **Windows "database is locked" under concurrent writers.** `journal_mode=DELETE`
  recreates the rollback journal per write; on Windows the file syscalls (plus
  AV scanning) make contended writes serialize past the old 5 s `busy_timeout`.
  Bumped to 20 s and added a bounded backoff-retry on the hot insert path (keeps
  the single-file `.agl` design — no WAL sidecars). Unblocks the Windows
  installer/ZIP.

## [0.1.0a5] — 2026-06-26

Fifth alpha. GPU Linux AppImage, faster dewarp, auto worker count.

### Added

- **Prebuilt slim-CUDA Linux AppImage** (`Aglaia-x86_64-cuda.AppImage`) for
  GPU-accelerated page dewarp on NVIDIA/Linux — no source / `--extra cuda`
  install needed. The dewarp is matmul-only L-BFGS-B, so the bundle ships only
  the CUDA libs it loads (cuBLAS, nvrtc, nvjitlink, ptxas, cupti, cudart) and
  drops ~2.6 GB of dead weight (cuDNN/NCCL/nvshmem/cuFFT/cuSPARSE/cuSOLVER) —
  1.3 GiB, under GitHub's 2 GiB release-asset cap. (#15)
- **Auto pipeline-worker count.** A worker count of `0` (config, `--workers 0`,
  or the Settings slider's leftmost notch) now means *auto* — sized to the CPU,
  platform-aware (Apple Silicon → performance-core count; x86 → ~half the
  physical cores). The pipeline sidebar shows `NN workers (auto|manual)`. Auto
  is the new default.

### Changed

- **Dewarp shape buckets right-sized** to real page geometry (measured ~45
  keypoints/line, not the ~70 the old caps assumed) plus finer steps — ~30%
  faster dewarp on GPU, ~40% on CPU, no memory or quality cost.

## [0.1.0a3] — 2026-06-25

Third alpha. Linux/GPU and tiling-WM fixes, plus dewarp robustness.

### Fixed

- **Dewarp produced no output on dense pages.** The padded JAX optimiser's
  over-cap fallback re-imported a function `install()` had already replaced
  with itself → infinite recursion (`maximum recursion depth exceeded`), which
  killed the dewarp branch. This was also the "QEMU" recursion crash — never
  QEMU-specific, just any page over the cap.
- **Sidebar "Tip the developer" link** opened a dead Ko-fi handle (redirected
  to the Ko-fi homepage); now points at the correct page.
- **Loading splash** was tiled/mangled by tiling window managers
  (Hyprland/sway/i3); it now floats.

### Changed

- **Over-cap dewarp** (dense or `baseline_source=both` pages — common now that
  line extraction is fixed) no longer falls back to the slow, cubic-only stock
  optimiser. It prunes text lines to fit (keeps the extremities, drops short
  lines in dense regions, protects sparse regions) and pads to size buckets
  (50 / 80 / 120 lines) so typical pages stay fast.

### Performance

- **Idle worker memory** is now released on Linux (`gc` + `malloc_trim`). An
  opt-in aggressive recycle (`AGLAIA_WORKER_IDLE_RECYCLE_S`) frees the
  JAX/CUDA resident stack (~1.6 GB → ~0.75 GB per worker) when idle.

## [0.1.0a2] — unreleased

Second alpha. Bug-fix pass over a1 from macOS release testing.

### Fixed

- **DPI estimation** (card + measure-a-distance) is applied as a per-session
  value again; it was silently lost in the frozen app (`camera_params.json`
  wrote to a read-only relative path). `camera_params.json` now stores only the
  camera matrix, under APP_DATA. Full chessboard calibration disabled for now (#16).
- **Quit crash** (SIGABRT) when closing mid-model-download — worker threads are
  now stopped on close.
- **Version** shown in About / Diagnostics / Bug report (was `0.0.0` / hardcoded
  `0.1.0`); added `aglaia --version`.
- **`roi_margin_mm`** now takes effect at any value (crop follows the extended
  ROI) — fixes DBnet clipping page margins.
- **Combo dropdowns** were see-through (transparent popup background).

### Performance

- A large project reprocess no longer balloons GUI memory (~3.9 GB → ~0.5 GB)
  or freezes the UI: stage thumbnails are deferred (spinner until the branch
  finishes, then render the final), and status-bar log updates are coalesced.
  Off-screen pixmap release for very large projects tracked in #17.

## [0.1.0a1] — 2026-06-24

First public **alpha**. Well tested on macOS; Linux and Windows are unverified.

### Added

- **End-to-end scanning pipeline.** Webcam capture or image/PDF import →
  deskew → ML page detection → per-page deskew → illumination-tolerant
  binarization → keystone + page-curvature (cubic-sheet) dewarp → a final
  *replay* pass that composes the geometric and morphological operators to
  avoid successive interpolation artifacts (especially on bilevel output).
- **Two entry points, one chain.** `aglaia <workspace>` (PySide6 capture GUI
  with voice control) and `aglaia <project.agl> --headless` (CLI batch) share
  the same multiprocess `IntegratedProcessingChain` and YAML pipeline.
- **Page detection backends.** `auto` resolves **DBnet → Apple Vision (macOS)
  → EAST**; DBnet (~5 MB ONNX) is the cross-platform default. Raises a clear
  error when no model is installed (no silent heuristic fallback).
- **OCR engines.** Apple Vision (macOS), Surya, PaddleOCR-VL (MLX), and Mistral
  Document AI (cloud, with a cheaper async batch mode). BCP-47 language
  selection; optional Markdown refinement.
- **Exports.** Searchable PDF (G4 / JBIG2 profiles) and structured Markdown,
  combinable in one run (`--export pdf:g4+md`).
- **First-run setup.** GUI onboarding wizard, and `aglaia --setup` — a Qt-free
  interactive TUI for CLI-only installs that picks/downloads models, seeds the
  default pipelines, and bootstraps the config. Headless runs refuse to start
  until configured.
- **Offline model downloader.** In-app (GUI) and via `--setup`; models are
  fetched on demand and live in the per-user app-data directory.
- **Voice control.** Vosk offline, constrained-grammar, cross-platform.
- **Extensibility.** Drop-in processors and OCR engines (auto-discovered), plus
  user plugins from the app-data folder gated by a startup trust prompt.
- **Cross-platform distribution.** Signed + notarized macOS DMG (Apple
  Silicon), Windows installer + portable ZIP, Linux AppImage, and
  `pip install aglaia` on any platform. Release artifacts use fixed "latest"
  names and ship `SHA256SUMS`.
- **Localization.** English and French UI (Qt translation catalogues).

### Known limitations

- The Windows build is **not code-signed**; SmartScreen warns on first run
  (bypassable via *More info → Run anyway*; verify with `SHA256SUMS-windows.txt`).
- Apple Vision can miss very faint, wide-spaced running heads — use DBnet or
  EAST for such pages.
- JAX Metal is disabled; the page dewarp runs on CPU (or CUDA/MLX where built).

[0.1.0rc2]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc2
[0.1.0rc1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0rc1
[0.1.0a6]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a6
[0.1.0a5]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a5
[0.1.0a1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a1
