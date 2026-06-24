# Changelog

All notable changes to Aglaïa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0a1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a1
