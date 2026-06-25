# Changelog

All notable changes to Aglaïa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0a1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a1
