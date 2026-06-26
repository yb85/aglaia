# Changelog

All notable changes to Aglaïa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0a6]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a6
[0.1.0a5]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a5
[0.1.0a1]: https://github.com/yb85/aglaia/releases/tag/v0.1.0a1
