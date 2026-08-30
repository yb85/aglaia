# Changelog

All notable changes to Aglaïa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LM dewarp solver** (`aglaia/processors/lm_solver.py`, #59). Levenberg-
  Marquardt with analytic Jacobians and the problem's bipartite sparsity —
  per-span *arrowhead* blocks eliminated in O(nᵢ), Schur complement down to one
  small solve per iteration. Now the default backend for the `cylindrical`
  sheet model (`backend: auto`), replacing ~10⁵ Powell objective evaluations
  with ~10-60 iterations: 28 s → 0.25 s per page on the test fixture, at a
  *better* final objective, on CPU with no GPU allocator to babysit. Ported
  from the iOS implementation; `AGLAIA_DEWARP_LM=0` keeps the old GPU chain.
- **Dewarp camera upgrades** (LM + `cylindrical` only, #60), from the iOS
  port's on-device research:
  - `principal_point` (default on) fits (cx, cy) alongside the pose. A
    homography composed with a perspective view of a curled page is a general
    projective camera the centred pinhole cannot represent — this is why the
    fit under-curled near the spine after `TrapezoidalCorrection`. Measured
    output line curvature 1.77 → 0.98 px on the fixture page.
  - `spine_weight_boost` / `spine_weight_zone` up-weight the binding-side
    residuals, which are otherwise a least-squares minority (2-4 px systematic
    under-curl at 300 dpi).
  - `spine_gammas` grid-searches a spine-localized directrix term
    `z += γ·exp(−|x−x_binding|/s)`, keeping the best objective (γ = 0 always
    competes). Only identifiable with the principal point; also supplies the
    surface slope the arc-length grid needs at the fold, fixing horizontal
    compression there.

  Replay params gained `camera_np` and `spine`; stamps written before this
  release default to the 8-param pinhole with no spine term, i.e. exactly the
  surface they were fitted on.

  Measured over `test_data/test_athanase` through `book_curved_x2`, mean
  output-baseline curvature: Powell 2.04 px @ 75.7 s/page → LM + the new
  camera 0.90 px @ 0.78 s/page. LM with the *old* 8-param camera is not
  uniformly better than Powell (2.96 px — it can park a shape DOF on the ±0.5
  curl clamp), which is why `principal_point` defaults on and a rejected LM
  fit now retries on Powell instead of falling straight through to grayscale.

### Added

- **Slope-based x decompression** (`slope_emphasis`, default 0 = unchanged).
  The arc-length grid corrects the *surface-length* term; it does not correct
  **projective foreshortening**, where the page recedes steeply near the spine
  and the camera compresses those glyphs however the surface is
  parameterised. `slope_emphasis` (k) weights the grid measure by
  `(1+z′²)^(k/2)` so steep regions claim more output width, and is carried in
  the replay stamp so a replayed page is sized like the live one.

  The effect scales with z′², so it is negligible on mild curl and large on
  strong: at k = 1, +0.3% output width at max|z′| = 0.1 but **+16.9%** at 1.67,
  with the steepest quarter's share of output columns going 0.318 → 0.403.

  **Ships at k = 1**, matching the iOS port where it is device-validated. That
  rests on iOS validation plus the maintainer's judgement, not on a local
  measurement: the A/B harness scores baseline straightness (vertical) and this
  changes horizontal scale, so it can only confirm no regression — which it
  does, 0.926 px vs 0.923 at the noise floor. `slope_emphasis: 0` recovers the
  exact previous geometry, and projects stamped before this option replay at 0
  regardless of the default.

- **Same-line footnotes** (`mistral_same_line`, default off). Critical editions
  pack several notes onto one physical line (`"(12) premier. (13) second."`).
  Every marker after the first never sits at a line start, so it never entered
  the entry set — and since a footnote is recognised by refs ∩ entries, it was
  never classified as a footnote **at all**: its ref stayed a bare `(13)` in
  the body and its text stayed glued to note 12. With the toggle on, markers
  inside a line that already starts with an entry marker also count, and such
  a line is split at each of them. Only markers already in that page's mapping
  cut (a citation inside a note never splits it), and the bare `N.` form never
  cuts mid-line (too ambiguous against dates and verse references).

### Fixed

- **The keystone de-rotate no longer shears a page whose deskew failed.**
  `detect_column_quad_from_baselines` re-centres near-parallel tilted margins
  onto the median baseline angle, on the premise that the page was deskewed
  upstream so level baselines are ground truth. That premise fails exactly
  when `pages_deskew` returns 0 on a fanned page (no peak in its projection
  profile): the baselines keep their tilt, and margins disagreeing with them
  are real shear, not invented rotation. Un-gated it forced the sides onto the
  baseline angle — flat text lines, sheared verticals, the quad corner ~100 px
  off the ink. Now gated on `|median baseline angle| < 1°`. Backported from
  the iOS port, where it was found on a real capture and verified against the
  desktop reference byte-for-byte.

### Added

- **A flat-fit rung in the dewarp failure ladder.** When a fit's remap
  overshoots the `max_oob` gate, the page now falls back to a zero-curl fit
  (pose and page dims around a flat sheet, ~6 ms) before being conceded to the
  grayscale passthrough. A zero-curl surface cannot produce the runaway remap
  that trips the gate, so this rung always has an answer — the page comes out
  perspective-corrected but not curl-corrected, rather than not processed.

  It also covers the one failure the rung above does not: LM parking a shape
  DOF on the ±0.5 curl clamp at a competitive objective but a wild remap (the
  case the iOS port answers with Powell, which costs 25-80 s here). Dormant on
  healthy pages — 0 firings across the 12-page corpus, output unchanged at
  0.923 px — and deliberately not recorded in the warm-start ring, since a
  curl of 0 by construction is not a measurement.

### Fixed

- **A headless run no longer abandons a page whose step outlives it** (#64).
  Completion decided the run had settled from a *silence timer*: once every
  scan had emitted one `branch_ready` and no further branch event arrived for
  8 s, it returned and the caller stopped the chain — with workers still
  mid-page. Those pages vanished: no node, no error, exit code 0. It now waits
  for the chain to report no work in flight (`is_idle`, debounced), and the
  worker holds an unconditional busy marker so that signal can't under-report
  (the resumable in-flight reference it used to rely on is only recorded for
  buffers carrying a `parent_node_id`).

  Measured on the same two spreads with the ~75 s/page `powell` backend:
  before, 2 of 4 pages lost in a 30 s "successful" run; after, 4 of 4 in 60 s.

### Removed

- **The twist/spline sheet models** (`sine_twist`, `bspline_twist`,
  `flat_spline`) and their options (`sheet_model`, `spline_modes`, `twist`,
  `knot_grading`, `flat_outer_penalty`). They were strictly more expressive on
  parameter count and strictly worse on results. A/B over the fixture corpus
  (3 books, 12 pages), mean deviation of the output baselines from straight:

  | model | mean | worst page |
  |---|---|---|
  | **cylindrical + spine curl** | **0.923 px** | **1.194** |
  | flat_spline | 1.679 px | 3.315 |
  | bspline_twist | 1.648 px | 3.053 |
  | sine_twist | 1.645 px | 3.051 |

  The cylindrical model wins on **every one of the 12 pages**. A fold
  concentrates curvature at the spine, which one localized exponential term
  captures robustly; the global spline families spread degrees of freedom
  where there is no curvature and overfit noisy spans.

  **Hard break**: a `.agl` node stamped with one of these models now raises on
  replay rather than being silently rebuilt against a different surface —
  re-process such a page from source.

  Removes ~900 lines: `sheet_models.py` 592 → 171, the spline objectives in
  the MLX and padded-JAX backends, the model plumbing in `PageDewarper` and
  the dewarp batcher, and two test modules. Output for the surviving model is
  bit-identical (verified: 0.923 px on all 12 pages, before and after).

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
