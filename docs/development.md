# Development

## Environment

- Python 3.12 (`.python-version`). `<3.13` constraint in `pyproject.toml`.
- `uv` for dependency management. Lockfile: `uv.lock`.

```bash
uv sync                                  # install all deps into .venv
uv run python -c "import cv2, jax, doxapy, PySide6, pikepdf"   # sanity check
```

## Runtime requirements

- **macOS** at runtime for any flow that uses `PageDetector` (Apple Vision) or `VoiceWorker` (Apple Speech). The default pipeline includes `PageDetector`, so cross-platform runs need a pipeline that omits it.
- **Dewarp backend**: MLX on arm64 macOS, JAX (CPU) elsewhere — `.jax_cache/` is created on first JAX dewarp run (gitignored). CUDA wheels via `--extra cuda`.
- **`model/` / `models/`** dirs hold downloaded ML weights (Surya / EAST / DBNet). Gitignored; managed via the `models_dir` config key (Settings → Models), default `<APP_DATA>/models`.

## Running

Quick smoke test (single SkewFinder step, synthetic image):

```bash
uv run python tests/test_processing_chain.py
```

Capture GUI (needs a camera at `--camera-id 0` by default):

```bash
uv run aglaia gui /tmp/test_scans
```

Headless batch (PDF import lives in the GUI import panel; CLI batch
reprocesses an existing project):

```bash
uv run aglaia run /tmp/test_scans.agl -p config/pipelines/book_curved_x2.yaml
```

## Module map (quick reference)

| Concern | File |
|---|---|
| Buffer envelope + write | `aglaia/ImageBuffer.py` |
| Processor interface | `aglaia/processors/abstraction.py` |
| Processors | `aglaia/processors/*.py` |
| Chain abstractions | `aglaia/workers/chain_abstraction.py` |
| Active chain | `aglaia/workers/IntegratedProcessingChain.py` |
| Initializer (args, factory, template eval) | `aglaia/workers/Initializer.py` |
| Camera calibration | `aglaia/workers/Calibrator.py` |
| PDF I/O | `aglaia/workers/PDFprocessor.py` |
| GUI process bridge | `aglaia/workers/ProcessMonitor.py` |
| GUI main window | `aglaia/gui/MainWindow.py` |
| GUI per-scan widget | `aglaia/gui/ScanItemWidget.py` |
| Webcam thread | `aglaia/gui/WebcamThread.py` |
| Voice recognition | `aglaia/gui/VoiceWorker.py` |

## Adding a processor

See `docs/processors.md` — "Writing a new processor". One step:

1. Drop a new file in `aglaia/processors/` with an `AbstractImageProcessor`
   subclass declaring `OPTIONS` (and an optional `OPTION_CLASS`). The
   registry (`aglaia/processors/registry.py`) auto-discovers it — there is no
   `OPTION_MAP` / `PROCESSOR_REGISTRY` to edit.

## Multiprocessing constraints

`set_start_method("spawn", force=True)` is enforced by both entry scripts. Consequences:

- Everything queued must be picklable. `ImageBuffer` is (numpy + dict). Pipeline option dataclasses must be too — that's why YAML lambdas are wrapped in `TemplateEvaluator` (picklable) instead of lambdas.
- Imports happen in each worker on spawn. Avoid expensive top-level imports in any module that the processor registry imports. pyobjc / MLX only load when actually used.
- OpenCV uses its defaults; if many workers contend, consider `cv2.setNumThreads(1)` explicitly.

## Logging / debugging

- Workers log via `log_queue` (no stdout in workers — use queue tuples).
- Main process drains via `ProcessMonitor` (GUI) or an inline loop (headless CLI). Rich `Console` prints everything.
- `('timing', stem, dims, dpi, proc_name, ms, success)` events come out per step — useful for spotting slow stages. Look for `log_warning` lines.
- `--debug` flag forwards to processors that opt in (currently `PageDewarper`).

## Testing

`pytest` is configured (`[tool.pytest.ini_options]` in `pyproject.toml`, `testpaths = ["tests"]`); run the suite with `uv run pytest`. Tests live under `tests/` (processors, storage, GUI, plugins, …).

`tests/test_processing_chain.py` is the integration smoke test: it builds a 1-element Integrated chain with `SkewFinder`, feeds a synthetic skewed line, and waits for the corresponding `image_event`.

## A/B a pipeline change (measure the output, not the objective)

`scripts/ab_pipeline.py` runs the real CLI once per variant over the committed
fixture corpus and prints a comparison. Every dewarp decision on record was
made with it — retiring the twist/spline sheet models, adopting the
principal-point DOF, sizing the spine-curl grid.

```bash
uv run python scripts/ab_pipeline.py \
    --variant "shipped:" \
    --variant "flat-camera:PageDewarper.principal_point=false" \
    --variant "no-gamma:PageDewarper.spine_gammas="
```

Each `--variant` is `label:Processor.option=value[,…]`; an empty override list
runs the pipeline as committed. Output is mean / median / worst deviation of
the output text baselines from a straight line, ms/page, and a per-page column.

**Use it whenever a change claims to improve output quality.** A green test
suite says a change didn't break anything; it says nothing about whether the
change is *better*. Four habits, each of which caught a real bug here:

1. **Score the output, not the objective.** A fit is judged by its objective,
   and the objective cannot see a bad page — a spine-curl γ at the grid edge
   beat γ = 0 on objective while producing a remap that blew the out-of-bounds
   gate. Only the rendered page tells you.
2. **Distrust a suspicious tie.** Two variants scoring *identically* is
   evidence of a bug, not of equivalence. That is how the `page_side` drop
   surfaced: `TrapezoidalCorrection` returns a fresh `ImageBuffer` and was
   dropping the meta one step before `PageDewarper` read it, so every
   binding-side feature had been silently inert — including `flat_spline`'s
   own defining features. The A/B before that fix was measuring two
   handicapped things and would have justified the wrong conclusion.
3. **Check pages in == pages out.** The harness compares `PageDetector`
   children against `PageDewarper` nodes and exits 2 on a mismatch. That
   column is how #64 surfaced — pages vanishing from slow runs with no node,
   no error and exit code 0. A quality metric averaged over the pages that
   *survived* would have looked fine.
4. **Re-measure after a refactor that should change nothing.** The
   twist/spline retirement was verified by re-running the whole corpus and
   getting the same aggregate and per-page numbers, which is a much stronger
   statement than "the tests still pass". Expect *equal*, not *bit-equal* —
   see the noise floor below.

Numbers quoted in commits and `docs/processors.md` come from this harness, so
they are reproducible — and re-runnable when you doubt them.

**How much delta to believe.** Runs are not bit-deterministic: the warm-start
ring seeds each page's curl from recent same-side fits, so the order pages
happen to complete (worker scheduling) can shift a page slightly. Observed
drift on repeat runs of an identical configuration is ~0.002 px. Treat
sub-1% differences as noise and re-run before believing one; the gaps that
decided anything here (0.92 vs 1.65 px) are far outside it.

## Headless UI debugging (no display / camera)

Three layers let you inspect + drive the app without a screen, camera, or the
startup dialog — useful for CI smoke shots and agent-driven UI debugging:

1. **Logic / data corners — headless CLI.** `aglaia run <img|pdf|.agl>
   -p <pipeline> --ocr <engine> --export pdf+md --md-refine apple_fm` creates
   a project, imports, processes, OCRs and exports without Qt. Inspect the
   resulting `.agl` DB or output files. Covers import / process / pipeline swap /
   OCR engines / export.
2. **Widget visual corners — `debug/ui_shot.py <scene>`.** Renders one seeded
   widget (CaptureTab, ExportTab, the capture-mode status bar, …) to a PNG
   offscreen. `QT_QPA_PLATFORM=offscreen uv run python debug/ui_shot.py list`.
   Add a scene function for whatever widget you're debugging.
3. **Full populated GUI — `AGLAIA_UI_SHOT_DIR`.** `aglaia <project.agl>` opens
   in project mode with no startup dialog, so
   `QT_QPA_PLATFORM=offscreen AGLAIA_UI_SHOT_DIR=/tmp/s uv run aglaia gui
   project.agl` screenshots the whole window + each sidebar tab into `/tmp/s`
   then quits (`AGLAIA_UI_SHOT_TABS=pipeline,export` limits which). Faithful view
   of the real app loaded with real data.

4. **Scripted GUI flows — scenarios.** `AGLAIA_UI_SCENARIO=debug/scenarios/<x>.py`
   runs a scenario against the live `MainWindow` (it gets a `Driver` with
   `shot()`, `tab()`, `pump()`, `wait_until()`), then quits. A **fake camera**
   (`AGLAIA_FAKE_CAMERA=/path/img.jpg`) feeds a still image as the live camera,
   so even the capture → pipeline flow runs with no hardware. Run all at once:

       uv run python debug/run_scenarios.py <project.agl>   # → /tmp/ui_scenarios/<name>/

   Shipped scenarios: `tour` (every tab), `corners` (Settings, Model Downloader,
   Pipeline editor), `capture_flow` (fake camera → activate → capture → process).
   Add `debug/scenarios/<name>.py` with `def run(d): …`.

**GUI vs headless split.** Use scenarios for *visual / interaction* corners
(tabs, dialogs that `show()`, capture preview, editor render). Use the headless
CLI for corners that hit a **blocking save-dialog** or need a **verifiable
result** — export (`--export pdf+md`), OCR (`--ocr <engine>`), and changing a
**pipeline element value** (edit the YAML, run headless, check the DB/output).
The GUI pipeline editor is a deep custom-widget tree; drive value *changes* via
YAML+headless, screenshot the editor only to check it renders.

Hard limits (need real hardware): live Continuity-Camera pixels, actual Apple
Vision / Speech recognition output, true macOS compositing.

## Duck-typing helpers

- `ImageBuffer.binarize() / detectLayouts() / dewarp()` — duck-typing
  convenience methods. Prefer the processor classes; note `binarize()` is
  used by `inspect_binarization.py`, so don't delete it blindly.

## Conventions

- All numeric pixel/length quantities are in **DPI-aware millimeters** where possible (`margin_mm`, `square_size_mm`). Convert with `px = int((mm / 25.4) * dpi)`.
- Results are persisted to the SQLite-backed project (`aglaia/storage/`), not to filesystem step folders. The `NN_` step prefix (used for timing rows / pipeline-preview labels) is generated automatically — don't put it in the YAML `name:` field.
- Buffers crossing process boundaries must be `.copy()`'d (or about-to-be-finalized).
- New CLI flags should default from `DEFAULT_ARGS` and be overridable in YAML `args:` blocks.

## Useful one-liners

```bash
# Print effective pipeline definition (with templates resolved literally)
uv run python -c "from aglaia.workers.Initializer import load_pipeline_def; from pprint import pp; pp(load_pipeline_def('config/pipelines/book_curved_x2.yaml'))"

# Dump current calibration
uv run python -c "from aglaia.workers.Calibrator import load_calibration; from pprint import pp; pp(load_calibration())"

# List doxapy binarization algorithms
uv run python -c "import doxapy; print([a for a in dir(doxapy.Binarization.Algorithms) if not a.startswith('_')])"
```
