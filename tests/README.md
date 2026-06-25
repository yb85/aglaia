# Tests

Two layers:

1. **Unit / integration tests** (this directory) — fast, hermetic, run on
   every change. `uv run pytest`. They build synthetic projects via the
   storage repos (`tests/conftest.py` fixtures `db` / `seeded_db`) and assert
   pipeline, storage, export, and OCR behaviour without the GUI or a camera.
   `testpaths = ["tests"]`, so only this directory is collected.

2. **Regression bench** (`aglaia-dev/tests/bench/`) — slow, fixture-driven,
   run on demand / per architecture. It drives the **real** CLI and GUI over
   a slim 175-page project and profiles timing + memory + OCR, gated against
   a per-arch baseline. The fixture and harness live in the **private**
   `aglaia-dev` repo (the 179 MB `.agl` can't ship in the source tree); they
   are gitignored here.

## Regression-bench method (for a new agent)

The goal is to catch performance / memory / output regressions across
machines, not just "does it crash". The method, and how to drive it, is
documented in **[`aglaia-dev/tests/bench/README.md`](../aglaia-dev/tests/bench/README.md)**.
Start there. In one breath:

```bash
python aglaia-dev/tests/bench/bench.py all                 # profile + diff vs baseline
python aglaia-dev/tests/bench/bench.py all --update-baseline   # adopt as this arch's baseline
uv run pytest aglaia-dev/tests/bench                       # fast gate (open/close + smokes)
```

It is **almost pure orchestration** — the instrumentation is already in the
app. Key facts a new agent needs:

- **Pipeline timing** comes from `nodes.elapsed_ms` in the project DB (the
  chain writes it per step). Run `--force-proc` to re-persist every step,
  then query, grouped by `processor_name`, scoped to non-deleted scans.
- **Memory** comes from the chain's own `[RSS-poll]` stdout lines (vmmap
  phys-footprint — accurate on Apple Silicon, unlike psutil RSS). In headless
  mode they reach stdout via the log drain; in GUI mode the log_queue feeds
  Qt, so the sampler additionally prints them under `AGLAIA_TEST`.
- **OCR timing** comes from the per-page `[ocr.<engine>]` op-log line in
  `_run_ocr`; **md accuracy** is a `difflib` ratio vs a committed reference.
- **Output pixel-diff** (pages 11 & 20) compares the reprocessed page image
  to the golden stored in the `.agl`, via a margin-robust perceptual metric
  (Otsu autocrop → NCC-max alignment → SSIM); see `bench/imgdiff.py`.
- Runs come in three **sizes** — small (10 pages) / medium (50) / full (175);
  baselines + refs are keyed per arch *and* size.
- The fixture is **copied to a tempdir** for every phase — never mutated.

### Test-only app hooks (shipped, env-gated, cheap)

These exist in the `aglaia` package so the harness can drive the app
headlessly. They are no-ops unless the env var is set:

| Hook | Where | Purpose |
|---|---|---|
| `AGLAIA_TEST=1` GUI auto-quit | `aglaia/app.py:_maybe_test_autoquit` | quit cleanly once the pipeline settles / reprocess drains — enables GUI bench + open/close without a human |
| `AGLAIA_TEST` → `[RSS-poll]` stdout | `IntegratedProcessingChain._rss_sampler_loop` | expose per-process memory in GUI mode (where the log_queue goes to Qt, not stdout) |
| `[ocr.<engine>]` per-page op-log | `aglaia/workers/headless.py:_run_ocr` | OCR timing (always on) |

Tunables: `AGLAIA_TEST_SETTLE_S`, `AGLAIA_TEST_QUIET_S`, `AGLAIA_TEST_MAX_S`.
Offscreen GUI runs add `QT_QPA_PLATFORM=offscreen` + `AGLAIA_FAKE_CAMERA=1`.
