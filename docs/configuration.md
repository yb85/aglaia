# Configuration

The user-facing CLI is the Typer app in `aglaia/cli/` (subcommands `gui` /
`run` / `setup` / `list` / `server` / `version`) — see the [CLI reference](cli.md).
This page documents the **internal** config layer those commands feed:
`aglaia/workers/cli.py` (`CliConfig` + the `--ocr`/`--export` spec parsers),
shared by the GUI and the headless runner. Per-invocation choices are flags;
runtime *preferences* (default engine, thumb size, theme, worker count) live in
the per-user config DB (`aglaia/app_data/db.py`). See [gui.md](gui.md) for the
app-data store.

The `DEFAULT_CONFIG` / `DEFAULT_ARGS` + `-c path.yml` YAML layer (below)
lives in `aglaia/workers/Initializer.py` and is fed a synthetic argv by the
headless path. The sections below document that internal layer for code
readers.

The pipeline definition (`-p` / `--pipeline`) is loaded **separately** by
`create_processing_chain`.

## DEFAULT_ARGS

```python
{
  "workers": 4,
  "max_pages": 2,
  "make_pdf": False,
  "camera_id": 0,
  "voice_control": False,
  "transform": "0",
  "debug": False,
  "input_dpi": None,
}
```

YAML key: `args:`. Top-level CLI flags map 1:1.

## DEFAULT_CONFIG

```python
{
  "keycontrols": { scan, trash, rotate },
  "voicecontrols": { scan, trash, quit, debounce_time },
  "processing": {},
  "page": {
    "margin_mm": 2.0,
    "rescale_threshold": 0.01,
    "binarize_threshold": 127
  },
  "dewarp": {                            # the active dewarp config lives in the pipeline step
    "max_oob": 1000, "margin": 5, "page_margin": 20, "shear_cost": 20.0,
    "remap_decimate": 16, "focal_length": 1.2,
    "debug_resize_w": 1280, "debug_resize_h": 700,
    "mask_vis_opacity": 0.4,
    "jax_metal": False
  },
  "calibration": {
    "board_cols_inner": 5,
    "board_rows_inner": 8,
    "square_size_mm": 30,
    "calnum": 10
  },
  "paths": {
    "raw_dir": "00_INPUT",
    "output_dir": "XX_OUTPUT"
  }
}
```

YAML key: `config:`. Only **top-level** keys are merged (shallow replace). If you supply `config.keycontrols`, you replace the entire dict — partial-override of a single action requires repeating all four keys.

## Args after initialize()

`args` (Namespace) keys, after the internal `initialize(mode)` runs (the
headless path synthesises the argv it parses):

| Key | Source | Notes |
|---|---|---|
| `config` | merged DEFAULT_CONFIG + YAML | `dict` |
| `pipeline` | `--pipeline` | `Path?` |
| `workers`, `max_pages`, `make_pdf`, `debug`, `input_dpi` | DEFAULT_ARGS / YAML | scalar |
| `workspace_dir` | derived project dir | `Path` |
| `camera_id`, `voice_control`, `transform` | DEFAULT_ARGS / YAML | |
| `options` | derived | nested dict — see below |

`make_pdf` is a DEFAULT_ARGS key used by the GUI page-PDF path; exports
are driven by `--export`.

## args.options structure

Populated by `initialize()`. The Integrated chain reads
`options["calibration"]` (injected into `PageDewarper` steps). The chain
persists step outputs to the project DB and does not write to
`00_INPUT`/`XX_OUTPUT` directories (see [pipeline.md](pipeline.md) /
[imagebuffer.md](imagebuffer.md)).

```python
args.options = {
    "dewarp":      { ...config["dewarp"] },
    "page":      { "workers", "max_pages", "config": config["page"] },
    "general":     { "debug", "input_dpi", "overwrite" },
    "calibration": { ...config["calibration"], "camera_matrix": None, "camera_matrix_resolution": None },
    "paths": {
        "root":   workspace_dir,
        "output": workspace_dir / config["paths"]["output_dir"],
        "raw":    workspace_dir / config["paths"]["raw_dir"],
        "debug":  workspace_dir / output_dir_name / "debug"
    }
}
```

## Calibration injection

After `initialize`, `aglaia` overwrites `args.options["calibration"]["camera_matrix"]` and `["camera_matrix_resolution"]` from `config/camera_params.json`. `create_processing_chain` then injects those into every `PageDewarper` step's options.

## Example: headless run

```bash
uv run aglaia run ~/scans/book.agl \
  -p full \
  --ocr auto --ocr-lang fr-FR+en-US \
  --export pdf:g4+md
```

Open an existing `.agl` project, run the `full` pipeline, OCR with the auto
engine (French + English), and export both a G4-compressed searchable PDF and
a Markdown file (`<project_dir>/<slug>.pdf` / `<slug>.md`).

## Where defaults are not overridable from YAML

The pipeline step options are **not** read from `default.yml`. They live inside the pipeline YAML file. CLI flags that override pipeline steps:

- `--debug` → injected into any step option with a `debug` key.
- `--max-pages` → overrides `PageDetector.max_pages` only.

Other step options must be edited directly in the pipeline YAML.

## Storage page

A project is a **single SQLite file** `<project_dir>/<slug>.agl`. There are
**no** per-step or `00_INPUT`/`XX_OUTPUT` output directories — everything
lives in the DB. See [storage.md](storage.md) for the schema.

| What | Where |
|---|---|
| Project file | `<project_dir>/<slug>.agl` |
| Raw captures / imported pages | `scans` rows in the DB |
| Per-step pipeline outputs | `nodes` + `images` rows (one node per step, labelled `03_pages_2ppf`, etc.) |
| Exports (`--export pdf…` / `md`) | `<project_dir>/<slug>.pdf` and `<slug>.md` (written by `aglaia/workers/headless.py:_run_exports`) |

For a new project, `<project_dir>` defaults to `--parent-dir` (or the input
file's own parent) and `<slug>` derives from `--project-name` (or the input
filename).
