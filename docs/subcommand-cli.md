# Subcommand CLI — implementation plan

Redesign the `aglaia` command from one flat flag-soup parser into a
**subcommand-based interface** (git/svn/cargo style), following the principles
in [jmmv.dev — CLI design: subcommand-based interfaces](https://jmmv.dev/2013/09/cli-design-subcommand-based-interfaces.html).

## Why

Today everything hangs off a single `argparse` parser
(`aglaia/workers/cli.py:build_parser`) and `main()` (`aglaia/app.py:1048`)
demultiplexes the mode from a tangle of flags:

- `--headless` switches batch vs GUI, but most flags only make sense for one of
  them (`--camera-id`/`--diagnose-memory` are GUI-only; `--ocr`/`--export`/
  `--project-name`/`--input-dpi` are batch-only) — exactly the "tool-level flags
  that imply universal applicability but only apply to a subset" anti-pattern.
- mode-ish behaviours are flags too: `--setup`, `--check-ocr`, `--pipeline-list`,
  `--ocr-list`, `--export-list`, `--version`. These are *different operations*,
  not options.
- the dispatch ladder in `main()` (`run_list_commands` → `setup` → `headless` →
  no-GUI fallback → GUI) is implicit and order-sensitive.

Subcommands make each mode self-documenting, scope flags to where they apply,
and give a clean place to add `server` (the iOS bridge receiver, `feat/47-…`)
and future modes.

## Principles applied (from the article)

1. `aglaia [global opts] <command> [command opts] [args]`. Anything **before**
   the command applies to *every* command; anything **after** belongs to *that*
   command only.
2. A flag repeated across some-but-not-all commands is fine **iff** it behaves
   identically and shares parsing code (e.g. `--pipeline`/`--workers` on both
   `gui` and `run`).
3. `help` and `version` are **subcommands**, not flags — different behaviour, not
   options. (We keep `-h`/`--help` working as an alias because users expect it,
   but `aglaia help <command>` is the documented form.)
4. Don't expose a universal flag that only some commands honour.

## Command tree

```
aglaia [-v|--quiet] [--config FILE] <command> …

  gui   [PROJECT]                 launch the capture GUI (DEFAULT when omitted)
  run   PATHS...                  headless batch: ingest → pipeline → (ocr) → (export)
  setup                          interactive first-run setup (CLI-only installs)
  server [--host H] [--port N]   long-running HTTP API: warm workers+models, run-equivalent jobs
  ocr     PROJECT                 (re)run OCR on an existing project
  export  PROJECT                 (re)run export on an existing project
  cloud-check PROJECT             poll + import pending Mistral (cloud) batch OCR jobs
  list    pipelines|ocr|exports   introspection
  help    [COMMAND]               help (general or per-command)
  version                        print version and exit
```

`gui` is the default: `aglaia` → `gui`, and `aglaia ~/book.agl` →
`gui ~/book.agl` (see *Default & backward-compat*).

### Naming notes
- **`run`** is the headless-batch command (not `cli` — the whole binary is a CLI;
  `aglaia run book.pdf` is unambiguous). No `cli` alias.
- `ocr`/`export`/`cloud-check` are the modular split of today's `--ocr`/
  `--export`/`--check-ocr`. They still ride **inline on `run`** for the one-shot
  path (`run … --ocr surya --export pdf:g4`); the standalone commands let you
  re-run a stage on an existing `.agl` without re-processing.
- `cloud-check` (not bare `check`) makes the **cloud** nature explicit — it
  polls a remote Mistral Document AI batch job, the one network/billing-touching
  command. Reads as cloud, not a local lint/health check.

## Global vs per-command options

| Scope | Options |
|---|---|
| **Global** (before the command) | `-v/--verbose`, `-q/--quiet` (log level); `--config FILE` (config DB / yaml override). Truly universal. |
| **`gui`** | `PROJECT` (positional, optional); `--camera-id N`; `--diagnose-memory`; + shared `--pipeline`, `--workers`, `--force-proc`. |
| **`run`** | `PATHS...` (positional, 1+); `--ocr [ENGINE[:opt…]]`; `--ocr-lang`; `--export SPEC`; `--md-refine`; `--project-name`; `--parent-dir`; `--input-dpi`; + shared `--pipeline`, `--workers`, `--force-proc`. (No `--headless` — `run` *is* headless.) |
| **`server`** | `--host`, `--port`; API-key auth (see *Server* below); + the same chain knobs as `run` as per-request defaults. |
| **`ocr`** | `PROJECT`; `--engine`/`--lang`/`--batch` (the parsed `--ocr` spec, exploded into real flags). |
| **`export`** | `PROJECT`; `--pdf [PROFILE]`, `--md`, `--md-refine`. |
| **shared** (on `gui`+`run`) | `--pipeline NAME|PATH`, `--workers N` (0=auto), `--force-proc`. Identical semantics + one parse helper. |

Flag → command migration (every current flag has a home):

| today | becomes |
|---|---|
| `--headless` | implied by `run` (removed) |
| `paths` | `run PATHS` / `gui PROJECT` |
| `-p/--pipeline`, `--workers`, `--force-proc` | shared on `gui`+`run` |
| `--ocr`, `--ocr-lang` | `run --ocr…` and `ocr` command |
| `--export`, `--md-refine` | `run --export…` and `export` command |
| `--project-name`, `--parent-dir`, `--input-dpi` | `run` |
| `--camera-id`, `--diagnose-memory` | `gui` |
| `--check-ocr` | `cloud-check` command |
| `--pipeline-list`/`--ocr-list`/`--export-list` | `list pipelines|ocr|exports` |
| `--setup` | `setup` command |
| `--version` | `version` command (+ `--version` alias) |

## Default command (the no-subcommand case)

The first non-global token decides:

1. it matches a known command → dispatch that command;
2. otherwise → inject the default command `gui` and treat the token as its arg.

So `aglaia`, `aglaia book.agl`, `aglaia ~/scans/` open the GUI.

**No back-compat shim.** The app is pre-1.0 (alpha/rc1), no stable CLI contract
to honour. The old flat flags (`--headless`, `--setup`, `--ocr`, `--pipeline-list`,
…) are simply **removed**; passing them with no matching command means they fall
to `gui` as an unknown arg and error out the normal way. No deprecation path, no
mapping layer — a clean break. (Typer/argparse already reject unknown options;
we don't special-case them.)

Implementation: a small pre-parse peeks `argv[0]` (after pulling out global
opts) against the command set; if it isn't a known command and isn't `-h`,
prepend `gui`.

## Library choice — Typer

**Chosen: [Typer](https://typer.tiangolo.com/).** Type-hint-driven CLI on top of
Click. Each command is a plain function with annotated params; Typer derives the
parser, help, and validation.

Why Typer here:
- **Subcommands are the native model.** A `typer.Typer()` app with
  `@app.command()` functions *is* the `tool cmd opts args` grammar — no manual
  sub-parser wiring. Nested groups (`list pipelines`) via `app.add_typer(...)`.
- **Per-command help for free**, and `help`/`version` map cleanly onto the
  article's "commands not flags" stance (`@app.command("version")`, plus a
  `--version` eager callback as the alias users expect).
- **Less code, fewer footguns** than hand-rolled argparse subparsers — the
  current `build_parser` boilerplate (defaults, `metavar`, dest wiring) collapses
  into typed signatures. Validation/enums (pdf profile, ocr engine) come from the
  type.
- **Cost is acceptable.** Typer + Click are small pure-Python wheels, no native
  deps. It lands in **base** deps (the CLI entry is always installed). That's a
  deliberate, bounded exception to the slim-base rule — one lightweight dep for
  the single most user-facing surface. (`rich` is already a base dep; Typer uses
  it for help rendering, so the marginal weight is tiny.)

Add to `pyproject.toml` base `dependencies`: `typer>=0.12`.

(`argparse` subparsers were the zero-dep alternative — rejected: more
boilerplate, weaker validation, worse help, for a binary whose CLI is its
primary interface.)

The command/handler split below is the Typer shape: one module per command, each
exposing a Typer-decorated function.

## Code structure

```
aglaia/cli/
  __init__.py        # the typer.Typer() app + run(argv)->int entry; root callback (global opts)
  shared.py          # shared annotated params (pipeline/workers/force-proc) + the spec parsers
  commands/
    gui.py           # @app.command("gui")     def gui(...): -> int
    run.py           # @app.command("run")      (the old _run_headless + CliConfig assembly)
    setup.py
    server.py        # TBD — wraps the bridge receiver
    ocr.py
    export.py
    cloud_check.py
    list.py          # a sub-Typer: list pipelines|ocr|exports
    version.py
```

- Each command module defines a Typer-decorated function registered on the root
  `app` (or a sub-Typer for `list`). Shared options are reusable annotated
  params (`Annotated[int, typer.Option(...)]`) imported from `shared.py` — the
  "identical behaviour, shared parsing code" the article allows.
- The root `@app.callback()` holds the **global** opts (`-v/-q`, `--config`) and
  the default-command pre-parse (prepend `gui` when `argv[0]` isn't a command).
- `aglaia/cli/__init__.py:run(argv)` invokes the Typer app and returns its exit
  code. This replaces the dispatch ladder in `aglaia/app.py:main()`.
- `help` is Typer-native (`aglaia --help`, `aglaia run --help`); add an explicit
  `help` command + `version` command/`--version` callback for the article's
  command form.
- `CliConfig` / `parse_argv` (`aglaia/workers/cli.py`) is **kept as the batch
  config object** but built by `commands/run.py` from that command's namespace
  (not the global flat parse). The spec parsers (`parse_spec`, `_parse_input_dpi`,
  `_parse_lang_arg`, `_parse_export_arg`) move under `aglaia/cli/` or stay and
  are imported — they're reused by `run`/`ocr`/`export`.
- `__main__.py:run` and the `pyproject` `[project.scripts] aglaia` entry point
  unchanged externally; they now call `aglaia.cli.run`.

## Migration steps (incremental, each shippable)

1. **Scaffold** `aglaia/cli/` (Typer app) with `gui`/`run` only, default → `gui`.
   Add `typer` to base deps. Wire `aglaia.cli.run` behind the console-script
   entry; reduce `app.main` to the GUI/headless handlers the commands call. Port
   the `main()` dispatch into the two command funcs. Delete the old flat
   `build_parser`. Add `tests/cli/test_dispatch.py`.
2. **Mode commands**: `setup`, `list`, `version`. Old `--setup`/`--*-list`/
   `--version` flags are gone (clean break — they now error as unknown).
3. **Modular stages**: `cloud-check`, then `ocr`/`export` as standalone commands
   reusing the spec parsers.
4. **`server`**: the warm-pool HTTP API (`server` extra: FastAPI/uvicorn).
   Reuses `run`'s job assembly over a persistent chain. Lands after `run`/`gui`
   stabilise; folds in the `feat/47` bridge as one client.

## Server

`aglaia server` is a **long-running daemon** exposing an HTTP API. Semantically
it's `run` as a service: same ingest → pipeline → (ocr) → (export) jobs, but
with the **workers and models preloaded and kept warm in memory** across
requests — so each job skips chain spin-up + model load (the dominant latency on
a cold `run`).

- **Auth**: API-key. Key(s) configured out-of-band (config DB / env); requests
  carry it (`Authorization: Bearer …`). Not just the phone bridge — any client.
- **Endpoints (sketch)**: `POST /jobs` (upload images/PDF + per-request pipeline/
  ocr/export overrides, mirrors `run` flags) → job id; `GET /jobs/{id}` status;
  `GET /jobs/{id}/result` artifact. Health/`GET /capabilities`.
- **Warm pool**: holds an `IntegratedProcessingChain` (or a worker pool) alive
  between jobs; concurrency bounded by `--workers`. Model handles (dewarp, OCR
  engine) loaded once at boot.
- **Stack**: FastAPI/uvicorn, behind a `server` extra (not base) — the old
  removed `web2scans.py` is the closest prior art. Out of v1 scope; lands after
  `run`/`gui` are stable.

## Help & version

- `aglaia help` → top-level help (lists commands).
- `aglaia help run` → `run` usage (== `aglaia run -h`).
- `aglaia version` → version string (== keep `--version` alias).

## Tests

- A table-driven `tests/cli/test_dispatch.py`: `(argv, expected_command,
  expected_key_args)` — covers defaults, back-compat (`aglaia book.agl`,
  `aglaia x.pdf --headless`), each command's required/optional args, and that
  unknown global+command combos error cleanly.
- Keep the existing `parse_spec`/`_parse_input_dpi` unit tests; add per-command
  arg-parsing tests.

## Decisions locked

- **Library: Typer** (base dep).
- **No back-compat** for old flat flags — clean break (pre-1.0).
- **`cloud-check`** for the Mistral batch poll (cloud-explicit name).
- **`run`** is the batch command — no `cli` name/alias.
- **`server`** = warm-pool HTTP API with API-key auth, `run`-equivalent jobs
  (broader than the phone bridge).

## Open questions

1. **`ocr`/`export` standalone**: ship in v1 or defer (inline-on-`run` covers the
   common path)?
2. **Server endpoint/auth detail**: design alongside the bridge work, not now.
