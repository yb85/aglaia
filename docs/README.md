# Aglaïa documentation

Detailed docs for the Aglaïa project. Start at the project root `CLAUDE.md` for the high-level orientation.

| Doc | Topic |
|---|---|
| [quickstart.md](./quickstart.md) | **Start here** — install → capture/import → process → OCR → export, GUI + headless |
| [cli.md](./cli.md) | CLI reference — subcommands (`gui` / `run` / `ocr` / `setup` / `list` / `plugins` / `server` / `version` / `skill`), shared options, the `--ocr`/`--export` spec format, migration from the old flat flags |
| [server.md](./server.md) | `aglaia server` HTTP job API — endpoints, auth, job lifecycle, Mistral-batch backoff, capability download URLs, completion email, admin panel, storage layout |
| [architecture.md](./architecture.md) | Process tree, IntegratedProcessingChain, log_queue protocol |
| [pipeline.md](./pipeline.md) | YAML pipeline schema, template substitution, default pipeline annotated |
| [processors-review.md](./processors-review.md) | **Architecture review** — where processors bleed into each other, what the pipeline still keys on a processor's NAME, and the processor-protocol plan (REQUIRES/PROVIDES, EDITABLE, render_debug, summary) |
| [processors.md](./processors.md) | DPIfixer, SkewFinder, PageDetector, Binarizer, PageDewarper — options, behavior, extension points, **drop-in user plugins** |
| [imagebuffer.md](./imagebuffer.md) | Standard image envelope, meta keys, write logic |
| [gui.md](./gui.md) | Capture GUI (aglaia), threads, key/voice bindings, calibration buttons |
| [calibration.md](./calibration.md) | Camera calibration workflow, DPI calibration, camera_params.json |
| [configuration.md](./configuration.md) | Config layers (defaults / YAML / CLI), `args.options` shape, path resolution |
| [ui-writing.md](./ui-writing.md) | **Writing for the user** — the UI/log register split, measured length budgets, error slots, case & punctuation, the do-not-write list |
| [development.md](./development.md) | Env setup, module map, multiprocessing constraints, adding processors, conventions |
| [storage.md](./storage.md) | SQLite schema (migrations 0001–0014): tables, branches, per-page `manual_overrides`, the stored raw Mistral output, query cookbook |
| [ocr.md](./ocr.md) | OCR engine interface + registry, bundled engines (Apple Vision / Apple Document, Surya, GLM-OCR, Unlimited, Mistral cloud), Mistral batch and its stored raw output, shared DPI / confidence knobs, cloud key storage |
| [export.md](./export.md) | PDF export (JBIG2 / G4 bitonal, JPEG colour, invisible OCR text layer that fails loudly when it cannot be written), the **OCR textpack** for corpus, Markdown pointer, visibility filtering |
| [destinations.md](./destinations.md) | Sending an export somewhere — the `destinations` plugin kind, the `Destination` contract, and the ones in the registry (folder, calibre, Kindle, corpus) |
| [plugin-store.md](./plugin-store.md) | The plugin store — curated registry, versioned `plugin_api`, namespaced secrets, install and update flows, review checklist, threat model |
| [app_data.md](./app_data.md) | Per-user dirs via platformdirs (data/cache/log/models/pipelines/plugins), env overrides, config DB schema |
| [markdown_export.md](./markdown_export.md) | `write_markdown` — OCR text → free-flowing Markdown; Apple Vision line-geometry heuristics (headings, paragraphs, dehyphenation, running-head removal, footnotes, lists, cross-page merge) |
| [lessons.md](./lessons.md) | Hard-won lessons: pitfalls (DPI-vulnerable constants, morphology line-bridging, multi-line bbox poisoning, input-polarity assumptions, MLX cache hygiene), implementation guidelines (new processor checklist, debug visualisation, span+baseline recipe), and a reference table of h_med-scaled constants. |
| [distribution.md](./distribution.md) | Release CI (signed/notarized macOS DMG, Windows installer, Linux AppImage — all tag-driven), `Aglaia.spec`, required secrets, the aglaia.bibli.cc site (Astro landing + Starlight docs) |
| [theme.md](./theme.md) | Colours, icons and artwork: the palette tokens, SVG tinting, and the sizing rule a new icon must follow |
| [i18n.md](./i18n.md) | Translation catalogues, the `tr()` discipline, and how a new language is added |
| [ocr-benchmark.md](./ocr-benchmark.md) | Engine-by-engine accuracy against input DPI, on a hard Greek + French corpus |
| [subcommand-cli.md](./subcommand-cli.md) | The plan behind the subcommand CLI, kept for the reasoning (the reference is [cli.md](./cli.md)) |

> **Advanced / CV-research docs** (the math-heavy algorithm + page-dewarp
> references, with figures) live in **`../private_docs/`** — dev-only material,
> not published to the site and not copied into the public `aglaia` repo.

