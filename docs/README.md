# Aglaïa documentation

Detailed docs for the Aglaïa project. Start at the project root `CLAUDE.md` for the high-level orientation.

| Doc | Topic |
|---|---|
| [quickstart.md](./quickstart.md) | **Start here** — install → capture/import → process → OCR → export, GUI + headless |
| [architecture.md](./architecture.md) | Process tree, IntegratedProcessingChain, log_queue protocol |
| [pipeline.md](./pipeline.md) | YAML pipeline schema, template substitution, default pipeline annotated |
| [processors.md](./processors.md) | DPIfixer, SkewFinder, PageDetector, Binarizer, PageDewarper — options, behavior, extension points, **drop-in user plugins** |
| [imagebuffer.md](./imagebuffer.md) | Standard image envelope, meta keys, write logic |
| [gui.md](./gui.md) | Capture GUI (aglaia.py), threads, key/voice bindings, calibration buttons |
| [calibration.md](./calibration.md) | Camera calibration workflow, DPI calibration, camera_params.json |
| [configuration.md](./configuration.md) | Config layers (defaults / YAML / CLI), `args.options` shape, path resolution |
| [development.md](./development.md) | Env setup, module map, multiprocessing constraints, adding processors, conventions |
| [storage.md](./storage.md) | M0 SQLite schema: tables, branches, query cookbook |
| [ocr.md](./ocr.md) | OCR engine interface + registry, bundled engines (Apple Vision/Document, Surya, PaddleOCR-VL, Mistral cloud), shared DPI / confidence knobs, cloud key storage |
| [export.md](./export.md) | PDF export (JBIG2 / G4 bitonal, JPEG colour, invisible OCR text layer) + Markdown pointer; visibility filtering |
| [app_data.md](./app_data.md) | Per-user dirs via platformdirs (data/cache/log/models/pipelines/plugins), env overrides, config DB schema |
| [markdown_export.md](./markdown_export.md) | `write_markdown` — OCR text → free-flowing Markdown; Apple Vision line-geometry heuristics (headings, paragraphs, dehyphenation, running-head removal, footnotes, lists, cross-page merge) |
| [lessons.md](./lessons.md) | Hard-won lessons: pitfalls (DPI-vulnerable constants, morphology line-bridging, multi-line bbox poisoning, input-polarity assumptions, MLX cache hygiene), implementation guidelines (new processor checklist, debug visualisation, span+baseline recipe), and a reference table of h_med-scaled constants. |
| [distribution.md](./distribution.md) | Release CI (signed/notarized macOS DMG, tag-driven), `Aglaia.spec`, required secrets, the aglaia.bibli.cc site (Astro landing + Starlight docs) |

> **Advanced / CV-research docs** (the math-heavy algorithm + page-dewarp
> references, with figures) live in **`../private_docs/`** — dev-only material,
> not published to the site and not copied into the public `aglaia` repo.

