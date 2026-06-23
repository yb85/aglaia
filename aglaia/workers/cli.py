# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Aglaïa command-line plumbing.

Single ``argparse`` schema shared between the GUI entry script
(``aglaia.py``) and the headless batch runner. The parser is
intentionally minimal — anything that's a *runtime preference* (theme,
default thumb size, default OCR engine) lives in the per-user config
DB; CLI flags are reserved for *per-invocation* choices.

Examples:

    # Open a project file
    aglaia ~/scans/myproject.agl

    # New project from PDFs, run OCR, export both PDF (G4) and Markdown
    aglaia ~/scans/source/*.pdf --ocr --ocr-lang fr-FR+en-US \
           --export pdf:g4+md --headless

    # New project from images, default engine and language
    aglaia ~/scans/page-*.png --ocr -p full

The positional arguments are auto-classified:

* exactly one ``.agl`` (or legacy ``.scanproj.sqlite``) file → open existing project
* one or more ``.pdf`` files          → new project from PDFs
* one or more image files             → new project from images
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# Canonical + legacy project file suffixes — files ending in any of
# these open as a project. Source of truth is aglaia.storage.
from aglaia.storage import (
    PROJECT_EXT as _PROJECT_EXT,
    LEGACY_PROJECT_EXT as _LEGACY_PROJECT_EXT,
    slug_from_project_file as _slug_from_project_file,
)
PROJECT_SUFFIXES = (_PROJECT_EXT, _LEGACY_PROJECT_EXT)
# Back-compat alias — older code reaches for the legacy single suffix.
PROJECT_SUFFIX = _LEGACY_PROJECT_EXT

OCR_ENGINE_CHOICES = ["auto", "apple", "surya"]
EXPORT_PDF_PROFILES = ["auto", "g4", "jbig2", "native"]


@dataclass
class Spec:
    """A parsed ``name[:tok|key=value][:…]`` spec (see ``parse_spec``)."""
    name: str
    tokens: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class ExportTask:
    kind: str             # "pdf" | "md"
    profile: Optional[str] = None  # for pdf: auto|g4|jbig2|native
    tokens: list[str] = field(default_factory=list)   # bare tokens from the spec
    params: dict[str, str] = field(default_factory=dict)  # key=value from the spec


@dataclass
class CliConfig:
    paths: list[Path] = field(default_factory=list)

    # Source classification (set by `classify_inputs`).
    source: str = "none"          # "project" | "pdfs" | "images" | "none"
    project_file: Optional[Path] = None
    inputs: list[Path] = field(default_factory=list)

    pipeline: Optional[str] = None        # name (config/pipelines/<name>.yaml) or path
    workers: Optional[int] = None
    input_dpi: Optional[float] = None
    # False → input_dpi only fills in inputs with NO embedded DPI (images
    # lacking DPI metadata). True ("force:" prefix) → override every input.
    input_dpi_force: bool = False
    camera_id: Optional[int] = None

    do_ocr: bool = False
    ocr_engine: str = "auto"              # auto | apple | surya | <registered engine>
    ocr_languages: list[str] = field(default_factory=list)  # empty == "auto"
    ocr_params: dict[str, str] = field(default_factory=dict)  # engine spec key=value
    ocr_batch: bool = False               # `--do-ocr mistral:batch` → submit a
                                          # Mistral batch job (vs `:stream`/sync)
    check_ocr: bool = False               # `--check-ocr <project>` → poll +
                                          # import pending Mistral batch jobs

    exports: list[ExportTask] = field(default_factory=list)
    md_refine: Optional[str] = None       # on-device LLM backend for MD cleanup

    headless: bool = False
    project_name: Optional[str] = None    # only honoured for new projects
    parent_dir: Optional[Path] = None

    diagnose_memory: bool = False
    force_proc: bool = False

    # Informational list-and-exit flags.
    list_pipelines: bool = False
    list_ocr: bool = False
    list_exports: bool = False

    def has_inputs(self) -> bool:
        return self.source != "none"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aglaia",
        description="Aglaïa scanner / page extraction pipeline.",
        allow_abbrev=False,
    )
    p.add_argument(
        "paths", nargs="*", type=Path,
        help="One .agl project file, OR one or more PDFs, OR one or more image files.",
    )
    p.add_argument("--workers", type=int, default=None,
                   help="Number of pipeline worker processes (overrides config).")
    p.add_argument("-p", "--pipeline", type=str, default=None,
                   help="Pipeline name (e.g. 'book_curved_x2') or path to a .yaml file.")
    p.add_argument(
        "--ocr", nargs="?", const="auto", default=None, metavar="ENGINE[:opt…]",
        help="Run OCR after the pipeline. Engine spec follows the standard "
             "name[:token|key=value] format: 'auto' (default, Apple Vision → "
             "Surya), 'apple', 'surya', or a registered engine, with optional "
             "params, e.g. 'surya:lang=fr-FR:beam=4'.",
    )
    p.add_argument(
        "--ocr-lang", type=str, default="auto",
        help="'+' joined ISO/BCP-47 language codes (e.g. 'fr-FR+en-US'). "
             "'auto' lets the engine decide. (Shorthand: a 'lang=' param in "
             "--ocr.)",
    )
    p.add_argument(
        "--export", type=str, default=None,
        help="'+' joined export specs in the standard name[:token|key=value] "
             "format: pdf, pdf:g4 (or pdf:profile=g4; auto|g4|jbig2|native), "
             "md, md:refine=apple_fm. E.g. 'pdf:g4+md'.",
    )
    p.add_argument(
        "--md-refine", type=str, default=None, metavar="BACKEND",
        help="Post-process the Markdown export with an on-device LLM "
             "(e.g. 'apple_fm' — Apple Foundation Models, macOS 26+). "
             "No-op when unavailable.",
    )
    p.add_argument("--headless", action="store_true",
                   help="Run end-to-end on the CLI without showing the UI.")
    p.add_argument("--check-ocr", action="store_true", dest="check_ocr",
                   help="Poll + import any pending Mistral batch OCR job(s) "
                        "for the given project, then exit. Pair with the "
                        "project path: aglaia --headless --check-ocr proj.agl")
    p.add_argument("--project-name", type=str, default=None,
                   help="Name for new projects (default: derive from input filename).")
    p.add_argument("--parent-dir", type=Path, default=None,
                   help="Parent folder for new projects (default: input file's parent).")
    p.add_argument(
        "--input-dpi", type=str, default=None, metavar="[force:]N",
        help="Input DPI for imported images. Bare 'N' fills only images "
             "with no embedded DPI metadata; 'force:N' overrides every "
             "input. (PDFs estimate DPI from page size regardless.)")
    p.add_argument("--camera-id", type=int, default=None)
    p.add_argument("--diagnose-memory", action="store_true",
                   help="Enable tracemalloc snapshots in the GUI process.")
    p.add_argument("--force-proc", action="store_true",
                   help="Reprocess every active scan on project open (wipes "
                        "branches + intermediate nodes, re-enqueues raws). "
                        "Without it, only scans whose objective is missing "
                        "from the DB are caught up.")
    p.add_argument("--pipeline-list", action="store_true",
                   help="List available pipelines and exit.")
    p.add_argument("--ocr-list", action="store_true",
                   help="List available OCR engines and exit.")
    p.add_argument("--export-list", action="store_true",
                   help="List available export formats and exit.")
    return p


def parse_argv(argv: list[str]) -> CliConfig:
    parser = build_parser()
    ns = parser.parse_args(argv)
    input_dpi, input_dpi_force = _parse_input_dpi(ns.input_dpi)
    ocr_spec = parse_spec(ns.ocr) if ns.ocr is not None else None
    ocr_languages = _parse_lang_arg(ns.ocr_lang)
    if not ocr_languages and ocr_spec and "lang" in ocr_spec.params:
        # `--ocr surya:lang=fr-FR+en-US` is shorthand for `--ocr-lang`.
        ocr_languages = _parse_lang_arg(ocr_spec.params["lang"])
    cfg = CliConfig(
        paths=[Path(p).expanduser() for p in (ns.paths or [])],
        pipeline=ns.pipeline,
        workers=ns.workers,
        input_dpi=input_dpi,
        input_dpi_force=input_dpi_force,
        camera_id=ns.camera_id,
        do_ocr=ocr_spec is not None,
        ocr_engine=ocr_spec.name if ocr_spec else "auto",
        ocr_languages=ocr_languages,
        ocr_params={k: v for k, v in (ocr_spec.params.items() if ocr_spec else ())
                    if k != "lang"},
        # `mistral:batch` (token) submits a batch job; `mistral:stream` (or no
        # token) is the synchronous default.
        ocr_batch=bool(ocr_spec and "batch" in ocr_spec.tokens),
        check_ocr=bool(getattr(ns, "check_ocr", False)),
        exports=_parse_export_arg(ns.export),
        md_refine=ns.md_refine,
        headless=bool(ns.headless),
        project_name=ns.project_name,
        parent_dir=Path(ns.parent_dir).expanduser() if ns.parent_dir else None,
        diagnose_memory=bool(ns.diagnose_memory),
        force_proc=bool(ns.force_proc),
        list_pipelines=bool(ns.pipeline_list),
        list_ocr=bool(ns.ocr_list),
        list_exports=bool(ns.export_list),
    )
    classify_inputs(cfg)
    return cfg


def run_list_commands(cfg: "CliConfig") -> bool:
    """Handle the `--*-list` flags: print to stdout and return True (the
    caller should exit). Returns False when no list flag was set."""
    if not (cfg.list_pipelines or cfg.list_ocr or cfg.list_exports):
        return False
    if cfg.list_pipelines:
        from aglaia.app_data import pipelines_dir
        print("Pipelines:")
        for p in sorted(pipelines_dir().glob("*.yaml")):
            name = p.stem
            try:
                import yaml as _yaml
                name = (_yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("name") or p.stem
            except Exception:
                pass
            print(f"  {p.stem:24} {name}")
    if cfg.list_ocr:
        print("OCR engines:")
        try:
            import aglaia.workers.ocr  # noqa: F401 — side-effect registers engines
            from aglaia.workers.ocr.engine import ENGINE_REGISTRY
            for name, cls in sorted(ENGINE_REGISTRY.items()):
                avail = "" if getattr(cls, "available", False) else " (unavailable)"
                desc = getattr(cls, "description", "") or ""
                print(f"  {name:14} {desc}{avail}")
        except Exception as e:  # noqa: BLE001
            print(f"  (could not load OCR engines: {e})")
    if cfg.list_exports:
        print("Export formats (use with --export):")
        print(f"  pdf            PDF profiles: {'|'.join(EXPORT_PDF_PROFILES)} "
              "(e.g. pdf:g4)")
        print("  md             Markdown; md:refine=<backend> for on-device LLM cleanup")
    return True


def _parse_input_dpi(raw: Optional[str]) -> tuple[Optional[float], bool]:
    """Parse ``--input-dpi`` → (dpi, force). Accepts ``N`` or ``force:N``."""
    if raw is None:
        return (None, False)
    s = str(raw).strip()
    force = False
    if s.lower().startswith("force:"):
        force = True
        s = s[len("force:"):].strip()
    try:
        return (float(s), force)
    except ValueError:
        raise SystemExit(f"--input-dpi: expected '[force:]<number>', got {raw!r}")


def _parse_lang_arg(raw: str) -> list[str]:
    """Split a '+'-joined language list. ``"auto"`` (default) → empty
    list, which downstream code interprets as 'engine picks'."""
    if not raw or raw.lower() == "auto":
        return []
    return [tok.strip() for tok in raw.split("+") if tok.strip()]


def _split_top(s: str, sep: str) -> list[str]:
    """Split ``s`` on single-char ``sep`` at the top level, honouring
    ``'…'`` / ``"…"`` quotes (which protect ``sep`` and are stripped from
    the output). Used for the reserved ``:`` and ``+`` separators so a
    value can contain them when quoted, e.g. ``opt="a:b"``."""
    out: list[str] = []
    buf: list[str] = []
    quote: Optional[str] = None
    for ch in s:
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        elif ch in ("'", '"'):
            quote = ch
        elif ch == sep:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def parse_spec(entry: str) -> Spec:
    """Parse one ``name[:arg][:arg]…`` spec — the standard option format
    for engines and exporters. ``:`` separates the name from its args and
    args from each other; each arg is either a bare **token** or a
    **key=value** pair (``=`` separates key from value). Both ``:`` and
    ``=`` are reserved; quote an arg (``'…'`` / ``"…"``) to use them
    literally in a value. The name is lower-cased; token/value case is
    preserved (language codes etc. are case-sensitive).

    Examples::

        pdf:g4                       → Spec('pdf', ['g4'], {})
        pdf:profile=jbig2            → Spec('pdf', [], {'profile': 'jbig2'})
        surya:lang=fr-FR:beam=4      → Spec('surya', [], {'lang': 'fr-FR', 'beam': '4'})
        md:refine=apple_fm           → Spec('md', [], {'refine': 'apple_fm'})
    """
    fields = _split_top(entry, ":")
    name = fields[0].strip().lower()
    tokens: list[str] = []
    params: dict[str, str] = {}
    for f in fields[1:]:
        f = f.strip()
        if not f:
            continue
        if "=" in f:
            k, v = f.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            tokens.append(f)
    return Spec(name=name, tokens=tokens, params=params)


def _parse_export_arg(raw: Optional[str]) -> list[ExportTask]:
    """Parse a '+'-joined list of export specs. Each spec follows the
    standard ``name[:token|key=value]…`` format (see ``parse_spec``):

        pdf            pdf:g4         pdf:profile=jbig2
        md             md:refine=apple_fm
        pdf:g4+md

    Unknown targets / PDF profiles raise SystemExit."""
    if not raw:
        return []
    out: list[ExportTask] = []
    for entry in _split_top(raw, "+"):
        entry = entry.strip()
        if not entry:
            continue
        spec = parse_spec(entry)
        if spec.name == "md":
            out.append(ExportTask(kind="md", tokens=spec.tokens, params=spec.params))
            continue
        if spec.name == "pdf":
            prof = (spec.tokens[0] if spec.tokens
                    else spec.params.get("profile", "auto")).lower()
            if prof not in EXPORT_PDF_PROFILES:
                raise SystemExit(
                    f"Unknown PDF profile in --export: {prof!r}. "
                    f"Choices: {EXPORT_PDF_PROFILES}"
                )
            out.append(ExportTask(kind="pdf", profile=prof,
                                  tokens=spec.tokens, params=spec.params))
            continue
        raise SystemExit(
            f"Unknown export target: {spec.name!r}. Use pdf[:profile] or md."
        )
    return out


def classify_inputs(cfg: CliConfig) -> None:
    """Decide whether the positional args open a project, ingest PDFs,
    or ingest images. Mixed types raise SystemExit."""
    if not cfg.paths:
        return
    abs_paths = [p.resolve() for p in cfg.paths]
    proj = [p for p in abs_paths if any(p.name.endswith(s) for s in PROJECT_SUFFIXES)]
    pdfs = [p for p in abs_paths if p.suffix.lower() == ".pdf"]
    imgs = [p for p in abs_paths if p.suffix.lower() in IMAGE_SUFFIXES]

    other = set(abs_paths) - (set(proj) | set(pdfs) | set(imgs))
    if other:
        raise SystemExit(
            "Unsupported input file(s): " + ", ".join(str(p) for p in sorted(other))
        )

    if proj:
        if len(proj) > 1 or pdfs or imgs:
            raise SystemExit(
                "When opening a project, pass exactly one .agl "
                "file and no other inputs."
            )
        cfg.source = "project"
        cfg.project_file = proj[0]
        return
    if pdfs and imgs:
        raise SystemExit("Cannot mix PDF and image inputs in the same run.")
    if pdfs:
        cfg.source = "pdfs"
        cfg.inputs = pdfs
        return
    if imgs:
        cfg.source = "images"
        cfg.inputs = imgs


def resolve_pipeline_path(pipeline_arg: Optional[str]) -> Optional[Path]:
    """Map a ``--pipeline`` argument to a YAML path.

    Accepts either a bare name (looked up under ``config/pipelines/``)
    or a direct path. Returns None when the argument is unset so the
    caller can apply its own default."""
    if not pipeline_arg:
        return None
    p = Path(pipeline_arg).expanduser()
    if p.is_file():
        return p.resolve()
    # Treat as a name under the bundled pipelines/ dir.
    from aglaia.assets import config_path
    candidate = config_path("pipelines", f"{pipeline_arg}.yaml")
    if candidate.is_file():
        return candidate
    raise SystemExit(
        f"--pipeline: not found as path or under config/pipelines/: {pipeline_arg!r}"
    )


def default_project_name(cfg: CliConfig) -> str:
    """When the user omits --project-name, derive one from the first
    input file's stem. PDFs → file stem. Images → parent dir name."""
    if cfg.project_name:
        return cfg.project_name
    if cfg.source == "pdfs" and cfg.inputs:
        return cfg.inputs[0].stem
    if cfg.source == "images" and cfg.inputs:
        # All images in one folder → use the folder name.
        return cfg.inputs[0].parent.name or cfg.inputs[0].stem
    if cfg.source == "project" and cfg.project_file:
        return _slug_from_project_file(cfg.project_file)
    return "project"


def effective_workers(cli_value: Optional[int]) -> int:
    """CLI flag wins; otherwise use the preference stored in the app-
    data config DB; otherwise the bootstrap default."""
    if cli_value is not None:
        return max(1, int(cli_value))
    try:
        from aglaia.app_data import db as app_db
        with app_db.session() as conn:
            app_db.bootstrap(conn)
            return max(1, int(app_db.get(conn, app_db.KEY_WORKERS, 4)))
    except Exception:
        return 4


def default_parent_dir(cfg: CliConfig) -> Path:
    """Best-effort parent directory when --parent-dir is unset. Falls
    back to the input file's own parent so the project file lands next
    to its source material."""
    if cfg.parent_dir:
        return cfg.parent_dir.expanduser().resolve()
    if cfg.source in ("pdfs", "images") and cfg.inputs:
        return cfg.inputs[0].parent.resolve()
    if cfg.source == "project" and cfg.project_file:
        return cfg.project_file.parent.resolve()
    from aglaia.app_data import default_documents_dir
    return default_documents_dir()
