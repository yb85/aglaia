# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import argparse
import importlib
import os
import sys
import yaml
import multiprocessing
from pathlib import Path
from aglaia.workers.IntegratedProcessingChain import IntegratedProcessingChain
from aglaia.workers.chain_abstraction import SimpleChainElement
from aglaia.processors import registry as _proc_registry
from slugify import slugify


# ── startup capability probe ───────────────────────────────────────

def _probe_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _probe_model_file(env_var: str, *candidates: str,
                      glob: str | None = None) -> tuple[bool, str]:
    """True if env var points at an existing file OR any candidate name
    (or `glob`) resolves under the user's configured `models_dir()` or
    repo-local `model/` / `models/`. Returns (found, source).

    Search order matches what the actual backend resolvers use:
      1. `<env_var>` absolute path
      2. user-configured `models_dir()` (Settings → Models)
      3. repo-relative `model/` then `models/`
    """
    override = os.environ.get(env_var, "").strip()
    if override and Path(override).expanduser().is_file():
        return True, override

    search_dirs: list[Path] = []
    try:
        from aglaia.app_data import models_dir as _md
        search_dirs.append(_md())
    except Exception:
        pass
    repo = Path(__file__).resolve().parents[2]
    for sub in ("model", "models"):
        search_dirs.append(repo / sub)

    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in candidates:
            p = d / name
            if p.is_file():
                try:
                    return True, str(p.relative_to(repo))
                except ValueError:
                    return True, str(p)
        if glob:
            hits = sorted(d.glob(glob))
            if hits:
                p = hits[0]
                try:
                    return True, str(p.relative_to(repo))
                except ValueError:
                    return True, str(p)
    return False, ""


def probe_capabilities() -> list[tuple[str, bool, str]]:
    """Returns list of (name, available, detail).

    Cheap — try/except imports only, no model loads."""
    caps: list[tuple[str, bool, str]] = []

    # JAX (CPU or accelerated)
    if _probe_import("jax"):
        try:
            import jax
            devs = jax.devices()
            kinds = ",".join(sorted({d.platform for d in devs}))
            caps.append(("jax", True, f"devices={kinds}"))
        except Exception as e:
            caps.append(("jax", True, f"present but query failed: {e}"))
    else:
        caps.append(("jax", False, "pip install jax"))

    # MLX (Apple Silicon Metal)
    if _probe_import("mlx.core"):
        caps.append(("mlx", True, "Apple Metal"))
    else:
        caps.append(("mlx", False, "pip install mlx (Apple Silicon only)"))

    # Apple Vision — text detection + OCR
    if sys.platform == "darwin":
        if _probe_import("Vision"):
            caps.append(("apple_vision", True, "text detection + OCR"))
        else:
            caps.append(("apple_vision", False, "pip install pyobjc-framework-vision"))
        # Apple Document engine — structured OCR (macOS 26+)
        try:
            import Vision as _V
            has_docs = hasattr(_V, "VNRecognizeDocumentsRequest")
        except Exception:
            has_docs = False
        caps.append(("apple_document", has_docs,
                     "structured document OCR" if has_docs
                     else "requires macOS 26+"))
    else:
        caps.append(("apple_vision", False, "macOS only"))
        caps.append(("apple_document", False, "macOS only"))

    # Vosk — offline voice control (cross-platform; needs the `voice` extra
    # and a downloaded model).
    if _probe_import("vosk"):
        caps.append(("vosk", True, "voice control"))
    else:
        caps.append(("vosk", False, "uv sync --extra voice"))

    # PaddleOCR-VL — needs the python orchestrator (paddleocr) + the
    # mlx-vlm server backend. The actual weights live under the user's
    # models dir; missing weights only block runtime, not the probe.
    if _probe_import("paddleocr") and _probe_import("mlx_vlm"):
        try:
            from aglaia.workers.ocr.paddle_vl import _paddle_weights_dir
            weights = _paddle_weights_dir()
        except Exception:
            weights = None
        if weights is not None and weights.exists():
            caps.append(("paddle", True,
                          f"PaddleOCR-VL + mlx-vlm ({weights})"))
        else:
            caps.append(("paddle", True,
                          "PaddleOCR-VL + mlx-vlm (weights not downloaded)"))
    else:
        caps.append(("paddle", False,
                      "pip install paddleocr[doc-parser] paddlepaddle mlx-vlm"))

    # Surya OCR — needs the python package AND a usable llama-server
    # binary (bundled inside the frozen .app, vendored in dev checkouts
    # under vendor/llama-server/<plat>/, or installed via brew).
    if _probe_import("surya"):
        try:
            from aglaia.workers.ocr.surya import _resolve_llama_server
            server_path = _resolve_llama_server()
        except Exception:
            server_path = None
        if server_path:
            caps.append(("surya", True,
                          f"OCR engine (llama.cpp at {server_path})"))
        else:
            caps.append(("surya", False,
                          "needs `brew install llama.cpp` (llama-server) "
                          "or `python scripts/fetch_llama_server.py`"))
    else:
        caps.append(("surya", False, "pip install surya-ocr"))

    # EAST model file
    found, src = _probe_model_file(
        "AGLAIA_EAST_MODEL",
        "frozen_east_text_detection.pb", "east_text_detection.pb",
    )
    caps.append(("east model", found, src or "set AGLAIA_EAST_MODEL"))

    # DBNet model file — same KNOWN_FILENAMES + glob as the actual
    # backend resolver (aglaia/processors/layout_backends/dbnet.py).
    found, src = _probe_model_file(
        "AGLAIA_DBNET_MODEL",
        "en_PP-OCRv5_mobile_det.onnx", "en_PP-OCRv5_server_det.onnx",
        "PP-OCRv5_mobile_det.onnx", "PP-OCRv5_server_det.onnx",
        "en_PP-OCRv4_mobile_det.onnx", "en_PP-OCRv4_server_det.onnx",
        "PP-OCRv4_mobile_det.onnx", "PP-OCRv4_server_det.onnx",
        "ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_det_server_infer.onnx",
        "en_PP-OCRv3_det_mobile.onnx", "en_PP-OCRv3_mobile_det.onnx",
        "PP-OCRv3_mobile_det.onnx", "en_PP-OCRv3_det_infer.onnx",
        glob="*PP-OCR*det*.onnx",
    )
    caps.append(("dbnet model", found, src or "set AGLAIA_DBNET_MODEL"))

    return caps


def print_startup_info(workers: int | None = None) -> None:
    """Plain-text startup banner. Reports APP_DATA dirs + worker count +
    backend capability probe. Replaces the old Rich CLI dump."""
    try:
        from aglaia.app_data import app_data_dir, cache_dir, log_dir
        app_dir = str(app_data_dir())
        cch_dir = str(cache_dir())
        lg_dir = str(log_dir())
    except Exception as e:
        app_dir = f"<unavailable: {e}>"
        cch_dir = lg_dir = "<unavailable>"

    OK = "\x1b[32m✓\x1b[0m"
    KO = "\x1b[31m✗\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    RST = "\x1b[0m"

    print(f"{BOLD}Aglaïa{RST}")
    if workers is not None:
        print(f"  workers   {workers}")
    print(f"  APP_DATA  {app_dir}")
    print(f"  cache     {cch_dir}")
    print(f"  log       {lg_dir}")
    print(f"{BOLD}backends{RST}")
    caps = probe_capabilities()
    name_w = max(len(n) for n, _, _ in caps)
    for name, ok, detail in caps:
        mark = OK if ok else KO
        print(f"  {mark} {name:<{name_w}}  {DIM}{detail}{RST}")
    # Pending drop-in plugins can't be acknowledged without the GUI trust
    # popup, so headless runs don't load them — warn so the user knows.
    try:
        from aglaia.app_data import plugins as _plugins
        pending = _plugins.scan_pending()
        if pending:
            print(f"{BOLD}plugins{RST}")
            for cand in pending:
                print(f"  {KO} {cand.path.name:<{name_w}}  {DIM}pending "
                      f"({cand.reason}) — not loaded; accept in the GUI{RST}")
    except Exception:
        pass
    print()


# Default non-CLI configuration
DEFAULT_CONFIG = {
    "keycontrols": {
        "scan": ["Space", "S"],
        "trash": ["Backspace", "D"],
        "rotate": ["R"]
    },
    # Two universal commands only. Keys are the dispatched actions
    # (handle_voice_command), values the spoken trigger word(s). Keep this
    # tiny: a short, distinct vocabulary is what keeps Vosk's constrained
    # grammar from shoehorning random speech onto a command.
    "voicecontrols": {
        "scan": ["photo"],
        "trash": ["delete"],
        "debounce_time": 1
    },
    "processing": {
    },
    "layout": {
        "margin_mm": 2.0,           # Margin around crop
        "rescale_threshold": 0.01,  # Factor diff to trigger rescale
        "binarize_threshold": 127   # fallback threshold
    },
    "dewarp": {
        "max_oob": 1000,
        "margin": 5,                 # mm (padding before dewarp)
        "page_margin": 20,           # mm (mask optimization)
        "shear_cost": 20.0,
        "remap_decimate": 16,
        "focal_length": 1.2,
        "mask_vis_opacity": 0.4,
        "jax_metal": False # the metal bindings development is not complete and seems abandonned
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
    },
    "display": {
        "card_max_width_px": 150,
        "zoom_tolerance": 0.2,
    },
}

# Default CLI arguments
DEFAULT_ARGS = {
    "workers": 4,
    "max_pages": 2,
    "make_pdf": False,
    "camera_id": 0,
    "voice_control": False,
    "transform": "0",
    "debug": False,
    "input_dpi": None
}

def load_yaml_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {path}: {e}")
        return {}

def initialize(mode="capture"):
    """
    Initialize configuration and parse arguments.
    mode: "capture" or "pdf"
    """
    # 1. Pre-scan for config file argument
    config_path = None
    if "-c" in sys.argv:
        idx = sys.argv.index("-c")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    elif "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]

    # 2. Prepare defaults
    current_args = DEFAULT_ARGS.copy()
    current_config = DEFAULT_CONFIG.copy()

    # 3. Load and merge YAML config if present
    if config_path:
        yaml_data = load_yaml_config(config_path)
        if yaml_data:
            # Update args defaults
            if "args" in yaml_data and isinstance(yaml_data["args"], dict):
                current_args.update(yaml_data["args"])
            
            # Update config defaults (deep merge for top-level keys)
            if "config" in yaml_data and isinstance(yaml_data["config"], dict):
                user_config = yaml_data["config"]
                for key in current_config:
                    if key in user_config:
                        current_config[key] = user_config[key]

    # 4. Setup ArgumentParser with updated defaults
    desc = "Capture scans from Webcam." if mode == "capture" else "Extract scans from PDFs."
    parser = argparse.ArgumentParser(description=desc)
    
    # Common Arguments
    parser.add_argument("-c", "--config", type=Path, help="Path to YAML config file")
    parser.add_argument("-p", "--pipeline", type=Path, help="Path to Pipeline YAML definition")
    
    # Mode Specific Arguments
    if mode == "capture":
        parser.add_argument("workspace_dir", type=Path, help="Directory to save captures and layouts")
        parser.add_argument("--camera-id", type=int, default=current_args["camera_id"], help="Camera Device ID")
        parser.add_argument("--voice-control", action="store_true", default=current_args["voice_control"], help="Enable voice commands")
        parser.add_argument("--transform", type=str, default=current_args["transform"], help="Transform camera feed: e.g. 90, 180+flip, mirror")
    
    elif mode == "pdf":
        parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to process")
        # PDF mode usually implies some CPU parallelism, overriding layout workers default?
        parser.add_argument("--pdf-workers", type=int, default=None, help="Number of concurrent PDF extractors (default: CPU count)")

    # Common Processing Arguments
    parser.add_argument("--workers", type=int, default=current_args["workers"], help="Number of workers for the processing chain")
    parser.add_argument("--max-pages", type=int, default=current_args["max_pages"], help="Max layouts per page (merging small ones). 0=Infinity. Default 2.")
    parser.add_argument("--make-pdf", action="store_true", default=current_args["make_pdf"], help="Create a PDF from the extracted layouts")
    parser.add_argument("--debug", action="store_true", default=current_args["debug"], help="Save debug images for analysis")
    parser.add_argument("--input-dpi", type=float, default=current_args["input_dpi"], help="Input DPI override (Default: 100 for Cam / Auto for PDF)")

    # 5. Parse arguments
    args = parser.parse_args()

    args.config = current_config
    
    args.options = {
        "dewarp": {
            **current_config["dewarp"],
        },
        "layout": {
            "workers": args.workers,
            "max_pages": args.max_pages,
            "config": current_config["layout"]
        },
        "general": {
            "debug": args.debug,
            "input_dpi": args.input_dpi
        },
        "calibration": {
             **current_config.get("calibration", {}),
             "camera_matrix": None,
             "camera_matrix_resolution": None
        }
    }
    
    # Mode specific paths
    if mode == "capture":
        # SQLite-backed projects: everything lives in `<slug>.scanproj.sqlite`.
        # Debug + export folders are siblings prefixed with the slug.
        workspace = args.workspace_dir
        slug = getattr(args, "project_slug", None) or workspace.name
        args.options["paths"] = {
            "root": workspace,
            # AbstractImageProcessor._debug_dir concatenates
            # `{prefix}_debug_<name>_<ts>` — debug dirs land as siblings of the sqlite.
            "debug_prefix": str(workspace / slug),
            "export": workspace / f"{slug}_export",
        }
        # Default DPI for capture if not set
        final_dpi = args.input_dpi if args.input_dpi is not None else 100.0
        args.options["general"]["input_dpi"] = final_dpi
        args.options["general"]["overwrite"] = False
        
    elif mode == "pdf":
        # For PDF mode, paths are determined per-file in the worker
        # But we initialize empty structure
        args.options["paths"] = {
            "output": None, 
            "raw": None,
            "layout": None,
            "dewarp": None,
            "debug": None
        }
        args.options["general"]["overwrite"] = True

    print_startup_info(workers=getattr(args, "workers", None))
    return args

def _option_map() -> dict:
    """Lazy registry view — adding a processor only needs its file in `aglaia/processors/`."""
    return _proc_registry.option_classes()

def load_pipeline_def(path):
    """Plain YAML load. The legacy `t:$dpi/…` template syntax was
    dropped in favour of per-processor unit-aware options (e.g.
    Binarizer's `window_mm` + `window_px`)."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading pipeline {path}: {e}")
        return None

def pipeline_step_descriptions(pipeline_def) -> dict:
    """Map ``{processor_name: (essential, full)}`` for the pipeline view.

    Builds each step's option object from its YAML and renders it via the
    processor's ``describe_options`` classmethod (no processor is
    constructed — so this never triggers PageDewarper's JAX/MLX init).
    Unknown processors (e.g. the Replay worker) and bad options are
    skipped. Keyed by processor name to match the timing view's row keys;
    on duplicate-named steps the last config wins."""
    import dataclasses as _dc
    out: dict[str, tuple[str, str]] = {}
    if not pipeline_def:
        return out
    for step in pipeline_def.get("pipeline", []):
        proc_name = step.get("processor") or step.get("name")
        info = _proc_registry.get_processor(proc_name)
        if info is None:
            continue
        valid = {f.name for f in _dc.fields(info.option_cls)}
        step_opts = {k: v for k, v in (step.get("options") or {}).items()
                     if k in valid}
        try:
            opts = info.option_cls(**step_opts)
            ess = info.processor_cls.describe_options(opts, "essential")
            full = info.processor_cls.describe_options(opts, "full")
        except Exception:
            continue
        out[proc_name] = (ess, full)
    return out


def _is_real_calibration(K) -> bool:
    """True for a real intrinsic (fx, fy in pixel units, hundreds-to-thousands).
    Identity / near-identity is the uncalibrated placeholder — passing it
    through underflows PageDewarper's focal_length to ~0.001."""
    if K is None:
        return False
    try:
        import numpy as _np
        K = _np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            return False
        fx, fy = K[0, 0], K[1, 1]
        if not (_np.isfinite(fx) and _np.isfinite(fy)):
            return False
        return fx >= 50.0 and fy >= 50.0  # pixel focal in real cameras
    except Exception:
        return False


def create_processing_chain(args, log_queue, queue_factory=multiprocessing.Queue, db_path=None):
    """
    Factory to create an IntegratedProcessingChain based on arguments and pipeline def.

    db_path: path to the project's SQLite file (M0). Required. If None, falls back to
             args.db_path or args.workspace_dir/<slug>.scanproj.sqlite.
    """
    from aglaia.assets import config_path as _bundled_config_path
    pipeline_path = (args.pipeline if args.pipeline
                     else _bundled_config_path("pipelines", "book_curved_x2.yaml"))

    if db_path is None:
        db_path = getattr(args, "db_path", None)
    if db_path is None:
        raise ValueError("create_processing_chain: db_path is required (M0 storage)")

    pipeline_def = load_pipeline_def(pipeline_path)
    if not pipeline_def:
        print("CRITICAL: Leading pipeline failed. Falling back to hardcoded empty chain.")
        return IntegratedProcessingChain([], 1, log_queue, db_path=db_path,
                                         paths=args.options.get("paths"), queue_factory=queue_factory)

    elements = []
    
    print(f"Building pipeline: {pipeline_def.get('name', 'Unknown')}")
    
    pipeline_list = pipeline_def.get("pipeline", [])
    if len(pipeline_list) > 99:
        raise ValueError(f"Pipeline has {len(pipeline_list)} steps. Maximum allowed is 99 to maintain consistent 2-digit padding.")

    for idx, step in enumerate(pipeline_list, 1):
        proc_name = step.get("processor")
        step_name = step.get("name", proc_name)

        # Any processor shall have an instance_name derived from the name in the yaml file
        # Prefix with NN_ for proper ordering
        instance_name = f"{idx:02d}_{slugify(step_name, separator='_')}"
        
        info = _proc_registry.get_processor(proc_name)
        if info is None:
            print(f"Warning: Unknown processor {proc_name} in pipeline step {step_name}. Skipping.")
            continue
        opt_class = info.option_cls

        # Get options from YAML
        step_opts = step.get("options", {})

        # Merge global CLI args into step options so flags like `--debug`
        # still reach individual processors.

        # Global Debug
        if args.debug:
            step_opts["debug"] = True
        # Inject slug-prefixed debug path: `<prefix>_debug_<name>_<ts>` sibling
        # of the .scanproj.sqlite file.
        paths = args.options.get("paths") or {}
        prefix = paths.get("debug_prefix") or paths.get("root")
        if prefix and "debug_dir" not in step_opts:
            step_opts["debug_dir"] = str(prefix)

        # Per-processor classmethod hook — each processor owns its runtime injection.
        if info.inject_step_options is not None:
            try:
                step_opts = info.inject_step_options(step_opts, args) or step_opts
            except Exception as e:
                print(f"Warning: {proc_name}.inject_step_options failed: {e}")

        # Drop unknown yaml keys. A stale option name would otherwise drop the
        # step silently, forwarding the previous buffer with no warning.
        import dataclasses as _dc
        _valid = {f.name for f in _dc.fields(opt_class)}
        _extras = {k: v for k, v in step_opts.items() if k not in _valid}
        if _extras:
            print(f"Warning: step '{step_name}' has unknown option(s) "
                  f"{list(_extras)}; ignoring (schema drift).")
            step_opts = {k: v for k, v in step_opts.items() if k in _valid}

        try:
            opts = opt_class(**step_opts)
            elements.append(SimpleChainElement(proc_name, opts, instance_name=instance_name))
        except Exception as e:
            print(f"Error configuring step {step_name}: {e}")
            continue

    num_workers = args.workers
    # Top-level `replay:` flag. Absent → default True (back-compat).
    # When True the chain still self-suppresses if the pipeline has no
    # replay-stamping step, so users can drop a no-warp pipeline through
    # without touching the flag.
    replay_enabled = bool(pipeline_def.get("replay", True))
    return IntegratedProcessingChain(
        elements, num_workers, log_queue, db_path=db_path,
        paths=args.options.get("paths"), queue_factory=queue_factory,
        replay_enabled=replay_enabled,
    )
