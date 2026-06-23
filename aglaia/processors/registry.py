# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Plugin registry for image processors.

Adding a new processor:
  1. Drop `aglaia/processors/<NewProc>.py` containing a subclass of
     `AbstractImageProcessor` with:
        - `SUMMARY: str` — one-liner shown in the add-step menu
        - `OPTIONS: dict[str, ParamSpec]` — option specs for the UI
        - (optional) `OPTION_CLASS: type[AbstractProcessorOption]` —
          dataclass that wraps the YAML/CLI options. Defaults to a
          freshly synthesised dataclass built from `OPTIONS`.
        - (optional) classmethods `inject_step_options(step_opts, args)`
          and `desired_worker_count(args, all_steps)` for the
          special-case hooks that used to live in `Initializer`.
  2. Done. The registry auto-discovers the class on first access; the
     GUI, the pipeline loader, and the worker chain all read through
     it — no further imports, maps, or `if proc_name == "X"` branches.

This file is the single source of truth for "what processors exist". The
old `OPTION_SPECS`, `list_processors`, `PROCESSOR_TYPE_TO_OPTION_CLASS`,
`PROCESSOR_CLASS_MAP` are now thin views over `all_processors()`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Callable, Optional

from aglaia.processors.abstraction import AbstractImageProcessor, AbstractProcessorOption


@dataclass
class ProcessorInfo:
    """Everything the rest of the codebase needs to know about a processor."""

    name: str
    processor_cls: type[AbstractImageProcessor]
    option_cls: type[AbstractProcessorOption]
    options: dict[str, Any]                          # name → ParamSpec
    summary: str
    inject_step_options: Optional[Callable] = None   # (step_opts, args) → step_opts


_REGISTRY: dict[str, ProcessorInfo] = {}
_DISCOVERED = False

# Backward-compat processor names. Pipelines frozen in existing `.agl`
# projects (pipeline_versions table) + older user-edited YAMLs reference the
# pre-rename class name; without this alias the step silently vanishes from
# the chain (unknown processor → skipped) — e.g. a two-page spread stops
# being split. Aliases resolve for loading/instantiation but are hidden from
# the add-step menu so they don't show as duplicates.
_ALIASES: dict[str, str] = {"LayoutDetector": "PageDetector"}
_ALIAS_KEYS: set[str] = set()


def _apply_aliases() -> None:
    for alias, canonical in _ALIASES.items():
        if alias not in _REGISTRY and canonical in _REGISTRY:
            _REGISTRY[alias] = _REGISTRY[canonical]
            _ALIAS_KEYS.add(alias)


def _kind_to_pytype(kind: str) -> type:
    """Map ParamSpec.kind → Python type for synthesised dataclass fields."""
    return {
        "bool": bool,
        "string": str,
        "enum": str,
        "bounded_int": int,
        "bounded_float": float,
    }.get(kind, Any)


def _synthesize_option_class(name: str, options: dict[str, Any]) -> type:
    """Build an Options dataclass from a `name → ParamSpec` dict.

    Used when a processor doesn't declare its own `OPTION_CLASS`. Fields
    inherit from `AbstractProcessorOption` (debug, timeout_s, …) so
    callers don't have to special-case the common set."""
    fields = []
    for fname, spec in options.items():
        default = spec.default
        # Replicate the "use field(default_factory=...)" gymnastic
        # for mutable defaults so callers can mutate without state leak.
        if isinstance(default, (list, dict, set)):
            default_factory = lambda d=default: type(d)(d)
            fields.append((fname, _kind_to_pytype(spec.kind),
                           field(default_factory=default_factory)))
        else:
            fields.append((fname, _kind_to_pytype(spec.kind),
                           field(default=default)))
    cls = make_dataclass(f"{name}AutoOption", fields,
                         bases=(AbstractProcessorOption,))
    cls.__module__ = "aglaia.processors.registry"
    return cls


# Built-in processor modules — the authoritative list for PyInstaller-frozen
# builds, where the package directory doesn't exist on disk to walk. These are
# force-bundled via `collect_submodules("aglaia.processors")` in Aglaia.spec, so
# importing them always succeeds in the bundle. KEEP IN SYNC when adding a new
# built-in processor (the source-mode filesystem walk auto-discovers it for
# dev; only the frozen app needs it listed here — and `test_registry` guards
# that the two agree).
_BUILTIN_PROCESSOR_MODULES = (
    "DPIfixer", "SkewFinder", "PageDetector", "Binarizer",
    "PageDewarper", "TrapezoidalCorrection", "MarginSetter",
)


def _iter_processor_module_names(pkg) -> list[str]:
    """Module names to import for built-in processor discovery.

    The filesystem walk (`pkgutil.iter_modules`) works from source but yields
    NOTHING inside a PyInstaller bundle — the package directory doesn't exist
    on disk, the modules live in the PYZ archive. Earlier attempts to detect
    "frozen" and read the PYZ table-of-contents proved unreliable (the bundle
    skipped the branch with no diagnostics), so we stop guessing: we ALWAYS
    union the filesystem walk with the hard-coded built-in list. The built-ins
    are force-bundled via `collect_submodules("aglaia.processors")`, so importing
    them succeeds in the bundle; from source the walk already finds them and
    the union just dedups. Registration downstream is idempotent.

    A new built-in must be added to ``_BUILTIN_PROCESSOR_MODULES`` to appear in
    frozen builds — ``test_registry_frozen`` enforces that the list stays in
    sync with what the source walk discovers.
    """
    fs_names = [m for _, m, _ in pkgutil.iter_modules(pkg.__path__)]
    # dict.fromkeys preserves order while de-duplicating.
    return list(dict.fromkeys(fs_names + list(_BUILTIN_PROCESSOR_MODULES)))


def _discover_once() -> None:
    """Discover built-in processors (frozen-safe, see
    :func:`_iter_processor_module_names`), import each, and register every
    concrete `AbstractImageProcessor` subclass that declares `OPTIONS`. Pure
    classes without `OPTIONS` are skipped — they're internal helpers
    (e.g. abstract bases, utility processors not exposed to the UI).
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    import aglaia.processors as _pkg
    for mod_name in _iter_processor_module_names(_pkg):
        if mod_name.startswith("_") or mod_name in {"abstraction", "registry",
                                                     "option_specs", "utils",
                                                     "geometry"}:
            continue
        try:
            module = importlib.import_module(f"aglaia.processors.{mod_name}")
        except Exception as e:
            print(f"[registry] failed to import aglaia.processors.{mod_name}: {e}")
            continue
        _register_from_module(module)

    # User drop-in processor plugins (<APP_DATA>/plugins/processors). Only
    # files the user acknowledged in the trust gate are imported — import
    # == code execution, so unacknowledged files are never touched here.
    # Spawned workers re-run this same path, which is why discovery is
    # DB-driven (accepted list) rather than popup-driven.
    try:
        from aglaia.app_data import plugins as _plugins
        for mod_name in _plugins.import_accepted(_plugins.KIND_PROCESSORS):
            mod = sys.modules.get(mod_name)
            if mod is not None:
                _register_from_module(mod)
    except Exception as e:  # noqa: BLE001 — plugin layer optional / best-effort
        print(f"[registry] processor plugin discovery skipped: {e}")

    _apply_aliases()
    # One-line summary — cheap support breadcrumb (esp. for frozen builds where
    # a discovery regression silently empties the pipeline).
    print(f"[registry] {len(_REGISTRY)} processors registered", flush=True)
    _DISCOVERED = True


def _register_from_module(module) -> None:
    """Register every concrete `AbstractImageProcessor` subclass *defined
    in* ``module`` that declares `OPTIONS`."""
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is AbstractImageProcessor:
            continue
        if not issubclass(obj, AbstractImageProcessor):
            continue
        # Gate on OPTIONS declared in the class's OWN body: the ABC now
        # ships a default `OPTIONS = {}` (so `hasattr` is always true), but
        # a real processor declares its own — even an empty one (no tunable
        # params). Helper / intermediate base classes don't, so stay out.
        if "OPTIONS" not in obj.__dict__:
            continue
        # Avoid double-registering imported subclasses.
        if obj.__module__ != module.__name__:
            continue
        name = getattr(obj, "REGISTRY_NAME", None) or obj.__name__
        if name in _REGISTRY:
            continue
        option_cls = getattr(obj, "OPTION_CLASS", None) \
            or _synthesize_option_class(name, obj.OPTIONS)
        _REGISTRY[name] = ProcessorInfo(
            name=name,
            processor_cls=obj,
            option_cls=option_cls,
            options=dict(obj.OPTIONS),
            summary=getattr(obj, "SUMMARY", ""),
            inject_step_options=getattr(obj, "inject_step_options", None),
        )


def all_processors() -> dict[str, ProcessorInfo]:
    _discover_once()
    return {n: i for n, i in _REGISTRY.items() if n not in _ALIAS_KEYS}


def get_processor(name: str) -> Optional[ProcessorInfo]:
    _discover_once()
    return _REGISTRY.get(name)


def option_specs() -> dict[str, dict[str, Any]]:
    """`{processor_name: {option_name: ParamSpec}}` — used by the GUI
    form generator and the web pipeline editor."""
    _discover_once()
    return {name: dict(info.options) for name, info in _REGISTRY.items()}


def list_summaries() -> list[dict]:
    """Used by the add-step menu in PipelineEditorWidget."""
    _discover_once()
    return [{"name": name, "summary": info.summary}
            for name, info in _REGISTRY.items() if name not in _ALIAS_KEYS]


def processor_classes() -> dict[str, type[AbstractImageProcessor]]:
    """Used by IntegratedProcessingChain when instantiating workers."""
    _discover_once()
    return {name: info.processor_cls for name, info in _REGISTRY.items()}


def option_classes() -> dict[str, type[AbstractProcessorOption]]:
    """Used by Initializer when materialising YAML options into dataclasses."""
    _discover_once()
    return {name: info.option_cls for name, info in _REGISTRY.items()}
