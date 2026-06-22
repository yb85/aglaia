# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from abc import ABC, abstractmethod
import dataclasses
import warnings
from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, ClassVar, Union, TYPE_CHECKING
import cv2
import numpy as np
from lib.ImageBuffer import ImageBuffer

if TYPE_CHECKING:
    import numpy as np

    from lib.processors.option_specs import ParamSpec
    from lib.processors.replay_transform import (
        AffineTransform, ReplayContext, SampleMapTransform,
    )


class ReplayTrait(Enum):
    """How a processor behaves under the end-of-chain replay algebra.

    Replay reorders + fuses steps to minimise interpolation/quantisation:
    contiguous COORDINATE maps compound into one remap, PIXEL_VALUE ops are
    pushed as late as possible (applied once on the final geometry), and ROI
    ops are fixed barriers nothing reorders across. See memory
    project_replay_trait_algebra.md.
    """
    # x'=f(x) coordinate remap, value-preserving (skew, keystone, dewarp,
    # DPI resample). Contiguous COORDINATE maps fuse into a single warp.
    COORDINATE = "coordinate"
    # v'=g(v, neighbourhood) value op (binarize, morphology). Pushed late;
    # boundary/missing pixels read from the carried ROI mask.
    PIXEL_VALUE = "pixel_value"
    # Changes the domain/region (layout crop+branch, margin crop+pad). A
    # fixed barrier + per-segment source anchor; its image must persist.
    ROI = "roi"


# Option fields shared by every processor (debug, replay, timeout).
# They're plumbing, not algorithm parameters, so the human-readable
# descriptions hide them.
_COMMON_IO_FIELDS = frozenset({
    "debug", "debug_dir", "timeout_s",
})


def _fmt_value(v: Any) -> str:
    """Compact, human-friendly rendering of an option value."""
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        # Trim trailing zeros: 1.0 → "1", 0.50 → "0.5".
        return f"{v:g}"
    if v is None:
        return "auto"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def render_param_description(options: Any,
                            essential_fields: tuple = (),
                            verbosity: str = "essential") -> str:
    """Render an option dataclass into a parameter description string.

    ``essential`` → a compact one-liner (``a 1 · b on``) of the
    ``essential_fields`` (falling back to the first few algorithm fields);
    ``full`` → every algorithm field, one per line (``name: value``).
    Common I/O / debug fields are always hidden. Used by both
    ``AbstractImageProcessor.describe_parameters`` and the pipeline view's
    no-instantiation fallback."""
    if not dataclasses.is_dataclass(options):
        return ""
    algo_fields = [f.name for f in dataclasses.fields(options)
                   if f.name not in _COMMON_IO_FIELDS]
    if verbosity == "essential":
        names = [n for n in (tuple(essential_fields) or tuple(algo_fields[:3]))
                 if hasattr(options, n)]
        parts = [f"{n} {_fmt_value(getattr(options, n))}" for n in names]
        return " · ".join(parts)
    # full
    lines = [f"{n}: {_fmt_value(getattr(options, n))}" for n in algo_fields]
    return "\n".join(lines) if lines else "no tunable parameters"

@dataclass
class AbstractProcessorOption:
    """
    Abstract base dataclass for processor options.
    Defines the common plumbing fields (debug, timeout).
    """
    # Per-processor debug. When True, processor dumps overlays/intermediate
    # images under `debug_dir / debug_<processor_name>_<timestamp>/`.
    debug: bool = False
    # Project root injected by the chain factory (Initializer).
    debug_dir: Optional[str] = None
    # Per-processor wall-clock budget in seconds. 0 = disabled. Enforced
    # via SIGALRM in the worker; on expiry the chain logs an error and
    # treats the step as failed (passthrough fallback). Useful for
    # JAX/XLA hangs in PageDewarper.
    timeout_s: float = 0.0


class AbstractImageProcessor(ABC):
    """Abstract base class for all image processors.

    Plugin contract — a processor exposed to the UI / pipeline MUST set the
    two class attributes the registry reads (`SUMMARY`, `OPTIONS`) and
    implement `process()`. The remaining class attributes are optional
    hooks; they are declared here (rather than only duck-typed by the
    registry / Initializer) so a plugin author sees the whole contract in
    one place. `__init_subclass__` validates the contract at import time.

    Output-format contract: `process()` returns one `ImageBuffer`, or a
    `list[ImageBuffer]` to branch (>1 element → branch point), or `None`
    to stop this branch. `run()` is the validated entry the chain calls —
    it enforces that shape so a malformed plugin fails loudly, in context.
    """

    name: str = "AbstractImageProcessor" # Static name field

    # ── plugin contract (read by lib.processors.registry) ──────────────
    # REQUIRED for a UI-exposed processor:
    SUMMARY: ClassVar[str] = ""                              # one-liner for the add-step menu
    OPTIONS: ClassVar[Dict[str, "ParamSpec"]] = {}          # option_name → ParamSpec
    # Replay behaviour (COORDINATE / PIXEL_VALUE / ROI). Drives the replay
    # engine's reorder+fuse algebra; None = treat as an opaque barrier (safe
    # default — replay falls back to re-processing, never reorders across it).
    REPLAY_TRAIT: ClassVar[Optional[ReplayTrait]] = None
    # OPTIONAL registry hooks (default = sensible fallback):
    REGISTRY_NAME: ClassVar[Optional[str]] = None           # registry key; default = class name
    OPTION_CLASS: ClassVar[Optional[type]] = None           # options dataclass; default = synthesised from OPTIONS

    # OPTIONAL declaration — the metadata keys this processor STAMPS into
    # the output buffer's `meta` dict, beyond the `replay_kind` /
    # `replay_params` plumbing the replay engine reads. Maps key → one-line
    # meaning. Purely documentary: it lets a downstream processor or plugin
    # author discover what upstream steps make available (e.g. a despeckler
    # reading `char_h_frac`), and feeds the generated processor reference.
    # Declaring a key here does not stamp it — the processor still writes
    # `buffer.meta[key]` itself; this is the contract, not the mechanism.
    PROVIDES_META: ClassVar[Dict[str, str]] = {}

    # Subclasses list the parameter field names that matter most — shown
    # in the pipeline card's one-line "essential" description. Empty →
    # the base falls back to the first few algorithm fields.
    _ESSENTIAL_PARAMS: ClassVar[tuple] = ()

    # ── replay contract (read by lib.workers.Replay) ───────────────────
    # A processor self-describes how it re-applies in end-of-chain replay,
    # so the engine never needs to special-case it (and a plugin joins
    # replay with zero edits to Replay.py). Which method you implement is
    # dictated by REPLAY_TRAIT:
    #
    #   COORDINATE   → replay_transform(): describe the transform as a composable
    #                  geometric primitive (AffineTransform / SampleMapTransform). The
    #                  engine fuses contiguous COORDINATE warps into one
    #                  interpolation. Touch NO pixels here.
    #   PIXEL_VALUE  → apply_replay(): act on pixels (binarise, morphology).
    #   ROI          → apply_replay(): act on pixels (crop+pad margin).
    #
    # The forward pass stamps the needed values into the output's
    # meta["replay_params"]; both methods receive that dict.
    @classmethod
    def replay_transform(cls, params: dict,
                    in_wh: "tuple[int, int]") -> "AffineTransform | SampleMapTransform":
        """COORDINATE processors only: the composable warp for an input of
        size ``in_wh = (w, h)``. Must be analytic (params + size, no pixels)."""
        raise NotImplementedError(
            f"{cls.__name__} is REPLAY_TRAIT.COORDINATE but has no replay_transform()")

    @classmethod
    def apply_replay(cls, buf: "np.ndarray", mask: "np.ndarray",
                     params: dict, ctx: "ReplayContext",
                     ) -> "tuple[np.ndarray, np.ndarray]":
        """PIXEL_VALUE / ROI processors only: re-apply the step to (buf, mask)
        and return the new (buf, mask)."""
        raise NotImplementedError(
            f"{cls.__name__} has no apply_replay()")

    def __init_subclass__(cls, **kwargs):
        """Validate the plugin contract when a processor class is defined.

        Only classes that declare a non-empty `OPTIONS` in their own body
        are UI-exposed processors (the registry uses the same gate); helper
        / intermediate base classes are skipped. Warnings (not exceptions)
        so one malformed plugin can't abort discovery of the rest."""
        super().__init_subclass__(**kwargs)
        own = cls.__dict__
        if "OPTIONS" not in own:
            return  # helper / abstract subclass — not registered
        opts = own.get("OPTIONS") or {}
        if not getattr(cls, "SUMMARY", ""):
            warnings.warn(
                f"{cls.__name__} declares OPTIONS but no SUMMARY — it will "
                f"show a blank line in the add-step menu.", stacklevel=2)
        oc = getattr(cls, "OPTION_CLASS", None)
        if oc is not None and dataclasses.is_dataclass(oc):
            valid = {f.name for f in dataclasses.fields(oc)}
            extra = sorted(set(opts) - valid)
            if extra:
                warnings.warn(
                    f"{cls.__name__}: OPTIONS keys {extra} have no matching "
                    f"field on {oc.__name__}; Initializer will drop them.",
                    stacklevel=2)

    def __init__(self, options: AbstractProcessorOption):
        self.options = options
        self._debug_dir_cache: Optional[Path] = None
        # Per-call structured stats — populated at the end of process()
        # by each subclass, drained + emitted by the chain worker as
        # part of the unified op-log line. Keep keys short, values JSON-
        # serialisable. See ``lib/workers/oplog.py``.
        self.last_stats: dict = {}

    @abstractmethod
    def process(self, buffer: ImageBuffer) -> "Union[ImageBuffer, List[ImageBuffer], None]":
        """Process the input ImageBuffer and return the result.

        Return one ImageBuffer, a list of ImageBuffers to branch, or None
        to stop this branch. Must be implemented by subclasses. The chain
        calls :meth:`run`, not this directly.
        """
        pass

    def run(self, buffer: ImageBuffer) -> "Union[ImageBuffer, List[ImageBuffer], None]":
        """Validated entry point the chain calls.

        Delegates to :meth:`process` then enforces the output-format
        contract — the result must be an ImageBuffer, a list of
        ImageBuffers, or None — so a misbehaving plugin fails with a clear
        error here rather than corrupting the chain downstream."""
        out = self.process(buffer)
        if out is None:
            return None
        candidates = out if isinstance(out, list) else [out]
        for c in candidates:
            if not isinstance(c, ImageBuffer):
                raise TypeError(
                    f"{self.name}.process() must return ImageBuffer, "
                    f"list[ImageBuffer], or None; got {type(c).__name__}")
        return out

    def replay(self, buffer: ImageBuffer) -> "Union[ImageBuffer, List[ImageBuffer], None]":
        """Re-derive this step's output during the end-of-chain replay pass.

        Default = re-run :meth:`process` (correct for any step; trivial steps
        like DPIfixer's resample need nothing more). Override when a cheaper
        or higher-quality reconstruction exists — e.g. a geometric processor
        (PageDewarper, TrapezoidalCorrection, SkewFinder) that can express
        its effect as a single warp the replay engine fuses with its
        neighbours so the final image takes only one interpolation."""
        return self.process(buffer)

    def clone(self) -> 'AbstractImageProcessor':
        return deepcopy(self)

    # ── human-readable descriptions (pipeline view / logging) ──────────

    @classmethod
    def describe_options(cls, options: Any,
                         verbosity: str = "essential") -> str:
        """Describe an option object WITHOUT constructing the processor.

        This is what the pipeline view calls — building a processor just to
        read its parameters would run heavy ``__init__`` work (PageDewarper
        spins up JAX/MLX). The base introspects the option dataclass via
        ``_ESSENTIAL_PARAMS``; subclasses with derived/conditional params
        (e.g. Binarizer's per-family window/k) override this classmethod."""
        return render_param_description(
            options, cls._ESSENTIAL_PARAMS, verbosity)

    def describe_parameters(self, verbosity: str = "essential") -> str:
        """Describe this processor's active parameters.

        ``verbosity='essential'`` → a compact one-liner for the pipeline
        card (shown under the name + timings); ``'full'`` → every tunable
        parameter, one per line, revealed when the card is expanded.
        Delegates to :meth:`describe_options` so instance and class paths
        agree."""
        return type(self).describe_options(self.options, verbosity)

    def describe_result(self) -> str:
        """One-line summary of the LAST processing call, for logging.

        Formats ``self.last_stats`` (populated by ``process()`` and also
        drained by the chain worker's op-log). Empty string when the step
        recorded no stats."""
        if not self.last_stats:
            return ""
        return " · ".join(f"{k}={_fmt_value(v)}"
                          for k, v in self.last_stats.items())

    def debug_enabled(self) -> bool:
        return bool(getattr(self.options, "debug", False))

    def _debug_dir(self) -> Optional[Path]:
        """Lazily create `<prefix>_debug_<name>_<timestamp>/` on first
        call. `<prefix>` is the project-slug-prefixed path injected by
        the chain factory (e.g. `~/Desktop/myproj`), so the
        resulting folder sits as a sibling of the `.scanproj.sqlite`
        file rather than inside a project folder.
        Returns None if debug disabled or no prefix injected."""
        if not self.debug_enabled():
            return None
        if self._debug_dir_cache is not None:
            return self._debug_dir_cache
        prefix = getattr(self.options, "debug_dir", None)
        if not prefix:
            return None
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = Path(f"{prefix}_debug_{self.name}_{ts}")
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        self._debug_dir_cache = d
        return d

    def debug_save(self, img: np.ndarray, tag: str, img_buf: Optional[ImageBuffer] = None) -> None:
        """Write `img` under the processor's debug dir with a stable name.
        Filename: `<step>_snap{NN}_br{X}_<tag>.png`."""
        d = self._debug_dir()
        if d is None:
            return
        parts = []
        if img_buf is not None:
            if img_buf.scan_id is not None:
                parts.append(f"scan{int(img_buf.scan_id):02d}")
            if img_buf.branch_label:
                parts.append(f"br{img_buf.branch_label}")
        parts.append(tag)
        fname = "_".join(parts) + ".png"
        try:
            cv2.imwrite(str(d / fname), img)
        except Exception as e:
            print(f"[{self.name}] debug_save({fname}) failed: {e}")
