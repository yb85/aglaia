# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""ParamSpec dataclass + helpers.

Each processor declares its own `OPTIONS` dict (and `SUMMARY` string) as a
class attribute in `lib/processors/<Name>.py`. The runtime registry walks the
package, collects every `AbstractImageProcessor` subclass that declares
`OPTIONS`, and exposes the merged metadata via `lib.processors.registry`.
"""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParamSpec:
    kind: str                                   # enum|string|bool|bounded_int|bounded_float
    default: Any
    minimum: Optional[float] = None             # bounded_*
    maximum: Optional[float] = None             # bounded_*
    step: Optional[float] = None                # bounded_float
    choices: Optional[list[str]] = None         # enum
    help: str = ""
    advanced: bool = False                      # GUI: hide behind "show advanced" toggle
    # Cross-field visibility predicate. `{"method": ["wolf", "wolf++"]}`
    # means: this field is only rendered when the step's `method` option
    # equals one of "wolf" / "wolf++". Used by the Binarizer to hide
    # window/k for methods (otsu, bht, gray, none) that don't read them.
    visible_when: Optional[dict[str, list[Any]]] = None

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "default": self.default,
            "minimum": self.minimum, "maximum": self.maximum,
            "step": self.step, "choices": self.choices,
            "help": self.help, "advanced": self.advanced,
            "visible_when": self.visible_when,
        }


def _i(default, lo, hi, help_text="", *, advanced=False, visible_when=None):
    return ParamSpec("bounded_int", default, minimum=lo, maximum=hi,
                     help=help_text, advanced=advanced,
                     visible_when=visible_when)


def _f(default, lo, hi, step, help_text="", *, advanced=False, visible_when=None):
    return ParamSpec("bounded_float", default, minimum=lo, maximum=hi,
                     step=step, help=help_text, advanced=advanced,
                     visible_when=visible_when)


def _b(default, help_text="", *, advanced=False, visible_when=None):
    return ParamSpec("bool", default, help=help_text, advanced=advanced,
                     visible_when=visible_when)


def _e(default, choices, help_text="", *, advanced=False, visible_when=None):
    return ParamSpec("enum", default, choices=choices,
                     help=help_text, advanced=advanced,
                     visible_when=visible_when)


def _s(default, help_text="", *, advanced=False, visible_when=None):
    return ParamSpec("string", default, help=help_text, advanced=advanced,
                     visible_when=visible_when)


# Generic options every processor inherits from AbstractProcessorOption.
# Surfaced as a separate dict so the form renders the inherited knobs
# alongside the processor-specific OPTIONS. Currently empty: replay order
# is derived from processor REPLAY_TRAIT (see Replay engine), and debug
# etc. are routed by the chain rather than shown as per-step knobs.
COMMON_OPTION_SPECS: dict[str, ParamSpec] = {}


