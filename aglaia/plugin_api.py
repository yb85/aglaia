# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The public surface a plugin may use — and the only one (#126).

A plugin imports from here and from nowhere else under ``aglaia``. That is not
politeness: an unbounded surface means every internal rename breaks somebody's
plugin, and a reviewer reading a PR cannot tell what the code is allowed to
touch. One façade fixes both. `docs/plugin-store.md` §5 is the contract.

    from aglaia.plugin_api import AbstractImageProcessor, Destination, …

**Versioning.** `API_VERSION` is semver-ish and independent of the app's own
version: a plugin declares `api = 1` in its manifest and the host refuses a
major it does not implement, naming both. Adding a name here is a minor bump.
Removing one, or changing a signature, is a major — expected to be rare and
loud.

Three kinds of plugin exist:

* **processors** — a pipeline step. Subclass `AbstractImageProcessor`.
* **ocr** — a recognition engine. Subclass `OcrEngine`, decorate `@register_ocr_engine`.
* **destinations** — somewhere a finished export goes. Subclass `Destination`.

Each gets a `PluginContext` (settings, secrets, a scratch dir, a log line)
assigned to ``self.ctx`` by the host after construction — so a plugin written
before contexts existed keeps working, and one that needs state does not have
to reach for `sqlite3` or `keyring`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── version ───────────────────────────────────────────────────────────
#: Major bumps break plugins; minors only add. See the module docstring.
API_VERSION = 1

# ── the data types a plugin handles ───────────────────────────────────
from aglaia.ImageBuffer import ImageBuffer, ImageType          # noqa: E402
from aglaia.Status import Status                               # noqa: E402
from aglaia.processors.abstraction import (                    # noqa: E402
    AbstractImageProcessor, AbstractProcessorOption, ReplayTrait,
)
from aglaia.processors.utils import (                          # noqa: E402
    is_binary, to_bw, to_gray, to_rgb,
)
# Erase masks (`meta["erase"]`): a processor that finds something the page
# should not contain says so with `add_erase`, and the host removes it —
# through the geometry, out of the binarizer's statistics, and white in the
# output. See `docs/processors.md`. Exposed here because a plugin must not
# reach into `aglaia.processors` for it; the import scan refuses that, and it
# refused the first draft of StampRemover for exactly this.
from aglaia.processors.erase import (                          # noqa: E402
    add as add_erase, get as get_erase,
)
from aglaia.workers.ocr.engine import (                        # noqa: E402
    OcrEngine, OcrLine, OcrResult, register as register_ocr_engine,
)

# ── option spec helpers, under names that read as English ─────────────
# The one-letter internal names (`_i`, `_f`, …) are fine inside a processor
# module a reader already has open; in someone else's plugin they are noise.
from aglaia.processors.option_specs import (                   # noqa: E402
    _b as option_bool, _e as option_enum, _f as option_float,
    _i as option_int, _s as option_str,
)

# ── the context the host hands over ───────────────────────────────────
from aglaia.app_data.plugin_ctx import (                       # noqa: E402
    PluginConfig, PluginContext, PluginSecrets,
)


# ── destinations ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    """One setting a destination needs, described so the GUI can render it
    without knowing what the destination is.

    Same idea as the OCR tab reading engine capability flags instead of
    hard-coding engine names: the host draws a form from data, so a plugin
    nobody anticipated still gets a proper settings panel.

    `kind` is one of ``"str"``, ``"int"``, ``"bool"``, ``"choice"``,
    ``"secret"``. A ``secret`` field is rendered masked and stored in
    `ctx.secrets`; every other kind is stored in `ctx.config`.
    """

    key: str
    label: str
    kind: str = "str"
    default: Any = ""
    help: str = ""
    required: bool = False
    choices: tuple[str, ...] = ()
    #: For "str": a hint the GUI can show greyed inside an empty box.
    placeholder: str = ""


@dataclass
class BookMeta:
    """What a destination may want to know about the document it is sending.

    Every field is optional because every field genuinely can be missing — a
    scan of an unidentified offprint has no ISBN and may have no author. A
    destination decides what to do with the gaps; it is not the host's place
    to invent them."""

    title: str = ""
    author: str = ""
    publisher: str = ""
    language: str = ""
    year: str = ""
    pages: int = 0
    isbn: str = ""
    categories: str = ""
    #: The .agl this came from, for a destination that wants to link back.
    project_path: str = ""

    def filled(self) -> dict[str, str]:
        """Only the fields that actually have a value — the shape most upload
        APIs want, where an empty field means "erase this" rather than
        "unknown"."""
        out: dict[str, str] = {}
        for k in ("title", "author", "publisher", "language", "year",
                  "isbn", "categories"):
            v = str(getattr(self, k) or "").strip()
            if v:
                out[k] = v
        if self.pages:
            out["pages"] = str(int(self.pages))
        return out


@dataclass(frozen=True)
class SendResult:
    """The outcome of one send, in a shape a toast and a log line can both use.

    `ok` is not a boolean verdict on the user's intent: a document the
    destination already had is `ok=True, already_there=True`. Flattening that
    into a failure trains people to ignore the failure."""

    ok: bool
    message: str
    url: str = ""
    already_there: bool = False
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    """The outcome of a credentials/reachability test.

    Separate from `send` so a user can prove a configuration without pushing a
    book into a library, and so "host unreachable", "credentials rejected" and
    "account cannot write" stay three different messages — they have three
    different fixes, and one "failed" for all of them is useless."""

    ok: bool
    message: str
    detail: dict = field(default_factory=dict)


class Destination:
    """Somewhere a finished export goes: a server, a library, a mailbox.

    Subclass, set `name`, declare your fields, implement `send` (and `check`
    if the destination can be tested), and the host does the rest: it renders
    the settings, keeps the secrets out of the config file, offers the
    destination wherever an export lands, and reports what came back.

    Deliberately **not** an HTTP abstraction. The three first-party
    destinations are a calibre server (raw body, parameters in the URL path,
    Basic/Digest), an SMTP mailbox (a MIME attachment) and a corpus API
    (multipart fields, an API-key header). A common `send()` over those would
    be a signature and a shrug. What is common — the settings schema, the
    credential storage, the check/send split, the result shape — is what lives
    here.
    """

    #: Registry key. Unique, non-empty, matches the plugin slug by convention.
    name: str = ""
    #: Shown in menus.
    display: str = ""
    #: One line under the title in the destinations list.
    description: str = ""
    #: Export formats this destination accepts, by extension.
    accepts: tuple[str, ...] = ("pdf",)
    #: Rendered in the settings panel; stored in `ctx.config`.
    CONFIG_FIELDS: tuple[Field, ...] = ()
    #: Rendered masked; stored in `ctx.secrets`. Declaring any of these means
    #: the manifest must declare `secrets = true`.
    SECRET_FIELDS: tuple[Field, ...] = ()

    #: Set by the host after construction. Never construct one yourself.
    ctx: Optional[PluginContext] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name") and cls.name == "":
            import warnings
            warnings.warn(
                f"{cls.__name__} does not set a `name` — it cannot be "
                f"registered as a destination.", stacklevel=2)

    # ── settings, read through the context ────────────────────────────
    def conf(self, key: str, default: Any = None) -> Any:
        """This destination's setting, falling back to the field's declared
        default before the caller's — so a plugin reads `self.conf("port")`
        and gets 587, not None, on a fresh install."""
        if self.ctx is not None:
            got = self.ctx.config.get(key, None)
            if got is not None and got != "":
                return got
        for f in self.CONFIG_FIELDS:
            if f.key == key:
                return f.default if default is None else default
        return default

    def secret(self, key: str) -> str:
        if self.ctx is None or self.ctx.secrets is None:
            return ""
        return self.ctx.secrets.get(key) or ""

    def missing_settings(self) -> list[str]:
        """Labels of required fields that are still empty. The host shows
        these instead of letting a send fail on the far end for a reason the
        user could have been told locally."""
        out: list[str] = []
        for f in self.CONFIG_FIELDS:
            if f.required and not str(self.conf(f.key) or "").strip():
                out.append(f.label)
        for f in self.SECRET_FIELDS:
            if f.required and not self.secret(f.key):
                out.append(f.label)
        return out

    # ── what a subclass implements ────────────────────────────────────
    def check(self) -> CheckResult:
        """Prove the configuration without sending anything. Optional."""
        return CheckResult(True, "No check implemented for this destination.")

    def send(self, path: Path, meta: BookMeta) -> SendResult:
        raise NotImplementedError


# ── plugin-owned windows ──────────────────────────────────────────────

@dataclass(frozen=True)
class PluginWindow:
    """A window a plugin contributes, listed under *Plugins* in the menu bar.

    Some plugins need a workspace, not a settings form: a stamp library wants
    to show snippets and let you trace a polygon on one. `Field` cannot
    express that, and inventing a UI description language that could would be
    a worse answer than letting the plugin build the widget.

    So `factory(ctx) -> QWidget` is a real Qt widget, and a plugin that wants
    one declares `ui = true` in its manifest — which puts `PySide6` on its
    import allow-list and puts "adds a window" in the install dialog.

    **This changes nothing about the threat model** (`docs/plugin-store.md`
    §1): a plugin already runs in-process with the host's privileges, and
    could import Qt whether or not the scan allowed it. What it does change is
    what a *reviewer* looks for, and the guidelines say it: a plugin that can
    draw can draw something that looks like Aglaïa asking for a password. UI
    plugins get read harder.
    """

    key: str
    title: str
    factory: Any            # Callable[[PluginContext], QWidget]
    menu: str = "Plugins"
    #: Shown beside the entry; keep it to a few words.
    summary: str = ""


#: Populated by `@register_window`, keyed by the owning plugin's slug.
WINDOW_REGISTRY: dict[str, list[PluginWindow]] = {}


def register_window(slug: str, window: PluginWindow) -> PluginWindow:
    """Contribute a window. The host groups entries under the plugin's name,
    so two plugins cannot collide in the menu even with the same title."""
    WINDOW_REGISTRY.setdefault(str(slug), []).append(window)
    return window


#: Populated by `@register_destination`.
DESTINATION_REGISTRY: dict[str, type[Destination]] = {}


def register_destination(cls: type[Destination]) -> type[Destination]:
    """Class decorator that puts a `Destination` in the registry."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} has no `name`; cannot register")
    DESTINATION_REGISTRY[cls.name] = cls
    return cls


__all__ = [
    "API_VERSION",
    # data types
    "ImageBuffer", "ImageType", "Status",
    # processors
    "AbstractImageProcessor", "AbstractProcessorOption", "ReplayTrait",
    "option_bool", "option_enum", "option_float", "option_int", "option_str",
    "to_gray", "to_rgb", "to_bw", "is_binary",
    "add_erase", "get_erase",
    # ocr
    "OcrEngine", "OcrResult", "OcrLine", "register_ocr_engine",
    # destinations
    "Destination", "DESTINATION_REGISTRY", "register_destination",
    # windows
    "PluginWindow", "WINDOW_REGISTRY", "register_window",
    "Field", "BookMeta", "SendResult", "CheckResult",
    # context
    "PluginContext", "PluginConfig", "PluginSecrets",
]
