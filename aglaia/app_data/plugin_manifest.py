# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Reading a plugin's manifest, and looking at what it imports (#128).

Two jobs, both of them "decide whether this is even a plugin before anything
of it runs":

* `parse_manifest` turns `aglaia-plugin.toml` into a validated record, or
  says exactly what is wrong with it.
* `scan_imports` walks the entry module's AST and reports what it imports,
  split into allowed / refused / undeclared / review-required.

The scan is a **lint, not a sandbox**. It reads `import` statements; it cannot
see `__import__(base64.b64decode(...))`. `docs/plugin-store.md` §1 is the
threat model and says so plainly. What the scan is good for is real all the
same: it catches accidents, it makes a reviewer's job finite, and — because
`keyring`, `sqlite3` and `os` are refused outright — it turns "this plugin
went around the context API" from something you would have to notice into
something the tooling says out loud.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .plugin_ctx import SLUG_RE

KINDS = ("processors", "ocr", "destinations")

#: The plugin API major this build implements. A manifest asking for another
#: is refused with both numbers named, rather than loaded and left to fail
#: somewhere less obvious.
from aglaia.plugin_api import API_VERSION  # noqa: E402

#: Standard-library modules a plugin may import. Deliberately short: what a
#: plugin legitimately needs from `os`, `sqlite3` or `keyring` it gets through
#: `PluginContext` instead, which is the entire point of having one.
STDLIB_ALLOWED = frozenset({
    "__future__",   # `from __future__ import annotations` is universal
    "abc", "base64", "collections", "contextlib", "dataclasses", "datetime",
    "enum", "functools", "hashlib", "io", "itertools", "json", "logging",
    "math", "pathlib", "re", "statistics", "textwrap", "time", "typing",
    "unicodedata", "uuid", "warnings", "email", "ssl", "string",
    "difflib", "bisect", "heapq", "copy", "fractions", "decimal", "random",
})

#: Standard-library modules that open a connection. Allowed, but only with
#: `capabilities.network = true` — the same rule third-party `httpx` follows,
#: because a user consenting to "network access" should not have to know
#: which library the plugin happened to pick.
STDLIB_NETWORK = frozenset({"smtplib", "imaplib", "poplib"})

#: Submodules of an otherwise-refused package that are themselves harmless.
#: `urllib` is refused because of `urllib.request`; `urllib.parse` is string
#: manipulation and is exactly what a plugin building a URL should use
#: instead of formatting one by hand.
SUBMODULE_ALLOWED = frozenset({"urllib.parse"})

#: Refused outright. Each one has a `PluginContext` replacement, or no
#: business in a plugin at all.
STDLIB_REFUSED = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "importlib",
    "pickle", "multiprocessing", "threading", "sqlite3", "keyring",
    "tempfile", "glob", "runpy", "marshal", "shelve", "signal", "pty",
    "webbrowser", "http", "urllib", "ftplib", "telnetlib", "xmlrpc",
})

#: Third-party modules the host already ships. A plugin may declare any of
#: these in `[requires].imports`; anything else is refused, because **no
#: plugin ever installs a dependency** — that single rule removes the whole
#: supply-chain surface, and is why this list can be closed rather than open.
THIRD_PARTY_ALLOWED = frozenset({
    "numpy", "cv2", "scipy", "PIL", "yaml", "httpx",
    # Only meaningful with `ui = true`; declaring it without is refused the
    # same way `httpx` is refused without `network`.
    "PySide6",
})

#: Only with `network = true` in the manifest.
THIRD_PARTY_NETWORK = frozenset({"httpx"})

#: Not refused — flagged. Each has legitimate uses, and a blanket ban would
#: be both leaky (they are reachable by other names) and annoying. A reviewer
#: is told to look.
REVIEW_CALLS = frozenset({"eval", "exec", "compile", "__import__", "globals",
                          "vars", "breakpoint", "getattr", "setattr"})

_VERSION_RE = re.compile(r"^\d+(\.\d+){0,3}([-+][0-9A-Za-z.-]+)?$")


@dataclass
class Manifest:
    slug: str
    kind: str
    name: str
    version: str
    summary: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    entry: str = ""
    requires_aglaia: str = ""
    requires_python: str = ""
    api: int = 1
    imports: tuple[str, ...] = ()
    config: bool = False
    secrets: bool = False
    network: bool = False
    files: bool = False
    ui: bool = False

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"config": self.config, "secrets": self.secrets,
                "network": self.network, "files": self.files, "ui": self.ui}

    def declared(self) -> list[str]:
        """Capability names that are on, for the install dialog."""
        labels = {"config": "its own settings",
                  "secrets": "stores secrets in your keychain",
                  "network": "network access",
                  "files": "reads/writes files outside its own folder",
                  "ui": "adds a window to the Plugins menu"}
        return [labels[k] for k, v in self.capabilities.items() if v]


class ManifestError(ValueError):
    """A manifest that cannot be trusted to describe itself."""


def parse_manifest(path: Path, *, kind: Optional[str] = None,
                   expect_slug: Optional[str] = None) -> Manifest:
    """Read and validate one `aglaia-plugin.toml`.

    `kind` comes from the directory, never from the file — the directory is
    what decides, because the slug also decides the keychain namespace and the
    settings file, and those are not things to take a plugin's word for."""
    import tomllib
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"no manifest at {path}")
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except Exception as e:
        raise ManifestError(f"manifest is not valid TOML: {e}") from e

    plugin = raw.get("plugin") or {}
    requires = raw.get("requires") or {}
    caps = raw.get("capabilities") or {}

    slug = str(plugin.get("slug") or "")
    if not SLUG_RE.match(slug):
        raise ManifestError(
            f"slug {slug!r} must be 3-40 chars of lowercase letters, digits "
            f"and hyphens, not starting or ending with a hyphen")
    if expect_slug is not None and slug != expect_slug:
        raise ManifestError(
            f"manifest says slug={slug!r} but it lives in a directory called "
            f"{expect_slug!r} — the directory decides, so this is ambiguous")
    if kind is not None and kind not in KINDS:
        raise ManifestError(f"unknown plugin kind {kind!r}")

    version = str(plugin.get("version") or "")
    if not _VERSION_RE.match(version):
        raise ManifestError(f"version {version!r} is not a version")

    entry = str(plugin.get("entry") or "")
    if not entry.endswith(".py") or "/" in entry or "\\" in entry:
        raise ManifestError(
            f"entry {entry!r} must be a single .py file in the plugin's own "
            f"directory")

    try:
        api = int(requires.get("api", 1))
    except (TypeError, ValueError):
        raise ManifestError(f"requires.api must be a number") from None

    imports = tuple(str(m) for m in (requires.get("imports") or []))
    network = bool(caps.get("network", False))
    for mod in imports:
        if mod not in THIRD_PARTY_ALLOWED:
            raise ManifestError(
                f"{mod!r} is not a module Aglaïa ships, and no plugin ever "
                f"installs a dependency. Allowed: "
                f"{', '.join(sorted(THIRD_PARTY_ALLOWED))}")
        if mod in THIRD_PARTY_NETWORK and not network:
            raise ManifestError(
                f"{mod!r} talks to the network, so the manifest must declare "
                f"capabilities.network = true")
        if mod == "PySide6" and not bool(caps.get("ui", False)):
            raise ManifestError(
                "PySide6 draws windows, so the manifest must declare "
                "capabilities.ui = true — it is what puts 'adds a window' in "
                "the install dialog")

    return Manifest(
        slug=slug, kind=kind or "", name=str(plugin.get("name") or slug),
        version=version, summary=str(plugin.get("summary") or ""),
        author=str(plugin.get("author") or ""),
        homepage=str(plugin.get("homepage") or ""),
        license=str(plugin.get("license") or ""), entry=entry,
        requires_aglaia=str(requires.get("aglaia") or ""),
        requires_python=str(requires.get("python") or ""),
        api=api, imports=imports,
        config=bool(caps.get("config", False)),
        secrets=bool(caps.get("secrets", False)),
        network=network, files=bool(caps.get("files", False)),
        ui=bool(caps.get("ui", False)),
    )


def api_compatible(man: Manifest) -> tuple[bool, str]:
    if man.api == API_VERSION:
        return True, ""
    return False, (f"{man.name} needs plugin API {man.api}; this build of "
                   f"Aglaïa implements {API_VERSION}")


# ── the import scan ───────────────────────────────────────────────────

@dataclass
class ScanResult:
    allowed: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    undeclared: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def clean(self) -> bool:
        return not (self.refused or self.undeclared or self.error)

    def summary(self) -> str:
        bits = []
        if self.error:
            bits.append(self.error)
        if self.refused:
            bits.append("refused: " + ", ".join(sorted(set(self.refused))))
        if self.undeclared:
            bits.append("undeclared: "
                        + ", ".join(sorted(set(self.undeclared))))
        if self.review:
            bits.append("needs a look: "
                        + ", ".join(sorted(set(self.review))))
        return " · ".join(bits) or "clean"


def _top(name: str) -> str:
    return (name or "").split(".", 1)[0]


def scan_source(source: str, man: Optional[Manifest] = None) -> ScanResult:
    """Classify every import in one module's source."""
    res = ScanResult()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        res.error = f"will not parse: {e}"
        return res

    declared = set(man.imports) if man else set()

    ui_ok = bool(man and man.ui)

    def classify(mod: str, full: str) -> None:
        if mod == "PySide6":
            # The widest surface a plugin gets, and only on request. It does
            # not widen what a plugin COULD do — it already runs in-process —
            # but it does mean it can draw, so the capability puts "adds a
            # window" in the install dialog and tells a reviewer to look
            # harder (docs/plugin-store.md §1).
            (res.allowed if ui_ok else res.undeclared).append(full)
            return
        if mod == "aglaia":
            # Only the façade. Everything else under `aglaia` is internal and
            # may move without notice.
            if full == "aglaia.plugin_api" or full.startswith(
                    "aglaia.plugin_api."):
                res.allowed.append(full)
            else:
                res.refused.append(full)
            return
        if full in SUBMODULE_ALLOWED:
            res.allowed.append(full)
            return
        if mod in STDLIB_NETWORK:
            if man is None or man.network:
                res.allowed.append(full)
            else:
                res.undeclared.append(full)
            return
        if mod in STDLIB_REFUSED:
            res.refused.append(full)
            return
        if mod in STDLIB_ALLOWED:
            res.allowed.append(full)
            return
        if mod in declared:
            res.allowed.append(full)
            return
        if mod in THIRD_PARTY_ALLOWED:
            # Shipped, but this manifest did not say it would use it. Not
            # refused — reported, so the dialog can show what the plugin does
            # beyond what it admitted to.
            res.undeclared.append(full)
            return
        res.refused.append(full)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                classify(_top(a.name), a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # a relative import
                res.allowed.append("." * node.level + (node.module or ""))
                continue
            classify(_top(node.module or ""), node.module or "")
        elif isinstance(node, ast.Call):
            # Bare names only. `re.compile` is an ATTRIBUTE call and is
            # perfectly ordinary — flagging it would fire on nearly every
            # plugin and teach a reviewer to skim past the flag, which is
            # the one thing a review flag must never do. A dangerous
            # attribute call is caught by its import instead.
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in REVIEW_CALLS:
                res.review.append(fn.id)
        elif isinstance(node, ast.Name) and node.id in REVIEW_CALLS:
            res.review.append(node.id)
    return res


def scan_plugin_dir(directory: Path,
                    man: Optional[Manifest] = None) -> ScanResult:
    """Scan the entry module and every private support module beside it."""
    directory = Path(directory)
    merged = ScanResult()
    files = sorted(directory.rglob("*.py"))
    if not files:
        merged.error = "no Python in the plugin directory"
        return merged
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except OSError as e:
            merged.error = f"cannot read {f.name}: {e}"
            continue
        r = scan_source(src, man)
        merged.allowed += r.allowed
        merged.refused += r.refused
        merged.undeclared += r.undeclared
        merged.review += r.review
        if r.error and not merged.error:
            merged.error = f"{f.name}: {r.error}"
    return merged
