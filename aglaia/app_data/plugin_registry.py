# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Installing plugins — from the registry, and from a local archive (#129/#130).

Two sources, and the difference between them is the whole point:

* **The registry** (`aglaia-plugins`) is PR-reviewed. `index.json` lists every
  plugin with a per-file sha256, so a download that does not match what the
  index says is refused before it is written. Install is one confirm — and a
  disclaimer naming the person who actually wrote it, because reviewing
  something does not make it ours.
* **A local `.aglplugin` archive** was reviewed by nobody. It is validated
  before anything is extracted, scanned for what it imports, and gated behind
  a typed sentence.

What is NOT here, deliberately: dependency installation. Not at install time,
not at runtime, never. A plugin that needs a library Aglaïa does not ship is a
PR against the host, not a `pip install` the installer runs on the user's
behalf. That one rule removes the entire supply-chain surface.

Signature verification of `index.json` is designed (`docs/plugin-store.md` §3)
and not implemented: it needs an offline signing key and a release step. Until
then the index is fetched over HTTPS and each file is checked against its
hash, and `IndexResult.signed` is False so the UI can say which guarantee the
user is actually getting. That is a smaller promise honestly labelled, rather
than a bigger one quietly unmet.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import app_data_dir
from .plugin_ctx import SLUG_RE, PluginConfig, PluginSecrets, plugin_data_dir
from .plugin_manifest import (KINDS, Manifest, ManifestError, ScanResult,
                              api_compatible, parse_manifest, scan_plugin_dir)

#: Where the registry lives. Raw files, so no API token and no rate limit
#: that matters for a once-a-day index fetch.
REGISTRY_REPO = "yb85/aglaia-plugins"
REGISTRY_BRANCH = "main"
REGISTRY_RAW = f"https://raw.githubusercontent.com/{REGISTRY_REPO}/{REGISTRY_BRANCH}"
REGISTRY_WEB = f"https://github.com/{REGISTRY_REPO}"

#: Archive limits, checked before extraction. A plugin is a few small text
#: files; anything else is either a mistake or a zip bomb.
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_MEMBERS = 200

#: Never extracted. A compiled extension cannot be reviewed by reading it,
#: which makes it exactly the thing this system cannot honestly accept.
BINARY_SUFFIXES = (".so", ".dylib", ".pyd", ".dll", ".exe", ".bin")


def installed_root(kind: str = "") -> Path:
    d = app_data_dir() / "plugins"
    if kind:
        d = d / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class RegistryEntry:
    slug: str
    kind: str
    name: str
    version: str
    summary: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    source_url: str = ""
    capabilities: dict = field(default_factory=dict)
    imports: tuple[str, ...] = ()
    files: dict[str, str] = field(default_factory=dict)
    api: int = 1

    @property
    def web_url(self) -> str:
        """Where to read the code. The index may pin it to the merged commit;
        without that, the directory on the default branch."""
        return self.source_url or f"{REGISTRY_WEB}/tree/{REGISTRY_BRANCH}/{self.kind}/{self.slug}"

    @property
    def first_party(self) -> bool:
        """Written by Aglaïa itself, rather than submitted by someone else.

        It changes what the install dialog can honestly say: "submitted by
        Aglaïa, not by Aglaïa" is not a disclaimer, it is a sentence that
        makes the reader distrust the rest of the dialog.

        Keyed off the maintainer address rather than a flag in the index,
        because a flag is something a submitted manifest could set."""
        return "aglaia@bibli.cc" in (self.author or "").lower()

    def declared(self) -> list[str]:
        labels = {"config": "its own settings",
                  "secrets": "stores secrets in your keychain",
                  "network": "network access",
                  "files": "reads/writes files outside its own folder",
                  "ui": "adds a window to the Plugins menu"}
        return [labels[k] for k, v in (self.capabilities or {}).items()
                if v and k in labels]


@dataclass
class IndexResult:
    entries: list[RegistryEntry] = field(default_factory=list)
    revoked: list[dict] = field(default_factory=list)
    error: str = ""
    #: False until index signing exists. The UI says which guarantee applies.
    signed: bool = False

    def get(self, slug: str) -> Optional[RegistryEntry]:
        for e in self.entries:
            if e.slug == slug:
                return e
        return None

    def is_revoked(self, slug: str, version: str) -> Optional[str]:
        for r in self.revoked:
            if r.get("slug") == slug and str(r.get("version", "")) in (
                    "", version):
                return str(r.get("reason") or "revoked")
        return None


def _client(timeout: float, *, ipv4_first: bool = True):
    """An httpx client that does not stall on a dead IPv6 route.

    `raw.githubusercontent.com` resolves to four IPv6 addresses before four
    IPv4 ones. On a network that advertises IPv6 but does not route it — a
    common home-router state — Python tries them **strictly in order** and
    burns the full connect timeout on each. Measured here: four IPv6
    timeouts, then IPv4 connecting in 0.05 s. That is 24 seconds to fetch
    15 KB, and it is why an install felt like a hang.

    curl is fast on the same machine because it does Happy Eyeballs: it
    starts an IPv4 attempt about 200 ms in and takes whichever answers.
    httpx has no such thing, so this binds the local address to IPv4, which
    makes the resolver hand back A records only.

    Not hard-forced: `ipv4_first=False` gives a plain client, and the callers
    fall back to it. A genuinely IPv6-only network must still work, and it is
    not this code's place to decide the machine has IPv4."""
    import httpx
    limits = httpx.Timeout(timeout, connect=6.0)
    transport = None
    if ipv4_first:
        try:
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
        except Exception:
            transport = None
    return httpx.Client(timeout=limits, follow_redirects=True,
                        transport=transport)


def _cache_path() -> Path:
    return app_data_dir() / "plugins" / "index-cache.json"


def fetch_index(*, timeout: float = 20.0,
                use_cache_on_error: bool = True) -> IndexResult:
    """Read `index.json` from the registry, falling back to the last copy.

    A network failure must not empty the list a user is looking at — they may
    be offline and still want to see what is installed and what was available
    yesterday. The error travels with the result rather than replacing it."""
    import httpx
    raw = None
    err = ""
    # A SHORT connect timeout, separate from the read timeout. A host that
    # resolves to several addresses is tried one at a time, and a route that
    # black-holes costs the full connect timeout before the next is tried —
    # so a generous single timeout turns "one dead address" into half a minute
    # of apparent hang. Six seconds is far more than a reachable host needs
    # and far less than a dead one takes to admit it.
    for ipv4_first in (True, False):
        try:
            with _client(timeout, ipv4_first=ipv4_first) as c:
                r = c.get(f"{REGISTRY_RAW}/index.json")
            if r.status_code >= 400:
                err = f"the registry answered {r.status_code}"
            else:
                raw = r.text
            break
        except Exception as e:  # noqa: BLE001
            err = f"cannot reach the registry — {type(e).__name__}"
            # IPv4-only failed; the machine may genuinely be IPv6-only.
            continue

    if raw is not None:
        try:
            _cache_path().parent.mkdir(parents=True, exist_ok=True)
            _cache_path().write_text(raw, encoding="utf-8")
        except OSError:
            pass
    elif use_cache_on_error and _cache_path().is_file():
        try:
            raw = _cache_path().read_text(encoding="utf-8")
            err = f"{err} — showing the last copy"
        except OSError:
            pass

    if raw is None:
        return IndexResult(error=err or "no index")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return IndexResult(error=f"the index is not valid JSON: {e}")

    out = IndexResult(error=err, revoked=list(data.get("revoked") or []))
    for item in data.get("plugins") or []:
        kind = str(item.get("kind") or "")
        slug = str(item.get("slug") or "")
        if kind not in KINDS or not SLUG_RE.match(slug):
            continue
        out.entries.append(RegistryEntry(
            slug=slug, kind=kind, name=str(item.get("name") or slug),
            version=str(item.get("version") or "0"),
            summary=str(item.get("summary") or ""),
            author=str(item.get("author") or ""),
            license=str(item.get("license") or ""),
            homepage=str(item.get("homepage") or ""),
            source_url=str(item.get("source_url") or ""),
            capabilities=dict(item.get("capabilities") or {}),
            imports=tuple(item.get("imports") or []),
            files=dict(item.get("files") or {}),
            api=int(item.get("api", (item.get("requires") or {}).get("api", 1))),
        ))
    return out


@dataclass
class InstallResult:
    ok: bool
    message: str
    slug: str = ""
    path: Optional[Path] = None
    scan: Optional[ScanResult] = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> bool:
    """A member path that cannot escape the plugin's own directory."""
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        return False
    if name.startswith(("/", "\\")):
        return False
    return not any(str(name).lower().endswith(s) for s in BINARY_SUFFIXES)


def install_from_registry(entry: RegistryEntry, *,
                          timeout: float = 60.0,
                          on_progress=None) -> InstallResult:
    """Download one registry plugin, verify every file against the index, and
    install it. Nothing is written outside a temp dir until every hash
    matches — a half-installed plugin is a plugin that loads and misbehaves."""
    import httpx
    if not entry.files:
        return InstallResult(False, "the index lists no files for this plugin")

    staged: dict[str, bytes] = {}
    try:
        with _client(timeout) as c:
            total = len(entry.files)
            for i, (rel, want) in enumerate(entry.files.items(), 1):
                if on_progress is not None:
                    on_progress(i, total, rel)
                if not _safe_member(rel):
                    return InstallResult(
                        False, f"the index lists an unsafe path: {rel}")
                url = f"{REGISTRY_RAW}/{entry.kind}/{entry.slug}/{rel}"
                r = c.get(url)
                if r.status_code >= 400:
                    return InstallResult(
                        False, f"{rel} — the registry answered "
                               f"{r.status_code}")
                data = r.content
                if len(data) > MAX_FILE_BYTES:
                    return InstallResult(
                        False, f"{rel} is larger than the "
                               f"{MAX_FILE_BYTES // (1024 * 1024)} MB limit")
                got = _sha256(data)
                want_hex = str(want).split(":", 1)[-1]
                if got != want_hex:
                    return InstallResult(
                        False, f"{rel} does not match the hash the index "
                               f"gives for it, so it was refused.\n\n"
                               f"Usually this means the cached index is out "
                               f"of date and the plugin was updated since — "
                               f"hit Refresh and install again. If it "
                               f"persists, the file really is not what the "
                               f"registry says it should be.")
                staged[rel] = data
    except Exception as e:  # noqa: BLE001
        return InstallResult(False, f"download failed — {type(e).__name__}: {e}")

    return _write_plugin(entry.kind, entry.slug, staged, source="registry")


def read_archive(path: Path) -> tuple[Optional[str], dict[str, bytes], str]:
    """Validate an `.aglplugin` and return `(slug, files, error)`.

    Everything is checked BEFORE anything is written: a zip that is going to
    be refused should never have touched the disk."""
    path = Path(path)
    try:
        zf = zipfile.ZipFile(path)
    except Exception as e:  # noqa: BLE001
        return None, {}, f"not a readable archive: {e}"
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            return None, {}, f"archive has {len(infos)} members; the limit is {MAX_MEMBERS}"
        total = sum(i.file_size for i in infos)
        if total > MAX_ARCHIVE_BYTES:
            return None, {}, (f"archive expands to {total // (1024*1024)} MB; "
                              f"the limit is {MAX_ARCHIVE_BYTES // (1024*1024)} MB")
        files: dict[str, bytes] = {}
        roots: set[str] = set()
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            if not _safe_member(name):
                return None, {}, (
                    f"refused: {name}. Archives may not contain absolute or "
                    f"'..' paths, or compiled extensions — a compiled "
                    f"extension cannot be reviewed by reading it.")
            if info.file_size > MAX_FILE_BYTES:
                return None, {}, f"{name} is over the per-file limit"
            parts = Path(name).parts
            # Tolerate both `slug/…` and a flat archive.
            if len(parts) > 1:
                roots.add(parts[0])
                rel = str(Path(*parts[1:]))
            else:
                rel = name
            files[rel] = zf.read(info)
        if len(roots) > 1:
            return None, {}, (f"archive holds several top-level directories "
                              f"({', '.join(sorted(roots))}); a plugin is one")
    if "aglaia-plugin.toml" not in files:
        return None, {}, "no aglaia-plugin.toml in the archive"
    slug = next(iter(roots), "")
    return (slug or None), files, ""


def stage_archive(path: Path) -> tuple[Optional[Manifest], dict[str, bytes],
                                       Optional[ScanResult], str]:
    """Validate + parse + scan an archive without installing it.

    The dialog needs all three before it can ask the user anything: what the
    plugin claims, what it actually imports, and where the two disagree."""
    import tempfile
    slug_hint, files, err = read_archive(path)
    if err:
        return None, {}, None, err
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for rel, data in files.items():
            fp = d / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(data)
        try:
            man = parse_manifest(d / "aglaia-plugin.toml")
        except ManifestError as e:
            return None, files, None, str(e)
        if slug_hint and slug_hint != man.slug:
            return None, files, None, (
                f"the archive's folder is {slug_hint!r} but the manifest says "
                f"{man.slug!r}")
        tops = [r for r in files if r.endswith(".py") and "/" not in r]
        if man.entry not in tops:
            return None, files, None, (
                f"the manifest's entry {man.entry!r} is not in the archive")
        if len(tops) > 1:
            return None, files, None, (
                f"a plugin is ONE top-level module; this has "
                f"{', '.join(sorted(tops))}")
        ok, why = api_compatible(man)
        if not ok:
            return man, files, None, why
        scan = scan_plugin_dir(d, man)
    return man, files, scan, ""


def install_from_archive(path: Path, kind: str) -> InstallResult:
    man, files, scan, err = stage_archive(path)
    if err or man is None:
        return InstallResult(False, err or "unreadable archive")
    return _write_plugin(kind or man.kind or "destinations", man.slug, files,
                         source="zip", scan=scan)


def _write_plugin(kind: str, slug: str, files: dict[str, bytes], *,
                  source: str, scan: Optional[ScanResult] = None
                  ) -> InstallResult:
    if kind not in KINDS:
        return InstallResult(False, f"unknown plugin kind {kind!r}")
    if not SLUG_RE.match(slug):
        return InstallResult(False, f"unusable slug {slug!r}")
    dest = installed_root(kind) / slug
    # Replace atomically-ish: write beside, then swap. A half-written plugin
    # directory is one that loads and misbehaves.
    staging = installed_root(kind) / f".{slug}.staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for rel, data in files.items():
            fp = staging / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(data)
        try:
            man = parse_manifest(staging / "aglaia-plugin.toml", kind=kind,
                                 expect_slug=slug)
        except ManifestError as e:
            shutil.rmtree(staging, ignore_errors=True)
            return InstallResult(False, str(e))
        ok, why = api_compatible(man)
        if not ok:
            shutil.rmtree(staging, ignore_errors=True)
            return InstallResult(False, why)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        return InstallResult(False, f"could not write the plugin: {e}")

    _record(slug, kind, man, source)
    return InstallResult(True, f"{man.name} {man.version} installed.",
                         slug=slug, path=dest, scan=scan)


# ── the installed record ──────────────────────────────────────────────

def _record(slug: str, kind: str, man: Manifest, source: str) -> None:
    """Remember where a plugin came from — the tab shows it for as long as it
    is installed. A one-time warning that vanishes is a warning the user
    forgets he accepted."""
    from . import db as cfg
    try:
        with cfg.session() as conn:
            data = cfg.get(conn, "installed_plugins", {}) or {}
            if not isinstance(data, dict):
                data = {}
            data[slug] = {"kind": kind, "version": man.version,
                          "name": man.name, "author": man.author,
                          "source": source, "license": man.license,
                          "capabilities": man.capabilities}
            cfg.set(conn, "installed_plugins", data)
            conn.commit()
    except Exception:
        pass


def installed_record(slug: str = "") -> dict:
    from . import db as cfg
    try:
        with cfg.session() as conn:
            data = cfg.get(conn, "installed_plugins", {}) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data.get(slug, {}) if slug else data


def set_disabled(slug: str, disabled: bool) -> None:
    from . import db as cfg
    try:
        with cfg.session() as conn:
            data = cfg.get(conn, "installed_plugins", {}) or {}
            rec = dict(data.get(slug) or {})
            rec["disabled"] = bool(disabled)
            data[slug] = rec
            cfg.set(conn, "installed_plugins", data)
            conn.commit()
    except Exception:
        pass


def is_disabled(slug: str) -> bool:
    return bool(installed_record(slug).get("disabled"))


def uninstall(slug: str) -> InstallResult:
    """Remove the code, the settings, the scratch dir and the secrets.

    All four, because "uninstalled" that leaves a password in the keychain is
    not uninstalled."""
    rec = installed_record(slug)
    kind = str(rec.get("kind") or "")
    removed = []
    for k in ([kind] if kind else list(KINDS)):
        d = installed_root(k) / slug
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    try:
        cfgdb = PluginConfig(slug)
        PluginSecrets(slug, cfgdb).purge()
    except Exception:
        pass
    data_dir = plugin_data_dir(slug)
    if data_dir.is_dir():
        shutil.rmtree(data_dir, ignore_errors=True)
        removed.append(str(data_dir))
    from . import db as cfg
    try:
        with cfg.session() as conn:
            data = cfg.get(conn, "installed_plugins", {}) or {}
            data.pop(slug, None)
            cfg.set(conn, "installed_plugins", data)
            conn.commit()
    except Exception:
        pass
    if not removed:
        return InstallResult(False, f"{slug} was not installed")
    return InstallResult(True, f"{slug} removed, with its settings and "
                               f"secrets.", slug=slug)


def list_installed() -> list[dict]:
    """Every plugin directory under APP_DATA, with its manifest and record."""
    out: list[dict] = []
    for kind in KINDS:
        root = installed_root(kind)
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith(("." , "_")):
                continue
            try:
                man = parse_manifest(d / "aglaia-plugin.toml", kind=kind,
                                     expect_slug=d.name)
            except ManifestError as e:
                out.append({"slug": d.name, "kind": kind, "dir": d,
                            "manifest": None, "error": str(e),
                            "record": installed_record(d.name)})
                continue
            out.append({"slug": d.name, "kind": kind, "dir": d,
                        "manifest": man, "error": "",
                        "record": installed_record(d.name)})
    return out
