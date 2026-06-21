#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Download llama.cpp pre-built binaries into ``vendor/llama-server/``.

PyInstaller's ``Aglaia.spec`` ships the per-platform binary from
``vendor/llama-server/<plat>/`` into the bundle. Run this script
before building the .app (or any other platform) so that directory is
populated with the correct llama-server.

Usage::

    # Fetch for the current host (auto-detect):
    python scripts/fetch_llama_server.py

    # Fetch a specific platform/arch:
    python scripts/fetch_llama_server.py --platform macos-arm64
    python scripts/fetch_llama_server.py --platform linux-x64
    python scripts/fetch_llama_server.py --platform windows-x64

    # Fetch every supported (plat, arch) — useful in CI before
    # packaging multiple installer flavours.
    python scripts/fetch_llama_server.py --all

The mapping of "platform key" → release-asset name template tracks the
ggml-org/llama.cpp GitHub releases. Update ``ASSETS`` when upstream
renames an artifact.

Defaults to the latest release. Pin a specific tag with
``--tag b1234``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform as _plat
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor" / "llama-server"

API_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
API_TAG = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"

# Substring matchers against release-asset names. Upstream switches
# naming periodically — keep these loose. We pick the first matching
# asset per platform.
ASSETS: dict[str, list[str]] = {
    "macos-arm64":   ["macos-arm64.tar.gz", "macos-arm64.zip"],
    "macos-x64":     ["macos-x64.tar.gz", "macos-x64.zip"],
    "linux-x64":     ["ubuntu-x64.tar.gz", "linux-x64.tar.gz",
                       "ubuntu-x64.zip", "linux-x64.zip"],
    "linux-arm64":   ["ubuntu-arm64.tar.gz", "linux-arm64.tar.gz",
                       "ubuntu-arm64.zip", "linux-arm64.zip"],
    "windows-x64":   ["win-vulkan-x64.zip", "win-cpu-x64.zip",
                       "win-avx2-x64.zip", "win-x64.zip"],
    "windows-arm64": ["win-cpu-arm64.zip", "win-arm64.zip"],
}

# Filenames we want to keep from each release zip. llama.cpp ships ~30
# binaries; we only need the server + its shared libs.
KEEP_NAME_FRAGMENTS = (
    "llama-server", "llama_server",
    "ggml", "llama",          # libs
    "metal", "cuda", "blas",  # backends (mac/linux/win variants)
    "mtmd",                    # multimodal (VLM) — llama-server depends on it
)


def _detect_host_key() -> str:
    sys_name = _plat.system().lower()
    machine = _plat.machine().lower()
    if sys_name == "darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x64"
    if sys_name == "linux":
        return "linux-arm64" if machine in ("arm64", "aarch64") else "linux-x64"
    if sys_name == "windows":
        return "windows-arm64" if machine in ("arm64", "aarch64") else "windows-x64"
    raise SystemExit(f"unsupported host platform: {sys_name}/{machine}")


def _fetch_release(tag: str | None) -> dict:
    url = API_LATEST if tag is None else API_TAG.format(tag=tag)
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "aglaia-build"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_asset(release: dict, key: str) -> dict | None:
    assets = release.get("assets") or []
    candidates = ASSETS.get(key, [])
    # Best match: first hint substring that resolves to an asset whose
    # name contains it.
    for hint in candidates:
        for asset in assets:
            name = asset.get("name", "")
            if hint in name:
                return asset
    return None


def _download(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "aglaia-build"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _should_keep(name: str) -> bool:
    low = name.lower()
    return any(frag in low for frag in KEEP_NAME_FRAGMENTS)


def _mark_exec(target: Path, base: str) -> None:
    if ("llama-server" in base.lower()
            or base.endswith(".dylib")
            or base.endswith(".so")
            or ".so." in base):
        target.chmod(
            target.stat().st_mode
            | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )


def _extract_zip(payload: bytes, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for info in zf.infolist():
            base = os.path.basename(info.filename)
            if not base or info.is_dir():
                continue
            if not _should_keep(base):
                continue
            target = dest / base
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(base)
            _mark_exec(target, base)
    return written


def _extract_tar_gz(payload: bytes, dest: Path) -> list[str]:
    """Extract llama.cpp's macOS / Linux tarball. Flattens the single
    top-level dir but preserves symlinks (the dylib aliases that the
    @rpath loader needs at runtime — e.g. ``libllama.0.dylib``
    pointing at ``libllama.0.0.9575.dylib``)."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        members = list(tf.getmembers())
        # Pass 1: regular files.
        for member in members:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if not base or not _should_keep(base):
                continue
            target = dest / base
            src = tf.extractfile(member)
            if src is None:
                continue
            with target.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(base)
            _mark_exec(target, base)
        # Pass 2: symlinks. The dylib @rpath loader keys off these.
        for member in members:
            if not member.issym() and not member.islnk():
                continue
            base = os.path.basename(member.name)
            if not base or not _should_keep(base):
                continue
            target = dest / base
            link_target = os.path.basename(member.linkname or "")
            if not link_target:
                continue
            try:
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(link_target)
            except OSError:
                continue
            written.append(base)
    return written


def _extract(payload: bytes, asset_name: str, dest: Path) -> list[str]:
    low = asset_name.lower()
    if low.endswith(".zip"):
        return _extract_zip(payload, dest)
    if low.endswith(".tar.gz") or low.endswith(".tgz"):
        return _extract_tar_gz(payload, dest)
    raise SystemExit(f"unsupported archive type: {asset_name}")


def fetch(keys: Iterable[str], tag: str | None) -> None:
    release = _fetch_release(tag)
    print(f"llama.cpp release: {release.get('tag_name', '<?>')}")
    for key in keys:
        asset = _pick_asset(release, key)
        if asset is None:
            print(f"  [{key}] no matching asset found; skipping")
            continue
        url = asset.get("browser_download_url")
        if not url:
            print(f"  [{key}] asset missing download_url; skipping")
            continue
        print(f"  [{key}] downloading {asset['name']} ({asset.get('size', 0):,} B)")
        try:
            payload = _download(url)
        except urllib.error.HTTPError as e:
            print(f"  [{key}] download failed: {e}")
            continue
        dest = VENDOR_ROOT / key
        # Wipe any previous contents so stale libs don't shadow the new ones.
        if dest.is_dir():
            shutil.rmtree(dest)
        written = _extract(payload, asset["name"], dest)
        print(f"  [{key}] wrote {len(written)} file(s) → {dest.relative_to(REPO_ROOT)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", help="Target platform key "
                        "(e.g. macos-arm64). Default: host.")
    parser.add_argument("--all", action="store_true",
                        help="Fetch every supported (os, arch).")
    parser.add_argument("--tag", help="Pin to a specific release tag "
                        "(e.g. b1234). Default: latest.")
    args = parser.parse_args(argv)

    if args.all:
        keys = list(ASSETS.keys())
    elif args.platform:
        if args.platform not in ASSETS:
            parser.error(f"unknown platform {args.platform!r}. "
                          f"Choices: {sorted(ASSETS)}")
        keys = [args.platform]
    else:
        keys = [_detect_host_key()]

    fetch(keys, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
