# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Qt-free model registry + synchronous downloader.

The GUI (``ModelDownloaderTab``) drives downloads through Qt ``QThread``
workers; the CLI ``aglaia --setup`` has no Qt (the ``--without-gui`` install
ships no PySide6). So the registry (``ModelSpec`` / ``_load_model_specs`` /
``is_model_installed``) and a blocking urllib downloader live here, importable
without PySide6. ``ModelDownloaderTab`` re-exports the registry for the GUI.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Optional

from aglaia.app_data import app_data_dir, models_dir

_USER_AGENT = "Aglaia/1.0 (+https://aglaia.bibli.cc)"
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class ModelSpec:
    key: str            # internal id (e.g. "dbnet", "vosk_en", "surya")
    title: str          # short card title
    filename: str       # on-disk filename inside models_dir (or subdir)
    url: str            # download URL (file kind) or HF repo id (hf-snapshot)
    approx_size_mb: int
    kind: str           # "file" | "hf-snapshot"
    section: str        # "recommended" | "other"
    purpose: str        # e.g. "Layout detection" / "OCR"
    project: str        # human-readable upstream project name
    source: str         # short host badge: "github.com" / "huggingface.co"
    sha1: str = ""      # SHA-1 of the canonical file (`kind="file"` only)
    required_files: tuple = ()   # ((path, size), …) for hf-snapshot completeness


def _load_model_specs() -> list[ModelSpec]:
    """Load the registry from ``model-list.json`` — a per-user override in
    ``<APP_DATA>/model-list.json`` wins over the bundled ``aglaia/app_data``
    copy. Unknown JSON fields are tolerated."""
    candidates = [
        app_data_dir() / "model-list.json",
        Path(__file__).resolve().parent / "model-list.json",
    ]
    raw: Optional[dict] = None
    for p in candidates:
        try:
            if p.is_file():
                raw = json.loads(p.read_text(encoding="utf-8"))
                break
        except Exception:
            continue
    if not raw:
        return []
    allowed = {f.name for f in fields(ModelSpec)}
    out: list[ModelSpec] = []
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        clean = {k: v for k, v in entry.items() if k in allowed}
        rf = clean.get("required_files") or ()
        if rf:
            clean["required_files"] = tuple(
                (str(e.get("path", "")), int(e.get("size", 0)))
                for e in rf if isinstance(e, dict) and e.get("path"))
        try:
            out.append(ModelSpec(**clean))
        except TypeError:
            continue  # missing required field — skip
    return out


MODEL_SPECS: list[ModelSpec] = _load_model_specs()


def spec_for(key: str) -> Optional[ModelSpec]:
    return next((s for s in (_load_model_specs() or MODEL_SPECS) if s.key == key), None)


def is_model_installed(key: str) -> bool:
    """Lightweight on-disk presence check (no Qt needed)."""
    spec = spec_for(key)
    if spec is None:
        return False
    d = models_dir() / spec.filename
    if spec.kind == "hf-snapshot":
        return d.is_dir() and any(d.iterdir())
    try:
        return d.exists() and d.stat().st_size > 1024
    except OSError:
        return False


# ── synchronous download (CLI) ───────────────────────────────────────
ProgressCb = Callable[[int, int], None]   # (bytes_done, bytes_total)


def _hf_list_files(repo: str) -> list[tuple[str, int]]:
    """[(rfilename, size_bytes)] for a HuggingFace repo via the tree API."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", _USER_AGENT)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:   # noqa: S310 (https only)
        data = json.load(r)
    out: list[tuple[str, int]] = []
    for s in data or []:
        if s.get("type") != "file" or not s.get("path"):
            continue
        lfs = s.get("lfs") or {}
        sz = int(lfs["size"]) if isinstance(lfs, dict) and lfs.get("size") \
            else int(s.get("size") or 0)
        out.append((s["path"], sz))
    return out


def _stream_to(url: str, dest: Path, on_chunk: Callable[[int], None]) -> int:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", _USER_AGENT)
    tmp = dest.with_name(dest.name + ".partialdl")
    got = 0
    with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310 (https only)
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                on_chunk(len(chunk))
    tmp.replace(dest)
    return got


def download_model(spec: ModelSpec, on_progress: Optional[ProgressCb] = None) -> None:
    """Synchronously fetch ``spec`` into ``models_dir()``. ``on_progress`` is
    called with cumulative ``(bytes_done, bytes_total)`` across all files."""
    if spec.kind == "hf-snapshot":
        dest_dir = models_dir() / spec.filename
        files = _hf_list_files(spec.url)
        total = sum(sz for _, sz in files) or (spec.approx_size_mb * 1024 * 1024) or 1
        done = 0

        def bump(n: int) -> None:
            nonlocal done
            done += n
            if on_progress:
                on_progress(done, total)
        for rfn, _sz in files:
            _stream_to(f"https://huggingface.co/{spec.url}/resolve/main/{rfn}",
                       dest_dir / rfn, bump)
    else:  # single file
        total = (spec.approx_size_mb * 1024 * 1024) or 1
        done = 0

        def bump(n: int) -> None:
            nonlocal done
            done += n
            if on_progress:
                on_progress(min(done, total), total)
        _stream_to(spec.url, models_dir() / spec.filename, bump)
