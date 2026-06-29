# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Central download registry — the source of truth for fetchable model assets.

Replaces the old static ``model-list.json``: core assets register themselves in
Python at import (``_register_core_targets`` below), and **drop-in plugins
register their own targets** the same way — a processor or OCR plugin calls
``register_download(DownloadTarget(...))`` at module top level, so its weights
show up in the GUI Model Downloader and gate the engine's availability, with no
edit to the core catalogue.

Two halves, mirroring how the ``plugins`` trust registry is split:

* **Catalogue** — an in-memory ``_REGISTRY`` rebuilt every process from the
  ``register_download`` calls (core + imported plugins). Holds the rich metadata
  (``DownloadTarget``).
* **State** — the ``downloads`` table in the config DB (``db.py``) persists the
  lifecycle status (``downloading`` / ``downloaded`` / ``failed``) across runs.
  Absence of a row = never fetched, exactly like the ``plugins`` convention.

Disk stays ground truth: ``is_downloaded()`` is a pure on-disk presence check
(cheap, no DB), and ``download_status()`` reconciles the table against disk so a
manually-deleted model is reported correctly. This module is Qt-free and import-
safe without the heavy engine deps (registration is metadata only).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aglaia.app_data import models_dir

_USER_AGENT = "Aglaia/1.0 (+https://aglaia.bibli.cc)"
_CHUNK = 64 * 1024


# ── the target descriptor ─────────────────────────────────────────────
@dataclass(frozen=True)
class DownloadTarget:
    """One fetchable asset. Field names match the legacy ``ModelSpec`` so the
    GUI Model Downloader and onboarding wizard keep working unchanged."""

    key: str  # stable id (e.g. "paddle_vl", "glm_ocr_mlx")
    title: str  # short card title
    filename: str  # on-disk dir/file name under models_dir()
    url: str  # HTTPS URL (kind="file") or HF repo id (kind="hf-snapshot")
    approx_size_mb: int
    kind: str  # "file" | "hf-snapshot"
    section: str = "other"  # "recommended" | "other" — GUI grouping
    purpose: str = "OCR"  # "Layout detection" / "OCR" / "Voice control"
    project: str = ""  # upstream repo string (pill)
    source: str = "huggingface.co"  # host badge
    sha1: str = ""  # SHA-1 of the canonical file (kind="file")
    required_files: tuple[tuple[str, int], ...] = ()  # snapshot completeness
    # New in the registry refactor:
    platform: str = "any"  # "any" | "darwin-arm64" | "cuda" — for filtering
    registered_by: str = "core"  # "core" or the plugin module that registered it


# ── catalogue (in-memory, rebuilt each process) ───────────────────────
_REGISTRY: dict[str, DownloadTarget] = {}


def register_download(target: DownloadTarget) -> None:
    """Add (or replace) a target in the catalogue. Idempotent — safe to call at
    every import. Core calls it for the bundled assets; plugins call it for
    theirs."""
    _REGISTRY[target.key] = target


def registry() -> list[DownloadTarget]:
    """All registered targets, in registration order (core first)."""
    return list(_REGISTRY.values())


def target_for(key: str) -> Optional[DownloadTarget]:
    return _REGISTRY.get(key)


# ── on-disk presence (ground truth, no DB) ────────────────────────────
def _dest(target: DownloadTarget) -> Path:
    return models_dir() / target.filename


def files_present(target: DownloadTarget) -> bool:
    """True iff the asset's files are physically on disk. For snapshots with a
    declared ``required_files`` manifest, each must exist at (close to) its
    recorded size; otherwise a non-empty dir suffices. For a single file, a
    >1 KiB file."""
    d = _dest(target)
    if target.kind == "hf-snapshot":
        if not d.is_dir():
            return False
        if target.required_files:
            for rel, size in target.required_files:
                f = d / rel
                try:
                    if not f.is_file():
                        return False
                    # Tolerate tiny metadata drift; guard against truncated pulls.
                    if size and f.stat().st_size < int(size * 0.99):
                        return False
                except OSError:
                    return False
            return True
        return any(d.iterdir())
    try:
        return d.exists() and d.stat().st_size > 1024
    except OSError:
        return False


def is_downloaded(key: str) -> bool:
    """Cheap, DB-free verdict used on hot paths (capability probes, engine
    availability). Pure disk check."""
    target = _REGISTRY.get(key)
    return bool(target and files_present(target))


# ── lifecycle state (persisted in the config DB) ──────────────────────
# Status strings stored in the `downloads` table.
STATUS_DOWNLOADING = "downloading"
STATUS_DOWNLOADED = "downloaded"
STATUS_FAILED = "failed"
STATUS_NONE = "not_downloaded"  # synthetic — never stored; means "no row"


def record_status(
    key: str, status: str, *, sha: str | None = None, size_bytes: int | None = None
) -> None:
    """Write the lifecycle status for ``key``. Best-effort — a config-DB hiccup
    must never break a download."""
    try:
        from aglaia.app_data import db as _db

        with _db.session() as conn:
            _db.set_download_status(conn, key, status, sha=sha, size_bytes=size_bytes)
    except Exception:
        pass


def clear_status(key: str) -> None:
    try:
        from aglaia.app_data import db as _db

        with _db.session() as conn:
            _db.clear_download_status(conn, key)
    except Exception:
        pass


def download_status(key: str) -> str:
    """Reconciled lifecycle verdict for the GUI: ``downloaded`` when the files
    are on disk (records/repairs the row), else the persisted ``downloading`` /
    ``failed`` row, else ``not_downloaded``. Deleting a model out-of-band is
    therefore reported honestly on the next read."""
    target = _REGISTRY.get(key)
    if target is None:
        return STATUS_NONE
    present = files_present(target)
    try:
        from aglaia.app_data import db as _db

        with _db.session() as conn:
            row = _db.get_download_status(conn, key)
            if present:
                if row != STATUS_DOWNLOADED:
                    _db.set_download_status(conn, key, STATUS_DOWNLOADED)
                return STATUS_DOWNLOADED
            # Not on disk — a stale "downloaded" row means it was deleted.
            if row == STATUS_DOWNLOADED:
                _db.clear_download_status(conn, key)
                return STATUS_NONE
            return row or STATUS_NONE
    except Exception:
        return STATUS_DOWNLOADED if present else STATUS_NONE


# ── synchronous download (CLI; no Qt) ─────────────────────────────────
# The GUI drives downloads through Qt QThread workers (ModelDownloaderTab);
# the `--without-gui` CLI (`aglaia setup`) has no PySide6, so a blocking urllib
# fetcher lives here. Both update the lifecycle state.
ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total)


def _hf_list_files(repo: str) -> list[tuple[str, int]]:
    """[(rfilename, size_bytes)] for a HuggingFace repo via the tree API."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", _USER_AGENT)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (https only)
        data = json.load(r)
    out: list[tuple[str, int]] = []
    for s in data or []:
        if s.get("type") != "file" or not s.get("path"):
            continue
        lfs = s.get("lfs") or {}
        sz = (
            int(lfs["size"])
            if isinstance(lfs, dict) and lfs.get("size")
            else int(s.get("size") or 0)
        )
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
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (https only)
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


def download_model(
    target: DownloadTarget, on_progress: Optional[ProgressCb] = None
) -> None:
    """Synchronously fetch ``target`` into ``models_dir()``, recording the
    lifecycle status. ``on_progress`` gets cumulative ``(done, total)`` bytes."""
    record_status(target.key, STATUS_DOWNLOADING)
    try:
        if target.kind == "hf-snapshot":
            dest_dir = _dest(target)
            files = _hf_list_files(target.url)
            total = (
                sum(sz for _, sz in files) or (target.approx_size_mb * 1024 * 1024) or 1
            )
            done = 0

            def bump(n: int) -> None:
                nonlocal done
                done += n
                if on_progress:
                    on_progress(done, total)

            for rfn, _sz in files:
                _stream_to(
                    f"https://huggingface.co/{target.url}/resolve/main/{rfn}",
                    dest_dir / rfn,
                    bump,
                )
        else:  # single file
            total = (target.approx_size_mb * 1024 * 1024) or 1
            done = 0

            def bump(n: int) -> None:
                nonlocal done
                done += n
                if on_progress:
                    on_progress(min(done, total), total)

            _stream_to(target.url, _dest(target), bump)
    except Exception:
        record_status(target.key, STATUS_FAILED)
        raise
    record_status(target.key, STATUS_DOWNLOADED)


# ── core asset registrations (was model-list.json) ────────────────────
def _register_core_targets() -> None:
    """Register the assets that ship with Aglaïa. Order = GUI display order."""
    register_download(
        DownloadTarget(
            key="vosk_en",
            title="Vosk — English (small)",
            filename="vosk-model-small-en-us",
            url="aglaia-models/vosk-model-small-en-us-0.15",
            approx_size_mb=40,
            kind="hf-snapshot",
            section="recommended",
            purpose="Voice control",
            project="aglaia-models/vosk-model-small-en-us-0.15",
        )
    )
    register_download(
        DownloadTarget(
            key="east",
            title="EAST",
            filename="frozen_east_text_detection.pb",
            url="https://huggingface.co/aglaia-models/east-text-detection/resolve/main/frozen_east_text_detection.pb",
            approx_size_mb=95,
            kind="file",
            section="recommended",
            purpose="Layout detection",
            project="aglaia-models/east-text-detection",
            sha1="fffabf5ac36f37bddf68e34e84b45f5c4247ed06",
        )
    )
    register_download(
        DownloadTarget(
            key="surya",
            title="Surya 2 — Q4_K_M (GGUF)",
            filename="surya-ocr-2-Q4_K_M-gguf",
            url="aglaia-models/surya-ocr-2-Q4_K_M-gguf",
            approx_size_mb=610,
            kind="hf-snapshot",
            section="recommended",
            purpose="OCR",
            project="aglaia-models/surya-ocr-2-Q4_K_M-gguf",
            required_files=(
                ("surya-2-Q4_K_M.gguf", 403585152),
                ("surya-2-mmproj.gguf", 204986688),
            ),
        )
    )
    register_download(
        DownloadTarget(
            key="paddle_vl",
            title="PaddleOCR-VL 1.5 (MLX 4-bit)",
            filename="PaddleOCR-VL-1.5-4bit",
            url="aglaia-models/paddleocr-vl-1.5-4bit",
            approx_size_mb=720,
            kind="hf-snapshot",
            section="recommended",
            purpose="OCR",
            project="aglaia-models/paddleocr-vl-1.5-4bit",
            platform="darwin-arm64",
            required_files=(
                ("model.safetensors", 703562711),
                ("processor_config.json", 843),
                ("config.json", 2483),
                ("tokenizer.json", 11189060),
            ),
        )
    )
    register_download(
        DownloadTarget(
            key="dbnet",
            title="PP-OCRv4 det",
            filename="ch_PP-OCRv4_det_infer.onnx",
            url="https://huggingface.co/aglaia-models/ppocrv4-det/resolve/main/ch_PP-OCRv4_det_infer.onnx",
            approx_size_mb=5,
            kind="file",
            section="other",
            purpose="Layout detection",
            project="aglaia-models/ppocrv4-det",
            sha1="0f1bf3fbf5e0ade20036429f3b50d4ab5626c6df",
        )
    )


_register_core_targets()
