# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Read an ``.aglbundle`` produced by **aglaia-bridge** (the iOS capture app).

The bundle is the lightweight handoff format from issue #47 — a directory (or a
``.zip`` of one) containing::

    manifest.json   {format_version, project, created, device, page_count, dpi?}
    pages.jsonl     one JSON object per page: {index, file, captured, w, h}
    images/NNNN.jpg

This module turns that into an ordered list of image paths the existing import
pipeline (:func:`aglaia.workers.ImportHelpers.enqueue_image_files`) can consume.
Pure stdlib — no new dependencies, headlessly testable.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_FORMAT_VERSION = 1


class BridgeBundleError(Exception):
    """Raised when a bundle is malformed or an unsupported format version."""


@dataclass(frozen=True)
class BridgePage:
    index: int
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class BridgeBundle:
    project: str
    device: str
    dpi: int | None
    pages: list[BridgePage]

    @property
    def image_paths(self) -> list[Path]:
        """Page image paths, in page order — ready for ``enqueue_image_files``."""
        return [p.path for p in self.pages]


def read_bundle(source: Path, *, extract_dir: Path | None = None) -> BridgeBundle:
    """Read an ``.aglbundle`` from a directory or a ``.zip``.

    For a zip, ``extract_dir`` is required (where image files are materialized).
    """
    root = _resolve_root(source, extract_dir)
    manifest = _read_manifest(root)

    version = manifest.get("format_version")
    if version != SUPPORTED_FORMAT_VERSION:
        raise BridgeBundleError(
            f"unsupported format_version {version!r} (expected {SUPPORTED_FORMAT_VERSION})"
        )

    pages = _read_pages(root)
    return BridgeBundle(
        project=str(manifest.get("project") or "Imported scan"),
        device=str(manifest.get("device") or ""),
        dpi=_opt_int(manifest.get("dpi")),
        pages=pages,
    )


# --------------------------------------------------------------------------- #


def _resolve_root(source: Path, extract_dir: Path | None) -> Path:
    """Return the directory holding ``manifest.json`` (extracting a zip first)."""
    if source.is_dir():
        return _find_manifest_root(source)
    if zipfile.is_zipfile(source):
        if extract_dir is None:
            raise BridgeBundleError("extract_dir is required to read a zipped bundle")
        with zipfile.ZipFile(source) as zf:
            _safe_extract(zf, extract_dir)
        return _find_manifest_root(extract_dir)
    raise BridgeBundleError(f"not a bundle directory or zip: {source}")


def _find_manifest_root(directory: Path) -> Path:
    """Find the bundle root — ``directory`` itself, or its single wrapping
    subdirectory (a zip may nest everything under one folder)."""
    if (directory / "manifest.json").is_file():
        return directory
    subdirs = [p for p in directory.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "manifest.json").is_file():
        return subdirs[0]
    raise BridgeBundleError(f"manifest.json not found in {directory}")


def _read_manifest(root: Path) -> dict[str, object]:
    try:
        data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeBundleError(f"cannot read manifest.json: {exc}") from exc
    if not isinstance(data, dict):
        raise BridgeBundleError("manifest.json is not a JSON object")
    return data


def _read_pages(root: Path) -> list[BridgePage]:
    text = (root / "pages.jsonl").read_text(encoding="utf-8")
    pages: list[BridgePage] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeBundleError(f"bad pages.jsonl line: {exc}") from exc
        rel = str(rec.get("file") or "")
        img = root / rel
        if not rel or not img.is_file():
            raise BridgeBundleError(f"page image missing: {rel!r}")
        pages.append(
            BridgePage(
                index=int(rec.get("index", len(pages) + 1)),
                path=img,
                width=int(rec.get("w", 0)),
                height=int(rec.get("h", 0)),
            )
        )
    pages.sort(key=lambda p: p.index)
    return pages


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract, refusing path-traversal entries (zip-slip)."""
    dest = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if not target.is_relative_to(dest):
            raise BridgeBundleError(f"unsafe path in zip: {member!r}")
    zf.extractall(dest)


def _opt_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None
