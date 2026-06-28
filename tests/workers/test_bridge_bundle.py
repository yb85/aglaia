"""Tests for the aglaia-bridge .aglbundle reader (issue #47)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aglaia.workers.bridge_bundle import (
    BridgeBundleError,
    read_bundle,
)


def _make_bundle(root: Path, *, dpi: int | None = 150, pages: int = 3) -> Path:
    """Write a minimal valid .aglbundle directory and return its path."""
    bundle = root / "Book.aglbundle"
    (bundle / "images").mkdir(parents=True)
    manifest = {
        "format_version": 1,
        "project": "Book",
        "created": "2026-06-28T12:00:00Z",
        "device": "Test iPhone",
        "page_count": pages,
    }
    if dpi is not None:
        manifest["dpi"] = dpi
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    lines = []
    # Write out of order to prove the reader sorts by index.
    for i in reversed(range(1, pages + 1)):
        name = f"{i:04d}.jpg"
        (bundle / "images" / name).write_bytes(b"\xff\xd8\xff\xd9")
        lines.append(
            json.dumps(
                {"index": i, "file": f"images/{name}", "captured": "2026-06-28T12:00:00Z", "w": 100, "h": 200}
            )
        )
    (bundle / "pages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bundle


def test_reads_directory_bundle_in_page_order(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, dpi=150, pages=3)
    result = read_bundle(bundle)

    assert result.project == "Book"
    assert result.device == "Test iPhone"
    assert result.dpi == 150
    assert [p.index for p in result.pages] == [1, 2, 3]
    assert [p.name for p in result.image_paths] == ["0001.jpg", "0002.jpg", "0003.jpg"]
    assert all(p.is_file() for p in result.image_paths)


def test_reads_zipped_bundle(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "src", dpi=110, pages=2)
    zip_path = tmp_path / "Book.aglbundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in bundle.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(bundle.parent))  # nested under "Book.aglbundle/"

    result = read_bundle(zip_path, extract_dir=tmp_path / "out")
    assert result.dpi == 110
    assert [p.index for p in result.pages] == [1, 2]


def test_missing_dpi_is_none(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, dpi=None, pages=1)
    assert read_bundle(bundle).dpi is None


def test_rejects_unsupported_format_version(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, pages=1)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["format_version"] = 99
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BridgeBundleError, match="format_version"):
        read_bundle(bundle)


def test_rejects_missing_image(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, pages=1)
    (bundle / "images" / "0001.jpg").unlink()
    with pytest.raises(BridgeBundleError, match="missing"):
        read_bundle(bundle)
