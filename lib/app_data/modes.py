# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Curated, beginner-friendly capture "modes".

Each mode is a friendly name + icon + one-line "good for" blurb bound to a
shipped pipeline yaml (seeded into `<APP_DATA>/pipelines`). The startup
new-project picker shows these as cards; advanced users edit the underlying
pipeline (or pick any other yaml) via the pipeline editor. Strings are
English for now — i18n later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lib.assets import asset_path as _asset_path
_ICON_DIR = _asset_path("modes")


@dataclass(frozen=True)
class PipelineMode:
    key: str
    name: str
    description: str
    pipeline: str  # yaml filename, resolved under app_data.pipelines_dir()
    icon: str      # basename in assets/modes/ (no extension)

    def icon_path(self, *, prefer_svg: bool = True) -> Optional[Path]:
        exts = ("svg", "png") if prefer_svg else ("png", "svg")
        for ext in exts:
            p = _ICON_DIR / f"{self.icon}.{ext}"
            if p.is_file():
                return p
        return None


# Order = display order in the picker.
MODES: tuple[PipelineMode, ...] = (
    PipelineMode(
        key="book_curved_x2",
        name="Book — curved pages",
        description="Open book with pages that bulge near the spine. Two "
                    "pages per photo, deskewed and flattened (dewarp). Best "
                    "for thick or tightly-bound books.",
        pipeline="book_curved_x2.yaml",
        icon="book_curved_x2",
    ),
    PipelineMode(
        key="book_flat_x2",
        name="Book — flat",
        description="Open book pressed flat. Two pages per photo, "
                    "perspective-corrected — no dewarp.",
        pipeline="book_flat_x2.yaml",
        icon="book_flat_x2",
    ),
    PipelineMode(
        key="sheet_flat_x1",
        name="Paper sheets",
        description="Loose sheets or printouts, one page per photo. "
                    "Deskewed and keystone-corrected.",
        pipeline="sheet_flat_x1.yaml",
        icon="sheet_flat_x1",
    ),
    PipelineMode(
        key="book_flat_x1",
        name="Book — one page",
        description="One book page at a time, flat. Text bleeding in from "
                    "the facing page is discarded.",
        pipeline="book_flat_x1.yaml",
        icon="book_flat_x1_reject",
    ),
)


def modes() -> list[PipelineMode]:
    return list(MODES)


def mode_for_pipeline(filename: str) -> Optional[PipelineMode]:
    """Map a pipeline yaml filename back to its curated mode, if any."""
    name = Path(filename).name
    for m in MODES:
        if m.pipeline == name:
            return m
    return None
