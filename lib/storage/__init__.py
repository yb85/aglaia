# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from lib.storage.db import open_db, ensure_schema
from lib.storage.repo import (
    ProjectRepo, PipelineRepo, CalibrationRepo,
    ImageRepo, ThumbRepo, ScanRepo, NodeRepo, BranchRepo, StepOverrideRepo, DebugRepo,
)
from lib.storage.persister import Persister


# ── Aglaïa project file conventions ──────────────────────────────────
#
# A project IS a single SQLite file. As of 2026-06 the canonical suffix
# is `.agl`; older projects use `.scanproj.sqlite` and continue to load
# unchanged. Anything that *creates* a project should use PROJECT_EXT;
# anything that *opens* one should walk both via `project_db_candidates`.

PROJECT_EXT = ".agl"
LEGACY_PROJECT_EXT = ".scanproj.sqlite"

PROJECT_DIALOG_FILTER = (
    "Aglaïa projects (*.agl);;"
    "Legacy projects (*.scanproj.sqlite);;"
    "All files (*)"
)


def project_filename(slug: str) -> str:
    """Canonical project filename for a new project with `slug`."""
    return f"{slug}{PROJECT_EXT}"


def project_db_candidates(project_dir, slug: str):
    """Yield the most-preferred-first list of project DB paths for a
    project directory + slug. Order: canonical `.agl`, then legacy
    `.scanproj.sqlite`. Callers pick the first that exists; if none
    exists, the first entry is the path a new project would create."""
    from pathlib import Path
    project_dir = Path(project_dir)
    yield project_dir / f"{slug}{PROJECT_EXT}"
    yield project_dir / f"{slug}{LEGACY_PROJECT_EXT}"


def resolve_existing_project_db(project_dir, slug: str):
    """Return the on-disk `.agl` (or legacy) DB for `slug` under
    `project_dir`, or None if neither exists."""
    for cand in project_db_candidates(project_dir, slug):
        if cand.exists():
            return cand
    return None


def is_project_file(path) -> bool:
    """True if `path` looks like a project DB (either canonical or legacy)."""
    from pathlib import Path
    p = Path(path)
    return p.name.endswith(PROJECT_EXT) or p.name.endswith(LEGACY_PROJECT_EXT)


def slug_from_project_file(path) -> str:
    """Strip the project suffix (either ext) from a project DB filename."""
    from pathlib import Path
    name = Path(path).name
    if name.endswith(LEGACY_PROJECT_EXT):
        return name[: -len(LEGACY_PROJECT_EXT)]
    if name.endswith(PROJECT_EXT):
        return name[: -len(PROJECT_EXT)]
    return Path(path).stem
