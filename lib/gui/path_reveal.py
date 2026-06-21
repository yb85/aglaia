# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Reveal a file/dir in the OS file manager — cross-platform.

Used to make path strings throughout the GUI clickable (Finder on macOS,
Explorer on Windows, the default file manager on Linux). Best-effort:
never raises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def reveal_path(path: str | Path) -> None:
    """Reveal ``path`` in the OS file manager. A file is highlighted in its
    folder; a directory is opened."""
    p = Path(str(path)).expanduser()
    try:
        if sys.platform == "darwin":
            # `open -R` reveals a file; `open` opens a directory.
            if p.is_dir():
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["open", "-R", str(p)])
        elif sys.platform.startswith("win"):
            if p.is_dir():
                subprocess.Popen(["explorer", str(p)])
            else:
                # Note the required trailing comma in /select,
                subprocess.Popen(["explorer", f"/select,{p}"])
        else:
            target = p if p.is_dir() else p.parent
            subprocess.Popen(["xdg-open", str(target)])
    except Exception:
        pass


def make_label_paths_clickable(label) -> None:
    """Wire a rich-text QLabel whose ``<a href="…">`` targets are file
    paths so clicking them reveals the path (instead of opening a URL)."""
    label.setOpenExternalLinks(False)
    label.linkActivated.connect(reveal_path)
