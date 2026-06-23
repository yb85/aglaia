#!/usr/bin/env python3
"""Insert the PolyForm Shield license header into our Python source files.

Idempotent: re-running skips files that already carry the SPDX line. The
header is placed after any shebang / coding line and before the module
docstring (comments are not statements, so the docstring and a leading
`from __future__ import` keep their required first-statement positions).

Scope = git-tracked product code. Vendored skills (.claude/), scratch
(debug/, spike_jbig2/) and generated trees are excluded.

    uv run python scripts/add_license_headers.py          # apply
    uv run python scripts/add_license_headers.py --check   # list files missing the header (exit 1 if any)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HEADER = """\
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""

MARKER = "SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0"

INCLUDE_DIRS = ("aglaia/", "scripts/", "tests/", "scanproj/", "aglaia_jbig2/")
INCLUDE_FILES = {
    "aglaia.py", "inspect_binarization.py", "process_1ppf_empty.py",
    "ocr_run.py", "replay_chain.py", "migrate_page_order.py",
    "normalize_widths.py",
}
EXCLUDE_DIRS = (".claude/", "debug/", "spike_jbig2/", "site/")

_CODING = re.compile(r"^#.*coding[:=]")


def in_scope(rel: str) -> bool:
    if any(rel.startswith(d) for d in EXCLUDE_DIRS):
        return False
    if rel in INCLUDE_FILES:
        return True
    return any(rel.startswith(d) for d in INCLUDE_DIRS)


def tracked_py() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "*.py"], text=True)
    return [p for p in out.splitlines() if in_scope(p)]


def apply_header(text: str) -> str:
    """Insert the current HEADER, or replace an existing PolyForm header
    block in place (so the header can be edited and re-rolled out)."""
    lines = text.splitlines(keepends=True)
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    if i < len(lines) and _CODING.match(lines[i]):
        i += 1
    block = HEADER if HEADER.endswith("\n") else HEADER + "\n"

    # If a header block already sits at i (contiguous comment run holding
    # the marker), replace exactly those lines; otherwise insert.
    j = i
    while j < len(lines) and lines[j].startswith("#"):
        j += 1
    existing = "".join(lines[i:j])
    if MARKER in existing:
        return "".join(lines[:i]) + block + "".join(lines[j:])
    return "".join(lines[:i]) + block + "\n" + "".join(lines[i:])


def main() -> int:
    check = "--check" in sys.argv
    root = Path(__file__).resolve().parents[1]
    stale: list[str] = []
    changed = 0
    for rel in tracked_py():
        p = root / rel
        text = p.read_text(encoding="utf-8")
        new = apply_header(text)
        if new == text:
            continue
        stale.append(rel)
        if not check:
            p.write_text(new, encoding="utf-8")
            changed += 1
    if check:
        for m in stale:
            print(m)
        print(f"[headers] {len(stale)} file(s) missing/stale header")
        return 1 if stale else 0
    print(f"[headers] updated {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
