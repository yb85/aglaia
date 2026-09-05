#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
"""Build `index.json` for an aglaia-plugins checkout.

The index is what the app reads: one entry per plugin, carrying its metadata,
its declared capabilities, and a sha256 for every file. The client refuses a
download that does not match, so this file is the integrity anchor — which is
why it is GENERATED and never hand-edited, and why CI regenerating it must be
a no-op diff.

    python scripts/build_plugin_index.py /path/to/aglaia-plugins [--check]

`--check` regenerates and exits non-zero if the result differs from what is
committed — the shape a CI gate wants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aglaia.app_data.plugin_manifest import (  # noqa: E402
    KINDS, ManifestError, parse_manifest, scan_plugin_dir)

#: Files that travel with a plugin. Anything else in the directory is repo
#: furniture (tests, fixtures) and is not shipped to a user.
SHIPPED = ("aglaia-plugin.toml", "README.md", "LICENSE")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def collect(root: Path, repo: str) -> tuple[dict, list[str]]:
    sha = _commit(root)
    out: dict = {"schema": 1, "api": 1,
                 "generated_at": datetime.now(timezone.utc).strftime(
                     "%Y-%m-%dT%H:%M:%SZ"),
                 "plugins": [], "revoked": []}
    problems: list[str] = []
    revoked_file = root / "revoked.json"
    if revoked_file.is_file():
        try:
            out["revoked"] = json.loads(revoked_file.read_text("utf-8"))
        except Exception as e:
            problems.append(f"revoked.json: {e}")

    for kind in KINDS:
        kdir = root / kind
        if not kdir.is_dir():
            continue
        for d in sorted(kdir.iterdir()):
            if not d.is_dir() or d.name.startswith(("." , "_")):
                continue
            try:
                man = parse_manifest(d / "aglaia-plugin.toml", kind=kind,
                                     expect_slug=d.name)
            except ManifestError as e:
                problems.append(f"{kind}/{d.name}: {e}")
                continue

            tops = sorted(p.name for p in d.glob("*.py"))
            if tops != [man.entry]:
                problems.append(
                    f"{kind}/{d.name}: a plugin is ONE top-level module; "
                    f"found {tops or 'none'}, manifest says {man.entry!r}")
                continue
            if not (d / "LICENSE").is_file():
                problems.append(f"{kind}/{d.name}: no LICENSE")
            scan = scan_plugin_dir(d, man)
            if not scan.clean:
                problems.append(f"{kind}/{d.name}: import scan — "
                                f"{scan.summary()}")
                continue

            files: dict[str, str] = {}
            for name in (man.entry, *SHIPPED):
                f = d / name
                if f.is_file():
                    files[name] = f"sha256:{_sha256(f)}"
            for sup in sorted(d.glob("_*/**/*.py")):
                files[str(sup.relative_to(d))] = f"sha256:{_sha256(sup)}"

            entry = {
                "slug": man.slug, "kind": kind, "name": man.name,
                "version": man.version, "summary": man.summary,
                "author": man.author, "license": man.license,
                "homepage": man.homepage,
                "source_url": (f"https://github.com/{repo}/tree/{sha}/"
                               f"{kind}/{man.slug}" if sha else ""),
                "requires": {"aglaia": man.requires_aglaia,
                             "python": man.requires_python},
                "api": man.api,
                "capabilities": man.capabilities,
                "imports": list(man.imports),
                "files": files,
            }
            out["plugins"].append(entry)
    out["plugins"].sort(key=lambda e: (e["kind"], e["slug"]))
    return out, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--repo", default="yb85/aglaia-plugins")
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed index differs")
    args = ap.parse_args()

    index, problems = collect(args.root, args.repo)
    for p in problems:
        print(f"✗ {p}", file=sys.stderr)
    if problems:
        return 1

    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    target = args.root / "index.json"
    if args.check:
        # `generated_at` and the commit sha move on every run, so compare
        # everything else — a timestamp is not a change to the index.
        try:
            old = json.loads(target.read_text("utf-8"))
        except Exception as e:
            print(f"✗ cannot read {target}: {e}", file=sys.stderr)
            return 1
        for doc in (old, index):
            doc.pop("generated_at", None)
            for e in doc.get("plugins", []):
                e.pop("source_url", None)
        if old != index:
            print("✗ index.json is stale — regenerate it", file=sys.stderr)
            return 1
        print("✓ index.json is current")
        return 0

    target.write_text(text, encoding="utf-8")
    print(f"✓ {len(index['plugins'])} plugin(s) → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
