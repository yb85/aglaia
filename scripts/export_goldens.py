#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Export per-stage golden fixtures from an .agl project file.

Extraction only — does not run the pipeline. For each scan the full node
tree (every stage output image + its meta_json, incl. replay params) is
written to a directory layout consumed by aglaia-ios parity tests:

    <out>/<set>/
      pipeline.yaml                  # active pipeline_versions.yaml_text
      manifest.json                  # scans, dpi, node index
      scan1/
        00_raw.jpg
        02_capture_deskew.jpg  02_capture_deskew.json
        03_pages_2ppf.A.jpg    03_pages_2ppf.A.json
        ...
        10_replay.A.png        10_replay.A.json

Pure stdlib (sqlite3) — runs without the aglaia venv.

    python3 scripts/export_goldens.py test_data/test_augustin/*.agl -o /tmp/goldens
    python3 scripts/export_goldens.py test_data/*/*.agl -o ../aglaia-ios/AglaiaCore/Tests/AglaiaCoreTests/Fixtures
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

EXT = {"JPG": "jpg", "PNG": "png"}


def slug(agl: Path) -> str:
    """test-augustin-confessions-vii.agl -> augustin (short set name)."""
    m = re.match(r"test[-_]?([a-z0-9]+)", agl.stem, re.IGNORECASE)
    return (m.group(1) if m else agl.stem).lower()


def export(agl: Path, out_root: Path) -> Path:
    db = sqlite3.connect(f"file:{agl}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    out = out_root / slug(agl)
    out.mkdir(parents=True, exist_ok=True)

    # The version the scans were actually processed with — NOT is_active
    # (the project may have a newer active pipeline than its existing nodes).
    pids = [
        r[0]
        for r in db.execute(
            "SELECT DISTINCT pipeline_version_id FROM scans WHERE deleted_at IS NULL"
        )
    ]
    if len(pids) != 1:
        print(f"warning: {agl.name} has scans from {len(pids)} pipeline versions; using first")
    pipe = db.execute(
        "SELECT name, yaml_text, yaml_sha256 FROM pipeline_versions WHERE id=?", (pids[0],)
    ).fetchone()
    (out / "pipeline.yaml").write_text(pipe["yaml_text"])

    manifest: dict = {
        "source_agl": agl.name,
        "pipeline": {"name": pipe["name"], "yaml_sha256": pipe["yaml_sha256"]},
        "scans": [],
    }

    scans = db.execute(
        "SELECT id, idx, source, capture_dpi FROM scans WHERE deleted_at IS NULL ORDER BY idx"
    ).fetchall()
    for scan in scans:
        scan_dir = out / f"scan{scan['idx']}"
        scan_dir.mkdir(exist_ok=True)
        nodes = db.execute(
            """SELECT n.step_idx, n.step_name, n.processor_name, n.branch_label,
                      n.is_leaf, n.meta_json, n.status_int,
                      i.format, i.type, i.width, i.height, i.dpi, i.blob
               FROM nodes n LEFT JOIN images i ON i.id = n.image_id
               WHERE n.scan_id = ? ORDER BY n.step_idx, n.branch_label""",
            (scan["id"],),
        ).fetchall()
        index = []
        for n in nodes:
            branch = f".{n['branch_label']}" if n["branch_label"] else ""
            stem = f"{n['step_name'] or '00_raw'}{branch}"
            entry = {
                "step_idx": n["step_idx"],
                "step_name": n["step_name"],
                "processor": n["processor_name"],
                "branch": n["branch_label"],
                "is_leaf": bool(n["is_leaf"]),
                "status": n["status_int"],
            }
            if n["blob"] is not None:
                img = f"{stem}.{EXT.get(n['format'], 'bin')}"
                (scan_dir / img).write_bytes(n["blob"])
                entry |= {
                    "image": img,
                    "image_type": n["type"],
                    "width": n["width"],
                    "height": n["height"],
                    "dpi": n["dpi"],
                }
            if n["meta_json"]:
                meta_file = f"{stem}.json"
                # round-trip for stable formatting
                (scan_dir / meta_file).write_text(
                    json.dumps(json.loads(n["meta_json"]), indent=1, sort_keys=True)
                )
                entry["meta"] = meta_file
            index.append(entry)
        manifest["scans"].append(
            {
                "idx": scan["idx"],
                "source": scan["source"],
                "capture_dpi": scan["capture_dpi"],
                "dir": f"scan{scan['idx']}",
                "nodes": index,
            }
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    db.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agl", nargs="+", type=Path, help=".agl file(s) to export")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output root dir")
    args = ap.parse_args()
    for agl in args.agl:
        out = export(agl, args.out)
        n_files = sum(1 for _ in out.rglob("*") if _.is_file())
        print(f"{agl.name} -> {out} ({n_files} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
