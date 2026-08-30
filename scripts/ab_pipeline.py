#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A/B pipeline variants over the committed fixture corpus.

Answers "is this change actually better?" by running the REAL CLI once per
variant and measuring the OUTPUT, not the objective. Every dewarp decision on
record here was made with it — retiring the twist/spline sheet models, adopting
the principal-point DOF, sizing the spine-curl grid.

Why measure the output. A sheet fit is scored by its objective, and the
objective cannot see a bad page: a lower objective happily coexists with a
remap that overshoots the frame (measured — a spine-curl γ at the grid edge
beat γ = 0 on objective while blowing the out-of-bounds gate). The number that
matters is whether the text comes out straight.

Two findings this harness produced that nothing else would have:

  * `page_side` was being dropped one step before PageDewarper read it, so
    every binding-side feature had been silently inert. It surfaced as two
    variants scoring *identically* when they had no business doing so — a
    suspicious tie is a bug until proven otherwise.
  * Pages were vanishing from slow runs (no node, no error, exit 0). It
    surfaced from the pages-detected-vs-dewarped column below, which is why
    that column is not optional.

How much delta to believe. Runs are not bit-deterministic: the warm-start ring
seeds each page's curl from recent same-side fits, so the ORDER pages happen to
complete — which varies with worker scheduling — can shift a page slightly.
Observed drift on repeat runs of an identical configuration is ~0.002 px, so
treat sub-1% differences as noise and re-run before believing one. Aggregate
gaps of the size that decided anything here (0.92 vs 1.65) are far outside it.

Usage:

    uv run python scripts/ab_pipeline.py \\
        --variant "baseline:" \\
        --variant "flat-camera:PageDewarper.principal_point=false" \\
        --variant "no-gamma:PageDewarper.spine_gammas="

Each `--variant` is `label:key=value[,key=value…]`, where key is
`<StepProcessor>.<option>` (an empty override list = the pipeline as committed).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
#: The committed corpus: three books, two facing pages each. Small enough to
#: iterate on, varied enough that a change that only helps one book shows up.
DEFAULT_IMAGES = [
    "test_data/test_athanase/athanase_150.jpg",
    "test_data/test_athanase/athanase_151.jpg",
    "test_data/test_augustin/augustin_286.jpg",
    "test_data/test_augustin/augustin_287.jpg",
    "test_data/test_balthasar/balthasar-theologique-iii_100.jpg",
    "test_data/test_balthasar/balthasar-theologique-iii_101.jpg",
]


def baseline_curvature(img: np.ndarray) -> tuple[float, int]:
    """Mean deviation of the output text baselines from a straight line, px.

    A proxy, and worth knowing its limits: it only sees lines the morphology
    finds (full-width, not too tall), it measures residual curvature *after*
    the remap, and it says nothing about horizontal scale. It is sensitive to
    exactly what dewarping is for, which is what makes it the right headline
    number — but read it alongside the page count, not instead of it."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (max(9, g.shape[1] // 40), 1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel), 8)
    devs = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if w < g.shape[1] * 0.3 or h > g.shape[0] * 0.05:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        if xs.size < 50:
            continue
        cols = np.unique(xs).astype(float)
        base = np.array([ys[xs == int(c)].max() for c in cols], float)
        fit = np.polyval(np.polyfit(cols, base, 1), cols)
        devs.append(float(np.mean(np.abs(base - fit))))
    return (float(np.mean(devs)), len(devs)) if devs else (float("nan"), 0)


def _write_variant_yaml(base_yaml: Path, overrides: dict, out: Path) -> None:
    spec = yaml.safe_load(base_yaml.read_text())
    for dotted, value in overrides.items():
        proc, _, key = dotted.partition(".")
        hit = False
        for step in spec.get("pipeline", []):
            if step.get("processor") == proc:
                step.setdefault("options", {})[key] = value
                hit = True
        if not hit:
            sys.exit(f"no step with processor {proc!r} in {base_yaml.name}")
    out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))


def _coerce(raw: str):
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("", "none", "null"):
        return ""
    try:
        return int(raw) if re.fullmatch(r"-?\d+", raw.strip()) else float(raw)
    except ValueError:
        return raw


def run_variant(label: str, yaml_path: Path, images: list[str],
                out_dir: Path) -> float:
    """Drive the real CLI, in its own process group so a stray run can be
    reaped as a group (spawn children reparent and survive a bare kill)."""
    t0 = time.time()
    log = out_dir / f"{label}.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(
            ["uv", "run", "--no-sync", "aglaia", "run", *images,
             "-p", str(yaml_path), "--project-name", label,
             "--parent-dir", str(out_dir)],
            cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True)
        rc = proc.wait()
    if rc != 0:
        print(f"  ! {label}: CLI exited {rc} (see {log})", file=sys.stderr)
    return time.time() - t0


def score(agl: Path) -> dict:
    conn = sqlite3.connect(agl)
    try:
        detected = {f"{s}{b}" for s, b in conn.execute(
            "select scan_id, branch_label from nodes "
            "where processor_name='PageDetector'")}
        rows = list(conn.execute(
            "select scan_id, branch_label, meta_json, elapsed_ms, image_id "
            "from nodes where processor_name='PageDewarper' "
            "order by scan_id, branch_label"))
        pages, ms, ok = {}, [], 0
        for sid, lab, meta, elapsed, image_id in rows:
            blob = conn.execute("select blob from images where id=?",
                                (image_id,)).fetchone()[0]
            img = cv2.imdecode(np.frombuffer(blob, np.uint8),
                               cv2.IMREAD_UNCHANGED)
            curv, _lines = baseline_curvature(img)
            pages[f"{sid}{lab}"] = curv
            ms.append(elapsed or 0.0)
            ok += bool(json.loads(meta or "{}").get("success"))
    finally:
        conn.close()
    return {"pages": pages, "ms": ms, "ok": ok,
            "lost": sorted(detected - set(pages))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", default="book_curved_x2",
                    help="pipeline name under aglaia/config/pipelines/")
    ap.add_argument("--variant", action="append", required=True,
                    metavar="LABEL:KEY=VAL[,KEY=VAL]")
    ap.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    ap.add_argument("--out", default=None,
                    help="work dir (default: a fresh temp dir)")
    args = ap.parse_args()

    base = REPO / "aglaia/config/pipelines" / f"{args.pipeline}.yaml"
    if not base.exists():
        return int(bool(sys.stderr.write(f"no pipeline {base}\n")))
    out_dir = Path(args.out) if args.out else Path(
        tempfile.mkdtemp(prefix="aglaia-ab-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"corpus: {len(args.images)} image(s)   work dir: {out_dir}\n")

    results = {}
    for spec in args.variant:
        label, _, raw = spec.partition(":")
        overrides = {}
        for pair in filter(None, (p.strip() for p in raw.split(","))):
            k, _, v = pair.partition("=")
            overrides[k.strip()] = _coerce(v)
        vyaml = out_dir / f"{label}.yaml"
        _write_variant_yaml(base, overrides, vyaml)
        print(f"running {label} … ", end="", flush=True)
        wall = run_variant(label, vyaml, args.images, out_dir)
        agl = out_dir / f"{label}.agl"
        if not agl.exists():
            print("no project produced")
            continue
        results[label] = score(agl)
        print(f"{wall:.0f}s")

    if not results:
        return 1
    w = max(len(k) for k in results) + 2
    print(f"\n{'variant':<{w}}{'pages':>7}{'ok':>4}{'mean':>9}"
          f"{'median':>9}{'worst':>8}{'ms/page':>9}  lost")
    for label, r in results.items():
        c = np.array(list(r["pages"].values()), float)
        print(f"{label:<{w}}{len(c):>7}{r['ok']:>4}{np.nanmean(c):>9.3f}"
              f"{np.nanmedian(c):>9.3f}{np.nanmax(c):>8.3f}"
              f"{np.mean(r['ms']):>9.0f}  {r['lost'] or '-'}")

    order = list(results)
    pages = sorted(results[order[0]]["pages"])
    print(f"\n{'page':<8}" + "".join(f"{k:>14}" for k in order))
    for pg in pages:
        print(f"{pg:<8}" + "".join(
            f"{results[k]['pages'].get(pg, float('nan')):>14.3f}"
            for k in order))

    if any(r["lost"] for r in results.values()):
        print("\n!! pages were DETECTED but never dewarped — the run dropped "
              "work. That is a bug in the chain, not a quality result.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
