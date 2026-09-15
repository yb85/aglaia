# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR textpack export (#148) — the searchable PDF, its Markdown and the raw
OCR response in one TextBundle archive, in the format corpus reads.

    <stem>_OCR.textpack                (zip)
    └── <stem>_OCR.textbundle/
        ├── info.json                  TextBundle v2 + "corpus" block
        ├── text.md                    `<!-- page N -->` before each page
        └── assets/
            ├── source.pdf             searchable PDF (fixed name, always)
            └── mistral-<job>.jsonl    raw batch output, one per job

The contract is corpus's (`docs/OCR-TEXTPACK.md` in yb85/corpus, reference
writer `corpus/cloud_ocr.py`): same names, same `info.json` shape, same
provenance keys. Do not vary them — corpus codes against them.

The PDF and `text.md` come from ONE pass over the pages: `create_pdf_from_db`
reports which OCR result it laid on each PDF page, and `text.md` is written
from exactly that list, so page N of the Markdown is page N of the PDF.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: `cc.bibli.corpus.ocr.version` — bump when this writer's output changes.
TEXTPACK_VERSION = 1
CREATOR = "cc.bibli.corpus.ocr"


class TextpackError(RuntimeError):
    """The textpack cannot be built (no OCR, no pages)."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def text_md(pages: list[str]) -> str:
    """`text.md`: each page preceded by `<!-- page N -->`, N from 1."""
    return "".join(f"<!-- page {n} -->\n\n{md.strip()}\n\n"
                   for n, md in enumerate(pages, start=1))


def source_of(conn) -> tuple[Optional[Path], str]:
    """The original scan behind the project: ``(path or None, stem)``.

    A PDF import records ``<file>#<page>`` per scan; when every scan comes
    from one file, that file is the scan source. Otherwise (captures, several
    files) the project slug stands in."""
    refs = {(r["source_ref"] or "").rsplit("#", 1)[0] for r in conn.execute(
        "SELECT source_ref FROM scans WHERE deleted_at IS NULL "
        "AND source = 'pdf'")}
    n_all = conn.execute(
        "SELECT count(*) FROM scans WHERE deleted_at IS NULL").fetchone()[0]
    n_pdf = conn.execute(
        "SELECT count(*) FROM scans WHERE deleted_at IS NULL "
        "AND source = 'pdf'").fetchone()[0]
    if len(refs) == 1 and n_pdf == n_all and n_all:
        p = Path(next(iter(refs)))
        return p, p.stem
    row = conn.execute("SELECT slug FROM project").fetchone()
    return None, (row["slug"] if row else "aglaia")


def _raw_outputs(conn, run_ids: list) -> list[dict]:
    """Stored batch outputs of the jobs that produced these runs, oldest
    first: ``{job_id, raw, submitted_at, completed_at}``."""
    from aglaia.storage.repo import MistralBatchRepo
    from aglaia.workers.ocr.mistral_batch import iso_time
    wanted = {r for r in run_ids if r is not None}
    repo = MistralBatchRepo(conn)
    out = []
    for job in conn.execute(
            "SELECT * FROM mistral_batch_jobs ORDER BY submitted_at, chunk"):
        if not wanted.intersection(MistralBatchRepo.run_ids_of(job)):
            continue
        raw = repo.output(job["job_id"])
        stored = conn.execute(
            "SELECT completed_at FROM mistral_batch_outputs WHERE job_id = ?",
            (job["job_id"],)).fetchone()
        out.append({"job_id": job["job_id"], "raw": raw,
                    "submitted_at": job["submitted_at"],
                    "completed_at": (iso_time(stored["completed_at"])
                                     if stored else None)})
    return out


def _billed_pages(raw: bytes) -> Optional[int]:
    """Sum of `usage_info.pages_processed` over a batch output's requests."""
    total, seen = 0, False
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        resp = obj.get("response") or {}
        body = (resp.get("body") if isinstance(resp, dict) else None) or resp
        n = ((body or {}).get("usage_info") or {}).get("pages_processed")
        if n is not None:
            total += int(n)
            seen = True
    return total if seen else None


def provenance(conn, *, results: list, run_ids: list, raw_assets: list[str],
               jobs: list[dict], source_sha256: Optional[str]) -> dict[str, Any]:
    """`corpus.ocr` — the provenance keys of corpus's reference writer."""
    from aglaia.version import get_version
    from aglaia.workers.ocr.mistral_cloud import (
        MODEL, PRICE_PER_PAGE_USD, PRICE_PER_PAGE_USD_BATCH)

    done = [r for r in results if r]
    engines = sorted({str(r.get("engine") or "") for r in done})
    engine = engines[0] if len(engines) == 1 else "+".join(engines)
    metas = [r.get("meta") or {} for r in done]
    mistral = engine == "mistral_cloud"
    batch = mistral and any(m.get("batch") for m in metas)
    pages = len(results)

    billed: Optional[int] = None
    if jobs and all(j["raw"] is not None for j in jobs):
        counts = [_billed_pages(j["raw"]) for j in jobs]
        billed = sum(c for c in counts if c is not None) if any(
            c is not None for c in counts) else None
    if mistral and batch:
        n = billed if billed is not None else len(done)
        cost = n * PRICE_PER_PAGE_USD_BATCH
        basis = ("pages processed per Mistral (usage_info) × $2/1000, batch rate"
                 if billed is not None else
                 "OCR'd pages × $2/1000, batch rate (usage_info not stored)")
    elif mistral:
        cost = len(done) * PRICE_PER_PAGE_USD
        basis = "OCR'd pages × $4/1000, standard rate (estimate)"
    else:
        cost = 0.0
        basis = "local engine, no charge"

    ids = [r for r in run_ids if r is not None]
    finished = None
    if ids:
        q = ",".join("?" * len(ids))
        row = conn.execute(
            f"SELECT min(started_at) AS s, max(finished_at) AS f "
            f"FROM ocr_runs WHERE id IN ({q})", ids).fetchone()
        finished = row["f"]
        started = row["s"]
    else:
        started = None
    submitted = min((j["submitted_at"] for j in jobs if j["submitted_at"]),
                    default=started)
    completed = max((j["completed_at"] for j in jobs if j["completed_at"]),
                    default=finished)
    models = sorted({str(m.get("model")) for m in metas if m.get("model")})
    return {
        "engine": "mistral-ocr" if mistral else engine,
        "model": (models[0] if len(models) == 1 else "+".join(models))
                 or (MODEL if mistral else engine),
        "mode": "batch" if batch else ("sync" if mistral else "local"),
        "jobs": [j["job_id"] for j in jobs],
        "requests": [],
        "pages": pages,
        "page_numbers": None,
        "pages_document": pages,
        "tokens": None,
        "billed_pages": billed,
        "cost_usd": round(cost, 6),
        "cost_basis": basis,
        "submitted_at": submitted,
        "completed_at": completed,
        "source_sha256": source_sha256,
        "tool": f"aglaia {get_version()}",
        "raw_assets": sorted(raw_assets),
    }


def info_json(*, stem: str, scan_source: str, prov: dict[str, Any],
              double_page: bool, zlib_id: Optional[int] = None) -> dict[str, Any]:
    corpus: dict[str, Any] = {"scan_source": scan_source, "ocr": dict(prov),
                              "double_page": double_page, "date": _now()}
    if zlib_id is not None:
        corpus["zlib_id"] = zlib_id
    return {
        "version": 2,
        "type": "net.daringfireball.markdown",
        "transient": False,
        "creatorIdentifier": CREATOR,
        "corpus": corpus,
        CREATOR: {"version": TEXTPACK_VERSION, "stem": stem},
    }


def write_archive(destination: Path, *, stem: str, info: dict[str, Any],
                  markdown: str, pdf: bytes, raws: dict[str, bytes]) -> Path:
    """Write the zip next to `destination`, then rename: never half a file."""
    for name in raws:
        if "/" in name or name in ("", "source.pdf") or name.startswith("."):
            raise ValueError(f"refused asset name: {name!r}")
    root = f"{stem}_OCR.textbundle"
    partial = destination.with_name(destination.name + ".partiel")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{root}/info.json",
                       json.dumps(info, ensure_ascii=False, indent=2))
            z.writestr(f"{root}/text.md", markdown)
            z.writestr(f"{root}/assets/source.pdf", pdf,
                       compress_type=zipfile.ZIP_STORED)
            for name, data in raws.items():
                z.writestr(f"{root}/assets/{name}", data)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def default_name(conn) -> str:
    """`<stem>_OCR.textpack`, stem = the original scan's."""
    return f"{source_of(conn)[1]}_OCR.textpack"


def write_textpack(conn, destination, *, compression: str = "auto",
                   engine: Optional[str] = None, zlib_id: Optional[int] = None,
                   scan_source: Optional[str] = None,
                   stem: Optional[str] = None) -> Path:
    """Build the OCR textpack of the project at `destination`.

    Raises `TextpackError` when there is nothing to pack, and lets
    `OcrLayerError` through: an archive whose `source.pdf` is not searchable
    is exactly what corpus must not receive (#149)."""
    from aglaia.workers.md_export import markdown_pages
    from aglaia.workers.PDFprocessor import create_pdf_from_db

    destination = Path(destination)
    src_path, src_stem = source_of(conn)
    stem = stem or src_stem
    scan_source = scan_source or (src_path.name if src_path else stem)
    sha = None
    if src_path is not None and src_path.is_file():
        sha = hashlib.sha256(src_path.read_bytes()).hexdigest()

    layer: dict[str, list] = {}
    with tempfile.TemporaryDirectory(prefix="aglaia-textpack-") as tmp:
        pdf_path = Path(tmp) / "source.pdf"
        if not create_pdf_from_db(conn, pdf_path, compression=compression,
                                  add_ocr_layer=True, engine=engine,
                                  layer=layer):
            raise TextpackError("no page to export")
        pdf = pdf_path.read_bytes()
    results, run_ids = layer["results"], layer["run_ids"]

    jobs = _raw_outputs(conn, run_ids)
    lacking = [j["job_id"] for j in jobs if j["raw"] is None]
    if lacking:
        # The raw response is part of the contract: it is what lets corpus
        # rebuild the text without paying for the OCR again.
        raise TextpackError(
            f"raw Mistral output not stored for job(s) {', '.join(lacking)} "
            f"— run `aglaia run <project>.agl --check-ocr` to fetch it")
    raws = {f"mistral-{j['job_id']}.jsonl": j["raw"]
            for j in jobs if j["raw"] is not None}
    prov = provenance(conn, results=results, run_ids=run_ids,
                      raw_assets=list(raws), jobs=jobs, source_sha256=sha)
    double = bool(conn.execute(
        "SELECT 1 FROM branches b JOIN scans s ON s.id = b.scan_id "
        "WHERE s.deleted_at IS NULL AND b.trashed_at IS NULL "
        "GROUP BY b.scan_id HAVING count(*) > 1 LIMIT 1").fetchone())
    info = info_json(stem=stem, scan_source=scan_source, prov=prov,
                     double_page=double, zlib_id=zlib_id)
    return write_archive(destination, stem=stem, info=info,
                         markdown=text_md(markdown_pages(results)),
                         pdf=pdf, raws=raws)
