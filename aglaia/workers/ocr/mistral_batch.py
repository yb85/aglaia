# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
"""Mistral **Batch** OCR — submit an async `/v1/ocr` batch job (cheaper than
the synchronous path), then poll / fetch / cancel.

Reuses :class:`MistralCloudEngine`'s capped-PDF assembly so a batch upload
respects the same 1000-page / 50 MB per-document caps; oversized selections
are split into multiple chunks, one batch job each (the project full path +
chunk number ride along in the job metadata so the Jobs tab can show, and
re-open, the owning project).

Status values (Mistral): QUEUED, RUNNING, SUCCESS, FAILED, TIMEOUT_EXCEEDED,
CANCELLATION_REQUESTED, CANCELLED.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .engine import OcrResult, engine_log
from .mistral_cloud import MODEL, MistralCloudEngine

# Job metadata keys (arbitrary strings on the Mistral batch job).
META_APP = "app"
META_APP_VALUE = "aglaia"
META_PROJECT = "aglaia_project"        # absolute path to the .agl
META_CHUNK = "aglaia_chunk"            # 0-based chunk index (only when split)
META_CHUNKS_TOTAL = "aglaia_chunks_total"
ENDPOINT = "/v1/ocr"

FAILED_STATUSES = ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED")


def _client(api_key: str):
    from .mistral_cloud import make_mistral_client
    return make_mistral_client(api_key)


def _norm_status(s) -> str:
    """Normalise a job status to a bare upper-case string. The SDK may hand
    back a str-enum, an enum whose ``str()`` is ``BatchJobStatus.SUCCESS``,
    or mixed case — so `get` and `list` agreed on the wire but compared
    unequal in code ('Check result' said *processing* while the Jobs table
    said *SUCCESS*)."""
    s = getattr(s, "value", s)
    return str(s or "").rsplit(".", 1)[-1].strip().upper()


# ── PDF chunking ─────────────────────────────────────────────────────────
def _chunk_pdfs(eng: MistralCloudEngine, img_rows: list[dict]
                ) -> list[tuple[bytes, int]]:
    """Split ``img_rows`` into consecutive chunks, each a single PDF that
    fits Mistral's page/byte caps. Returns ``[(pdf_bytes, n_rows), …]``
    covering the rows in order (so chunk *k* owns the next ``n_rows`` rows)."""
    chunks: list[tuple[bytes, int]] = []
    remaining = list(img_rows)
    with tempfile.TemporaryDirectory(prefix="aglaia-mistral-batch-") as td:
        i = 0
        while remaining:
            pdf_path = Path(td) / f"chunk-{i}.pdf"
            pdf_bytes, n = eng._build_capped_pdf_from_rows(remaining, pdf_path)
            if n <= 0:
                raise RuntimeError("Failed to assemble a batch PDF chunk.")
            chunks.append((pdf_bytes, n))
            remaining = remaining[n:]
            i += 1
    return chunks


# ── submit ───────────────────────────────────────────────────────────────
def submit(api_key: str, img_rows: list[dict], run_ids: list[int],
           project_path: str) -> list[dict]:
    """Upload + create one batch job per chunk. Returns a list of dicts:
    ``{job_id, input_file_id, chunk, chunks_total, page_count, run_ids}`` —
    persist these (MistralBatchRepo.add) so results can be pulled later."""
    if len(run_ids) != len(img_rows):
        raise ValueError("run_ids must align 1:1 with img_rows")
    eng = MistralCloudEngine()
    client = _client(api_key)
    chunks = _chunk_pdfs(eng, img_rows)
    total = len(chunks)
    out: list[dict] = []
    offset = 0
    for ci, (pdf_bytes, n) in enumerate(chunks):
        uploaded = client.files.upload(
            file={"file_name": f"aglaia-ocr-{ci}.pdf", "content": pdf_bytes},
            purpose="ocr")
        signed = client.files.get_signed_url(file_id=uploaded.id)
        # One OCR request line per chunk PDF.
        line = {"custom_id": "0", "body": {
            "document": {"type": "document_url", "document_url": signed.url},
            "include_image_base64": False}}
        jsonl = (json.dumps(line) + "\n").encode("utf-8")
        binput = client.files.upload(
            file={"file_name": f"aglaia-batch-{ci}.jsonl", "content": jsonl},
            purpose="batch")
        meta = {META_APP: META_APP_VALUE, META_PROJECT: str(project_path)}
        if total > 1:
            meta[META_CHUNK] = str(ci)
            meta[META_CHUNKS_TOTAL] = str(total)
        job = client.batch.jobs.create(
            input_files=[binput.id], model=MODEL, endpoint=ENDPOINT,
            metadata=meta)
        out.append({
            "job_id": job.id, "input_file_id": binput.id, "chunk": ci,
            "chunks_total": total, "page_count": n,
            "run_ids": run_ids[offset:offset + n],
        })
        engine_log(f"[mistral_batch] submitted job {job.id} "
                   f"(chunk {ci + 1}/{total}, {n} page(s))", "info")
        offset += n
    return out


# ── poll / fetch / cancel ────────────────────────────────────────────────
def poll(api_key: str, job_id: str) -> tuple[str, Optional[str]]:
    """Return ``(status, error_text|None)`` for a job."""
    client = _client(api_key)   # keep a ref — an inline temporary's httpx
    job = client.batch.jobs.get(job_id=job_id)  # client gets torn down mid-call
    status = _norm_status(getattr(job, "status", ""))
    err = getattr(job, "errors", None)
    return status, (str(err) if err else None)


def fetch_pages(api_key: str, job_id: str) -> list[dict]:
    """Download a SUCCESS job's output and return the per-page OCR objects
    (full OCR-4 structure: markdown + typed blocks / bboxes / confidence) in
    page order. Raises if the job isn't SUCCESS or the output is unreadable."""
    client = _client(api_key)
    job = client.batch.jobs.get(job_id=job_id)
    status = _norm_status(getattr(job, "status", ""))
    if status != "SUCCESS":
        raise RuntimeError(f"job {job_id} not ready (status={status})")
    out_id = getattr(job, "output_file", None) or getattr(job, "output_file_id", None)
    if not out_id:
        raise RuntimeError(f"job {job_id} has no output_file")
    dl = client.files.download(file_id=out_id)
    # The SDK's download return shape varies by version: a stream with
    # .read(), an object with .content/.text, or raw bytes/str.
    if hasattr(dl, "read"):
        data = dl.read()
    elif hasattr(dl, "content"):
        data = dl.content
    elif hasattr(dl, "text"):
        data = dl.text
    else:
        data = dl
    text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
    pages: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        resp = obj.get("response") or {}
        body = resp.get("body") if isinstance(resp, dict) else None
        body = body or resp or obj
        for pg in (body.get("pages") or []):
            pages.append(pg if isinstance(pg, dict)
                         else {"markdown": str(pg or "")})
    return pages


def cancel(api_key: str, job_id: str) -> str:
    client = _client(api_key)   # keep a ref (see poll)
    job = client.batch.jobs.cancel(job_id=job_id)
    return _norm_status(getattr(job, "status", "")) or "CANCELLATION_REQUESTED"


def _job_created_iso(job: Any) -> str:
    ts = getattr(job, "created_at", None)
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
                timespec="seconds")
        except Exception:
            return ""
    return str(ts or "")


def list_jobs(api_key: str) -> list[dict]:
    """All Aglaïa batch jobs on the account, newest first. Normalized dicts:
    ``{id, status, created_at, project, chunk, chunks_total, total, succeeded,
    failed}``."""
    client = _client(api_key)
    try:
        res = client.batch.jobs.list(
            metadata={META_APP: META_APP_VALUE}, page_size=100)
    except TypeError:
        res = client.batch.jobs.list(metadata={META_APP: META_APP_VALUE})
    data = getattr(res, "data", None)
    if data is None:
        data = res if isinstance(res, list) else []
    rows: list[dict] = []
    for j in data:
        md = getattr(j, "metadata", None) or {}
        rows.append({
            "id": getattr(j, "id", ""),
            "status": _norm_status(getattr(j, "status", "")),
            "created_at": _job_created_iso(j),
            "project": md.get(META_PROJECT, ""),
            "chunk": md.get(META_CHUNK, ""),
            "chunks_total": md.get(META_CHUNKS_TOTAL, ""),
            "total": getattr(j, "total_requests", None),
            "succeeded": getattr(j, "succeeded_requests", None),
            "failed": getattr(j, "failed_requests", None),
        })
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


# ── result shaping ───────────────────────────────────────────────────────
def page_to_result(page: dict, page_w: int, page_h: int,
                   languages: list[str]) -> OcrResult:
    """Build the same OcrResult shape the sync path produces, from one
    Mistral page object — markdown for md_export plus the rich page
    (``meta.mistral_page``) so deferred batch results persist via
    ``ocr_repo.finish`` exactly like a synchronous run, structure intact."""
    md = page.get("markdown", "") if isinstance(page, dict) else (page or "")
    base: OcrResult = {
        "engine": "mistral_cloud", "languages": list(languages),
        "page_w": int(page_w), "page_h": int(page_h),
    }
    line = {"text": md, "bbox": (0, 0, int(page_w), int(page_h)),
            "confidence": 1.0}
    base["lines"] = [line] if md else []
    meta = {"source": "mistral", "model": MODEL, "markdown": md, "batch": True}
    if isinstance(page, dict):
        meta["mistral_page"] = page
    base["meta"] = meta
    return base
