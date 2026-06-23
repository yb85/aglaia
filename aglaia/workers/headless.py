# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Headless batch driver — runs the same chain as the GUI without Qt.

Flow:
    1. Build args from CLI config.
    2. Open project DB, register pipeline_version.
    3. Spawn IntegratedProcessingChain.
    4. Enqueue inputs (PDFs or images).
    5. Drain log_queue until every imported scan has emitted branch_ready
       (or a timeout is exceeded).
    6. Optionally run OCR on each branch.
    7. Optionally run exports (PDF, Markdown).
    8. Stop chain, return exit code.

Mirrors the GUI's bring-up logic in `aglaia.py` minus the Qt
event loop and live-edit affordances.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import queue as _q
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from aglaia.app_data import db as app_db
from aglaia.workers.Initializer import create_processing_chain, initialize, load_pipeline_def
from aglaia.workers.ImportHelpers import (
    catchup_active_scans, enqueue_image_files, enqueue_pdf_files,
)
from aglaia.workers.PDFprocessor import create_pdf_from_db
from aglaia.workers.md_export import write_markdown
from aglaia.workers.cli import (
    CliConfig, default_parent_dir, default_project_name, effective_workers,
    resolve_pipeline_path,
)
from aglaia.storage.db import open_db
from aglaia.storage.repo import PipelineRepo, ProjectRepo, ScanRepo, OcrRepo


# Hard ceiling so a stuck chain can't keep the CLI hanging forever; the
# GUI's RSS-watchdog SIGKILL respawn logic doesn't help here because the
# only signal we have is per-scan branch_ready events.
DEFAULT_TIMEOUT_S = 60 * 60


def _build_initialize_argv(cfg: CliConfig, project_dir: Path) -> list[str]:
    """Synthesise sys.argv for `initialize()`. The Initializer parser
    expects a positional workspace dir + recognises a handful of
    capture-mode flags."""
    argv = ["aglaia", str(project_dir)]
    argv += ["--workers", str(effective_workers(cfg.workers))]
    if cfg.input_dpi is not None:
        argv += ["--input-dpi", str(cfg.input_dpi)]
    if cfg.camera_id is not None:
        argv += ["--camera-id", str(cfg.camera_id)]
    return argv


def _run_ocr(db_path: str, *, engine_name: str, languages: list[str],
             params: dict | None = None) -> int:
    """Run OCR sync on every branch that's missing or stale.

    `engine_name` follows the CLI semantics: 'auto' tries apple then
    surya; 'apple' / 'surya' force the named engine. `params` are the
    standard engine-spec key=value pairs (``--do-ocr name:k=v``), applied
    via ``engine.configure`` before recognition."""
    from aglaia.workers.ocr import get_engine

    def _pick_engine(name: str):
        if name == "auto":
            for n in ("apple_vision", "surya"):
                eng = get_engine(n)
                if eng.available:
                    return n, eng
            raise SystemExit("OCR: no engine available (apple_vision, surya).")
        canonical = {"apple": "apple_vision", "surya": "surya",
                     "mistral": "mistral_cloud"}.get(name, name)
        try:
            eng = get_engine(canonical)
        except KeyError as e:
            raise SystemExit(f"OCR: {e}") from None
        if not eng.available:
            raise SystemExit(f"OCR: engine {canonical!r} not available.")
        return canonical, eng

    engine_canonical, engine = _pick_engine(engine_name)
    if params:
        engine.configure(params)
    conn = open_db(db_path)
    try:
        ocr = OcrRepo(conn)
        rows = ocr.branches_needing_ocr(include_stale=True)
        if not rows:
            print("OCR: nothing to do.")
            return 0
        print(f"OCR: {len(rows)} branch(es), engine={engine_canonical}, langs={languages or '<auto>'}")
        done = 0
        for r in rows:
            scan_id = int(r["scan_id"])
            branch_path = r["branch_path"] or ""
            node_id = int(r["chosen_node_id"])
            image_id = int(r["image_id"])
            blob_row = conn.execute(
                "SELECT blob FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if blob_row is None:
                continue
            run_id = ocr.start(
                scan_id=scan_id, node_id=node_id, branch_path=branch_path,
                engine=engine_canonical, languages=languages,
            )
            try:
                pil = Image.open(io.BytesIO(blob_row["blob"])).convert("RGB")
                arr = np.array(pil, dtype=np.uint8)
                result = engine.recognize(arr, languages)
                ocr.finish(run_id, result)
                done += 1
                print(f"  scan {scan_id}{('.' + branch_path) if branch_path else ''}: "
                      f"{len(result.get('lines', []))} line(s)")
            except Exception as e:
                ocr.fail(run_id, f"{type(e).__name__}: {e}")
                print(f"  scan {scan_id}: ERROR {e}", file=sys.stderr)
        return 0 if done > 0 else 1
    finally:
        conn.close()


_OCR_CANON = {"apple": "apple_vision", "surya": "surya",
              "mistral": "mistral_cloud"}


def _submit_batch_ocr(db_path: str, *, engine_name: str,
                      languages: list[str]) -> int:
    """Submit a Mistral batch OCR job for every missing/stale branch and
    leave it pending — no Qt. Prints the `--check-ocr` command to retrieve."""
    from aglaia.workers.ocr import get_engine
    from aglaia.workers.ocr.engine import BatchableOCR
    from aglaia.workers.ocr import mistral_batch
    from aglaia.app_data.secrets import get_mistral_api_key
    from aglaia.storage.repo import MistralBatchRepo

    canonical = _OCR_CANON.get(engine_name, engine_name)
    try:
        eng = get_engine(canonical)
    except KeyError as e:
        raise SystemExit(f"OCR: {e}") from None
    if not isinstance(eng, BatchableOCR):
        raise SystemExit(f"OCR: engine {canonical!r} does not support batch "
                         f"— drop ':batch' (or use ':stream').")
    api_key = get_mistral_api_key()
    if not api_key:
        raise SystemExit("OCR: no Mistral API key (set MISTRAL_API_KEY or the "
                         "OCR tab's Cloud card).")
    conn = open_db(db_path)
    try:
        ocr = OcrRepo(conn)
        repo = MistralBatchRepo(conn)
        rows = ocr.branches_needing_ocr(include_stale=True)
        if not rows:
            print("OCR: nothing to do.")
            return 0
        run_ids: list[int] = []
        img_rows: list[dict] = []
        for r in rows:
            rec = conn.execute(
                "SELECT blob, dpi, type, format, width, height "
                "FROM images WHERE id = ?", (int(r["image_id"]),)).fetchone()
            if rec is None:
                continue
            rid = ocr.start(
                scan_id=int(r["scan_id"]), node_id=int(r["chosen_node_id"]),
                branch_path=r["branch_path"] or "", engine=canonical,
                languages=languages)
            img_rows.append({
                "blob": rec["blob"], "dpi": rec["dpi"], "type": rec["type"],
                "format": rec["format"], "width": rec["width"],
                "height": rec["height"]})
            run_ids.append(rid)
        if not run_ids:
            print("OCR: nothing to do.")
            return 0
        print(f"OCR (batch): submitting {len(run_ids)} page(s) to Mistral…")
        jobs = mistral_batch.submit(api_key, img_rows, run_ids, db_path)
        for j in jobs:
            repo.add(j["job_id"], input_file_id=j.get("input_file_id"),
                     chunk=j.get("chunk", 0),
                     chunks_total=j.get("chunks_total", 1),
                     page_count=j.get("page_count"), status="QUEUED",
                     run_ids=j.get("run_ids"))
        conn.commit()
        print(f"OCR (batch): submitted {len(jobs)} job(s):")
        for j in jobs:
            print(f"  {j['job_id']}")
        print("\nRetrieve the results when ready with:\n"
              f"  aglaia --headless --check-ocr {db_path}")
        return 0
    finally:
        conn.close()


def _check_ocr(db_path: str) -> int:
    """Poll this project's pending Mistral batch jobs; import SUCCESS results
    into their OCR runs (rich page JSON + markdown). No Qt."""
    from aglaia.workers.ocr import mistral_batch
    from aglaia.storage.repo import MistralBatchRepo
    from aglaia.app_data.secrets import get_mistral_api_key

    api_key = get_mistral_api_key()
    if not api_key:
        raise SystemExit("No Mistral API key.")
    conn = open_db(db_path)
    try:
        repo = MistralBatchRepo(conn)
        ocr = OcrRepo(conn)
        pend = repo.pending()
        if not pend:
            print("No pending Mistral batch jobs.")
            return 0
        imported = still = failed = 0
        for job in pend:
            jid = job["job_id"]
            status, err = mistral_batch.poll(api_key, jid)
            repo.set_status(jid, status, err)
            run_ids = MistralBatchRepo.run_ids_of(job)
            if status == "SUCCESS":
                pages = mistral_batch.fetch_pages(api_key, jid)
                for i, rid in enumerate(run_ids):
                    page = pages[i] if i < len(pages) else {}
                    row = conn.execute(
                        "SELECT i.width AS w, i.height AS h FROM ocr_runs r "
                        "JOIN nodes n ON n.id = r.node_id "
                        "JOIN images i ON i.id = n.image_id WHERE r.id = ?",
                        (rid,)).fetchone()
                    w, h = (int(row["w"] or 0), int(row["h"] or 0)) if row \
                        else (0, 0)
                    ocr.finish(rid, mistral_batch.page_to_result(
                        page, w, h, []))
                repo.mark_imported(jid)
                imported += 1
                print(f"  job {jid}: imported {len(run_ids)} page(s)")
            elif status in mistral_batch.FAILED_STATUSES:
                for rid in run_ids:
                    ocr.fail(rid, f"batch {status}")
                failed += 1
                print(f"  job {jid}: {status}")
            else:
                still += 1
                print(f"  job {jid}: {status} (not ready)")
        conn.commit()
        print(f"Batch check: {imported} imported, {still} pending, "
              f"{failed} failed.")
        return 0 if still == 0 else 2
    finally:
        conn.close()


def _run_exports(db_path: str, project_dir: Path, slug: str,
                 exports: list, ocr_layer: bool,
                 md_refine: str | None = None) -> int:
    """Execute every export task. Returns 0 on success, non-zero if any
    task failed."""
    fail = 0
    for task in exports:
        if task.kind == "pdf":
            out = project_dir / f"{slug}.pdf"
            print(f"Export PDF ({task.profile}) → {out}")
            with open_db(db_path) as _:  # noqa: F841 — just to surface DB errors early
                pass
            conn = open_db(db_path)
            try:
                ok = create_pdf_from_db(
                    conn, out, step_name=None,
                    compression=task.profile or "auto",
                    add_ocr_layer=ocr_layer,
                )
            finally:
                conn.close()
            if not ok:
                print(f"  ! PDF export failed", file=sys.stderr)
                fail += 1
        elif task.kind == "md":
            out = project_dir / f"{slug}.md"
            # `--export md:refine=apple_fm` overrides the global --md-refine.
            refine = task.params.get("refine", md_refine)
            print(f"Export Markdown → {out}"
                  + (f" (LLM refine: {refine})" if refine else ""))
            conn = open_db(db_path)
            try:
                ok = write_markdown(conn, out, refine=refine)
            finally:
                conn.close()
            if not ok:
                print("  ! Markdown export: no OCR data.", file=sys.stderr)
                fail += 1
    return fail


def _wait_for_chain(log_queue, *, total_expected: int,
                    timeout_s: float) -> int:
    """Drain log_queue until `total_expected` distinct scans have emitted
    branch_ready or `timeout_s` elapses. Returns the count of completed
    scans. Forwards log messages to stdout.

    A multi-branch scan fires branch_ready once per branch, so completion
    is deduped by scan_id. Replay runs *after* the branch_ready emit, so
    once the target is reached we keep draining until the queue stays
    quiet for a grace period before handing back to chain.stop()."""
    done_scans: set = set()
    deadline = time.monotonic() + timeout_s
    imported = 0

    def _handle(msg) -> None:
        nonlocal imported
        if isinstance(msg, str):
            line = msg.strip()
            if line:
                print(line, flush=True)
            return
        # Chain protocol: every event is a tuple ('tag', payload…) — see
        # docs/architecture.md "log_queue protocol".
        if isinstance(msg, tuple) and msg:
            tag = msg[0]
            if tag == "scan_imported":
                imported += 1
            elif tag == "branch_ready":
                payload = msg[1] if len(msg) > 1 else {}
                sid = payload.get("scan_id")
                done_scans.add(sid)
                print(f"branch ready: scan={sid} "
                      f"path={payload.get('branch_path') or '-'} "
                      f"({len(done_scans)}/{total_expected})", flush=True)
            elif tag in ("log_info", "log_warning", "error"):
                txt = msg[1] if len(msg) > 1 else ""
                if txt:
                    print(txt, flush=True)

    while len(done_scans) < total_expected and time.monotonic() < deadline:
        try:
            msg = log_queue.get(timeout=1.0)
        except _q.Empty:
            continue
        _handle(msg)
    if len(done_scans) < total_expected:
        print(f"WARN: only {len(done_scans)}/{total_expected} scans finished "
              f"within timeout.", file=sys.stderr)
        return len(done_scans)
    # Grace drain: sibling branches / replay of the last scan may still be
    # in flight. Exit after 5 s of queue silence.
    while time.monotonic() < deadline:
        try:
            msg = log_queue.get(timeout=5.0)
        except _q.Empty:
            break
        _handle(msg)
    return len(done_scans)


def run(cfg: CliConfig) -> int:
    """Top-level headless entry. Returns process exit code."""
    if not cfg.has_inputs():
        print("No inputs. Pass a .agl file, PDFs, or images.", file=sys.stderr)
        return 2

    from aglaia.storage import (
        project_filename, resolve_existing_project_db, slug_from_project_file,
    )
    # Project dir + slug
    if cfg.source == "project":
        project_file = cfg.project_file
        project_dir = project_file.parent
        slug = slug_from_project_file(project_file)
        # --check-ocr: just poll + import pending batch jobs, then exit. No
        # pipeline / import.
        if cfg.check_ocr:
            return _check_ocr(str(project_file))
    else:
        slug = default_project_name(cfg)
        parent = default_parent_dir(cfg)
        parent.mkdir(parents=True, exist_ok=True)
        project_dir = parent
        project_file = (resolve_existing_project_db(project_dir, slug)
                        or project_dir / project_filename(slug))

    # initialize() builds args, resolves config, pipeline path, etc.
    saved_argv = sys.argv[:]
    sys.argv = _build_initialize_argv(cfg, project_dir)
    try:
        args = initialize(mode="pdf" if cfg.source == "pdfs" else "capture")
    finally:
        sys.argv = saved_argv

    pipeline_path = resolve_pipeline_path(cfg.pipeline)
    if pipeline_path is None:
        from aglaia.assets import config_path
        pipeline_path = config_path("pipelines", "book_curved_x2.yaml").resolve()
    args.pipeline = pipeline_path
    args.project_slug = slug

    pipeline_def = load_pipeline_def(pipeline_path) or {}

    # DB bring-up
    conn = open_db(project_file)
    try:
        ProjectRepo(conn).init(name=slug, slug=slug)
        pipeline_version_id = PipelineRepo(conn).upsert(
            pipeline_path.read_text(encoding="utf-8"),
            pipeline_def.get("name"),
            step_count=len(pipeline_def.get("pipeline", [])),
        )
    finally:
        conn.close()
    args.db_path = str(project_file)

    # Remember in app-data recent list (best-effort).
    try:
        with app_db.session() as cdb:
            app_db.remember_project(cdb, project_file, slug)
    except Exception:
        pass

    # Chain
    log_queue = multiprocessing.Queue()
    chain = create_processing_chain(args, log_queue, db_path=str(project_file))
    chain.start()

    try:
        # Inputs
        expected_branches = 0
        n_caught = 0
        if cfg.source == "pdfs":
            print(f"Importing {len(cfg.inputs)} PDF(s)…")
            enqueue_pdf_files(
                db_path=str(project_file), pipeline_version_id=pipeline_version_id,
                slug=slug, chain=chain, pdf_paths=cfg.inputs, log_queue=log_queue,
            )
        elif cfg.source == "images":
            print(f"Importing {len(cfg.inputs)} image(s)…")
            enqueue_image_files(
                db_path=str(project_file), pipeline_version_id=pipeline_version_id,
                slug=slug, chain=chain, image_paths=cfg.inputs,
                default_dpi=float(cfg.input_dpi) if cfg.input_dpi else 120.0,
                force_dpi=cfg.input_dpi_force,
                log_queue=log_queue,
            )
        elif cfg.source == "project":
            # Re-open: surface scans whose pipeline objective is missing
            # from the DB (or every scan when --force-proc).
            n_caught = catchup_active_scans(
                db_path=str(project_file),
                pipeline_version_id=pipeline_version_id,
                chain=chain, force=cfg.force_proc,
            )
            if n_caught:
                kind = "force-reprocessed" if cfg.force_proc else "caught up"
                print(f"{n_caught} scan(s) {kind}.")
            else:
                print("All scans already have a pipeline objective in the DB.")
        # Total scans imported = post-ingest scan count. Branch count
        # (one chosen_node_id per branch) is what the chain emits.
        # Re-query to get the actual import count.
        conn = open_db(project_file)
        try:
            n_scans = len(ScanRepo(conn).list_active())
            n_branches = conn.execute(
                "SELECT COUNT(*) AS n FROM scans s WHERE s.deleted_at IS NULL"
            ).fetchone()["n"]
        finally:
            conn.close()
        if cfg.source == "project":
            # Re-open: only the scans `catchup` actually re-enqueued emit
            # branch_ready. An already-complete project re-enqueues nothing
            # (n_caught == 0), so waiting for the full scan count would block
            # for DEFAULT_TIMEOUT_S (1 h) on events that never come.
            expected_branches = n_caught
        else:
            expected_branches = max(n_scans, n_branches)
        if expected_branches:
            print(f"Waiting for {expected_branches} scan(s) to finish processing…")
            _wait_for_chain(log_queue, total_expected=expected_branches,
                            timeout_s=DEFAULT_TIMEOUT_S)
        else:
            print("Nothing to process — pipeline objective already in DB.")
    finally:
        try:
            chain.stop()
        except Exception:
            pass

    # OCR
    if cfg.do_ocr:
        if cfg.ocr_batch:
            # `--do-ocr mistral:batch` — submit + leave pending (retrieve
            # later with `--check-ocr`). Skip exports below: results aren't
            # in yet.
            _submit_batch_ocr(str(project_file), engine_name=cfg.ocr_engine,
                              languages=cfg.ocr_languages)
            return 0
        _run_ocr(str(project_file), engine_name=cfg.ocr_engine,
                 languages=cfg.ocr_languages, params=cfg.ocr_params)

    # Exports
    if cfg.exports:
        ocr_layer = cfg.do_ocr  # if user asked for OCR, embed the layer in PDFs
        rc = _run_exports(str(project_file), project_dir, slug,
                          cfg.exports, ocr_layer=ocr_layer,
                          md_refine=cfg.md_refine)
        if rc:
            return rc
    return 0
