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
import multiprocessing
import queue as _q
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from aglaia.app_data import db as app_db
from aglaia.workers.Initializer import (
    create_processing_chain,
    initialize,
    load_pipeline_def,
)
from aglaia.workers.ImportHelpers import (
    _embedded_dpi,
    _persist_raw,
    catchup_active_scans,
    enqueue_image_files,
    enqueue_pdf_files,
)
from aglaia.workers.PDFprocessor import create_pdf_from_db
from aglaia.workers.md_export import write_markdown
from aglaia.workers.cli import (
    CliConfig,
    default_parent_dir,
    default_project_name,
    effective_workers,
    resolve_pipeline_path,
)
from aglaia.storage.db import open_db
from aglaia.storage.repo import (
    BranchRepo,
    OcrRepo,
    PipelineRepo,
    ProjectRepo,
    ScanRepo,
)


# Hard ceiling so a stuck chain can't keep the CLI hanging forever; the
# GUI's RSS-watchdog SIGKILL respawn logic doesn't help here because the
# only signal we have is per-scan branch_ready events.
DEFAULT_TIMEOUT_S = 60 * 60


def _warn_workers_ram(raw_workers) -> None:
    """Print a swap-risk warning when a MANUAL --workers count won't fit in RAM
    (auto is already RAM-capped). Respects the GUI 'don't warn again' flag."""
    try:
        from aglaia.worker_count import resolve_workers, ram_warning

        count, is_auto = resolve_workers(raw_workers)
        if is_auto:
            return
        msg = ram_warning(count)
        if not msg:
            return
        from aglaia.app_data import db as cfg

        with cfg.session() as conn:
            if bool(cfg.get(conn, cfg.KEY_WORKERS_RAM_WARN_DISMISSED, False)):
                return
        print(f"WARN: {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _build_initialize_argv(cfg: CliConfig, project_dir: Path) -> list[str]:
    """Synthesise sys.argv for `initialize()`. The Initializer parser
    expects a positional workspace dir + recognises a handful of
    capture-mode flags."""
    argv = ["aglaia", str(project_dir)]
    _w = effective_workers(cfg.workers)
    argv += ["--workers", str(_w)]
    _warn_workers_ram(_w)
    if cfg.input_dpi is not None:
        argv += ["--input-dpi", str(cfg.input_dpi)]
    if cfg.camera_id is not None:
        argv += ["--camera-id", str(cfg.camera_id)]
    return argv


def _run_ocr(
    db_path: str, *, engine_name: str, languages: list[str], params: dict | None = None
) -> int:
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
        canonical = {
            "apple": "apple_vision",
            "surya": "surya",
            "mistral": "mistral_cloud",
        }.get(name, name)
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
        # Per-engine: a branch needs OCR for THIS engine even if another engine
        # already ran it → re-running a different engine adds its layer instead
        # of being a no-op, keeping the existing engine's layer.
        rows = ocr.branches_needing_ocr(include_stale=True, engine=engine_canonical)
        if not rows:
            print("OCR: nothing to do.")
            return 0
        print(
            f"OCR: {len(rows)} branch(es), engine={engine_canonical}, langs={languages or '<auto>'}"
        )
        # Pay the model-load cost up front and time it SEPARATELY, so the
        # per-page numbers below are steady-state processing — not load +
        # first-page conflated. No-op (≈0 s) for Apple Vision; spins up the
        # local server for the VLMs.
        _tl0 = time.perf_counter()
        try:
            engine.warmup(languages)
        except Exception as e:  # noqa: BLE001 — surface, keep going
            print(f"OCR: warmup failed: {e}", file=sys.stderr, flush=True)
        load_s = time.perf_counter() - _tl0
        print(f"OCR: model load {load_s:.1f}s", flush=True)

        # Cloud WHOLE-DOCUMENT engines (Mistral) OCR the entire project in ONE
        # request — all pages assembled into a single PDF, one API call. The
        # per-page loop below would upload one PDF per page (24 calls, billed,
        # and not how the engine is meant to run); route them through the same
        # recognize_rows() the GUI's OcrWorker uses. (CLI/GUI still have two OCR
        # drivers — that divergence needs an audit; see issue note.)
        if callable(getattr(engine, "recognize_rows", None)):
            img_rows: list[dict] = []
            handles: list[tuple[int, int, str]] = []  # (run_id, scan_id, bp)
            for r in rows:
                irow = conn.execute(
                    "SELECT blob, dpi, type, format, width, height "
                    "FROM images WHERE id = ?", (int(r["image_id"]),)
                ).fetchone()
                if irow is None:
                    continue
                run_id = ocr.start(
                    scan_id=int(r["scan_id"]), node_id=int(r["chosen_node_id"]),
                    branch_path=r["branch_path"] or "",
                    engine=engine_canonical, languages=languages,
                )
                img_rows.append(dict(irow))
                handles.append((run_id, int(r["scan_id"]), r["branch_path"] or ""))
            if not img_rows:
                return 1
            print(f"OCR: 1 whole-document request → {len(img_rows)} page(s)…",
                  flush=True)
            _t0 = time.perf_counter()
            try:
                results = engine.recognize_rows(img_rows, languages)
            except Exception as e:  # noqa: BLE001
                for run_id, _, _ in handles:
                    ocr.fail(run_id, f"{type(e).__name__}: {e}")
                print(f"OCR: ERROR {e}", file=sys.stderr, flush=True)
                return 1
            req_s = time.perf_counter() - _t0
            done = 0
            for (run_id, _sid, _bp), result in zip(handles, results):
                try:
                    ocr.finish(run_id, result)
                    done += 1
                except Exception as e:  # noqa: BLE001
                    ocr.fail(run_id, f"{type(e).__name__}: {e}")
            print(
                f"OCR: {done} page(s) processed in {req_s:.1f}s "
                f"(mean {req_s / max(done, 1):.2f}s/page amortized, "
                f"one whole-document request) + {load_s:.1f}s model load",
                flush=True,
            )
            return 0 if done > 0 else 1

        done = 0
        page_ms: list[float] = []
        for r in rows:
            scan_id = int(r["scan_id"])
            branch_path = r["branch_path"] or ""
            node_id = int(r["chosen_node_id"])
            image_id = int(r["image_id"])
            blob_row = conn.execute(
                "SELECT blob, dpi FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if blob_row is None:
                continue
            # Pass the page's true DPI so the engine downsamples to the
            # configured ocr_dpi. Without it the engine sees src_dpi=0 and
            # falls back to a coarse longest-edge budget that barely shrinks
            # a 300-dpi page → it OCRs ~2× the pixels (≈2× slower) at the
            # wrong resolution, unlike the GUI which passes src_dpi.
            src_dpi = float(blob_row["dpi"] or 0)
            run_id = ocr.start(
                scan_id=scan_id,
                node_id=node_id,
                branch_path=branch_path,
                engine=engine_canonical,
                languages=languages,
            )
            try:
                pil = Image.open(io.BytesIO(blob_row["blob"])).convert("RGB")
                arr = np.array(pil, dtype=np.uint8)
                _t0 = time.perf_counter()
                result = engine.recognize(arr, languages, src_dpi=src_dpi)
                _ms = (time.perf_counter() - _t0) * 1000.0
                ocr.finish(run_id, result)
                done += 1
                page_ms.append(_ms)
                n_lines = len(result.get("lines", []))
                # Per-page op-log line (parseable by the bench harness for
                # OCR p5/p50/p95). Mirrors the pipeline op-log shape so the
                # log strip / log tab read identically across subsystems.
                from aglaia.workers.oplog import format_op

                scope = {"scan": scan_id}
                if branch_path:
                    scope["layout"] = branch_path
                print(
                    format_op(
                        f"ocr.{engine_canonical}",
                        elapsed_ms=_ms,
                        color=False,
                        lines=n_lines,
                        **scope,
                    ),
                    flush=True,
                )
            except Exception as e:
                ocr.fail(run_id, f"{type(e).__name__}: {e}")
                print(f"  scan {scan_id}: ERROR {e}", file=sys.stderr)
        # Steady-state page summary, kept separate from the model-load line
        # above so a benchmark can attribute time correctly.
        if page_ms:
            srt = sorted(page_ms)
            total_s = sum(page_ms) / 1000.0
            mean_s = total_s / len(page_ms)
            p50_s = srt[len(srt) // 2] / 1000.0
            print(
                f"OCR: {done} page(s) processed in {total_s:.1f}s "
                f"(mean {mean_s:.2f}s/page, median {p50_s:.2f}s/page) "
                f"+ {load_s:.1f}s model load",
                flush=True,
            )
        return 0 if done > 0 else 1
    finally:
        conn.close()


_OCR_CANON = {"apple": "apple_vision", "surya": "surya", "mistral": "mistral_cloud"}


def _submit_batch_ocr(db_path: str, *, engine_name: str, languages: list[str]) -> int:
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
        raise SystemExit(
            f"OCR: engine {canonical!r} does not support batch "
            f"— drop ':batch' (or use ':stream')."
        )
    api_key = get_mistral_api_key()
    if not api_key:
        raise SystemExit(
            "OCR: no Mistral API key (set MISTRAL_API_KEY or the OCR tab's Cloud card)."
        )
    conn = open_db(db_path)
    try:
        ocr = OcrRepo(conn)
        repo = MistralBatchRepo(conn)
        rows = ocr.branches_needing_ocr(include_stale=True, engine=canonical)
        if not rows:
            print("OCR: nothing to do.")
            return 0
        run_ids: list[int] = []
        img_rows: list[dict] = []
        for r in rows:
            rec = conn.execute(
                "SELECT blob, dpi, type, format, width, height "
                "FROM images WHERE id = ?",
                (int(r["image_id"]),),
            ).fetchone()
            if rec is None:
                continue
            rid = ocr.start(
                scan_id=int(r["scan_id"]),
                node_id=int(r["chosen_node_id"]),
                branch_path=r["branch_path"] or "",
                engine=canonical,
                languages=languages,
            )
            img_rows.append(
                {
                    "blob": rec["blob"],
                    "dpi": rec["dpi"],
                    "type": rec["type"],
                    "format": rec["format"],
                    "width": rec["width"],
                    "height": rec["height"],
                }
            )
            run_ids.append(rid)
        if not run_ids:
            print("OCR: nothing to do.")
            return 0
        print(f"OCR (batch): submitting {len(run_ids)} page(s) to Mistral…")
        jobs = mistral_batch.submit(api_key, img_rows, run_ids, db_path)
        for j in jobs:
            repo.add(
                j["job_id"],
                input_file_id=j.get("input_file_id"),
                chunk=j.get("chunk", 0),
                chunks_total=j.get("chunks_total", 1),
                page_count=j.get("page_count"),
                status="QUEUED",
                run_ids=j.get("run_ids"),
            )
        conn.commit()
        print(f"OCR (batch): submitted {len(jobs)} job(s):")
        for j in jobs:
            print(f"  {j['job_id']}")
        print(
            "\nRetrieve the results when ready with:\n"
            f"  aglaia --headless --check-ocr {db_path}"
        )
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
                        (rid,),
                    ).fetchone()
                    w, h = (int(row["w"] or 0), int(row["h"] or 0)) if row else (0, 0)
                    ocr.finish(rid, mistral_batch.page_to_result(page, w, h, []))
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
        print(f"Batch check: {imported} imported, {still} pending, {failed} failed.")
        return 0 if still == 0 else 2
    finally:
        conn.close()


def _ocr_layer_available(conn, engine: str) -> bool:
    """True if `engine` has a done OCR layer; else print the available layers
    (latest-generated first) so the user can pick a valid one."""
    layers = OcrRepo(conn).available_ocr_layers()
    if any(r["engine"] == engine for r in layers):
        return True
    avail = (
        ", ".join(f"{r['engine']} ({r['n_branches']} pp)" for r in layers) or "none"
    )
    print(
        f"  ! OCR layer '{engine}' not found. Available (latest first): {avail}",
        file=sys.stderr,
    )
    return False


def _run_exports(
    db_path: str,
    project_dir: Path,
    slug: str,
    exports: list,
    ocr_layer: bool,
    md_refine: str | None = None,
) -> int:
    """Execute every export task. Returns 0 on success, non-zero if any
    task failed."""
    fail = 0
    for task in exports:
        # `pdf:ocr=surya` / `md:ocr=apple` select which OCR layer to export
        # (default = the latest layer regardless of engine). Aliases honoured.
        raw_eng = task.params.get("ocr")
        ocr_engine = _OCR_CANON.get(raw_eng, raw_eng) if raw_eng else None
        tag = f" [OCR layer: {ocr_engine}]" if ocr_engine else ""
        if task.kind == "pdf":
            out = project_dir / f"{slug}.pdf"
            conn = open_db(db_path)
            try:
                if ocr_engine and not _ocr_layer_available(conn, ocr_engine):
                    fail += 1
                    continue
                print(f"Export PDF ({task.profile}) → {out}{tag}")
                ok = create_pdf_from_db(
                    conn,
                    out,
                    step_name=None,
                    compression=task.profile or "auto",
                    add_ocr_layer=ocr_layer,
                    engine=ocr_engine,
                )
            finally:
                conn.close()
            if not ok:
                print("  ! PDF export failed", file=sys.stderr)
                fail += 1
        elif task.kind == "md":
            out = project_dir / f"{slug}.md"
            # `--export md:refine=apple_fm` overrides the global --md-refine.
            refine = task.params.get("refine", md_refine)
            conn = open_db(db_path)
            try:
                if ocr_engine and not _ocr_layer_available(conn, ocr_engine):
                    fail += 1
                    continue
                print(
                    f"Export Markdown → {out}{tag}"
                    + (f" (LLM refine: {refine})" if refine else "")
                )
                ok = write_markdown(conn, out, refine=refine, engine=ocr_engine)
            finally:
                conn.close()
            if not ok:
                print("  ! Markdown export: no OCR data.", file=sys.stderr)
                fail += 1
    return fail


class _LogDrainer(threading.Thread):
    """Drains the chain's ``log_queue`` on its OWN thread, continuously,
    from the moment the chain starts — INCLUDING while the main thread is
    still feeding scans in.

    Why a thread and not an inline drain-after-feed loop: ``log_queue`` is an
    ``mp.Queue`` backed by a finite OS pipe. Workers flood it (an op-log line
    + image_event per step). If the main thread is busy in
    ``catchup_active_scans`` → ``chain.enqueue`` (which blocks on the bounded
    input_queue's backpressure) and isn't draining log_queue, the pipe fills,
    the workers' queue-feeder threads block on it, they stop consuming
    input_queue, and enqueue() never unblocks → a circular deadlock that bites
    at scale (confirmed via faulthandler: main in input_queue.put, workers in
    queues._feed). Draining concurrently keeps the pipe clear so backpressure
    always resolves. See memory project_worker_queue_deadlock.

    Completion is deduped by scan_id (a multi-branch scan fires branch_ready
    once per branch); replay runs after the emit, so the drainer keeps running
    (and printing) until ``stop()``."""

    def __init__(self, log_queue, *, total_expected: int = 0):
        super().__init__(daemon=True, name="log-drainer")
        self._log_queue = log_queue
        # Settable after construction: the drainer starts BEFORE feeding (to
        # keep the pipe clear), but the expected scan count is only known
        # once feeding/import finishes. Int assignment is atomic in CPython.
        self._total_expected = total_expected
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._done_scans: set = set()
        # Count distinct branches (scan_id, branch_path) and stamp the last
        # branch_ready — a scan emits one per layout (A/B…), so scan-id dedup
        # alone declared a multi-layout scan "done" after its FIRST branch and
        # tore the run down with siblings still in flight. Completion now waits
        # for every scan to start finishing AND for branch activity to go quiet.
        self._done_branches: set = set()
        self._last_branch_ts = 0.0
        self.imported = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._log_queue.get(timeout=0.3)
            except _q.Empty:
                continue
            except (OSError, EOFError):
                break
            self._handle(msg)
        # Final non-blocking sweep so late replay/sibling lines aren't lost.
        while True:
            try:
                self._handle(self._log_queue.get_nowait())
            except Exception:
                break

    def _handle(self, msg) -> None:
        if isinstance(msg, str):
            line = msg.strip()
            if line:
                print(line, flush=True)
            return
        if isinstance(msg, tuple) and msg:
            tag = msg[0]
            if tag == "scan_imported":
                self.imported += 1
            elif tag == "branch_ready":
                payload = msg[1] if len(msg) > 1 else {}
                sid = payload.get("scan_id")
                with self._lock:
                    self._done_scans.add(sid)
                    self._done_branches.add((sid, payload.get("branch_path") or ""))
                    self._last_branch_ts = time.monotonic()
                    n = len(self._done_scans)
                print(
                    f"branch ready: scan={sid} "
                    f"path={payload.get('branch_path') or '-'} "
                    f"({n}/{self._total_expected})",
                    flush=True,
                )
            elif tag in ("log_info", "log_warning", "error"):
                txt = msg[1] if len(msg) > 1 else ""
                if txt:
                    print(txt, flush=True)

    def set_expected(self, n: int) -> None:
        self._total_expected = n

    def done_count(self) -> int:
        with self._lock:
            return len(self._done_scans)

    def wait_for_completion(self, timeout_s: float, quiesce_s: float = 8.0) -> int:
        """Block until the run settles or ``timeout_s`` elapses.

        Two phases, because a scan's branch count is dynamic (PageDetector
        emits 1–N layouts) so the total isn't known up front:
          1. every expected scan has emitted at least one branch (all scans
             have started finishing), then
          2. branch activity goes quiet for ``quiesce_s`` (the remaining
             sibling branches — incl. parked/batched dewarps — and replay
             settle). Returning on the scan count alone tore multi-layout
             scans down after their first branch."""
        deadline = time.monotonic() + timeout_s
        while self.done_count() < self._total_expected and time.monotonic() < deadline:
            time.sleep(0.2)
        n = self.done_count()
        if n < self._total_expected:
            print(
                f"WARN: only {n}/{self._total_expected} scans finished within timeout.",
                file=sys.stderr,
            )
            return n
        # Phase 2: wait for branch activity to go silent (all siblings done).
        while time.monotonic() < deadline:
            with self._lock:
                quiet = time.monotonic() - self._last_branch_ts
            if quiet >= quiesce_s:
                break
            time.sleep(0.2)
        return n

    def stop(self) -> None:
        self._stop.set()


# ── chain-free OCR (the `aglaia ocr` command) ────────────────────────
#
# For already-clean docs (born-digital PDFs, flat scans) that need no geometric
# processing: ingest each page as the raw COLOR root node, register a single
# branch pointing straight at it, then reuse the same OCR + export primitives as
# `run`. No IntegratedProcessingChain — no worker pool, no dewarp/binarize.

# A synthetic, zero-step pipeline so the nodes/branches have a valid
# pipeline_version to reference (OCR never reads it).
_OCR_PASSTHROUGH_YAML = "name: ocr-passthrough\npipeline: []\nreplay: false\n"


def _ingest_one(
    db_path: str,
    pv_id: int,
    slug: str,
    arr,
    dpi: float,
    *,
    source: str,
    source_ref: str,
) -> None:
    """Persist a raw page + a single branch (chosen node = the raw image)."""
    scan_id, root_node_id, _fs, _idx, _img = _persist_raw(
        db_path,
        pv_id,
        slug,
        arr,
        dpi,
        source=source,
        source_ref=source_ref,
    )
    conn = open_db(db_path)  # autocommit (isolation_level=None)
    try:
        BranchRepo(conn).upsert(scan_id, "", root_node_id)
    finally:
        conn.close()


def _ingest_images_no_chain(
    db_path: str,
    pv_id: int,
    slug: str,
    image_paths,
    *,
    default_dpi: float,
    force_dpi: bool,
) -> int:
    import cv2

    paths = sorted(image_paths, key=lambda p: Path(p).name.lower())
    n = 0
    for p in paths:
        arr = cv2.imread(str(p))
        if arr is None:
            print(f"  ! skipped (unreadable): {p}", file=sys.stderr)
            continue
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        embedded = None if force_dpi else _embedded_dpi(Path(p))
        dpi = float(embedded) if embedded else float(default_dpi)
        _ingest_one(db_path, pv_id, slug, rgb, dpi, source="import", source_ref=str(p))
        n += 1
    return n


def _ingest_pdfs_no_chain(
    db_path: str, pv_id: int, slug: str, pdf_paths, *, render_dpi: float
) -> int:
    from aglaia.workers.pdf_extract import open_pdf, render_page

    pdfs = sorted(pdf_paths, key=lambda p: Path(p).name.lower())
    n = 0
    for pdf in pdfs:
        try:
            doc = open_pdf(pdf)
        except Exception as e:
            print(f"  ! skipped (cannot open): {pdf} ({e})", file=sys.stderr)
            continue
        try:
            for pno in range(len(doc)):
                arr = render_page(doc, pno, render_dpi)
                _ingest_one(
                    db_path,
                    pv_id,
                    slug,
                    arr,
                    render_dpi,
                    source="pdf",
                    source_ref=f"{pdf}#{pno + 1}",
                )
                n += 1
        finally:
            doc.close()
    return n


def run_ocr_only(cfg: CliConfig) -> int:
    """`aglaia ocr PATHS…` — OCR PDFs/images (or re-OCR a .agl) with NO geometric
    processing. Ingest → OCR → export; reuses run()'s OCR/export primitives."""
    if not cfg.has_inputs():
        print("No inputs. Pass PDFs, images, or one .agl project.", file=sys.stderr)
        return 2

    from aglaia.storage import (
        project_filename,
        resolve_existing_project_db,
        slug_from_project_file,
    )

    if cfg.source == "project":
        project_file = cfg.project_file
        project_dir = project_file.parent
        slug = slug_from_project_file(project_file)
        if cfg.check_ocr:
            return _check_ocr(str(project_file))
        # else: re-OCR the project's existing branches (no ingest).
    else:
        slug = default_project_name(cfg)
        parent = default_parent_dir(cfg)
        parent.mkdir(parents=True, exist_ok=True)
        project_dir = parent
        project_file = resolve_existing_project_db(
            project_dir, slug
        ) or project_dir / project_filename(slug)

        conn = open_db(project_file)
        try:
            ProjectRepo(conn).init(name=slug, slug=slug)
            pv_id = PipelineRepo(conn).upsert(
                _OCR_PASSTHROUGH_YAML, "ocr-passthrough", step_count=0
            )
        finally:
            conn.close()
        try:
            with app_db.session() as cdb:
                app_db.remember_project(cdb, project_file, slug)
        except Exception:
            pass

        if cfg.source == "images":
            print(f"Ingesting {len(cfg.inputs)} image(s) (no processing)…")
            n = _ingest_images_no_chain(
                str(project_file),
                pv_id,
                slug,
                cfg.inputs,
                default_dpi=float(cfg.input_dpi) if cfg.input_dpi else 120.0,
                force_dpi=cfg.input_dpi_force,
            )
        else:  # pdfs
            print(f"Ingesting {len(cfg.inputs)} PDF(s) (no processing)…")
            n = _ingest_pdfs_no_chain(
                str(project_file),
                pv_id,
                slug,
                cfg.inputs,
                render_dpi=float(cfg.input_dpi) if cfg.input_dpi else 200.0,
            )
        if n == 0:
            print("No pages ingested.", file=sys.stderr)
            return 2
        print(f"Ingested {n} page(s).")

    # OCR — the whole point of this command, so default to 'auto' if unset.
    engine = cfg.ocr_engine if cfg.do_ocr else "auto"
    if cfg.ocr_batch:
        return _submit_batch_ocr(
            str(project_file), engine_name=engine, languages=cfg.ocr_languages
        )
    _run_ocr(
        str(project_file),
        engine_name=engine,
        languages=cfg.ocr_languages,
        params=cfg.ocr_params,
    )

    if cfg.exports:
        rc = _run_exports(
            str(project_file),
            project_dir,
            slug,
            cfg.exports,
            ocr_layer=True,
            md_refine=cfg.md_refine,
        )
        if rc:
            return rc
    try:
        from aglaia.storage.db import compact_db

        compact_db(str(project_file))
    except Exception:
        pass
    return 0


def run(cfg: CliConfig) -> int:
    """Top-level headless entry. Returns process exit code."""
    if not cfg.has_inputs():
        print("No inputs. Pass a .agl file, PDFs, or images.", file=sys.stderr)
        return 2

    from aglaia.storage import (
        project_filename,
        resolve_existing_project_db,
        slug_from_project_file,
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
        project_file = resolve_existing_project_db(
            project_dir, slug
        ) or project_dir / project_filename(slug)

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

    # Start draining log_queue NOW, on its own thread — before any feeding —
    # so the pipe can't saturate while the main thread blocks in enqueue
    # backpressure (the deadlock; see _LogDrainer).
    drainer = _LogDrainer(log_queue)
    drainer.start()

    try:
        # Inputs
        expected_branches = 0
        n_caught = 0
        if cfg.source == "pdfs":
            print(f"Importing {len(cfg.inputs)} PDF(s)…")
            enqueue_pdf_files(
                db_path=str(project_file),
                pipeline_version_id=pipeline_version_id,
                slug=slug,
                chain=chain,
                pdf_paths=cfg.inputs,
                log_queue=log_queue,
            )
        elif cfg.source == "images":
            print(f"Importing {len(cfg.inputs)} image(s)…")
            enqueue_image_files(
                db_path=str(project_file),
                pipeline_version_id=pipeline_version_id,
                slug=slug,
                chain=chain,
                image_paths=cfg.inputs,
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
                chain=chain,
                force=cfg.force_proc,
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
            drainer.set_expected(expected_branches)
            drainer.wait_for_completion(DEFAULT_TIMEOUT_S)
        else:
            print("Nothing to process — pipeline objective already in DB.")
    finally:
        drainer.stop()
        try:
            drainer.join(timeout=5)
        except Exception:
            pass
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
            _submit_batch_ocr(
                str(project_file),
                engine_name=cfg.ocr_engine,
                languages=cfg.ocr_languages,
            )
            return 0
        _run_ocr(
            str(project_file),
            engine_name=cfg.ocr_engine,
            languages=cfg.ocr_languages,
            params=cfg.ocr_params,
        )

    # Exports
    if cfg.exports:
        ocr_layer = cfg.do_ocr  # if user asked for OCR, embed the layer in PDFs
        rc = _run_exports(
            str(project_file),
            project_dir,
            slug,
            cfg.exports,
            ocr_layer=ocr_layer,
            md_refine=cfg.md_refine,
        )
        if rc:
            return rc
    # Fold the WAL back into the .agl + drop -wal/-shm sidecars at rest.
    try:
        from aglaia.storage.db import compact_db

        compact_db(str(project_file))
    except Exception:
        pass
    return 0
