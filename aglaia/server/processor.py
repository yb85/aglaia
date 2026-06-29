# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Background job processing for the server (#52, slices 2-3).

A daemon thread drains pending jobs: ingest (bundle/PDF) → pipeline → export
(PDF; +OCR+MD when OCR is on) → artifacts in the job folder, then **discard the
transient `.agl`**. Mistral *batch* OCR submits and parks the job `ocr_pending`;
it is then polled — on `/check`/`/get` and by this ticker — with exponential
backoff (1′ … 512′, then give up).

Reuses the proven headless pipeline; the heavy calls (`run_pipeline`,
`run_exports`, `check_batch`, `cancel_batch`) are module functions so tests can
stub them.
"""

from __future__ import annotations

import shutil
import threading
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from aglaia.server import db as sdb

#: Exponential backoff (minutes) for polling a pending Mistral batch job.
BACKOFF_MINUTES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

SLUG = "job"   # transient project name inside each job's proj/ dir


# ── heavy operations (stubbable in tests) ───────────────────────────────

def run_pipeline(cfg) -> int:
    from aglaia.workers.headless import run as _run
    return _run(cfg)


def run_exports(project_file: Path, project_dir: Path, slug: str, exports, *,
                ocr_layer: bool, md_refine: Optional[str] = None) -> int:
    from aglaia.workers.headless import _run_exports
    return _run_exports(str(project_file), project_dir, slug, exports,
                        ocr_layer=ocr_layer, md_refine=md_refine)


def check_batch(project_file: Path) -> str:
    """Poll pending Mistral batch OCR for a project. → 'done'|'pending'|'error'."""
    from aglaia.workers.headless import _check_ocr
    try:
        rc = _check_ocr(str(project_file))
    except SystemExit:
        return "error"
    return "pending" if rc == 2 else "done"


def cancel_batch(project_file: Path) -> None:
    """Best-effort cancel of a project's pending Mistral batch jobs."""
    try:
        from aglaia.app_data.secrets import get_mistral_api_key
        from aglaia.storage.db import open_db
        from aglaia.storage.repo import MistralBatchRepo
        from aglaia.workers.ocr import mistral_batch
        api_key = get_mistral_api_key()
        if not api_key:
            return
        conn = open_db(str(project_file))
        try:
            for job in MistralBatchRepo(conn).pending():
                try:
                    mistral_batch.cancel(api_key, job["job_id"])
                except Exception:  # noqa: BLE001
                    pass
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        pass


# ── helpers ──────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _plus_minutes(minutes: int) -> str:
    return (_now() + timedelta(minutes=minutes)).isoformat(timespec="seconds")


# ── processor ──────────────────────────────────────────────────────────────

class JobProcessor:
    def __init__(self, db_file: Path, data_root: Path) -> None:
        self.db_file = Path(db_file)
        self.data_root = Path(data_root)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="aglaia-server-jobs")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop.wait(2.0):
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def tick(self) -> None:
        with sdb.session(self.db_file) as conn:
            pending = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
                (sdb.STATUS_PENDING,)).fetchone()
            due = [r["id"] for r in sdb.due_ocr_jobs(conn)]
        if pending:
            self.process_job(pending["id"])
        for job_id in due:
            self.check_job(job_id)

    # — processing —

    def process_job(self, job_id: str) -> None:
        with sdb.session(self.db_file) as conn:
            job = sdb.get_job(conn, job_id)
            if job is None or job["status"] != sdb.STATUS_PENDING:
                return
            sdb.update_job(conn, job_id, status=sdb.STATUS_PROCESSING)
            row = dict(job)

        jobdir = self.data_root / job_id
        projdir = jobdir / "proj"
        try:
            cfg, exports, is_batch = self._build_cfg(row, jobdir, projdir)
            run_pipeline(cfg)
            if is_batch:
                with sdb.session(self.db_file) as conn:
                    sdb.update_job(conn, job_id, status=sdb.STATUS_OCR_PENDING,
                                   attempt=1, next_check_at=_plus_minutes(BACKOFF_MINUTES[0]))
            else:
                self._finalize(job_id, jobdir, projdir, row, did_ocr=bool(row["ocr_spec"]))
        except Exception as exc:  # noqa: BLE001
            self._fail(job_id, exc)

    def check_job(self, job_id: str) -> None:
        """Poll a pending batch job; finalize, reschedule, or give up."""
        with sdb.session(self.db_file) as conn:
            job = sdb.get_job(conn, job_id)
            if job is None or job["status"] != sdb.STATUS_OCR_PENDING:
                return
            row = dict(job)

        jobdir = self.data_root / job_id
        projdir = jobdir / "proj"
        project_file = projdir / f"{SLUG}.agl"
        status = check_batch(project_file)
        if status == "done":
            try:
                run_exports(project_file, projdir, SLUG, self._exports_for(row),
                            ocr_layer=True)
                self._finalize(job_id, jobdir, projdir, row, did_ocr=True)
            except Exception as exc:  # noqa: BLE001
                self._fail(job_id, exc)
        elif status == "error":
            self._fail(job_id, RuntimeError("Mistral batch check failed (API key configured?)"))
        else:  # still pending → reschedule with exponential backoff
            attempt = int(row["attempt"] or 1)
            if attempt >= len(BACKOFF_MINUTES):
                self._fail(job_id, RuntimeError("OCR batch timed out (max backoff reached)"))
                return
            with sdb.session(self.db_file) as conn:
                sdb.update_job(conn, job_id, attempt=attempt + 1,
                               next_check_at=_plus_minutes(BACKOFF_MINUTES[attempt]))

    # — finalize / fail —

    def _finalize(self, job_id, jobdir: Path, projdir: Path, row, *, did_ocr: bool) -> None:
        pdf_src, md_src = projdir / f"{SLUG}.pdf", projdir / f"{SLUG}.md"
        pdf_path = md_path = None
        if pdf_src.is_file():
            shutil.copyfile(pdf_src, jobdir / "output.pdf")
            pdf_path = str(jobdir / "output.pdf")
        if did_ocr and md_src.is_file():
            shutil.copyfile(md_src, jobdir / "output.md")
            md_path = str(jobdir / "output.md")
        shutil.rmtree(projdir, ignore_errors=True)   # never persist the .agl

        with sdb.session(self.db_file) as conn:
            sdb.update_job(conn, job_id, status=sdb.STATUS_DONE,
                           pdf_path=pdf_path, md_path=md_path, error=None)
            job = sdb.get_job(conn, job_id)
            email = sdb.email_for_job(conn, job)
            base_url = sdb.get_config(conn, sdb.CONFIG_BASE_URL)
        if row["email_notif"] and email:
            self._notify(job, email, base_url)

    def _fail(self, job_id: str, exc: Exception) -> None:
        with sdb.session(self.db_file) as conn:
            sdb.update_job(conn, job_id, status=sdb.STATUS_FAILED,
                           error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    def _notify(self, job, email: str, base_url) -> None:
        try:
            from aglaia.server.mailer import send_completion
            send_completion(self.db_file, to=email, job=job, base_url=base_url)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    # — config build —

    def _exports_for(self, row):
        from aglaia.workers.cli import ExportTask
        exports = [ExportTask(kind="pdf", profile="auto")]
        if row["ocr_spec"]:
            exports.append(ExportTask(kind="md"))
        return exports

    def _build_cfg(self, row, jobdir: Path, projdir: Path):
        from aglaia.workers.cli import CliConfig, build_ocr_fields, classify_inputs
        projdir.mkdir(parents=True, exist_ok=True)

        if row["kind"] == "bundle":
            from aglaia.workers.bridge_bundle import read_bundle
            bundle = read_bundle(jobdir / "input.zip", extract_dir=jobdir / "extract")
            inputs = bundle.image_paths
            source = "images"
            dpi = bundle.dpi or row["dpi"]
        else:
            inputs = [jobdir / "input.pdf"]
            source = "pdfs"
            dpi = row["dpi"]

        ocr_fields = build_ocr_fields(row["ocr_spec"], "auto") if row["ocr_spec"] else {}
        is_batch = bool(ocr_fields.get("ocr_batch"))
        exports = self._exports_for(row)

        cfg = CliConfig(
            paths=list(inputs), source=source, inputs=list(inputs),
            project_name=SLUG, parent_dir=projdir, headless=True,
            input_dpi=float(dpi) if dpi else None, input_dpi_force=bool(dpi),
            exports=exports, **ocr_fields,
        )
        classify_inputs(cfg)   # re-affirm source from the resolved paths
        return cfg, exports, is_batch
