# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
"""Off-chain worker for Mistral batch OCR jobs: poll the project's pending
jobs and import finished results, cancel a job, or list all account jobs for
the Jobs tab. Network I/O, so it runs off the GUI thread."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from aglaia.storage.db import open_db
from aglaia.storage.repo import MistralBatchRepo, OcrRepo
from aglaia.workers.ocr import mistral_batch


def _dims_for_run(conn, run_id: int) -> tuple[int, int]:
    row = conn.execute(
        "SELECT i.width AS w, i.height AS h FROM ocr_runs r "
        "JOIN nodes n ON n.id = r.node_id "
        "JOIN images i ON i.id = n.image_id WHERE r.id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return (0, 0)
    return (int(row["w"] or 0), int(row["h"] or 0))


class MistralBatchWorker(QThread):
    """One-shot worker. ``action``:
      - ``"check"``  poll this project's pending jobs; import SUCCESS results
                     into their OCR runs; emits ``check_done``.
      - ``"cancel"`` cancel ``job_id``; emits ``cancel_done``.
      - ``"list"``   list all Aglaïa jobs on the account; emits ``list_done``.
    """

    check_done = Signal(int, int, int, str)   # imported, pending, failed, msg
    cancel_done = Signal(str, str, str)       # job_id, status, error
    list_done = Signal(list, str)             # rows, error
    log_line = Signal(str, str)               # level, text

    def __init__(self, *, action: str, db_path: str = "",
                 job_id: str = "", parent=None) -> None:
        super().__init__(parent)
        self._action = action
        self._db_path = db_path
        self._job_id = job_id

    def run(self) -> None:
        from aglaia.app_data.secrets import get_mistral_api_key
        api_key = get_mistral_api_key()
        if not api_key:
            err = "No Mistral API key — set it in the OCR tab's Cloud card."
            if self._action == "list":
                self.list_done.emit([], err)
            elif self._action == "cancel":
                self.cancel_done.emit(self._job_id, "", err)
            else:
                self.check_done.emit(0, 0, 0, err)
            return
        try:
            if self._action == "list":
                self._do_list(api_key)
            elif self._action == "cancel":
                self._do_cancel(api_key)
            else:
                self._do_check(api_key)
        except Exception as e:  # network / SDK failure — report, don't crash
            err = f"{type(e).__name__}: {e}"
            if self._action == "list":
                self.list_done.emit([], err)
            elif self._action == "cancel":
                self.cancel_done.emit(self._job_id, "", err)
            else:
                self.check_done.emit(0, 0, 0, err)

    # ── list ──────────────────────────────────────────────────────────
    def _do_list(self, api_key: str) -> None:
        self.list_done.emit(mistral_batch.list_jobs(api_key), "")

    # ── cancel ────────────────────────────────────────────────────────
    def _do_cancel(self, api_key: str) -> None:
        status = mistral_batch.cancel(api_key, self._job_id)
        if self._db_path:
            conn = open_db(self._db_path)
            try:
                repo = MistralBatchRepo(conn)
                repo.set_status(self._job_id, status)
                # Fail the cancelled job's OCR runs so they don't sit
                # 'running' forever; the pages just stay un-OCR'd.
                row = repo.get(self._job_id)
                if row is not None:
                    ocr_repo = OcrRepo(conn)
                    for rid in MistralBatchRepo.run_ids_of(row):
                        ocr_repo.fail(rid, f"batch {status}")
                conn.commit()
            finally:
                conn.close()
        self.cancel_done.emit(self._job_id, status, "")

    # ── check + import ────────────────────────────────────────────────
    def _do_check(self, api_key: str) -> None:
        conn = open_db(self._db_path)
        imported = pending = failed = 0
        try:
            repo = MistralBatchRepo(conn)
            ocr_repo = OcrRepo(conn)
            jobs = repo.pending()
            for job in jobs:
                jid = job["job_id"]
                try:
                    status, err = mistral_batch.poll(api_key, jid)
                except Exception as e:
                    self.log_line.emit(
                        "error", f"[mistral_batch] poll {jid} failed: "
                        f"{type(e).__name__}: {e}")
                    pending += 1
                    continue
                repo.set_status(jid, status, err)
                run_ids = MistralBatchRepo.run_ids_of(job)
                if status == "SUCCESS":
                    try:
                        pages = mistral_batch.fetch_pages(api_key, jid)
                    except Exception as e:
                        self.log_line.emit(
                            "error", f"[mistral_batch] fetch {jid} failed: "
                            f"{type(e).__name__}: {e}")
                        pending += 1
                        continue
                    from aglaia.workers.ocr.md_postprocess import batch_markers
                    _mk = batch_markers(pages)  # doc-wide (refs/defs straddle)
                    for i, rid in enumerate(run_ids):
                        page = pages[i] if i < len(pages) else {}
                        w, h = _dims_for_run(conn, rid)
                        ocr_repo.finish(rid, mistral_batch.page_to_result(
                            page, w, h, [], markers=_mk))
                    repo.mark_imported(jid)
                    imported += 1
                    self.log_line.emit(
                        "info", f"[mistral_batch] imported job {jid} "
                        f"({len(run_ids)} run(s), {len(pages)} page(s))")
                elif status in mistral_batch.FAILED_STATUSES:
                    for rid in run_ids:
                        ocr_repo.fail(rid, f"batch {status}: {err or ''}")
                    failed += 1
                    self.log_line.emit(
                        "warn", f"[mistral_batch] job {jid} {status}")
                else:
                    pending += 1
                    self.log_line.emit(
                        "info", f"[mistral_batch] job {jid} {status}")
            conn.commit()
        finally:
            conn.close()
        if imported:
            msg = f"Imported {imported} job(s)."
            if pending:
                msg += f" {pending} still processing."
        elif pending:
            msg = f"{pending} job(s) still processing — check again later."
        elif failed:
            msg = f"{failed} job(s) failed."
        else:
            msg = "No pending batch jobs."
        self.check_done.emit(imported, pending, failed, msg)
