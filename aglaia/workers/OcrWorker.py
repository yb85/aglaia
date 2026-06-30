# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR worker thread.

One run iterates the set of branches that need OCR (per UI mode) and
calls the chosen engine for each. Persists each result via OcrRepo.

Runs in-process on a QThread because:
  * Apple Vision is CPU-bound, releases GIL during pyobjc calls.
  * Surya (future) will load heavy torch state — keep one process so
    weights stay resident across scans.
  * Pipeline workers and OCR worker are mutually exclusive (UI mutex),
    so RSS contention is not an issue.

Signals follow the same vocabulary as `PipelineProgressBar` so the
bottom status bar can be reused as-is:
  * `started(total)` → caller calls `progress.set_imported(total)`.
  * `snap_done(scan_id)` → `progress.mark_done(scan_id)`.
  * `finished(ok, error_text)` → final disable/re-enable in MainWindow.
"""

from __future__ import annotations

import io
import os
import sqlite3
import traceback
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from aglaia.workers.ocr import get_engine
from aglaia.storage.db import open_db
from aglaia.storage.repo import OcrRepo


class OcrWorker(QThread):

    started_total = Signal(int)                # total branches to OCR
    progress_scan = Signal(int)                # scan_id just finished
    log_line = Signal(str, str)                # level, text
    finished_ok = Signal(bool, str)            # ok, error_text
    batch_submitted = Signal(int, str)         # n_jobs, error_text ("" = ok)

    MODE_DEFAULT = "default"     # OCR on missing + stale
    MODE_FORCE = "force"         # OCR on every branch (re-OCR fresh too)
    MODE_MISSING = "missing"     # OCR only branches with no OCR at all

    def __init__(self, *, db_path: str, engine_name: str, languages: list[str],
                 mode: str, complement: str = "", batch: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._engine_name = engine_name
        self._languages = list(languages)
        self._mode = mode
        # When True (Cloud OCR + the card's batch toggle), submit a Mistral
        # batch job instead of running OCR synchronously: create the runs,
        # upload, and leave them pending for a later "Check result" import.
        self._batch = bool(batch)
        # Only meaningful for the apple_docs engine — the confidence-gated
        # complement engine ("surya" / "paddle_vl" / "none"). "" leaves the
        # engine on its own default.
        self._complement = complement or ""
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        # Plumb engine diagnostic prints through this worker's log_line
        # signal — surya.py / paddle_vl.py used to call ``print()`` which
        # went to terminal stdout (= /dev/null in .app bundles) and never
        # reached the GUI Log tab.
        from aglaia.workers.ocr.engine import set_engine_log_sink
        set_engine_log_sink(self.log_line.emit)
        try:
            conn = open_db(self._db_path)
            ocr_repo = OcrRepo(conn)
            engine = get_engine(self._engine_name)
            if not engine.available:
                self.finished_ok.emit(False, f"Engine {self._engine_name} not available.")
                return
            # Carry the GUI-picked complement onto the engine so its
            # (batched) recognize() picks the right gated backend.
            if self._complement and hasattr(engine, "complement"):
                engine.complement = self._complement

            if self._mode == self.MODE_FORCE:
                rows = ocr_repo.all_branches()
            elif self._mode == self.MODE_MISSING:
                rows = ocr_repo.branches_needing_ocr(
                    include_stale=False, engine=self._engine_name)
            else:
                rows = ocr_repo.branches_needing_ocr(
                    include_stale=True, engine=self._engine_name)

            total = len(rows)
            if total == 0:
                self.started_total.emit(0)
                self.log_line.emit("info", "OCR: nothing to do.")
                self.finished_ok.emit(True, "")
                return

            self.log_line.emit(
                "info",
                f"OCR: {total} branch(es), engine={self._engine_name}, "
                f"langs={self._languages or '<auto>'}, mode={self._mode}",
            )

            # Batch path: submit a Mistral batch job and leave the runs
            # pending (filled later by "Check result"). Only for whole-doc
            # cloud engines. Don't emit started_total — batch isn't a running
            # OCR pass, so it must NOT show the "Génération OCR…" spinner.
            if self._batch and getattr(engine, "whole_doc", False):
                self._submit_batch(rows, ocr_repo, conn)
                return

            # Non-batch: now show the progress bar.
            self.started_total.emit(total)

            # Batched OCR: feed N pages per call so Surya's llama-server
            # `--parallel` slots stay busy across pages instead of doing
            # one round-trip per scan. Default tracks the engine's
            # actual slot count (Surya stashes it on the class) so a
            # quant-aware budget that opens 8 slots also gets 8-page
            # batches, not 4-page ones.
            engine_slots = getattr(engine, "workers_chosen", 0) or 0
            try:
                batch_size = max(1, int(os.environ.get(
                    "AGLAIA_OCR_BATCH",
                    str(engine_slots or 4),
                )))
            except ValueError:
                batch_size = engine_slots or 4
            # Count of pages the engine flagged as truncated (whole-doc
            # provider cap) — reported at the end so the user knows to re-run.
            truncated_count = 0
            # Apple Vision / Apple Document run per-image (Vision + a
            # complement crop pass); batching doesn't help so collapse to 1.
            if self._engine_name in ("apple_vision", "apple_docs"):
                batch_size = 1
            # Whole-document engines (Cloud OCR) take a dedicated low-memory
            # path: build ONE upload PDF straight from the stored blobs
            # (decoded a page at a time) instead of holding every page as a
            # full-res RGB array — peak RSS drops from GBs to tens of MB.
            if getattr(engine, "whole_doc", False):
                self._run_whole_doc(engine, rows, ocr_repo, conn, total)
                return

            for chunk_start in range(0, len(rows), batch_size):
                if self._cancel:
                    self.log_line.emit("warn", "OCR cancelled.")
                    self.finished_ok.emit(False, "cancelled")
                    return
                chunk = rows[chunk_start: chunk_start + batch_size]

                pending: list[tuple] = []  # (scan_id, branch_path, run_id, arr)
                for r in chunk:
                    scan_id = int(r["scan_id"])
                    branch_path = r["branch_path"]
                    node_id = int(r["chosen_node_id"])
                    image_id = int(r["image_id"])

                    blob_row = conn.execute(
                        "SELECT blob, dpi FROM images WHERE id = ?",
                        (image_id,),
                    ).fetchone()
                    if blob_row is None:
                        self.log_line.emit(
                            "warn",
                            f"scan {scan_id}: image {image_id} missing, skip",
                        )
                        self.progress_scan.emit(scan_id)
                        continue

                    run_id = ocr_repo.start(
                        scan_id=scan_id, node_id=node_id,
                        branch_path=branch_path,
                        engine=self._engine_name, languages=self._languages,
                    )
                    try:
                        pil = Image.open(io.BytesIO(blob_row["blob"]))
                        if pil.mode != "RGB":
                            pil = pil.convert("RGB")
                        arr = np.array(pil, dtype=np.uint8)
                        src_dpi = float(blob_row["dpi"] or 0)
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        self.log_line.emit("error",
                                            f"scan {scan_id}: decode failed: {err}")
                        ocr_repo.fail(run_id, err + "\n" + traceback.format_exc())
                        self.progress_scan.emit(scan_id)
                        continue
                    pending.append((scan_id, branch_path, run_id, arr, src_dpi))

                if not pending:
                    continue

                # Heartbeat before the (possibly multi-minute) batch call so
                # the UI shows live progress instead of looking frozen while a
                # dense VLM page grinds — progress_scan only ticks *after* the
                # whole batch returns.
                lo = chunk_start + 1
                hi = min(chunk_start + len(pending), total)
                span = f"page {lo}" if lo == hi else f"pages {lo}–{hi}"
                hint = (" — loading the model on first run, please wait…"
                        if chunk_start == 0 else
                        " — VLM OCR; dense pages can take a few minutes")
                self.log_line.emit(
                    "info", f"OCR: {span} of {total}{hint}")

                images = [t[3] for t in pending]
                dpis = [t[4] for t in pending]
                try:
                    results = engine.recognize_batch(
                        images, self._languages, src_dpis=dpis,
                    )
                except Exception as e:
                    # Batch-level failure (e.g. llama-server died, HTTP
                    # timeout reading the OpenAI-compat completion).
                    # Mark every pending run as failed and surface the
                    # error. Then ABORT — continuing the chunk loop just
                    # rolls another 60 s timeout per page until the
                    # user's whole project is "OCR'd" with empties.
                    err = f"{type(e).__name__}: {e}"
                    is_timeout = isinstance(
                        e, (TimeoutError,)
                    ) or "timeout" in str(e).lower() or "timed out" in str(e).lower()
                    if is_timeout:
                        self.log_line.emit(
                            "error",
                            f"OCR cancelled — batch timed out: {err}. "
                            "Check the engine backend is reachable.",
                        )
                    else:
                        self.log_line.emit(
                            "error", f"OCR cancelled — batch failed: {err}"
                        )
                    tb = traceback.format_exc()
                    for scan_id, branch_path, run_id, *_ in pending:
                        ocr_repo.fail(run_id, err + "\n" + tb)
                        self.progress_scan.emit(scan_id)
                    # Bail out of the chunk loop entirely; downstream
                    # scans would all hit the same broken backend.
                    self.finished_ok.emit(False, err)
                    return

                for (scan_id, branch_path, run_id, _arr, _dpi), result in zip(
                        pending, results):
                    # Whole-doc engines (Cloud OCR) may flag a page as
                    # `meta.truncated` when the upload hit the provider's
                    # page/size cap. Don't finish() it — fail() so the
                    # branch stays "needing OCR" and a re-run continues it.
                    rmeta = result.get("meta") or {}
                    if rmeta.get("truncated"):
                        truncated_count += 1
                        ocr_repo.fail(
                            run_id,
                            "truncated: " + str(rmeta.get("reason", "over "
                            "provider limit")) + " — run OCR again to continue")
                        self.progress_scan.emit(scan_id)
                        continue
                    try:
                        ocr_repo.finish(run_id, result)
                        n_lines = len(result.get("lines", []))
                        suffix = f".{branch_path}" if branch_path else ""
                        self.log_line.emit(
                            "info",
                            f"scan {scan_id}{suffix}: {n_lines} line(s)",
                        )
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        self.log_line.emit("error",
                                            f"scan {scan_id}: persist failed: {err}")
                        ocr_repo.fail(run_id, err + "\n" + traceback.format_exc())
                    self.progress_scan.emit(scan_id)

            if truncated_count:
                msg = (
                    f"Cloud OCR truncated: {total - truncated_count} of "
                    f"{total} page(s) OCR'd; {truncated_count} exceeded the "
                    f"provider's per-upload limit and were left pending. "
                    f"Run OCR again to continue the remaining pages."
                )
                self.log_line.emit("warn", msg)
                # Carry the note up on a *successful* run so the GUI surfaces
                # it (error_text doubles as an info channel when ok=True).
                self.finished_ok.emit(True, msg)
            else:
                self.finished_ok.emit(True, "")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.log_line.emit("error", f"OCR worker crashed: {err}")
            self.finished_ok.emit(False, err)
        finally:
            # Clear the sink so a stale signal ref doesn't outlive the
            # worker (Qt would still deliver, but the slot is gone).
            try:
                set_engine_log_sink(None)
            except Exception:
                pass
            if conn is not None:
                conn.close()

    def _submit_batch(self, rows, ocr_repo, conn) -> None:
        """Create OCR runs for the selected branches, assemble the upload,
        and submit a Mistral batch job (chunked if over the page/size cap).
        Runs are left PENDING — results are pulled later by 'Check result'.
        Emits ``batch_submitted(n_jobs, error)``."""
        from aglaia.app_data.secrets import get_mistral_api_key
        from aglaia.storage.repo import MistralBatchRepo
        from aglaia.workers.ocr import mistral_batch

        api_key = get_mistral_api_key()
        if not api_key:
            self.batch_submitted.emit(
                0, "No Mistral API key — set it in the OCR tab's Cloud card.")
            return

        run_ids: list[int] = []
        img_rows: list[dict] = []
        for r in rows:
            if self._cancel:
                self.batch_submitted.emit(0, "cancelled")
                return
            image_id = int(r["image_id"])
            rec = conn.execute(
                "SELECT blob, dpi, type, format, width, height "
                "FROM images WHERE id = ?", (image_id,)).fetchone()
            if rec is None:
                continue
            run_id = ocr_repo.start(
                scan_id=int(r["scan_id"]), node_id=int(r["chosen_node_id"]),
                branch_path=r["branch_path"], engine=self._engine_name,
                languages=self._languages)
            img_rows.append({
                "blob": rec["blob"], "dpi": rec["dpi"], "type": rec["type"],
                "format": rec["format"], "width": rec["width"],
                "height": rec["height"]})
            run_ids.append(run_id)

        if not run_ids:
            self.batch_submitted.emit(0, "No pages to submit.")
            return

        self.log_line.emit(
            "info", f"Mistral batch: assembling + uploading {len(run_ids)} "
            f"page(s)…")
        try:
            jobs = mistral_batch.submit(api_key, img_rows, run_ids,
                                        self._db_path)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.log_line.emit("error", f"Batch submit failed: {err}")
            for run_id in run_ids:
                ocr_repo.fail(run_id, "batch submit failed: " + err)
            self.batch_submitted.emit(0, err)
            return

        repo = MistralBatchRepo(conn)
        for j in jobs:
            repo.add(j["job_id"], input_file_id=j.get("input_file_id"),
                     chunk=j.get("chunk", 0),
                     chunks_total=j.get("chunks_total", 1),
                     page_count=j.get("page_count"), status="QUEUED",
                     run_ids=j.get("run_ids"))
        conn.commit()
        self.log_line.emit(
            "info", f"Mistral batch: submitted {len(jobs)} job(s). Track them "
            f"on the OCR card or the Jobs tab; pull results with 'Check "
            f"result'.")
        self.batch_submitted.emit(len(jobs), "")

    def _run_whole_doc(self, engine, rows, ocr_repo, conn, total: int) -> None:
        """Low-memory path for whole-document engines (Cloud OCR).

        Builds ONE upload PDF straight from the stored image blobs — the
        engine (`recognize_rows`) decodes a single page at a time during
        assembly, so peak RSS stays at tens of MB instead of the GBs it
        took to hold every page as a full-res RGB array. Emits
        `finished_ok` itself; the caller returns right after."""
        pending: list[tuple] = []     # (scan_id, branch_path, run_id)
        img_rows: list[dict] = []     # lightweight rows for pdf_export
        for r in rows:
            if self._cancel:
                self.log_line.emit("warn", "OCR cancelled.")
                self.finished_ok.emit(False, "cancelled")
                return
            scan_id = int(r["scan_id"])
            branch_path = r["branch_path"]
            node_id = int(r["chosen_node_id"])
            image_id = int(r["image_id"])
            rec = conn.execute(
                "SELECT blob, dpi, type, format, width, height "
                "FROM images WHERE id = ?", (image_id,)).fetchone()
            if rec is None:
                self.log_line.emit(
                    "warn", f"scan {scan_id}: image {image_id} missing, skip")
                self.progress_scan.emit(scan_id)
                continue
            run_id = ocr_repo.start(
                scan_id=scan_id, node_id=node_id, branch_path=branch_path,
                engine=self._engine_name, languages=self._languages)
            img_rows.append({
                "blob": rec["blob"], "dpi": rec["dpi"], "type": rec["type"],
                "format": rec["format"], "width": rec["width"],
                "height": rec["height"],
            })
            pending.append((scan_id, branch_path, run_id))

        if not pending:
            self.finished_ok.emit(True, "")
            return

        self.log_line.emit(
            "info",
            f"OCR: {len(pending)} page(s) → assembling one PDF and uploading "
            f"to the cloud service (no local model to load)…")
        try:
            results = engine.recognize_rows(img_rows, self._languages)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            self.log_line.emit("error", f"Cloud OCR failed: {err}")
            for scan_id, _bp, run_id in pending:
                ocr_repo.fail(run_id, err + "\n" + tb)
                self.progress_scan.emit(scan_id)
            self.finished_ok.emit(False, err)
            return

        truncated = 0
        for (scan_id, branch_path, run_id), result in zip(pending, results):
            rmeta = result.get("meta") or {}
            if rmeta.get("truncated"):
                truncated += 1
                ocr_repo.fail(
                    run_id, "truncated: " + str(rmeta.get("reason", "over "
                    "provider limit")) + " — run OCR again to continue")
                self.progress_scan.emit(scan_id)
                continue
            try:
                ocr_repo.finish(run_id, result)
                n_lines = len(result.get("lines", []))
                suffix = f".{branch_path}" if branch_path else ""
                self.log_line.emit(
                    "info", f"scan {scan_id}{suffix}: {n_lines} line(s)")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                self.log_line.emit(
                    "error", f"scan {scan_id}: persist failed: {err}")
                ocr_repo.fail(run_id, err + "\n" + traceback.format_exc())
            self.progress_scan.emit(scan_id)

        if truncated:
            msg = (
                f"Cloud OCR truncated: {len(pending) - truncated} of "
                f"{len(pending)} page(s) OCR'd; {truncated} exceeded the "
                f"provider's per-upload limit and were left pending. Run OCR "
                f"again to continue the remaining pages.")
            self.log_line.emit("warn", msg)
            self.finished_ok.emit(True, msg)
        else:
            self.finished_ok.emit(True, "")
