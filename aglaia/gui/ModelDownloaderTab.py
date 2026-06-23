# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Model Downloader tab.

Three cards — EAST, Surya (recommended) and DBNet (other) — each lets
the user fetch the ML weights the corresponding PageDetector backend
needs into the configured `models_dir()`.

Per-card UX:

* Idle      → editable URL field + filesize hint + [Download] button.
* Active    → progress bar (downloaded / total, speed, ETA) + [Pause] [Stop].
* Paused    → progress bar frozen + [Resume] [Stop].
* Installed → green tick + [Re-download] (lets the user fetch over a
              corrupt or outdated file).

Downloads land in `<models_dir>/<filename>` via streamed `urllib.request`
with HTTP Range support — a paused download keeps a `.partialdl` file and
resumes from byte N. Stop deletes the partial.

Surya is special-cased: its weights are pulled by `huggingface_hub`
(via `snapshot_download`) into a `surya/` sub-directory rather than as a
single file — the card still shows a progress bar but it's
indeterminate (HF does its own per-file progress in stderr).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from aglaia.app_data import models_dir
from aglaia.gui.colors import (
    COLOR_BG_OVERLAY_HOVER,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_ERROR,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_FONT_SECTION_LABEL,
    COLOR_OUTLINE,
    COLOR_OUTLINE_GHOST,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_PRIMARY_BG_STRONG,
    COLOR_PRIMARY_BORDER,
    COLOR_SUCCESS,
    COLOR_SUCCESS_BG_SOFT,
    COLOR_SUCCESS_BORDER,
    COLOR_TERTIARY,
)
from aglaia.gui.widgets import Card


# ── model registry ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    key: str            # internal id
    title: str          # short card title
    filename: str       # on-disk filename inside models_dir (or subdir)
    url: str            # download URL (file kind) or HF repo id (hf-snapshot)
    approx_size_mb: int
    # Discriminates the worker that drives the download:
    #   "file"        — single HTTPS file pulled via `DownloadWorker`.
    #   "hf-snapshot" — recursive HuggingFace repo snapshot via the
    #                   `SuryaWorker` engine (despite the name, it works
    #                   for any HF model id, not just Surya).
    # Transport is HTTPS in both cases; plain http:// URLs are
    # rejected at start time.
    kind: str
    section: str        # "recommended" or "other"
    purpose: str        # e.g. "Layout detection" / "OCR"
    project: str        # human-readable upstream project name
    source: str         # short host badge: "github.com" / "huggingface.co"
    # SHA-1 of the canonical file. `kind="file"` only — directory
    # snapshots skip the per-file hash check.
    sha1: str = ""
    # For hf-snapshot kinds: list of ``{"path": "...", "size": N}``
    # entries the download MUST land on disk before the card is
    # considered "Installed". Without this, a partial snapshot with
    # only README + config would still be flagged as installed because
    # the directory is non-empty. Tuple of frozen dicts because
    # ``frozen=True`` dataclasses can't hold mutable defaults.
    required_files: tuple = ()


def _load_model_specs() -> list[ModelSpec]:
    """Load the model registry from `aglaia/app_data/model-list.json`.

    Lookup order — first hit wins:
      1. `<APP_DATA>/model-list.json` (per-user override; future remote
         refresh writes here).
      2. Bundled `aglaia/app_data/model-list.json` (ships with the app).

    Top-level keys (EAST / SURYA / DBNET …) are friendly labels — the
    actual `key` field inside each entry is what code branches on.
    Unknown JSON fields are tolerated so newer files stay
    backwards-compatible with older builds.
    """
    import json
    from aglaia.app_data import app_data_dir
    candidates = [
        app_data_dir() / "model-list.json",
        Path(__file__).resolve().parents[1] / "app_data" / "model-list.json",
    ]
    raw: dict | None = None
    for p in candidates:
        try:
            if p.is_file():
                raw = json.loads(p.read_text())
                break
        except Exception:
            continue
    if not raw:
        return []
    allowed = {f.name for f in fields(ModelSpec)}
    out: list[ModelSpec] = []
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        clean = {k: v for k, v in entry.items() if k in allowed}
        # ``required_files`` is JSON-shaped as a list[dict]; ModelSpec is
        # ``frozen=True`` so the value has to be hashable → tuple of
        # tuples (path, size). Older entries without the field default to
        # the empty tuple (existence-only fallback).
        rf = clean.get("required_files") or ()
        if rf:
            clean["required_files"] = tuple(
                (str(e.get("path", "")), int(e.get("size", 0)))
                for e in rf if isinstance(e, dict) and e.get("path")
            )
        try:
            out.append(ModelSpec(**clean))
        except TypeError:
            continue  # missing required field — skip silently
    return out


MODEL_SPECS: list[ModelSpec] = _load_model_specs()


# ── download worker ──────────────────────────────────────────────────

class _Stopped(Exception):
    """Raised inside the worker thread to bail cleanly on stop()."""


class _Paused(Exception):
    """Raised inside the worker thread to bail cleanly on pause()."""


class DownloadWorker(QObject):
    """Resumable HTTP downloader. One per active card.

    Lives on its own QThread; runs `run()` once. State transitions
    (pause/stop) flip booleans the streaming loop polls between chunks.
    """

    progress = Signal(int, int, float)  # downloaded, total, speed_bps
    finished = Signal(bool, str)        # ok, message
    paused = Signal()

    CHUNK = 64 * 1024  # 64 KiB — balances throughput against poll latency

    def __init__(self, url: str, dest: Path):
        super().__init__()
        self._url = url
        self._dest = dest
        self._stop = False
        self._pause = False

    def stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._pause = True

    def run(self) -> None:
        part = self._dest.with_suffix(self._dest.suffix + ".partialdl")
        try:
            already = part.stat().st_size if part.exists() else 0
        except OSError:
            already = 0

        req = urllib.request.Request(self._url)
        # Some hosts (Hugging Face's CDN, GitHub raw) 403 Python-urllib's
        # default UA. Always present as a real browser-ish client.
        req.add_header("User-Agent", "Aglaia/1.0 (+https://aglaia.bibli.cc)")
        if already > 0:
            req.add_header("Range", f"bytes={already}-")
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            # 416 = the server has nothing past `already` → file already
            # complete; promote .partialdl atomically and call it a win.
            if e.code == 416 and already > 0:
                part.replace(self._dest)
                self.finished.emit(True, self.tr("Already complete ({n} bytes).").format(n=already))
                return
            self.finished.emit(False, self.tr("HTTP {code}: {reason}").format(code=e.code, reason=e.reason))
            return
        except Exception as e:
            self.finished.emit(False, f"{type(e).__name__}: {e}")
            return

        total = already
        try:
            content_len = int(resp.headers.get("Content-Length", "0"))
            if content_len > 0:
                total = already + content_len
        except (TypeError, ValueError):
            pass

        mode = "ab" if already > 0 else "wb"
        downloaded = already
        last_emit = time.monotonic()
        emit_window_bytes = downloaded
        window_start = last_emit

        try:
            with open(part, mode) as f:
                while True:
                    if self._stop:
                        raise _Stopped()
                    if self._pause:
                        raise _Paused()
                    try:
                        chunk = resp.read(self.CHUNK)
                    except Exception as e:
                        self.finished.emit(False, self.tr("Read error: {err}").format(err=e))
                        return
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_emit >= 0.25:
                        elapsed = max(1e-6, now - window_start)
                        speed = (downloaded - emit_window_bytes) / elapsed
                        self.progress.emit(downloaded, total, speed)
                        last_emit = now
                        if now - window_start >= 1.0:
                            window_start = now
                            emit_window_bytes = downloaded
        except _Stopped:
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            self.finished.emit(False, self.tr("Cancelled."))
            return
        except _Paused:
            self.paused.emit()
            return
        finally:
            try:
                resp.close()
            except Exception:
                pass

        # Done — emit final progress so the bar lands at 100%, then
        # promote .partialdl to the real name atomically.
        if total > 0:
            self.progress.emit(downloaded, total, 0.0)
        try:
            part.replace(self._dest)
        except OSError as e:
            self.finished.emit(False, self.tr("Rename failed: {err}").format(err=e))
            return
        self.finished.emit(True, self.tr("Download complete."))


class SuryaWorker(QObject):
    """HuggingFace repo snapshot via plain HTTPS — no `huggingface_hub`
    dependency.

    Lists files via `https://huggingface.co/api/models/<repo>` and pulls
    each through the same resumable urllib path as `DownloadWorker`.
    Resume per file via HTTP `Range`. Pause leaves `.partialdl` files in
    place; resume re-issues from where they stopped. Stop deletes the
    in-flight `.partialdl` only.

    Why bypass huggingface_hub:
      * removes a runtime dependency (~3 MB + transitive in the bundle),
      * sidesteps `hf_xet` permission errors on macOS,
      * keeps progress reporting in our own code — no opaque blocking
        snapshot_download call.
    """

    progress = Signal(int, int, float)  # downloaded, total, speed_bps
    finished = Signal(bool, str)
    paused = Signal()

    USER_AGENT = "Aglaia/1.0 (+https://aglaia.bibli.cc)"
    CHUNK = 64 * 1024

    def __init__(self, repo_id: str, dest_dir: Path):
        super().__init__()
        self._repo = repo_id
        self._dest = dest_dir
        self._stop = False
        self._pause = False

    def stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._pause = True

    # ── HF API ──────────────────────────────────────────────────
    def _list_files(self) -> list[tuple[str, int]]:
        """Return [(rfilename, size_bytes)] via the HF tree endpoint with
        `recursive=true` — this is the only public API that ships sizes
        for non-LFS files. Falls back to size 0 (HEAD'd later) on any
        per-entry parse glitch."""
        import json
        url = (f"https://huggingface.co/api/models/{self._repo}"
               f"/tree/main?recursive=true")
        req = urllib.request.Request(url)
        req.add_header("User-Agent", self.USER_AGENT)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        out: list[tuple[str, int]] = []
        for s in data or []:
            if s.get("type") != "file":
                continue
            path = s.get("path")
            if not path:
                continue
            # LFS files prefer `lfs.size` (true blob bytes); plain text
            # entries use the top-level `size`. Either gives the byte
            # count the resolver-CDN will actually serve.
            sz = 0
            lfs = s.get("lfs") or {}
            if isinstance(lfs, dict) and lfs.get("size"):
                sz = int(lfs["size"])
            elif s.get("size"):
                sz = int(s["size"])
            out.append((path, sz))
        return out

    def _head_size(self, rfilename: str) -> int:
        url = f"https://huggingface.co/{self._repo}/resolve/main/{rfilename}"
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", self.USER_AGENT)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                cl = r.headers.get("Content-Length")
                return int(cl) if cl else 0
        except Exception:
            return 0

    # ── per-file resumable download ─────────────────────────────
    def _download_file(self, rfilename: str, expected: int,
                       on_chunk) -> tuple[bool, str]:
        """Download `rfilename` into `self._dest / rfilename` with HTTP
        Range resume. `on_chunk(n_bytes)` lets the caller update the
        cumulative byte counter for progress. Returns (ok, msg).
        Bails on stop/pause flags between chunks."""
        target = self._dest / rfilename
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".partialdl")

        # Already done? Skip without touching the network.
        if target.exists() and expected and target.stat().st_size == expected:
            on_chunk(target.stat().st_size)
            return True, ""

        already = part.stat().st_size if part.exists() else 0
        url = f"https://huggingface.co/{self._repo}/resolve/main/{rfilename}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", self.USER_AGENT)
        if already > 0:
            req.add_header("Range", f"bytes={already}-")
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 416 and already > 0:
                # Server says we're past EOF — partial is the full file.
                part.replace(target)
                on_chunk(already)
                return True, ""
            return False, f"{rfilename}: HTTP {e.code}"
        except Exception as e:
            return False, f"{rfilename}: {type(e).__name__}: {e}"

        on_chunk(already)  # account for resumed bytes immediately
        mode = "ab" if already > 0 else "wb"
        try:
            with open(part, mode) as f:
                while True:
                    if self._stop:
                        try:
                            part.unlink(missing_ok=True)
                        except OSError:
                            pass
                        try:
                            resp.close()
                        except Exception:
                            pass
                        return False, self.tr("Cancelled.")
                    if self._pause:
                        try:
                            resp.close()
                        except Exception:
                            pass
                        return False, "__paused__"
                    chunk = resp.read(self.CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    on_chunk(len(chunk))
        except Exception as e:
            return False, f"{rfilename}: {type(e).__name__}: {e}"
        finally:
            try:
                resp.close()
            except Exception:
                pass

        try:
            part.replace(target)
        except OSError as e:
            return False, f"{rfilename}: rename failed: {e}"
        return True, ""

    # ── top-level ──────────────────────────────────────────────
    def run(self) -> None:
        import time as _time
        self._dest.mkdir(parents=True, exist_ok=True)

        try:
            files = self._list_files()
        except Exception as e:
            self.finished.emit(False, self.tr("Repo list failed: {err}").format(err=e))
            return
        if not files:
            self.finished.emit(False, self.tr("Repo has no files."))
            return

        # Fill in sizes that the JSON API didn't expose, so the bar can
        # show a real percentage from the first byte. Cheap — one HEAD
        # per missing entry, all small config files.
        for i, (rfn, sz) in enumerate(files):
            if sz == 0:
                files[i] = (rfn, self._head_size(rfn))
        total = sum(sz for _, sz in files)

        downloaded = 0
        win_bytes = 0
        win_start = _time.monotonic()
        last_emit = win_start

        def on_chunk(n: int) -> None:
            nonlocal downloaded, win_bytes, last_emit, win_start
            downloaded += int(n)
            now = _time.monotonic()
            if now - last_emit >= 0.25:
                elapsed = max(1e-6, now - win_start)
                speed = (downloaded - win_bytes) / elapsed
                self.progress.emit(downloaded, total, max(0.0, speed))
                last_emit = now
                if now - win_start >= 1.0:
                    win_start = now
                    win_bytes = downloaded

        for rfn, sz in files:
            ok, msg = self._download_file(rfn, sz, on_chunk)
            if not ok:
                if msg == "__paused__":
                    self.paused.emit()
                    return
                self.finished.emit(False, msg)
                return

        if total > 0:
            self.progress.emit(total, total, 0.0)
        self.finished.emit(True, self.tr("Surya weights ready."))


# ── per-model card widget ────────────────────────────────────────────

class _Pill(QLabel):
    """Compact rounded chip used for purpose / project / size / source.

    Color slot lets the source / purpose stand out without screaming —
    everything stays slate-grey by default, only the accent fields tint."""

    def __init__(self, text: str, *, color: str = COLOR_FONT_SECTION_LABEL,
                 parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"QLabel {{"
            f"  color: {color};"
            f"  background-color: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 1px solid {COLOR_OUTLINE_GHOST};"
            f"  border-radius: 9px;"
            f"  padding: 1px 8px;"
            f"  font-size: 10px;"
            f"  font-weight: 500;"
            f"}}"
        )


def _icon_btn(icon_name: str, *, color: str = COLOR_FONT_MUTED,
              tooltip: str = "") -> QPushButton:
    """Compact icon-only pill button used in the card action strip."""
    from aglaia.gui.theme import lucide as _lucide
    btn = QPushButton()
    btn.setIcon(_lucide(icon_name, color=color, size=14))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(28, 28)
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setStyleSheet(
        f"QPushButton {{"
        f"  background: transparent;"
        f"  border: 1px solid {COLOR_OUTLINE};"
        f"  border-radius: 6px;"
        f"}}"
        f"QPushButton:hover {{ background: {COLOR_BG_OVERLAY_HOVER}; }}"
    )
    return btn


class _ModelCard(Card):
    """Streamlined per-model card.

    Single header row [title · found tag · actions] with a quiet pills
    row underneath conveying purpose / project / size / source. The
    progress bar + speed label appear only while a download is in
    flight. State machine:

      * empty       → [Download]
      * partial     → [Resume] [Trash]
      * downloading → progress + speed + [Pause-toggle] [Stop]
      * paused      → progress (frozen) + [Resume-toggle] [Stop]
      * installed   → [Trash]  (plus the green Installed tag)
    """

    def __init__(self, spec: ModelSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self._spec = spec
        self._thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None
        self._is_paused = False

        body = self.layout()
        body.setContentsMargins(12, 10, 12, 10)
        body.setSpacing(6)

        # ── header row: title + found tag + actions ─────────────
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(8)
        title = QLabel(spec.title)
        title.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 600; font-size: 14px;"
        )
        head_row.addWidget(title)

        self.found_tag = QLabel(self.tr("Installed"))
        self.found_tag.setStyleSheet(
            f"color: {COLOR_SUCCESS};"
            f" background-color: {COLOR_SUCCESS_BG_SOFT};"
            f" border: 1px solid {COLOR_SUCCESS_BORDER};"
            f" border-radius: 9px;"
            f" padding: 1px 8px;"
            f" font-size: 10px;"
            f" font-weight: 600;"
        )
        self.found_tag.setVisible(False)
        head_row.addWidget(self.found_tag)
        # Set once a model is downloaded *this session*: engines/backends load
        # their models at startup, so a freshly-fetched model isn't usable
        # until Aglaïa is restarted. Surfaced on the Installed badge.
        self._needs_restart = False
        head_row.addStretch(1)

        # Action buttons — visibility flipped in `_refresh_state`.
        self.btn_download = QPushButton(self.tr("Download"))
        self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_download.setStyleSheet(
            f"QPushButton {{"
            f"  background: {COLOR_PRIMARY_BG_SOFT};"
            f"  color: {COLOR_PRIMARY};"
            f"  border: 1px solid {COLOR_PRIMARY_BORDER};"
            f"  border-radius: 6px;"
            f"  padding: 4px 12px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{ background: {COLOR_PRIMARY_BG_STRONG}; }}"
        )
        self.btn_download.clicked.connect(self._start)
        head_row.addWidget(self.btn_download)

        # Single toggle button: pause ↔ resume while a DL is live.
        self.btn_pause = _icon_btn("pause", tooltip=self.tr("Pause download"))
        self.btn_pause.clicked.connect(self._toggle_pause)
        head_row.addWidget(self.btn_pause)

        self.btn_stop = _icon_btn("square", color=COLOR_ERROR,
                                  tooltip=self.tr("Stop and delete partial file"))
        self.btn_stop.clicked.connect(self._stop)
        head_row.addWidget(self.btn_stop)

        self.btn_trash = _icon_btn("trash-2", color=COLOR_ERROR,
                                   tooltip=self.tr("Delete on-disk weights"))
        self.btn_trash.clicked.connect(self._delete)
        head_row.addWidget(self.btn_trash)

        body.addLayout(head_row)

        # ── pills row: purpose · project · size · source ────────
        pills_row = QHBoxLayout()
        pills_row.setContentsMargins(0, 0, 0, 0)
        pills_row.setSpacing(6)
        pills_row.addWidget(_Pill(spec.purpose, color=COLOR_TERTIARY))
        pills_row.addWidget(_Pill(spec.project))
        pills_row.addWidget(_Pill(self.tr("~{n} MB").format(n=spec.approx_size_mb)))
        pills_row.addWidget(_Pill(spec.source, color=COLOR_PRIMARY))
        pills_row.addStretch(1)
        body.addLayout(pills_row)

        # ── progress row (hidden until DL starts) ───────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            f"color: {COLOR_FONT_SECTION_LABEL}; font-size: 11px;"
        )
        self.status_label.setVisible(False)
        prog_wrap = QWidget()
        prog_v = QVBoxLayout(prog_wrap)
        prog_v.setContentsMargins(0, 4, 0, 0)
        prog_v.setSpacing(2)
        prog_v.addWidget(self.progress_bar)
        prog_v.addWidget(self.status_label)
        self._prog_wrap = prog_wrap
        self._prog_wrap.setVisible(False)
        body.addWidget(prog_wrap)

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self._refresh_state()

    # ── compact pause/resume toggle ─────────────────────────────
    def _toggle_pause(self) -> None:
        if self._is_paused:
            # _reset_paused() cleared _worker after the pause request.
            # Resume rebuilds the worker from scratch; the underlying
            # streamer honours `.partialdl` byte-range resume.
            self._start()
            return
        if self._worker is None:
            return
        self._worker.pause()
        self._is_paused = True
        from aglaia.gui.theme import lucide as _lucide
        self.btn_pause.setIcon(_lucide("play", color=COLOR_FONT_MUTED, size=14))
        self.btn_pause.setToolTip(self.tr("Resume download"))

    # ── destination path ─────────────────────────────────────────
    def _dest(self) -> Path:
        return models_dir() / self._spec.filename

    def _is_installed(self) -> bool:
        """Installed = on-disk content matches the spec's SHA-1. For
        directory snapshots (Surya), fall back to "directory non-empty"
        because individual file hashes are checked elsewhere. Cached
        per-(path, mtime, size) so a 95 MB file doesn't get re-hashed
        on every UI refresh."""
        d = self._dest()
        if self._spec.kind == "hf-snapshot":
            if not (d.is_dir() and any(d.iterdir())):
                return False
            # Preferred check — every listed `required_files` entry
            # exists on disk with the expected byte count. Catches the
            # "config.json + README landed, GGUF didn't" case the SHA1
            # manifest also caught but without needing to hash 1.4 GB.
            if self._spec.required_files:
                for rel, expected_size in self._spec.required_files:
                    f = d / rel
                    if not f.is_file():
                        return False
                    try:
                        if f.stat().st_size != expected_size:
                            return False
                    except OSError:
                        return False
                return True
            expected = (self._spec.sha1 or "").lower()
            if not expected:
                return True  # snapshot-without-hash: existence is enough
            try:
                return self._snapshot_sha1(d).lower() == expected
            except Exception:
                return False
        if not d.exists() or d.stat().st_size <= 1024:
            return False
        expected = (self._spec.sha1 or "").lower()
        if not expected:
            # No reference hash → fall back to plain existence test so
            # the card still works for ad-hoc model entries.
            return True
        try:
            return self._cached_sha1(d) == expected
        except Exception:
            return False

    _SHA1_CACHE: dict[tuple, str] = {}

    @classmethod
    def _cached_sha1(cls, path: Path) -> str:
        """Streaming SHA-1, memoised by (path, mtime_ns, size). Reading
        a 95 MB EAST file is ~50 ms cold but adds up if you call
        `_is_installed` from every status refresh — the cache keeps
        idle UI redraws free."""
        import hashlib
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        cached = cls._SHA1_CACHE.get(key)
        if cached is not None:
            return cached
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        cls._SHA1_CACHE[key] = digest
        return digest

    _SNAPSHOT_SHA1_CACHE: dict[tuple, str] = {}

    @classmethod
    def _snapshot_sha1(cls, root: Path) -> str:
        """Deterministic SHA-1 over an entire hf-snapshot directory.

        Algorithm (matches what we use to populate `sha1` in
        model-list.json):
          1. Walk `root` recursively, sort by POSIX-relative path.
          2. Skip `.cache/` (HF tracking metadata — re-created on every
             download, not part of the model itself).
          3. For each file, compute per-file SHA-1 of bytes.
          4. Manifest = newline-joined `<sha1>  <relpath>` lines.
          5. Snapshot SHA-1 = SHA-1 of the manifest bytes.

        Memoised by (root, mtime_ns of root, total bytes) so the 1.4 GB
        Surya snapshot is hashed once per session, not on every UI tick."""
        import hashlib
        if not root.is_dir():
            return ""
        total_bytes = 0
        items: list[Path] = []
        for p in sorted(root.rglob("*"), key=lambda x: x.as_posix()):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if rel.startswith(".cache/"):
                continue
            items.append(p)
            try:
                total_bytes += p.stat().st_size
            except OSError:
                continue
        try:
            mtime = root.stat().st_mtime_ns
        except OSError:
            mtime = 0
        cache_key = (str(root), mtime, total_bytes)
        cached = cls._SNAPSHOT_SHA1_CACHE.get(cache_key)
        if cached is not None:
            return cached
        lines: list[str] = []
        for p in items:
            digest = cls._cached_sha1(p)
            rel = p.relative_to(root).as_posix()
            lines.append(f"{digest}  {rel}")
        manifest = "\n".join(lines).encode()
        out = hashlib.sha1(manifest).hexdigest()
        cls._SNAPSHOT_SHA1_CACHE[cache_key] = out
        return out

    def _refresh_state(self) -> None:
        """Single visibility pass driven by 4 booleans. Keeps the
        action strip impossible to land in a half-visible inconsistent
        state during reconnects."""
        is_dl = self._thread is not None and not self._is_paused
        is_paused = self._is_paused
        installed = self._is_installed()
        has_partial = self._has_partial()

        show_found = installed and not is_dl and not is_paused
        if show_found and self._needs_restart:
            self.found_tag.setText(self.tr("Installed · restart to use"))
            self.found_tag.setToolTip(self.tr(
                "Models are loaded at startup — restart Aglaïa for this "
                "newly-downloaded model to become available."))
        else:
            self.found_tag.setText(self.tr("Installed"))
            self.found_tag.setToolTip("")
        self.found_tag.setVisible(show_found)

        # In-flight controls: pause-toggle + stop are the only buttons
        # visible while bytes are flowing or the download is parked.
        in_flight = is_dl or is_paused
        self.btn_pause.setVisible(in_flight)
        self.btn_stop.setVisible(in_flight)

        # Trash: visible when there's something to delete AND we're
        # not actively writing to it. Hidden during in-flight DL so a
        # mis-click doesn't take out a file mid-stream.
        self.btn_trash.setVisible(
            (installed or has_partial) and not in_flight
        )

        # Download / Resume button.
        if installed or in_flight:
            self.btn_download.setVisible(False)
        else:
            self.btn_download.setText(self.tr("Resume") if has_partial else self.tr("Download"))
            self.btn_download.setVisible(True)

        self._prog_wrap.setVisible(in_flight)

    # Back-compat alias for callers that still poke _refresh_status.
    def _refresh_status(self) -> None:
        self._refresh_state()

    def _has_partial(self) -> bool:
        d = self._dest()
        return d.with_suffix(d.suffix + ".partialdl").exists()

    def _delete(self) -> None:
        """Wipe whatever's on disk for this model. No confirmation —
        downloads are cheap to redo and the trash button is explicit
        enough on its own."""
        import shutil
        d = self._dest()
        # Always also remove the matching .partialdl if present.
        for p in (d, d.with_suffix(d.suffix + ".partialdl")):
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
            except Exception:
                continue
        self._refresh_state()

    # ── state transitions ───────────────────────────────────────
    def _start(self) -> None:
        """Kick off (or resume) a download. Same path for first click,
        post-pause resume, and Resume-button-after-app-restart: the
        underlying workers honour `.partialdl` byte-Range resume on
        their own."""
        url = self._spec.url
        dest = self._dest()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self._spec.kind == "hf-snapshot":
            self._worker = SuryaWorker(url, dest)
            self.progress_bar.setRange(0, 0)  # indeterminate until HF size known
        else:
            # Refuse cleartext transports. Any model URL surfaced to
            # the UI must be HTTPS — we sign builds, but won't trust
            # the bytes coming back from an unencrypted hop.
            if not url.lower().startswith("https://"):
                self.status_label.setText(self.tr(
                    "Refusing non-HTTPS URL — edit "
                    "model-list.json to use https://"
                ))
                self.status_label.setVisible(True)
                self._refresh_state()
                return
            self._worker = DownloadWorker(url, dest)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.paused.connect(self._on_paused)
        # Standard QThread cleanup chain — avoids the
        # "QThread: Destroyed while thread is still running" race
        # that hits when the user pauses mid-network-read. Letting
        # signals do the teardown (vs ``quit() + wait(2000)`` in the
        # slot) means the widget never holds a stale ref to a QThread
        # whose OS thread hasn't exited yet.
        self._worker.finished.connect(self._thread.quit)
        self._worker.paused.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        # Capture the locals so the lambda doesn't dangle on a freshly
        # reassigned ``self._thread`` after Resume.
        _t, _w = self._thread, self._worker
        self._thread.finished.connect(
            lambda: self._on_thread_done(_t, _w)
        )
        self._thread.start()

        self._is_paused = False
        self.status_label.setVisible(True)
        self.status_label.setText(self.tr("Starting…"))
        # Make the pause icon a "pause" again in case we just resumed
        # from a paused state.
        from aglaia.gui.theme import lucide as _lucide
        self.btn_pause.setIcon(_lucide("pause", color=COLOR_FONT_MUTED, size=14))
        self.btn_pause.setToolTip(self.tr("Pause download"))
        self._refresh_state()

    def _stop(self) -> None:
        """Stop = trash the partial. Worker tears down via its own stop
        flag; the partial file is deleted by the worker as part of the
        Cancelled path. Stop is destructive on purpose — Pause is the
        non-destructive alternative."""
        if self._worker is not None:
            self._worker.stop()
        else:
            self._reset_idle()

    def _reset_idle(self, *, keep_status: bool = False) -> None:
        self.status_label.setVisible(keep_status)
        self._thread = None
        self._worker = None
        self._is_paused = False
        self._refresh_state()

    def _reset_paused(self) -> None:
        """Thread exited after pause request — keep the bar frozen at
        the last progress %, swap the pause-icon for a play-icon so the
        same button now resumes."""
        from aglaia.gui.theme import lucide as _lucide
        self._thread = None
        self._worker = None
        self.btn_pause.setIcon(_lucide("play", color=COLOR_FONT_MUTED, size=14))
        self.btn_pause.setToolTip(self.tr("Resume download"))
        if self.status_label.text() and "paused" not in self.status_label.text():
            self.status_label.setText(self.status_label.text() + self.tr("  — paused"))
        self._refresh_state()

    # ── worker callbacks ─────────────────────────────────────────
    def _on_progress(self, done: int, total: int, speed: float) -> None:
        if total > 0:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            pct = max(0, min(100, int(done * 100 / total)))
            self.progress_bar.setValue(pct)
            self.status_label.setText(self.tr(
                "{done} / {total}  •  {speed}  •  ETA {eta}"
            ).format(
                done=_fmt_bytes(done), total=_fmt_bytes(total),
                speed=_fmt_speed(speed),
                eta=_fmt_eta(done, total, speed),
            ))
        else:
            self.status_label.setText(self.tr("{done} downloaded").format(done=_fmt_bytes(done)))

    def _on_finished(self, ok: bool, msg: str) -> None:
        # Thread teardown happens via the signal chain wired in
        # ``_start``; don't quit/wait here (the blocking wait was the
        # source of the "Destroyed while thread is still running" race).
        if ok:
            verified, vmsg = self._verify_installed()
            if verified:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(100)
                self.status_label.setVisible(False)
                self._needs_restart = True
                self._refresh_state()
            else:
                self.status_label.setText(vmsg)
                self._reset_idle(keep_status=True)
        else:
            self.status_label.setText(msg)
            self._reset_idle(keep_status=True)

    def _verify_installed(self) -> tuple[bool, str]:
        """Run the same hash check `_is_installed` uses, with a
        user-readable explanation when the hash mismatches. Single-file
        (`kind=file`) verifies via SHA-1 of file bytes; directory
        snapshots (`kind=hf-snapshot`) verify via the per-file manifest
        SHA-1 (`_snapshot_sha1`)."""
        if not self._spec.sha1:
            return True, ""
        d = self._dest()
        if not d.exists():
            return False, self.tr("Download finished but file is missing.")
        try:
            if self._spec.kind == "hf-snapshot":
                got = self._snapshot_sha1(d).lower()
            else:
                got = self._cached_sha1(d).lower()
        except Exception as e:
            return False, self.tr("SHA-1 check failed: {err}").format(err=e)
        if got != self._spec.sha1.lower():
            return False, self.tr(
                "SHA-1 mismatch — got {got}…, expected "
                "{expected}…. File may be "
                "corrupted or from a different revision."
            ).format(got=got[:10], expected=self._spec.sha1[:10])
        return True, ""

    def _on_paused(self) -> None:
        # Same as _on_finished: rely on signal-chained teardown. We
        # only update state here so the UI flips immediately; the actual
        # thread exit is observed via ``_on_thread_done`` below.
        self._reset_paused()

    def _on_thread_done(self, thread: QThread, worker: QObject) -> None:
        """Final cleanup once the QThread has fully exited. Called via
        ``thread.finished`` signal. Clears refs only when the just-
        exited thread is the one still recorded on ``self`` — otherwise
        a Resume already replaced it with a fresh thread and we must
        not stomp the new pair."""
        if self._thread is thread:
            self._thread = None
        if self._worker is worker:
            self._worker = None


# ── dialog ──────────────────────────────────────────────────────────

class ModelDownloaderDialog(QDialog):
    """Modeless dialog version of the model downloader.

    Built as a dialog instead of a tab so it can pop on first launch
    (when no MainWindow tab strip exists yet) and so the user can keep
    interacting with the main UI underneath while a 1.3 GB Surya pull
    streams in the background."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Model Downloader"))
        self.setModal(False)
        self.setMinimumSize(760, 460)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header strip — quiet intro, target dir.
        header = QWidget()
        head_v = QVBoxLayout(header)
        head_v.setContentsMargins(20, 16, 20, 12)
        head_v.setSpacing(4)
        title = QLabel(self.tr("Models"))
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {COLOR_FONT_PRIMARY};"
        )
        head_v.addWidget(title)
        from aglaia.app_data import models_dir as _md
        sub = QLabel(self.tr("Saved to <code>{path}</code>").format(path=_md()))
        sub.setStyleSheet(
            f"color: {COLOR_FONT_SECTION_LABEL}; font-size: 11px;"
        )
        sub.setTextFormat(Qt.TextFormat.RichText)
        head_v.addWidget(sub)
        outer.addWidget(header)

        # Body — scrollable card list.
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(20, 4, 20, 20)
        v.setSpacing(12)

        # Reload the JSON registry on every dialog open so user edits to
        # `model-list.json` (or a future remote-pull refresh) take effect
        # without an app restart. Stale module-level `MODEL_SPECS` would
        # otherwise hold the values from process startup.
        specs = _load_model_specs() or MODEL_SPECS
        rec = [s for s in specs if s.section == "recommended"]
        oth = [s for s in specs if s.section == "other"]
        if rec:
            v.addWidget(_section_header(self.tr("Recommended")))
            for spec in rec:
                v.addWidget(_ModelCard(spec))
        if oth:
            v.addWidget(_section_header(self.tr("Other")))
            for spec in oth:
                v.addWidget(_ModelCard(spec))
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Footer — Close.
        footer = QHBoxLayout()
        footer.setContentsMargins(20, 10, 20, 14)
        footer.addStretch(1)
        close = QPushButton(self.tr("Close"))
        close.clicked.connect(self.close)
        footer.addWidget(close)
        outer.addLayout(footer)


# Backwards-compat alias: previous code (MainWindow tab opener,
# PageWarningDialog) referenced `ModelDownloaderTab`. Point it at
# the new dialog so we don't have to chase every call site at once.
ModelDownloaderTab = ModelDownloaderDialog


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLOR_FONT_PRIMARY}; font-weight: 700; font-size: 16px;"
        f" margin-top: 8px;"
    )
    return lbl


# ── formatting helpers ───────────────────────────────────────────────

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_speed(bps: float) -> str:
    return f"{_fmt_bytes(bps)}/s"


def _fmt_eta(done: int, total: int, speed: float) -> str:
    if speed <= 0 or total <= 0 or done >= total:
        return "—"
    remaining = (total - done) / speed
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining/60:.0f}m"
    return f"{remaining/3600:.1f}h"
