# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Single-window first-run onboarding wizard.

Replaces the old two-dialog flow (WelcomeDialog → ModelInstallPrompt → a
deferred download hosted inside the MainWindow that then needed a broken
"restart"). Three steps in one fixed-size, floating window:

  1. Welcome  — logo, intro, language selection.
  2. Permissions — what the OS will ask for and why.
  3. Models — pick the offline models to fetch, downloaded *here* on one
     global progress bar.

Because the wizard runs in ``app.py:_qt_app`` BEFORE the StartupWindow — and
before any OCR/voice/layout engine is imported — the downloads land on disk
ahead of first use, so the engines see them with no restart. Finishing the
wizard simply flows into the normal StartupWindow.

Gated once via the ``welcome_seen`` config flag; never blocks launch.
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from aglaia.assets import asset_path
from aglaia.gui.colors import (
    COLOR_FONT_DIM, COLOR_FONT_MUTED, COLOR_FONT_PRIMARY, COLOR_PRIMARY,
    active_palette_name,
)
from aglaia.i18n import SUPPORTED_LOCALES

_IS_MAC = sys.platform == "darwin"

# Permission review copy — platform-neutral so it reads correctly on every OS.
_PERM_INTRO = (
    "Aglaïa turns photos of book and document pages into clean, "
    "<b>searchable PDFs and Markdown</b> — fully offline by default."
)
_PERM_ROWS = [
    ("📷", "Camera & microphone",
     "The camera captures pages with your webcam; the microphone powers "
     "optional hands-free voice capture. Your system asks the first time you "
     "use each — or skip both and just import existing images and PDFs."),
    ("🔑", "System keychain",
     "Asked <b>only if you choose to save a Cloud OCR API key</b>. The key is "
     "kept in your system keychain; everything else, including offline OCR, "
     "works without it."),
    ("📁", "Local files",
     "Projects, settings and models live in Aglaïa's app-data folder. Pages "
     "stay on your machine unless you explicitly pick a cloud OCR engine."),
]

# Per-model display copy keyed by ModelSpec.key. Size is pulled live from the
# spec; the caption is the small grey line under the description.
_MODEL_COPY = {
    "east": ("EAST — page detection",
             "Finds the text region on each photo so pages crop cleanly."),
    "vosk_en": ("Vosk — voice control",
                "Hands-free page capture by voice, fully offline."),
    "surya": ("Surya — neural OCR",
              "Higher-quality OCR for difficult scans (large download)."),
}
_MODEL_ORDER = ["east", "vosk_en", "surya"]


class _StepDots(QWidget):
    """Progress stepper: a labelled dot per step. Upcoming dots are dimmed;
    completed + current use the primary colour; the current dot gets an extra
    outline ring and a bold primary label."""

    def __init__(self, labels: list[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._labels = labels
        self._current = 0
        self.setMinimumHeight(44)

    def set_current(self, idx: int) -> None:
        self._current = idx
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self._labels)
        if n == 0:
            return
        slot = self.width() / n
        dot_y = 11
        r = 6
        prim = QColor(COLOR_PRIMARY)
        dim = QColor(COLOR_FONT_DIM)
        font = QFont(self.font())
        font.setPixelSize(10)
        for i, label in enumerate(self._labels):
            cx = slot * (i + 0.5)
            done_or_current = i <= self._current
            # Dot.
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(prim if done_or_current else dim)
            p.drawEllipse(int(cx - r), dot_y, 2 * r, 2 * r)
            # Outline ring on the current dot.
            if i == self._current:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(prim, 2))
                p.drawEllipse(int(cx - r - 4), dot_y - 4, 2 * (r + 4), 2 * (r + 4))
            # Label.
            if i == self._current:
                font.setBold(True)
                p.setPen(QColor(COLOR_FONT_PRIMARY))
            else:
                font.setBold(False)
                p.setPen(QColor(COLOR_FONT_MUTED))
            p.setFont(font)
            p.drawText(int(cx - slot / 2), 26, int(slot), 16,
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       label)
        p.end()


class OnboardingWizard(QDialog):
    """First-run setup wizard. Use :meth:`run_if_first_run`."""

    _W, _H = 560, 600

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Set up Aglaïa"))
        self.setModal(True)
        # Fixed size + Dialog window type → tiling WMs float it instead of
        # tiling/stretching it into something weird.
        self.setWindowFlag(Qt.WindowType.Dialog, True)
        self.setFixedSize(self._W, self._H)

        self._downloading = False
        self._dl_queue: list = []          # remaining ModelSpec to fetch
        self._dl_total_mb = 0              # sum of selected sizes
        self._dl_done_mb = 0              # completed models' size
        self._dl_failures: list[str] = []
        self._thread: Optional[QThread] = None
        self._worker = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_welcome_page())      # 0
        self._stack.addWidget(self._build_permissions_page())  # 1
        self._stack.addWidget(self._build_models_page())       # 2
        outer.addWidget(self._stack, 1)

        # Footer: Back · step-dots · Next/Finish.
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {COLOR_FONT_DIM};")
        outer.addWidget(sep)

        footer = QHBoxLayout()
        footer.setContentsMargins(20, 10, 20, 14)
        footer.setSpacing(12)
        self._btn_back = QPushButton(self.tr("← Back"))
        self._btn_back.clicked.connect(lambda: self._go(-1))
        footer.addWidget(self._btn_back)
        self._dots = _StepDots([self.tr("Welcome"), self.tr("Permissions"),
                                self.tr("Models")])
        footer.addWidget(self._dots, 1)
        self._btn_next = QPushButton(self.tr("Next →"))
        self._btn_next.setDefault(True)
        self._btn_next.clicked.connect(self._on_next)
        footer.addWidget(self._btn_next)
        outer.addLayout(footer)

        self._update_nav()

    # ── pages ────────────────────────────────────────────────────────────
    def _page_host(self) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(28, 26, 28, 18)
        v.setSpacing(14)
        return w, v

    def _build_welcome_page(self) -> QWidget:
        w, v = self._page_host()
        v.addStretch(1)
        scheme = "dark" if active_palette_name() == "dark" else "light"
        pm = QPixmap(str(asset_path("brand", f"aglaia-{scheme}.png")))
        if not pm.isNull():
            dpr = self.devicePixelRatioF() or 1.0
            pm.setDevicePixelRatio(dpr)
            scaled = pm.scaledToHeight(int(96 * dpr),
                                       Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(dpr)
            logo = QLabel()
            logo.setPixmap(scaled)
            logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            v.addWidget(logo)
        else:
            t = QLabel("Aglaïa")
            t.setStyleSheet("font-size: 30px; font-weight: 800;")
            t.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            v.addWidget(t)

        sub = QLabel(self.tr("Take a few minutes to set up Aglaïa."))
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 14px;")
        v.addWidget(sub)
        v.addSpacing(18)

        lang_row = QHBoxLayout()
        lang_row.addStretch(1)
        lang_lbl = QLabel(self.tr("Language"))
        lang_row.addWidget(lang_lbl)
        self._lang_combo = QComboBox()
        for code, label in SUPPORTED_LOCALES:
            self._lang_combo.addItem(label, userData=code)
        self._preselect_language()
        self._lang_combo.setMinimumWidth(200)
        lang_row.addWidget(self._lang_combo)
        lang_row.addStretch(1)
        v.addLayout(lang_row)

        hint = QLabel(self.tr("A language change applies from the next launch."))
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hint.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        v.addWidget(hint)
        v.addStretch(2)
        return w

    def _build_permissions_page(self) -> QWidget:
        w, v = self._page_host()
        title = QLabel(self.tr("Permissions"))
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        v.addWidget(title)
        intro = QLabel(self.tr(_PERM_INTRO))
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        v.addWidget(intro)
        for emoji, head, body in _PERM_ROWS:
            v.addWidget(self._icon_row(emoji, self.tr(head), self.tr(body)))
        v.addStretch(1)
        foot = QLabel(self.tr(
            "Source-available, signed & built by CI. You stay in control of "
            "every permission."))
        foot.setWordWrap(True)
        foot.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        v.addWidget(foot)
        return w

    def _build_models_page(self) -> QWidget:
        from aglaia.gui.ModelDownloaderTab import (
            MODEL_SPECS, _load_model_specs, is_model_installed,
        )
        w, v = self._page_host()
        title = QLabel(self.tr("Offline models"))
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        v.addWidget(title)
        intro = QLabel(self.tr(
            "Aglaïa runs everything offline. Pick the models to download now "
            "— you can always change this later in the downloader."))
        intro.setWordWrap(True)
        v.addWidget(intro)

        specs = {s.key: s for s in (_load_model_specs() or MODEL_SPECS)}
        self._model_checks: dict[str, tuple[QCheckBox, object]] = {}
        for key in _MODEL_ORDER:
            spec = specs.get(key)
            if spec is None:
                continue
            head, desc = _MODEL_COPY[key]
            installed = is_model_installed(key)
            chk = QCheckBox()
            # Default state per platform.
            if key == "east":
                # Required off-Apple (Apple Vision covers detection on macOS).
                chk.setChecked(not _IS_MAC)
                chk.setEnabled(_IS_MAC)         # off-mac: checked + locked
            elif key == "vosk_en":
                chk.setChecked(True)            # recommended everywhere, optional
            else:  # surya
                chk.setChecked(False)           # opt-in everywhere
            if installed:
                chk.setChecked(True)
                chk.setEnabled(False)
            self._model_checks[key] = (chk, spec)

            caption_bits = [f"~{spec.approx_size_mb} MB"]
            if installed:
                caption_bits.append(self.tr("already installed"))
            elif key == "east" and not _IS_MAC:
                caption_bits.append(self.tr("required on this platform"))
            else:
                caption_bits.append(self.tr("optional"))
            v.addWidget(self._model_row(chk, self.tr(head), self.tr(desc),
                                        " · ".join(caption_bits)))

        v.addStretch(1)
        self._dl_bar = QProgressBar()
        self._dl_bar.setRange(0, 1000)
        self._dl_bar.setTextVisible(False)
        self._dl_bar.setVisible(False)
        v.addWidget(self._dl_bar)
        self._dl_status = QLabel("")
        self._dl_status.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        self._dl_status.setVisible(False)
        v.addWidget(self._dl_status)
        return w

    # ── small row builders ───────────────────────────────────────────────
    def _icon_row(self, emoji: str, head: str, body: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        icon = QLabel(emoji)
        icon.setStyleSheet("font-size: 22px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        icon.setFixedWidth(30)
        h.addWidget(icon)
        txt = QLabel(f"<b>{head}</b><br>{body}")
        txt.setWordWrap(True)
        txt.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(txt, 1)
        return w

    def _model_row(self, chk: QCheckBox, head: str, desc: str,
                   caption: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(10)
        h.addWidget(chk, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        title = QLabel(f"<b>{head}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(title)
        d = QLabel(desc)
        d.setWordWrap(True)
        col.addWidget(d)
        cap = QLabel(caption)
        cap.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        col.addWidget(cap)
        h.addLayout(col, 1)
        return w

    def _preselect_language(self) -> None:
        try:
            from aglaia.app_data import db
            with db.session() as conn:
                cur = db.get(conn, db.KEY_LANGUAGE, "") or ""
        except Exception:
            cur = ""
        idx = self._lang_combo.findData(cur)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)

    # ── navigation ───────────────────────────────────────────────────────
    def _go(self, delta: int) -> None:
        if self._downloading:
            return
        i = max(0, min(self._stack.count() - 1, self._stack.currentIndex() + delta))
        self._stack.setCurrentIndex(i)
        self._update_nav()

    def _on_next(self) -> None:
        if self._stack.currentIndex() < self._stack.count() - 1:
            self._go(+1)
        else:
            self._finish()

    def _update_nav(self) -> None:
        i = self._stack.currentIndex()
        last = i == self._stack.count() - 1
        self._dots.set_current(i)
        self._btn_back.setEnabled(i > 0 and not self._downloading)
        self._btn_next.setText(self.tr("Finish") if last else self.tr("Next →"))
        self._btn_next.setEnabled(not self._downloading)

    # ── finish + downloads ───────────────────────────────────────────────
    def _finish(self) -> None:
        # Persist + apply the language choice (applies to the StartupWindow
        # shown next and onward this session).
        code = self._lang_combo.currentData() or ""
        try:
            from aglaia.app_data import db
            with db.session() as conn:
                if (db.get(conn, db.KEY_LANGUAGE, "") or "") != code:
                    db.set(conn, db.KEY_LANGUAGE, code)
                    conn.commit()
                    app = QApplication.instance()
                    if app is not None:
                        from aglaia.i18n import install_translator
                        install_translator(app, code)
        except Exception:
            pass

        # Collect models to fetch: checked, not already installed.
        from aglaia.gui.ModelDownloaderTab import is_model_installed
        queue = [spec for key, (chk, spec) in self._model_checks.items()
                 if chk.isChecked() and not is_model_installed(key)]
        if not queue:
            self.accept()
            return

        self._dl_queue = queue
        self._dl_total_mb = sum(max(1, s.approx_size_mb) for s in queue) or 1
        self._dl_done_mb = 0
        self._dl_failures = []
        self._downloading = True
        self._dl_bar.setVisible(True)
        self._dl_bar.setValue(0)
        self._dl_status.setVisible(True)
        self._update_nav()
        self._start_next()

    def _start_next(self) -> None:
        from aglaia.app_data import models_dir
        from aglaia.gui.ModelDownloaderTab import DownloadWorker, SuryaWorker
        if not self._dl_queue:
            self._downloading = False
            if self._dl_failures:
                # Don't block launch on a failed model; report and finish.
                self._dl_status.setText(self.tr(
                    "Some downloads failed: {names}. You can retry later in "
                    "the downloader.").format(names=", ".join(self._dl_failures)))
            self.accept()
            return

        spec = self._dl_queue[0]
        self._cur_spec = spec
        head = _MODEL_COPY.get(spec.key, (spec.title, ""))[0]
        self._dl_status.setText(self.tr("Downloading {name}…").format(name=head))
        dest = models_dir() / spec.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if spec.kind == "hf-snapshot":
            worker = SuryaWorker(spec.url, dest)
        else:
            worker = DownloadWorker(spec.url, dest)

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_one_finished)
        # Teardown chain (no blocking wait → avoids "destroyed while running").
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._worker, self._thread = worker, thread
        thread.start()

    def _on_progress(self, downloaded: int, total: int, _speed: float) -> None:
        frac = (downloaded / total) if total > 0 else 0.0
        size = max(1, getattr(self._cur_spec, "approx_size_mb", 1))
        overall = (self._dl_done_mb + frac * size) / self._dl_total_mb
        self._dl_bar.setValue(int(max(0.0, min(1.0, overall)) * 1000))

    def _on_one_finished(self, ok: bool, message: str) -> None:
        spec = self._cur_spec
        self._dl_done_mb += max(1, spec.approx_size_mb)
        if not ok:
            self._dl_failures.append(
                _MODEL_COPY.get(spec.key, (spec.title, ""))[0])
        self._dl_bar.setValue(int(min(1.0, self._dl_done_mb / self._dl_total_mb) * 1000))
        self._worker = None
        self._thread = None
        if self._dl_queue:
            self._dl_queue.pop(0)
        self._start_next()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Block the window-close button while a download thread is live —
        # otherwise the dialog (and its thread refs) could be torn down
        # mid-stream. accept() flips _downloading off first, so programmatic
        # completion still closes cleanly.
        if self._downloading:
            event.ignore()
            return
        super().closeEvent(event)

    # ── entry point ──────────────────────────────────────────────────────
    @classmethod
    def run_if_first_run(cls, parent: Optional[QWidget] = None) -> None:
        """Show the wizard once (gated by ``welcome_seen``), running its
        downloads inline. Any failure is swallowed — onboarding must never
        block launch."""
        try:
            from aglaia.app_data import db
            with db.session() as conn:
                if db.get(conn, db.KEY_WELCOME_SEEN, False):
                    return
            cls(parent).exec()
            with db.session() as conn:
                db.set(conn, db.KEY_WELCOME_SEEN, True)
                # Also retire the legacy standalone model invite.
                db.set(conn, db.KEY_MODELS_PROMPT_DISMISSED, True)
                conn.commit()
        except Exception as e:
            print(f"onboarding: skipped ({e})", file=sys.stderr)
