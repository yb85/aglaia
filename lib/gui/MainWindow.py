# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import os
import time
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from slugify import slugify
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QGroupBox, QMessageBox, QApplication, QTabWidget, QTabBar, QComboBox, QSlider, QDialog, QStackedWidget, QButtonGroup, QToolButton
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThreadPool, QRunnable
from PySide6.QtGui import QKeySequence, QPixmap

from lib.gui.WebcamThread import WebcamThread
from lib.workers.ProcessMonitor import ProcessMonitor, console
from lib.ImageBuffer import ImageBuffer, ImageType
from lib.gui.ScanItemWidget import ScanItemWidget
from lib.workers.PDFprocessor import create_pdf_from_images, create_pdf_from_db
from lib.workers.Calibrator import Calibrator, load_calibration, save_calibration

from lib.workers.Initializer import load_pipeline_def, pipeline_step_descriptions
from lib.storage.db import db_session
from lib.storage.persister import Persister, make_thumb
from lib.storage.repo import (
    ScanRepo, NodeRepo, ImageRepo, ThumbRepo, OcrRepo, StepOverrideRepo,
)
from lib.gui.PipelineEditorWidget import PipelineEditorDialog
from lib.workers.OcrWorker import OcrWorker
from lib.gui.sidebar import SidebarPanel
from lib.gui.sidebar.tabs import (
    CaptureTab, ExportTab, ImportTab, OcrTab, PipelineTab,
)
from lib.gui.colors import (
    COLOR_BG,
    COLOR_BG_BUTTON,
    COLOR_BG_BUTTON_HOVER,
    COLOR_BG_BUTTON_PRESSED,
    COLOR_BG_OVERLAY_HOVER,
    COLOR_BG_SURFACE_ALT,
    COLOR_BG_VIDEO,
    COLOR_ERROR,
    COLOR_FONT_DISABLED,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_ON_BUTTON,
    COLOR_FONT_PLACEHOLDER,
    COLOR_FONT_PRIMARY,
    COLOR_INFO,
    COLOR_OUTLINE_BUTTON,
    COLOR_OUTLINE_GHOST,
    COLOR_OUTLINE_STRONG,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_SUCCESS,
)


def _ocr_rank(state: str) -> int:
    return {"none": 0, "stale": 1, "fresh": 2}.get(state, 0)


class _ClickableLabel(QLabel):
    """QLabel that emits a callback on left-click. Used for the bottom
    status bar's status-message slot so the user can click it open to
    the log tab (mirrors the LogStrip behaviour it replaces)."""

    def __init__(self, text: str = "", on_click=None, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, ev):  # noqa: N802 — Qt API
        if ev.button() == Qt.MouseButton.LeftButton and self._on_click is not None:
            self._on_click()
        super().mousePressEvent(ev)


import threading as _threading

_thumb_tls = _threading.local()


def _thumb_worker_conn(db_path: str):
    """One SQLite connection per thumb-pool worker thread, reused across
    jobs (SQLite connections are not safe to share across threads). Keyed by
    db_path so a project switch on a reused thread reopens the right DB."""
    entry = getattr(_thumb_tls, "entry", None)
    if entry is None or entry[0] != db_path:
        if entry is not None:
            try:
                entry[1].close()
            except Exception:
                pass
        from lib.storage.db import open_db
        conn = open_db(db_path)
        _thumb_tls.entry = (db_path, conn)
        return conn
    return entry[1]


class _ThumbJob(QRunnable):
    """Build one thumbnail off the GUI thread: read the full image blob,
    decode + resize + encode, write the thumb back. Emits `done` (a signal
    owned by the long-lived ThumbLoader, so there is no lifetime race)."""

    def __init__(self, db_path: str, image_id: int, max_dim: int, done):
        super().__init__()
        self._db_path = db_path
        self._image_id = image_id
        self._max_dim = max_dim
        self._done = done

    def run(self) -> None:  # noqa: D401 — Qt API
        ok = False
        try:
            conn = _thumb_worker_conn(self._db_path)
            thumbs = ThumbRepo(conn)
            if thumbs.get(self._image_id, self._max_dim) is None:
                src = ImageRepo(conn).get(self._image_id)
                if src is not None:
                    blob, w, h = make_thumb(bytes(src["blob"]), self._max_dim)
                    thumbs.upsert(self._image_id, self._max_dim, w, h, blob)
                    conn.commit()
            ok = True
        except Exception:
            ok = False
        # Emitted from the worker thread; the receiver (ThumbLoader) lives on
        # the GUI thread, so Qt delivers it queued there.
        self._done.emit(self._image_id, self._max_dim, ok)


class ThumbLoader(QObject):
    """Async thumbnail cache.

    The widgets ask for thumbnails one-by-one as they paint / as pipeline
    steps stream in. A cache HIT returns the small thumb blob via one fast
    indexed read on the GUI thread. A MISS does NOT decode/encode/write on
    the GUI thread — that froze the event loop during processing (full
    multi-MB blob read + 2 resamples + JPEG encode + a DB write contending
    with the worker processes for SQLite's single writer). Instead it
    schedules the build on a small thread pool and returns None (the caller
    shows its "pending" placeholder); when the build lands, `ready` fires and
    the owning view re-requests — now a cache hit.
    """

    ready = Signal(int)           # image_id whose thumb just became available
    _done = Signal(int, int, bool)  # image_id, max_dim, ok — worker → GUI

    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
        from lib.storage.db import open_db
        self._conn = open_db(db_path)     # GUI thread: cached reads only
        self._thumbs = ThumbRepo(self._conn)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._inflight: set[tuple[int, int]] = set()
        self._done.connect(self._on_done)

    def __call__(self, image_id: int, max_dim: int = 256) -> bytes | None:
        if image_id is None:
            return None
        row = self._thumbs.get(image_id, max_dim)
        if row is not None:
            return bytes(row["blob"])
        key = (int(image_id), int(max_dim))
        if key not in self._inflight:
            self._inflight.add(key)
            self._pool.start(_ThumbJob(self._db_path, key[0], key[1], self._done))
        return None

    def _on_done(self, image_id: int, max_dim: int, ok: bool) -> None:
        self._inflight.discard((image_id, max_dim))
        if ok:
            self.ready.emit(image_id)

    def close(self) -> None:
        try:
            self._pool.waitForDone(2000)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


class MainWindow(QMainWindow):
    # Broadcast: per-layout selected stage / visibility changed in DB.
    # All three scan views subscribe and resync their UI. Single source
    # of truth = `branches` table; one signal per kind of change.
    branch_visibility_changed = Signal(int, str, bool)   # scan_id, branch_label, hidden
    # Per-page processor disable: a step toggled on/off for one layout.
    # Replaces exit-stage navigation. Views dim/undim the cell immediately;
    # the scan reruns from raw (worker re-applies the override per branch).
    step_disabled_changed = Signal(int, str, int, bool)  # scan_id, branch_path, step_idx, disabled
    # Fired after any write that affects OCR staleness (chosen_node_id
    # move, OCR run inserted as fresh, OCR forced). Single broadcast =
    # every view + bottom-bar OCR frame resyncs from DB.
    ocr_state_changed = Signal()

    def __init__(self, args, processing_queue, ignored_queue, log_queue,
                 slug_name, start_idx, *, db_path: str, pipeline_version_id: int,
                 source: str = "capture",
                 pipeline_yaml_path: Optional[Path] = None,
                 apply_pipeline_callback=None,
                 force_reprocess_callback=None,
                 reprocess_scans_callback=None,
                 reprocess_branch_callback=None,
                 stop_pipeline_callback=None):
        super().__init__()
        # Calculate pipeline length for progress indication
        pipeline_path = args.pipeline if args.pipeline else Path("config/pipelines/book_curved_x2.yaml")
        pdef = load_pipeline_def(pipeline_path)
        self.pipeline_steps = []
        # Sidebar timing view keys off the bare processor class name
        # — that's what the chain's `timing` log_queue events carry.
        # Keeping the slugified `pipeline_steps` separately so the
        # progress bar (which counts produced nodes by step index)
        # stays untouched.
        self.pipeline_proc_names: list[str] = []
        if pdef:
            for i, step in enumerate(pdef.get("pipeline", []), 1):
                sname = step.get("name", step.get("processor"))
                inst = f"{i:02d}_{slugify(sname, separator='_')}"
                self.pipeline_steps.append(inst)
                self.pipeline_proc_names.append(
                    step.get("processor") or sname
                )
        # Replay runs as final step. Skipped when:
        #   * top-level `replay: false` in the pipeline yaml
        #   * `AGLAIA_NO_REPLAY=1` env var (debug)
        # The step is counted in the denominator so the progress bar's
        # 0/N matches the actual node count.
        import os as _os_replay
        replay_on = bool(pdef.get("replay", True)) if pdef else True
        if (self.pipeline_steps and replay_on
                and not _os_replay.environ.get("AGLAIA_NO_REPLAY")):
            self.pipeline_steps.append(f"{len(self.pipeline_steps) + 1:02d}_replay")
            self.pipeline_proc_names.append("Replay")

        self.pipeline_length = len(self.pipeline_steps)
        # Per-step parameter descriptions for the pipeline view (essential
        # one-liner under the row, full on click). Keyed by processor name.
        self.pipeline_descriptions = pipeline_step_descriptions(pdef)

        self.setWindowTitle(self.tr("Aglaïa · {name}").format(name=Path(db_path).stem))
        self.showMaximized()
        # Clamp window to screen geometry; growing child minimum sizes can
        # otherwise un-maximise the window mid-session. Reapplied on show/move.
        QTimer.singleShot(0, self._clamp_to_screen)
        
        self.args = args
        # Snapshot the theme value this session was launched with so
        # _apply_settings_changes can spot a user-driven change without
        # re-applying the theme live (every inline f-string baked at
        # widget-construction time would mismatch).
        try:
            from lib.app_data import db as _cfg
            with _cfg.session() as _conn:
                self._session_theme = str(
                    _cfg.get(_conn, _cfg.KEY_THEME, "system") or "system"
                )
        except Exception:
            self._session_theme = "system"
        self.processing_queue = processing_queue
        self.log_queue = log_queue
        self.slug_name = slug_name
        self.current_idx = start_idx

        # M0 DB context
        self.db_path = db_path
        self.pipeline_version_id = pipeline_version_id
        self.source = source
        self.pipeline_yaml_path = pipeline_yaml_path
        self._apply_pipeline_callback = apply_pipeline_callback
        self._force_reprocess_callback = force_reprocess_callback
        self._reprocess_snaps_callback = reprocess_scans_callback
        self._reprocess_branch_callback = reprocess_branch_callback
        self._stop_pipeline_callback = stop_pipeline_callback
        # Maps scan_id -> ScanItemWidget
        self.scan_widgets_by_scan: dict[int, ScanItemWidget] = {}
        # One connection per session — widgets call it as a function.
        self.thumb_loader = ThumbLoader(self.db_path)
        
        # Calibration state
        self.calibration = load_calibration()
        # Manual DPI override (set by clicking the DPI readout). Kept apart
        # from `self.calibration` on purpose: fabricating an identity-matrix
        # calibration just to carry a DPI would feed a bogus camera_matrix to
        # PageDewarper (see the identity-K focal-underflow leak). When set, it
        # overrides the uncalibrated input_dpi default; a real calibration
        # takes precedence over it.
        self._manual_dpi: Optional[float] = None
        cal_cfg = self.args.config.get("calibration", {})
        self.cal_target_count = cal_cfg.get("calnum", 10)
        self.calibrator = Calibrator(
            board_size=(cal_cfg.get("board_cols_inner", 5), cal_cfg.get("board_rows_inner", 8)),
            square_size_mm=cal_cfg.get("square_size_mm", 30)
        )
        self.path_config = self.args.config.get("paths", {})
        self.output_dir_name = self.path_config.get("output_dir", "XX_OUTPUT")
        self.is_calibrating = False
        self.history: list[int] = []  # scan_ids in capture order (for undo)

        # Card-grid zoom band. `_global_zoom` tracks min observed fit_zoom
        # across all cards' final-step thumbs. Cards above (1+tol)*global
        # are clamped down to that upper bound. A new fit_zoom below
        # global*(1-tol) becomes the new global and triggers broadcast.
        display_cfg = self.args.config.get("display", {})
        self._card_max_width_px = int(display_cfg.get("card_max_width_px", 260))
        self._zoom_tolerance = float(display_cfg.get("zoom_tolerance", 0.2))
        self._global_zoom: Optional[float] = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # Outer vertical: horizontal body + bottom status bar.
        outer_layout = QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self._outer_layout = outer_layout
        
        left_panel = QWidget()
        # Cap the side-panel width so the right-hand tab area dominates.
        # Matches the visual weight of non-capture modes where the left
        # column is narrow (no video preview competing for space).
        left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_panel.setLayout(left_layout)
        self._left_panel = left_panel
        # Top-left collapse button — same flat icon style as the pipeline
        # editor button. Click hides the panel; an overlaid "open" button
        # at the scans tab edge brings it back.
        from lib.gui.theme import lucide as _lucide_icn
        self._left_panel_close_btn = QPushButton()
        self._left_panel_close_btn.setIcon(_lucide_icn("panel-left-close",
                                                        color=COLOR_FONT_MUTED,
                                                        size=14))
        self._left_panel_close_btn.setFixedSize(26, 26)
        self._left_panel_close_btn.setToolTip(self.tr("Hide side panel"))
        self._left_panel_close_btn.setStyleSheet(
            f"QPushButton{{background:{COLOR_BG_SURFACE_ALT}; "
            f"border:1px solid {COLOR_OUTLINE_GHOST}; border-radius:5px; "
            "padding:0; margin:0;}"
            f"QPushButton:hover{{background:{COLOR_BG_OVERLAY_HOVER};}}"
        )
        self._left_panel_close_btn.clicked.connect(
            lambda: self._toggle_left_panel(False)
        )
        # Mount in a top-aligned row so it lives at the corner.
        _topbar = QHBoxLayout()
        _topbar.setContentsMargins(0, 0, 0, 0)
        _topbar.setSpacing(0)
        _topbar.addWidget(self._left_panel_close_btn, 0,
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        _topbar.addStretch(1)
        left_layout.addLayout(_topbar)

        is_capture = (self.source == "capture")

        # The live webcam preview now lives in the sidebar Capture tab for
        # every mode (compact preview slot), so there's no left-side video
        # pane any more. Hide the panel + its close-toggle entirely; the
        # right-side sidebar owns every control and the status bar handles
        # progress / log / RSS.
        self.video_label = None
        left_panel.setMaximumWidth(0)
        left_panel.setMinimumWidth(0)
        left_panel.hide()
        self._left_panel_close_btn.hide()
        
        # Modern Styled Buttons
        button_style = f"""
            QPushButton {{
                background-color: {COLOR_BG_BUTTON};
                color: {COLOR_FONT_ON_BUTTON};
                border: 1px solid {COLOR_OUTLINE_BUTTON};
                border-radius: 6px;
                padding: 6px 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BG_BUTTON_HOVER};
                border: 1px solid {COLOR_OUTLINE_STRONG};
            }}
            QPushButton:pressed {{
                background-color: {COLOR_BG_BUTTON_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_BG};
                color: {COLOR_FONT_DISABLED};
            }}
        """

        # Transform table — used by `_apply_selected_transform` to map
        # combo index → operation. Always defined so non-capture mode
        # can still reference it without an `is_capture` guard.
        self._transform_items = [
            (None,                  self.tr("— (no change)"), None),
            ("rotate-ccw-square",   self.tr("Rotate -90°"),   -90),
            ("flip-horizontal-2",   self.tr("Mirror"),        "mirror"),
            ("flip-vertical-2",     self.tr("Flip"),          "flip"),
            ("rotate-cw-square",    self.tr("Rotate 90°"),    90),
            ("refresh-cw",          self.tr("Rotate 180°"),   180),
        ]

        # ── Sidebar (replaces every old per-feature group above) ────
        # VS Code-style activity bar + per-tab content stack. Hosts
        # capture controls, import, pipeline, OCR, export — and the
        # tip + settings buttons at the bottom of the bar.
        self._build_sidebar(is_capture)

        # Freehand SIFT-tracked capture — fields exist even when the
        # feature is off so `_freehand_overlay` and friends don't have
        # to guard each attribute access.
        self._sift_tracker = None
        self._sift_gate = None
        self._sift_timer: Optional[QTimer] = None
        self._sift_last_pts = None
        self._sift_last_fraction = 0.0
        self._sift_roi = None
        self._sift_last_quad = None
        self._sift_flash_until: float = 0.0
        import threading as _t
        self._sift_state_lock = _t.Lock()

        self._ocr_worker: Optional[OcrWorker] = None

        # Now that the checkbox exists, set its initial visibility from
        # the pipeline yaml.
        self.refresh_norm_widths_visibility()

        # Status + voice + loading labels are no longer parented to the
        # left panel: in capture mode they're moved to the bottom status
        # bar (see `use_capture_widgets` call below). They're still
        # constructed here so the rest of MainWindow can `.setText()` on
        # them without conditional checks.
        self.status_label = _ClickableLabel(
            self.tr("Ready. Say 'Scan' or Press Space."),
            on_click=lambda: self._open_log_tab(),
        )
        self.status_label.setStyleSheet(
            f"QLabel {{ color: {COLOR_FONT_PRIMARY}; padding: 0 8px; }}"
            f"QLabel:hover {{ color: {COLOR_FONT_INVERSE}; text-decoration: underline; }}"
        )

        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet(
            f"color: {COLOR_PRIMARY}; font-style: italic; padding: 0 8px;"
        )

        # Capture mode shows the spinner in the side panel until the status
        # bar lights up. Non-capture modes have no left panel, so the
        # labels stay un-parented (still constructed because other code
        # calls `setText()` on them); the bottom status bar handles
        # progress + log itself.
        if is_capture:
            left_layout.addWidget(self.loading_label)

        self.workers_started = 0
        # For Integrated Chain, total workers is just the count of integrated workers
        self.total_workers = args.workers
        
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self.update_spinner)
        self.spinner_timer.start(100)
        
        # ── tabbed area (spans full window width) ──────────────────────
        # Tab 0 = "Scans" (pinned). It owns the left-side controls
        # panel + the scans flow so every other tab (debug, settings,
        # pipeline editor, calibration, …) gets the full window width
        # for its content.
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close_requested)

        scans_tab = QWidget()
        scans_h = QHBoxLayout(scans_tab)
        scans_h.setContentsMargins(0, 0, 0, 0)
        scans_h.setSpacing(0)
        scans_h.addWidget(left_panel)
        self._scans_tab_widget = scans_tab
        # Overlay "open" button — child of the tab, absolutely positioned
        # near where the close button would sit on the collapsed side.
        self._left_panel_open_btn = QPushButton(scans_tab)
        self._left_panel_open_btn.setIcon(_lucide_icn("panel-left-open",
                                                       color=COLOR_FONT_MUTED,
                                                       size=14))
        self._left_panel_open_btn.setFixedSize(26, 26)
        self._left_panel_open_btn.setToolTip(self.tr("Show side panel"))
        self._left_panel_open_btn.setStyleSheet(
            f"QPushButton{{background:{COLOR_BG_SURFACE_ALT}; "
            f"border:1px solid {COLOR_OUTLINE_GHOST}; border-radius:5px; "
            "padding:0; margin:0;}"
            f"QPushButton:hover{{background:{COLOR_BG_OVERLAY_HOVER};}}"
        )
        self._left_panel_open_btn.clicked.connect(
            lambda: self._toggle_left_panel(True)
        )
        self._left_panel_open_btn.move(6, 6)
        self._left_panel_open_btn.hide()
        self._left_panel_open_btn.raise_()
        if not is_capture:
            # No left pane to re-open outside capture — disable the
            # floating affordance so the corner stays clean.
            self._left_panel_open_btn.setEnabled(False)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scans_scroll = scroll
        from lib.gui.FlowLayout import FlowLayout, FlowContentWidget
        self.scroll_content = FlowContentWidget()
        self.scroll_content.card_dropped.connect(self._on_card_dropped)
        self.scroll_layout = FlowLayout(self.scroll_content,
                                        margin=8, h_spacing=10, v_spacing=10)
        scroll.setWidget(self.scroll_content)

        # View-mode stack. Grid is the persistent base; the table and gallery
        # are built lazily on first selection and destroyed when deselected
        # (see set_view_mode / _ensure_table / _ensure_gallery / _destroy_*).
        # On a 300+ scan project an always-resident table is a lot of widgets
        # and pixmaps to hold for a view the user may never open.
        self._scans_stack = QStackedWidget()
        self._scans_stack.addWidget(scroll)                  # grid (persistent)
        self._grid_widget = scroll
        self._scans_table = None
        self._scans_gallery = None
        # Single-source-of-truth broadcast: any view's selection/visibility
        # change writes to DB then re-emits; whichever view is live resyncs.
        self.branch_visibility_changed.connect(self._on_branch_visibility_changed)
        self.step_disabled_changed.connect(self._on_step_disabled_changed)
        # OCR state changes fan out to badges + bottom-bar OCR frame.
        self.ocr_state_changed.connect(self._on_ocr_state_changed)
        scans_h.addWidget(self._scans_stack, 1)
        # Sidebar (right edge): ActivityBar + content stack.
        scans_h.addWidget(self.sidebar)

        from lib.gui.theme import lucide as _lucide_tab
        self.tabs.addTab(scans_tab, _lucide_tab("layout-grid", size=14), self.tr("Scans"))
        self._install_thumb_size_slider()
        # Hide the close button on tab 0 — it must stay open.
        for side in (QTabBar.ButtonPosition.RightSide,
                     QTabBar.ButtonPosition.LeftSide):
            btn = self.tabs.tabBar().tabButton(0, side)
            if btn is not None:
                btn.deleteLater()
                self.tabs.tabBar().setTabButton(0, side, None)

        outer_layout.addWidget(self.tabs, 1)

        # Per-leaf-node debug viewer cache. Re-opening the same node
        # focuses the existing tab instead of building a second one.
        self._debug_tabs: dict[int, QWidget] = {}
        # Singleton tabs — re-opening focuses the existing tab.
        self._pipeline_editor_tab: Optional[QWidget] = None
        self._dpi_dialog: Optional[QDialog] = None
        self._settings_tab: Optional[QWidget] = None
        self._freehand_tab: Optional[QWidget] = None

        # Toast overlay for "the action just succeeded but the only
        # other visual cue is that the tab closed". Parented on the
        # central widget so it floats above tab content.
        from lib.gui.Toast import Toast as _Toast
        self._toast = _Toast(main_widget)

        # Bottom status bar: per-worker RSS bars + pipeline % + last log line.
        from lib.gui.StatusBarWidget import StatusBarWidget, LogViewerWidget
        self._LogViewerWidget = LogViewerWidget  # captured for tab opening
        self.status_bar_widget = StatusBarWidget()
        self.status_bar_widget.log_clicked.connect(self._open_log_tab)
        self.status_bar_widget.settings_clicked.connect(self._on_settings_clicked)
        self.status_bar_widget.stop_clicked.connect(self._on_stop_pipeline_clicked)
        self.status_bar_widget.stop_ocr_clicked.connect(self._on_stop_ocr_clicked)
        self._outer_layout.addWidget(self.status_bar_widget)
        # Cache for the open Log tab (so a second click focuses instead
        # of building a duplicate viewer).
        self._log_tab: QWidget | None = None

        # Capture mode: keep the bottom-bar progress + stop, show the capture
        # status message. Voice control lives entirely in the capture sidebar
        # (the split Voice button), so there's no bottom-bar voice widget.
        if is_capture:
            self.status_bar_widget.use_capture_widgets(self.status_label)

        if is_capture:
            self._active_cam_id = getattr(args, "camera_id", 0)
            self.webcam_thread = WebcamThread(args.camera_id)
            if hasattr(args, 'transform'):
                self.webcam_thread.set_transform(args.transform)
            self.webcam_thread.change_pixmap_signal.connect(self.update_image)
            # Capture-flash overlay installed for the camera's lifetime
            # so voice / keyboard captures get the same visual cue SIFT
            # already had. SIFT still wraps additional draws (the
            # tracked-quad polyline) on top.
            self.webcam_thread.set_overlay_fn(self._freehand_overlay)
            self.webcam_thread.start()
        else:
            self.webcam_thread = None

        self.monitor_thread = ProcessMonitor(log_queue)
        # Image events arrive batched (see ProcessMonitor) to keep the GUI
        # responsive during a large reprocess.
        self.monitor_thread.image_events_batch_signal.connect(self.on_image_events_batch)
        self.monitor_thread.snap_imported_signal.connect(self.on_scan_imported)
        self.monitor_thread.worker_started_signal.connect(self.on_worker_started)
        # Status bar wiring
        self.monitor_thread.rss_signal.connect(
            self.status_bar_widget.rss.update_values
        )
        self.monitor_thread.log_signal.connect(self._on_log_line)
        # Live per-step pipeline timing → sidebar Pipeline tab.
        # First sample swaps the idle step list out for the live view.
        self.monitor_thread.timing_signal.connect(
            self._pipeline_tab.record_timing
        )
        # Seed the idle step list with the loaded pipeline's steps so
        # the tab has something to show before any sample lands.
        try:
            self._pipeline_tab.set_steps(self.pipeline_proc_names,
                                          self.pipeline_descriptions)
        except Exception:
            pass
        self.monitor_thread.snap_imported_signal.connect(
            self._on_status_scan_imported
        )
        self.monitor_thread.branch_ready_signal.connect(
            self._on_status_branch_ready
        )
        self.monitor_thread.start()

        # Voice control. Auto-start when the user passed --voice-control;
        # otherwise stays inactive but can be flipped on by clicking the
        # voice label in the status bar. _start_voice / _stop_voice live
        # on the class so the click handler can reuse them.
        self.voice_thread = None
        self.last_voice_cmd_time = 0
        if is_capture and args.voice_control:
            self._start_voice()

        # Load existing scans from project DB
        QTimer.singleShot(100, self.load_existing_scans)

        # Platform-aware keyboard shortcuts. `StandardKey.Quit` maps to
        # ⌘Q on macOS and Ctrl+Q elsewhere; `StandardKey.Close` to ⌘W /
        # Ctrl+W. Letter-only bindings (Q for quit, etc.) fired even when
        # an input field had focus, which is too easy to trigger by
        # accident. Modifier-gated shortcuts fix that.
        self._install_global_shortcuts()

    def _install_global_shortcuts(self) -> None:
        """⌘Q quit · ⌘W close project → launcher · ⌘N new · ⌘O open ·
        ⌘, settings. Built on every platform so Help (docs / bug report /
        diagnostics / about) is always reachable: macOS shows a native menu
        bar (Qt relocates Settings/Quit/About via their MenuRoles), Linux /
        Windows show an in-window menu bar. StandardKey resolves the platform
        modifier automatically."""
        self._build_menu_bar()

    def _build_menu_bar(self) -> None:
        """Native macOS menu bar. Qt relocates the PreferencesRole +
        QuitRole actions into the application menu; New/Open/Close sit in
        a standard File menu."""
        from PySide6.QtGui import QAction
        mb = self.menuBar()

        def _act(text, seq, fn, role=None):
            a = QAction(text, self)
            if seq is not None:
                a.setShortcut(QKeySequence(seq))
            if role is not None:
                a.setMenuRole(role)
            a.triggered.connect(fn)
            return a

        app_menu = mb.addMenu("Aglaïa")
        app_menu.addAction(_act(
            self.tr("Settings…"), QKeySequence.StandardKey.Preferences,
            lambda: self._on_settings_clicked(),
            QAction.MenuRole.PreferencesRole))
        app_menu.addAction(_act(
            self.tr("Quit Aglaïa"), QKeySequence.StandardKey.Quit,
            self.close, QAction.MenuRole.QuitRole))

        file_menu = mb.addMenu(self.tr("File"))
        # Disabled cue where a Save item would normally sit: Aglaïa
        # projects persist to their SQLite DB continuously, so there is no
        # manual save. Greyed-out + no shortcut so it reads as a hint.
        _autosave = _act(self.tr("No save : project autosaves"), None,
                         lambda: None)
        _autosave.setEnabled(False)
        file_menu.addAction(_autosave)
        file_menu.addSeparator()
        file_menu.addAction(_act(
            self.tr("New Project…"), QKeySequence.StandardKey.New,
            lambda: self._confirm_then_restart("new")))
        file_menu.addAction(_act(
            self.tr("Open Project…"), QKeySequence.StandardKey.Open,
            lambda: self._confirm_then_restart("open")))
        file_menu.addSeparator()
        file_menu.addAction(_act(
            self.tr("Slim-down current project…"), None,
            lambda: self._on_slim_down_in_place()))
        file_menu.addSeparator()
        # The square-x icon / this no-shortcut item returns the whole
        # project to the launcher (⌘W is Close Tab, in the View menu).
        file_menu.addAction(_act(
            self.tr("Close Project"), None,
            lambda: self._request_restart("landing")))

        # View menu — downloader, close tab, and the scans-view selector.
        from PySide6.QtGui import QActionGroup
        view_menu = mb.addMenu(self.tr("View"))
        view_menu.addAction(_act(
            self.tr("Show Downloader"), None,
            lambda: self._open_model_downloader()))
        view_menu.addAction(_act(
            self.tr("Close Tab"), QKeySequence.StandardKey.Close,
            self._close_current_tab))
        view_menu.addSeparator()
        self._view_mode_actions = {}
        _vm_group = QActionGroup(self)
        _vm_group.setExclusive(True)
        for _label, _mode in ((self.tr("Table"), "list"),
                              (self.tr("Grid"), "grid"),
                              (self.tr("Gallery"), "gallery")):
            a = _act(_label, None,
                     lambda checked=False, m=_mode: self.set_view_mode(m))
            a.setCheckable(True)
            a.setChecked(getattr(self, "_view_mode", "grid") == _mode)
            _vm_group.addAction(a)
            view_menu.addAction(a)
            self._view_mode_actions[_mode] = a

        # Help menu — docs link, diagnostics, bug report, about.
        help_menu = mb.addMenu(self.tr("Help"))
        help_menu.addAction(_act(
            self.tr("Aglaïa Documentation"), None, self._open_docs))
        help_menu.addAction(_act(
            self.tr("Diagnostics…"), None, lambda: self._on_diagnostics()))
        help_menu.addAction(_act(
            self.tr("Report a Bug…"), None, lambda: self._on_report_bug()))
        help_menu.addSeparator()
        help_menu.addAction(_act(
            self.tr("About Aglaïa"), None, lambda: self._open_about(),
            QAction.MenuRole.AboutRole))

    def _open_about(self) -> None:
        from lib.gui.AboutDialog import show_about
        show_about(self)

    def _open_docs(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://aglaia.bibli.cc/docs"))

    # ── launcher round-trip ────────────────────────────────────────────

    def _request_restart(self, action: str) -> None:
        """Close this project window and hand back to the launcher. The
        ``action`` ("landing" / "new" / "open") pre-navigates the
        relaunched StartupWindow; the main() loop reads it off the app
        property once this window has closed (which stops the chain)."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.setProperty("aglaia_restart", action)
        self.close()

    def _confirm_then_restart(self, action: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, self.tr("Close current project?"),
            self.tr("Close the current project and return to the launcher?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._request_restart(action)

    def _on_report_bug(self) -> None:
        from lib.gui.BugReportDialog import open_bug_report
        open_bug_report(self)

    def _on_diagnostics(self) -> None:
        from lib.gui.BugReportDialog import open_diagnostics
        open_diagnostics(self)

    def _close_current_tab(self) -> None:
        """⌘W / Ctrl+W handler. Closes the active right-side tab unless
        it's the pinned Scans tab (idx 0). Re-uses the same teardown
        path as the close button on the tab itself."""
        idx = self.tabs.currentIndex()
        if idx <= 0:
            return
        self._on_tab_close_requested(idx)

    def load_existing_scans(self):
        """Rebuild widgets from the project SQLite DB (M0 source of truth)."""
        self.status_label.setText(self.tr("Loading existing scans..."))
        n_spawned = 0
        completed_scan_ids: list[int] = []
        with db_session(self.db_path) as conn:
            scans = ScanRepo(conn).list_active(newest_first=False)
            node_repo = NodeRepo(conn)
            max_idx = 0
            for scan_row in scans:
                scan_id = scan_row["id"]
                idx = int(scan_row["idx"])
                max_idx = max(max_idx, idx)
                root_node_id = scan_row["root_node_id"]
                if root_node_id is None:
                    # Orphan scan (no root node persisted) — skip widget
                    # AND don't count it toward the progress denominator,
                    # otherwise the bar reads "N/N+1 (…%)" forever.
                    continue
                root_node = node_repo.get(root_node_id)
                if root_node is None:
                    continue
                widget = self._spawn_widget(scan_id, idx, root_node["id"], root_node["image_id"],
                                            root_node["filestem"])
                n_spawned += 1
                # Replay non-root nodes as restored events
                max_step = 0
                for n in node_repo.by_scan(scan_id):
                    if n["id"] == root_node_id:
                        continue
                    max_step = max(max_step, int(n["step_idx"] or 0))
                    widget.restore_node(
                        node_id=n["id"],
                        parent_node_id=n["parent_id"],
                        image_id=n["image_id"],
                        step_name=n["step_name"] or "",
                        filestem=n["filestem"],
                        meta=None,
                    )
                # Restore per-page hidden state from `branches.trashed_at`
                # (a branch's nodes share one filestem, so the chosen
                # node's filestem is the page's card stem).
                for b in conn.execute(
                    "SELECT chosen_node_id FROM branches "
                    "WHERE scan_id = ? AND trashed_at IS NOT NULL",
                    (scan_id,),
                ).fetchall():
                    bn = node_repo.get(b["chosen_node_id"])
                    if bn is not None:
                        widget.set_stem_trashed(bn["filestem"], True)
                # If the scan already reached the last pipeline step (or
                # the Replay/MarginSetter that follows it), it's done —
                # don't keep the spinner spinning for completed scans.
                if max_step >= len(self.pipeline_steps):
                    widget.set_processing(False)
                    completed_scan_ids.append(scan_id)

        if max_idx >= self.current_idx:
            self.current_idx = max_idx + 1
        self.status_label.setText(self.tr("Ready. Scans loaded from project DB."))
        # Seed the pipeline progress bar with the count of WIDGETS actually
        # spawned, not raw `len(scans)` — orphan rows skipped above never
        # emit branch_ready, so counting them inflates the denominator
        # forever ("10/11 (91%)" with one mystery missing scan).
        self.status_bar_widget.progress.bump_imported_if_below(n_spawned)
        # For scans that already reached the last pipeline step on disk,
        # mark them done now: the chain won't reprocess them, so
        # branch_ready won't fire. Without this the bar reads 0/N for
        # an already-finished project until a fresh capture/import hits.
        for sid in completed_scan_ids:
            self.status_bar_widget.progress.mark_done(sid)
        # Initial OCR-state pull so already-OCR'd scans light up their badges.
        self._refresh_ocr_ui()
        self._update_ocr_frame_state()
        # If the persisted view mode is `list` or `gallery`, those panes
        # were instantiated before `load_existing_scans` ran and therefore
        # showed empty. Repopulate now that the scan widgets are alive.
        self._refresh_alt_views_if_visible()

    def _spawn_widget(self, scan_id: int, idx: int, raw_node_id: int, raw_image_id: int,
                      filestem: str) -> ScanItemWidget:
        # Idempotent: race between `load_existing_scans` (T=100ms after
        # window init) and `on_scan_imported` events fired by the catchup
        # thread can otherwise spawn the same scan twice → duplicate row
        # in the scans column. Reuse the existing widget instead.
        existing = self.scan_widgets_by_scan.get(scan_id)
        if existing is not None:
            return existing
        # Look up the raw image's source DPI so the scan title can show
        # it — a quick visual consistency check that the importer / PDF
        # extractor handed in what we expected (e.g. 300 dpi, not 72).
        raw_dpi = None
        try:
            with db_session(self.db_path) as conn:
                row = ImageRepo(conn).get(raw_image_id)
                if row is not None:
                    raw_dpi = float(row["dpi"]) if row["dpi"] is not None else None
        except Exception:
            pass

        widget = ScanItemWidget(
            scan_id=scan_id,
            idx=idx,
            raw_node_id=raw_node_id,
            raw_image_id=raw_image_id,
            raw_filestem=filestem,
            pipeline_steps=self.pipeline_steps,
            thumb_loader=self.thumb_loader,
            raw_dpi=raw_dpi,
            max_card_width_px=self._card_max_width_px,
            zoom_tolerance=self._zoom_tolerance,
            # Parent the card straight to its eventual host so it never
            # exists as a parentless QWidget (which macOS would briefly
            # render as a top-level window with native chrome).
            parent=self.scroll_content,
        )
        widget.delete_requested.connect(self.delete_scan)
        widget.debug_requested.connect(self._open_debug_viewer)
        widget.final_zoom_observed.connect(self._on_card_final_zoom)
        # Per-page disable: round stage-toggle → flip override + rerun.
        # State for the toggle + band comes from a per-scan provider.
        widget.step_states_provider = self.cell_disable_states
        widget.step_toggle_requested.connect(self.toggle_step_disabled)
        # Visibility (eye) still writes branches.trashed_at so gallery +
        # table see the same hide state. (selection_changed is vestigial.)
        widget.visibility_changed.connect(self._on_card_visibility_changed)
        if self._global_zoom is not None:
            widget.set_global_zoom(self._global_zoom)
        # Mark in-flight by default — every freshly spawned widget is
        # for either a new capture / import or a reloaded scan whose
        # pipeline is about to run. `branch_ready_signal` clears it.
        widget.set_processing(True)
        # Append: oldest scans stay at the top, newest at the bottom.
        self.scroll_layout.insertWidget(self.scroll_layout.count(), widget)
        self.scan_widgets_by_scan[scan_id] = widget
        self.history.append(scan_id)
        # FlowLayout needs to lay out the new row before the scrollbar
        # range updates — defer the scroll-to-bottom by one event loop.
        QTimer.singleShot(0, self._scroll_scans_to_bottom)
        return widget

    def _scroll_scans_to_bottom(self):
        bar = self._scans_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _open_debug_viewer(self, node_id: int, label: str):
        """Open a debug viewer for `node_id` as a new closable tab.

        If the same leaf already has a tab, switch to it rather than
        spawning a duplicate — the debug render is ~250 ms and the user
        usually clicks the same node multiple times while iterating.
        """
        existing = self._debug_tabs.get(node_id)
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return

        from lib.gui.DebugViewerTab import DebugViewerWidget
        from lib.gui.theme import lucide as _lucide_tab
        viewer = DebugViewerWidget(self.db_path, node_id, title_hint=label)
        tab_label = (self.tr("Inspect · {label}").format(label=label) if label
                     else self.tr("Inspect · node {nid}").format(nid=node_id))
        idx = self.tabs.addTab(viewer, _lucide_tab("image", size=14), tab_label)
        self.tabs.setTabToolTip(idx, tab_label)
        self.tabs.setCurrentIndex(idx)
        self._debug_tabs[node_id] = viewer

    def _on_tab_close_requested(self, idx: int):
        """Close a debug / log / pipeline-editor tab. Tab 0 (Scans) has
        no close button so this is never called for it."""
        w = self.tabs.widget(idx)
        if w is None or idx == 0:
            return
        # Drop from the leaf→tab cache so the next click on that leaf
        # opens a fresh tab.
        for node_id, tab in list(self._debug_tabs.items()):
            if tab is w:
                del self._debug_tabs[node_id]
                break
        if self._log_tab is w:
            self._log_tab = None
        if getattr(self, "_pipeline_editor_tab", None) is w:
            try:
                w.stop_preview_worker()
            except Exception:
                pass
            self._pipeline_editor_tab = None
        if getattr(self, "_settings_tab", None) is w:
            self._settings_tab = None
        if getattr(self, "_fix_dpi_tab", None) is w:
            self._fix_dpi_tab = None
        self.tabs.removeTab(idx)
        w.deleteLater()

    def toast(self, text: str, duration_ms: int | None = None) -> None:
        """Show a brief overlay confirmation. Used after invisible
        operations (settings saved, pipeline applied, export finished)
        so the user has a quick visual receipt that the action ran."""
        if getattr(self, "_toast", None) is not None:
            self._toast.show_message(text, duration_ms)

    def _reveal_in_finder(self, path: Path) -> None:
        """Reveal a just-written export in the OS file manager so the user
        can see where it landed. Cross-platform, best-effort."""
        from lib.gui.path_reveal import reveal_path
        reveal_path(path)

    def resizeEvent(self, ev):  # noqa: N802 — Qt API
        super().resizeEvent(ev)
        # Reposition the toast …
        if getattr(self, "_toast", None) is not None:
            self._toast.reposition()
        # … and keep the window within the active screen (oversized restore).
        # (Merged from a second resizeEvent that used to silently shadow this
        # one, dropping the toast reposition.)
        scr = self.screen() if hasattr(self, "screen") else None
        if scr is not None:
            avail = scr.availableGeometry()
            if self.width() > avail.width() or self.height() > avail.height():
                self.setGeometry(avail)

    def update_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        frame = self.spinner_frames[self.spinner_idx]
        progress = f"({self.workers_started}/{self.total_workers})"
        self.loading_label.setText(self.tr("{frame} Workers loading {progress}...").format(
            frame=frame, progress=progress,
        ))

    def on_worker_started(self):
        self.workers_started += 1
        if self.workers_started >= self.total_workers:
            self.loading_label.hide()
            self.spinner_timer.stop()
            self.status_label.setText(self.tr("Ready. All workers active."))

    def rotate_camera(self, delta=90):
        self.webcam_thread.rotate(delta)
        self.status_label.setText(self.tr("Rotated to {deg}°").format(deg=self.webcam_thread.rotation))

    def toggle_mirror(self):
        self.webcam_thread.toggle_mirror()
        state = self.tr("ON") if self.webcam_thread.mirror else self.tr("OFF")
        self.status_label.setText(self.tr("Mirror: {state}").format(state=state))

    def toggle_flip(self):
        self.webcam_thread.toggle_flip()
        state = self.tr("ON") if self.webcam_thread.flip else self.tr("OFF")
        self.status_label.setText(self.tr("Flip: {state}").format(state=state))

    def _apply_selected_transform(self):
        """Run the action picked in `transform_combo`, then scan back to
        the "—" default so the Apply button disables itself again."""
        idx = self.transform_combo.currentIndex()
        if idx <= 0:
            return
        _, _, value = self._transform_items[idx]
        if value == "mirror":
            self.toggle_mirror()
        elif value == "flip":
            self.toggle_flip()
        elif isinstance(value, int):
            self.rotate_camera(value)
        # Reset to default; the index change handler disables Apply.
        self.transform_combo.blockSignals(True)
        self.transform_combo.setCurrentIndex(0)
        self.transform_combo.blockSignals(False)
        self.btn_apply_transform.setEnabled(False)

    def _build_capture_placeholder(self) -> QWidget:
        """Capture-tab body used when the project was opened from disk
        (no webcam thread spun up at launch).

        Shows a camera picker + "Activate Capture" button. Clicking
        the button lazily builds a real ``CaptureTab`` and spins the
        ``WebcamThread`` against the chosen device — the live CaptureTab
        then replaces the placeholder inside a small QStackedWidget so
        the sidebar icon stays put and the user keeps their place in
        the project."""
        from lib.gui.CameraEnum import list_cameras

        host = QWidget()
        host_v = QVBoxLayout(host)
        host_v.setContentsMargins(0, 0, 0, 0)
        host_v.setSpacing(0)

        # Stack: 0 = picker, 1 = future live CaptureTab.
        stack = QStackedWidget()
        host_v.addWidget(stack)
        self._capture_stack = stack

        picker_wrap = QWidget()
        v = QVBoxLayout(picker_wrap)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        title = QLabel(self.tr("Capture"))
        title.setObjectName("SectionTitle")
        v.addWidget(title)

        cam_label = QLabel(self.tr("Camera"))
        cam_label.setObjectName("FieldLabel")
        v.addWidget(cam_label)

        cam_combo = QComboBox()
        cam_combo.setMinimumHeight(28)
        cams = list_cameras() or [(0, self.tr("Default camera"))]
        for cam_id, name in cams:
            cam_combo.addItem(self.tr("{name}  (id {cid})").format(name=name, cid=cam_id), int(cam_id))
        v.addWidget(cam_combo)
        self._capture_cam_combo = cam_combo

        btn = QPushButton(self.tr("Activate Capture"))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(36)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_BG_BUTTON}; color: {COLOR_FONT_ON_BUTTON}; "
            f"border: 1px solid {COLOR_OUTLINE_BUTTON}; border-radius: 6px; "
            "padding: 8px 14px; font-weight: 600; font-size: 13px; }"
            f"QPushButton:hover {{ background-color: {COLOR_BG_BUTTON_HOVER}; }}"
        )
        btn.clicked.connect(self._activate_capture_clicked)
        v.addWidget(btn)
        v.addStretch(1)

        stack.addWidget(picker_wrap)  # index 0
        return host

    def _activate_capture_clicked(self) -> None:
        """Slot for the placeholder's Activate button. Spins the
        ``WebcamThread`` against the picked camera, builds a real
        ``CaptureTab``, wires it the same way the launch-time
        ``is_capture`` branch does, then swaps the QStackedWidget so
        the icon now opens the live capture UI."""
        if getattr(self, "webcam_thread", None) is not None:
            return  # already activated
        cam_id = 0
        try:
            cam_id = int(self._capture_cam_combo.currentData() or 0)
        except Exception:
            cam_id = 0
        self._active_cam_id = cam_id
        try:
            self.webcam_thread = WebcamThread(cam_id)
            self.webcam_thread.change_pixmap_signal.connect(self.update_image)
            self.webcam_thread.set_overlay_fn(self._freehand_overlay)
            self.webcam_thread.start()
        except Exception as e:
            QMessageBox.critical(
                self, self.tr("Capture"),
                self.tr("Failed to start camera #{cid}: {err}").format(cid=cam_id, err=e),
            )
            self.webcam_thread = None
            return

        # Apply any launch-time transform the user passed on the CLI.
        try:
            if getattr(self.args, "transform", None):
                self.webcam_thread.set_transform(self.args.transform)
        except Exception:
            pass

        ct = self._make_live_capture_tab()
        # Swap the picker for the live tab.
        self._capture_stack.addWidget(ct)  # index 1
        self._capture_stack.setCurrentIndex(1)
        self._capture_tab = ct
        self.toast("Capture activated — webcam ready.")

    def _deactivate_capture_clicked(self) -> None:
        """Tear down the late-activated webcam session: stop the
        ``WebcamThread``, drop the live ``CaptureTab`` from the sidebar
        stack, and swap back to the picker so the user can pick a
        different device or simply free the camera."""
        wt = getattr(self, "webcam_thread", None)
        if wt is not None:
            try:
                wt.change_pixmap_signal.disconnect(self.update_image)
            except Exception:
                pass
            try:
                wt.stop()
            except Exception:
                pass
            self.webcam_thread = None
        ct = getattr(self, "_capture_tab", None)
        if ct is not None and hasattr(ct, "set_preview_pixmap"):
            # Cleanly toggle off any hands-free trigger before tearing
            # the widget down so the underlying worker stops too.
            try:
                if ct.btn_voice.isChecked():
                    ct.btn_voice.setChecked(False)
            except Exception:
                pass
            try:
                if ct.btn_freehand.isChecked():
                    ct.btn_freehand.setChecked(False)
            except Exception:
                pass
            try:
                self._capture_stack.removeWidget(ct)
                ct.deleteLater()
            except Exception:
                pass
        self._capture_tab = None
        # Drop legacy button aliases so reactivation rebuilds them.
        self.btn_full_calibrate = None
        self.btn_dpi_calibrate = None
        self.btn_freehand = None
        if getattr(self, "_capture_stack", None) is not None:
            self._capture_stack.setCurrentIndex(0)
        self.toast("Capture deactivated — camera released.")

    def _build_sidebar(self, is_capture: bool) -> None:
        """Construct the right-side activity bar + tab content stack.

        Builds five tabs (Capture, Import, Pipeline, OCR, Export) and
        the two bottom items (Tip dev, Settings). Pipes the
        ProcessMonitor's ``timing_signal`` into the PipelineTab's live
        timing view.
        """
        from lib.gui.theme import icon as _icon

        # ── Build tabs ─────────────────────────────────────────────
        # Capture tab is always the picker/stack host. In capture mode we
        # immediately build + show the live tab inside its stack (at the end
        # of this method) so the session opens straight into the preview; in
        # open-from-disk mode it stays on the picker until the user clicks
        # "Activate Capture".
        self._capture_tab = self._build_capture_placeholder()
        self._import_tab = ImportTab()
        self._pipeline_tab = PipelineTab()
        self._ocr_tab = OcrTab()
        self._export_tab = ExportTab()

        # Capture-tab live wiring is handled by `_make_live_capture_tab`,
        # called either here (capture mode, at the end of this method) or
        # later from `_activate_capture_clicked` (open-from-disk).

        # ── Pipeline-tab wiring ────────────────────────────────────
        self._pipeline_tab.btn_edit.clicked.connect(self.open_pipeline_editor)
        self._pipeline_tab.btn_force.clicked.connect(self._on_force_rerun_clicked)
        self._pipeline_tab.btn_fix_dpi.clicked.connect(self.open_fix_dpi_tab)

        # ── OCR-tab seeding + wiring ───────────────────────────────
        try:
            from lib.app_data import db as _cfg
            with _cfg.session() as _conn:
                _defaults = _cfg.get(_conn, _cfg.KEY_OCR_DEFAULTS, {}) or {}
            _langs = list(_defaults.get("languages") or [])
            if _langs:
                self._ocr_tab.lang_input.set_tags(_langs)
            _engine = _defaults.get("engine")
            # Only restore a persisted engine if its card exists AND is
            # enabled on this platform; otherwise keep the gated default
            # the OCR tab already selected (e.g. a stored apple_docs on a
            # pre-26 Mac falls back to the legacy card).
            if _engine and _engine in self._ocr_tab.engine_group.keys():
                _card = self._ocr_tab.engine_group._cards.get(_engine)
                if _card is not None and _card.frame.isEnabled():
                    self._ocr_tab.engine_group.set_current_key(_engine)
        except Exception:
            pass
        self._ocr_tab.run_requested.connect(self._on_ocr_run_requested)
        # Live-OCR toggle — sidebar fires a bool; we install / tear down
        # the auto-OCR scheduler accordingly.
        self._ocr_tab.live_ocr_toggled.connect(self._on_live_ocr_toggled)
        # Engine switch — mark all done runs of the OLD engine as
        # stale so badges + default OCR run mode pick them up.
        self._ocr_tab.engine_changed.connect(self._on_ocr_engine_changed)

        # ── Export-tab wiring ──────────────────────────────────────
        self._export_tab.btn_export.clicked.connect(self._on_export_clicked)
        self._export_tab.chk_norm_widths.stateChanged.connect(
            self._on_norm_widths_toggled
        )

        # ── Import-tab wiring ──────────────────────────────────────
        self._import_tab.import_requested.connect(self._on_sidebar_import_requested)

        # ── Assemble panel ────────────────────────────────────────
        self.sidebar = SidebarPanel(self)
        # Capture tab is ALWAYS visible — in non-capture project mode
        # the tab body is a placeholder pointing the user back to the
        # startup window. Hiding the icon entirely made the sidebar
        # look incomplete when reopening an existing project.
        self.sidebar.add_tab("capture", self._capture_tab,
                              icon_name="camera", tooltip=self.tr("Capture"))
        self.sidebar.add_tab("import", self._import_tab,
                              icon_name="download", tooltip=self.tr("Import"))
        self.sidebar.add_tab("pipeline", self._pipeline_tab,
                              icon_name="sliders-horizontal",
                              tooltip=self.tr("Pipeline"), scrollable=False)
        self.sidebar.add_tab("ocr", self._ocr_tab,
                              icon_name="scan-text", tooltip=self.tr("OCR"))
        self.sidebar.add_tab("export", self._export_tab,
                              icon_name="export", tooltip=self.tr("Export"))
        # Bottom toolbox, top-to-bottom: close project, report bug, settings.
        self.sidebar.add_bottom_action(
            "close_project", "square-x", self.tr("Close project (⌘W)"),
            lambda: self._request_restart("landing"),
        )
        self.sidebar.add_bottom_action(
            "report_bug", "bug", self.tr("Report a bug"),
            lambda: self._on_report_bug(),
        )
        self.sidebar.add_bottom_action(
            "settings", "settings", self.tr("Settings"),
            lambda: self._on_settings_clicked(),
        )
        # Tip heart sits centred in the mid block (see ActivityBar).
        self.sidebar.add_tip_button(lambda: self._open_tip_url())
        # Restore last-session tab + collapsed state if any.
        stored_tab: Optional[str] = None
        stored_collapsed = False
        try:
            from lib.app_data import db as _cfg
            with _cfg.session() as _conn:
                stored_tab = _cfg.get(_conn, _cfg.KEY_SIDEBAR_TAB, None)
                stored_collapsed = bool(
                    _cfg.get(_conn, _cfg.KEY_SIDEBAR_COLLAPSED, False)
                )
        except Exception:
            pass
        # Fall back to capture (if available) → pipeline.
        default_tab = "capture" if is_capture else "pipeline"
        active = stored_tab if stored_tab in self.sidebar._tabs else default_tab
        self.sidebar.set_active(active)
        if stored_collapsed:
            self.sidebar.set_collapsed(True)
        # Persist state on every change.
        self.sidebar.state_changed.connect(self._persist_sidebar_state)

        # ── Legacy attribute aliases ──────────────────────────────
        # Old call sites keep working without per-line rewrites.
        self.ocr_frame = self._ocr_tab
        self.btn_force_rerun = self._pipeline_tab.btn_force
        self.btn_export = self._export_tab.btn_export
        self.chk_pdf_ocr_layer = self._export_tab.chk_ocr_layer
        self.chk_norm_widths = self._export_tab.chk_norm_widths

        # In capture mode, open straight into the live tab: pre-select the
        # chosen camera in the picker and build + show the live CaptureTab
        # inside the stack now. The webcam itself is started in __init__
        # (auto-start); the preview slot fills once the first frame lands.
        # "Deactivate camera" swaps the stack back to the picker, where
        # "Activate Capture" rebuilds the live tab on the chosen device.
        if is_capture:
            self._select_capture_camera(getattr(self.args, "camera_id", 0))
            ct = self._make_live_capture_tab()
            self._capture_stack.addWidget(ct)        # index 1 = live
            self._capture_stack.setCurrentIndex(1)
            self._capture_tab = ct

    def _select_capture_camera(self, cam_id) -> None:
        """Pre-select ``cam_id`` in the picker combo (best-effort)."""
        combo = getattr(self, "_capture_cam_combo", None)
        if combo is None:
            return
        try:
            cam_id = int(cam_id)
        except (TypeError, ValueError):
            return
        idx = combo.findData(cam_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _make_live_capture_tab(self) -> "CaptureTab":
        """Build a live ``CaptureTab``, wire its handlers, show + connect
        the Deactivate button, refresh the legacy button aliases, and kick
        off the max-zoom probe. Shared by launch-from-camera and the
        late-activation (open-from-disk) path. Does NOT touch the webcam
        thread or the stack — callers own that."""
        from lib.gui.theme import icon as _icon
        ct = CaptureTab()
        for ico_name, label, _ in self._transform_items:
            if ico_name is None:
                ct.transform_combo.addItem(label)
            else:
                ct.transform_combo.addItem(_icon(ico_name), label)
        ct.transform_combo.setCurrentIndex(0)
        ct.btn_full_calibrate.clicked.connect(self.calibrate_camera)
        ct.btn_dpi_calibrate.clicked.connect(self.calibrate_dpi)
        ct.dpi_label.clicked.connect(self._prompt_manual_dpi)
        ct.btn_freehand.clicked.connect(self._on_freehand_clicked)
        ct.btn_apply_transform.clicked.connect(self._apply_selected_transform)
        ct.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        ct.zoom_spin.valueChanged.connect(self._on_zoom_spin)
        ct.btn_voice.toggled.connect(self._on_voice_toggle)
        # Voice is Vosk-only now — one engine, so the split button auto-hides
        # its ▾ chevron and renders as a plain toggle.
        _engines = self._available_voice_engines()
        self._voice_engine = _engines[0][0] if _engines else "vosk"
        ct.set_voice_engines(_engines, self._voice_engine)
        ct.voice_engine_changed.connect(self._on_voice_engine_changed)
        try:
            keys = (self.args.config.get("keycontrols") or {})
            ct.set_shortcut_legend({
                "scan": keys.get("scan", []),
                "trash": keys.get("trash", []),
                "rotate": keys.get("rotate", []),
            })
            ct.set_voice_command_legend(self.args.config.get("voicecontrols") or {})
        except Exception:
            pass
        # Primary capture button — same action as SPACE / voice "photo".
        ct.btn_capture.clicked.connect(self.capture)
        ct.btn_deactivate.show()
        ct.btn_deactivate.clicked.connect(self._deactivate_capture_clicked)
        # Legacy button aliases so other call sites keep working.
        self.btn_full_calibrate = ct.btn_full_calibrate
        self.btn_dpi_calibrate = ct.btn_dpi_calibrate
        self.btn_freehand = ct.btn_freehand
        self.transform_combo = ct.transform_combo
        self.btn_apply_transform = ct.btn_apply_transform
        self.zoom_slider = ct.zoom_slider
        self.zoom_spin = ct.zoom_spin
        # Webcam max-zoom resolves after the first frame.
        self._zoom_init_timer = QTimer(self)
        self._zoom_init_timer.setSingleShot(True)
        self._zoom_init_timer.timeout.connect(self._init_zoom_range)
        self._zoom_init_timer.start(1500)
        # Camera + format pickers (Continuity Camera exposes several modes).
        self._populate_capture_devices(ct)
        ct.camera_combo.currentIndexChanged.connect(self._on_capture_device_changed)
        ct.format_combo.currentIndexChanged.connect(self._on_capture_format_changed)
        # Seed the DPI readout (uncalibrated default until calibration runs).
        try:
            ct.set_dpi(self.effective_dpi(), calibrated=bool(self.calibration),
                       manual=bool(self._manual_dpi is not None and not self.calibration))
        except Exception:
            pass
        return ct

    def _refresh_dpi_readout(self) -> None:
        """Push the current effective DPI into the capture tab's readout."""
        ct = getattr(self, "_capture_tab", None)
        if ct is not None and hasattr(ct, "set_dpi"):
            try:
                ct.set_dpi(self.effective_dpi(), calibrated=bool(self.calibration),
                           manual=bool(self._manual_dpi is not None and not self.calibration))
            except Exception:
                pass

    def _prompt_manual_dpi(self) -> None:
        """Let the user type the scan DPI directly (the readout is clickable).

        Stored as a standalone override (`self._manual_dpi`), NOT a fake
        calibration — see the note where `_manual_dpi` is initialised. Also
        updates the `input_dpi` general option so the spawned workers pick the
        new value up on the `reload_params` broadcast below."""
        from PySide6.QtWidgets import QInputDialog
        cur = self.effective_dpi()
        val, ok = QInputDialog.getDouble(
            self, self.tr("Set DPI manually"),
            self.tr("Scan resolution in DPI:"),
            float(cur), 10.0, 4800.0, 0)
        if not ok:
            return
        self._manual_dpi = float(val)
        # Feed the workers: input_dpi is the uncalibrated DPI source.
        try:
            self.args.options["general"]["input_dpi"] = float(val)
        except Exception:
            pass
        if self.processing_queue:
            for _ in range(self.total_workers + 2):
                self.processing_queue.put(('reload_params',))
        self.toast(self.tr("DPI set manually — {dpi:.0f} dpi.").format(dpi=float(val)))
        self._refresh_dpi_readout()

    def _populate_capture_devices(self, ct) -> None:
        """Fill the camera + format combos and pre-select the active device.
        Signals are blocked so populating doesn't trigger a restart."""
        from lib.gui.CameraEnum import list_cameras, list_camera_formats
        cam_id = getattr(self, "_active_cam_id", getattr(self.args, "camera_id", 0))
        ct.camera_combo.blockSignals(True)
        ct.camera_combo.clear()
        for cid, name in (list_cameras() or [(0, "Default camera")]):
            ct.camera_combo.addItem(name, int(cid))
        i = ct.camera_combo.findData(int(cam_id))
        if i >= 0:
            ct.camera_combo.setCurrentIndex(i)
        ct.camera_combo.blockSignals(False)
        self._populate_capture_formats(ct, cam_id)

    def _populate_capture_formats(self, ct, cam_id) -> None:
        """Fill the format combo for ``cam_id``; first entry = auto-pick."""
        from lib.gui.CameraEnum import list_camera_formats
        ct.format_combo.blockSignals(True)
        ct.format_combo.clear()
        # First item = auto (None → WebcamThread picks widest FOV / sensor).
        ct.format_combo.addItem(self.tr("Auto (widest)"), None)
        for f in list_camera_formats(int(cam_id)):
            ct.format_combo.addItem(f["label"], int(f["index"]))
        ct.format_combo.setCurrentIndex(0)
        ct.format_combo.blockSignals(False)

    def _on_capture_device_changed(self, _idx: int) -> None:
        ct = getattr(self, "_capture_tab", None)
        if ct is None or not hasattr(ct, "camera_combo"):
            return
        cam_id = ct.camera_combo.currentData()
        if cam_id is None:
            return
        self._populate_capture_formats(ct, cam_id)   # reset to Auto for new device
        self._restart_webcam(int(cam_id), None)

    def _on_capture_format_changed(self, _idx: int) -> None:
        ct = getattr(self, "_capture_tab", None)
        if ct is None or not hasattr(ct, "format_combo"):
            return
        fmt_index = ct.format_combo.currentData()   # None = auto
        self._restart_webcam(getattr(self, "_active_cam_id", 0), fmt_index)

    def _restart_webcam(self, cam_id: int, format_index) -> None:
        """Stop the current webcam and start a fresh one on ``cam_id`` with
        the given AVFoundation format index (None = auto). Re-wires the
        preview, overlay, and transform; re-probes the zoom range."""
        old = getattr(self, "webcam_thread", None)
        if old is not None:
            try:
                old.change_pixmap_signal.disconnect(self.update_image)
            except Exception:
                pass
            try:
                old.stop()
            except Exception:
                pass
        try:
            self.webcam_thread = WebcamThread(cam_id, format_index)
            self.webcam_thread.change_pixmap_signal.connect(self.update_image)
            self.webcam_thread.set_overlay_fn(self._freehand_overlay)
            if getattr(self.args, "transform", None):
                self.webcam_thread.set_transform(self.args.transform)
            self.webcam_thread.start()
            self._active_cam_id = int(cam_id)
        except Exception as e:
            QMessageBox.critical(
                self, self.tr("Capture"),
                self.tr("Failed to start camera #{cid}: {err}").format(cid=cam_id, err=e))
            self.webcam_thread = None
            return
        # Re-probe max-zoom against the new device/format.
        self._zoom_init_timer = QTimer(self)
        self._zoom_init_timer.setSingleShot(True)
        self._zoom_init_timer.timeout.connect(self._init_zoom_range)
        self._zoom_init_timer.start(1500)

    def _persist_sidebar_state(self) -> None:
        try:
            from lib.app_data import db as _cfg
            with _cfg.session() as _conn:
                _cfg.set(_conn, _cfg.KEY_SIDEBAR_TAB, self.sidebar.active())
                _cfg.set(_conn, _cfg.KEY_SIDEBAR_COLLAPSED,
                         bool(self.sidebar.collapsed()))
        except Exception:
            pass

    def _open_tip_url(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://ko-fi.com/yannbarbotin"))

    def _on_voice_toggle(self, on: bool) -> None:
        if on:
            self._start_voice()
        else:
            self._stop_voice()

    def _on_sidebar_import_requested(self, items, dpi: float) -> None:
        """Handle Import tab's import-button click. Fans out to the
        existing ``enqueue_image_files`` / ``enqueue_pdf_files`` workers,
        preserving the in-queue order — files land in the scans gallery
        in the same order they appear in the tab."""
        from pathlib import Path as _P
        from lib.workers.ImportHelpers import (
            enqueue_image_files, enqueue_pdf_files,
        )
        images = [p for p, k in items if k == "image"]
        pdfs = [p for p, k in items if k == "pdf"]
        slug = slugify(_P(self.db_path).stem) or "scan"
        try:
            if images:
                enqueue_image_files(
                    db_path=self.db_path,
                    pipeline_version_id=self.pipeline_version_id,
                    slug=slug, chain=self.chain,
                    image_paths=images, default_dpi=float(dpi),
                    log_queue=self.log_queue,
                )
            if pdfs:
                enqueue_pdf_files(
                    db_path=self.db_path,
                    pipeline_version_id=self.pipeline_version_id,
                    slug=slug, chain=self.chain,
                    pdf_paths=pdfs, render_dpi=float(dpi),
                    log_queue=self.log_queue,
                )
        finally:
            self._import_tab.clear_queue()

    def _on_export_clicked(self):
        """Single Export button — dispatch by currently selected format
        card."""
        fmt = self._export_tab.current_format()
        if fmt == "pdf":
            self.make_pdf("output")
        elif fmt == "markdown":
            self._export_markdown()
        elif fmt == "slim":
            self._export_slim_project()

    def _init_zoom_range(self):
        """Pull the device max_zoom from WebcamThread once it has resolved."""
        if self.webcam_thread is None:
            return
        # Late-activated capture sessions populate the legacy aliases
        # (``self.zoom_slider`` / ``self.zoom_spin``) inside
        # ``_activate_capture_clicked``; pull from the live CaptureTab
        # as a fallback so this timer-driven slot doesn't AttributeError
        # if it fires before the alias was wired.
        zoom_slider = getattr(self, "zoom_slider", None)
        zoom_spin = getattr(self, "zoom_spin", None)
        if zoom_slider is None or zoom_spin is None:
            ct = getattr(self, "_capture_tab", None)
            if ct is not None and hasattr(ct, "zoom_slider"):
                zoom_slider = ct.zoom_slider
                zoom_spin = ct.zoom_spin
                self.zoom_slider = zoom_slider
                self.zoom_spin = zoom_spin
            else:
                return
        max_z = float(getattr(self.webcam_thread, "max_zoom", 1.0))
        cur_z = float(getattr(self.webcam_thread, "current_zoom", 1.0))
        if max_z <= 1.0:
            # Device doesn't support zoom — leave widgets disabled.
            zoom_slider.setEnabled(False)
            zoom_spin.setEnabled(False)
            return
        self.zoom_slider.blockSignals(True)
        self.zoom_spin.blockSignals(True)
        self.zoom_slider.setRange(100, int(max_z * 100))
        self.zoom_slider.setValue(int(cur_z * 100))
        self.zoom_spin.setRange(1.0, max_z)
        self.zoom_spin.setValue(cur_z)
        self.zoom_slider.blockSignals(False)
        self.zoom_spin.blockSignals(False)

    def _on_zoom_slider(self, val: int):
        if self.webcam_thread is None:
            return
        f = self.webcam_thread.set_zoom(val / 100.0)
        self.zoom_spin.blockSignals(True)
        self.zoom_spin.setValue(f)
        self.zoom_spin.blockSignals(False)
        self.status_label.setText(self.tr("Zoom: {f:.2f}x").format(f=f))
        self._refresh_dpi_readout()   # DPI scales with zoom

    def _on_zoom_spin(self, val: float):
        if self.webcam_thread is None:
            return
        f = self.webcam_thread.set_zoom(val)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(f * 100))
        self.zoom_slider.blockSignals(False)
        self.status_label.setText(self.tr("Zoom: {f:.2f}x").format(f=f))
        self._refresh_dpi_readout()   # DPI scales with zoom

    def update_image(self, cv_img):
        from PySide6.QtGui import QPixmap
        pix = QPixmap.fromImage(cv_img)
        if pix.isNull() or pix.width() <= 0:
            return
        # Mirror the frame into the CaptureTab preview slot when present
        # so the sidebar's "Capture" tab shows the live camera too.
        # Done unconditionally — in open-from-disk projects the
        # left-pane video_label is None, but the sidebar CaptureTab is
        # the only surface that needs the feed.
        ct = getattr(self, "_capture_tab", None)
        if ct is not None and hasattr(ct, "set_preview_pixmap"):
            try:
                ct.set_preview_pixmap(pix)
            except Exception:
                pass
        if self.video_label is None:
            return
        # Width comes from the layout (column cap). Drive the label
        # height from the camera's native aspect so the preview shows
        # the full frame, no letterbox, no crop.
        w = self.video_label.width()
        if w <= 0:
            return
        target_h = int(round(w * pix.height() / pix.width()))
        if self.video_label.height() != target_h:
            self.video_label.setFixedHeight(target_h)
        scaled = pix.scaled(
            w, target_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,  # already proportional
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def update_voice_label(self, text):
        # The worker sends pre-trimmed rich-text (coloured <span> words) —
        # forward it verbatim. Splitting on whitespace here would cut through
        # the HTML tags (each span contains spaces) and corrupt the markup.
        # Only plain text gets the last-10-words trim.
        if "<span" not in text:
            words = text.split()
            if len(words) > 10:
                text = "... " + " ".join(words[-10:])
        ct = getattr(self, "_capture_tab", None)
        if ct is not None and hasattr(ct, "set_voice_transcript"):
            ct.set_voice_transcript(text)

    # ── voice engines + on/off ─────────────────────────────────────────
    def _available_voice_engines(self):
        """[(key, label)] usable on this platform. Voice control is Vosk only
        (offline, constrained grammar, cross-platform). Apple's recognizers
        were dropped: free-form transcription mishears the command words and
        SFSpeechRecognizer/SpeechAnalyzer add multi-second latency."""
        engines = [("vosk", self.tr("Vosk — offline"))]
        return engines

    def _on_voice_engine_changed(self, key: str) -> None:
        self._voice_engine = key
        if self.voice_thread is not None and self.voice_thread.isRunning():
            self._stop_voice()
            self._start_voice()

    def _start_voice(self):
        # Voice control is Vosk only (offline, constrained grammar).
        try:
            from lib.gui.VoiceWorkerVosk import (
                VoskVoiceWorker, HAS_VOSK, _model_dir)
        except Exception:
            HAS_VOSK, _model_dir = False, (lambda: None)
        if not HAS_VOSK:
            self._voice_unavailable(self.tr(
                "Offline voice needs the 'voice' extra "
                "(uv sync --extra voice)."), offer_download=False)
            return
        if _model_dir() is None:
            self._voice_unavailable(self.tr(
                "The Vosk voice model isn't downloaded yet."),
                offer_download=True)
            return
        # QThread can't restart once finished — spin a fresh worker.
        self.voice_thread = VoskVoiceWorker(self.args.config)
        self.voice_thread.command_detected.connect(self.handle_voice_command)
        self.voice_thread.transcription_update.connect(self.update_voice_label)
        self.voice_thread.start()

    def _voice_unavailable(self, message: str, *, offer_download: bool) -> None:
        """Reset the toggle + explain why voice didn't start; when a model
        download would fix it, offer to open the downloader."""
        ct = getattr(self, "_capture_tab", None)
        if ct is not None and hasattr(ct, "btn_voice"):
            ct.btn_voice.blockSignals(True)
            ct.btn_voice.setChecked(False)
            ct.btn_voice.blockSignals(False)
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(self.tr("Voice control"))
        box.setText(message)
        if offer_download:
            dl = box.addButton(self.tr("Open downloader"),
                               QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is dl:
                self._open_model_downloader()
        else:
            box.exec()

    def _stop_voice(self):
        if self.voice_thread is not None:
            try:
                self.voice_thread.stop()
            except Exception:
                pass
            self.voice_thread.quit()
            self.voice_thread.wait(500)
        self.voice_thread = None

    # ── Freehand SIFT-tracked capture ──────────────────────────────
    def _on_freehand_clicked(self, checked: bool) -> None:
        """Toolbar entry point. When toggled on, open the registration
        tab; when toggled off, stop tracking. The toggle state mirrors
        whether the tracker is live, not the tab — registration may be
        in flight while the toggle stays off until success."""
        if checked:
            self._open_freehand_tab()
            # The button stays toggled-off until registration succeeds;
            # toggle back if the user clicked while not registered.
            if self._sift_tracker is None:
                self.btn_freehand.setChecked(False)
        else:
            self._freehand_stop()

    def _open_freehand_tab(self) -> None:
        if self.webcam_thread is None:
            QMessageBox.warning(self, self.tr("Hands-free"),
                                self.tr("Hands-free capture needs the webcam preview."))
            return
        if self._freehand_tab is not None:
            self._freehand_tab.raise_()
            self._freehand_tab.activateWindow()
            return
        from lib.gui.CalibrationDialogs import FreehandRegistrationDialog
        # Pixel-based ROI — independent of zoom / DPI so the registration
        # box covers the same chunk of screen real estate no matter how
        # the user has framed the live preview.
        dlg = FreehandRegistrationDialog(
            self.webcam_thread, side_px=160, parent=self,
        )
        dlg.registered.connect(self._on_freehand_registered)
        dlg.finished.connect(self._close_freehand_tab)
        self._freehand_tab = dlg
        dlg.show()

    def _on_freehand_registered(self, patch, roi) -> None:
        from lib.workers.SiftTracker import ClickGate, SiftTracker
        if patch is None or roi is None:
            self._close_freehand_tab()
            return
        frame = self.webcam_thread.get_frame() if self.webcam_thread else None
        if frame is None:
            self._close_freehand_tab()
            return
        tracker = SiftTracker()
        n_kp = tracker.register(frame, roi)
        if n_kp < 12:
            QMessageBox.information(
                self, self.tr("Hands-free"),
                self.tr(
                    "Could not extract a stable pattern from that frame "
                    "(only {n} keypoints). Try a busier pattern."
                ).format(n=n_kp))
            return  # leave the tab open so the user can re-try
        with self._sift_state_lock:
            self._sift_tracker = tracker
            self._sift_gate = ClickGate()
            self._sift_last_pts = None
            self._sift_last_fraction = 0.0
            self._sift_roi = roi
        # 5 Hz matcher tick — combined with the in-tracker frame
        # downsample (≤960 px) this keeps the GUI thread under ~10 ms
        # per tick on Apple Silicon. The IoU-based click gate works
        # off ≥0.5 s occlusion so 5 Hz is plenty of sampling.
        self._sift_timer = QTimer(self)
        self._sift_timer.setInterval(200)
        self._sift_timer.timeout.connect(self._freehand_tick)
        self._sift_timer.start()
        self.webcam_thread.set_overlay_fn(self._freehand_overlay)
        self.btn_freehand.setChecked(True)
        self.status_label.setText(
            self.tr("Hands-free armed ({n} keypoints) — cover pattern to capture.").format(n=n_kp)
        )
        self._close_freehand_tab()

    def _close_freehand_tab(self, *_args) -> None:
        dlg = self._freehand_tab
        if dlg is None:
            return
        try:
            dlg.accept()
        except Exception:
            pass
        try:
            dlg.deleteLater()
        except Exception:
            pass
        self._freehand_tab = None

    def _freehand_stop(self) -> None:
        t = self._sift_timer
        if t is not None:
            t.stop()
        with self._sift_state_lock:
            self._sift_timer = None
            self._sift_tracker = None
            self._sift_gate = None
            self._sift_last_pts = None
            self._sift_last_fraction = 0.0
            self._sift_roi = None
            self._sift_last_quad = None
            self._sift_flash_until = 0.0
        # Keep overlay fn installed — it now also paints the shared
        # capture flash (voice / keyboard). SIFT-specific polyline drops
        # out automatically because tracker is None.
        self.status_label.setText(self.tr("Hands-free disabled."))

    def _freehand_tick(self) -> None:
        tracker = getattr(self, "_sift_tracker", None)
        gate = getattr(self, "_sift_gate", None)
        if tracker is None or gate is None:
            return
        if self.webcam_thread is None:
            return
        frame = self.webcam_thread.get_frame()
        if frame is None:
            return
        res = tracker.update(frame)
        # Hand the per-frame fraction + projected quad to the click gate;
        # the gate uses the quad to confirm the pattern came back to the
        # same spot before firing.
        fired = gate.feed(res.fraction, res.quad)
        with self._sift_state_lock:
            self._sift_last_pts = res.points
            self._sift_last_fraction = float(res.fraction)
            self._sift_last_quad = res.quad
        if fired:
            # capture() sets the shared flash flag.
            self.capture()

    def _freehand_overlay(self, bgr):
        """Runs on the webcam thread. Draws the inferred tracked-square
        as a green polyline + a green capture-fired flash on the live
        preview. Single polyline keeps the per-frame paint cost
        negligible."""
        import time as _time
        with self._sift_state_lock:
            quad = self._sift_last_quad
            pts_match = self._sift_last_pts
            flash_until = float(self._sift_flash_until)
        # Capture-fired feedback: translucent green wash + thick border
        # around the whole preview for ~450 ms.
        if flash_until > _time.monotonic():
            h, w = bgr.shape[:2]
            wash = bgr.copy()
            cv2.rectangle(wash, (0, 0), (w - 1, h - 1), (0, 220, 0), -1)
            cv2.addWeighted(wash, 0.18, bgr, 0.82, 0, dst=bgr)
            cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 220, 0),
                          thickness=max(14, h // 60), lineType=cv2.LINE_AA)
        # Tracked-quad polyline. Gate on ≥10 matches so a degenerate
        # 4-match homography doesn't paint a flickering rhombus.
        n_matches = 0 if pts_match is None else len(pts_match)
        if quad is not None and len(quad) == 4 and n_matches >= 10:
            pts = quad.astype(np.int32).reshape(-1, 1, 2)
            thick = max(3, bgr.shape[0] // 90)
            cv2.polylines(bgr, [pts], isClosed=True, color=(0, 220, 0),
                          thickness=thick, lineType=cv2.LINE_AA)
        return bgr

    def _match_key(self, event, action_name):
        keys = self.args.config["keycontrols"].get(action_name, [])
        for k in keys:
            if len(k) == 1: # Single char check
                if event.text().upper() == k.upper(): return True
                # Also check Key_X enum for single chars if needed, but text() usually works
            
            # Map common key names to Qt Enums
            key_map = {
                "Space": Qt.Key.Key_Space,
                "Backspace": Qt.Key.Key_Backspace,
                "Return": Qt.Key.Key_Return,
                "Enter": Qt.Key.Key_Enter,
                "Escape": Qt.Key.Key_Escape,
                "Tab": Qt.Key.Key_Tab,
                "Delete": Qt.Key.Key_Delete,
            }
            
            if k in key_map and event.key() == key_map[k]:
                return True
                
            # If specified as "S" or "Q" but passed as Qt.Key.Key_S
            if hasattr(Qt.Key, f"Key_{k.upper()}"):
                if event.key() == getattr(Qt.Key, f"Key_{k.upper()}"):
                    return True
                    
        return False

    def keyPressEvent(self, event):
        # Quit is bound to the platform-standard ⌘Q / Ctrl+Q via
        # `_install_global_shortcuts`. Don't accept a bare letter here:
        # it would fire whenever an input field briefly lost focus.
        if self._match_key(event, "scan"):
            self.capture()
        elif self._match_key(event, "trash"):
            self.undo()
        elif self._match_key(event, "rotate"):
            self.rotate_camera()
            
    def handle_voice_command(self, cmd):
        if time.time() - self.last_voice_cmd_time < 1.0:
            return
        self.last_voice_cmd_time = time.time()
        
        self.status_label.setText(self.tr("Voice Command Detected: {cmd}").format(cmd=cmd.upper()))
        QTimer.singleShot(2000, lambda: self.status_label.setText(self.tr("Ready.")))

        # Run the action on the next event-loop tick, never inline. The voice
        # callback may land on the main thread; doing heavy work (capture /
        # delete) inline would block recognition until it finishes. Deferring
        # lets the callback return immediately so listening stays responsive.
        if cmd == "scan":
            QTimer.singleShot(0, self.capture)
        elif cmd == "trash":
            QTimer.singleShot(0, self.undo)
        elif cmd == "quit":
            QTimer.singleShot(0, self.close)

    def capture(self):
        if self.webcam_thread is None:
            self.status_label.setText(self.tr("Capture unavailable in this mode."))
            return
        # Shared visual cue — every capture trigger (SIFT / voice /
        # keyboard / button) flashes the preview frame. Single source of
        # truth for the flash so the timing reads the same regardless of
        # how the user fired the capture.
        import time as _time
        self._sift_flash_until = _time.monotonic() + 0.45
        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            self.status_label.setText(self.tr("OCR running — capture disabled until it finishes."))
            return
        frame = self.webcam_thread.get_frame()
        if frame is None:
            return

        # Apply Undistortion if available
        current_dpi = None
        if self.calibration:
            mtx = self.calibration["mtx"]
            dist = self.calibration["dist"]
            # Use the zoom-scaled effective DPI rather than the raw
            # calibration value, so DPI tracks the live camera zoom.
            current_dpi = self.effective_dpi()
            h, w = frame.shape[:2]
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        input_dpi = self.args.options["general"].get("input_dpi", 100.0)
        capture_dpi = float(current_dpi or input_dpi)

        # Convert BGR → RGB (the chain expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Persist scan + root node (raw) in the project DB.
        with db_session(self.db_path) as conn:
            persister = Persister(conn)
            scan_repo = ScanRepo(conn)
            # Encode the camera id in source_ref ("webcam#<id>") so the
            # Fix-input-DPI view can show "capture #<id>" without a schema
            # change. Legacy scans carry the bare "webcam".
            _cam_id = getattr(self, "_active_cam_id", getattr(self.args, "camera_id", 0))
            scan_id = scan_repo.create(
                "capture", self.pipeline_version_id,
                source_ref=f"webcam#{_cam_id}", capture_dpi=capture_dpi,
            )
            idx_str = f"{scan_repo.get(scan_id)['idx']:03d}"
            filestem = f"{self.slug_name}_{idx_str}"
            image_id = persister.persist_image(rgb_frame, ImageType.COLOR.value, capture_dpi)
            root_node_id = persister.persist_node(
                scan_id=scan_id, parent_id=None,
                pipeline_version_id=self.pipeline_version_id,
                step_idx=0, step_name=None, processor_name=None,
                branch_label=None, depth=0, filestem=filestem, image_id=image_id,
            )
            scan_repo.set_root(scan_id, root_node_id)

        scan_idx = int(idx_str)
        widget = self._spawn_widget(scan_id, scan_idx, root_node_id, image_id, filestem)
        widget.restore_node(node_id=root_node_id, parent_node_id=None,
                            image_id=image_id, step_name="raw", filestem=filestem, meta=None)

        # Enqueue the raw for processing as an ImageBuffer — the same shape
        # the import path uses (`ImportHelpers.enqueue_image_files`). The
        # worker loop only consumes ImageBuffers / ("routed", …) tuples; an
        # earlier ("ref", …) DB-reference form was produced here but never
        # implemented worker-side, so captures were silently never processed.
        buf = ImageBuffer(
            rgb_frame, ImageType.COLOR, dpi=capture_dpi,
            filestem=filestem, scan_id=int(scan_id),
            parent_node_id=int(root_node_id),
            pipeline_version_id=self.pipeline_version_id, depth=0,
        )
        self.processing_queue.put(buf)
        self.current_idx = max(self.current_idx, scan_idx + 1)
        self.status_label.setText(self.tr("Captured {stem}").format(stem=filestem))
        self._scan_added_to_views(scan_id)

    def undo(self):
        if not self.history:
            self.status_label.setText(self.tr("Nothing to undo."))
            return
        last_scan_id = self.history[-1]
        self.delete_scan(last_scan_id)
        self.status_label.setText(self.tr("Undid scan {sid}").format(sid=last_scan_id))

    def delete_scan(self, scan_id: int):
        if scan_id in self.history:
            self.history.remove(scan_id)
        widget = self.scan_widgets_by_scan.pop(scan_id, None)
        if widget is not None:
            self.scroll_layout.removeWidget(widget)
            widget.deleteLater()
        # Soft-delete in DB so the scan disappears from list_active.
        with db_session(self.db_path) as conn:
            ScanRepo(conn).soft_delete(scan_id)
        self.status_label.setText(self.tr("Deleted scan #{sid}").format(sid=scan_id))
        # Drop the row from the visible alt view only (incremental). Hidden
        # views (incl. the common grid mode) do nothing and rebuild lazily
        # when selected — no 300-row rebuild on a single delete.
        self._scan_removed_from_views(scan_id)

    def on_scan_imported(self, payload: dict):
        """Background import → spawn raw widget immediately so the user sees
        the scan before any worker stage completes."""
        if not isinstance(payload, dict):
            return
        scan_id = payload.get("scan_id")
        if scan_id is None or scan_id in self.scan_widgets_by_scan:
            return
        try:
            self._spawn_widget(
                scan_id,
                int(payload.get("idx") or 0),
                int(payload.get("root_node_id") or 0),
                int(payload.get("raw_image_id") or 0),
                payload.get("filestem") or "",
            )
            idx = int(payload.get("idx") or 0)
            if idx + 1 > self.current_idx:
                self.current_idx = idx + 1
            # Incremental add to the visible alt view — importing a 300-page
            # PDF must not trigger a full table rebuild per page.
            self._scan_added_to_views(scan_id)
        except Exception:
            pass

    def on_image_event(self, payload: dict):
        """M0 dict-form event handler. Routes by scan_id."""
        if not isinstance(payload, dict):
            return
        scan_id = payload.get("scan_id")
        target = self.scan_widgets_by_scan.get(scan_id) if scan_id is not None else None
        if target is None:
            return
        target.handle_event(
            node_id=payload.get("node_id"),
            parent_node_id=payload.get("parent_node_id"),
            image_id=payload.get("image_id"),
            step_name=payload.get("event_type") or "",
            filestem=payload.get("filestem") or "",
            meta=payload.get("meta") or {},
        )

    def on_image_events_batch(self, events: list):
        """Batched image events (see ProcessMonitor.image_events_batch_signal).

        Applies each event's node registration, but defers the per-widget
        header refresh to ONCE per affected widget per batch — the dominant
        per-event cost during a large reprocess. Thumbnails already rebuild via
        each widget's coalesced refresh timer."""
        affected = set()
        widgets = self.scan_widgets_by_scan
        for payload in events:
            if not isinstance(payload, dict):
                continue
            scan_id = payload.get("scan_id")
            target = widgets.get(scan_id) if scan_id is not None else None
            if target is None:
                continue
            target.handle_event(
                node_id=payload.get("node_id"),
                parent_node_id=payload.get("parent_node_id"),
                image_id=payload.get("image_id"),
                step_name=payload.get("event_type") or "",
                filestem=payload.get("filestem") or "",
                meta=payload.get("meta") or {},
                defer_header=True,
            )
            affected.add(target)
        for w in affected:
            try:
                w.update_header()
            except Exception:
                pass

    def make_pdf(self, source_type, *, step_name: Optional[str] = None):
        """Build a PDF from DB blobs.

        source_type:
          - 'output' → chosen branch terminals (final result per scan)
          - any other → an intermediate `step_name` (also passed via step_name=)
        """
        if source_type == "output":
            suffix = ""
            step_filter = None
        else:
            step_filter = step_name or source_type
            suffix = f"_{step_filter}"

        compression = "auto"
        if hasattr(self, "_export_tab"):
            compression = self._export_tab.compression_hint()
        add_ocr_layer = bool(
            getattr(self, "chk_pdf_ocr_layer", None)
            and self.chk_pdf_ocr_layer.isEnabled()
            and self.chk_pdf_ocr_layer.isChecked()
        )
        # Tag the file with the OCR engine used when the user opted to
        # embed a text layer — makes it obvious which Aglaïa run
        # produced the searchable PDF (apple / surya / paddle).
        ocr_suffix = ""
        if add_ocr_layer:
            try:
                from lib.workers.md_export import ocr_engine_suffix
                with db_session(self.db_path) as _conn:
                    ocr_suffix = ocr_engine_suffix(_conn)
            except Exception:
                ocr_suffix = ""
        output_filename = f"{self.slug_name}{suffix}{ocr_suffix}.pdf"
        from PySide6.QtWidgets import QFileDialog
        default = self.args.workspace_dir / output_filename
        dest, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export PDF"), str(default), self.tr("PDF (*.pdf)"))
        if not dest:
            return
        output_path = Path(dest)
        self.status_label.setText(
            self.tr("Generating PDF ({src}, {comp})…").format(src=source_type, comp=compression)
        )
        QTimer.singleShot(100, lambda: self._run_pdf_maker(
            step_filter, output_path, compression, add_ocr_layer))

    def _run_pdf_maker(self, step_name: Optional[str], output_path: Path,
                       compression: str = "auto", add_ocr_layer: bool = False):
        with db_session(self.db_path) as conn:
            success = create_pdf_from_db(
                conn, output_path, step_name=step_name, compression=compression,
                add_ocr_layer=add_ocr_layer,
            )
        if success:
            self.status_label.setText(self.tr("Saved: {name}").format(name=output_path.name))
            self.toast(self.tr("PDF saved — {name}").format(name=output_path.name))
            self._reveal_in_finder(output_path)
        else:
            self.status_label.setText(self.tr("Failed to create PDF (no images)."))
            self.toast(self.tr("PDF export failed."), 3000)
        QTimer.singleShot(3000, lambda: self.status_label.setText(self.tr("Ready.")))

    def _on_slim_down_in_place(self):
        """Prune the *current* project file in place (same keep-set as the
        slim export: raw captures + chosen outputs + their OCR), dropping
        all intermediate pipeline states. Confirms first, then closes and
        reopens the project so the view rebuilds against the slimmed DB.

        The prune itself runs in main() once this window has closed (chain
        stopped → DB free); here we only gate, confirm, and arm the
        reopen round-trip."""
        from PySide6.QtWidgets import QApplication

        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            QMessageBox.warning(self, self.tr("Slim-down"),
                                self.tr("Wait for the OCR pass to finish first."))
            return
        if not self._pipeline_idle():
            QMessageBox.warning(
                self, self.tr("Slim-down"),
                self.tr("Wait for the pipeline to finish before slimming down."))
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(self.tr("Slim-down current project"))
        box.setText(self.tr(
            "After slimming down, the intermediate processing states are "
            "not available anymore (they can be regenerated since the "
            "originals are kept)."))
        box.setInformativeText(self.tr(
            "The current view will close and re-open."))
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        slim_btn = box.addButton(self.tr("Slim down"),
                                 QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        if box.clickedButton() is not slim_btn:
            return

        # Arm the in-place slim + reopen round-trip handled by main().
        app = QApplication.instance()
        if app is not None:
            app.setProperty("aglaia_reopen_path", str(self.db_path))
            app.setProperty("aglaia_slim_before_reopen", True)
            app.setProperty("aglaia_restart", "reopen")
        self.close()

    def _export_slim_project(self):
        """Save a pruned copy of the project DB. Asks for a destination
        path (defaulting to `<project>-slim.scanproj.sqlite`), then runs
        the slim_export pass on a worker thread so the GUI stays
        responsive on large projects."""
        from PySide6.QtWidgets import QFileDialog
        from lib.workers.slim_export import slim_export, default_slim_path

        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            QMessageBox.warning(self, self.tr("Slim export"),
                                self.tr("Wait for the OCR pass to finish first."))
            return
        if not self._pipeline_idle():
            reply = QMessageBox.question(
                self, self.tr("Slim export"),
                self.tr(
                    "The pipeline is still running. Export the current "
                    "DB anyway? (In-flight nodes won't be in the result.)"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        from lib.storage import PROJECT_DIALOG_FILTER
        src = Path(self.db_path)
        default = default_slim_path(src)
        dest, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export slimmed project"), str(default),
            PROJECT_DIALOG_FILTER,
        )
        if not dest:
            return
        dest_path = Path(dest)
        if dest_path == src:
            QMessageBox.warning(self, self.tr("Slim export"),
                                self.tr("Destination must differ from the active project."))
            return

        self.status_label.setText(self.tr("Building slim project…"))
        # NB: don't disable btn_export here. The done/fail callbacks are
        # dispatched via QTimer.singleShot from a worker thread (no event
        # loop) so they may never fire — which left the button greyed for
        # every format after a slim export. The modal save dialog already
        # guards against re-entry, so keep it enabled (matches PDF/MD).

        import threading as _t

        def _run():
            try:
                stats = slim_export(src, dest_path)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                QTimer.singleShot(0, lambda: self._on_slim_export_failed(err))
                return
            QTimer.singleShot(0, lambda: self._on_slim_export_done(dest_path, stats))

        _t.Thread(target=_run, daemon=True, name="SlimExport").start()

    def _on_slim_export_done(self, dest: Path, stats: dict) -> None:
        before_mb = stats["size_before"] / (1024 * 1024)
        after_mb = stats["size_after"] / (1024 * 1024)
        msg = self.tr(
            "Slim export saved: {name}\n"
            "{after_mb:.1f} MB (was {before_mb:.1f} MB) · "
            "{kept} kept, {dropped} dropped image(s)"
        ).format(
            name=dest.name, after_mb=after_mb, before_mb=before_mb,
            kept=stats['kept_images'], dropped=stats['dropped_images'],
        )
        self.status_label.setText(self.tr("Saved: {name}").format(name=dest.name))
        self.toast(msg, 4000)
        QTimer.singleShot(3000, lambda: self.status_label.setText(self.tr("Ready.")))

    def _on_slim_export_failed(self, err: str) -> None:
        self._on_log_line("error", f"Slim export failed: {err}")
        self.status_label.setText(self.tr("Slim export failed."))
        QTimer.singleShot(3000, lambda: self.status_label.setText(self.tr("Ready.")))
        QMessageBox.critical(self, self.tr("Slim export"), err)

    def _export_markdown(self):
        from lib.workers.md_export import write_markdown, ocr_engine_suffix
        from PySide6.QtWidgets import QFileDialog
        try:
            with db_session(self.db_path) as conn:
                suffix = ocr_engine_suffix(conn)
        except Exception:
            suffix = ""
        default = self.args.workspace_dir / f"{self.slug_name}{suffix}.md"
        dest, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Markdown"), str(default),
            self.tr("Markdown (*.md)"))
        if not dest:
            return
        output_path = Path(dest)
        use_llm = bool(
            getattr(self._export_tab, "chk_llm_refine", None)
            and self._export_tab.chk_llm_refine.isEnabled()
            and self._export_tab.chk_llm_refine.isChecked()
        )
        refine = "apple_fm" if use_llm else None
        self.status_label.setText(
            self.tr("Polishing with Apple Intelligence…") if use_llm
            else self.tr("Writing Markdown…"))
        try:
            with db_session(self.db_path) as conn:
                ok = write_markdown(conn, output_path, refine=refine)
        except Exception as e:
            self._on_log_line("error", f"Markdown export failed: {e}")
            self.status_label.setText(self.tr("Markdown export failed."))
            QTimer.singleShot(3000, lambda: self.status_label.setText(self.tr("Ready.")))
            return
        if ok:
            self.status_label.setText(self.tr("Saved: {name}").format(name=output_path.name))
            self.toast(self.tr("Markdown saved — {name}").format(name=output_path.name))
            self._reveal_in_finder(output_path)
        else:
            self.status_label.setText(self.tr("No OCR text to export."))
            self.toast(self.tr("Markdown export skipped — no OCR text."), 3000)
        QTimer.singleShot(3000, lambda: self.status_label.setText(self.tr("Ready.")))

    def refresh_norm_widths_visibility(self):
        """Show + pre-tick the checkbox when the loaded pipeline asks
        for batch width normalisation (any TrapezoidalCorrection or
        PageDewarper step with `norm_width: true`). Hide otherwise.

        Called after pipeline (re)load. Reads the current yaml from
        `pipeline_yaml_path` rather than relying on a live `args`
        attribute — that way we don't need a separate elements
        accessor on the chain.
        """
        wants = False
        if self.pipeline_yaml_path:
            try:
                import yaml as _yaml
                doc = _yaml.safe_load(
                    Path(self.pipeline_yaml_path).read_text()
                ) or {}
                for step in doc.get("pipeline", []) or []:
                    proc = step.get("processor", "")
                    if proc not in ("TrapezoidalCorrection", "PageDewarper"):
                        continue
                    if bool((step.get("options") or {}).get("norm_width")):
                        wants = True
                        break
            except Exception:
                wants = False
        # Block stateChanged during the programmatic tick — we only
        # want the user click to trigger the batch run.
        self.chk_norm_widths.blockSignals(True)
        self.chk_norm_widths.setChecked(wants)
        self.chk_norm_widths.blockSignals(False)
        self.chk_norm_widths.setVisible(wants)

    def _on_norm_widths_toggled(self, state: int):
        from PySide6.QtCore import Qt
        if state != Qt.CheckState.Checked.value:
            return  # untick = passive; output stays in DB
        out_dir = self.args.workspace_dir / "results-norm"
        self.status_label.setText(self.tr("Normalising widths (computing scales)…"))
        QTimer.singleShot(100, lambda: self._run_blob_normalizer(out_dir))

    def _run_blob_normalizer(self, out_dir: Path):
        from lib.workers.BlobNormalizer import compute_scales, apply_scales
        try:
            scales = compute_scales(self.db_path)
            if not scales:
                self.status_label.setText(
                    self.tr("Normalise: no terminal images found."))
                QTimer.singleShot(3000,
                                  lambda: self.status_label.setText(self.tr("Ready.")))
                return
            self.status_label.setText(
                self.tr("Normalising widths (writing {n} files)…").format(n=len(scales)))
            QApplication.processEvents()
            n = apply_scales(self.db_path, scales, out_dir)
            self.status_label.setText(
                self.tr("Normalised {n} files → {dir}/").format(n=n, dir=out_dir.name))
        except Exception as e:
            self.status_label.setText(self.tr("Normalise failed: {err}").format(err=e))
        QTimer.singleShot(4000, lambda: self.status_label.setText(self.tr("Ready.")))


    def calibrate_camera(self):
        frame = self.webcam_thread.get_frame()
        if frame is None:
            self.status_label.setText(self.tr("Error: No frame from camera."))
            return

        if not self.is_calibrating:
            # Start fresh
            self.is_calibrating = True
            self.calibrator.reset()
            self.status_label.setText(self.tr("Calibration Mode: Capture 1/{total}").format(
                total=self.cal_target_count,
            ))

        success, msg = self.calibrator.collect_sample(frame)

        if not success:
            self.status_label.setText(self.tr("Sample failed: {msg}").format(msg=msg))
            return

        current_count = len(self.calibrator.imgpoints)

        if current_count < self.cal_target_count:
            remaining = self.cal_target_count - current_count
            if current_count == self.cal_target_count - 1:
                self.btn_full_calibrate.setText(self.tr("Last one : put the board flat, at book distance"))
                self.status_label.setText(self.tr("Prepare final sample: {n}/{total}").format(
                    n=current_count + 1, total=self.cal_target_count,
                ))
            else:
                self.btn_full_calibrate.setText(self.tr("Retake ({n} more ...)").format(n=remaining))
                self.status_label.setText(self.tr("Sample added! Capture {n}/{total}").format(
                    n=current_count + 1, total=self.cal_target_count,
                ))
        else:
            # We have enough samples, finalize
            self.status_label.setText(self.tr("Finalizing calibration... please wait."))
            self.btn_full_calibrate.setText(self.tr("Processing..."))
            self.btn_full_calibrate.setEnabled(False)
            QApplication.processEvents()

            success, mtx, dist, dpi, msg = self.calibrator.finalize_calibration()

            if success:
                h, w = frame.shape[:2]
                newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # Update persistent config/parameters
                save_calibration(mtx, dist, dpi, (h, w), new_mtx=newcameramtx)
                self.calibration = {"mtx": mtx, "dist": dist, "dpi": dpi, "resolution": (h, w), "new_mtx": newcameramtx}
                self.status_label.setText(self.tr("Full Calibration Success! DPI: {dpi:.1f}").format(dpi=dpi))

                # Notify workers to reload parameters
                num_signals = self.total_workers + 2
                if self.processing_queue:
                    for _ in range(num_signals):
                        self.processing_queue.put(('reload_params',))
            else:
                self.status_label.setText(self.tr("Full Calibration Failed: {msg}").format(msg=msg))

            # Reset state
            self.is_calibrating = False
            self.btn_full_calibrate.setText(self.tr("Full Calibration"))
            self.btn_full_calibrate.setEnabled(True)

    def calibrate_dpi(self):
        """Open the medium DPI calibration dialog. Live preview + card
        auto-detect; two action buttons (capture-and-refine, trace-
        manually) close the dialog as soon as a calibration is
        committed."""
        if self.webcam_thread is None:
            self.status_label.setText(self.tr("No webcam available."))
            return
        existing = getattr(self, "_dpi_dialog", None)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        from lib.gui.CalibrationDialogs import DpiCalibrationDialog
        from lib.workers.CreditCardDPI import ID1_LONG_MM, ID1_SHORT_MM
        dlg = DpiCalibrationDialog(
            self.webcam_thread,
            id1_long_mm=ID1_LONG_MM, id1_short_mm=ID1_SHORT_MM,
            parent=self,
        )
        dlg.calibration_committed.connect(self._on_dpi_calibration_committed)
        dlg.finished.connect(lambda _r: setattr(self, "_dpi_dialog", None))
        self._dpi_dialog = dlg
        dlg.show()

    def _on_dpi_calibration_committed(self, dpi: float, base_dpi: float,
                                      zoom: float, frame_bgr, _quad) -> None:
        ref_shape = frame_bgr.shape
        if self.calibration:
            mtx = self.calibration["mtx"]
            dist = self.calibration["dist"]
            new_mtx = self.calibration.get("new_mtx")
            res = self.calibration["resolution"]
        else:
            h, w = ref_shape[:2]
            mtx = np.eye(3)
            dist = np.zeros((1, 5))
            new_mtx = None
            res = [h, w]
        save_calibration(mtx, dist, dpi, res, new_mtx=new_mtx,
                         base_dpi=base_dpi, zoom_at_capture=zoom)
        if self.calibration is None:
            self.calibration = {
                "mtx": mtx, "dist": dist, "dpi": dpi,
                "resolution": res, "new_mtx": new_mtx,
                "base_dpi": base_dpi, "zoom_at_capture": zoom,
            }
        else:
            self.calibration["dpi"] = dpi
            self.calibration["base_dpi"] = base_dpi
            self.calibration["zoom_at_capture"] = zoom

        num_signals = self.total_workers + 2
        if self.processing_queue:
            for _ in range(num_signals):
                self.processing_queue.put(('reload_params',))
        self.status_label.setText(
            self.tr(
                "DPI calibrated: base {base:.1f} @1.0x (now {now:.1f})"
            ).format(base=base_dpi, now=self.effective_dpi())
        )
        self.toast(self.tr("DPI calibrated — {dpi:.0f} dpi.").format(dpi=self.effective_dpi()))
        self._refresh_dpi_readout()
        # The dialog auto-closes on `accept()`; nothing more to do here.


    def effective_dpi(self) -> float:
        """Current DPI scaled by the camera's live zoom factor."""
        if not self.calibration:
            if self._manual_dpi is not None:
                return float(self._manual_dpi)
            return float(self.args.options["general"].get("input_dpi", 100.0))
        base = self.calibration.get("base_dpi")
        if base is None:
            return float(self.calibration.get("dpi") or 100.0)
        zoom = 1.0
        if self.webcam_thread is not None:
            zoom = float(getattr(self.webcam_thread, "current_zoom", 1.0))
        return float(base) * zoom

    # ── Pipeline editor ────────────────────────────────────────────────
    def open_fix_dpi_tab(self):
        """Open the Fix-input-DPI scans table as a closable main tab
        (singleton — re-clicking focuses the existing one)."""
        if getattr(self, "_fix_dpi_tab", None) is not None:
            self.tabs.setCurrentWidget(self._fix_dpi_tab)
            return

        def _reprocess(scan_ids: set) -> None:
            cb = self._reprocess_snaps_callback
            if cb is None or not scan_ids:
                return
            cb(set(scan_ids))
            self.status_bar_widget.progress.reset()

        from lib.gui.ScansDpiTab import ScansDpiTab
        from lib.gui.theme import lucide as _lucide_tab
        tab = ScansDpiTab(str(self.db_path), self.thumb_loader, _reprocess)
        self._fix_dpi_tab = tab
        idx = self.tabs.addTab(tab, _lucide_tab("search-alert", size=14),
                               self.tr("Fix input DPI"))
        self.tabs.setCurrentIndex(idx)

    def open_pipeline_editor(self):
        """Open the pipeline editor as a closable tab in the right-side
        strip. Only one such tab is allowed: re-clicking the action just
        focuses the existing one."""
        if not self.pipeline_yaml_path:
            QMessageBox.warning(self, self.tr("Pipeline editor"),
                                self.tr("No pipeline yaml path was set for this session."))
            return
        # Singleton: focus existing tab if it's already open.
        if getattr(self, "_pipeline_editor_tab", None) is not None:
            self.tabs.setCurrentWidget(self._pipeline_editor_tab)
            return
        try:
            initial_yaml = Path(self.pipeline_yaml_path).read_text()
        except Exception as e:
            QMessageBox.warning(self, self.tr("Pipeline editor"),
                                self.tr("Could not load pipeline: {err}").format(err=e))
            return
        default_scan_id: Optional[int] = None
        try:
            ids = sorted(self.scan_widgets_by_scan.keys())
            if ids:
                default_scan_id = int(ids[0])
        except Exception:
            default_scan_id = None
        from lib.gui.PipelineEditorWidget import PipelineEditorTab
        from lib.gui.theme import lucide as _lucide_tab
        tab = PipelineEditorTab(
            initial_yaml, allow_reprocess=True,
            db_path=str(self.db_path) if self.db_path else None,
            default_scan_id=default_scan_id,
        )
        tab.apply_requested.connect(self._on_pipeline_tab_apply)
        tab.cancel_requested.connect(self._close_pipeline_tab)
        idx = self.tabs.addTab(tab, _lucide_tab("sliders", size=14), self.tr("Edit pipeline"))
        self.tabs.setTabToolTip(idx, self.tr("Edit pipeline"))
        self.tabs.setCurrentIndex(idx)
        self._pipeline_editor_tab = tab

    def _on_pipeline_tab_apply(self, new_yaml: str, reprocess: bool) -> None:
        if self._apply_pipeline_callback is None:
            QMessageBox.information(self, self.tr("Pipeline editor"),
                                    self.tr("Pipeline saved but no live-swap callback is wired."))
            Path(self.pipeline_yaml_path).write_text(new_yaml)
            self._close_pipeline_tab()
            return
        self.status_label.setText(
            self.tr("Applying pipeline + reprocessing…") if reprocess
            else self.tr("Applying pipeline…")
        )
        # Only seed the progress bar / spinners when there's actual work
        # coming. Without reprocess the chain is swapped but no scans
        # get re-enqueued — the spinner overlay would lie about activity.
        if reprocess:
            self.status_bar_widget.progress.set_label_prefix(self.tr("Pipeline"))
            self.status_bar_widget.progress.reset()
            n_active = len(self.scan_widgets_by_scan)
            if n_active:
                self.status_bar_widget.progress.set_imported(n_active)
            for w in self.scan_widgets_by_scan.values():
                w.set_processing(True)
        try:
            self._apply_pipeline_callback(new_yaml, reprocess)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Pipeline editor"),
                                 self.tr("Apply failed: {err}").format(err=e))
            return
        self.status_label.setText(self.tr("Pipeline updated."))
        self.toast(self.tr("Pipeline applied. Reprocessing started.") if reprocess
                   else self.tr("Pipeline applied."))
        self._close_pipeline_tab()
        # Reprocess path wipes ocr_runs.node_id + marks is_stale=1 in
        # `reprocess_active_scans` — broadcast so badges turn yellow now.
        if reprocess:
            self.ocr_state_changed.emit()

    def _close_pipeline_tab(self) -> None:
        tab = getattr(self, "_pipeline_editor_tab", None)
        if tab is None:
            return
        try:
            tab.stop_preview_worker()
        except Exception:
            pass
        idx = self.tabs.indexOf(tab)
        if idx >= 0:
            self.tabs.removeTab(idx)
        tab.deleteLater()
        self._pipeline_editor_tab = None

    def update_pipeline_context(self, *, processing_queue, pipeline_version_id: int,
                                 reprocess: bool = False):
        """Called by the entry script after a chain rebuild — wires the new queue.

        `reprocess=True` means scans will be re-enqueued shortly; only
        in that case do we seed the progress bar + spinners. A plain
        apply (no reprocess) leaves the existing UI state alone."""
        self.processing_queue = processing_queue
        self.pipeline_version_id = pipeline_version_id
        # Pipeline yaml may have flipped the `norm_width` flag; re-evaluate.
        self.refresh_norm_widths_visibility()
        # Rebuild step list from the (possibly edited) yaml — denominator
        # used by ScanItemWidget headers must match the new chain length.
        try:
            pdef = load_pipeline_def(Path(self.pipeline_yaml_path))
        except Exception:
            pdef = None
        if pdef:
            steps = []
            proc_names: list[str] = []
            for i, step in enumerate(pdef.get("pipeline", []), 1):
                sname = step.get("name", step.get("processor"))
                steps.append(f"{i:02d}_{slugify(sname, separator='_')}")
                proc_names.append(step.get("processor") or sname)
            import os as _os_replay
            replay_on = bool(pdef.get("replay", True))
            if (steps and replay_on
                    and not _os_replay.environ.get("AGLAIA_NO_REPLAY")):
                steps.append(f"{len(steps) + 1:02d}_replay")
                proc_names.append("Replay")
            self.pipeline_steps = steps
            self.pipeline_proc_names = proc_names
            self.pipeline_length = len(steps)
            self.pipeline_descriptions = pipeline_step_descriptions(pdef)
            # Reseed the sidebar's step list + drop stale samples so
            # the rows match the new pipeline.
            try:
                self._pipeline_tab.clear_timing()
                self._pipeline_tab.set_steps(self.pipeline_proc_names,
                                              self.pipeline_descriptions)
            except Exception:
                pass
        if not reprocess:
            return
        # Reprocess re-fires branch_ready but NOT scan_imported (scans exist).
        # Reseed totals from on-screen widget count, not list_active() — DB
        # orphans (root_node_id IS NULL) never produce a widget and would stall
        # the bar at "N-1/N".
        n_active = len(self.scan_widgets_by_scan)
        self.status_bar_widget.progress.reset()
        if n_active:
            self.status_bar_widget.progress.set_imported(n_active)
        # All existing scan widgets are about to be re-fed through the
        # new pipeline → spin them up again. branch_ready clears each
        # one as its replay finishes.
        for w in self.scan_widgets_by_scan.values():
            w.set_processing(True)

    # ── status bar wiring ──────────────────────────────────────────────
    def _on_log_line(self, level: str, text: str):
        """Show in the status bar's compact log strip and append to any
        open log tab. ProcessMonitor's internal buffer keeps history for
        late-opened tabs.

        Lines emitted by Qt-thread workers (OcrWorker.log_line) skip the
        multiprocessing log_queue, so we ALSO need to push them into the
        rolling buffer here — otherwise a later-opened Log tab seeds
        from a buffer that's missing every OCR error / engine print."""
        self.status_bar_widget.log.push(level, text)
        try:
            buf = self.monitor_thread.log_buffer
            buf.append(f"[{level.upper()}] {text}")
        except Exception:
            pass
        if self._log_tab is not None:
            self._log_tab.append(level, text)

    def _on_status_scan_imported(self, _payload: dict):
        # Pipeline activity resumed — make sure the bar header reads
        # "Pipeline" again even if the previous run was an OCR pass.
        self.status_bar_widget.progress.set_label_prefix(self.tr("Pipeline"))
        sid = _payload.get("scan_id") if isinstance(_payload, dict) else None
        self.status_bar_widget.progress.note_imported(sid)
        self._update_ocr_frame_state()

    def _on_status_branch_ready(self, payload: dict):
        scan_id = payload.get("scan_id")
        self.status_bar_widget.progress.mark_done(scan_id)
        # Clear the in-flight spinner / dim on the matching scan widget.
        # A multi-branch scan reaches branch_ready once per branch, but
        # `set_processing(False)` is idempotent so extra fires are cheap.
        if scan_id is not None:
            w = self.scan_widgets_by_scan.get(int(scan_id))
            if w is not None:
                w.set_processing(False)   # per-scan spinner clear — immediate
        # The expensive bits — OCR badge fan-out (DB queries) + a full
        # ScansTableView.refresh() (rebuild every block, re-decode every
        # thumb) — are THROTTLED to ~1 per 150 ms. A 100-page × 8-branch
        # batch fires branch_ready hundreds of times; refreshing per event
        # was a GUI-thread storm. Leading-edge + re-arm = at most one refresh
        # per window, with a final one ~150 ms after the last branch.
        t = getattr(self, "_batch_refresh_timer", None)
        if t is None:
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(150)
            t.timeout.connect(self._do_batch_refresh)
            self._batch_refresh_timer = t
        if not t.isActive():
            t.start()
        # Live-OCR: a freshly-finished branch is fair game for auto-OCR
        # once the 10 s grace window elapses (gives the user time to
        # delete the scan first). The debounce timer collapses several
        # branch_ready events in a row into a single OCR pass.
        self._maybe_schedule_live_ocr()

    def _do_batch_refresh(self) -> None:
        """Coalesced badge + alt-view refresh (throttled branch_ready storm)."""
        self.ocr_state_changed.emit()
        self._refresh_alt_views_if_visible()

    def _scan_added_to_views(self, scan_id: int) -> None:
        """One scan appeared (capture / import) → add its row to the visible
        alt view only. Grid already has the live widget; hidden views rebuild
        lazily on `set_view_mode`. Incremental so a 300+ scan table doesn't
        rebuild on every photo."""
        mode = getattr(self, "_view_mode", "grid")
        if mode == "list" and self._scans_table is not None:
            try:
                self._scans_table.add_snap(int(scan_id))
            except Exception:
                pass
        elif mode == "gallery" and self._scans_gallery is not None:
            try:
                self._scans_gallery.reload()
            except Exception:
                pass

    def _scan_removed_from_views(self, scan_id: int) -> None:
        """One scan deleted → drop its row from the visible alt view only.
        Grid widget is removed by the caller; hidden views rebuild lazily."""
        mode = getattr(self, "_view_mode", "grid")
        if mode == "list" and self._scans_table is not None:
            try:
                self._scans_table.remove_snap(int(scan_id))
            except Exception:
                pass
        elif mode == "gallery" and self._scans_gallery is not None:
            try:
                self._scans_gallery.reload()
            except Exception:
                pass

    def _refresh_alt_views_if_visible(self) -> None:
        """Push fresh data into whichever view is currently shown.
        Grid uses the live widgets, no refresh needed."""
        mode = getattr(self, "_view_mode", "grid")
        if mode == "list" and self._scans_table is not None:
            self._scans_table.refresh()
        elif mode == "gallery" and self._scans_gallery is not None:
            # Preserve the user's current focus across background events
            # (scan_imported, branch_ready). Yanking them to the newest
            # scan mid-review is annoying. `reload()` without
            # `jump_to_latest` keeps `prev_scan` + `prev_stage`.
            self._scans_gallery.reload()

    # ── live-OCR debounced scheduler ──────────────────────────────────

    LIVE_OCR_GRACE_MS = 10_000
    """Wall-time grace after the last branch_ready before the auto-OCR
    fires. Lets the user delete an unwanted scan before any OCR token
    cost is spent. Restarts on every fresh branch_ready, so a burst of
    8 pages produces a single batched run 10 s after the last one."""

    def _on_ocr_engine_changed(self, new_engine: str) -> None:
        """Flip every done OCR row produced by a different engine to
        ``is_stale = 1``. Triggered when the user picks a new engine
        in the sidebar — the badges + Run-OCR-default mode then treat
        old rows as needing re-OCR, which matches the user's intent
        of "switching engines invalidates the prior output"."""
        if not new_engine:
            return
        try:
            with db_session(str(self.db_path)) as conn:
                n = OcrRepo(conn).mark_stale_for_engine_switch(new_engine)
        except Exception as e:
            self._on_log_line(
                "error", f"engine switch → stale-update failed: {e}"
            )
            return
        if n > 0:
            self._on_log_line(
                "info",
                f"engine → {new_engine}: marked {n} previous OCR "
                "row(s) as stale",
            )
        self.ocr_state_changed.emit()
        self._refresh_alt_views_if_visible()

    def _on_live_ocr_toggled(self, on: bool) -> None:
        """Wired from OcrTab's checkbox. When the user flips off mid-
        debounce, cancel any pending timer so we don't surprise them
        with a delayed OCR pass."""
        if on:
            # The OCR pass is kicked from ``_maybe_schedule_live_ocr``
            # which is called on every branch_ready. Nothing to do here
            # except make sure no stale timer survives a toggle cycle.
            return
        timer = getattr(self, "_live_ocr_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _maybe_schedule_live_ocr(self) -> None:
        ocr_tab = getattr(self, "_ocr_tab", None)
        if ocr_tab is None or not ocr_tab.is_live_ocr_on():
            return
        # Lazy-init: keep one shared timer + restart on every fresh
        # event so we batch within the grace window.
        timer = getattr(self, "_live_ocr_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._fire_live_ocr)
            self._live_ocr_timer = timer
        timer.start(self.LIVE_OCR_GRACE_MS)

    def _fire_live_ocr(self) -> None:
        """Run an OCR pass only on branches that still have no OCR
        history. Concurrent with pipeline workers — Apple Vision is
        cheap; Surya/Paddle's extra ~700 MB of weights fits the M4
        16 GB unified mem budget alongside the chain workers. Only
        bails when another OCR worker is already in flight, so we
        don't stomp it; the next branch_ready re-arms the timer."""
        ocr_tab = getattr(self, "_ocr_tab", None)
        if ocr_tab is None or not ocr_tab.is_live_ocr_on():
            return
        if (self._ocr_worker is not None
                and self._ocr_worker.isRunning()):
            # Another OCR pass is mid-flight — re-arm. When it ends,
            # the next branch_ready will trigger a fresh debounce.
            self._maybe_schedule_live_ocr()
            return
        # Mode MISSING = only branches without any prior OCR. Avoids
        # eating cycles re-OCRing pages the user kept from a stale pass.
        engine = ocr_tab.engine_group.current_key() or "apple_vision"
        langs = list(ocr_tab.lang_input.tags() or [])
        comp = ocr_tab.current_complement()
        self._on_ocr_run_requested(engine, langs, OcrWorker.MODE_MISSING, comp)

    def _pipeline_idle(self) -> bool:
        """True iff no pipeline worker is currently processing a scan
        and no OCR worker is in flight. Drives the OCR run button's
        enabled state — pipeline + OCR are mutually exclusive.

        Uses scan-widget processing state rather than bar counts so the
        gate doesn't lock up after edge cases like an OCR run with
        nothing to do (which leaves the bar at 0/0)."""
        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            return False
        for w in self.scan_widgets_by_scan.values():
            if w.is_processing():
                return False
        return True

    def _update_ocr_frame_state(self) -> None:
        if not hasattr(self, "ocr_frame"):
            return
        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            self.ocr_frame.set_ocr_running(True)
            self._update_stop_btn_state(False)
            self._update_ocr_stop_btn_state(True)
            return
        pipeline_running = not self._pipeline_idle()
        self.ocr_frame.set_pipeline_running(pipeline_running)
        self._update_stop_btn_state(pipeline_running)
        self._update_ocr_stop_btn_state(False)
        # DB-derived pending counts → keeps the OCR frame in sync with
        # whatever the badges show. Single source of truth.
        if not pipeline_running:
            try:
                with db_session(str(self.db_path)) as conn:
                    state = OcrRepo(conn).branch_status_map()
                missing = sum(1 for v in state.values() if v["state"] == "none")
                stale = sum(1 for v in state.values() if v["state"] == "stale")
                self.ocr_frame.set_pending_count(missing, stale)
            except Exception:
                pass

    def _update_stop_btn_state(self, running: bool) -> None:
        bar = getattr(self, "status_bar_widget", None)
        if bar is None:
            return
        bar.set_pipeline_running(bool(running))

    def _update_ocr_stop_btn_state(self, running: bool) -> None:
        bar = getattr(self, "status_bar_widget", None)
        if bar is None:
            return
        bar.set_ocr_running(bool(running))

    def _on_stop_ocr_clicked(self) -> None:
        w = self._ocr_worker
        if w is None or not w.isRunning():
            return
        try:
            w.cancel()
            self.status_label.setText(self.tr("Stopping OCR…"))
        except Exception:
            pass

    def _on_force_rerun_clicked(self) -> None:
        if self._force_reprocess_callback is None:
            self.toast(self.tr("Force rerun not wired in this build."))
            return
        if not self.scan_widgets_by_scan:
            self.toast(self.tr("No scans to reprocess."))
            return
        n = len(self.scan_widgets_by_scan)
        msg = self.tr(
            "Reprocess every active scan ({n})?\n\n"
            "This wipes every branch + intermediate node for those "
            "scans (including any page selection) and re-enqueues "
            "the raw inputs."
        ).format(n=n)
        reply = QMessageBox.question(
            self, self.tr("Force rerun"), msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._force_reprocess_callback()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Force rerun"),
                                 self.tr("Failed: {err}").format(err=e))
            return
        # Reseed progress bar + spinners — workers run in background.
        self.status_bar_widget.progress.reset()
        self.status_bar_widget.progress.set_imported(n)
        for w in self.scan_widgets_by_scan.values():
            w.set_processing(True)
        self.toast(self.tr("Reprocessing {n} scan(s)…").format(n=n))
        # Reprocess wipe writes `is_stale = 1` to every ocr_run touched.
        # Broadcast now so badges + the OCR frame's pending count refresh
        # immediately, not only after the first branch_ready.
        self.ocr_state_changed.emit()

    def _on_stop_pipeline_clicked(self) -> None:
        if self._stop_pipeline_callback is None:
            self.toast(self.tr("Stop not wired in this build."))
            return
        try:
            dropped = self._stop_pipeline_callback()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Stop pipeline"),
                                 self.tr("Failed: {err}").format(err=e))
            return
        # Workers terminated their scan mid-flight — spinners would
        # otherwise spin forever. Clear them.
        for w in self.scan_widgets_by_scan.values():
            w.set_processing(False)
        self.status_bar_widget.progress.reset()
        try:
            n_drop = int(dropped or 0)
        except Exception:
            n_drop = 0
        self.toast(self.tr("Pipeline stopped. Dropped {n} queued item(s).").format(n=n_drop))
        self._update_ocr_frame_state()

    # ── OCR worker glue ─────────────────────────────────────────────────
    def _on_ocr_run_requested(self, engine: str, languages: list, mode: str,
                              complement: str = ""):
        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            return
        self._ocr_worker = OcrWorker(
            db_path=str(self.db_path), engine_name=engine,
            languages=list(languages), mode=mode, complement=complement,
        )
        self._ocr_worker.started_total.connect(self._on_ocr_started_total)
        self._ocr_worker.progress_scan.connect(self._on_ocr_progress)
        self._ocr_worker.log_line.connect(self._on_log_line)
        self._ocr_worker.finished_ok.connect(self._on_ocr_finished)
        self._update_ocr_frame_state()
        self._ocr_worker.start()

    def _on_ocr_started_total(self, total: int):
        # `total == 0` means "nothing to do" — don't repaint the bar as
        # `OCR · 0/0` in that case; leave the previous (pipeline / OCR)
        # final state visible and let the toast / log carry the message.
        if total <= 0:
            return
        bar = self.status_bar_widget.progress
        bar.set_label_prefix(self.tr("OCR"))
        bar.reset()
        bar.set_imported(total)
        # No per-page ticks until the first result lands (whole-doc Cloud
        # OCR, or the slow first Surya/Paddle batch) — show an activity
        # animation instead of a stuck 0%. The first tick flips it off.
        bar.set_indeterminate(True, self.tr("OCR · working…"))

    def _on_ocr_progress(self, scan_id: int):
        # OCR completes per (scan, branch). Single broadcast → all views
        # + bottom-bar OCR frame resync from the single source of truth.
        self.status_bar_widget.progress.set_indeterminate(False)
        self.status_bar_widget.progress.mark_tick()
        self.ocr_state_changed.emit()
        self._refresh_alt_views_if_visible()

    def _on_ocr_finished(self, ok: bool, error_text: str):
        # Leave the label on "OCR" so the user sees "OCR · N/N · 100%"
        # after the pass settles. scan_imported / pipeline reprocess flip
        # it back to "Pipeline" when new work arrives.
        # Stop any activity animation (e.g. a failed cloud run that never
        # ticked).
        self.status_bar_widget.progress.set_indeterminate(False)
        # Single broadcast → badges + bottom-bar resync.
        self.ocr_state_changed.emit()
        self._refresh_alt_views_if_visible()
        if not ok and error_text:
            self._on_log_line("error", f"OCR finished with error: {error_text}")
        elif ok and error_text:
            # ok=True with text = an advisory (e.g. Cloud OCR truncation).
            # Surface it prominently so the user knows to re-run.
            self._on_log_line("warning", error_text)

    def _refresh_ocr_ui(self) -> None:
        """Re-query OCR state per branch and push into scan widgets +
        export controls. Called on OCR finish and on app load."""
        try:
            with db_session(str(self.db_path)) as conn:
                state = OcrRepo(conn).branch_status_map()
        except Exception:
            return
        # Group by scan_id → pick the dominant state. Order: fresh > stale > none.
        snap_state: dict[int, str] = {}
        for (scan_id, _bp), info in state.items():
            cur = snap_state.get(scan_id)
            new = info["state"]
            if cur is None or _ocr_rank(new) > _ocr_rank(cur):
                snap_state[scan_id] = new
        for sid, w in self.scan_widgets_by_scan.items():
            s = snap_state.get(sid, "none")
            if hasattr(w, "set_ocr_state"):
                w.set_ocr_state(s)
            # Per-stem state lookup so the badge for layout A can read
            # "stale" while B reads "fresh" within the same scan.
            if hasattr(w, "set_ocr_state_per_stem"):
                raw_stem = getattr(w, "raw_filestem", "")
                per_stem: dict[str, str] = {}
                items = getattr(w, "items", {}) or {}
                for (msid, bp), info in state.items():
                    if int(msid) != int(sid):
                        continue
                    leaf = str(bp).split(".")[-1] if bp else ""
                    if leaf:
                        cand = f"{raw_stem}_{leaf}"
                        if cand in items:
                            per_stem[cand] = str(info.get("state", "none"))
                    else:
                        # Empty branch_path → apply to every non-root
                        # stem (single-layout scans land here).
                        for stem in items.keys():
                            if stem != raw_stem and stem not in per_stem:
                                per_stem[stem] = str(info.get("state", "none"))
                w.set_ocr_state_per_stem(per_stem)
        # Export controls — Markdown card + OCR layer toggle. Stale OCR
        # still counts: the text is correct relative to the OCR run, and
        # users would rather have a slightly-out-of-sync overlay than no
        # overlay at all. Only "missing" blocks.
        has_any = any(v in ("fresh", "stale") for v in snap_state.values())
        self._ocr_has_any = has_any
        self._refresh_ocr_layer_toggle()
        if hasattr(self, "_export_tab"):
            self._export_tab.set_markdown_available(has_any)

    def _refresh_ocr_layer_toggle(self) -> None:
        """Mirror OCR availability onto the PDF card's OCR-layer
        checkbox. Auto-checks on the enable transition: when OCR becomes
        available the layer is what the user almost always wants."""
        if not hasattr(self, "_export_tab"):
            return
        has_any = bool(getattr(self, "_ocr_has_any", False))
        self._export_tab.set_ocr_layer_available(has_any)

    def _clamp_to_screen(self):
        """Force the window back to maximised inside the current screen's
        available geometry. Called after show + on resize so a child
        layout asking for more space can't push the window off-screen."""
        scr = self.screen() if hasattr(self, "screen") else None
        if scr is None:
            from PySide6.QtGui import QGuiApplication
            scr = QGuiApplication.primaryScreen()
        if scr is None:
            return
        avail = scr.availableGeometry()
        self.setMaximumSize(avail.width(), avail.height())
        if self.width() > avail.width() or self.height() > avail.height():
            self.setGeometry(avail)

    def _on_card_dropped(self, scan_id: int, target_idx: int):
        """Card dragged + released over the grid. Compute a page_order
        value as the mean of the new neighbours' page_order, persist it,
        and move the widget in the FlowLayout to match."""
        widget = self.scan_widgets_by_scan.get(int(scan_id))
        if widget is None:
            return
        # Find the widget's current index, skip the no-op cases (drop on
        # itself or on either side of itself within the same slot).
        cur_idx = -1
        for i in range(self.scroll_layout.count()):
            if self.scroll_layout.itemAt(i).widget() is widget:
                cur_idx = i
                break
        if cur_idx == -1:
            return
        if target_idx in (cur_idx, cur_idx + 1):
            return

        # Translate flow-layout index → DB neighbours. After the move the
        # card will sit between `prev_w` and `next_w` in display order, so
        # take the mean of their page_order values for the new value. If
        # we're at one end, offset by ±1 (no underflow concern — float).
        order_for_idx = {}
        for i in range(self.scroll_layout.count()):
            w = self.scroll_layout.itemAt(i).widget()
            sid = getattr(w, "scan_id", None)
            if sid is not None:
                order_for_idx[i] = sid

        # Build the post-move sequence of scan_ids (skip the moving card,
        # then insert it at target_idx).
        ids = [order_for_idx[i] for i in range(self.scroll_layout.count())
               if i in order_for_idx]
        ids = [s for s in ids if s != int(scan_id)]
        insert_at = target_idx if target_idx <= cur_idx else target_idx - 1
        insert_at = max(0, min(insert_at, len(ids)))
        ids.insert(insert_at, int(scan_id))

        # Load current page_order for each surviving sibling — needed to
        # average between neighbours.
        with db_session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, page_order FROM scans WHERE deleted_at IS NULL"
            ).fetchall()
            order_by_id = {int(r["id"]): (float(r["page_order"])
                                          if r["page_order"] is not None
                                          else float(r["id"]))
                           for r in rows}
            prev_sid = ids[insert_at - 1] if insert_at > 0 else None
            next_sid = ids[insert_at + 1] if insert_at + 1 < len(ids) else None
            prev_o = order_by_id.get(prev_sid) if prev_sid is not None else None
            next_o = order_by_id.get(next_sid) if next_sid is not None else None
            if prev_o is not None and next_o is not None:
                new_order = (prev_o + next_o) / 2.0
            elif prev_o is not None:
                new_order = prev_o + 1.0
            elif next_o is not None:
                new_order = next_o - 1.0
            else:
                new_order = 1.0
            ScanRepo(conn).set_page_order(int(scan_id), new_order)

        # Move the widget in the FlowLayout.
        self.scroll_layout.removeWidget(widget)
        self.scroll_layout.insertWidget(insert_at, widget)
        self.scroll_content.updateGeometry()
        # Reorder is a scan-set mutation — table + gallery (which both
        # enumerate scans in sorted order) need to pick up the new
        # `page_order`. Cheap full rebuild = guaranteed in sync.
        self._broadcast_scan_set_changed()

    def _ocr_branch_state_map(self) -> dict:
        """Per-branch OCR state for `ScansTableView` row badges. Single
        DB query; format matches `OcrRepo.branch_status_map`."""
        try:
            with db_session(str(self.db_path)) as conn:
                return OcrRepo(conn).branch_status_map()
        except Exception:
            return {}

    # ── ScansGalleryView providers ────────────────────────────────
    def _gallery_snaps(self) -> list[tuple[int, str]]:
        """Ordered (scan_id, raw_filestem) — vertical axis of the gallery."""
        out: list[tuple[int, str]] = []
        for sid in sorted(self.scan_widgets_by_scan.keys()):
            w = self.scan_widgets_by_scan[sid]
            out.append((int(sid), str(getattr(w, "raw_filestem", f"scan-{sid}"))))
        return out

    def _gallery_stages(self) -> list[str]:
        """Pipeline stage names — horizontal axis of the gallery.

        `raw` first, then every step in the active pipeline, then
        `replay` if any node in the DB carries that step name."""
        stages = ["raw"] + list(self.pipeline_steps or [])
        # Detect replay nodes (post-pipeline composite pass).
        try:
            with db_session(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT 1 FROM nodes WHERE step_name = 'replay' LIMIT 1"
                ).fetchone()
                if row is not None and "replay" not in stages:
                    stages.append("replay")
        except Exception:
            pass
        return stages

    def _gallery_default_stage(self, scan_id: int) -> Optional[str]:
        """Return the scan's last user-chosen stage name (or None).

        Reads `branches.chosen_node_id → nodes.step_name`. With multiple
        branches (per-layout), all share the same chosen stage in
        practice — pick the first encountered."""
        try:
            with db_session(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT n.step_name FROM branches b "
                    "JOIN nodes n ON n.id = b.chosen_node_id "
                    "WHERE b.scan_id = ? "
                    "ORDER BY b.branch_path ASC LIMIT 1",
                    (int(scan_id),),
                ).fetchone()
                if row and row["step_name"]:
                    return str(row["step_name"])
        except Exception:
            return None
        return None

    def _gallery_branch_trashed(self, scan_id: int,
                                  branch_label: str) -> bool:
        """True iff the matching branches row has `trashed_at IS NOT NULL`.

        Uses the same single-page fallback as the writer (see
        `_resolve_branch_ids`) so a hidden one-page scan reads back hidden."""
        try:
            with db_session(str(self.db_path)) as conn:
                ids = self._resolve_branch_ids(conn, int(scan_id),
                                                str(branch_label))
                if not ids:
                    return False
                placeholders = ",".join("?" * len(ids))
                row = conn.execute(
                    "SELECT 1 FROM branches "
                    f"WHERE id IN ({placeholders}) AND trashed_at IS NOT NULL "
                    "LIMIT 1",
                    ids,
                ).fetchone()
                return bool(row)
        except Exception:
            return False

    def _gallery_set_branch_trashed(self, scan_id: int,
                                      branch_label: str, trashed: bool) -> None:
        """Gallery eye toggle → central writer (persists + broadcasts)."""
        self.set_branch_visibility(int(scan_id), str(branch_label),
                                     bool(trashed))

    def _broadcast_scan_set_changed(self) -> None:
        """Scan set mutated (deleted, reordered, imported) → rebuild only the
        currently-visible alt view. Hidden views are rebuilt lazily by
        `set_view_mode` when the user switches to them, so they never drift.

        Previously this rebuilt BOTH the table and the gallery unconditionally
        on every mutation — a full re-query + thumbnail re-decode of every
        scan. On a 300+ scan project a single voice "delete" froze the UI
        (beach ball) for 3-4 s while it rebuilt views the user wasn't even
        looking at. The grid (default view) uses live widgets and needs no
        refresh here at all."""
        try:
            self._refresh_alt_views_if_visible()
        except Exception:
            pass

    def _toggle_left_panel(self, show: bool) -> None:
        """Hide/show the scans-tab left panel. When hidden, an absolute
        "open" button overlays the corner so the user can bring it back."""
        if not hasattr(self, "_left_panel"):
            return
        self._left_panel.setVisible(bool(show))
        if hasattr(self, "_left_panel_open_btn"):
            self._left_panel_open_btn.setVisible(not show)
            if not show:
                self._left_panel_open_btn.raise_()

    def _on_ocr_state_changed(self) -> None:
        """Single fan-out for any OCR-relevant DB change. Refreshes
        badges across all views + the bottom-bar OCR frame counts."""
        try:
            self._refresh_ocr_ui()
        except Exception:
            pass
        try:
            self._update_ocr_frame_state()
        except Exception:
            pass

    # ── broadcast subscribers ─────────────────────────────────────
    def _on_step_disabled_changed(self, scan_id: int, branch_path: str,
                                     step_idx: int, disabled: bool) -> None:
        """A step was toggled → repaint the active view's stage state now.

        Optimistic: the disabled marker shows immediately; the scan's
        rerun (triggered by the same writer) repopulates thumbnails as
        nodes land, and the grid's live image events refresh those."""
        if getattr(self, "_view_mode", "grid") == "list":
            try:
                self._scans_table.refresh()
            except Exception:
                pass
        elif getattr(self, "_view_mode", "grid") == "gallery":
            try:
                self._scans_gallery.reload()
            except Exception:
                pass
        else:
            w = self.scan_widgets_by_scan.get(int(scan_id))
            if w is not None:
                try:
                    w.refresh_composite()
                except Exception:
                    pass

    def _on_branch_visibility_changed(self, scan_id: int, branch_label: str,
                                         hidden: bool) -> None:
        """DB trashed_at changed → resync all three views from DB."""
        # Grid: locate stem, set `trashed`, repaint.
        w = self.scan_widgets_by_scan.get(int(scan_id))
        if w is not None:
            raw_stem = getattr(w, "raw_filestem", "")
            items = getattr(w, "items", {}) or {}
            target_stem = None
            if branch_label:
                cand = f"{raw_stem}_{branch_label}"
                if cand in items:
                    target_stem = cand
                else:
                    for s in items.keys():
                        if s.split("_")[-1] == branch_label:
                            target_stem = s
                            break
            if target_stem is None:
                non_root = [s for s in items.keys() if s != raw_stem]
                target_stem = non_root[0] if non_root else raw_stem
            entry = items.get(target_stem)
            if entry is not None and entry.get("trashed", False) != hidden:
                entry["trashed"] = bool(hidden)
                try:
                    w.refresh_composite()
                except Exception:
                    pass
        # Table: full rebuild.
        if getattr(self, "_view_mode", "grid") == "list":
            try:
                self._scans_table.refresh()
            except Exception:
                pass
        # Gallery: `_present` if focused on this scan.
        if getattr(self, "_view_mode", "grid") == "gallery":
            try:
                if self._scans_gallery.focused_scan_id() == int(scan_id):
                    self._scans_gallery._present()
            except Exception:
                pass

    def _on_table_trash_requested(self, scan_id: int, stem: str,
                                     hidden: bool) -> None:
        """Table view eye toggle → central writer. The caller passes the
        desired new state so we don't need to re-query (the DB query
        misses single-page scans whose branches row uses ``branch_path =
        'A'`` instead of ``''``)."""
        w = self.scan_widgets_by_scan.get(int(scan_id))
        raw_stem = getattr(w, "raw_filestem", "") if w is not None else ""
        if stem and raw_stem and stem.startswith(raw_stem + "_"):
            suffix = stem[len(raw_stem) + 1:]
            branch_label = suffix.split("_")[-1] if "_" in suffix else suffix
        else:
            branch_label = ""
        self.set_branch_visibility(int(scan_id), str(branch_label),
                                     bool(hidden))

    # ── single source of truth: branch visibility ───────────────
    # (chosen_node_id is no longer user-writable — it tracks the rerun
    #  terminal; per-page output is shaped by `step_overrides`.)

    def set_branch_visibility(self, scan_id: int, branch_label: str,
                                 hidden: bool) -> None:
        """Authoritative writer for `branches.trashed_at`. Persists +
        broadcasts `branch_visibility_changed`."""
        try:
            with db_session(str(self.db_path)) as conn:
                set_clause = ("trashed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"
                              if hidden else "trashed_at = NULL")
                ids = self._resolve_branch_ids(conn, int(scan_id),
                                                str(branch_label))
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    conn.execute(
                        f"UPDATE branches SET {set_clause} WHERE id IN ({placeholders})",
                        ids,
                    )
                conn.commit()
        except Exception:
            return
        self.branch_visibility_changed.emit(int(scan_id), str(branch_label),
                                              bool(hidden))

    def _resolve_branch_ids(self, conn, scan_id: int,
                              branch_label: str) -> list[int]:
        """Branch-row ids matching (scan, page-label) for visibility/chosen.

        A layout step labels every page `A`, `B`, … — so even a scan with
        a *single* page has `branch_path = 'A'`, not `''`. The root-level
        toggle computes an empty label; without this fallback its
        `branch_path = ''` match finds nothing and the hide never
        persists. When the label is empty and there's no literal root
        row, target the scan's lone branch (only if exactly one exists)."""
        if branch_label:
            rows = conn.execute(
                "SELECT id FROM branches WHERE scan_id = ? "
                "  AND (branch_path = ? OR branch_path LIKE ?)",
                (int(scan_id), str(branch_label), f"%.{branch_label}"),
            ).fetchall()
            return [int(r["id"]) for r in rows]
        rows = conn.execute(
            "SELECT id FROM branches WHERE scan_id = ? "
            "  AND (branch_path = '' OR branch_path IS NULL)",
            (int(scan_id),),
        ).fetchall()
        if not rows:
            all_rows = conn.execute(
                "SELECT id FROM branches WHERE scan_id = ?", (int(scan_id),),
            ).fetchall()
            if len(all_rows) == 1:
                rows = all_rows
        return [int(r["id"]) for r in rows]

    # ── per-page processor disable ───────────────────────────────
    @staticmethod
    def _proc_toggleable(processor_name) -> bool:
        """True iff a processor may be per-page disabled — only linear
        COORDINATE / PIXEL_VALUE steps. ROI / branch-emitting (PageDetector)
        and untagged steps are locked: toggling them would restructure the
        branch tree."""
        try:
            from lib.processors.registry import get_processor
            from lib.processors.abstraction import ReplayTrait
            info = get_processor(processor_name)
            trait = getattr(info.processor_cls, "REPLAY_TRAIT", None) if info else None
            return trait in (ReplayTrait.COORDINATE, ReplayTrait.PIXEL_VALUE)
        except Exception:
            return False

    def node_toggleable(self, processor_name) -> bool:
        """Public alias for views deciding whether to show an active toggle."""
        return self._proc_toggleable(processor_name)

    def _node_override_key(self, conn, node_id: int):
        """(branch_path, step_idx, processor_name) for a node, or None.

        `branch_path` is the node's `branch_label` (or "" for pre-split
        trunk steps) — matches the worker's `current.branch_path` for the
        non-nested splits PageDetector produces. (Nested A.1 branches would
        need the full ancestry path; deferred — see issue #68.)"""
        row = conn.execute(
            "SELECT scan_id, branch_label, step_idx, processor_name "
            "FROM nodes WHERE id = ?", (int(node_id),),
        ).fetchone()
        if row is None:
            return None
        return (int(row["scan_id"]), str(row["branch_label"] or ""),
                int(row["step_idx"]), row["processor_name"])

    def step_disabled_for_node(self, node_id: int) -> bool:
        try:
            with db_session(str(self.db_path)) as conn:
                key = self._node_override_key(conn, int(node_id))
                if key is None:
                    return False
                scan_id, bp, sidx, _ = key
                return StepOverrideRepo(conn).is_disabled(scan_id, bp, sidx)
        except Exception:
            return False

    def disabled_steps_for_layout(self, scan_id: int, branch_path: str) -> set:
        """Disabled `step_idx`s visible on (scan, layout): trunk ("") +
        this branch. Used by the grid band + table strip to render state."""
        try:
            with db_session(str(self.db_path)) as conn:
                allset = StepOverrideRepo(conn).map_for_scan(int(scan_id))
            bp = str(branch_path or "")
            return {s for (b, s) in allset if b == "" or b == bp}
        except Exception:
            return set()

    def set_step_disabled(self, scan_id: int, node_id: int, disabled: bool) -> None:
        """Authoritative writer for `step_overrides`. Persists the toggle,
        broadcasts for instant cell feedback, then reruns the scan from raw
        (the worker re-applies the override per branch; `chosen_node_id`
        lands on the new terminal automatically). No-op for locked steps."""
        bp = ""
        sidx = -1
        try:
            with db_session(str(self.db_path)) as conn:
                key = self._node_override_key(conn, int(node_id))
                if key is None:
                    return
                ks, bp, sidx, pname = key
                if not self._proc_toggleable(pname):
                    return
                StepOverrideRepo(conn).set(int(scan_id), bp, sidx, bool(disabled))
                conn.commit()
        except Exception:
            return
        self.step_disabled_changed.emit(int(scan_id), bp, int(sidx), bool(disabled))
        # Rerun only the toggled page-branch (its sibling pages are unaffected).
        # `bp` is the branch path; reprocess_branch falls back to a whole-scan
        # rerun when the scan isn't split. If no branch callback is wired, fall
        # back to the scan-level reprocess.
        branch_cb = self._reprocess_branch_callback
        if branch_cb is not None:
            try:
                branch_cb(int(scan_id), bp)
                return
            except Exception:
                pass
        cb = self._reprocess_snaps_callback
        if cb is not None:
            try:
                cb({int(scan_id)})
            except Exception:
                pass

    def toggle_step_disabled(self, scan_id: int, node_id: int) -> None:
        """Flip a step's disabled state for its layout (view click handler)."""
        self.set_step_disabled(int(scan_id), int(node_id),
                                not self.step_disabled_for_node(int(node_id)))

    def cell_disable_states(self, scan_id: int) -> dict:
        """`{node_id: (toggleable, disabled)}` for every node of a scan —
        one query, so a view can render its whole stage strip without a
        per-cell DB round-trip."""
        out: dict[int, tuple[bool, bool]] = {}
        try:
            with db_session(str(self.db_path)) as conn:
                disabled = StepOverrideRepo(conn).map_for_scan(int(scan_id))
                rows = conn.execute(
                    "SELECT id, branch_label, step_idx, processor_name "
                    "FROM nodes WHERE scan_id = ?", (int(scan_id),),
                ).fetchall()
                for r in rows:
                    bp = str(r["branch_label"] or "")
                    out[int(r["id"])] = (
                        self._proc_toggleable(r["processor_name"]),
                        (bp, int(r["step_idx"])) in disabled,
                    )
        except Exception:
            pass
        return out

    def _on_card_visibility_changed(self, scan_id: int, branch_label: str,
                                      hidden: bool) -> None:
        self.set_branch_visibility(scan_id, branch_label, hidden)

    def _sync_grid_current_idx_from_chosen(self, scan_id: int,
                                            branch_label: str,
                                            node_id: int) -> None:
        """Push the gallery's new chosen-node selection into the matching
        grid ScanItemWidget so its mini-thumb cursor (`current_idx` per
        stem) tracks the DB without a re-spawn."""
        w = self.scan_widgets_by_scan.get(int(scan_id))
        if w is None:
            return
        # Resolve the new chosen node's step_name → its index in
        # `global_history`. Need DB lookup for step_name.
        try:
            with db_session(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT step_name FROM nodes WHERE id = ?", (int(node_id),)
                ).fetchone()
        except Exception:
            row = None
        if not row:
            return
        step_name = row["step_name"] or "raw"
        hist = getattr(w, "global_history", None) or []
        if step_name not in hist:
            return
        new_idx = hist.index(step_name)
        # Identify the stem from branch_label. branch_label might be the
        # leaf token "A"; stem in items uses `raw_filestem + "_A"`.
        raw_stem = getattr(w, "raw_filestem", "")
        items = getattr(w, "items", {}) or {}
        target_stem = None
        if branch_label:
            candidate = f"{raw_stem}_{branch_label}"
            if candidate in items:
                target_stem = candidate
            else:
                # Nested branch labels — find any stem ending in the leaf token.
                for s in items.keys():
                    if s.split("_")[-1] == branch_label:
                        target_stem = s
                        break
        if target_stem is None:
            # Single-layout case: walk the only non-root stem if any,
            # else fall through to the root.
            non_root = [s for s in items.keys() if s != raw_stem]
            target_stem = non_root[0] if non_root else raw_stem
        entry = items.get(target_stem)
        if entry is None:
            return
        entry["current_idx"] = new_idx
        # Repaint the card so the mini-thumb cursor + header read the
        # new selection straight away.
        try:
            w.refresh_composite()
            w.update_header()
        except Exception:
            pass

    def _gallery_stage_resolve(self, scan_id: int, stage: str
                                ) -> list[tuple[str, Optional[int], Optional[int]]]:
        """All layouts of `scan_id` at pipeline `stage`.

        Returns `[(layout_label, image_id, node_id), …]` — one entry
        per branch. `node_id` is the per-branch node id (the gallery's
        star button writes that as `chosen_node_id`)."""
        try:
            with db_session(str(self.db_path)) as conn:
                rows = conn.execute(
                    "SELECT id, branch_label, image_id "
                    "FROM nodes "
                    "WHERE scan_id = ? AND step_name = ? "
                    "ORDER BY branch_label IS NULL DESC, branch_label ASC, id ASC",
                    (int(scan_id), str(stage)),
                ).fetchall()
                if not rows:
                    if stage == "raw":
                        srow = conn.execute(
                            "SELECT root_node_id FROM scans WHERE id = ?",
                            (int(scan_id),),
                        ).fetchone()
                        if srow and srow["root_node_id"] is not None:
                            nrow = conn.execute(
                                "SELECT id, image_id FROM nodes WHERE id = ?",
                                (int(srow["root_node_id"]),),
                            ).fetchone()
                            if nrow and nrow["image_id"] is not None:
                                return [("", int(nrow["image_id"]),
                                         int(nrow["id"]))]
                    return []
                return [
                    (str(r["branch_label"] or ""),
                     int(r["image_id"]) if r["image_id"] is not None else None,
                     int(r["id"]))
                    for r in rows
                ]
        except Exception:
            return []

    def _install_thumb_size_slider(self):
        """Right corner of the tab row: view-mode toggle + zoom slider.
        Slider controls the card thumb max width (50–600px, step 50);
        toggle picks list / grid / gallery view of the scans tab.
        Corner widget hides on non-Scans tabs to avoid confusion."""
        from lib.gui.theme import lucide_pixmap
        host = QWidget()
        host.setFixedHeight(30)
        h = QHBoxLayout(host)
        h.setContentsMargins(8, 0, 8, 0)
        h.setSpacing(4)
        h.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        # View-mode button group.
        from lib.app_data import db as cfg
        try:
            with cfg.session() as conn:
                stored_mode = cfg.get(conn, cfg.KEY_VIEW_MODE, "grid")
        except Exception:
            stored_mode = "grid"
        if stored_mode not in ("list", "grid", "gallery"):
            stored_mode = "grid"
        self._view_mode = stored_mode
        self._view_btn_group = QButtonGroup(host)
        self._view_btn_group.setExclusive(True)
        from PySide6.QtCore import QSize as _QSize
        for mode, glyph, tip in (
            ("list", "list", self.tr("Compact table view")),
            ("grid", "layout-grid", self.tr("Card grid view")),
            ("gallery", "gallery-horizontal", self.tr("Full-size carousel")),
        ):
            btn = QToolButton()
            btn.setIcon(QPixmap(lucide_pixmap(glyph, color=COLOR_FONT_PLACEHOLDER, size=14)))
            btn.setIconSize(_QSize(14, 14))
            btn.setCheckable(True)
            # 26×26 leaves enough headroom for the 1-px border + 2-px
            # checked outline without clipping at the host's 30-px height.
            btn.setFixedSize(26, 26)
            btn.setToolTip(tip)
            btn.setStyleSheet(
                "QToolButton{background:transparent; border:1px solid transparent; "
                "border-radius:4px; padding:0px; margin:0px;}"
                f"QToolButton:hover{{background:{COLOR_BG_SURFACE_ALT};}}"
                f"QToolButton:checked{{background:{COLOR_PRIMARY_BG_SOFT}; border-color:{COLOR_PRIMARY};}}"
            )
            btn.setProperty("view_mode", mode)
            self._view_btn_group.addButton(btn)
            h.addWidget(btn, 0, Qt.AlignmentFlag.AlignVCenter)
            if mode == self._view_mode:
                btn.setChecked(True)
        self._view_btn_group.buttonClicked.connect(self._on_view_mode_clicked)
        # Initial view is applied lazily at the end of this method (once the
        # corner widgets exist) via `set_view_mode`, so a non-grid stored mode
        # builds its view and grid stays the only resident one otherwise.
        # Sync the View-menu radio (if the menu was built first).
        for _m, _a in getattr(self, "_view_mode_actions", {}).items():
            _a.setChecked(_m == self._view_mode)
        # Tiny gap before the zoom slider.
        sep = QLabel(" ")
        sep.setFixedWidth(8)
        h.addWidget(sep)
        icon_lbl = QLabel()
        # lucide_pixmap renders at 2× (32×32 here) for HiDPI sharpness;
        # tag the DPR so QLabel renders it at 16×16 logical pixels
        # instead of cropping to the top-left corner.
        _spx = lucide_pixmap("search", color=COLOR_FONT_PLACEHOLDER, size=16)
        _spx.setDevicePixelRatio(2.0)
        icon_lbl.setPixmap(_spx)
        icon_lbl.setFixedSize(16, 16)
        h.addWidget(icon_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(50)
        slider.setMaximum(600)
        slider.setSingleStep(50)
        slider.setPageStep(50)
        slider.setTickInterval(50)
        slider.setTickPosition(QSlider.TickPosition.NoTicks)
        slider.setFixedWidth(160)
        slider.setValue(int(self._card_max_width_px))
        slider.valueChanged.connect(self._on_thumb_slider_changed)
        # Slider natively renders ~22 px tall; pin to 20 + AlignVCenter so
        # it lands on the same baseline as the 26-px view buttons.
        slider.setFixedHeight(20)
        h.addWidget(slider, 0, Qt.AlignmentFlag.AlignVCenter)
        self._thumb_slider = slider
        # Gallery-only "Show selected" toggle. Lives in the same slot as
        # the slider; visibility flips with the active view mode.
        self._icon_lbl_search = icon_lbl
        from PySide6.QtWidgets import QCheckBox as _QCb
        self._chk_show_selected = _QCb(self.tr("Show selected"))
        self._chk_show_selected.setToolTip(
            self.tr(
                "Per-page view: each cell shows the starred (chosen) stage. "
                "Stage navigation is hidden."
            )
        )
        self._chk_show_selected.setStyleSheet(
            f"QCheckBox{{color:{COLOR_FONT_MUTED}; padding:0 4px;}}"
        )
        self._chk_show_selected.setVisible(False)
        self._chk_show_selected.toggled.connect(
            lambda on: (self._scans_gallery is not None
                        and self._scans_gallery.set_show_selected(bool(on)))
        )
        h.addWidget(self._chk_show_selected, 0, Qt.AlignmentFlag.AlignVCenter)
        self.tabs.setCornerWidget(host, Qt.Corner.TopRightCorner)
        self._tab_corner_host = host
        # Hide the corner on non-Scans tabs — both slider + view-mode
        # group are scans-tab-only controls.
        self.tabs.currentChanged.connect(self._on_tab_changed_corner)
        self._on_tab_changed_corner(self.tabs.currentIndex())
        # Initial slider-vs-checkbox visibility based on persisted view mode.
        self._sync_corner_mode_widgets()
        # Now that every corner widget exists, build the initial view lazily.
        # Grid stays the only resident view; a stored list/gallery mode builds
        # exactly one alt view.
        self.set_view_mode(getattr(self, "_view_mode", "grid"))

    def _on_tab_changed_corner(self, idx: int) -> None:
        host = getattr(self, "_tab_corner_host", None)
        if host is None:
            return
        host.setVisible(idx == 0)

    def _sync_corner_mode_widgets(self) -> None:
        """Show slider vs. 'Show selected' checkbox based on current view.
        Gallery hides the slider + magnifier; grid/list keep slider.
        Called on every view-mode change and at startup."""
        mode = getattr(self, "_view_mode", "grid")
        is_gallery = (mode == "gallery")
        for w in (self._thumb_slider, self._icon_lbl_search):
            w.setVisible(not is_gallery)
        # "Show selected" mode retired with exit-stage nav — always hidden.
        self._chk_show_selected.setVisible(False)

    def _on_view_mode_clicked(self, btn) -> None:
        self.set_view_mode(btn.property("view_mode"))

    def _ensure_table(self):
        """Lazily build the table view + wire it. Returns the live widget."""
        if self._scans_table is None:
            from lib.gui.ScansTableView import ScansTableView
            t = ScansTableView(
                get_snap_widgets=lambda: self.scan_widgets_by_scan,
                thumb_loader=self.thumb_loader,
                ocr_state_provider=self._ocr_branch_state_map,
                cell_states_provider=self.cell_disable_states,
            )
            t.delete_requested.connect(self.delete_scan)
            t.debug_requested.connect(self._open_debug_viewer)
            t.trash_requested.connect(self._on_table_trash_requested)
            t.step_toggle_requested.connect(self.toggle_step_disabled)
            # Drag-grip reorder reuses the grid card-drop handler.
            t.card_dropped.connect(self._on_card_dropped)
            # Seed the thumb-row height from the current zoom slider so a
            # freshly-built table matches the rest of the UI.
            try:
                snapped = int(self._card_max_width_px)
                t.set_thumb_height(16 + int(round((snapped - 50) / 550 * 80)))
            except Exception:
                pass
            self._scans_stack.addWidget(t)
            self._scans_table = t
        return self._scans_table

    def _ensure_gallery(self):
        """Lazily build the gallery view + wire it. Returns the live widget."""
        if self._scans_gallery is None:
            from lib.gui.ScansGalleryView import ScansGalleryView
            g = ScansGalleryView(
                scans_provider=self._gallery_snaps,
                stages_provider=self._gallery_stages,
                stage_resolver=self._gallery_stage_resolve,
                thumb_loader=self.thumb_loader,
                default_stage_provider=self._gallery_default_stage,
                cell_states_provider=self.cell_disable_states,
                step_toggle_writer=self.toggle_step_disabled,
                branch_trashed_provider=self._gallery_branch_trashed,
                branch_trashed_writer=self._gallery_set_branch_trashed,
            )
            g.debug_requested.connect(self._open_debug_viewer)
            self._scans_stack.addWidget(g)
            self._scans_gallery = g
        return self._scans_gallery

    def _destroy_alt_view(self, which: str) -> None:
        """Tear down the table/gallery widget to free its memory once the
        user navigates away. Grid is never destroyed."""
        if which == "list" and self._scans_table is not None:
            self._scans_stack.removeWidget(self._scans_table)
            self._scans_table.deleteLater()
            self._scans_table = None
        elif which == "gallery" and self._scans_gallery is not None:
            self._scans_stack.removeWidget(self._scans_gallery)
            self._scans_gallery.deleteLater()
            self._scans_gallery = None

    def set_view_mode(self, mode: str) -> None:
        """Switch the scans view (grid / list[=table] / gallery), keeping
        the toolbar toggles, the View menu, the stack and the persisted
        setting all in sync. Safe to call from any of them.

        Only the active alt view exists: switching builds the target lazily
        and destroys the view we're leaving (grid is the persistent base)."""
        if mode not in ("list", "grid", "gallery"):
            return
        self._view_mode = mode
        # Free whichever alt view we're NOT showing.
        if mode != "list":
            self._destroy_alt_view("list")
        if mode != "gallery":
            self._destroy_alt_view("gallery")
        # Build + show the target.
        if mode == "grid":
            self._scans_stack.setCurrentWidget(self._grid_widget)
        elif mode == "list":
            t = self._ensure_table()
            self._scans_stack.setCurrentWidget(t)
            t.refresh()
        elif mode == "gallery":
            g = self._ensure_gallery()
            self._scans_stack.setCurrentWidget(g)
            # Fresh build has no prior focus; land on the latest scan the
            # first time, otherwise honour the gallery's own fallback.
            already_visited = getattr(self, "_gallery_visited", False)
            g.reload(jump_to_latest=not already_visited)
            self._gallery_visited = True
        # Sync corner widgets (slider hides on gallery, "Show selected"
        # checkbox appears in its place).
        self._sync_corner_mode_widgets()
        # Sync the toolbar toggle buttons.
        grp = getattr(self, "_view_btn_group", None)
        if grp is not None:
            for b in grp.buttons():
                b.setChecked(b.property("view_mode") == mode)
        # Sync the View-menu radio items.
        for m, act in getattr(self, "_view_mode_actions", {}).items():
            act.setChecked(m == mode)
        # Persist.
        try:
            from lib.app_data import db as cfg
            with cfg.session() as conn:
                cfg.set(conn, cfg.KEY_VIEW_MODE, mode)
        except Exception:
            pass

    def _on_thumb_slider_changed(self, raw: int):
        # Scan to nearest multiple of 50 within [50, 600]. setValue here
        # re-emits valueChanged once but is a no-op when already scanned.
        snapped = max(50, min(600, int(round(raw / 50.0)) * 50))
        if snapped != raw:
            self._thumb_slider.blockSignals(True)
            self._thumb_slider.setValue(snapped)
            self._thumb_slider.blockSignals(False)
        if snapped == self._card_max_width_px:
            return
        self._card_max_width_px = snapped
        # Max width changed — clear global zoom; cards will re-seed it via
        # `final_zoom_observed` on the next refresh_composite call.
        self._global_zoom = None
        for w in self.scan_widgets_by_scan.values():
            w.set_global_zoom(None)
            w.set_max_card_width(snapped)
        # Drive the table view's thumb-row height off the same slider —
        # 50 .. 600 → 16 .. 96 (clamped inside set_thumb_height).
        if self._scans_table is not None:
            table_h = 16 + int(round((snapped - 50) / 550 * 80))
            self._scans_table.set_thumb_height(table_h)

    def _on_card_final_zoom(self, _snap_id: int, fit_zoom: float):
        """A card observed a final-step thumb at natural fit_zoom. If it's
        the smallest seen so far (or first), promote it to global zoom and
        broadcast so other cards that are now above (1+tol)*global get
        clamped down."""
        if fit_zoom <= 0:
            return
        tol = self._zoom_tolerance
        prev = self._global_zoom
        if prev is None or fit_zoom < prev * (1.0 - tol):
            self._global_zoom = fit_zoom
            for w in self.scan_widgets_by_scan.values():
                w.set_global_zoom(fit_zoom)

    def _on_settings_clicked(self):
        """Open Settings as a singleton tab in the right-side strip.
        Saved values land in the per-user config DB on Apply; the tab
        closes itself on Apply or Cancel."""
        if self._settings_tab is not None:
            self.tabs.setCurrentWidget(self._settings_tab)
            return
        from lib.gui.SettingsTab import SettingsTab
        from lib.gui.theme import lucide as _lucide_tab
        tab = SettingsTab()
        tab.applied.connect(self._on_settings_applied)
        tab.cancel_requested.connect(self._close_settings_tab)
        tab.open_downloader_requested.connect(self._open_model_downloader)
        idx = self.tabs.addTab(tab, _lucide_tab("settings", size=14), self.tr("Settings"))
        self.tabs.setTabToolTip(idx, self.tr("Settings"))
        self.tabs.setCurrentIndex(idx)
        self._settings_tab = tab

    def _open_model_downloader(self) -> None:
        """Open the Model Downloader dialog (singleton, modeless)."""
        existing = getattr(self, "_downloader_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        from lib.gui.ModelDownloaderTab import ModelDownloaderDialog
        dlg = ModelDownloaderDialog(self)

        def _on_close(*_):
            setattr(self, "_downloader_dialog", None)
            # Refresh engine availability — the user may have just
            # finished downloading the engine they're trying to use.
            try:
                self._ocr_tab.refresh_engines()
            except Exception:
                pass

        dlg.finished.connect(_on_close)
        self._downloader_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_settings_applied(self) -> None:
        self._apply_settings_changes()
        self._close_settings_tab()
        self.toast(self.tr("Settings applied."))

    def _close_settings_tab(self) -> None:
        tab = self._settings_tab
        if tab is None:
            return
        idx = self.tabs.indexOf(tab)
        if idx >= 0:
            self.tabs.removeTab(idx)
        tab.deleteLater()
        self._settings_tab = None

    def _apply_settings_changes(self) -> None:
        """Pick up settings the running session can adopt without a
        restart — thumb size + OCR defaults flip live. Theme is
        intentionally NOT applied live: every inline f-string in the
        widget tree was baked against the import-time palette, so a
        runtime swap leaves half the UI in the wrong colours. Prompt
        the user to restart instead."""
        from lib.app_data import db as cfg
        with cfg.session() as conn:
            theme = str(cfg.get(conn, cfg.KEY_THEME, "system") or "system")
            thumb = int(cfg.get(conn, cfg.KEY_THUMB_SIZE, 150))
            ocr_defaults = cfg.get(conn, cfg.KEY_OCR_DEFAULTS, {}) or {}
        # Compare against the theme this session was started with —
        # ``_session_theme`` is stashed at MainWindow construction.
        if theme != getattr(self, "_session_theme", theme):
            self._prompt_theme_restart()
        if hasattr(self, "_thumb_slider"):
            self._thumb_slider.setValue(thumb)
        if hasattr(self, "ocr_frame"):
            langs = list(ocr_defaults.get("languages") or [])
            engine = ocr_defaults.get("engine")
            if langs:
                self.ocr_frame.lang_input.set_tags(langs)
            if engine:
                self.ocr_frame.engine_group.set_current_key(engine)

    def _prompt_theme_restart(self) -> None:
        """Tell the user the new theme will fully apply on next startup."""
        QMessageBox.information(
            self, self.tr("Theme"),
            self.tr("Will apply on next startup."),
        )

    def _open_log_tab(self):
        """Open or focus the closable Log tab. Seeded with the rolling
        buffer from ProcessMonitor so users see prior history."""
        if self._log_tab is not None:
            self.tabs.setCurrentWidget(self._log_tab)
            return
        from lib.gui.theme import lucide as _lucide_tab
        viewer = self._LogViewerWidget(self.monitor_thread.log_buffer)
        idx = self.tabs.addTab(viewer, _lucide_tab("logs", size=14), self.tr("Log"))
        self.tabs.setTabToolTip(idx, self.tr("Console log history"))
        self.tabs.setCurrentIndex(idx)
        self._log_tab = viewer

    def closeEvent(self, event):
        if self.webcam_thread is not None:
            self.webcam_thread.stop()
        self.monitor_thread.stop()
        if self.voice_thread is not None:
            try:
                self.voice_thread.stop()
            except Exception:
                pass
        try:
            self.thumb_loader.close()
        except Exception:
            pass

        # Clear ._temp directory
        temp_dir = self.args.workspace_dir / self.output_dir_name / "._temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Failed to clear temp directory: {e}")

        event.accept()
