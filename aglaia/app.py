#!/usr/bin/env python3
# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Aglaïa entry point.

Three operating modes, selected by the CLI arguments:

* **No arguments**  → show the StartupWindow wizard, then a normal GUI.
* **--headless + paths** → run end-to-end on the CLI (no Qt at all).
* **paths (no --headless)** → bootstrap the GUI with the supplied
  inputs / settings, skipping the StartupWindow.

See `aglaia/workers/cli.py` for the full argument schema.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# JAX/XLA host-memory tuning — must be set before any descendant process
# imports `jax`. See aglaia/processors/PageDewarper.py for the leak this prevents.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

# Ensure ./lib is importable when run from a checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from aglaia.workers.cli import (
    CliConfig, ExportTask, default_parent_dir, default_project_name,
    effective_workers, parse_argv, resolve_pipeline_path,
)


def _sigterm(_sig, _frame):
    raise KeyboardInterrupt


# ── headless path ─────────────────────────────────────────────────────

def _run_headless(cfg: CliConfig) -> int:
    # CLI-only first-run gate: processing assumes a configured install (a
    # detection model, seeded pipelines, defaults). If nothing's set up yet,
    # point the user at `aglaia --setup` instead of failing deep in the chain.
    if cfg.has_inputs():
        from aglaia.workers.setup_cli import has_user_config
        if not has_user_config():
            print("Aglaïa isn't set up yet. Run `aglaia --setup` (interactive) "
                  "or configure it via the GUI, then re-run.", file=sys.stderr)
            return 2
    from aglaia.workers.headless import run
    return run(cfg)


# ── GUI path ─────────────────────────────────────────────────────────

def _maybe_ui_shot(app, window) -> None:
    """If ``AGLAIA_UI_SHOT_DIR`` is set, screenshot the window + each sidebar
    tab into that dir once the UI settles, then quit. No-op otherwise.

    Enables headless (offscreen) inspection of the real, project-populated
    GUI — for agent-driven UI debugging and CI smoke shots. Optional
    ``AGLAIA_UI_SHOT_TABS`` (comma-separated tab names) limits which tabs."""
    out_dir = os.environ.get("AGLAIA_UI_SHOT_DIR")
    if not out_dir or os.environ.get("AGLAIA_UI_SCENARIO"):
        return  # a scenario, if set, owns the screenshots + quit
    from pathlib import Path
    from PySide6.QtCore import QTimer
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sidebar = getattr(window, "sidebar", None)
    tabs = list(getattr(sidebar, "_tabs", {}).keys()) if sidebar else []
    want = os.environ.get("AGLAIA_UI_SHOT_TABS", "").strip()
    if want:
        wanted = [t.strip() for t in want.split(",") if t.strip()]
        tabs = [t for t in tabs if t in wanted]
    seq = ["__window__"] + tabs

    def _shot(i: int) -> None:
        if i >= len(seq):
            QTimer.singleShot(200, app.quit)
            return
        name = seq[i]
        try:
            if name != "__window__" and sidebar is not None:
                sidebar.set_active(name)
            app.processEvents()
            window.grab().save(str(Path(out_dir) / f"{i:02d}_{name}.png"))
            print(f"[ui_shot] {out_dir}/{i:02d}_{name}.png", flush=True)
        except Exception as e:
            print(f"[ui_shot] {name}: {e}", flush=True)
        QTimer.singleShot(400, lambda: _shot(i + 1))

    QTimer.singleShot(1500, lambda: _shot(0))


def _maybe_test_autoquit(app, window) -> None:
    """If ``AGLAIA_TEST`` is set, quit the app cleanly once the pipeline has
    settled — enabling headless GUI benchmarking + open/close verification
    without a human (or a SIGTERM) in the loop. No-op otherwise.

    Two cases, one rule. Poll ``window._pipeline_idle()`` every 500 ms:

      * **Open/close** (no ``--force-proc``): the project is already
        complete, so the pipeline never goes busy — quit after a short
        settle so bring-up + teardown are exercised end to end.
      * **Reprocess** (``--force-proc``): the catch-up thread re-enqueues
        scans a beat after launch, so we wait to *see* the busy phase, then
        quit once it drains back to idle for ``quiet_s`` — bounding a full
        memory/timing profile run.

    ``AGLAIA_TEST_MAX_S`` (default 3600) is a hard ceiling: on expiry we
    quit anyway (exit 0) so a wedged run can't hang CI; the harness judges
    completion from the DB, not the exit path.

    Disabled by ``AGLAIA_TEST_AUTOQUIT=0`` — the memory bench keeps
    ``AGLAIA_TEST`` on (for the [RSS-poll] stdout echo) but turns the autoquit
    OFF and bounds the run by wall-clock + process-group kill instead, because
    idle-detection during a large force-proc reprocess is unreliable (the
    catch-up thread feeds while no widget yet reports processing → a premature
    quit that stops the log_queue drainer mid-feed → shutdown deadlock)."""
    if not os.environ.get("AGLAIA_TEST") or \
            os.environ.get("AGLAIA_TEST_AUTOQUIT", "1") == "0":
        return
    from PySide6.QtCore import QTimer

    min_settle_s = float(os.environ.get("AGLAIA_TEST_SETTLE_S", "4.0"))
    # Generous sustained-idle window: during a large reprocess the pipeline is
    # busy almost continuously, with only sub-second lulls between scans while
    # catch-up feeds; a transient lull must NOT trigger a quit. 10 s of
    # *continuous* idle reliably means done. (We deliberately do NOT poll the
    # DB here — a read every tick on the delete-journal project DB serialises
    # against worker writes and slowed a full reprocess ~9×.)
    quiet_s = float(os.environ.get("AGLAIA_TEST_QUIET_S", "10.0"))
    max_s = float(os.environ.get("AGLAIA_TEST_MAX_S", "3600"))

    import time as _time
    state = {"start": _time.monotonic(), "saw_busy": False, "last_busy": 0.0,
             "quit": False}

    def _idle() -> bool:
        try:
            return bool(window._pipeline_idle())
        except Exception:
            return True

    def _quit(reason: str) -> None:
        if state["quit"]:
            return
        state["quit"] = True
        print(f"[aglaia-test] {reason} → quit", flush=True)
        app.quit()

    def _tick() -> None:
        if state["quit"]:
            return
        now = _time.monotonic()
        elapsed = now - state["start"]
        if elapsed >= max_s:
            _quit(f"max {max_s:g}s reached")
            return
        if not _idle():
            state["saw_busy"] = True
            state["last_busy"] = now
            QTimer.singleShot(500, _tick)
            return
        # idle now — quit only after `quiet_s` of CONTINUOUS idle.
        if state["saw_busy"]:
            if now - state["last_busy"] >= quiet_s:
                _quit(f"reprocess drained, idle {quiet_s:g}s")
                return
        elif elapsed >= min_settle_s + quiet_s:
            _quit(f"settled, nothing to process ({elapsed:.0f}s)")
            return
        QTimer.singleShot(500, _tick)

    QTimer.singleShot(int(min_settle_s * 1000), _tick)


def _install_flash_debug(app) -> None:
    """AGLAIA_FLASH_DEBUG=1: log every parentless top-level widget Show.

    A QWidget shown without a parent briefly renders as its own bare
    window on macOS — this filter prints the widget and the Python stack
    of the offending show() so the culprit creation site is identifiable.
    """
    import traceback
    import weakref
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QWidget

    # Record the Python creation stack of every QWidget so we can name
    # the culprit even when the Show comes from C++ (no Python frames).
    birth: "weakref.WeakKeyDictionary[QWidget, str]" = weakref.WeakKeyDictionary()
    _orig_init = QWidget.__init__

    def _traced_init(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            birth[self] = "".join(traceback.format_stack(limit=12)[:-1])
        except Exception:
            pass

    QWidget.__init__ = _traced_init

    class _FlashSpy(QObject):
        def eventFilter(self, obj, ev):
            try:
                if ev.type() != QEvent.Type.Show:
                    return False
                if obj.isWidgetType() and obj.isWindow():
                    kids = [type(c).__name__ for c in obj.children()]
                    print(f"[flash-debug] window Show: "
                          f"class={type(obj).__name__} "
                          f"size={obj.width()}x{obj.height()} "
                          f"parent={type(obj.parent()).__name__ if obj.parent() else None} "
                          f"flags={hex(int(obj.windowFlags()))} "
                          f"children={kids}",
                          file=sys.stderr)
                    created = birth.get(obj)
                    if created:
                        print(f"[flash-debug] created at:\n{created}",
                              file=sys.stderr)
                    print("[flash-debug] shown from:", file=sys.stderr)
                    traceback.print_stack(file=sys.stderr)
                elif obj.isWindowType():  # raw QWindow (e.g. drag pixmap)
                    print(f"[flash-debug] QWindow Show: "
                          f"class={type(obj).__name__} "
                          f"title={obj.title()!r} "
                          f"size={obj.width()}x{obj.height()}",
                          file=sys.stderr)
                    print("[flash-debug] shown from:", file=sys.stderr)
                    traceback.print_stack(file=sys.stderr)
            except Exception:
                pass
            return False

    spy = _FlashSpy(app)
    app.installEventFilter(spy)
    app._flash_spy = spy  # keep alive
    print("[flash-debug] active — watching parentless top-level shows",
          file=sys.stderr)


def _qt_available() -> bool:
    """True iff PySide6 is importable — i.e. a GUI-capable install. A bare
    `pip install aglaia` / `aglaia-cli --without-gui` ships no Qt, so the
    GUI can't launch and `main()` routes to headless without --headless."""
    import importlib.util
    return importlib.util.find_spec("PySide6") is not None


def _qt_app() -> "QApplication":
    from PySide6.QtWidgets import QApplication
    # Patch CFBundleName BEFORE QApplication registers the process with the
    # window server — the Cmd-Tab / Dock-tile label is read once at that
    # moment. Doing it after (the macOS block lower down) only relabels the
    # menu bar. Best-effort: when launched as a bare `python -m aglaia`, the
    # Python framework may already have registered "Python" before our code
    # imports, in which case only the .app bundle can fully fix Cmd-Tab.
    if sys.platform == "darwin":
        try:
            from Foundation import NSBundle
            _b = NSBundle.mainBundle()
            _info = _b.localizedInfoDictionary() or _b.infoDictionary()
            if _info is not None:
                _info["CFBundleName"] = "Aglaïa"
                _info["CFBundleDisplayName"] = "Aglaïa"
        except Exception:
            pass
    app = QApplication(sys.argv[:1])  # do not pass user argv to Qt
    # App identity — window grouping, About box, and (with the macOS block
    # below) the Dock tile name when launched as the bare `aglaia` script.
    app.setApplicationName("Aglaïa")
    app.setApplicationDisplayName("Aglaïa")
    app.setOrganizationName("Aglaïa")
    app.setDesktopFileName("aglaia")  # Linux: ties the window to aglaia.desktop

    # macOS delivers a double-clicked .agl as a FileOpen event (not argv).
    # Capture it: stash the path so the launcher loop opens that project
    # instead of the StartupWindow; if a project window is already up, hand
    # back to the launcher to reopen the new file.
    from PySide6.QtCore import QEvent as _QEvent, QObject as _QObject

    class _FileOpenFilter(_QObject):
        def eventFilter(self, obj, ev):
            try:
                if ev.type() == _QEvent.Type.FileOpen:
                    path = ev.file()
                    if path:
                        app.setProperty("aglaia_open_file", path)
                        from PySide6.QtWidgets import QMainWindow
                        live = [w for w in app.topLevelWidgets()
                                if isinstance(w, QMainWindow) and w.isVisible()]
                        if live:
                            app.setProperty("aglaia_restart", "reopen")
                            app.setProperty("aglaia_reopen_path", path)
                            app.setProperty("aglaia_open_file", None)
                            for w in live:
                                w.close()
                    return True
            except Exception:
                pass
            return False

    _fof = _FileOpenFilter()
    app.installEventFilter(_fof)
    app._aglaia_file_open_filter = _fof  # keep a ref alive

    if os.environ.get("AGLAIA_FLASH_DEBUG"):
        _install_flash_debug(app)
    # Install i18n translator BEFORE any widget is built so the first
    # paint already shows translated strings.
    try:
        from aglaia.app_data import db as cfg
        from aglaia.i18n import install_translator
        try:
            with cfg.session() as conn:
                lang_pref = str(cfg.get(conn, cfg.KEY_LANGUAGE, "") or "")
        except Exception:
            lang_pref = ""
        install_translator(app, lang_pref)
    except Exception as e:
        print(f"i18n: skipped ({e})", file=sys.stderr)
    try:
        from aglaia.gui.theme import apply_modern_theme
        from aglaia.app_data import db as cfg
        try:
            with cfg.session() as conn:
                pref = str(cfg.get(conn, cfg.KEY_THEME, "system") or "system")
        except Exception:
            pref = "system"
        mode = "auto" if pref == "system" else pref
        apply_modern_theme(app, mode=mode)
    except Exception as e:
        print(f"theme: skipped ({e})", file=sys.stderr)
    # macOS bundle launch quirk: when started via the .app bundle (open
    # / Finder double-click), the process can come up *unactivated*,
    # leaving the first window painted but never raised in front of
    # Finder. Force NSApplication to foreground so the StartupWindow is
    # actually visible. No-op outside macOS / when pyobjc is missing.
    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyRegular
            ns = NSApplication.sharedApplication()
            ns.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            ns.activateIgnoringOtherApps_(True)
            # Dock identity for non-bundled launches (the bare `aglaia`
            # script from pip / brew / uv): macOS otherwise shows the
            # generic Python rocket + "Python". Set the real icon + name at
            # runtime. The .app bundle already carries these via Info.plist,
            # so this is a harmless no-op there.
            try:
                from AppKit import NSImage
                from aglaia.assets import asset_path
                icon = NSImage.alloc().initByReferencingFile_(
                    str(asset_path("app", "Aglaia.icns")))
                if icon is not None and icon.isValid():
                    ns.setApplicationIconImage_(icon)
            except Exception:
                pass
            try:
                from Foundation import NSBundle
                bundle = NSBundle.mainBundle()
                info = (bundle.localizedInfoDictionary()
                        or bundle.infoDictionary())
                if info is not None:
                    info["CFBundleName"] = "Aglaïa"
                    info["CFBundleDisplayName"] = "Aglaïa"
            except Exception:
                pass
        except Exception:
            pass
    # First launch as the bundled .app: auto-register the .agl ↔ app binding
    # so double-clicking a project just works (no trip to Settings). Once,
    # gated on running from a real .app (not a CLI/source run).
    try:
        from aglaia.app_data.filetype_register import (
            filetype_registration_available, register_filetype)
        if filetype_registration_available():
            from aglaia.app_data import db as cfg
            with cfg.session() as conn:
                if not cfg.get(conn, cfg.KEY_FILETYPE_ASSOC_DONE, False):
                    register_filetype()
                    cfg.set(conn, cfg.KEY_FILETYPE_ASSOC_DONE, True)
                    conn.commit()
    except Exception as e:
        print(f"filetype auto-register: skipped ({e})", file=sys.stderr)
    # First-run onboarding wizard: welcome + language + permissions + model
    # downloads in ONE window. Runs before the StartupWindow and before any
    # OCR/voice/layout engine is imported, so freshly downloaded models are
    # seen with no restart (the old flow deferred the download into a
    # MainWindow and then needed a restart that didn't work).
    try:
        from aglaia.gui.OnboardingWizard import OnboardingWizard
        if not OnboardingWizard.run_if_first_run(None):
            # User closed first-run setup before finishing → don't launch.
            # welcome_seen stays unset, so the wizard runs again next time.
            app.setProperty("aglaia_abort_launch", True)
            return app
    except Exception as e:
        print(f"onboarding: skipped ({e})", file=sys.stderr)
    # Trust gate for drop-in plugins — must run before any widget reads
    # the processor / OCR registries (which import accepted plugins).
    try:
        from aglaia.gui.plugin_trust import prompt_pending_plugins
        prompt_pending_plugins(None)
    except Exception as e:
        print(f"plugin-trust: skipped ({e})", file=sys.stderr)
    return app


def _choice_from_cfg(cfg: CliConfig) -> Optional["StartupChoice"]:
    """Translate CLI inputs into a `StartupChoice`, the same shape the
    StartupWindow would have produced. Returns None when the CLI
    arguments are insufficient to skip the wizard."""
    from aglaia.gui.StartupWindow import StartupChoice, StartupWindow

    if cfg.source == "project":
        from aglaia.storage import slug_from_project_file
        slug = slug_from_project_file(cfg.project_file)
        return StartupChoice(
            mode=StartupWindow.MODE_OPEN,
            project_dir=cfg.project_file.parent,
            project_name=slug,
            project_slug=slug,
            pipeline_yaml="",  # rehydrated from the DB
        )

    if cfg.source in ("pdfs", "images"):
        name = default_project_name(cfg)
        parent = default_parent_dir(cfg)
        parent.mkdir(parents=True, exist_ok=True)
        from slugify import slugify
        pipeline_path = resolve_pipeline_path(cfg.pipeline)
        yaml_text = pipeline_path.read_text(encoding="utf-8") if pipeline_path else ""
        mode = (StartupWindow.MODE_PDF if cfg.source == "pdfs"
                else StartupWindow.MODE_IMAGES)
        choice = StartupChoice(
            mode=mode,
            project_dir=parent,
            parent_dir=parent,
            project_name=name,
            project_slug=slugify(name) or "project",
            input_files=list(cfg.inputs),
            pipeline_yaml=yaml_text,
        )
        if cfg.input_dpi is not None:
            choice.input_dpi = float(cfg.input_dpi)
        return choice
    return None


def _choice_for_open_path(project_file: Path) -> "StartupChoice":
    """Build a MODE_OPEN choice for an explicit project file path — used
    by the in-place "Slim-down" round-trip, which reopens the same file
    without going through the StartupWindow."""
    from aglaia.gui.StartupWindow import StartupChoice, StartupWindow
    from aglaia.storage import slug_from_project_file
    project_file = Path(project_file)
    slug = slug_from_project_file(project_file)
    return StartupChoice(
        mode=StartupWindow.MODE_OPEN,
        project_dir=project_file.parent,
        project_name=slug,
        project_slug=slug,
        pipeline_yaml="",  # rehydrated from the DB
    )


def _bootstrap_with_choice(app, choice, cfg: CliConfig) -> int:
    """Run the regular GUI launch using a pre-built StartupChoice.

    Mirrors the legacy `main()` flow but skips the StartupWindow.
    """
    import threading as _threading
    from slugify import slugify
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QPixmap
    from PySide6.QtWidgets import QSplashScreen
    from aglaia.gui.MainWindow import MainWindow
    from aglaia.gui.StartupWindow import StartupWindow
    from aglaia.workers.Calibrator import load_calibration
    from aglaia.workers.ImportHelpers import (
        catchup_active_scans, enqueue_image_files, enqueue_pdf_files,
        reprocess_active_scans,
    )
    from aglaia.workers.Initializer import (
        create_processing_chain, initialize, load_pipeline_def,
    )
    from aglaia.workers.PDFprocessor import create_pdf_from_images
    from aglaia.storage.db import open_db
    from aglaia.storage.repo import PipelineRepo, ProjectRepo, ScanRepo

    # ── splash ────────────────────────────────────────────────────
    # Palette-aware bg + text so the loading panel reads correctly on
    # both dark and light themes (was hardcoded dark slate).
    from aglaia.gui.colors import active_palette_name as _palette_name
    if _palette_name() == "light":
        _splash_bg = QColor("#f4f4f5")
        _splash_fg = QColor("#18181b")
    else:
        _splash_bg = QColor(30, 30, 40)
        _splash_fg = QColor(245, 245, 250)
    pix = QPixmap(480, 220)
    pix.fill(_splash_bg)
    # X11BypassWindowManagerHint: stop tiling WMs (Hyprland/sway/i3) from
    # managing the splash — without it they tile/resize this transient
    # loading panel into the layout, which looks broken. Bypassed, Qt centres
    # it on screen as a floating overlay. No-op on macOS/Windows.
    splash = QSplashScreen(
        pix,
        Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.X11BypassWindowManagerHint,
    )
    splash.setFont(QFont("Helvetica", 14))
    splash.showMessage("Setting up project…", Qt.AlignmentFlag.AlignCenter,
                       _splash_fg)
    splash.show()
    app.processEvents()

    # ── args ──────────────────────────────────────────────────────
    project_dir = choice.project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    saved_argv = sys.argv[:]
    sys.argv = ["aglaia", str(project_dir)]
    if choice.mode == StartupWindow.MODE_CAPTURE and getattr(choice, "camera_index", None) is not None:
        sys.argv += ["--camera-id", str(choice.camera_index)]
    elif cfg.camera_id is not None:
        sys.argv += ["--camera-id", str(cfg.camera_id)]
    input_dpi = getattr(choice, "input_dpi", None) or cfg.input_dpi
    if input_dpi is not None:
        sys.argv += ["--input-dpi", str(input_dpi)]
    sys.argv += ["--workers", str(effective_workers(cfg.workers))]
    try:
        args = initialize(mode="capture")
    finally:
        sys.argv = saved_argv

    slug = choice.project_slug or slugify(
        choice.project_name or choice.project_dir.name
    )
    if slug:
        args.project_slug = slug
        paths = args.options.get("paths") or {}
        ws = args.workspace_dir
        paths["debug_prefix"] = str(ws / slug)
        paths["export"] = ws / f"{slug}_export"
        args.options["paths"] = paths

    # OPEN-mode: rehydrate yaml from the project DB.
    if choice.mode == StartupWindow.MODE_OPEN and not choice.pipeline_yaml:
        choice.pipeline_yaml = _load_active_pipeline_yaml(project_dir, slug)

    # Pipeline override from --pipeline if user passed one.
    pipeline_arg_path = resolve_pipeline_path(cfg.pipeline)
    if pipeline_arg_path is not None and choice.mode != StartupWindow.MODE_OPEN:
        choice.pipeline_yaml = pipeline_arg_path.read_text(encoding="utf-8")

    pipeline_path = _write_project_pipeline(slug, choice.pipeline_yaml)
    args.pipeline = pipeline_path

    # Calibration (capture only — harmless otherwise).
    calibration = load_calibration()
    cm = calibration.get("new_mtx") if calibration else None
    if cm is None and calibration:
        cm = calibration.get("mtx")
    args.options["calibration"] = {
        "camera_matrix": cm,
        "camera_matrix_resolution": (calibration.get("resolution")
                                     if calibration else None),
    }
    # NB: don't pre-create `<slug>_export/` — exports go to user-chosen
    # save-dialog paths, so this dir was always created empty and unused.

    # ── DB ────────────────────────────────────────────────────────
    # Resolve to existing .agl / legacy .scanproj.sqlite if any, else
    # create new as .agl.
    from aglaia.storage import (
        resolve_existing_project_db, project_filename,
    )
    db_path = (resolve_existing_project_db(project_dir, slug)
               or project_dir / project_filename(slug))
    pipeline_def = load_pipeline_def(pipeline_path) or {}
    conn = open_db(db_path)
    try:
        ProjectRepo(conn).init(name=slug, slug=slug)
        pipeline_version_id = PipelineRepo(conn).upsert(
            pipeline_path.read_text(encoding="utf-8"),
            pipeline_def.get("name"),
            step_count=len(pipeline_def.get("pipeline", [])),
        )
        start_idx = ScanRepo(conn).next_idx()
        try:
            scan_count = int(conn.execute(
                "SELECT COUNT(*) FROM scans WHERE deleted_at IS NULL "
                "AND root_node_id IS NOT NULL").fetchone()[0])
        except Exception:
            scan_count = None
    finally:
        conn.close()
    args.db_path = str(db_path)

    # Remember recent project (best effort) + cache its scan count so the
    # startup card can show it without reopening every project DB.
    try:
        from aglaia.app_data import db as app_db
        with app_db.session() as cdb:
            app_db.remember_project(cdb, db_path, slug, scan_count=scan_count)
    except Exception:
        pass

    splash.showMessage("Starting processing chain…",
                       Qt.AlignmentFlag.AlignCenter, _splash_fg)
    app.processEvents()

    # ── chain ────────────────────────────────────────────────────
    log_queue = multiprocessing.Queue()
    chain = create_processing_chain(args, log_queue, db_path=str(db_path))
    chain.start()
    state = {"chain": chain, "log_queue": log_queue,
             "pipeline_version_id": pipeline_version_id, "args": args}

    def apply_pipeline_callback(new_yaml: str, reprocess: bool):
        pipeline_path.write_text(new_yaml, encoding="utf-8")

        def _worker():
            new_def = load_pipeline_def(pipeline_path) or {}
            conn = open_db(db_path)
            try:
                new_pvid = PipelineRepo(conn).upsert(
                    new_yaml, new_def.get("name"),
                    step_count=len(new_def.get("pipeline", [])),
                )
            finally:
                conn.close()
            try:
                state["chain"].stop()
            except Exception:
                pass
            new_chain = create_processing_chain(
                state["args"], state["log_queue"], db_path=str(db_path),
            )
            new_chain.start()
            state["chain"] = new_chain
            state["pipeline_version_id"] = new_pvid

            def _swap():
                window.update_pipeline_context(
                    processing_queue=new_chain.get_input_queue(),
                    pipeline_version_id=new_pvid,
                    reprocess=reprocess,
                )
            QTimer.singleShot(0, _swap)

            if reprocess:
                from aglaia.workers.ImportHelpers import reprocess_active_scans
                reprocess_active_scans(db_path=str(db_path),
                                       pipeline_version_id=new_pvid,
                                       chain=new_chain)

        _threading.Thread(target=_worker, daemon=True,
                          name="ApplyPipeline").start()

    def force_reprocess_callback():
        """Reprocess every active scan against the current chain. Wipes
        existing branches + intermediate nodes; chosen-layout selection
        is reset by the wipe (raw root survives, the rest re-runs)."""
        cur_chain = state["chain"]
        cur_pvid = state["pipeline_version_id"]
        # Background thread: the SQL wipe walks every scan's subtree and
        # blocking the UI for that is rough on large projects.
        def _worker():
            try:
                reprocess_active_scans(
                    db_path=str(db_path), pipeline_version_id=cur_pvid,
                    chain=cur_chain,
                )
            except Exception as e:
                print(f"force rerun: {e}", file=sys.stderr)
        _threading.Thread(target=_worker, daemon=True,
                          name="ForceRerun").start()

    def reprocess_scans_callback(scan_ids):
        """Reprocess only the given scans (Fix-input-DPI). Same wipe-and-
        rerun as force, scoped to a subset."""
        cur_chain = state["chain"]
        cur_pvid = state["pipeline_version_id"]
        ids = set(int(s) for s in scan_ids)

        def _worker():
            try:
                reprocess_active_scans(
                    db_path=str(db_path), pipeline_version_id=cur_pvid,
                    chain=cur_chain, scan_ids=ids,
                )
            except Exception as e:
                print(f"reprocess scans: {e}", file=sys.stderr)
        _threading.Thread(target=_worker, daemon=True,
                          name="ReprocessSnaps").start()

    def reprocess_branch_callback(scan_id, branch_label):
        """Reprocess ONE page-branch of a scan (per-page step toggle). Reruns
        only that branch from its split point — sibling pages untouched —
        falling back to a whole-scan rerun when the scan isn't split."""
        from aglaia.workers.ImportHelpers import reprocess_branch
        cur_chain = state["chain"]
        cur_pvid = state["pipeline_version_id"]
        sid, blabel = int(scan_id), str(branch_label or "")

        def _worker():
            try:
                reprocess_branch(
                    db_path=str(db_path), pipeline_version_id=cur_pvid,
                    chain=cur_chain, scan_id=sid, branch_label=blabel,
                )
            except Exception as e:
                print(f"reprocess branch: {e}", file=sys.stderr)
        _threading.Thread(target=_worker, daemon=True,
                          name="ReprocessBranch").start()

    def is_pipeline_idle_callback() -> bool:
        """True when the current chain has no work in flight or queued — lets
        the GUI reconcile a progress bar stuck below 100%."""
        ch = state.get("chain")
        try:
            return ch.is_idle() if ch is not None else True
        except Exception:
            return False

    def stop_pipeline_callback() -> int:
        """Hard-stop the current chain + spin up a fresh, idle one.
        Returns the count of items discarded across the queues."""
        old = state["chain"]
        try:
            dropped = old.hard_stop()
        except Exception:
            dropped = 0
        new_chain = create_processing_chain(
            state["args"], state["log_queue"], db_path=str(db_path),
        )
        new_chain.start()
        state["chain"] = new_chain
        def _swap():
            window.update_pipeline_context(
                processing_queue=new_chain.get_input_queue(),
                pipeline_version_id=state["pipeline_version_id"],
            )
        QTimer.singleShot(0, _swap)
        return int(dropped)

    # ── window ────────────────────────────────────────────────────
    window = MainWindow(
        args, chain.get_input_queue(), chain.get_input_queue(), log_queue,
        slug, start_idx,
        db_path=str(db_path), pipeline_version_id=pipeline_version_id,
        source=choice.mode if choice.mode == "capture" else choice.mode,
        pipeline_yaml_path=pipeline_path,
        apply_pipeline_callback=apply_pipeline_callback,
        force_reprocess_callback=force_reprocess_callback,
        reprocess_scans_callback=reprocess_scans_callback,
        reprocess_branch_callback=reprocess_branch_callback,
        pipeline_idle_callback=is_pipeline_idle_callback,
        stop_pipeline_callback=stop_pipeline_callback,
    )
    # showMaximized (not show): __init__ already maximised, but a plain
    # show() here reverts the window to its never-sized Normal state on
    # Windows → a tiny default window. Re-assert maximised on all platforms.
    window.showMaximized()
    splash.finish(window)

    # ── layout heuristic-fallback gate ──────────────────────────
    # If the pipeline asks for `auto` layout detection and the only
    # backend available is the heuristic, warn the user before any
    # ingest can kick off any auto-processing. They either accept
    # heuristic (proceed) or open the downloader (skip auto-ingest;
    # they will trigger reprocess manually once a model is fetched).
    from aglaia.gui.PageWarningDialog import maybe_show_heuristic_warning
    _heuristic_choice = maybe_show_heuristic_warning(pipeline_def, parent=window)
    if _heuristic_choice == "open_downloader":
        QTimer.singleShot(0, window._open_model_downloader)

    # (First-run model downloads now happen in the OnboardingWizard, before
    # the StartupWindow — no MainWindow-hosted deferral needed.)

    # ── ingest ────────────────────────────────────────────────────
    if _heuristic_choice == "open_downloader":
        pass  # skip auto-ingest; user will retrigger after downloading.
    elif choice.mode == StartupWindow.MODE_PDF and choice.input_files:
        render_dpi = float(args.input_dpi) if args.input_dpi else 200.0
        _threading.Thread(
            target=enqueue_pdf_files,
            kwargs=dict(db_path=str(db_path), pipeline_version_id=pipeline_version_id,
                        slug=slug, chain=chain, pdf_paths=choice.input_files,
                        render_dpi=render_dpi, log_queue=log_queue),
            daemon=True, name="ImportPDF").start()
    elif choice.mode == StartupWindow.MODE_IMAGES and choice.input_files:
        import_dpi = float(args.input_dpi) if args.input_dpi else 120.0
        _threading.Thread(
            target=enqueue_image_files,
            kwargs=dict(db_path=str(db_path), pipeline_version_id=pipeline_version_id,
                        slug=slug, chain=chain, image_paths=choice.input_files,
                        default_dpi=import_dpi, log_queue=log_queue),
            daemon=True, name="ImportImages").start()
    elif choice.mode == StartupWindow.MODE_OPEN:
        # Reopen: surface scans whose pipeline objective is missing
        # (or all when --force-proc). Runs off the GUI thread because
        # reprocess wipes branches/nodes synchronously.
        def _catchup():
            try:
                n = catchup_active_scans(
                    db_path=str(db_path),
                    pipeline_version_id=pipeline_version_id,
                    chain=chain, force=cfg.force_proc,
                )
            except Exception as e:
                print(f"catchup: {e}", file=sys.stderr)
                return
            if n > 0 and hasattr(window, "status_bar_widget"):
                kind = "Force rerun" if cfg.force_proc else "Catch-up"
                QTimer.singleShot(0, lambda: window.toast(
                    f"{kind}: reprocessing {n} scan(s)…"))
        _threading.Thread(target=_catchup, daemon=True,
                          name="Catchup").start()

    # ── CLI-driven OCR + export hooks (GUI mode) ────────────────
    if cfg.do_ocr or cfg.exports:
        _wire_ocr_export_on_done(window, cfg, db_path=str(db_path), slug=slug)

    # ── Debug: offscreen UI screenshots ─────────────────────────
    # `AGLAIA_UI_SHOT_DIR=/tmp/shots` → once the window settles, scan the
    # whole window and each sidebar tab into that dir, then quit. Lets an
    # agent (or CI) inspect the real, project-populated GUI headlessly:
    #   QT_QPA_PLATFORM=offscreen AGLAIA_UI_SHOT_DIR=/tmp/s uv run \
    #       python aglaia.py path/to/project.agl
    _maybe_ui_shot(app, window)

    # ── Test harness: clean auto-quit once the pipeline settles ──
    # `AGLAIA_TEST=1` → quit after bring-up / reprocess drains, for
    # headless GUI benchmarking + open/close verification. No-op otherwise.
    _maybe_test_autoquit(app, window)

    # Dev memory profiling of the GUI process (AGLAIA_MEMRAY_DIR). Flushes on
    # clean quit — pair with AGLAIA_TEST autoquit or quit the window normally.
    from aglaia.workers.worker_lifecycle import maybe_start_memray, stop_memray
    _gui_memray = maybe_start_memray("gui")

    # Ctrl-C: Qt's C++ event loop never yields to Python's SIGINT handler, so
    # the GUI ignores Ctrl-C while a worker (same terminal group) dies from it
    # and the watchdog respawns it. Route SIGINT → app.quit() and run an idle
    # QTimer so the interpreter wakes often enough to deliver the signal →
    # app.exec() returns → the finally below reaps the chain cleanly.
    from PySide6.QtCore import QTimer as _QTimer
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _sigint_pump = _QTimer()
    _sigint_pump.timeout.connect(lambda: None)
    _sigint_pump.start(150)

    try:
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        stop_memray(_gui_memray)
        try:
            state["chain"].stop()
        except Exception:
            pass

    if args.make_pdf:
        layout_dir = args.options["paths"].get("layout")
        if layout_dir and layout_dir.exists():
            create_pdf_from_images(layout_dir, args.workspace_dir / f"{slug}.pdf")

    return 0


def _wire_ocr_export_on_done(window, cfg: CliConfig, *, db_path: str, slug: str):
    """Pre-fill the OCR frame from CLI values, and once the pipeline is
    idle auto-trigger OCR / export. Skipped if no CLI hooks were given."""
    from PySide6.QtCore import QTimer

    # Apply OCR defaults to the frame.
    if cfg.do_ocr and hasattr(window, "ocr_frame"):
        engine_map = {"auto": "apple_vision", "apple": "apple_vision",
                      "surya": "surya"}
        eng = engine_map.get(cfg.ocr_engine, "apple_vision")
        for i in range(window.ocr_frame.engine_combo.count()):
            if window.ocr_frame.engine_combo.itemData(i) == eng:
                window.ocr_frame.engine_combo.setCurrentIndex(i)
                break
        if cfg.ocr_languages:
            window.ocr_frame.lang_input.set_tags(cfg.ocr_languages)

    state = {"ocr_started": False, "exports_done": False}

    def _maybe_run():
        if not window._pipeline_idle():
            return
        if cfg.do_ocr and not state["ocr_started"]:
            state["ocr_started"] = True
            window.ocr_frame.run_requested.emit(
                window.ocr_frame.engine_group.current_key() or "apple_vision",
                window.ocr_frame.lang_input.tags(),
                window.ocr_frame.MODE_DEFAULT,
            )
            return
        if cfg.exports and not state["exports_done"]:
            state["exports_done"] = True
            for task in cfg.exports:
                if task.kind == "pdf":
                    if hasattr(window, "_export_tab"):
                        window._export_tab.set_compression(task.profile or "jbig2")
                    if cfg.do_ocr and hasattr(window, "chk_pdf_ocr_layer"):
                        window.chk_pdf_ocr_layer.setChecked(True)
                    window.make_pdf("output")
                elif task.kind == "md":
                    window._export_markdown()

    poll = QTimer(window)
    poll.timeout.connect(_maybe_run)
    poll.start(1000)


# ── helpers shared with the legacy GUI flow ──────────────────────────

def _write_project_pipeline(slug: str, yaml_text: str) -> Path:
    """Persist the active pipeline yaml to a process-lifetime tempfile —
    the chain ingests yaml via path. The DB's pipeline_versions table
    is the canonical store; this file is a shim."""
    import atexit
    import tempfile
    fd, name = tempfile.mkstemp(prefix=f"{slug}-", suffix=".pipeline.yaml")
    os.close(fd)
    target = Path(name)
    target.write_text(yaml_text, encoding="utf-8")
    atexit.register(lambda p=target: p.unlink(missing_ok=True))
    return target


def _load_active_pipeline_yaml(project_dir: Path, slug: str) -> str:
    from aglaia.storage import resolve_existing_project_db
    db_path = resolve_existing_project_db(project_dir, slug)
    if db_path is None:
        return ""
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT yaml_text FROM pipeline_versions "
                "ORDER BY is_active DESC, id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else ""
        finally:
            conn.close()
    except Exception:
        return ""


# ── main ─────────────────────────────────────────────────────────────

def _trace(msg: str) -> None:
    """Append a timestamped trace line to <log_dir>/aglaia-launch.log.
    PyInstaller-frozen builds swallow stdout/stderr, so file-logged
    breadcrumbs are the only way to see where main() got to."""
    try:
        from aglaia.app_data import log_dir
        from datetime import datetime
        p = log_dir() / "aglaia-launch.log"
        with open(p, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='milliseconds')} "
                    f"pid={os.getpid()} {msg}\n")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    _trace("main: enter")
    signal.signal(signal.SIGTERM, _sigterm)
    cfg = parse_argv(list(argv if argv is not None else sys.argv[1:]))
    _trace(f"main: parsed argv, headless={cfg.headless} has_inputs={cfg.has_inputs()}")

    from aglaia.workers.cli import run_list_commands
    if run_list_commands(cfg):
        return 0

    if cfg.setup:
        from aglaia.workers.setup_cli import run_setup
        return run_setup()

    if cfg.headless:
        if not cfg.has_inputs():
            print("--headless requires positional inputs.", file=sys.stderr)
            return 2
        return _run_headless(cfg)

    # No-GUI install (`aglaia-cli --without-gui`, `pip install aglaia` base):
    # PySide6 isn't present, so there's no GUI to launch. Fall back to the
    # headless pipeline automatically — the user shouldn't have to pass
    # --headless when the GUI simply isn't installed.
    if not _qt_available():
        if cfg.has_inputs():
            _trace("main: PySide6 absent → auto headless")
            print("No GUI (PySide6 not installed) — running headless.",
                  file=sys.stderr)
            return _run_headless(cfg)
        print("No GUI: PySide6 is not installed and no inputs were given.\n"
              "Pass image / PDF / .agl paths to run the headless pipeline, or\n"
              "install the GUI: pip install \"aglaia[gui,macos]\".",
              file=sys.stderr)
        return 2

    # GUI path.
    _trace("main: building QApplication")
    app = _qt_app()
    if app.property("aglaia_abort_launch"):
        return 0  # first-run setup was closed before completion — exit.
    _trace("main: QApplication built")

    # Optional tracemalloc loop for GUI debugging.
    if cfg.diagnose_memory:
        _spawn_memory_dump_loop()

    from aglaia.gui.StartupWindow import StartupWindow

    # GUI loop. A project window can hand control back to the launcher
    # (⌘W close · ⌘N new · ⌘O open · the close icon) by setting the
    # "aglaia_restart" app property and closing — which stops its chain in
    # _bootstrap's finally. We then re-show the launcher, pre-navigated.
    restart_action: Optional[str] = None
    first = True
    while True:
        if first:
            # Drain a macOS FileOpen (double-clicked .agl) queued at launch so
            # we open that project instead of the StartupWindow.
            app.processEvents()
        _open_file = app.property("aglaia_open_file")
        if first and cfg.has_inputs():
            choice = _choice_from_cfg(cfg)
            if choice is None:
                return 2
        elif _open_file:
            app.setProperty("aglaia_open_file", None)
            choice = _choice_for_open_path(Path(str(_open_file)))
        elif restart_action == "reopen":
            # In-place "Slim-down" round-trip: the prior window closed
            # (its chain is stopped, so the DB file is free), optionally
            # prune it in place, then reopen the same file directly.
            reopen_path = Path(str(app.property("aglaia_reopen_path")))
            if app.property("aglaia_slim_before_reopen"):
                app.setProperty("aglaia_slim_before_reopen", None)
                try:
                    from aglaia.workers.slim_export import slim_in_place
                    slim_in_place(reopen_path)
                except Exception as e:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(
                        None, "Slim-down failed",
                        f"Could not slim down the project:\n"
                        f"{type(e).__name__}: {e}")
            choice = _choice_for_open_path(reopen_path)
        else:
            _trace("main: constructing StartupWindow")
            startup = StartupWindow(initial_action=restart_action)
            startup.show()
            startup.raise_()
            startup.activateWindow()
            _trace("main: calling startup.exec()")
            rc = startup.exec()
            _trace(f"main: startup.exec() returned {rc}")
            if rc != StartupWindow.DialogCode.Accepted:
                return 0
            choice = startup.choice()
            if choice is None or choice.project_dir is None:
                return 0
        first = False

        app.setProperty("aglaia_restart", None)
        _trace("main: handing off to _bootstrap_with_choice")
        _bootstrap_with_choice(app, choice, cfg)
        restart_action = app.property("aglaia_restart")
        if not restart_action:
            return 0
        _trace(f"main: restart requested → {restart_action}")


def _spawn_memory_dump_loop() -> None:
    import tracemalloc, threading, time as _time
    tracemalloc.start(25)
    pid = os.getpid()

    def _loop():
        import psutil
        p = psutil.Process(pid)
        while True:
            _time.sleep(20)
            scan = tracemalloc.take_snapshot().filter_traces((
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, tracemalloc.__file__),
            ))
            top = scan.statistics("lineno")[:10]
            rss_mb = p.memory_info().rss / 1024 / 1024
            print(f"\n[MEM-DUMP gui pid={pid} RSS={rss_mb:.0f} MB] top allocs:",
                  flush=True)
            for s in top:
                print(f"  {s.size/1024:8.0f} KB  {s.traceback.format()[0].strip()}",
                      flush=True)
    threading.Thread(target=_loop, daemon=True, name="GuiMemoryDump").start()


# The process-startup wiring (multiprocessing freeze_support / spawn) lives in
# aglaia/__main__.py so it runs identically for `python -m aglaia`, the
# `aglaia` console script, and the PyInstaller-frozen app. `main()` above is
# the importable entry those all call.
