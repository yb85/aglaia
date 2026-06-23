# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Settings dialog backed by the per-user config DB.

Tabbed layout:

* **Appearance** — UI theme.
* **View**      — list-view thumbnail target size (px).
* **OCR**       — default engine + default language tags.
* **Export**    — default PDF compression, default stage, image format.
* **Pipeline**  — workers, input DPI, voice control toggle.
* **Paths**     — last-used project dir / export dir (editable + browse).

Widgets are stocked from existing reusable pieces wherever possible
(LanguageTagInput, the thumb-size slider geometry, the PDF-compression
combo entries) so the look stays consistent with the main window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QToolButton, QVBoxLayout, QWidget,
)

from aglaia.app_data import db as cfg
from aglaia.gui.LanguageTagInput import LanguageTagInput
from aglaia.gui.colors import (
    COLOR_FONT_PRIMARY,
    COLOR_FONT_SECTION_LABEL,
    COLOR_OUTLINE_GHOST,
)
from aglaia.gui.widgets import Card


# Responsive column breakpoints (window width in px):
#   < 900            → 1 col  (narrow side panes)
#   900 .. 1600      → 2 cols
#   >= 1600          → 3 cols (wide / full-screen Settings window)
# Cards stay readable: the widest form label ("Langues par défaut (Apple
# uniquement)") still fits in a 3-col cell at ~530 px.
_FLEX_BREAKPOINT_PX = 900
_FLEX_BREAKPOINT_3COL_PX = 1600


class _FlexGrid(QWidget):
    """Responsive grid: 1 / 2 / 3 columns by width (see breakpoints).

    Stretch is split equally across the active column count so cards
    grow / shrink in lockstep without slanting the layout."""

    def __init__(self, items: list[QWidget], parent: QWidget | None = None):
        super().__init__(parent)
        self._items = items
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(16)
        self._cols = 0
        self._relayout(2)

    def _relayout(self, cols: int) -> None:
        if cols == self._cols:
            return
        self._cols = cols
        # Detach every item without destroying it so we can re-add to
        # the freshly-rebuilt grid.
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent; with no parent that's a bare
                # top-level window flash.
                w.hide()
                w.setParent(None)
        for i, w in enumerate(self._items):
            row, col = divmod(i, cols)
            self._grid.addWidget(w, row, col)
            w.show()  # undo the explicit hide() from the clear above
        # Reset stretch — only the active columns share width.
        for c in range(4):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)

    def resizeEvent(self, ev) -> None:  # noqa: N802
        w = self.width()
        if w >= _FLEX_BREAKPOINT_3COL_PX:
            cols = 3
        elif w >= _FLEX_BREAKPOINT_PX:
            cols = 2
        else:
            cols = 1
        self._relayout(cols)
        super().resizeEvent(ev)


def QT_TRANSLATE_NOOP(context: str, text: str) -> str:  # noqa: N802 — Qt naming
    """Local stand-in for Qt's macro: marks the string for lupdate while
    returning the raw English at module-load time. The actual translation
    is looked up at use time via `self.tr(label)`."""
    return text


THEMES = [
    (QT_TRANSLATE_NOOP("SettingsTab", "System"), "system"),
    (QT_TRANSLATE_NOOP("SettingsTab", "Light"), "light"),
    (QT_TRANSLATE_NOOP("SettingsTab", "Dark"), "dark"),
]
COMPRESSION_OPTIONS = [
    (QT_TRANSLATE_NOOP("SettingsTab", "Auto"), "auto"),
    (QT_TRANSLATE_NOOP("SettingsTab", "JBIG2 (smallest)"), "jbig2"),
    (QT_TRANSLATE_NOOP("SettingsTab", "CCITT G4"), "g4"),
    (QT_TRANSLATE_NOOP("SettingsTab", "Native (color)"), "native"),
]
IMAGE_FORMATS = [
    (QT_TRANSLATE_NOOP("SettingsTab", "JPG"), "jpg"),
    (QT_TRANSLATE_NOOP("SettingsTab", "PNG"), "png"),
]


class _PathRow(QWidget):
    """Editable line edit + browse button → directory picker."""

    def __init__(self, initial: str, *, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.edit = QLineEdit(initial)
        browse = QToolButton()
        browse.setText("…")
        browse.setToolTip(self.tr("Browse…"))
        browse.clicked.connect(self._pick)
        row.addWidget(self.edit, 1)
        row.addWidget(browse)

    def _pick(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, self.tr("Pick directory"),
                                                  self.edit.text() or str(Path.home()))
        if chosen:
            self.edit.setText(chosen)

    def value(self) -> str:
        return self.edit.text().strip()


class SettingsTab(QWidget):
    """Tab-hosted config editor. Saves on Apply, discards on Cancel.

    `applied` fires after the in-memory state has been flushed to the
    per-user config DB so the host (MainWindow) can re-read its caches;
    `cancel_requested` fires when the user backs out without saving.
    The host owns the tab lifecycle and removes it on either signal.
    """

    applied = Signal()
    cancel_requested = Signal()
    open_downloader_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(540)

        # ── load current values ──────────────────────────────────
        with cfg.session() as conn:
            cfg.bootstrap(conn)
            self._values: dict[str, Any] = cfg.items(conn)

        # Each section is its own Card; a responsive 2-col flex grid
        # lays them out side-by-side above 900 px and stacks them into
        # one column below. Recovers the vertical real estate that the
        # full-width one-column layout was wasting on wide windows.
        cards = [
            self._card(self.tr("Appearance"), self._build_appearance_section()),
            self._card(self.tr("View"),       self._build_view_section()),
            self._card(self.tr("OCR"),        self._build_ocr_section()),
            self._card(self.tr("Export"),     self._build_export_section()),
            self._card(self.tr("Pipeline"),   self._build_pipeline_section()),
            self._card(self.tr("Paths"),      self._build_paths_section()),
            self._card(self.tr("Models"),     self._build_models_section()),
            self._card(self.tr("About"),      self._build_about_section()),
        ]
        body = QWidget()
        body_v = QVBoxLayout(body)
        body_v.setContentsMargins(0, 0, 0, 0)
        body_v.setSpacing(0)
        body_v.addWidget(_FlexGrid(cards))
        body_v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)

        # ── buttons ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.cancel_requested)
        btn_row.addWidget(cancel_btn)
        apply_btn = QPushButton(self.tr("Apply"))
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._save_and_apply)
        btn_row.addWidget(apply_btn)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)
        v.addWidget(scroll, 1)
        v.addLayout(btn_row)

    # ── section chrome ──────────────────────────────────────────
    @staticmethod
    def _card(title: str, content: QWidget) -> Card:
        """Card with a heading + thin underline + the section's body.

        Same visual rhythm as before, just wrapped in a `Card` so the
        2-col flex layout has discrete blocks to flow around."""
        card = Card()
        body = card.layout()
        head = QLabel(title)
        # Section title color follows the active text palette so light
        # mode shows dark text (not white-on-white).
        head.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 600; font-size: 14px;"
            " letter-spacing: 0.3px;"
        )
        body.addWidget(head)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setStyleSheet(f"color: {COLOR_OUTLINE_GHOST};"
                           f" background-color: {COLOR_OUTLINE_GHOST};")
        rule.setFixedHeight(1)
        body.addWidget(rule)
        content.setSizePolicy(QSizePolicy.Policy.Preferred,
                              QSizePolicy.Policy.Maximum)
        body.addWidget(content)
        card.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)
        return card

    # ── tabs ────────────────────────────────────────────────────

    def _build_appearance_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.theme_combo = QComboBox()
        for label, value in THEMES:
            self.theme_combo.addItem(self.tr(label), value)
        self._select_by_data(self.theme_combo, self._values.get(cfg.KEY_THEME, "system"))
        form.addRow(self.tr("Theme:"), self.theme_combo)

        from aglaia.i18n import SUPPORTED_LOCALES
        self.language_combo = QComboBox()
        for value, label in SUPPORTED_LOCALES:
            self.language_combo.addItem(self.tr(label), value)
        self._select_by_data(self.language_combo,
                             self._values.get(cfg.KEY_LANGUAGE, ""))
        self.language_combo.setToolTip(
            self.tr("Restart Aglaïa to apply a language change.")
        )
        form.addRow(self.tr("Language:"), self.language_combo)
        return w

    def _build_view_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        cur = int(self._values.get(cfg.KEY_THUMB_SIZE, 150))
        slider_row = QHBoxLayout()
        self.thumb_slider = QSlider(Qt.Orientation.Horizontal)
        self.thumb_slider.setRange(80, 400)
        self.thumb_slider.setValue(cur)
        self.thumb_value = QLabel(self.tr("{n} px").format(n=cur))
        self.thumb_value.setMinimumWidth(60)
        self.thumb_slider.valueChanged.connect(
            lambda v: self.thumb_value.setText(self.tr("{n} px").format(n=v))
        )
        slider_row.addWidget(self.thumb_slider, 1)
        slider_row.addWidget(self.thumb_value)
        wrap = QWidget()
        wrap.setLayout(slider_row)
        form.addRow(self.tr("Thumbnail size:"), wrap)
        return w

    def _build_ocr_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        from aglaia.workers.ocr import ENGINE_REGISTRY
        self.ocr_engine_combo = QComboBox()
        for name in ENGINE_REGISTRY.keys():
            self.ocr_engine_combo.addItem(name, name)
        defaults = self._values.get(cfg.KEY_OCR_DEFAULTS) or {}
        self._select_by_data(self.ocr_engine_combo,
                             defaults.get("engine", "apple_vision"))
        self.ocr_langs = LanguageTagInput()
        self.ocr_langs.set_tags(list(defaults.get("languages") or ["fr-FR"]))
        # Vision's actual supported set — only the Apple engines use it.
        try:
            from aglaia.workers.ocr.apple_vision import supported_languages
            self.ocr_langs.set_allowed_languages(supported_languages())
        except Exception:
            pass
        form.addRow(self.tr("Default engine:"), self.ocr_engine_combo)
        form.addRow(self.tr("Default languages (Apple only):"), self.ocr_langs)
        return w

    def _build_export_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        defaults = self._values.get(cfg.KEY_EXPORT_DEFAULTS) or {}
        self.export_compression = QComboBox()
        for label, value in COMPRESSION_OPTIONS:
            self.export_compression.addItem(self.tr(label), value)
        self._select_by_data(self.export_compression,
                             defaults.get("compression", "auto"))
        self.export_format = QComboBox()
        for label, value in IMAGE_FORMATS:
            self.export_format.addItem(self.tr(label), value)
        self._select_by_data(self.export_format,
                             defaults.get("image_format", "jpg"))
        self.export_stage = QLineEdit(defaults.get("stage") or "")
        self.export_stage.setPlaceholderText(
            self.tr("Leave empty for chosen-output (Selected scans)")
        )
        form.addRow(self.tr("Default compression:"), self.export_compression)
        form.addRow(self.tr("Default image format:"), self.export_format)
        form.addRow(self.tr("Default export stage:"), self.export_stage)
        return w

    def _build_pipeline_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        # Workers slider: 1..10 with live read-out. The hard floor is 1
        # (no parallelism); the 10-worker ceiling matches the practical
        # limit before SQLite contention + RSS pressure dominates gains.
        cur_workers = int(self._values.get(cfg.KEY_WORKERS, 4))
        cur_workers = max(1, min(10, cur_workers))
        workers_row = QHBoxLayout()
        self.workers_slider = QSlider(Qt.Orientation.Horizontal)
        self.workers_slider.setRange(1, 10)
        self.workers_slider.setValue(cur_workers)
        self.workers_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.workers_slider.setTickInterval(1)
        self.workers_value = QLabel(str(cur_workers))
        self.workers_value.setMinimumWidth(28)
        self.workers_slider.valueChanged.connect(
            lambda v: self.workers_value.setText(str(v))
        )
        workers_row.addWidget(self.workers_slider, 1)
        workers_row.addWidget(self.workers_value)
        workers_wrap = QWidget()
        workers_wrap.setLayout(workers_row)
        self.input_dpi_spin = QDoubleSpinBox()
        self.input_dpi_spin.setRange(50.0, 1200.0)
        self.input_dpi_spin.setSingleStep(50.0)
        self.input_dpi_spin.setValue(float(self._values.get(cfg.KEY_INPUT_DPI, 100.0)))
        self.input_dpi_spin.setSuffix(" dpi")
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 15)
        self.camera_spin.setValue(int(self._values.get(cfg.KEY_CAMERA_ID, 0)))
        self.voice_check = QCheckBox(self.tr("Enable voice commands at startup"))
        self.voice_check.setChecked(bool(self._values.get(cfg.KEY_VOICE_CONTROL, False)))
        self.tip_disable_check = QCheckBox(self.tr("Hide tip buttons"))
        self.tip_disable_check.setChecked(
            bool(self._values.get(cfg.KEY_DISABLE_TIP_BUTTONS, False))
        )
        # Tooltip mirrors what the Ko-Fi tip-jar page itself says so the
        # user knows what they'd otherwise be clicking.
        self.tip_disable_check.setToolTip(
            self.tr(
                "Hide the heart-shaped “Tip” shortcut in the status bar and "
                "the “Tip the developer!” link on the startup screen.\n\n"
                "What the tip button links to:\n"
                "Buy yb85 a coffee — tips keep Aglaïa free and ad-free."
            )
        )
        form.addRow(self.tr("Worker processes:"), workers_wrap)
        form.addRow(self.tr("Input DPI:"), self.input_dpi_spin)
        form.addRow(self.tr("Camera ID:"), self.camera_spin)
        form.addRow("", self.voice_check)
        form.addRow("", self.tip_disable_check)
        return w

    def _build_models_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.models_dir_row = _PathRow(
            str(self._values.get(cfg.KEY_MODELS_DIR, "") or "")
        )
        self.models_dir_row.edit.setPlaceholderText(
            self.tr("models  (relative → APP_DATA; absolute path also accepted)")
        )
        form.addRow(self.tr("ML models directory:"), self.models_dir_row)

        from aglaia.app_data import models_dir as _md
        from aglaia.gui.path_reveal import make_label_paths_clickable
        _mdp = _md()
        effective = QLabel(
            self.tr('<i>Effective: <a href="{p}">{p}</a></i>').format(p=_mdp))
        effective.setStyleSheet(f"color: {COLOR_FONT_SECTION_LABEL};")
        effective.setTextFormat(Qt.TextFormat.RichText)
        effective.setWordWrap(True)
        effective.setToolTip(self.tr("Click to reveal in your file manager"))
        make_label_paths_clickable(effective)
        self._models_effective_label = effective
        self.models_dir_row.edit.textChanged.connect(self._refresh_models_effective)
        form.addRow(effective)

        open_btn = QPushButton(self.tr("Open Model Downloader…"))
        open_btn.clicked.connect(self._open_downloader)
        form.addRow(open_btn)
        return w

    def _build_about_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        from aglaia.app_data import APP_NAME
        from aglaia.gui.AboutDialog import app_version
        ver = QLabel(self.tr("{name} v{ver}").format(
            name=APP_NAME, ver=app_version()))
        ver.setStyleSheet(f"color: {COLOR_FONT_SECTION_LABEL};")
        form.addRow(ver)
        about_btn = QPushButton(self.tr("About Aglaïa…"))
        about_btn.clicked.connect(self._open_about)
        form.addRow(about_btn)
        return w

    def _open_about(self) -> None:
        from aglaia.gui.AboutDialog import show_about
        show_about(self.window())

    def _refresh_models_effective(self) -> None:
        # Show what `models_dir()` *would* return given the live edit
        # (without persisting), so the user can sanity-check before Apply.
        from pathlib import Path
        from aglaia.app_data import app_data_dir
        raw = self.models_dir_row.edit.text().strip()
        if not raw:
            shown = app_data_dir() / "models"
        else:
            p = Path(raw).expanduser()
            shown = p if p.is_absolute() else app_data_dir() / p
        self._models_effective_label.setText(
            self.tr('<i>Effective: <a href="{p}">{p}</a></i>').format(p=shown)
        )

    def _open_downloader(self) -> None:
        self.open_downloader_requested.emit()

    def _build_paths_section(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.cwd_project_row = _PathRow(str(self._values.get(cfg.KEY_CWD_PROJECT, "")))
        self.cwd_export_row = _PathRow(str(self._values.get(cfg.KEY_CWD_EXPORT, "")))
        form.addRow(self.tr("Project dir:"), self.cwd_project_row)
        form.addRow(self.tr("Export dir:"), self.cwd_export_row)

        # Show APP_DATA dir for transparency — clickable to reveal in the
        # OS file manager.
        from aglaia.app_data import app_data_dir
        from aglaia.gui.path_reveal import make_label_paths_clickable
        _adp = app_data_dir()
        _db = _adp / "aglaia-config.db"
        info = QLabel(
            self.tr(
                '<i>APP_DATA: <a href="{p}">{p}</a></i><br>'
                '<i>config DB: <a href="{db}">aglaia-config.db</a> '
                "(inside APP_DATA)</i>"
            ).format(p=_adp, db=_db)
        )
        info.setStyleSheet(f"color: {COLOR_FONT_SECTION_LABEL};")
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)
        info.setToolTip(self.tr("Click to reveal in your file manager"))
        make_label_paths_clickable(info)
        form.addRow(info)

        # File-type registration — best-effort per-platform binding so
        # double-clicking a .agl in Finder/Explorer/file manager opens
        # Aglaïa. Implementation lives in aglaia.app_data.filetype_register.
        reg_row = QHBoxLayout()
        reg_btn = QPushButton(self.tr("Open .agl files with this app"))
        reg_btn.clicked.connect(self._register_filetype)
        unreg_btn = QPushButton(self.tr("Unregister"))
        unreg_btn.clicked.connect(self._unregister_filetype)
        reg_row.addWidget(reg_btn)
        reg_row.addWidget(unreg_btn)
        reg_row.addStretch(1)
        reg_wrap = QWidget()
        reg_wrap.setLayout(reg_row)
        form.addRow(reg_wrap)
        return w

    def _register_filetype(self) -> None:
        from aglaia.app_data.filetype_register import register_filetype
        ok, msg = register_filetype()
        self._toast_or_message(ok, msg)

    def _unregister_filetype(self) -> None:
        from aglaia.app_data.filetype_register import unregister_filetype
        ok, msg = unregister_filetype()
        self._toast_or_message(ok, msg)

    def _toast_or_message(self, ok: bool, msg: str) -> None:
        # No toast service in SettingsTab — pop a simple message box.
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            self.tr("File-type registration") if ok else self.tr("File-type registration failed"),
            msg,
        )

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _select_by_data(combo: QComboBox, data: Any) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    # ── save ────────────────────────────────────────────────────

    def _save_and_apply(self) -> None:
        with cfg.session() as conn:
            cfg.set(conn, cfg.KEY_THEME, self.theme_combo.currentData())
            cfg.set(conn, cfg.KEY_LANGUAGE,
                    self.language_combo.currentData() or "")
            cfg.set(conn, cfg.KEY_THUMB_SIZE, int(self.thumb_slider.value()))
            # Preserve the apple_docs complement choice (owned by the OCR
            # tab dropdown) — Settings doesn't surface it but must not
            # clobber it.
            _prev_ocr = cfg.get(conn, cfg.KEY_OCR_DEFAULTS, {}) or {}
            _ocr_defaults = {
                "engine": self.ocr_engine_combo.currentData(),
                "languages": self.ocr_langs.tags(),
            }
            if _prev_ocr.get("complement"):
                _ocr_defaults["complement"] = _prev_ocr["complement"]
            cfg.set(conn, cfg.KEY_OCR_DEFAULTS, _ocr_defaults)
            stage = self.export_stage.text().strip() or None
            cfg.set(conn, cfg.KEY_EXPORT_DEFAULTS, {
                "compression": self.export_compression.currentData(),
                "image_format": self.export_format.currentData(),
                "stage": stage,
            })
            cfg.set(conn, cfg.KEY_WORKERS, int(self.workers_slider.value()))
            cfg.set(conn, cfg.KEY_INPUT_DPI, float(self.input_dpi_spin.value()))
            cfg.set(conn, cfg.KEY_CAMERA_ID, int(self.camera_spin.value()))
            cfg.set(conn, cfg.KEY_VOICE_CONTROL, bool(self.voice_check.isChecked()))
            cfg.set(conn, cfg.KEY_DISABLE_TIP_BUTTONS,
                    bool(self.tip_disable_check.isChecked()))
            cfg.set(conn, cfg.KEY_CWD_PROJECT, self.cwd_project_row.value())
            cfg.set(conn, cfg.KEY_CWD_EXPORT, self.cwd_export_row.value())
            cfg.set(conn, cfg.KEY_MODELS_DIR, self.models_dir_row.value())
        self.applied.emit()
