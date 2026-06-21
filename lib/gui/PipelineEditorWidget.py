# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Reusable pipeline editor widget for the Qt UI.

Shared between the StartupWindow (initial pipeline choice for a new project)
and the in-project "Edit Pipeline" dialog. Mirrors the web M11 editor but
native to Qt.

Layout (modernised 2026-06):

    ┌────────────────────────────────────────────────────────────────┐
    │  Card: header  (name | saved-pipelines dropdown)               │
    ├──────────────────────────┬─────────────────────────────────────┤
    │  Card: Steps             │  Card: Parameters (per-selected)    │
    │    • step list           │    • step name                      │
    │    • ↑ ↓ ✕ toolbar       │    • scrollable 2-col options grid  │
    │    • add-step gallery    │      with labels above inputs       │
    └──────────────────────────┴─────────────────────────────────────┘

Each "card" is `lib.gui.widgets.Card` styled via the QSS in
`lib.gui.theme`. Booleans render as the `Switch` toggle pill.

Inputs/outputs are plain YAML text — same format the chain loader consumes.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Optional

import yaml
from PySide6.QtCore import Qt, QRect, QSize, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QStackedWidget, QStyle, QStyledItemDelegate, QToolButton,
    QVBoxLayout, QWidget,
)

from lib.processors import registry as _proc_registry
from lib.processors.option_specs import COMMON_OPTION_SPECS
from lib.gui.colors import (
    COLOR_BG_OVERLAY_HOVER,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_FONT_DIM,
    COLOR_FONT_DISABLED,
    COLOR_FONT_SECONDARY,
    COLOR_OUTLINE,
    COLOR_OUTLINE_FAINT,
    COLOR_OUTLINE_GHOST,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_SECONDARY,
    COLOR_SECONDARY_BG_SOFT,
    qcolor,
)
from lib.gui.theme import icon
from lib.gui.widgets import Card, Switch


def _decimals_for_step(step: Optional[float]) -> int:
    """Number of decimals implied by a numeric step. step=1 ⇒ 0, step=0.1 ⇒ 1,
    step=0.01 ⇒ 2. Avoids "300.0000" in DPI-style fields."""
    if step is None or step <= 0:
        return 0
    return max(int(math.ceil(-math.log10(step))), 0)


# User-editable pipelines live in APP_DATA, seeded from the bundle on first
# access (the CLI keeps reading the bundled config/pipelines independently).
from lib.app_data import pipelines_dir as _pipelines_dir
PIPELINES_DIR = _pipelines_dir()
DEFAULT_PIPELINE_PATH = PIPELINES_DIR / "book_curved_x2.yaml"


def load_yaml_text(path: Path) -> str:
    return Path(path).read_text()


def parse_yaml(text: str) -> dict:
    return yaml.safe_load(text) or {}


def dump_yaml(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def _section_label(text: str, *, object_name: str = "SectionTitle") -> QLabel:
    """Small uppercase section header used inside cards."""
    lbl = QLabel(text)
    lbl.setObjectName(object_name)
    return lbl


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("FieldLabel")
    return lbl


def _help_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("HelpText")
    lbl.setWordWrap(True)
    return lbl


def _field_cell(label_text: str, editor: QWidget, help_text: str = "") -> QWidget:
    """Vertical (label / editor / optional help) cell that the params grid
    arranges in 2 columns. Wrapped in a subtle bordered container so the
    two-column grid reads as two independent fields — without the frame
    a label in column 1 visually rides next to the editor in column 0
    and gets misread as its caption."""
    cell = QFrame()
    cell.setObjectName("FieldCell")
    cell.setFrameShape(QFrame.Shape.NoFrame)
    cell.setStyleSheet(
        f"QFrame#FieldCell {{"
        f"  background-color: {COLOR_BG_OVERLAY_SOFT};"
        f"  border: 1px solid {COLOR_OUTLINE_FAINT};"
        f"  border-radius: 8px;"
        f"}}"
    )
    v = QVBoxLayout(cell)
    v.setContentsMargins(10, 8, 10, 10)
    v.setSpacing(4)
    v.addWidget(_field_label(label_text))
    v.addWidget(editor)
    if help_text:
        v.addWidget(_help_label(help_text))
    return cell


class _StepDelegate(QStyledItemDelegate):
    """Custom row painter for the Steps list.

    Layout per row: `[01] [editable name] [· processor]`.

    Item data roles used:
      * `Qt.UserRole + 2` — processor class name (string)
    The item's own `text()` carries the editable display name; double-
    click / SelectedClicked opens an in-place QLineEdit positioned over
    the name region only (see `updateEditorGeometry`).
    """

    PADDING_L = 10
    NAME_GAP = 14    # number → name
    PROC_GAP = 12    # name → processor
    ROW_H = 36

    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self, opt, idx):  # noqa: N802 — Qt API
        sz = super().sizeHint(opt, idx)
        return QSize(sz.width(), self.ROW_H)

    def _name_rect(self, opt, num_w: int, proc_w: int) -> QRect:
        x = opt.rect.x() + self.PADDING_L + num_w + self.NAME_GAP
        w = opt.rect.right() - x - proc_w - self.PROC_GAP
        return QRect(x, opt.rect.y(), max(0, w), opt.rect.height())

    def paint(self, p: QPainter, opt, idx):  # noqa: N802
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Selection / hover background. Tints derived from canonical
        # palette tokens so the row colours follow the active theme.
        if opt.state & QStyle.StateFlag.State_Selected:
            p.fillRect(opt.rect, qcolor(COLOR_PRIMARY_BG_SOFT))
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.fillRect(opt.rect, qcolor(COLOR_BG_OVERLAY_SOFT))

        proc = str(idx.data(Qt.ItemDataRole.UserRole + 2) or "")
        name = str(idx.data(Qt.ItemDataRole.DisplayRole) or "")
        row_num = idx.row() + 1

        fm = p.fontMetrics()
        num_str = f"{row_num:02d}"
        num_w = fm.horizontalAdvance(num_str)
        num_rect = QRect(opt.rect.x() + self.PADDING_L,
                         opt.rect.y(), num_w, opt.rect.height())
        p.setPen(qcolor(COLOR_FONT_DIM))
        p.drawText(num_rect, int(Qt.AlignmentFlag.AlignVCenter), num_str)

        # No leading "·" — the gap from right-alignment is enough of a
        # visual separator from the name.
        proc_str = proc
        proc_w = fm.horizontalAdvance(proc_str)
        proc_rect = QRect(opt.rect.right() - proc_w - self.PROC_GAP,
                          opt.rect.y(), proc_w, opt.rect.height())
        p.setPen(qcolor(COLOR_FONT_DISABLED))
        p.drawText(proc_rect, int(Qt.AlignmentFlag.AlignVCenter), proc_str)

        name_rect = self._name_rect(opt, num_w, proc_w)
        # Writeout state mirrored in name colour: dim when off, primary
        # when on. Primary (not INVERSE) so the active name stays
        # legible on both dark and light themes — INVERSE = white on
        # light, invisible against the row tint.
        from lib.gui.colors import COLOR_FONT_PRIMARY
        p.setPen(qcolor(COLOR_FONT_PRIMARY))
        p.drawText(name_rect, int(Qt.AlignmentFlag.AlignVCenter), name)

        p.restore()

    def updateEditorGeometry(self, editor, opt, idx):  # noqa: N802
        fm = editor.fontMetrics()
        num_w = fm.horizontalAdvance(f"{idx.row() + 1:02d}")
        proc = str(idx.data(Qt.ItemDataRole.UserRole + 2) or "")
        proc_w = fm.horizontalAdvance(proc) if proc else 0
        editor.setGeometry(self._name_rect(opt, num_w, proc_w))


class PipelinePillBar(QWidget):
    """Horizontal row of "pill" buttons — one per saved pipeline file.
    Selected pill is highlighted via the QSS `[active="true"]` rule.

    Overflow handling: pills are added in declaration order. If the row's
    width can't fit them all, the tail spills into a "… more" menu so the
    bar never wraps to two lines. Re-evaluated on every `resizeEvent`.

    `selected_changed(path)` fires when the user clicks a pill (signals
    aren't fired for programmatic `set_selected` calls so we don't loop
    through the load handler twice).
    """

    selected_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._items: list[tuple[str, str]] = []   # (name, path)
        self._selected_path: Optional[str] = None
        self._pills: list[QPushButton] = []
        self._more_btn: Optional[QToolButton] = None
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_items(self, items: list[tuple[str, str]], selected_path: Optional[str] = None):
        self._items = list(items)
        self._selected_path = selected_path
        self._rebuild()

    def set_selected(self, path: Optional[str]):
        self._selected_path = path
        for btn in self._pills:
            p = btn.property("path")
            btn.setProperty("active", "true" if p == path else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _rebuild(self):
        # Clear
        while self._row.count():
            it = self._row.takeAt(0)
            w = it.widget()
            if w is not None:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent (bare top-level window flash).
                w.hide()
                w.setParent(None)
        self._pills = []
        self._more_btn = None

        for name, path in self._items:
            btn = QPushButton(name)
            btn.setProperty("pill", "true")
            btn.setProperty("path", path)
            btn.setProperty("active",
                            "true" if path == self._selected_path else "false")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, p=path: self._on_clicked(p))
            self._row.addWidget(btn)
            self._pills.append(btn)

        # "… more" toggle that pops a menu of overflow items. Always
        # present; visibility flips in resizeEvent.
        self._more_btn = QToolButton()
        self._more_btn.setText("…")
        self._more_btn.setToolTip(self.tr("More pipelines"))
        self._more_btn.setProperty("pill", "true")
        self._more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._more_btn.setStyleSheet(
            f"QToolButton {{"
            f"  background-color: {COLOR_BG_OVERLAY_SOFT};"
            f"  color: {COLOR_FONT_SECONDARY};"
            f"  border: 1px solid {COLOR_OUTLINE};"
            f"  border-radius: 14px;"
            f"  padding: 4px 12px;"
            f"  font-weight: 600;"
            f"}}"
            f"QToolButton:hover {{ background-color: {COLOR_BG_OVERLAY_HOVER}; }}"
            f"QToolButton::menu-indicator {{ image: none; width: 0px; }}"
        )
        self._more_btn.hide()
        self._row.addWidget(self._more_btn)
        self._row.addStretch(1)
        self._relayout_overflow()

    def _on_clicked(self, path: str):
        self.set_selected(path)
        self.selected_changed.emit(path)

    def resizeEvent(self, ev):  # noqa: N802 — Qt API
        super().resizeEvent(ev)
        self._relayout_overflow()

    def _relayout_overflow(self):
        if not self._pills or self._more_btn is None:
            return
        # Greedy fit: first pass shows all, then hide tail until it fits.
        for btn in self._pills:
            btn.show()
        self._more_btn.hide()
        # Force layout activation so sizeHint reflects current contents.
        self.layout().activate()
        avail = self.width()
        if self.sizeHint().width() <= avail:
            return
        # Reserve room for the more button itself.
        more_w = self._more_btn.sizeHint().width() + 8
        # Walk from the end, hiding pills until the row fits.
        hidden: list[QPushButton] = []
        for btn in reversed(self._pills):
            if self.sizeHint().width() + more_w <= avail:
                break
            btn.hide()
            hidden.insert(0, btn)
            self.layout().activate()
        if not hidden:
            return
        menu = QMenu(self._more_btn)
        for btn in hidden:
            name = btn.text()
            path = btn.property("path")
            act = menu.addAction(name)
            act.triggered.connect(lambda _=False, p=path: self._on_clicked(p))
        self._more_btn.setMenu(menu)
        self._more_btn.show()


class StepParamsForm(QWidget):
    """Edits a single pipeline step's options based on its processor's specs."""

    changed = Signal()
    processor_changed = Signal(str)  # processor class name, for an external badge

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # The step name lives in the left-side Steps list (click-to-edit
        # handles renaming). The form now only owns the per-processor
        # option grid, so its
        # vertical footprint shrinks.

        # ── options section ────────────────────────────────────────
        opts_head = QHBoxLayout()
        opts_head.addWidget(_section_label(self.tr("Options")))
        opts_head.addStretch(1)
        opts_head.addWidget(_field_label(self.tr("Show advanced")))
        self._adv_switch = Switch()
        self._adv_switch.toggled.connect(self._on_advanced_toggled)
        opts_head.addWidget(self._adv_switch)
        v.addLayout(opts_head)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(10)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)
        v.addWidget(self._grid_host)
        v.addStretch(1)

        self._step: dict = {}
        # Tracks (cell_widget, is_advanced) so the toggle can re-show /
        # hide cells without rebuilding the grid on every flip.
        self._cells: list[tuple[QWidget, bool]] = []
        # Per-field metadata for cross-field visibility updates
        # (binarizer hides window/k for methods that ignore them).
        # `_field_meta[name] = (cell_widget, ParamSpec)`.
        self._field_meta: dict = {}

    def set_step(self, step: dict):
        self._step = step
        self.processor_changed.emit(step.get("processor") or "step")

        # Clear options grid
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent (bare top-level window flash).
                w.hide()
                w.setParent(None)
        self._cells = []
        self._field_meta = {}

        proc = step.get("processor") or ""
        specs = {**COMMON_OPTION_SPECS, **_proc_registry.option_specs().get(proc, {})}
        options = step.setdefault("options", {}) or {}

        # Pack non-advanced cells first, advanced last. Hiding advanced
        # then just chops the tail — no holes left between basic cells
        # (which the old in-place iteration produced because hidden grid
        # cells still occupy their slot).
        ordered = ([(f, s) for f, s in specs.items() if not s.advanced]
                   + [(f, s) for f, s in specs.items() if s.advanced])

        row, col = 0, 0
        for field, spec in ordered:
            value = options.get(field, spec.default)
            editor = self._editor_for(field, spec, value)
            cell = _field_cell(field, editor, spec.help or "")
            self._cells.append((cell, bool(spec.advanced)))
            self._field_meta[field] = (cell, spec)
            self._grid.addWidget(cell, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        # One pass at the end so advanced + cross-field rules combine.
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        """Show/hide cells based on (a) the advanced toggle and
        (b) per-field `visible_when` predicates that look at sibling
        option values."""
        show_adv = bool(self._adv_switch.isChecked())
        opts = self._step.get("options") or {}
        for field, (cell, spec) in self._field_meta.items():
            if spec.advanced and not show_adv:
                cell.hide()
                continue
            if spec.visible_when:
                ok = True
                for dep, allowed in spec.visible_when.items():
                    cur = opts.get(dep)
                    if cur is None:
                        # Fall back to spec default of the dep so first
                        # render before the user touches anything still
                        # applies the rule correctly.
                        dep_spec = self._field_meta.get(dep, (None, None))[1]
                        if dep_spec is not None:
                            cur = dep_spec.default
                    if cur not in allowed:
                        ok = False
                        break
                cell.setVisible(ok)
            else:
                cell.setVisible(True)

    def _on_advanced_toggled(self, _on: bool):
        """Show / hide advanced cells without rebuilding. Re-runs the
        full visibility pass so `visible_when` predicates still apply
        on top of the advanced filter."""
        self._apply_visibility()

    # ── editors ────────────────────────────────────────────────────
    def _editor_for(self, field: str, spec, value) -> QWidget:
        kind = spec.kind
        if kind == "bool":
            w = Switch()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, f=field: self._set_opt(f, bool(v)))
            return w
        if kind == "enum":
            w = QComboBox()
            w.addItems([str(c) for c in (spec.choices or [])])
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(lambda v, f=field: self._set_opt(f, v))
            # Contextual help: for PageDetector.backend, surface a
            # "download" affordance next to the dropdown whenever the
            # currently-picked backend's model is missing on disk.
            # Clicking the icon opens the Model Downloader tab.
            if (self._step.get("processor") == "PageDetector"
                    and field == "backend"):
                return self._wrap_page_backend_combo(w)
            return self._expand(w)
        if kind == "bounded_int":
            w = QSpinBox()
            w.setRange(int(spec.minimum or 0), int(spec.maximum or 999999))
            w.setValue(int(value if value is not None else spec.default))
            w.valueChanged.connect(lambda v, f=field: self._set_opt(f, int(v)))
            return self._expand(w)
        if kind == "bounded_float":
            w = QDoubleSpinBox()
            w.setRange(float(spec.minimum or 0.0), float(spec.maximum or 1e9))
            step = float(spec.step) if spec.step is not None else 0.1
            w.setSingleStep(step)
            w.setDecimals(_decimals_for_step(step))
            w.setValue(float(value if value is not None else spec.default))
            w.valueChanged.connect(lambda v, f=field: self._set_opt(f, float(v)))
            return self._expand(w)
        # default: free text (covers "string", incl. t:-templates)
        w = QLineEdit(str(value if value is not None else ""))
        w.textChanged.connect(lambda v, f=field: self._set_opt(f, v))
        return self._expand(w)

    @staticmethod
    def _expand(w: QWidget) -> QWidget:
        """Let inputs grow horizontally to fill their grid cell so the form
        breathes instead of pinning everything to a single fixed width."""
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        w.setMinimumHeight(28)
        return w

    def _wrap_page_backend_combo(self, combo: QComboBox) -> QWidget:
        """Pack `combo` with a right-aligned download icon button that
        shows whenever the currently-selected backend has no model on
        disk. Click → open the Model Downloader tab."""
        from PySide6.QtWidgets import QToolButton
        from lib.gui.theme import lucide as _lucide
        host = QWidget()
        h = QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(combo, 1)
        btn = QToolButton()
        btn.setIcon(_lucide("download", color=COLOR_SECONDARY, size=16))
        btn.setToolTip(self.tr(
            "No model installed for this backend — open the Model Downloader."
        ))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QToolButton{{border:0;padding:2px;}}"
                          f"QToolButton:hover{{background:{COLOR_SECONDARY_BG_SOFT};"
                          f"border-radius:4px;}}")
        btn.clicked.connect(self._open_model_downloader)
        h.addWidget(btn)
        self._expand(host)

        def _refresh():
            btn.setVisible(self._page_backend_missing(combo.currentText()))
        combo.currentTextChanged.connect(lambda _v: _refresh())
        _refresh()
        return host

    @staticmethod
    def _page_backend_missing(name: str) -> bool:
        """True if `name` would silently fall back to heuristic (i.e.
        no usable model on disk)."""
        try:
            from lib.processors.layout_backends.factory import probe_active_backend
            return probe_active_backend(name) == "heuristic" and name != "heuristic"
        except Exception:
            return False

    def _open_model_downloader(self) -> None:
        """Walk up to the MainWindow and invoke its tab opener. Decoupled
        from any direct import so this widget stays reusable in the
        standalone Pipeline Editor dialog."""
        w = self.window()
        opener = getattr(w, "_open_model_downloader", None)
        if callable(opener):
            opener()

    def _set(self, key: str, value):
        self._step[key] = value
        self.changed.emit()

    def _set_opt(self, key: str, value):
        self._step.setdefault("options", {})[key] = value
        # If this field is a dependency for any `visible_when` predicate,
        # other cells may need to show/hide. Cheap — runs over the
        # already-built `_field_meta` map.
        self._apply_visibility()
        self.changed.emit()


class PipelineEditorWidget(QWidget):
    """Edits a pipeline document (name + ordered list of steps).

    Emits `changed` whenever the in-memory doc mutates. Caller reads the
    current doc via `to_yaml()` / `to_doc()`.
    """

    changed = Signal()
    requested_close = Signal(str)  # action key: "apply" | "apply_reprocess" | "cancel"

    def __init__(self, initial_yaml: str | None = None,
                 saved_loader: Optional[Callable[[], list[dict]]] = None,
                 parent=None,
                 *,
                 db_path: Optional[str] = None,
                 default_scan_id: Optional[int] = None):
        super().__init__(parent)
        self._saved_loader = saved_loader or self._default_saved_loader
        self._db_path = db_path
        self._default_snap_id = default_scan_id

        if initial_yaml is None:
            initial_yaml = load_yaml_text(DEFAULT_PIPELINE_PATH)
        self._initial_yaml = initial_yaml
        self._doc = parse_yaml(initial_yaml)
        self._doc.setdefault("name", "pipeline")
        self._doc.setdefault("pipeline", [])

        self._build()
        self._refresh_steps()
        self._refresh_saved()
        # Initial preview run uses the loaded doc.
        if self._preview is not None:
            self._preview.set_pipeline_doc(self._doc)
        # Propagate every doc mutation to the preview (it debounces).
        self.changed.connect(self._push_doc_to_preview)

    # ── public API ─────────────────────────────────────────────────
    def to_doc(self) -> dict:
        return copy.deepcopy(self._doc)

    def to_yaml(self) -> str:
        return dump_yaml(self._doc)

    def load_yaml(self, text: str):
        self._doc = parse_yaml(text)
        self._doc.setdefault("name", "pipeline")
        self._doc.setdefault("pipeline", [])
        self._name_edit.setText(str(self._doc.get("name", "")))
        # Keep the replay toggle in sync with the freshly-loaded doc;
        # absent flag → default True.
        if hasattr(self, "_replay_switch"):
            self._replay_switch.blockSignals(True)
            self._replay_switch.setChecked(bool(self._doc.get("replay", True)))
            self._replay_switch.blockSignals(False)
        self._refresh_steps()
        self.changed.emit()

    # ── construction ───────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        # ── header card ────────────────────────────────────────────
        # Single row of pipeline pills. The doc's `name` field is set by
        # pill selection (or by Save as…) — no separate editable name
        # box, that was redundant with the pill labels.
        head_card = Card()
        head = head_card.layout()
        head.setContentsMargins(12, 8, 12, 8)
        head.setSpacing(6)

        pill_row = QHBoxLayout()
        pill_row.setSpacing(8)
        pill_row.addWidget(_field_label(self.tr("Pipeline")))
        self._pill_bar = PipelinePillBar()
        self._pill_bar.selected_changed.connect(self._on_pill_selected)
        pill_row.addWidget(self._pill_bar, 1)

        # Top-level `replay` switch. Lives in the pipeline header so the
        # user can flip the whole second-pass machinery on / off without
        # editing yaml. The replay engine still self-suppresses if no
        # step contributed a `replay_kind` stamp, so toggling it on for
        # a no-warp pipeline is harmless.
        replay_lbl = _field_label(self.tr("Replay"))
        replay_lbl.setToolTip(self.tr(
            "When activated does a second pass based on estimated "
            "transform parameters and reorganized processing elements "
            "(see per-element replay_order property) to maximize output "
            "quality."
        ))
        self._replay_switch = Switch()
        self._replay_switch.setToolTip(replay_lbl.toolTip())
        self._replay_switch.setChecked(bool(self._doc.get("replay", True)))
        self._replay_switch.toggled.connect(self._on_replay_toggled)
        pill_row.addWidget(replay_lbl)
        pill_row.addWidget(self._replay_switch)

        head.addLayout(pill_row)
        # Hidden line edit retained so save_as / pill load can mutate
        # the doc's name without a UI control. Other code (refresh) also
        # called .setText on it; cheap stub avoids spreading None checks.
        self._name_edit = QLineEdit(self._doc.get("name", ""))
        self._name_edit.hide()
        outer.addWidget(head_card)

        # ── body: stacked Steps/Params on the left, preview on the right.
        # `QSplitter` so the user can drag boundaries — tabs are narrower
        # than the old modal, so squeezing a 3rd column horizontally cost
        # the params card too much width.
        body_h = QSplitter(Qt.Orientation.Horizontal)
        body_h.setChildrenCollapsible(False)
        left_v = QSplitter(Qt.Orientation.Vertical)
        left_v.setChildrenCollapsible(False)

        # Left: Steps card. Toolbar (up / down / x) lives on the same
        # row as the STEPS title so the list claims the full inner
        # height. "Add step" combo + plus collapses onto one row at the
        # bottom — its own STEPS label was redundant.
        left_card = Card()
        left = left_card.layout()
        left.setSpacing(6)

        steps_head = QHBoxLayout()
        steps_head.setSpacing(4)
        steps_head.addWidget(_section_label(self.tr("Steps")))
        steps_head.addStretch(1)
        for ico, slot, tip in [
            ("chevron-up",   self._move_step_up,   self.tr("Move up")),
            ("chevron-down", self._move_step_down, self.tr("Move down")),
            ("trash-2",      self._remove_step,    self.tr("Remove step")),
        ]:
            b = QToolButton()
            b.setIcon(icon(ico))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            steps_head.addWidget(b)
        # Plus button → scrollable popup menu of all processors.
        # Replaces the bottom-row combo + plus pair that ate a full row
        # of card real-estate.
        self._add_btn = QToolButton()
        self._add_btn.setIcon(icon("plus"))
        self._add_btn.setToolTip(self.tr("Add step"))
        self._add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._add_btn.setStyleSheet(
            "QToolButton::menu-indicator { image: none; width: 0; }"
        )
        self._build_add_menu()
        steps_head.addWidget(self._add_btn)
        left.addLayout(steps_head)

        self._steps_list = QListWidget()
        self._steps_list.currentRowChanged.connect(self._on_step_selected)
        self._steps_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._steps_list.setMouseTracking(True)
        self._steps_list.setEditTriggers(
            QListWidget.EditTrigger.SelectedClicked
            | QListWidget.EditTrigger.DoubleClicked
            | QListWidget.EditTrigger.EditKeyPressed
        )
        self._step_delegate = _StepDelegate(self._steps_list)
        self._steps_list.setItemDelegate(self._step_delegate)
        # itemChanged fires when in-place rename commits — propagate to
        # the doc + UI badge.
        self._steps_list.itemChanged.connect(self._on_step_renamed)
        left.addWidget(self._steps_list, 1)

        left_v.addWidget(left_card)

        # Right: Params card (scrollable — large processors can have
        # many options and the dialog must not balloon vertically).
        right_card = Card(elevated=True)
        right = right_card.layout()
        right.setSpacing(6)
        # PARAMETERS title + processor badge inline. Saves a vertical
        # row vs. the old "title \n badge \n name" stack. Badge text is
        # populated by StepParamsForm via `set_external_badge`.
        params_head = QHBoxLayout()
        params_head.setSpacing(8)
        params_head.addWidget(_section_label(self.tr("Parameters")))
        self._params_badge = QLabel("—")
        self._params_badge.setObjectName("Badge")
        params_head.addWidget(self._params_badge)
        params_head.addStretch(1)
        right.addLayout(params_head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._params_stack = QStackedWidget()

        empty = QWidget()
        empty_v = QVBoxLayout(empty)
        empty_v.setContentsMargins(0, 24, 0, 0)
        msg = QLabel(self.tr("Select a step on the left to edit its parameters."))
        msg.setObjectName("Subtle")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_v.addWidget(msg)
        empty_v.addStretch(1)
        self._params_stack.addWidget(empty)
        self._empty_widget = empty

        self._params_form = StepParamsForm()
        self._params_form.changed.connect(self.changed)
        self._params_form.processor_changed.connect(self._params_badge.setText)
        self._params_stack.addWidget(self._params_form)

        scroll.setWidget(self._params_stack)
        right.addWidget(scroll, 1)
        left_v.addWidget(right_card)
        # Initial vertical split: steps gets ~40 %, params gets ~60 %. The
        # params card hosts the per-field grid that grows fastest with
        # config complexity.
        left_v.setStretchFactor(0, 2)
        left_v.setStretchFactor(1, 3)

        body_h.addWidget(left_v)

        # Right-side preview (only when a project DB is wired). Roughly
        # one third of the horizontal real estate; the rest goes to the
        # stacked steps + params.
        self._preview = None
        if self._db_path:
            from lib.gui.PipelinePreviewPanel import PipelinePreviewPanel
            self._preview = PipelinePreviewPanel(
                db_path=self._db_path,
                default_scan_id=self._default_snap_id,
            )
            body_h.addWidget(self._preview)
            body_h.setStretchFactor(0, 2)
            body_h.setStretchFactor(1, 1)

        outer.addWidget(body_h, 1)

    def _push_doc_to_preview(self) -> None:
        if self._preview is not None:
            self._preview.set_pipeline_doc(self._doc)

    # ── steps list ─────────────────────────────────────────────────
    def _refresh_steps(self):
        prev = self._steps_list.currentRow()
        self._steps_list.blockSignals(True)
        self._steps_list.clear()
        for step in self._doc.get("pipeline", []):
            label = step.get("name") or step.get("processor") or self.tr("step")
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole + 2,
                         step.get("processor", "?"))
            self._steps_list.addItem(item)
        self._steps_list.blockSignals(False)
        if 0 <= prev < self._steps_list.count():
            self._steps_list.setCurrentRow(prev)
        elif self._steps_list.count() > 0:
            self._steps_list.setCurrentRow(0)
        else:
            self._params_stack.setCurrentWidget(self._empty_widget)

    def _on_step_renamed(self, item: QListWidgetItem):
        row = self._steps_list.row(item)
        steps = self._doc.get("pipeline", [])
        if not (0 <= row < len(steps)):
            return
        new_name = item.text().strip() or steps[row].get("processor", "step")
        steps[row]["name"] = new_name
        # Keep the params form's hidden name in sync (still used by
        # set_step to populate its hidden name edit).
        if (self._steps_list.currentRow() == row
                and hasattr(self._params_form, "set_name_display")):
            self._params_form.set_name_display(new_name)
        self.changed.emit()

    def _on_step_selected(self, row: int):
        steps = self._doc.get("pipeline", [])
        if not (0 <= row < len(steps)):
            self._params_stack.setCurrentWidget(self._empty_widget)
            self._params_badge.setText("—")
            return
        self._params_form.set_step(steps[row])
        self._params_stack.setCurrentWidget(self._params_form)

    def _build_add_menu(self):
        """Populate the + button's popup menu with every available
        processor. The menu is set to scroll when it overflows the
        screen so long processor lists stay reachable."""
        menu = QMenu(self._add_btn)
        menu.setStyleSheet("QMenu { menu-scrollable: 1; }")
        for p in _proc_registry.list_summaries():
            name = p["name"]
            act = menu.addAction(name)
            summary = p.get("summary", "")
            if summary:
                act.setToolTip(summary)
            act.triggered.connect(lambda _=False, n=name: self._add_step(n))
        self._add_btn.setMenu(menu)

    def _add_step(self, proc: str):
        if not proc:
            return
        defaults = {f: s.default for f, s in _proc_registry.option_specs().get(proc, {}).items()}
        self._doc["pipeline"].append({
            "name": proc,
            "processor": proc,
            "options": defaults,
        })
        self._refresh_steps()
        self._steps_list.setCurrentRow(len(self._doc["pipeline"]) - 1)
        self.changed.emit()

    def _remove_step(self):
        row = self._steps_list.currentRow()
        steps = self._doc.get("pipeline", [])
        if 0 <= row < len(steps):
            del steps[row]
            self._refresh_steps()
            self.changed.emit()

    def _move_step_up(self):
        row = self._steps_list.currentRow()
        steps = self._doc.get("pipeline", [])
        if row > 0:
            steps[row - 1], steps[row] = steps[row], steps[row - 1]
            self._refresh_steps()
            self._steps_list.setCurrentRow(row - 1)
            self.changed.emit()

    def _move_step_down(self):
        row = self._steps_list.currentRow()
        steps = self._doc.get("pipeline", [])
        if 0 <= row < len(steps) - 1:
            steps[row + 1], steps[row] = steps[row], steps[row + 1]
            self._refresh_steps()
            self._steps_list.setCurrentRow(row + 1)
            self.changed.emit()

    # ── header actions ─────────────────────────────────────────────
    def _on_name_changed(self, value: str):
        self._doc["name"] = value
        self.changed.emit()

    def _on_replay_toggled(self, on: bool) -> None:
        self._doc["replay"] = bool(on)
        self.changed.emit()

    def _default_saved_loader(self) -> list[dict]:
        """List yaml pipelines on disk under `config/pipelines/`. Each
        entry: {"name": stem, "path": Path}. Sorted by name."""
        try:
            paths = sorted(PIPELINES_DIR.glob("*.yaml"))
        except OSError:
            paths = []
        return [{"name": p.stem, "path": p} for p in paths]

    def _refresh_saved(self):
        items = [(r["name"], str(r["path"])) for r in self._saved_loader()]
        # Selected pill: whichever path matches the doc's current name
        # (or `_loaded_path` if a pill was just clicked).
        sel = getattr(self, "_loaded_path", None)
        if sel is None:
            current_name = (self._doc.get("name") or "").strip()
            for name, path in items:
                if name == current_name:
                    sel = path
                    break
        self._pill_bar.set_items(items, selected_path=sel)

    def _on_pill_selected(self, path_str: str):
        path = Path(path_str)
        if not path.is_file():
            return
        try:
            self.load_yaml(path.read_text())
            self._loaded_path = path_str
        except OSError as e:
            QMessageBox.warning(
                self, self.tr("Load failed"),
                self.tr("{path}: {err}").format(path=path, err=e),
            )
        self._pill_bar.set_selected(path_str)

    # ── save / restore (driven by the footer gear menu) ──────────────
    def save_current(self):
        """Update the on-disk yaml that matches the doc's current name."""
        name = (self._doc.get("name") or "pipeline").strip() or "pipeline"
        self._save_named(name)

    def save_as(self):
        name, ok = QInputDialog.getText(
            self, self.tr("Save pipeline as…"), self.tr("Name:"),
            text=str(self._doc.get("name") or ""),
        )
        if ok and name.strip():
            self._doc["name"] = name.strip()
            self._name_edit.setText(name.strip())
            self._save_named(name.strip())

    def restore_defaults(self):
        if QMessageBox.question(
            self, self.tr("Restore defaults"),
            self.tr("Replace the current pipeline with the bundled default?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self.load_yaml(load_yaml_text(DEFAULT_PIPELINE_PATH))

    def make_save_menu_button(self) -> QToolButton:
        """Build a gear-style button wired to a popup menu containing the
        Update / Save as / Restore actions. Hosts (PipelineEditorDialog,
        StartupWindow) drop this into their bottom-left button row so the
        verbs live next to Cancel/Apply, opposite them."""
        btn = QToolButton()
        btn.setIcon(icon("square-chart-gantt"))
        btn.setIconSize(QSize(22, 22))
        btn.setToolTip(self.tr("Pipeline file actions"))
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setStyleSheet(
            f"QToolButton {{"
            f"  background-color: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 1px solid {COLOR_OUTLINE};"
            f"  border-radius: 8px;"
            f"  padding: 6px 10px;"
            f"}}"
            f"QToolButton:hover {{ background-color: {COLOR_OUTLINE_GHOST}; }}"
            f"QToolButton::menu-indicator {{ image: none; width: 0; }}"
        )
        menu = QMenu(btn)
        current_name = (self._doc.get("name") or "pipeline").strip() or "pipeline"
        act_update = menu.addAction(
            icon("save"),
            self.tr("Update '{name}.yaml'").format(name=current_name),
        )
        act_update.triggered.connect(self.save_current)
        act_save_as = menu.addAction(icon("save"), self.tr("Save as…"))
        act_save_as.triggered.connect(self.save_as)
        menu.addSeparator()
        act_restore = menu.addAction(icon("undo"), self.tr("Restore defaults"))
        act_restore.triggered.connect(self.restore_defaults)
        btn.setMenu(menu)
        return btn

    def _save_named(self, name: str):
        from slugify import slugify
        slug = slugify(name) or "pipeline"
        path = PIPELINES_DIR / f"{slug}.yaml"
        try:
            PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_yaml())
        except OSError as e:
            QMessageBox.warning(
                self, self.tr("Save failed"),
                self.tr("{path}: {err}").format(path=path, err=e),
            )
            return
        self._loaded_path = str(path)
        self._refresh_saved()


class PipelineEditorDialog(QDialog):
    """Modal wrapper: pipeline editor + buttons (Cancel / Apply / Apply + reprocess)."""

    APPLY = "apply"
    APPLY_REPROCESS = "apply_reprocess"
    CANCEL = "cancel"

    def __init__(self, initial_yaml: str, *, parent=None, allow_reprocess: bool = True,
                 title: Optional[str] = None,
                 db_path: Optional[str] = None,
                 default_scan_id: Optional[int] = None):
        super().__init__(parent)
        if title is None:
            title = self.tr("Edit pipeline")
        self.setWindowTitle(title)
        self.setModal(True)
        # Wider when the preview is enabled so the 3-column body breathes.
        self.resize(1400 if db_path else 1000, 720)
        self._action: str = self.CANCEL

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.editor = PipelineEditorWidget(
            initial_yaml=initial_yaml,
            db_path=db_path,
            default_scan_id=default_scan_id,
        )
        layout.addWidget(self.editor, 1)
        # On accept/reject, give the preview worker a beat to wind down
        # so its source ndarray doesn't get freed mid-process.
        self.finished.connect(self._stop_preview_worker)

        # Reprocess option
        if allow_reprocess:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(self.tr("Reprocess existing scans with new pipeline"))
            lbl.setObjectName("FieldLabel")
            row.addWidget(lbl)
            self.reprocess_box = Switch()
            self.reprocess_box.setToolTip(self.tr(
                "If on, every active scan is re-run through the updated pipeline."
            ))
            row.addWidget(self.reprocess_box)
            row.addStretch(1)
            layout.addLayout(row)
        else:
            self.reprocess_box = None

        # Buttons row — gear menu (Update / Save as / Restore) on the
        # left, Cancel + Apply on the right. QDialogButtonBox doesn't
        # accept widgets at custom positions, so the row is hand-built.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        gear_btn = self.editor.make_save_menu_button()
        btn_row.addWidget(gear_btn, 0, Qt.AlignmentFlag.AlignLeft)
        btn_row.addStretch(1)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        apply_btn = QPushButton(self.tr("Apply"))
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def _apply(self):
        if self.reprocess_box is not None and self.reprocess_box.isChecked():
            self._action = self.APPLY_REPROCESS
        else:
            self._action = self.APPLY
        self.accept()

    def action(self) -> str:
        return self._action

    def yaml_text(self) -> str:
        return self.editor.to_yaml()

    def _stop_preview_worker(self, _result: int = 0) -> None:
        prev = getattr(self.editor, "_preview", None)
        if prev is None:
            return
        try:
            alive = prev._is_worker_alive()
        except Exception:
            alive = False
        if alive:
            prev._worker.cancel()
            prev._worker.wait(2000)


class _SavedPipelinesListView(QWidget):
    """Landing page for `PipelineEditorTab`.

    Shows the list of yaml files in `config/pipelines/`. Each row is a
    card: title + step count, "Properties" button (→ editor view), and
    the active pipeline gets a primary frame + "Active" badge.

    Signals:
      * `open_editor(path)` — user clicked Properties on a row
      * `back_to_project()` — user dismissed the tab without picking
    """

    open_editor = Signal(str)
    back_to_project = Signal()

    def __init__(self, active_name: str, parent=None):
        super().__init__(parent)
        self._active_name = active_name or ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(8)
        title = QLabel(self.tr("Pipelines"))
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        head.addStretch(1)
        new_btn = QPushButton(self.tr("New pipeline"))
        new_btn.setToolTip(self.tr("Start from the bundled default"))
        new_btn.clicked.connect(lambda: self.open_editor.emit(""))
        head.addWidget(new_btn)
        outer.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_host = QWidget()
        self._list_v = QVBoxLayout(self._list_host)
        self._list_v.setContentsMargins(0, 0, 0, 0)
        self._list_v.setSpacing(8)
        scroll.setWidget(self._list_host)
        outer.addWidget(scroll, 1)

        self._refresh()

    def set_active_name(self, name: str) -> None:
        self._active_name = name or ""
        self._refresh()

    def _refresh(self) -> None:
        while self._list_v.count():
            it = self._list_v.takeAt(0)
            w = it.widget()
            if w is not None:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent (bare top-level window flash).
                w.hide()
                w.setParent(None)
        try:
            paths = sorted(PIPELINES_DIR.glob("*.yaml"))
        except OSError:
            paths = []
        for p in paths:
            self._list_v.addWidget(self._make_row(p))
        self._list_v.addStretch(1)

    def _make_row(self, path: Path) -> QWidget:
        try:
            doc = parse_yaml(path.read_text())
        except OSError:
            doc = {}
        name = doc.get("name") or path.stem
        steps = doc.get("pipeline") or []
        is_active = (name.strip() == self._active_name.strip())

        row = QFrame()
        row.setObjectName("PipelineRow")
        border = COLOR_PRIMARY if is_active else COLOR_OUTLINE_FAINT
        row.setStyleSheet(
            f"QFrame#PipelineRow {{"
            f"  background-color: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 2px solid {border};"
            f"  border-radius: 8px;"
            f"}}"
        )
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 10, 12, 10)
        h.setSpacing(10)

        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel(name)
        title.setStyleSheet("font-weight: 600; font-size: 13px;")
        col.addWidget(title)
        if len(steps) == 1:
            sub = QLabel(self.tr("1 step  ·  {filename}").format(filename=path.name))
        else:
            sub = QLabel(
                self.tr("{n} steps  ·  {filename}").format(
                    n=len(steps), filename=path.name,
                )
            )
        sub.setObjectName("HelpText")
        col.addWidget(sub)
        h.addLayout(col, 1)

        if is_active:
            badge = QLabel(self.tr("Active"))
            badge.setObjectName("Badge")
            badge.setStyleSheet(
                f"QLabel {{ color: {COLOR_PRIMARY}; "
                f"border: 1px solid {COLOR_PRIMARY}; "
                f"border-radius: 9px; padding: 2px 8px; font-weight: 600; }}"
            )
            h.addWidget(badge)

        props_btn = QPushButton(self.tr("Properties"))
        props_btn.setToolTip(self.tr("Open this pipeline in the editor"))
        props_btn.clicked.connect(lambda _=False, p=str(path): self.open_editor.emit(p))
        h.addWidget(props_btn)
        return row


class PipelineEditorTab(QWidget):
    """Two-view stack hosted inside a MainWindow tab.

    * View 1 (landing) — full `PipelineEditorWidget` opened directly on
      the project's attached pipeline, with "Save as new" /
      "Update <name>" footer buttons.
    * View 0 — `_SavedPipelinesListView`: saved-pipelines browser,
      reached via the editor's "← All pipelines" button.

    Signals:
      * `apply_requested(yaml, reprocess)` — Update flow (writes file
        and applies to project).
      * `cancel_requested()` — user clicked Back to project.
    """

    apply_requested = Signal(str, bool)   # (yaml_text, reprocess)
    cancel_requested = Signal()

    def __init__(self, initial_yaml: str, *, parent=None, allow_reprocess: bool = True,
                 db_path: Optional[str] = None,
                 default_scan_id: Optional[int] = None):
        super().__init__(parent)
        self._initial_yaml = initial_yaml
        self._db_path = db_path
        self._default_snap_id = default_scan_id
        self._allow_reprocess = allow_reprocess

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        # Landing list — derives "active" from the doc passed in by host.
        active_name = (parse_yaml(initial_yaml).get("name") or "").strip()
        self._list_view = _SavedPipelinesListView(active_name)
        self._list_view.open_editor.connect(self._open_editor)
        self._list_view.back_to_project.connect(self.cancel_requested)
        self._stack.addWidget(self._list_view)

        self._editor_page: Optional[QWidget] = None
        self.editor: Optional[PipelineEditorWidget] = None
        self.reprocess_box: Optional[Switch] = None
        self._update_btn: Optional[QPushButton] = None

        # Open the project's attached pipeline directly — the saved-
        # pipelines list (view 0) stays reachable via the editor's back
        # button, but is never the landing screen for an open project.
        self._open_editor("")

    # ── editor page (built lazily) ─────────────────────────────────
    def _build_editor_page(self, yaml_text: str) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        # Top: back-to-project bar.
        top = QHBoxLayout()
        top.setSpacing(8)
        back_btn = QPushButton(self.tr("← All pipelines"))
        back_btn.setToolTip(self.tr("Browse saved pipelines"))
        back_btn.clicked.connect(self._on_back)
        top.addWidget(back_btn)
        top.addStretch(1)
        v.addLayout(top)

        self.editor = PipelineEditorWidget(
            initial_yaml=yaml_text,
            db_path=self._db_path,
            default_scan_id=self._default_snap_id,
        )
        self.editor.changed.connect(self._refresh_update_label)
        v.addWidget(self.editor, 1)

        if self._allow_reprocess:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(self.tr("Reprocess existing scans with new pipeline"))
            lbl.setObjectName("FieldLabel")
            row.addWidget(lbl)
            self.reprocess_box = Switch()
            self.reprocess_box.setToolTip(self.tr(
                "If on, every active scan is re-run through the updated pipeline."
            ))
            row.addWidget(self.reprocess_box)
            row.addStretch(1)
            v.addLayout(row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        save_as_btn = QPushButton(self.tr("Save as new + apply"))
        save_as_btn.setToolTip(self.tr("Save current edits under a new pipeline name, then apply to the project"))
        save_as_btn.clicked.connect(self._on_save_as_new)
        btn_row.addWidget(save_as_btn)
        self._update_btn = QPushButton("")
        self._update_btn.clicked.connect(self._on_update)
        btn_row.addWidget(self._update_btn)
        apply_btn = QPushButton(self.tr("Apply"))
        apply_btn.setDefault(True)
        apply_btn.setToolTip(self.tr("Apply to the project without saving the pipeline file"))
        apply_btn.clicked.connect(self._on_apply_only)
        btn_row.addWidget(apply_btn)
        v.addLayout(btn_row)

        self._refresh_update_label()
        return page

    def _open_editor(self, path_str: str) -> None:
        if path_str:
            try:
                yaml_text = Path(path_str).read_text()
            except OSError:
                yaml_text = self._initial_yaml
        else:
            yaml_text = self._initial_yaml
        if self._editor_page is None:
            self._editor_page = self._build_editor_page(yaml_text)
            self._stack.addWidget(self._editor_page)
        else:
            self.editor.load_yaml(yaml_text)
        self._stack.setCurrentWidget(self._editor_page)

    def _on_back(self) -> None:
        # Stop the preview worker before returning, otherwise it keeps
        # munching CPU on a screen the user can't see.
        self.stop_preview_worker()
        # Refresh the landing list so newly-saved files appear, and the
        # "Active" badge tracks the current doc name.
        if self.editor is not None:
            doc_name = (self.editor._doc.get("name") or "").strip()
            self._list_view.set_active_name(doc_name)
        self._stack.setCurrentWidget(self._list_view)

    def _refresh_update_label(self) -> None:
        if self._update_btn is None or self.editor is None:
            return
        name = (self.editor._doc.get("name") or "pipeline").strip() or "pipeline"
        self._update_btn.setText(
            self.tr("Update “{name}” + apply").format(name=name))

    def _on_save_as_new(self) -> None:
        if self.editor is None:
            return
        name, ok = QInputDialog.getText(
            self, self.tr("Save as new pipeline"), self.tr("Name:"),
            text=str(self.editor._doc.get("name") or ""),
        )
        if not (ok and name.strip()):
            return
        self.editor._doc["name"] = name.strip()
        self.editor._name_edit.setText(name.strip())
        self.editor._save_named(name.strip())
        self._refresh_update_label()
        self._list_view.set_active_name(name.strip())
        reprocess = bool(
            self.reprocess_box is not None and self.reprocess_box.isChecked()
        )
        self.apply_requested.emit(self.editor.to_yaml(), reprocess)

    def _on_update(self) -> None:
        if self.editor is None:
            return
        self.editor.save_current()
        reprocess = bool(
            self.reprocess_box is not None and self.reprocess_box.isChecked()
        )
        self.apply_requested.emit(self.editor.to_yaml(), reprocess)

    def _on_apply_only(self) -> None:
        """Apply to the project session without writing the saved
        pipeline file — for trying edits before committing a name."""
        if self.editor is None:
            return
        reprocess = bool(
            self.reprocess_box is not None and self.reprocess_box.isChecked()
        )
        self.apply_requested.emit(self.editor.to_yaml(), reprocess)

    def stop_preview_worker(self) -> None:
        """Called by the host before removing this tab so the preview
        worker isn't freed mid-flight."""
        if self.editor is None:
            return
        prev = getattr(self.editor, "_preview", None)
        if prev is None:
            return
        try:
            if prev._is_worker_alive():
                prev._worker.cancel()
                prev._worker.wait(2000)
        except Exception:
            pass
