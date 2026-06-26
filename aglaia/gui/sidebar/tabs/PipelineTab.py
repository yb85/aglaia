# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Pipeline sidebar tab.

Layout
------
1. **Edit pipeline…** + **Force rerun** button row.
2. **Steps** — one card per pipeline step. Left side is the
   ``NN  StepName`` glyph; right side is empty until samples arrive,
   then fills with ``p5 · median · p95`` (centered dots, median
   coloured by share of total pipeline time).
3. **Backends** footer — what's available at runtime (JAX devices,
   MLX, Apple Vision, Surya, EAST / DBNet weights).

MainWindow wires the two buttons to its existing slots
(``open_pipeline_editor`` / ``_on_force_rerun_clicked``) and forwards
``ProcessMonitor.timing_signal`` into ``record_timing`` so the rows
fill in live.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_BUTTON,
    COLOR_BG_BUTTON_HOVER,
    COLOR_ERROR,
    COLOR_FONT_DIM,
    COLOR_FONT_MUTED,
    COLOR_FONT_INVERSE,
    COLOR_FONT_ON_BUTTON,
    COLOR_FONT_TIMING_NAME,
    COLOR_OUTLINE_BUTTON,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
)
from aglaia.gui.sidebar.widgets import PipelineTimingView


_BTN_QSS = f"""
QPushButton {{
    background-color: {COLOR_BG_BUTTON}; color: {COLOR_FONT_ON_BUTTON};
    border: 1px solid {COLOR_OUTLINE_BUTTON}; border-radius: 6px;
    padding: 6px 12px; font-weight: 600;
}}
QPushButton:hover {{ background-color: {COLOR_BG_BUTTON_HOVER}; }}
QPushButton:disabled {{ color: {COLOR_FONT_DIM}; }}
"""


# ── backends footer ────────────────────────────────────────────────


class _BackendsView(QWidget):
    """Footer block listing runtime backends + their availability."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(4)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(
            f"color: {COLOR_OUTLINE_SUBTLE}; background: {COLOR_OUTLINE_SUBTLE};"
        )
        sep.setFixedHeight(1)
        outer.addWidget(sep)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(self.tr("Backends"))
        title.setObjectName("FieldLabel")
        title_row.addWidget(title)
        title_row.addStretch(1)
        # "NN workers (auto|manual)" — resolved pipeline worker count + mode.
        self._workers_lbl = QLabel("")
        self._workers_lbl.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        self._workers_lbl.setToolTip(self.tr(
            "Pipeline worker processes. Set in Settings (0 = auto)."))
        title_row.addWidget(self._workers_lbl)
        outer.addLayout(title_row)

        self._rows = QVBoxLayout()
        self._rows.setSpacing(2)
        outer.addLayout(self._rows)
        self.refresh_workers()
        self.refresh()

    def refresh_workers(self) -> None:
        """Read the configured worker count and render 'NN workers (auto|manual)'."""
        from aglaia.worker_count import resolve_workers
        raw = None
        try:
            from aglaia.app_data import db as _cfg
            with _cfg.session() as conn:
                _cfg.bootstrap(conn)
                raw = _cfg.get(conn, _cfg.KEY_WORKERS, 0)
        except Exception:
            raw = 0
        count, is_auto = resolve_workers(raw)
        mode = self.tr("auto") if is_auto else self.tr("manual")
        self._workers_lbl.setText(self.tr("{n} workers ({mode})").format(
            n=count, mode=mode))

    def refresh(self) -> None:
        while self._rows.count():
            item = self._rows.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent (bare top-level window flash).
                w.hide()
                w.setParent(None)
        try:
            from aglaia.workers.Initializer import probe_capabilities
            caps = probe_capabilities()
        except Exception:
            caps = []
        for name, ok, detail in caps:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)

            glyph = QLabel("●")
            glyph.setStyleSheet(
                f"color: {COLOR_SUCCESS if ok else COLOR_ERROR}; "
                "font-size: 10px;"
            )
            row.addWidget(glyph)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(
                f"color: {COLOR_FONT_TIMING_NAME}; font-weight: 600; "
                "min-width: 96px;"
            )
            row.addWidget(name_lbl)

            # One line, ellipsised; full text on hover (paddle / surya /
            # model paths are long).
            from aglaia.gui.sidebar.widgets.RadioCardGroup import _ElidedLabel
            detail_lbl = _ElidedLabel(detail, max_lines=1)
            detail_lbl.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: 11px;"
            )
            detail_lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                                     QSizePolicy.Policy.Preferred)
            detail_lbl.setMinimumWidth(0)
            row.addWidget(detail_lbl, 1)

            wrap = QWidget()
            wrap.setLayout(row)
            self._rows.addWidget(wrap)


# ── PipelineTab ────────────────────────────────────────────────────


class PipelineTab(QWidget):
    """Pipeline controls + step list w/ live timing + backends footer."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel(self.tr("Pipeline"))
        title.setObjectName("SectionTitle")
        outer.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)
        # Primary action of this tab is now fixing input DPI; editing the
        # pipeline moves to a small tag beside the "Steps" header below.
        self.btn_fix_dpi = QPushButton(self.tr("Fix input DPI"))
        self.btn_fix_dpi.setStyleSheet(_BTN_QSS)
        self.btn_fix_dpi.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self.btn_fix_dpi.setToolTip(
            self.tr("Review and batch-correct each scan's import DPI."))
        try:
            from aglaia.gui.theme import icon as _icon
            self.btn_fix_dpi.setIcon(
                _icon("search-alert", color=COLOR_FONT_ON_BUTTON, size=14)
            )
        except Exception:
            pass

        self.btn_force = QPushButton(self.tr("Force rerun"))
        self.btn_force.setStyleSheet(_BTN_QSS)
        self.btn_force.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Fixed)
        self.btn_force.setToolTip(
            self.tr(
                "Reprocess every active scan from the raw input. "
                "Wipes any existing layout selection."
            )
        )
        try:
            from aglaia.gui.theme import icon as _icon
            self.btn_force.setIcon(
                _icon("refresh-cw", color=COLOR_FONT_ON_BUTTON, size=14)
            )
        except Exception:
            pass

        row.addWidget(self.btn_fix_dpi, 1)
        row.addWidget(self.btn_force, 1)
        outer.addLayout(row)

        outer.addSpacing(4)

        # "Steps" header with a small primary "Edit…" tag — editing the
        # pipeline definition is a property of the steps shown below.
        steps_row = QHBoxLayout()
        steps_row.setSpacing(6)
        steps_title = QLabel(self.tr("Steps"))
        steps_title.setObjectName("FieldLabel")
        steps_row.addWidget(steps_title)
        steps_row.addStretch(1)
        self.btn_edit = QPushButton(self.tr("Edit…"))
        self.btn_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit.setToolTip(self.tr("Edit the pipeline definition"))
        self.btn_edit.setStyleSheet(
            "QPushButton {"
            f"  color: {COLOR_FONT_INVERSE};"
            f"  background: {COLOR_PRIMARY};"
            "  border: none; border-radius: 9px;"
            "  padding: 2px 10px; font-size: 11px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: " + COLOR_PRIMARY + "; }"
        )
        steps_row.addWidget(self.btn_edit, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(steps_row)

        # Single view — same row card pre- and post-timing. The right
        # column stays empty until samples land. The steps live in their
        # OWN scroll area so a long pipeline scrolls locally instead of
        # forcing a scrollbar on the whole sidebar tab (the Backends footer
        # below stays put).
        self.timing_view = PipelineTimingView()
        steps_scroll = QScrollArea()
        steps_scroll.setWidgetResizable(True)
        steps_scroll.setFrameShape(QFrame.Shape.NoFrame)
        steps_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        steps_scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        steps_scroll.viewport().setAutoFillBackground(False)
        steps_scroll.setMinimumHeight(120)
        steps_scroll.setWidget(self.timing_view)
        outer.addWidget(steps_scroll, 1)  # fills available space; scrolls locally

        self._backends = _BackendsView()
        outer.addWidget(self._backends)

    # ── public API ─────────────────────────────────────────────────

    def set_steps(self, names: Iterable[str],
                  descriptions: Optional[dict] = None) -> None:
        """Seed the row order. Timing fills in per row when samples
        arrive. ``descriptions`` maps ``name → (essential, full)`` for the
        per-row parameter blurb (essential shown, full on click)."""
        self.timing_view.set_steps(names, descriptions)

    def record_timing(self, name: str, elapsed_ms: float,
                      success: bool = True) -> None:
        self.timing_view.record(name, elapsed_ms, success)

    def clear_timing(self) -> None:
        self.timing_view.clear()
