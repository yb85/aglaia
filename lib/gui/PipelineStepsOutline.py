# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Read-only pipeline steps outline.

A standalone, scrollable list of a pipeline's steps — one row per step
showing its number + name + a succinct parameter line, expanding on click
to the full per-parameter list. Built straight from a pipeline def dict via
each processor's ``describe_options`` classmethod (no processor is
constructed, so PageDewarper's JAX/MLX init never fires). Used by the
startup mode picker; decoupled from the live sidebar PipelineTab (which is
timing/DB-bound).
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from lib.gui.colors import (
    COLOR_BG_OVERLAY_SOFT,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_FAINT,
    COLOR_PRIMARY,
)


class _StepRow(QFrame):
    """One collapsible step row: header always visible, detail on click."""

    def __init__(self, idx: int, title: str, essential: str, full: str,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = full
        self.setObjectName("StepRow")
        self.setStyleSheet(
            "QFrame#StepRow {"
            f"  background: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 1px solid {COLOR_OUTLINE_FAINT};"
            "  border-radius: 8px;"
            "}"
            "QFrame#StepRow:hover { border-color: " + COLOR_PRIMARY + "; }"
            "QFrame#StepRow QLabel { background: transparent; border: none; }"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 7, 10, 7)
        v.setSpacing(2)

        head = QLabel(f"<b>{idx:02d}</b>&nbsp;&nbsp;{title}")
        head.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-size: 12px;")
        v.addWidget(head)

        if essential:
            sub = QLabel(essential)
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
            v.addWidget(sub)

        self._detail = QLabel(full)
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px;"
            "padding-top: 4px;")
        self._detail.setVisible(False)
        v.addWidget(self._detail)

        if full:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        if self._full:
            self._detail.setVisible(not self._detail.isVisible())
        super().mousePressEvent(event)


class PipelineStepsOutline(QScrollArea):
    """Scrollable read-only outline of a pipeline's steps."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Transparent so a coloured parent (e.g. the mode picker's grouped
        # surface) shows through the gaps between step rows.
        self.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.viewport().setAutoFillBackground(False)
        self._host = QWidget()
        self._host.setStyleSheet("background: transparent;")
        self._v = QVBoxLayout(self._host)
        self._v.setContentsMargins(0, 0, 0, 0)
        self._v.setSpacing(6)
        self._v.addStretch(1)
        self.setWidget(self._host)

    def set_pipeline(self, pipeline_def: Optional[dict]) -> None:
        # Clear existing rows (keep the trailing stretch).
        while self._v.count() > 1:
            item = self._v.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        steps = (pipeline_def or {}).get("pipeline", []) or []
        for i, step in enumerate(steps, 1):
            title = step.get("name") or step.get("processor") or "?"
            essential, full = self._describe(step)
            self._v.insertWidget(self._v.count() - 1,
                                 _StepRow(i, str(title), essential, full))

    @staticmethod
    def _describe(step: dict) -> tuple[str, str]:
        """(essential, full) parameter descriptions for a step, or ("","")
        for processors not in the registry (e.g. the Replay worker)."""
        from lib.processors import registry
        info = registry.get_processor(step.get("processor"))
        if info is None:
            return ("", "")
        try:
            opts: Any = info.option_cls(**(step.get("options") or {}))
            ess = info.processor_cls.describe_options(opts, "essential")
            full = info.processor_cls.describe_options(opts, "full")
        except Exception:
            return ("", "")
        return ess or "", full or ""
