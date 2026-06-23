# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Pipeline step list with optional live timing.

One row per step. Left side: index + name (middle-ellipsis at half
width when the row gets narrow). Right side: p5 · median · p95 in
milliseconds, separated by a centered dot, painted in a colour keyed
off the step's share of the total pipeline time. Until the first
timing sample arrives the timing column stays empty — the step list
itself is fully usable as a static index.

Public API:
  * ``set_steps(names)`` — seed rows in pipeline order (idempotent
    against name reordering: rows for missing steps drop, rows for
    new steps appear at the end).
  * ``record(name, elapsed_ms, success=True)`` — slot for
    ``ProcessMonitor.timing_signal``. Successful samples grow the
    rolling ``deque(maxlen=300)`` per step.
  * ``clear()`` — wipe samples (call when pipeline YAML swaps). Step
    rows remain so the list view stays consistent.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Iterable, Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_OVERLAY_SOFT,
    COLOR_BG_SURFACE,
    COLOR_FONT_DIM,
    COLOR_FONT_DISABLED,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_FONT_TIMING_NAME,
    COLOR_OUTLINE,
    COLOR_OUTLINE_SUBTLE,
    COLOR_SECONDARY,
    COLOR_TIMING_FAST,
    COLOR_TIMING_MID,
    COLOR_TIMING_SLOW,
    COLOR_TIMING_STOP_CHARTREUSE,
    COLOR_TIMING_STOP_CRIMSON,
    COLOR_TIMING_STOP_DARK_ORANGE,
    COLOR_TIMING_STOP_DARK_RED,
    COLOR_TIMING_STOP_GREEN,
    COLOR_TIMING_STOP_RED,
)


_DEQUE_MAX = 300

# Ten-stop scale, 0–10 % → 90–100 % of total pipeline time.
_SCALE = [
    COLOR_TIMING_STOP_GREEN,
    COLOR_TIMING_FAST,
    COLOR_TIMING_STOP_CHARTREUSE,
    COLOR_TIMING_MID,
    COLOR_SECONDARY,
    COLOR_TIMING_SLOW,
    COLOR_TIMING_STOP_DARK_ORANGE,
    COLOR_TIMING_STOP_RED,
    COLOR_TIMING_STOP_CRIMSON,
    COLOR_TIMING_STOP_DARK_RED,
]

_IDLE_TIMING_COLOR = COLOR_FONT_DISABLED


def _colour_for_share(share: float) -> str:
    if share <= 0.0:
        return _SCALE[0]
    bucket = min(int(share * 10), 9)
    return _SCALE[bucket]


# Total row stays neutral — its number is the SUM of every step, so a
# share/wall-time-based colour always reads "slow" and adds no signal.
# Keep the per-step scale for the rows that actually compete for time.


class _StepRow(QFrame):
    """One row: ``NN  step_name`` on the left, ``p5 · median · p95``
    on the right, both on the same baseline."""

    def __init__(self, index: int, name: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PipelineStepRow")
        self.setStyleSheet(
            f"QFrame#PipelineStepRow {{"
            f"  background: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 1px solid {COLOR_OUTLINE_SUBTLE};"
            f"  border-radius: 6px;"
            f"}}"
        )
        self._name = name
        self._desc_essential = ""
        self._desc_full = ""
        self._expanded = False

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(2)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        v.addLayout(h)

        self._num = QLabel(f"{index:02d}")
        self._num.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px;"
        )
        self._num.setFixedWidth(22)
        h.addWidget(self._num)

        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet(
            f"color: {COLOR_FONT_TIMING_NAME}; font-weight: 600;"
        )
        self._name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Preferred)
        self._name_lbl.setMinimumWidth(0)
        # Middle-ellipsis at half row width handled in resizeEvent.
        h.addWidget(self._name_lbl, 1)

        self._timing_lbl = QLabel("")
        self._timing_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._timing_lbl.setStyleSheet("font-size: 11px;")
        self._timing_lbl.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(self._timing_lbl)

        # Parameter blurb under the name/timing row: the "essential"
        # one-liner; clicking the row expands to the "full" listing.
        self._desc_lbl = QLabel("")
        self._desc_lbl.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 10px;")
        self._desc_lbl.setWordWrap(True)
        # Ignored horizontal policy: don't let the (single-line) sizeHint
        # widen the row — take the row's width and wrap to it, never
        # overflow the panel to the right.
        self._desc_lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                                     QSizePolicy.Policy.Minimum)
        self._desc_lbl.setMinimumWidth(0)
        self._desc_lbl.setVisible(False)
        v.addWidget(self._desc_lbl)

    def set_description(self, essential: str, full: str) -> None:
        self._desc_essential = essential or ""
        self._desc_full = full or ""
        self._render_desc()

    def _render_desc(self) -> None:
        if not (self._desc_essential or self._desc_full):
            self._desc_lbl.setVisible(False)
            return
        text = self._desc_full if self._expanded else self._desc_essential
        self._desc_lbl.setText(text or self._desc_essential or self._desc_full)
        self._desc_lbl.setVisible(bool(text))
        self.setCursor(Qt.CursorShape.PointingHandCursor
                       if self._desc_full else Qt.CursorShape.ArrowCursor)
        self.setToolTip("" if self._expanded else self.tr("Click for all parameters"))

    def mousePressEvent(self, ev) -> None:  # noqa: N802 — Qt API
        # Toggle essential ↔ full on click, when there's a full listing.
        if (ev.button() == Qt.MouseButton.LeftButton and self._desc_full
                and self._desc_full != self._desc_essential):
            self._expanded = not self._expanded
            self._render_desc()
        super().mousePressEvent(ev)

    def set_index(self, index: int) -> None:
        self._num.setText(f"{index:02d}")

    def set_total_marker(self) -> None:
        """Style the TOTAL pseudo-row: drop the index number, use a Σ
        symbol instead, and bold the step name so it reads as a header."""
        self._num.setText("Σ")
        self._num.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 12px;"
            f"font-weight: 700;"
        )
        self._name_lbl.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 700;"
        )

    def step_name(self) -> str:
        return self._name

    def clear_timing(self) -> None:
        self._timing_lbl.setText("")

    def update_timing(self, p5: Optional[float], median: Optional[float],
                      p95: Optional[float], share: float,
                      *, colour_override: Optional[str] = None) -> None:
        if median is None:
            self._timing_lbl.setText("")
            return
        colour = colour_override or _colour_for_share(share)
        p5_s = f"{p5:.0f}" if p5 is not None else "—"
        med_s = f"{median:.0f}"
        p95_s = f"{p95:.0f}" if p95 is not None else "—"
        # Centered dot separator; the median takes the share-keyed
        # colour, the surrounding stats stay dim so the eye lands on
        # the central number.
        self._timing_lbl.setText(
            f"<span style='color:{_IDLE_TIMING_COLOR}'>{p5_s}</span>"
            f"<span style='color:{_IDLE_TIMING_COLOR}'> · </span>"
            f"<span style='color:{colour}; font-weight:700'>{med_s}</span>"
            f"<span style='color:{_IDLE_TIMING_COLOR}'> · {p95_s} ms</span>"
        )

    def resizeEvent(self, ev) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(ev)
        # Middle-ellipsis at half the row width so the timing column
        # always has room. Skip when the row is wide enough to host
        # the full name.
        fm = QFontMetrics(self._name_lbl.font())
        full = fm.horizontalAdvance(self._name)
        budget = max(40, self._name_lbl.width())
        half_budget = max(40, self.width() // 2)
        if full <= budget:
            self._name_lbl.setText(self._name)
        else:
            self._name_lbl.setText(
                fm.elidedText(self._name, Qt.TextElideMode.ElideMiddle,
                               min(budget, half_budget))
            )


class PipelineTimingView(QWidget):
    """Vertical list of step rows. Rows are static cards; the timing
    column on each row updates live as samples arrive."""

    _REPAINT_INTERVAL_MS = 150

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._samples: dict[str, deque[float]] = {}
        self._rows: dict[str, _StepRow] = {}
        self._order: list[str] = []
        self._descriptions: dict[str, tuple[str, str]] = {}

        # Aggregate "TOTAL" row pinned above the step list. Shows p5 /
        # median / p95 over the sum-of-per-step pNs — approximate (a
        # true per-run sum would need run-id grouping the chain
        # doesn't currently emit) but useful for spotting whether the
        # whole pipeline regressed.
        self._total_row = _StepRow(0, self.tr("Total pipeline"), self)
        self._total_row.setStyleSheet(
            f"QFrame#PipelineStepRow {{"
            f"  background: {COLOR_BG_SURFACE};"
            f"  border: 1px solid {COLOR_OUTLINE};"
            f"  border-radius: 6px;"
            f"}}"
        )
        self._total_row.set_total_marker()
        self._layout.addWidget(self._total_row)
        self._total_row.setVisible(False)

        self._empty = QLabel(
            self.tr("No pipeline loaded — open one via Edit pipeline…")
        )
        self._empty.setStyleSheet(f"color: {COLOR_FONT_DIM};")
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._empty)

        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(self._REPAINT_INTERVAL_MS)
        self._timer.timeout.connect(self._flush)
        self._timer.start()

    # ── public API ──────────────────────────────────────────────────

    def set_steps(self, names: Iterable[str],
                  descriptions: Optional[dict] = None) -> None:
        """Build rows in the supplied order. Existing samples for any
        names that survive the new ordering are preserved. ``descriptions``
        maps ``name → (essential, full)`` for the per-row parameter blurb."""
        names = list(names)
        self._descriptions = dict(descriptions or {})
        # Drop rows for steps that left the pipeline.
        surviving = set(names)
        for n in list(self._rows):
            if n not in surviving:
                row = self._rows.pop(n)
                self._layout.removeWidget(row)
                row.deleteLater()
                self._samples.pop(n, None)
        self._order = list(names)
        # Wipe and re-add rows in the requested order.
        for n in names:
            row = self._rows.get(n)
            if row is None:
                row = _StepRow(0, n, self)
                self._rows[n] = row
            self._layout.removeWidget(row)
        # Re-attach in order, hiding the empty placeholder if any.
        self._empty.setVisible(not names)
        for i, n in enumerate(names, 1):
            row = self._rows[n]
            row.set_index(i)
            ess, full = self._descriptions.get(n, ("", ""))
            row.set_description(ess, full)
            self._layout.addWidget(row)
        self._dirty = True

    @Slot(str, float, bool)
    def record(self, name: str, elapsed_ms: float,
               success: bool = True) -> None:
        if not success:
            return
        dq = self._samples.get(name)
        if dq is None:
            dq = deque(maxlen=_DEQUE_MAX)
            self._samples[name] = dq
        dq.append(float(elapsed_ms))
        # Create a row on the fly if we get timing for a step we
        # didn't pre-seed (e.g. dynamic stage names from the chain
        # that aren't in the static YAML order).
        if name not in self._rows:
            row = _StepRow(len(self._order) + 1, name, self)
            self._rows[name] = row
            self._order.append(name)
            self._empty.setVisible(False)
            self._layout.addWidget(row)
        self._dirty = True

    def clear(self) -> None:
        """Wipe samples; keep the step rows so the list stays visible."""
        self._samples.clear()
        for row in self._rows.values():
            row.clear_timing()

    # ── internals ───────────────────────────────────────────────────

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        medians: dict[str, float] = {}
        for name, dq in self._samples.items():
            if dq:
                medians[name] = statistics.median(dq)
        total_median = sum(medians.values())
        # Avoid div-by-zero in per-step share when no samples yet.
        total_for_share = total_median or 1.0

        # Per-step rows.
        per_step_pcts: dict[str, tuple[float, float, float]] = {}
        for name, row in self._rows.items():
            dq = self._samples.get(name)
            if not dq:
                row.update_timing(None, None, None, 0.0)
                continue
            data = sorted(dq)
            n = len(data)

            def pct(p: float, data=data, n=n) -> float:
                if n == 1:
                    return data[0]
                idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
                return data[idx]

            p5 = pct(5)
            med = medians[name]
            p95 = pct(95)
            per_step_pcts[name] = (p5, med, p95)
            row.update_timing(p5, med, p95, med / total_for_share)

        # Aggregate TOTAL row — sum-of-pNs across all known steps. Not
        # statistically exact (sum(p95s) ≠ p95(sums)) but tracks the
        # same per-step trend the user just read above.
        if per_step_pcts and self._order:
            total_p5 = sum(v[0] for v in per_step_pcts.values())
            total_med = sum(v[1] for v in per_step_pcts.values())
            total_p95 = sum(v[2] for v in per_step_pcts.values())
            # Neutral colour on the total row — its number is by
            # definition the sum of every step, so any time-based scale
            # would always paint it as "slow" and the user would tune
            # it out.
            self._total_row.setVisible(True)
            self._total_row.update_timing(
                total_p5, total_med, total_p95, share=0.0,
                colour_override=_IDLE_TIMING_COLOR,
            )
        else:
            self._total_row.setVisible(False)
            self._total_row.update_timing(None, None, None, 0.0)
