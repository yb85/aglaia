# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Bottom status bar for the capture/edit MainWindow.

Three live indicators, left-to-right:

  * `WorkerRssStrip` — one tinted bar per worker (plus a separate one for
    the GUI process). Bar fill goes 0–100 % of the SIGKILL cap
    (`AGLAIA_WORKER_CAP_MB`, default 3072 MB). Color lerps green → orange
    → red so the user spots a runaway worker before the watchdog kills it.
  * `PipelineProgressBar` — overall pipeline progress for the current
    session. Counts `scan_imported` vs `branch_ready` events on log_queue.
  * `LogStrip` — last log line, clickable. Click opens a `LogViewerWidget`
    in a new MainWindow tab so the user can scroll the full history.

Everything in this file is pure-Qt; the parsing of `[RSS-poll …]` log
lines into per-worker mb values lives in `aglaia.workers.ProcessMonitor` so
the bar widgets only consume already-structured dicts.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

from PySide6.QtCore import Qt, Signal, QSize, QTimer, QRectF
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPixmap, QTextOption,
)
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPlainTextEdit, QSizePolicy,
    QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_OVERLAY_SOFT,
    COLOR_BG_SURFACE,
    COLOR_BG_TOAST,
    COLOR_ERROR,
    COLOR_ERROR_STRONG,
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_LINK_HOVER,
    COLOR_FONT_MUTED,
    COLOR_FONT_ON_TOAST,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_TIPPING,
    COLOR_WARNING,
    qcolor,
)


class _TipButton(QWidget):
    """Flat clickable heart-and-text shortcut to the Ko-Fi tip jar.
    Heart glows rosy-red so it sells the tip-jar invitation without
    being a loud button."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from aglaia.gui.theme import lucide_pixmap
        from aglaia.gui.StartupWindow import TIPPING_URL
        self._url = TIPPING_URL
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("TipButton")
        self.setToolTip(
            self.tr(
                "Buy yb85 a coffee\n"
                "Tips help keep Aglaïa free and ad-free. Thank you!\n"
                "{url}"
            ).format(url=TIPPING_URL)
        )
        accent = COLOR_ERROR_STRONG
        # Bordered pill around icon + label so the affordance reads as
        # one rosy-red chip. Border + text colour share the accent.
        self.setStyleSheet(
            f"#TipButton {{border:1px solid {accent}; border-radius:6px;}} "
            f"#TipButton:hover {{background:{COLOR_TIPPING};}}"
        )
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 2, 8, 2)
        h.setSpacing(4)
        pix = lucide_pixmap("heart", color=accent, size=14)
        pix.setDevicePixelRatio(2.0)
        icon = QLabel()
        icon.setPixmap(pix)
        icon.setFixedSize(14, 14)
        icon.setStyleSheet("background:transparent; border:none;")
        glow = QGraphicsDropShadowEffect(icon)
        glow.setOffset(0, 0)
        glow.setBlurRadius(80)
        glow.setColor(QColor(accent))
        icon.setGraphicsEffect(glow)
        h.addWidget(icon)
        lbl = QLabel(self.tr("Tip"))
        lbl.setStyleSheet(
            f"QLabel{{color:{accent}; font-size:12px; font-weight:600; "
            f"background:transparent; border:none;}}"
        )
        text_glow = QGraphicsDropShadowEffect(lbl)
        text_glow.setOffset(0, 0)
        text_glow.setBlurRadius(50)
        text_glow.setColor(QColor(accent))
        lbl.setGraphicsEffect(text_glow)
        h.addWidget(lbl)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(self._url))
        super().mouseReleaseEvent(ev)


def _kill_cap_mb() -> float:
    """Match the cap the watchdog enforces (see
    `IntegratedProcessingChain._watchdog_loop`)."""
    try:
        return float(os.environ.get("AGLAIA_WORKER_CAP_MB", "3072"))
    except ValueError:
        return 3072.0


# Per-worker "pressure" scale for the worst-worker bar: 100% = 1024 MB. A
# tight scale so normal worker RSS is visible (the kill cap is much higher).
_WORKER_PRESSURE_CAP_MB = 1024.0


def _lerp_color(t: float) -> QColor:
    """Green → orange → red as t goes 0 → 1. Clamped."""
    t = max(0.0, min(1.0, t))
    # Two segment lerp: 0..0.5 green→orange, 0.5..1 orange→red.
    if t < 0.5:
        u = t / 0.5
        r = int(34 + (245 - 34) * u)       # 22 → f5
        g = int(197 + (158 - 197) * u)     # c5 → 9e
        b = int(94 + (11 - 94) * u)        # 5e → 0b
    else:
        u = (t - 0.5) / 0.5
        r = int(245 + (239 - 245) * u)
        g = int(158 + (68 - 158) * u)
        b = int(11 + (68 - 11) * u)
    return QColor(r, g, b)


_WRENCH_PIX_CACHE: dict[tuple[str, int], QPixmap] = {}


def _wrench_pixmap(color: str, size: int) -> QPixmap:
    """Lucide wrench → QPixmap, cached by (color, size). Module-scope cache —
    re-rendering the SVG on every paint tick is too slow."""
    key = (color, size)
    cached = _WRENCH_PIX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from aglaia.gui.theme import lucide_pixmap
        pix = lucide_pixmap("wrench", color=color, size=size)
    except Exception:
        pix = QPixmap()
    _WRENCH_PIX_CACHE[key] = pix
    return pix


class WorkerRssBar(QWidget):
    """Single rounded bar — fill colour + width tracks RSS / cap ratio."""

    def __init__(self, name: str, cap_mb: float, parent: QWidget | None = None):
        super().__init__(parent)
        self._name = name
        self._cap = max(1.0, float(cap_mb))
        self._mb = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(120, 18)

    def set_cap(self, cap_mb: float):
        self._cap = max(1.0, float(cap_mb))

    def set_mb(self, mb: float):
        if mb is None or mb < 0:
            self._mb = 0.0
        else:
            self._mb = float(mb)
        self.setToolTip(self.tr("{name}: {mb:.0f} MB / {cap:.0f} MB cap").format(
            name=self._name, mb=self._mb, cap=self._cap,
        ))
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        radius = h / 2.0
        # Track — translucent fill + 1 px palette-aware outline so the
        # bar's full extent stays visible even when the cap-fraction is
        # tiny. Without the outline a near-empty bar collapses into the
        # status bar background and the user can't tell the widget is
        # there at all.
        bg = QPainterPath()
        bg.addRoundedRect(0.5, 0.5, w - 1, h - 1, radius, radius)
        p.fillPath(bg, qcolor(COLOR_BG_OVERLAY_SOFT))
        p.save()
        p.setPen(qcolor(COLOR_OUTLINE_SUBTLE))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(bg)
        p.restore()
        # Fill
        t = self._mb / self._cap if self._cap > 0 else 0.0
        fw = max(0.0, min(1.0, t)) * w
        if fw > 1.0:
            fill = QPainterPath()
            fill.addRoundedRect(0, 0, fw, h, radius, radius)
            p.fillPath(fill, _lerp_color(t))
        # Wrench icon on worker bars only — would mislead on the GUI bar.
        icon_h = int(h * 0.65)
        icon_x = 6
        icon_y = (h - icon_h) // 2
        # Wrench only on the worst-worker bar — not GUI, not the Σ total.
        wrench_visible = self._name == "max"
        if wrench_visible:
            # Same colour as the label text so it tracks the theme (was a
            # static light COLOR_FONT_INVERSE — invisible on light theme).
            wrench = _wrench_pixmap(COLOR_FONT_PRIMARY, icon_h)
            if not wrench.isNull():
                target = QRectF(icon_x, icon_y, icon_h, icon_h)
                p.drawPixmap(target, wrench, QRectF(wrench.rect()))
            text_left = icon_x + icon_h + 4
        else:
            text_left = icon_x + 2
        label = f"{self._short_name()}: {self._mb:.0f} {self.tr('MB')}"
        font = p.font()
        font.setPixelSize(11)
        font.setBold(True)
        p.setFont(font)
        text_rect = self.rect().adjusted(text_left, 0, -4, 0)
        p.setPen(qcolor(COLOR_FONT_PRIMARY))
        p.drawText(text_rect,
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   label)
        p.end()

    def _short_name(self) -> str:
        n = self._name
        if n == "gui":
            return "GUI"
        # "Worker-Integrated-0" → "0"
        if "Worker-Integrated-" in n:
            return n.rsplit("-", 1)[-1]
        return n[:6]


class WorkerRssStrip(QWidget):
    """Row of `WorkerRssBar`s. Builds bars lazily when new worker names
    appear in an `update(...)` call so respawned workers under different
    pids share the same bar slot."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._cap = _kill_cap_mb()
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(4)
        # Three bars: GUI process (own cap), the single worst-case worker
        # (fixed 600 MB pressure scale), and a total = GUI + Σworkers
        # (scale = GUI cap + 600 MB × workers).
        self._gui_bar = WorkerRssBar("gui", self._cap)
        self._max_bar = WorkerRssBar("max", _WORKER_PRESSURE_CAP_MB)
        self._total_bar = WorkerRssBar("Σ", self._cap)
        self._row.addWidget(self._gui_bar)
        self._row.addWidget(self._max_bar)
        self._row.addWidget(self._total_bar)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def update_values(self, values: dict[str, float]):
        if not values:
            return
        # Last poll kept for the bug report (same numbers the user sees).
        self.last_values = dict(values)
        gui_mb = values.get("gui", 0.0)
        worker_vals = [v for k, v in values.items() if k != "gui"]
        self._gui_bar.set_mb(gui_mb)
        if worker_vals:
            self._max_bar.set_mb(max(worker_vals))
            lines = [f"{k}: {v:.0f} MB" for k, v in sorted(values.items()) if k != "gui"]
            self._max_bar.setToolTip(
                self.tr("worst worker · 100% = {cap:.0f} MB\n").format(
                    cap=_WORKER_PRESSURE_CAP_MB) + "\n".join(lines))
        # Total app footprint: GUI + every worker. Scale grows with the
        # live worker count so 100% = the worst-case sum.
        n = len(worker_vals)
        total_mb = gui_mb + sum(worker_vals)
        total_cap = self._cap + _WORKER_PRESSURE_CAP_MB * max(0, n)
        self._total_bar.set_cap(total_cap)
        self._total_bar.set_mb(total_mb)
        self._total_bar.setToolTip(self.tr(
            "total app · {mb:.0f} MB / {cap:.0f} MB "
            "(GUI + {n} worker(s))").format(mb=total_mb, cap=total_cap, n=n))
        self._gui_bar.setToolTip(self.tr("GUI: {mb:.0f} MB / {cap:.0f} MB cap").format(
            mb=gui_mb, cap=self._cap,
        ))


class PipelineProgressBar(QWidget):
    """Compact label + filled bar. Scan counts kept here as plain ints —
    the MainWindow feeds `set_totals(imported)` on scan_imported and
    `mark_done(scan_id)` on branch_ready.

    Timing:
      * `_start_t` is stamped when the first work signal lands (import
        or mark_done). Used for total-elapsed display once done.
      * `_done_times` keeps the last `_ETA_WINDOW` mark_done timestamps;
        their pairwise deltas give a moving-average scan throughput
        that drives the ETA shown next to the percentage.
      * Until at least 2 mark_done events arrive, ETA shows as "—".
    """

    _ETA_WINDOW = 8  # scan timestamps kept for the moving-average

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._imported = 0
        self._imported_snaps: set[int] = set()
        self._done_snaps: set[int] = set()
        self._start_t: Optional[float] = None
        self._done_t: Optional[float] = None  # frozen at 100%
        self._done_times: deque[float] = deque(maxlen=self._ETA_WINDOW)
        self._label_prefix = self.tr("Pipeline")
        self._tick_count = 0
        # Indeterminate ("activity") mode for progress with no per-item ticks.
        self._indeterminate = False
        self._indet_phase = 0.0
        self._indet_label = ""
        self._indet_timer = QTimer(self)
        self._indet_timer.setInterval(33)   # ~30 fps sweep
        self._indet_timer.timeout.connect(self._tick_indeterminate)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_indeterminate(self, on: bool, label: str = "") -> None:
        """Toggle the moving 'activity' band (no percentage). Use while a
        slow op has produced no progress ticks yet."""
        self._indeterminate = bool(on)
        if on:
            self._indet_label = label or self.tr("Working…")
            if not self._indet_timer.isActive():
                self._indet_timer.start()
        else:
            self._indet_timer.stop()
        self.update()

    def _tick_indeterminate(self) -> None:
        self._indet_phase = (self._indet_phase + 0.016) % 1.0
        self.update()

    def set_label_prefix(self, prefix: str) -> None:
        """Switch the head label between e.g. 'Pipeline' and 'OCR'.

        Reused for the OCR pass so the bar keeps the same widget identity
        and ETA accounting, only the head text changes."""
        self._label_prefix = prefix
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(260, 18)

    def reset(self):
        self._imported = 0
        self._imported_snaps.clear()
        self._done_snaps.clear()
        self._start_t = None
        self._done_t = None
        self._done_times.clear()
        self._tick_count = 0
        self._indeterminate = False
        if self._indet_timer.isActive():
            self._indet_timer.stop()
        self.update()

    def set_imported(self, n: int):
        """Force the running total. Used by load_existing_scans /
        reprocess paths where scan_imported events don't fire (the
        scans already exist in DB and skip the importer)."""
        self._reset_if_finished()
        self._imported = int(n)
        self._stamp_start_if_needed()
        self.update()

    def bump_imported_if_below(self, n: int):
        """Raise the running total to at least n. For paths that may run
        in parallel with scan_imported events — never undercount."""
        if int(n) > self._imported:
            self._reset_if_finished()
            self._imported = int(n)
            self._stamp_start_if_needed()
            self.update()

    def increment_imported(self, n: int = 1):
        """Called per scan_imported. Auto-resets the bar if a prior run
        had already hit 100 % — otherwise a fresh capture / import would
        land into a frozen "Done" label and look like a stuck session."""
        self._reset_if_finished()
        self._imported += int(n)
        self._stamp_start_if_needed()
        self.update()

    def note_imported(self, scan_id) -> None:
        """Dedup'd import count by scan_id — a re-emitted scan_imported
        event (catchup / reprocess) must not inflate the total past the
        real scan count (the 94-scans-shows-96 bug)."""
        self._reset_if_finished()
        if scan_id is None:
            self._imported += 1            # unknown id — best-effort
        elif scan_id not in self._imported_snaps:
            self._imported_snaps.add(scan_id)
            self._imported = max(self._imported, len(self._imported_snaps))
        self._stamp_start_if_needed()
        self.update()

    def mark_tick(self) -> None:
        """Increment done by 1 without dedupe. Pipeline mode counts per
        scan (one branch_ready per scan_id) so `mark_done` dedupes by
        scan_id; OCR mode counts per branch (a multi-branch scan should
        tick twice). This path bypasses the set, using a private
        monotonic counter so the two modes don't collide."""
        self._tick_count += 1
        # Mirror into `_done_snaps` so `_ratio` / final-state stamping
        # keep working without splitting their plumbing.
        self._done_snaps.add(-self._tick_count)
        self._stamp_start_if_needed()
        self._done_times.append(time.monotonic())
        if (self._imported > 0
                and len(self._done_snaps) >= self._imported
                and self._done_t is None):
            self._done_t = time.monotonic()
        self.update()

    def mark_done(self, scan_id: Optional[int]):
        if scan_id is None:
            return
        sid = int(scan_id)
        if sid in self._done_snaps:
            return  # dedup — same branch may emit branch_ready twice
        self._done_snaps.add(sid)
        self._stamp_start_if_needed()
        self._done_times.append(time.monotonic())
        # Freeze the total-elapsed timestamp once we hit 100 %.
        if (self._imported > 0
                and len(self._done_snaps) >= self._imported
                and self._done_t is None):
            self._done_t = time.monotonic()
        self.update()

    def is_finished(self) -> bool:
        """True once the bar has reached (or been snapped to) 100 %."""
        return (self._imported > 0
                and len(self._done_snaps) >= self._imported
                and self._done_t is not None)

    def force_complete(self) -> None:
        """Snap the bar to finished. Called when the pipeline has drained while
        the event-counted ``done`` is still below ``total`` — a desync that
        would otherwise leave the bar stuck at e.g. 309/311 forever even though
        no work remains (partial-open edge cases, work added/skipped mid-run, a
        lost branch_ready event). An idle chain means whatever's on disk is the
        final state, so reconcile the label to 100 %."""
        if self._imported <= 0:
            return
        pad = self._imported - len(self._done_snaps)
        if pad > 0:
            # Sentinels well clear of real scan_ids (positive) and mark_tick's
            # small negatives, so dedup/ratio keep working.
            base = -1_000_000
            for k in range(pad):
                self._done_snaps.add(base - k)
        if self._done_t is None:
            self._done_t = time.monotonic()
        self.update()

    def _stamp_start_if_needed(self):
        if self._start_t is None:
            self._start_t = time.monotonic()

    def _reset_if_finished(self):
        """If the previous session reached 100 % and froze, clear the
        counters so the next piece of work starts a fresh bar instead
        of stacking on top of the old "Done" label."""
        if self._done_t is not None:
            self.reset()

    def _ratio(self) -> float:
        if self._imported <= 0:
            return 0.0
        return min(1.0, len(self._done_snaps) / self._imported)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Compact h:mm:ss / m:ss / Ns rendering for the label."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    def _moving_avg_dt(self) -> Optional[float]:
        """Average seconds-per-scan over the last `_ETA_WINDOW`
        completions. Needs ≥ 2 timestamps to produce any value."""
        if len(self._done_times) < 2:
            return None
        ts = list(self._done_times)
        deltas = [b - a for a, b in zip(ts, ts[1:])]
        if not deltas:
            return None
        return sum(deltas) / len(deltas)

    def _build_label(self) -> str:
        done = len(self._done_snaps)
        total = self._imported
        pct = self._ratio() * 100.0
        head = f"{self._label_prefix} · {done}/{total}  ({pct:.0f}%)"
        # Final state — replace ETA with actual elapsed + s/scan.
        if (total > 0 and done >= total and self._start_t is not None
                and self._done_t is not None):
            elapsed = self._done_t - self._start_t
            per_scan = elapsed / total if total > 0 else 0.0
            return self.tr("{head}  ·  {elapsed} · {per_scan:.1f}s/scan").format(
                head=head, elapsed=self._fmt_duration(elapsed), per_scan=per_scan,
            )
        # Running — show ETA from moving-average dt.
        avg_dt = self._moving_avg_dt()
        if avg_dt is None or total <= 0 or done >= total:
            eta_str = "—"
        else:
            eta_str = self._fmt_duration(avg_dt * (total - done))
        return self.tr("{head}  ·  ETA {eta}").format(head=head, eta=eta_str)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # Gentle corners (was a full h/2 pill — too round).
        radius = min(7.0, h / 2.0)
        # Match WorkerRssBar: translucent fill + palette outline so the
        # full extent stays visible on both light + dark.
        bg = QPainterPath()
        bg.addRoundedRect(0.5, 0.5, w - 1, h - 1, radius, radius)
        p.fillPath(bg, qcolor(COLOR_BG_OVERLAY_SOFT))
        p.save()
        p.setPen(qcolor(COLOR_OUTLINE_SUBTLE))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(bg)
        p.restore()
        if getattr(self, "_indeterminate", False):
            # Moving "activity" band so single-shot / slow-first-batch OCR
            # (Mistral upload, first Surya/Paddle batch) doesn't read as a
            # stuck 0%.
            band_w = max(40.0, w * 0.30)
            cx = self._indet_phase * (w + band_w) - band_w
            clip = QPainterPath()
            clip.addRoundedRect(0.5, 0.5, w - 1, h - 1, radius, radius)
            p.save()
            p.setClipPath(clip)
            band = QPainterPath()
            band.addRoundedRect(cx, 0, band_w, h, radius, radius)
            p.fillPath(band, qcolor(COLOR_PRIMARY))
            p.restore()
            label = self._indet_label
        else:
            t = self._ratio()
            fw = t * w
            if fw > 1.0:
                fill = QPainterPath()
                fill.addRoundedRect(0, 0, fw, h, radius, radius)
                p.fillPath(fill, qcolor(COLOR_PRIMARY))
            label = self._build_label()
        font = p.font()
        font.setPixelSize(11)
        font.setBold(True)
        p.setFont(font)
        p.setPen(qcolor(COLOR_FONT_PRIMARY))
        p.drawText(self.rect(),
                   int(Qt.AlignmentFlag.AlignCenter), label)
        p.end()


class LogStrip(QLabel):
    """Single-line clickable label showing the most recent log entry.
    Emits `clicked` so the MainWindow can open a Log tab."""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._full_text = self.tr("ready.")
        self._prefix_pix: QPixmap | None = None
        super().setText(self._full_text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QLabel {{ color: {COLOR_FONT_PRIMARY}; padding: 0 8px; }}"
            f"QLabel:hover {{ color: {COLOR_FONT_LINK_HOVER}; text-decoration: underline; }}"
        )
        # Ignored width: the natural text width was leaking through to
        # the status bar's sizeHint and pushing the main window past the
        # screen on long log lines. The label paints whatever fits, the
        # rest is elided.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        f = QFont()
        f.setPixelSize(11)
        self.setFont(f)
        # Coalesce log updates: a full reprocess fires push() thousands of
        # times. Doing setText + repaint per call floods the event loop and
        # starves user input (unresponsive UI). Store the latest line and flush
        # to the label at ~12 Hz; push() itself stays trivial so the queued
        # log-signal backlog drains fast.
        self._pending_log: Optional[str] = None
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(80)
        self._log_flush_timer.timeout.connect(self._flush_log)

    def _flush_log(self) -> None:
        if self._pending_log is None:
            self._log_flush_timer.stop()
            return
        text = self._pending_log
        self._pending_log = None
        self.setText(text)
        self.setToolTip(text)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt API
        return QSize(0, QFontMetrics(self.font()).height() + 4)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt API
        return self.sizeHint()

    def set_prefix_icon(self, pix: QPixmap | None):
        """Pixmap painted to the left of the text in the same widget so
        the icon stays glued to the log line (no row-spacing gap)."""
        self._prefix_pix = pix
        if pix is not None and not pix.isNull():
            dpr = max(int(pix.devicePixelRatio()), 1)
            iw = pix.width() // dpr
            # Bake left-padding = base 8 px + icon room so the QLabel's
            # text never overlaps the icon.
            self.setStyleSheet(
                f"QLabel {{ color: {COLOR_FONT_PRIMARY}; padding: 0 8px 0 {8 + iw + 4}px; }}"
                f"QLabel:hover {{ color: {COLOR_FONT_LINK_HOVER}; text-decoration: underline; }}"
            )
        self._apply_elided()
        self.update()

    def setText(self, text: str):  # noqa: N802 — Qt API
        self._full_text = text
        self._apply_elided()

    def _apply_elided(self):
        metrics = QFontMetrics(self.font())
        icon_room = 0
        if self._prefix_pix is not None and not self._prefix_pix.isNull():
            dpr = max(int(self._prefix_pix.devicePixelRatio()), 1)
            icon_room = self._prefix_pix.width() // dpr + 4
        avail = max(0, self.width() - 16 - icon_room)
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideRight, avail)
        super().setText(elided)

    def paintEvent(self, ev):  # noqa: N802 — Qt API
        super().paintEvent(ev)
        pix = self._prefix_pix
        if pix is None or pix.isNull():
            return
        dpr = max(int(pix.devicePixelRatio()), 1)
        iw = pix.width() // dpr
        ih = pix.height() // dpr
        p = QPainter(self)
        y = (self.height() - ih) // 2
        p.drawPixmap(8, y, iw, ih, pix)
        p.end()

    def resizeEvent(self, ev):  # noqa: N802 — Qt API
        super().resizeEvent(ev)
        self._apply_elided()

    def push(self, level: str, text: str):
        compact = " ".join(text.strip().splitlines())
        prefix = {"warn": "⚠ ", "error": "✕ ", "info": ""}.get(level, "")
        # Trivial: just record the latest; the timer flushes to the label.
        self._pending_log = prefix + compact
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class StatusBarWidget(QWidget):
    """Composite bar wired by the MainWindow.

    Signals:
      * `log_clicked` — the strip was clicked; open the log tab.
      * `settings_clicked` — the leftmost settings button was clicked.
      * `stop_clicked` — the red stop icon next to the progress bar
        was clicked; MainWindow hard-stops the chain.
    """

    log_clicked = Signal()
    settings_clicked = Signal()
    stop_clicked = Signal()
    stop_ocr_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StatusBarWidget")
        self.setStyleSheet(
            f"QWidget#StatusBarWidget {{"
            f"  background-color: {COLOR_BG_SURFACE};"
            f"  border-top: 1px solid {COLOR_OUTLINE_SUBTLE};"
            f"}}"
        )
        # Leftmost gear button — currently fires a TBD toast via the
        # `settings_clicked` signal. Wired here rather than relying on
        # MainWindow to add it, so the bar stays self-contained.
        self.settings_btn = QToolButton()
        self.settings_btn.setAutoRaise(True)
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setToolTip(self.tr("Settings"))
        try:
            from aglaia.gui.theme import icon as _icon
            self.settings_btn.setIcon(_icon("settings"))
        except Exception:
            self.settings_btn.setText("⚙")
        self.settings_btn.clicked.connect(self.settings_clicked)

        self.rss = WorkerRssStrip()
        self.progress = PipelineProgressBar()
        # Stop pipeline button. Visible only while pipeline is running
        # (MainWindow toggles via `set_pipeline_running(bool)`). Red
        # circle-stop glyph reads as a destructive action.
        self.stop_btn = QToolButton()
        self.stop_btn.setAutoRaise(True)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setToolTip(
            self.tr(
                "Stop pipeline — terminates in-flight workers and discards "
                "queued scans."
            )
        )
        try:
            from aglaia.gui.theme import icon as _icon
            self.stop_btn.setIcon(_icon("circle-stop", color=COLOR_ERROR))
        except Exception:
            self.stop_btn.setText("■")
            self.stop_btn.setStyleSheet(f"color: {COLOR_ERROR};")
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.stop_btn.hide()
        # Mirror stop affordance for the OCR pass — same red circle-stop
        # glyph, separate signal. Visible only while OCR is running so a
        # user never confuses pipeline-stop with OCR-stop.
        self.stop_ocr_btn = QToolButton()
        self.stop_ocr_btn.setAutoRaise(True)
        self.stop_ocr_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_ocr_btn.setToolTip(
            self.tr("Stop OCR — cancels the run after the current page.")
        )
        try:
            from aglaia.gui.theme import icon as _icon
            self.stop_ocr_btn.setIcon(_icon("circle-stop", color=COLOR_ERROR))
        except Exception:
            self.stop_ocr_btn.setText("■")
            self.stop_ocr_btn.setStyleSheet(f"color: {COLOR_ERROR};")
        self.stop_ocr_btn.clicked.connect(self.stop_ocr_clicked)
        self.stop_ocr_btn.hide()
        self.log = LogStrip()
        self.log.clicked.connect(self.log_clicked)

        # Lucide "logs" glyph sat to the left of the log strip — visually
        # ties the strip to the matching tab so users associate the two.
        self._log_icon_lbl = QLabel()
        try:
            from aglaia.gui.theme import lucide_pixmap as _lp
            _lp_pix = _lp("logs", color=COLOR_FONT_PRIMARY, size=14)
            _lp_pix.setDevicePixelRatio(2.0)
            self._log_icon_lbl.setPixmap(_lp_pix)
        except Exception:
            self._log_icon_lbl.setText("≡")
        self._log_icon_lbl.setFixedSize(14, 14)
        self._log_icon_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._log_icon_lbl.mousePressEvent = (  # type: ignore[assignment]
            lambda ev: self.log_clicked.emit() if ev.button() == Qt.MouseButton.LeftButton else None
        )

        # LogStrip paints the icon inline right before the elided text.
        self.log.set_prefix_icon(self._log_icon_lbl.pixmap())

        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(8, 4, 8, 4)
        self._row.setSpacing(10)
        # Settings + tip both moved to the sidebar bottom block. The
        # status bar keeps RSS + progress + stop + log only.
        self.settings_btn.hide()
        self._row.addWidget(self.rss)
        self._row.addWidget(self.progress, 1)
        self._row.addWidget(self.stop_btn)
        self._row.addWidget(self.stop_ocr_btn)
        self._row.addWidget(self.log, 2)
        self.tip_btn = None

    def set_pipeline_running(self, running: bool) -> None:
        """Show/hide the stop button to match pipeline activity."""
        self.stop_btn.setVisible(bool(running))

    def set_ocr_running(self, running: bool) -> None:
        """Show/hide the OCR stop button to match OCR activity."""
        self.stop_ocr_btn.setVisible(bool(running))

    def use_capture_widgets(self, status_widget: QWidget):
        """Capture mode: keep the progress bar (so pipeline AND OCR progress
        stay visible) and the stop buttons; swap only the log strip for the
        capture status message. Voice control lives in the capture sidebar
        (its own toggle + transcript), not the bottom bar.
        """
        self._row.removeWidget(self.log)
        self.log.hide()
        self._row.addWidget(status_widget, 2)


class ToastLabel(QLabel):
    """Tiny self-hiding popup. Built lazily by `show_toast`. Floats over
    the parent widget at the bottom-centre and fades out after `ms`."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setStyleSheet(
            f"QLabel#Toast {{"
            f"  background-color: {COLOR_BG_TOAST};"
            f"  color: {COLOR_FONT_ON_TOAST};"
            f"  border: 1px solid {COLOR_OUTLINE};"
            f"  border-radius: 8px;"
            f"  padding: 8px 14px;"
            f"}}"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

    def pop(self, text: str, ms: int = 1800):
        self.setText(text)
        self.adjustSize()
        parent = self.parentWidget()
        if parent is not None:
            x = (parent.width() - self.width()) // 2
            y = parent.height() - self.height() - 60
            self.move(max(0, x), max(0, y))
        self.show()
        self.raise_()
        QTimer.singleShot(ms, self.hide)


def show_toast(parent: QWidget, text: str, ms: int = 1800):
    """Convenience: create-or-reuse a ToastLabel on `parent` and show
    `text` for `ms`. Reused via the `_toast` attribute on the parent."""
    toast = getattr(parent, "_toast", None)
    if toast is None or toast.parent() is not parent:
        toast = ToastLabel(parent)
        parent._toast = toast
    toast.pop(text, ms)


_LOG_LEVEL_COLORS = {
    "TRACE":   COLOR_FONT_DIM,
    "DEBUG":   COLOR_FONT_DIM,
    "INFO":    COLOR_FONT_PRIMARY,
    "SUCCESS": COLOR_SUCCESS,
    "OK":      COLOR_SUCCESS,
    "WARN":    COLOR_WARNING,
    "WARNING": COLOR_WARNING,
    "ERROR":   COLOR_ERROR,
    "ERR":     COLOR_ERROR,
    "FATAL":   COLOR_ERROR_STRONG,
    "CRITICAL": COLOR_ERROR_STRONG,
}


def _level_color(level: str) -> str:
    return _LOG_LEVEL_COLORS.get(level.upper(), COLOR_FONT_PRIMARY)


def _highlight_message(escaped: str) -> str:
    """Apply rich-style token highlighting on an already-HTML-escaped
    log message. Covers the patterns the CLI's rich console colourises:

      * ``[pipeline.X]`` step tags → primary blue
      * ``key=value`` pairs (scan/layout/method/angle/…) → key dim,
        value in success green
      * dimension triples like ``4000×3000@200`` → success green
      * trailing numeric + unit (``423ms``, ``+0.30°``, ``200dpi``)
        → success green
      * arrows ``→`` → muted
    """
    import re
    out = escaped
    # Step tag first — sits inside square brackets so do this before
    # generic number patterns can chew the digits inside ``4000×3000``.
    out = re.sub(
        r"\[(pipeline\.[\w.]+)\]",
        lambda m: (
            f"<span style='color:{COLOR_FONT_MUTED};'>[</span>"
            f"<span style='color:{COLOR_PRIMARY}; font-weight:600;'>{m.group(1)}</span>"
            f"<span style='color:{COLOR_FONT_MUTED};'>]</span>"
        ),
        out,
    )
    # key=value (whitespace-delimited).
    out = re.sub(
        r"\b(scan|layout|method|angle|blobs|workers|gui_pid|stem|stage|engine)=([^\s,]+)",
        lambda m: (
            f"<span style='color:{COLOR_FONT_MUTED};'>{m.group(1)}=</span>"
            f"<span style='color:{COLOR_SUCCESS}; font-weight:600;'>{m.group(2)}</span>"
        ),
        out,
    )
    # WxH@DPI dimensions. Accept both ``×`` and ``x``.
    out = re.sub(
        r"\b(\d+[×x]\d+(?:@\d+)?)",
        lambda m: (
            f"<span style='color:{COLOR_SUCCESS};'>{m.group(1)}</span>"
        ),
        out,
    )
    # Trailing number + unit. Match signed floats / ints.
    out = re.sub(
        r"([+-]?\d+(?:\.\d+)?)(ms|s|°|%|dpi|DPI|px|mb|MB)\b",
        lambda m: (
            f"<span style='color:{COLOR_SUCCESS}; font-weight:600;'>{m.group(1)}</span>"
            f"<span style='color:{COLOR_FONT_MUTED};'>{m.group(2)}</span>"
        ),
        out,
    )
    # Arrows.
    out = out.replace(
        "→",
        f"<span style='color:{COLOR_FONT_DIM};'>→</span>",
    )
    return out


def _format_log_html(level: str, text: str) -> str:
    """Render one log entry as a single `<div>` with the level tag in
    its severity colour and the message tokenised à la rich. Escapes
    the message — log lines often contain ``<`` / ``>`` from type repr
    or path fragments."""
    from html import escape
    lvl = level.upper()
    lvl_color = _level_color(lvl)
    msg = escape(text).replace("\n", "<br/>")
    msg = _highlight_message(msg)
    return (
        f"<div style='margin:0; padding:0;'>"
        f"<span style='color:{COLOR_FONT_MUTED};'>[</span>"
        f"<span style='color:{lvl_color}; font-weight:700;'>{lvl}</span>"
        f"<span style='color:{COLOR_FONT_MUTED};'>]</span> "
        f"<span style='color:{COLOR_FONT_PRIMARY};'>{msg}</span>"
        f"</div>"
    )


def _parse_seed_line(line: str) -> tuple[str, str]:
    """Buffer lines are pre-formatted ``[LEVEL] text``. Split back into
    (level, text); fall back to INFO if the prefix is missing or
    malformed."""
    if line.startswith("[") and "]" in line:
        end = line.index("]")
        return line[1:end], line[end + 1:].lstrip()
    return "INFO", line


class LogViewerWidget(QWidget):
    """Read-only scrollback over the live console log. Built once per
    "Log" tab; appends new lines as the ProcessMonitor emits them.

    Each entry rendered as one rich-HTML `<div>` so the level tag picks
    up a severity colour (red for errors, amber for warnings, dim grey
    for debug). Mirrors the colourisation the CLI gets from `rich`.

    The MainWindow keeps a rolling buffer (deque) of N recent lines and
    seeds the viewer with them on construction, so opening the tab after
    a lot of activity doesn't show an empty pane."""

    def __init__(self, seed: deque[str], parent: QWidget | None = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        # Soft-wrap long lines so the user never has to scroll
        # horizontally just to read a single message. Wrap inside the
        # widget width — anywhere mid-word is fine since the log is
        # dense + monospaced.
        self.view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.view.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPixelSize(11)
        self.view.setFont(mono)
        if seed:
            html = "".join(
                _format_log_html(*_parse_seed_line(line)) for line in seed
            )
            self.view.setHtml(html)
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().maximum()
            )
        v.addWidget(self.view)

    def append(self, level: str, text: str):
        self.view.append(_format_log_html(level, text))
