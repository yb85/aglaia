# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Per-node debug viewer for the Qt GUI.

Click a thumb in `ScanItemWidget`, get a closable tab inside the main
window that walks the node's chain (root → leaf) and renders every step
through the shared renderers (`aglaia/storage/debug_chain.py`). The strip on
the left shows step minis; the
main pane on the right shows the currently-selected step at its full
rendered size.

Public surface:

  * `DebugViewerWidget` — QWidget used inside the MainWindow tab strip.
  * `DebugViewerDialog` — thin QDialog wrapper kept for any caller that
    still wants a free-floating window (currently none in tree).
"""
from __future__ import annotations

import base64
import json
from typing import Optional

from PySide6.QtCore import QCoreApplication, QPointF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from aglaia.gui.DebugEditCanvas import EditCanvas
from aglaia.gui.ZoomCanvas import ZoomToolbar
from aglaia.gui.colors import (
    active_palette_name,
    COLOR_BG_ZEBRA_EVEN,
    COLOR_BG_ZEBRA_ODD,
    COLOR_ERROR,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_BUTTON,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_STRONG,
    COLOR_SUCCESS,
    COLOR_WARNING,
)
from aglaia.storage.db import db_session, open_db
from aglaia.storage.debug_chain import _render_one, _walk_chain
from aglaia.storage.repo import ManualOverrideRepo
from aglaia.storage.debug_renderers import render_chain_overlays


class _OverlayJob(QThread):
    """Background worker that runs the per-processor renderers from
    ``debug_renderers`` and emits one list of ``{label, url}`` dicts
    when done. Heavy (trap + dewarp recompute spans on the raw ink) so
    we don't block the GUI thread."""

    done = Signal(list)
    failed = Signal(str)

    def __init__(self, db_path: str, leaf_node_id: int, parent=None):
        super().__init__(parent)
        self._db_path = db_path
        self._leaf = leaf_node_id

    def run(self) -> None:
        try:
            conn = open_db(self._db_path)
            try:
                images = render_chain_overlays(conn, self._leaf)
            finally:
                conn.close()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.done.emit(images)


def _draw_overlay(pix: QPixmap, processor: str, meta: dict) -> QPixmap:
    """Paint per-processor debug overlays on a copy of ``pix``.

    Currently supported:

    * ROI polygon (``meta['roi']``) — red outline. Every processor that
      tracks the image's content rect stores this.
    * SkewFinder ``skew`` — magenta tilt indicator + angle caption.
    * PageDewarper ``oob`` / ``fallback_reason`` — amber badge.
    * Status badge from ``meta['status']`` (Status enum int) — green /
      amber / red dot upper-left.

    Returns the original pixmap if nothing applies."""
    if pix is None or pix.isNull():
        return pix
    out = QPixmap(pix)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    w, h = out.width(), out.height()
    thick = max(2, int(min(w, h) / 320))

    roi = meta.get("roi")
    if isinstance(roi, list) and len(roi) >= 3:
        try:
            pts = [QPointF(float(x), float(y)) for (x, y) in roi]
            poly = QPolygonF(pts)
            pen = QPen(QColor(COLOR_ERROR))
            pen.setWidth(thick)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(poly)
        except Exception:
            pass

    proc = (processor or "").lower()
    skew = meta.get("skew")
    if "skew" in proc and isinstance(skew, (int, float)):
        import math
        # Tilt axis through image centre at the detected angle.
        cx, cy = w / 2, h / 2
        rad = math.radians(float(skew))
        half = max(w, h) * 0.4
        dx = half * math.cos(rad)
        dy = half * math.sin(rad)
        pen = QPen(QColor(COLOR_PRIMARY))
        pen.setWidth(thick)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(cx - dx, cy - dy), QPointF(cx + dx, cy + dy))
        # Caption upper-right.
        pad = max(8, int(min(w, h) / 120))
        _draw_caption(p, w - pad, pad,
                       QCoreApplication.translate(
                           "DebugViewer", "skew = {value:+.2f}°"
                       ).format(value=skew),
                       COLOR_PRIMARY, anchor_right=True,
                       image_size=(w, h))

    if "dewarp" in proc:
        pad = max(8, int(min(w, h) / 120))
        gap = max(28, int(min(w, h) / 30))
        if meta.get("fallback_reason"):
            _draw_caption(p, pad, h - gap,
                          QCoreApplication.translate(
                              "DebugViewer", "fallback: {reason}"
                          ).format(reason=meta.get("fallback_reason")),
                          COLOR_WARNING, image_size=(w, h))
        if meta.get("oob"):
            _draw_caption(p, pad, h - gap * 2,
                          QCoreApplication.translate("DebugViewer", "OOB"),
                          COLOR_WARNING,
                          image_size=(w, h))

    status = meta.get("status")
    if isinstance(status, int):
        # Status: 0=SUCCESS, 1=WARNING, 2=ERROR (matches Status enum).
        color = {0: COLOR_SUCCESS, 1: COLOR_WARNING, 2: COLOR_ERROR}.get(
            int(status), COLOR_FONT_MUTED
        )
        p.setBrush(QColor(color))
        p.setPen(Qt.PenStyle.NoPen)
        dot = max(14, int(min(w, h) / 80))
        pad = max(8, int(min(w, h) / 120))
        p.drawEllipse(pad, pad, dot, dot)

    p.end()
    return out


def _draw_caption(p: QPainter, x: float, y: float, text: str,
                   color: str, anchor_right: bool = False,
                   image_size: Optional[tuple[int, int]] = None) -> None:
    """Translucent black pill + coloured text. Used by every overlay
    that adds a corner label so callers don't repeat font / box code.
    Font scales with image dimensions so the caption stays readable on
    multi-megapixel scans (fixed 13 px is invisible at source res)."""
    if image_size is not None:
        iw, ih = image_size
        px = max(16, int(min(iw, ih) / 55))
    else:
        px = 13
    font = QFont()
    font.setPixelSize(px)
    font.setBold(True)
    p.setFont(font)
    fm = p.fontMetrics()
    pad_x = max(6, px // 2)
    pad_y = max(3, px // 4)
    tw = fm.horizontalAdvance(text) + pad_x * 2
    th = fm.height() + pad_y * 2
    if anchor_right:
        x = x - tw
    p.setBrush(QColor(0, 0, 0, 180))
    p.setPen(Qt.PenStyle.NoPen)
    radius = max(4, px // 3)
    p.drawRoundedRect(int(x), int(y), tw, th, radius, radius)
    p.setPen(QColor(color))
    p.drawText(int(x + pad_x), int(y + th - pad_y - fm.descent()), text)


#: Row background per PROCESSOR, not per row index (M9 #98).
#
# The strip used to zebra on the row INDEX, which carries no information: two
# adjacent look-alike stages got two different shades and two unrelated stages
# got the same one. Scrolling then had no visible effect, because consecutive
# stages often produce near-identical thumbnails (a deskew that found 0.2°, a
# passthrough) and the banding did not distinguish them either.
#
# Keyed by processor, the strip reads as BANDS — one colour per kind of stage —
# so any movement is immediate. The pipeline runs DPIfixer and SkewFinder twice
# each, so two consecutive rows of the same processor also get a light/dark
# variant (see `_band_for`); the pair reads as one band with a seam.
_BAND_HUES: dict[str, tuple[str, str]] = {
    "DPIfixer":              ("#2a2f3a", "#232833"),
    "SkewFinder":            ("#2b3327", "#242b21"),
    "PageDetector":          ("#3a2f26", "#312820"),
    "Binarizer":             ("#2a2a2e", "#232327"),
    "TrapezoidalCorrection": ("#33262f", "#2b2029"),
    "PageDewarper":          ("#26313a", "#202a33"),
    "MarginSetter":          ("#2e2a35", "#27232e"),
}
_BAND_LIGHT: dict[str, tuple[str, str]] = {
    "DPIfixer":              ("#e8ecf5", "#dde2ee"),
    "SkewFinder":            ("#e9f0e4", "#dee7d8"),
    "PageDetector":          ("#f5eae0", "#eddfd2"),
    "Binarizer":             ("#eeeef1", "#e4e4e9"),
    "TrapezoidalCorrection": ("#f4e6ed", "#ecdae3"),
    "PageDewarper":          ("#e2ecf5", "#d5e2ee"),
    "MarginSetter":          ("#ece7f3", "#e2dbec"),
}


def _band_for(processor: str, run: int) -> str:
    """Row background for `processor`, `run` counting repeats of that same
    processor down the chain. An unknown processor (a drop-in plugin) falls
    back to the old zebra, which is still better than nothing."""
    table = _BAND_HUES if active_palette_name() == "dark" else _BAND_LIGHT
    pair = table.get(processor)
    if pair is None:
        return COLOR_BG_ZEBRA_EVEN if run % 2 == 0 else COLOR_BG_ZEBRA_ODD
    return pair[run % 2]


class DebugViewerWidget(QWidget):
    """Strip-of-thumbs + zoomable main pane. Lives inside a tab; the
    enclosing `QTabWidget` owns close + lifecycle.

    Re-renders the full chain at construction time (~250 ms typical).
    Re-fits the current pixmap on resize so the picture always fills the
    available pane without manual scrollbar dancing.
    """

    #: Auto-process, remembered for the SESSION rather than persisted: it is
    #: a working mode, not a preference, and a user who turned it off to
    #: batch a page's edits does not want that decision to outlive the app.
    _AUTO_PROCESS_DEFAULT = True
    _auto_process_session: Optional[bool] = None

    def __init__(self, db_path: str, leaf_node_id: int,
                 title_hint: str = "", parent=None, *, reprocess_cb=None):
        super().__init__(parent)
        self.db_path = db_path
        self.leaf_node_id = leaf_node_id
        #: ``cb(scan_id, branch_path)`` — reruns one page-branch from its
        #: split point. Without it the editor still stores overrides; only
        #: the rerun is unavailable.
        self._reprocess_cb = reprocess_cb
        self._row_keys: list[tuple] = []
        self._overlay_geom: list[dict] = []
        self._pending_edit: dict = {}
        self._current_geom: dict = {}
        # Must exist before `_load()`: seeding the strip selects a row, which
        # paints through `_on_row_changed`.
        self._overlay_bytes: list[Optional[bytes]] = []
        self.title_hint = title_hint or self.tr("node {n}").format(n=leaf_node_id)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self.strip = QListWidget()
        self.strip.setFixedWidth(220)
        # Title-above-thumb rows are drawn via custom item widgets, so
        # the built-in icon slot is unused. Drop spacing/margins so the
        # zebra-banded rows sit flush.
        self.strip.setSpacing(0)
        self.strip.setStyleSheet(
            f"QListWidget {{ border: 1px solid {COLOR_OUTLINE_BUTTON}; }}"
            "QListWidget::item { padding: 0px; border: none; }"
            f"QListWidget::item:selected {{ background-color: {COLOR_PRIMARY_BG_STRONG}; }}"
        )
        self.strip.currentRowChanged.connect(self._on_row_changed)
        root.addWidget(self.strip)

        # Right pane: reusable ZoomCanvas (fit + PiP-on-hover), plus a
        # small zoom toolbar above it. Replaces the old QScrollArea +
        # QLabel pair that re-fit on resize manually.
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(4)
        self.canvas = EditCanvas(placeholder=self.tr("Select a step"))
        self.canvas.edited.connect(self._on_canvas_edited)
        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 2, 0)
        self.zoom_bar = ZoomToolbar(self.canvas, default=2.0)
        bar.addWidget(self.zoom_bar)
        bar.addStretch(1)
        # This view exists to show the debug data. Overlays are the point,
        # not an option, so there is no toggle: they replace the bare image
        # as soon as the background renderer is done (M9 #97).
        self._overlay_note = QLabel(self.tr("Rendering overlays…"))
        self._overlay_note.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        bar.addWidget(self._overlay_note)
        right_v.addLayout(bar)
        right_v.addWidget(self.canvas, 1)
        right_v.addWidget(self._build_editor())
        # Meta panel below the image — shows the node's ``meta_json``
        # dict pretty-printed. Lets the user inspect the diagnostic
        # data the overlay is rendering from (angle, status, ROI, …)
        # without re-hitting the DB.
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setFixedHeight(120)
        self.meta_view.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background: transparent;"
            f"  color: {COLOR_FONT_MUTED};"
            f"  border-top: 1px solid {COLOR_OUTLINE_BUTTON};"
            f"  font-size: 11px;"
            f"}}"
        )
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPixelSize(11)
        self.meta_view.setFont(mono)
        right_v.addWidget(self.meta_view)
        root.addWidget(right, 1)

        # Row widgets parallel `self.strip.item(i)`; tracked so the
        # selection-frame paint can target the active row directly
        # (setItemWidget masks the built-in `item:selected` highlight,
        # so the visible "selected" cue has to live on the row widget).
        # Must be initialised BEFORE `_load()` because `_build_strip_row`
        # appends to them per row.
        self._row_widgets: list[QWidget] = []
        self._row_zebra: list[str] = []
        self._load()
        # Kick off the overlay renderer after the bare images are up.
        # Slot fires on the GUI thread; pixmaps swap as soon as the
        # toggle is flipped.
        self._overlay_job = _OverlayJob(self.db_path, self.leaf_node_id, self)
        self._overlay_job.done.connect(self._on_overlay_ready)
        self._overlay_job.failed.connect(self._on_overlay_failed)
        self._overlay_job.start()

    #: Slider ranges, chosen for MANUAL tuning rather than for the fit's own
    #: freedom. The solver clamps curl at ±0.5 (`sheet_models.CURL_CLIP`) but
    #: a page past ±0.35 is already extreme, and a full-width slider over the
    #: solver's range would make every useful value a two-pixel move. Skew is
    #: the deskew step's own ±15° working band; γ is the spine term of #60,
    #: whose automatic grid searches ±0.10.
    _RANGES = {
        "skew_deg": (-15.0, 15.0, 0.05),
        "alpha": (-0.35, 0.35, 0.002),
        "beta": (-0.35, 0.35, 0.002),
        "gamma": (-0.15, 0.15, 0.002),
    }

    def _build_editor(self) -> QWidget:
        """Per-stage manual controls, plus the run mode.

        A stack rather than a growing column: only one stage is selected, and
        showing three sets of dead sliders would suggest they all apply."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(2, 4, 2, 2)
        v.setSpacing(4)

        self._editor_stack = QStackedWidget()
        self._editor_pages: dict[str, int] = {}

        blank = QLabel(self.tr("This step has no manual parameters."))
        blank.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        self._editor_pages[""] = self._editor_stack.addWidget(blank)

        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}

        skew_page, _ = self._slider_page([("skew_deg", self.tr("Angle"))],
                                         self.tr("Drag the handle on the "
                                                 "image, or use the slider."))
        self._editor_pages["SkewFinder"] = self._editor_stack.addWidget(skew_page)

        roi_page = QWidget()
        rv = QVBoxLayout(roi_page)
        rv.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(self.tr("Drag a vertex to reshape the page ROI. "
                              "Double-click an edge to add one."))
        hint.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        hint.setWordWrap(True)
        rv.addWidget(hint)
        self._editor_pages["PageDetector"] = self._editor_stack.addWidget(roi_page)

        curl_page, _ = self._slider_page(
            [("alpha", self.tr("Curl α")), ("beta", self.tr("Curl β")),
             ("gamma", self.tr("Spine γ"))],
            self.tr("The sheet is frozen at these; the pose is re-fitted."))
        self._editor_pages["PageDewarper"] = self._editor_stack.addWidget(curl_page)
        v.addWidget(self._editor_stack)

        run = QHBoxLayout()
        run.setContentsMargins(0, 0, 0, 0)
        self.auto_chk = QCheckBox(self.tr("Auto-process"))
        self.auto_chk.setChecked(self._auto_process_session
                                 if self._auto_process_session is not None
                                 else self._AUTO_PROCESS_DEFAULT)
        self.auto_chk.setToolTip(self.tr(
            "Rerun this page as soon as a parameter changes. Turn it off to "
            "make several edits and run once."))
        self.auto_chk.toggled.connect(self._on_auto_toggled)
        run.addWidget(self.auto_chk)
        self.reprocess_btn = QPushButton(self.tr("Reprocess"))
        self.reprocess_btn.clicked.connect(lambda: self._reprocess(force=True))
        run.addWidget(self.reprocess_btn)
        run.addStretch(1)
        self._manual_note = QLabel("")
        self._manual_note.setStyleSheet(
            f"color: {COLOR_WARNING}; font-size: 11px;")
        run.addWidget(self._manual_note)
        self._clear_btn = QPushButton(self.tr("Clear override"))
        self._clear_btn.clicked.connect(self._clear_override)
        run.addWidget(self._clear_btn)
        v.addLayout(run)
        self._sync_run_controls()
        return box

    def _slider_page(self, fields, hint_text: str):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        hint = QLabel(hint_text)
        hint.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        v.addWidget(hint)
        for key, label in fields:
            lo, hi, step = self._RANGES[key]
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            name = QLabel(label)
            name.setFixedWidth(64)
            name.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
            row.addWidget(name)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setMinimum(0)
            sl.setMaximum(int(round((hi - lo) / step)))
            sl.valueChanged.connect(
                lambda raw, k=key: self._on_slider(k, raw))
            row.addWidget(sl, 1)
            val = QLabel("—")
            val.setFixedWidth(56)
            val.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-size: 11px;")
            row.addWidget(val)
            self._sliders[key] = sl
            self._slider_labels[key] = val
            v.addLayout(row)
        return page, None

    def _load(self):
        conn = open_db(self.db_path)
        try:
            chain = _walk_chain(conn, self.leaf_node_id)
            images: list[dict] = []
            keys: list[tuple] = []
            for node in chain:
                if node.get("processor_name") is None:
                    continue
                rendered = _render_one(conn, node)
                images.extend(rendered)
                # `branch_path` is the node's `branch_label` (or "" for the
                # pre-split trunk) — the same rule `step_overrides` uses.
                keys.extend([(node.get("scan_id"),
                              str(node.get("branch_label") or ""),
                              node.get("processor_name") or "")] * len(rendered))
            self._row_keys = keys
        finally:
            conn.close()

        runs: dict[str, int] = {}
        for i, im in enumerate(images):
            label = im.get("label") or self.tr("step {n}").format(n=i)
            meta = im.get("meta") or {}
            processor = im.get("processor") or ""
            # Memory: a full-res RGBA QPixmap is ~45 MB per step; a dozen
            # steps held that way costs 500+ MB. Keep only the COMPRESSED
            # PNG bytes per row and decode lazily on selection — the
            # thumbnail decode below is transient.
            url = im.get("url", "")
            raw: Optional[bytes] = None
            if url.startswith("data:image/"):
                try:
                    raw = base64.b64decode(url.split(",", 1)[1])
                except Exception:
                    raw = None
            base_pix = None
            if raw is not None:
                img = QImage.fromData(raw)
                if not img.isNull():
                    base_pix = QPixmap.fromImage(img)
            thumb = (base_pix.scaledToWidth(
                self.THUMB_BOX * 2, Qt.TransformationMode.SmoothTransformation)
                if base_pix is not None else None)
            del base_pix  # full-res copy not retained
            # `run` counts repeats of this processor down the chain, so the
            # pipeline's two DPIfixers and two SkewFinders stay distinguishable
            # inside their own band.
            run = runs.get(processor, 0)
            runs[processor] = run + 1
            row_w = self._build_strip_row(label, thumb, processor, run)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, raw)
            item.setData(Qt.ItemDataRole.UserRole + 1, meta)
            item.setData(Qt.ItemDataRole.UserRole + 2, processor)
            item.setSizeHint(row_w.sizeHint())
            self.strip.addItem(item)
            self.strip.setItemWidget(item, row_w)
            self._row_widgets.append(row_w)

        if images:
            self.strip.setCurrentRow(len(images) - 1)

    #: Thumb box in the strip. Small on purpose: a ten-step chain at the old
    #: 200-px portrait thumb needed ~3700 px of strip, so the whole view was a
    #: scroll with no landmarks. At this size the default `book_curved_x2`
    #: chain fits a normal window without scrolling at all (M9 #98).
    THUMB_BOX = 56

    def _build_strip_row(self, label: str, pix: Optional[QPixmap],
                         processor: str, run: int) -> QWidget:
        """Per-step row: thumbnail left, step label right, on a background
        that identifies the PROCESSOR (see `_band_for`)."""
        w = QWidget()
        w.setObjectName("debugRow")
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(8)
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(self.THUMB_BOX, self.THUMB_BOX)
        if pix is not None:
            thumb_lbl.setPixmap(pix.scaled(
                self.THUMB_BOX, self.THUMB_BOX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(thumb_lbl)
        title = QLabel(label)
        title.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: bold; font-size: 11px;")
        title.setWordWrap(True)
        h.addWidget(title, 1)
        # The band lives on the row container so it stays visible under the
        # (transparent) child labels.
        w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bg = _band_for(processor, run)
        self._row_zebra.append(bg)
        w.setStyleSheet(
            f"QWidget#debugRow {{ background-color: {bg}; "
            f"border: 2px solid transparent; border-radius: 4px; }}"
        )
        return w

    def _on_row_changed(self, row: int):
        if row < 0 or row >= self.strip.count():
            return
        # Repaint row borders: selected row gets primary frame, others
        # transparent (preserves zebra fill). Must rewrite full QSS each
        # call because setStyleSheet replaces, not merges.
        for i, rw in enumerate(self._row_widgets):
            bg = self._row_zebra[i] if i < len(self._row_zebra) else COLOR_BG_ZEBRA_EVEN
            border = COLOR_PRIMARY if i == row else "transparent"
            rw.setStyleSheet(
                f"QWidget#debugRow {{ background-color: {bg}; "
                f"border: 2px solid {border}; border-radius: 4px; }}"
            )
        item = self.strip.item(row)
        meta_d = item.data(Qt.ItemDataRole.UserRole + 1)
        meta_d = meta_d if isinstance(meta_d, dict) else {}
        # Overlays are the point of this view, so the per-processor composite
        # wins as soon as it exists. The bare image + the light Qt overlay is
        # what is shown while the background render is still running.
        pix = None
        if (0 <= row < len(self._overlay_bytes)
                and self._overlay_bytes[row]):
            pix = self._decode(self._overlay_bytes[row])
        if pix is None:
            raw = item.data(Qt.ItemDataRole.UserRole)
            pix = self._decode(raw)
            if pix is not None and meta_d:
                processor = item.data(Qt.ItemDataRole.UserRole + 2) or ""
                pix = _draw_overlay(pix, processor, meta_d)
        if isinstance(pix, QPixmap):
            self.canvas.set_image(pix)
        self._configure_editor(row)
        meta = item.data(Qt.ItemDataRole.UserRole + 1)
        if isinstance(meta, dict) and meta:
            self.meta_view.setPlainText(
                json.dumps(meta, indent=2, ensure_ascii=False, default=str)
            )
        else:
            self.meta_view.setPlainText("")

    @staticmethod
    def _decode(raw) -> Optional[QPixmap]:
        if not raw:
            return None
        img = QImage.fromData(bytes(raw))
        if img.isNull():
            return None
        return QPixmap.fromImage(img)

    def _on_overlay_ready(self, images: list) -> None:
        """Background renderer finished. Keep only the COMPRESSED bytes
        per row (decode happens lazily on selection) and enable the
        toggle. Order must match the bare strip (renderer skips the same
        root node and walks the chain in identical order)."""
        blobs: list[Optional[bytes]] = []
        for im in images:
            url = im.get("url", "")
            raw = None
            if url.startswith("data:image/"):
                try:
                    raw = base64.b64decode(url.split(",", 1)[1])
                except Exception:
                    raw = None
            blobs.append(raw)
        geoms = [(im.get("geom") or {}) for im in images]
        # Pad/truncate to match the bare strip count.
        target = self.strip.count()
        while len(blobs) < target:
            blobs.append(None)
        while len(geoms) < target:
            geoms.append({})
        self._overlay_bytes = blobs[:target]
        self._overlay_geom = geoms[:target]
        self._overlay_note.setText("")
        # Redraw the selected row now that the overlay exists.
        self._on_row_changed(self.strip.currentRow())

    def _on_overlay_failed(self, err: str) -> None:
        self._overlay_note.setText(
            self.tr("Overlay render failed: {err}").format(err=err))
        self._overlay_note.setStyleSheet(
            f"color: {COLOR_ERROR}; font-size: 11px;")

    # ── refresh after a rerun ─────────────────────────────────────
    def scan_key(self) -> Optional[tuple]:
        """``(scan_id, branch_path)`` this viewer is showing, from its LEAF
        row — the deepest node of the chain, whose branch is the page."""
        for key in reversed(self._row_keys):
            if key[0] is not None:
                return (int(key[0]), key[1])
        return None

    def reload_for(self, scan_id: int, branch_path: str,
                   node_id: Optional[int] = None) -> bool:
        """Rebuild the view after that page-branch was reprocessed.

        A rerun WIPES the branch's subtree and writes fresh nodes, so this
        viewer's `leaf_node_id` is a dead row: without this the tab keeps
        showing the pre-edit chain and the editor looks broken (M9 #97).
        Returns whether it applied.
        """
        key = self.scan_key()
        if key is None or key[0] != int(scan_id):
            return False
        if key[1] and str(branch_path or "") and key[1] != str(branch_path):
            return False
        leaf = node_id or self._resolve_leaf(int(scan_id), key[1])
        if leaf is None:
            return False
        self.leaf_node_id = int(leaf)
        row = self.strip.currentRow()
        self._rebuild(keep_row=row)
        return True

    def _resolve_leaf(self, scan_id: int, branch_path: str) -> Optional[int]:
        """Deepest node of that branch — the chain's new terminal."""
        try:
            with db_session(self.db_path) as conn:
                sql = ("SELECT id FROM nodes WHERE scan_id = ? "
                       + ("AND branch_label = ? " if branch_path else "")
                       + "ORDER BY step_idx DESC, id DESC LIMIT 1")
                args = ((scan_id, branch_path) if branch_path else (scan_id,))
                row = conn.execute(sql, args).fetchone()
                return int(row["id"]) if row else None
        except Exception:
            return None

    def _rebuild(self, *, keep_row: int = -1) -> None:
        """Re-read the chain and restart the overlay render, in place."""
        job = getattr(self, "_overlay_job", None)
        if job is not None and job.isRunning():
            # Its `done` would otherwise land on the NEW strip with the OLD
            # chain's images and mis-pair every row with its geometry.
            try:
                job.done.disconnect()
                job.failed.disconnect()
            except Exception:
                pass
        self.strip.blockSignals(True)
        self.strip.clear()
        self.strip.blockSignals(False)
        self._row_widgets.clear()
        self._row_zebra.clear()
        self._overlay_bytes = []
        self._overlay_geom = []
        self._overlay_note.setText(self.tr("Rendering overlays…"))
        self._load()
        if 0 <= keep_row < self.strip.count():
            self.strip.setCurrentRow(keep_row)
        self._overlay_job = _OverlayJob(self.db_path, self.leaf_node_id, self)
        self._overlay_job.done.connect(self._on_overlay_ready)
        self._overlay_job.failed.connect(self._on_overlay_failed)
        self._overlay_job.start()

    # ── manual tuning (M9 #97) ────────────────────────────────────
    def _row_key(self, row: int):
        """``(scan_id, branch_path, processor)`` for a strip row, or None."""
        if 0 <= row < len(self._row_keys):
            key = self._row_keys[row]
            if key[0] is not None:
                return key
        return None

    def _stored(self, row: int) -> dict:
        key = self._row_key(row)
        if key is None:
            return {}
        try:
            with db_session(self.db_path) as conn:
                return ManualOverrideRepo(conn).get(int(key[0]), key[1])
        except Exception:
            return {}

    def _configure_editor(self, row: int) -> None:
        """Point the handle layer and the sliders at this row's stage."""
        key = self._row_key(row)
        processor = key[2] if key else ""
        geom = (self._overlay_geom[row]
                if 0 <= row < len(self._overlay_geom) else {}) or {}
        stored = self._stored(row)
        self._editor_stack.setCurrentIndex(
            self._editor_pages.get(processor, self._editor_pages[""]))
        self._current_geom = geom

        origin = geom.get("origin") or (0, 0)
        frame = geom.get("frame_wh")
        roi = stored.get("roi") or geom.get("roi")
        skew = stored.get("skew_deg")
        if skew is None:
            skew = geom.get("skew_deg")
        self.canvas.set_editable(
            polygon=roi if processor == "PageDetector" else None,
            rotation_deg=skew if processor == "SkewFinder" else None,
            origin=origin, frame_wh=frame,
            scale=float(geom.get("scale", 1.0) or 1.0))

        curl = stored.get("curl") or (geom.get("curl") or {})
        values = {"skew_deg": skew, "alpha": curl.get("alpha"),
                  "beta": curl.get("beta"), "gamma": curl.get("gamma")}
        for k, sl in self._sliders.items():
            v = values.get(k)
            lo, _hi, step = self._RANGES[k]
            sl.blockSignals(True)
            sl.setEnabled(v is not None)
            sl.setValue(0 if v is None else int(round((float(v) - lo) / step)))
            sl.blockSignals(False)
            self._slider_labels[k].setText(
                "—" if v is None else self._fmt(k, float(v)))
        self._manual_note.setText(
            self.tr("manual: {fields}").format(fields=", ".join(stored))
            if stored else "")
        self._sync_run_controls()

    @staticmethod
    def _fmt(key: str, value: float) -> str:
        return f"{value:+.2f}°" if key == "skew_deg" else f"{value:+.3f}"

    def _on_slider(self, key: str, raw: int) -> None:
        lo, _hi, step = self._RANGES[key]
        value = lo + raw * step
        self._slider_labels[key].setText(self._fmt(key, value))
        if key == "skew_deg":
            self.canvas.set_rotation_deg(value)
            self._store({"skew_deg": float(value)})
        else:
            curl = dict(self._stored(self.strip.currentRow()).get("curl")
                        or (self._current_geom.get("curl") or {}))
            curl[key] = float(value)
            self._store({"curl": curl})

    def _on_canvas_edited(self, kind: str, value) -> None:
        if kind == "skew_deg":
            lo, _hi, step = self._RANGES["skew_deg"]
            sl = self._sliders["skew_deg"]
            sl.blockSignals(True)
            sl.setEnabled(True)
            sl.setValue(max(sl.minimum(),
                            min(sl.maximum(),
                                int(round((float(value) - lo) / step)))))
            sl.blockSignals(False)
            self._slider_labels["skew_deg"].setText(
                self._fmt("skew_deg", float(value)))
            self._store({"skew_deg": float(value)})
        elif kind == "roi":
            self._store({"roi": value})

    def _store(self, fields: dict) -> None:
        """Persist an edit for the selected row, then rerun if asked to.

        Every spatial edit carries the frame it was made on — a polygon
        applied to a frame of another size would be silently shifted, and the
        consumer drops it rather than guessing (see `repo.validate_frame`)."""
        row = self.strip.currentRow()
        key = self._row_key(row)
        if key is None:
            return
        frame = (self._current_geom or {}).get("frame_wh")
        if frame and any(f in fields for f in ("roi",)):
            fields = dict(fields, frame_wh=list(frame))
        try:
            with db_session(self.db_path) as conn:
                stored = ManualOverrideRepo(conn).set(
                    int(key[0]), key[1], fields)
                conn.commit()
        except Exception:
            return
        self._manual_note.setText(
            self.tr("manual: {fields}").format(fields=", ".join(stored))
            if stored else "")
        self._pending_edit = {"scan_id": int(key[0]), "branch_path": key[1]}
        self._sync_run_controls()
        if self.auto_chk.isChecked():
            self._reprocess()

    def _clear_override(self) -> None:
        row = self.strip.currentRow()
        key = self._row_key(row)
        if key is None:
            return
        try:
            with db_session(self.db_path) as conn:
                ManualOverrideRepo(conn).clear(int(key[0]), key[1])
                conn.commit()
        except Exception:
            return
        self._pending_edit = {"scan_id": int(key[0]), "branch_path": key[1]}
        self._configure_editor(row)
        if self.auto_chk.isChecked():
            self._reprocess()

    def _on_auto_toggled(self, on: bool) -> None:
        # Session-wide, so opening another page keeps the mode.
        type(self)._auto_process_session = bool(on)
        self._sync_run_controls()

    def _sync_run_controls(self) -> None:
        auto = self.auto_chk.isChecked()
        # Dimmed while auto-process is on: there is nothing for it to do.
        self.reprocess_btn.setEnabled(bool(self._pending_edit) and not auto)
        self._clear_btn.setEnabled(bool(self._stored(self.strip.currentRow())))

    def _reprocess(self, *, force: bool = False) -> None:
        pending = self._pending_edit
        if not pending:
            return
        if self._reprocess_cb is None:
            self._overlay_note.setText(
                self.tr("Saved. Reprocess from the scans view."))
            return
        try:
            self._reprocess_cb(pending["scan_id"], pending["branch_path"])
        except Exception as e:
            self._overlay_note.setText(
                self.tr("Reprocess failed: {err}").format(err=e))
            return
        self._pending_edit = {}
        self._sync_run_controls()


class DebugViewerDialog(QDialog):
    """Free-floating window wrapper around DebugViewerWidget. Kept for
    callers that still want a modeless dialog instead of a tab — the
    MainWindow now embeds DebugViewerWidget directly in its tab strip."""

    def __init__(self, db_path: str, leaf_node_id: int,
                 title_hint: str = "", parent=None):
        super().__init__(parent)
        hint = title_hint or self.tr("node {n}").format(n=leaf_node_id)
        self.setWindowTitle(self.tr("Inspect · {hint}").format(hint=hint))
        self.setModal(False)
        self.resize(1400, 900)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = DebugViewerWidget(db_path, leaf_node_id, title_hint, self)
        layout.addWidget(self.viewer)
