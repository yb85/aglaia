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
    QPlainTextEdit, QVBoxLayout, QWidget,
)

from aglaia.gui.ZoomCanvas import ZoomCanvas, ZoomToolbar
from aglaia.gui.colors import (
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
from aglaia.storage.db import open_db
from aglaia.storage.debug_chain import _render_one, _walk_chain
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


class DebugViewerWidget(QWidget):
    """Strip-of-thumbs + zoomable main pane. Lives inside a tab; the
    enclosing `QTabWidget` owns close + lifecycle.

    Re-renders the full chain at construction time (~250 ms typical).
    Re-fits the current pixmap on resize so the picture always fills the
    available pane without manual scrollbar dancing.
    """

    def __init__(self, db_path: str, leaf_node_id: int,
                 title_hint: str = "", parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.leaf_node_id = leaf_node_id
        self.title_hint = title_hint or self.tr("node {n}").format(n=leaf_node_id)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self.strip = QListWidget()
        self.strip.setFixedWidth(232)
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
        self.canvas = ZoomCanvas(placeholder=self.tr("Select a step"))
        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 2, 0)
        self.zoom_bar = ZoomToolbar(self.canvas, default=2.0)
        bar.addWidget(self.zoom_bar)
        bar.addStretch(1)
        # Toggle "Show debug overlays" — disabled until the background
        # renderer finishes. Flipping it swaps the canvas pixmap
        # between bare (intermediate image) and overlay (per-processor
        # debug composite).
        self.overlay_chk = QCheckBox(self.tr("Show debug overlays"))
        self.overlay_chk.setEnabled(False)
        self.overlay_chk.setToolTip(self.tr(
            "Re-render the chain with per-processor overlays "
            "(spans, baselines, quad, grid). Computing…"
        ))
        self.overlay_chk.toggled.connect(self._on_overlay_toggled)
        bar.addWidget(self.overlay_chk)
        right_v.addLayout(bar)
        right_v.addWidget(self.canvas, 1)
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
        self._overlay_bytes: list[Optional[bytes]] = []
        self._overlay_job = _OverlayJob(self.db_path, self.leaf_node_id, self)
        self._overlay_job.done.connect(self._on_overlay_ready)
        self._overlay_job.failed.connect(self._on_overlay_failed)
        self._overlay_job.start()

    def _load(self):
        conn = open_db(self.db_path)
        try:
            chain = _walk_chain(conn, self.leaf_node_id)
            images: list[dict] = []
            for node in chain:
                if node.get("processor_name") is None:
                    continue
                images.extend(_render_one(conn, node))
        finally:
            conn.close()

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
                200, Qt.TransformationMode.SmoothTransformation)
                if base_pix is not None else None)
            del base_pix  # full-res copy not retained
            row_w = self._build_strip_row(label, thumb, i)
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

    def _build_strip_row(self, label: str, pix: Optional[QPixmap], idx: int) -> QWidget:
        """Per-step row: bold step label on top, thumbnail below. Zebra
        background (even/odd) so adjacent rows are visually separated
        even when the thumbnails have similar content."""
        w = QWidget()
        w.setObjectName("debugRow")
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 4, 6, 6)
        v.setSpacing(3)
        title = QLabel(label)
        title.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-weight: bold; font-size: 11px;")
        title.setWordWrap(True)
        v.addWidget(title)
        thumb_lbl = QLabel()
        if pix is not None:
            # Caller passes a pre-scaled (≤200 px) thumb; rescale only
            # if a full-res pixmap slipped through.
            thumb = (pix if pix.width() <= 200 else pix.scaledToWidth(
                200, Qt.TransformationMode.SmoothTransformation))
            thumb_lbl.setPixmap(thumb)
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(thumb_lbl)
        # Zebra banding lives on the row container so it stays visible
        # under the (transparent) child labels.
        w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bg = COLOR_BG_ZEBRA_EVEN if (idx % 2 == 0) else COLOR_BG_ZEBRA_ODD
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
        pix = None
        if (self.overlay_chk.isChecked()
                and 0 <= row < len(self._overlay_bytes)
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
        # Pad/truncate to match the bare strip count.
        target = self.strip.count()
        while len(blobs) < target:
            blobs.append(None)
        self._overlay_bytes = blobs[:target]
        self.overlay_chk.setEnabled(True)
        self.overlay_chk.setToolTip(self.tr(
            "Toggle per-processor overlays (spans, baselines, quad, grid)."
        ))
        # Restore the remembered choice now that the toggle is live.
        try:
            from aglaia.app_data import db as _cfg
            with _cfg.session() as _c:
                want = bool(_cfg.get(_c, _cfg.KEY_DEBUG_OVERLAYS, False))
            if want and not self.overlay_chk.isChecked():
                self.overlay_chk.setChecked(True)   # fires the re-render
        except Exception:
            pass

    def _on_overlay_failed(self, err: str) -> None:
        self.overlay_chk.setEnabled(False)
        self.overlay_chk.setToolTip(
            self.tr("Overlay render failed: {err}").format(err=err)
        )

    def _on_overlay_toggled(self, _on: bool) -> None:
        # Remember the choice across sessions.
        try:
            from aglaia.app_data import db as _cfg
            with _cfg.session() as _c:
                _cfg.set(_c, _cfg.KEY_DEBUG_OVERLAYS, bool(_on))
        except Exception:
            pass
        # Re-paint the currently-selected row using the new mode.
        self._on_row_changed(self.strip.currentRow())


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
