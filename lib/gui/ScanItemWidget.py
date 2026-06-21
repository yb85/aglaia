# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Per-scan widget for the Qt capture GUI (M0 / DB-backed).

A scan corresponds to one `scans` row. Its tree (root node + pipeline children +
branches) is replayed/extended in real time as `image_event` payloads arrive.

The widget loads thumbnail bytes lazily through the `thumb_loader` callable
provided by `MainWindow` — it never touches the filesystem.
"""

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal, QTimer, QSize, QMimeData, QByteArray, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QFontMetrics, QDrag, QCursor
from PySide6.QtWidgets import (QGraphicsOpacityEffect, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QApplication)


SCAN_DRAG_MIME = "application/x-aglaia-scan-id"

from lib.Status import Status, STATUS_COLORS
from lib.gui.colors import (
    COLOR_BG_BUTTON,
    COLOR_BG_BUTTON_PRESSED,
    COLOR_BG_HINT,
    COLOR_BG_OVERLAY_HOVER,
    COLOR_ERROR,
    COLOR_ERROR_BG_SOFT,
    COLOR_FONT_DIM,
    COLOR_FONT_DISABLED,
    COLOR_FONT_MUTED,
    COLOR_FONT_PLACEHOLDER,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE,
    COLOR_OUTLINE_BUTTON,
    COLOR_OUTLINE_BUTTON_STRONG,
    COLOR_OUTLINE_STRONG,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_STRONG,
    COLOR_SCRIM_MEDIUM,
    COLOR_SCRIM_STRONG,
    COLOR_SUCCESS,
    COLOR_WARNING,
    qcolor,
)


ThumbLoader = Callable[[int, int], Optional[bytes]]


class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        # Ignore natural text width in sizeHint — long filenames would
        # otherwise push the card off-screen.
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def sizeHint(self):  # noqa: N802 — Qt API
        metrics = QFontMetrics(self.font())
        return QSize(0, metrics.height())

    def minimumSizeHint(self):  # noqa: N802 — Qt API
        return self.sizeHint()

    def setFullText(self, text):
        self._full_text = text
        self.update_elided_text()

    def update_elided_text(self):
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideRight, self.width())
        super().setText(elided)

    def resizeEvent(self, event):
        self.update_elided_text()
        super().resizeEvent(event)


class _ClickableLabel(QLabel):
    """QLabel that emits `clicked` on left mouse press. Used so a
    thumbnail can open the debug viewer without a separate button."""
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _SpinnerOverlay(QWidget):
    """Translucent overlay covering its parent. Renders a centred braille
    spinner via a shared QTimer-driven frame index. Used to mark a scan
    as "still being processed" — paired with a 50 % opacity effect on
    the thumbs row.

    Click-through: `setAttribute(WA_TransparentForMouseEvents)` so the
    underlying nav buttons + thumbs stay clickable.
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._idx = 0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self):
        if not self._timer.isActive():
            self._timer.start()
        self.show()
        self.raise_()
        self._resize_to_parent()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._idx = (self._idx + 1) % len(self._FRAMES)
        self.update()

    def _resize_to_parent(self):
        p = self.parentWidget()
        if p is not None:
            self.setGeometry(0, 0, p.width(), p.height())

    def paintEvent(self, _ev):
        self._resize_to_parent()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Soft scrim so the dimmed thumb stays legible. Black-alpha
        # scrim works on both themes (light bg → light gray; dark bg →
        # darker gray). Bumped from 50 to 90 so the spinner reads even
        # on the brightest light-mode thumbnails.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))
        # Centred spinner glyph in the brand accent — visible on both
        # themes (was white, which vanished into the light bg).
        font = QFont()
        font.setPixelSize(min(56, max(28, self.height() // 4)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(qcolor(COLOR_PRIMARY))
        painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter),
                         self._FRAMES[self._idx])
        painter.end()


class _DisabledBand(QWidget):
    """3px mini-map pinned to a thumbnail's top edge. Invisible until a
    stage is disabled. `slots` = ordered [(node_id, disabled)] for the
    layout's deactivable steps; the band splits into N equal slots and
    paints a red square over each disabled one (faint green elsewhere)."""

    def __init__(self, slots, width, parent=None):
        super().__init__(parent)
        self._slots = list(slots)
        self.setFixedSize(int(width), 3)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, ev):  # noqa: N802
        n = len(self._slots)
        if n == 0:
            return
        p = QPainter(self)
        green = QColor(COLOR_SUCCESS)
        green.setAlpha(55)
        p.fillRect(self.rect(), green)
        red = QColor(COLOR_ERROR)
        red.setAlpha(205)
        slot_w = self.width() / n
        for k, (_nid, dis) in enumerate(self._slots):
            if dis:
                x0 = int(round(k * slot_w))
                x1 = int(round((k + 1) * slot_w))
                p.fillRect(x0, 0, max(1, x1 - x0), self.height(), red)
        p.end()


class _DragHandle(QWidget):
    """Header band that initiates a card drag. Cursor is open-hand when
    hovering, closed-hand once the press passes the drag threshold. The
    drag carries the owning ScanItemWidget's scan_id as raw bytes — the
    drop target (FlowContentWidget) resolves it back to the widget."""

    def __init__(self, card: "ScanItemWidget"):
        super().__init__(card)
        self._card = card
        self._press_pos: Optional[QPoint] = None
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position().toPoint()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._press_pos = None
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_pos is None:
            return
        delta = ev.position().toPoint() - self._press_pos
        if delta.manhattanLength() < QApplication.startDragDistance():
            return
        # Drag preview = full card pixmap so identity is unambiguous in the flow grid.
        pix = self._card.grab()
        # Cap drag preview height — bigger looks silly under the cursor.
        max_h = 220
        if pix.height() > max_h:
            pix = pix.scaledToHeight(max_h, Qt.TransformationMode.SmoothTransformation)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(SCAN_DRAG_MIME, QByteArray(str(self._card.scan_id).encode()))
        drag.setMimeData(mime)
        drag.setPixmap(pix)
        drag.setHotSpot(QPoint(pix.width() // 2, 12))
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._press_pos = None
        # Dim the source card while dragging — `drag.exec` blocks, so the
        # restore right after runs immediately on drop/cancel.
        prev_effect = self._card.graphicsEffect()
        dim_effect = QGraphicsOpacityEffect(self._card)
        dim_effect.setOpacity(0.35)
        self._card.setGraphicsEffect(dim_effect)
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self._card.setGraphicsEffect(prev_effect)


class ScanItemWidget(QWidget):
    # Emits the scan_id this widget represents (MainWindow soft-deletes on receipt).
    delete_requested = Signal(int)
    # Emits (leaf_node_id, label) when a thumb is clicked; MainWindow
    # opens the debug viewer dialog.
    debug_requested = Signal(int, str)

    # Emits (scan_id, fit_zoom) when the *final-step* thumb's natural fit
    # zoom is observed. Parent (MainWindow) maintains the global min and
    # broadcasts via `set_global_zoom`.
    final_zoom_observed = Signal(int, float)
    # Emits (scan_id, branch_label, node_id) whenever the card's
    # per-stem selection cursor moves. Vestigial — exit-stage navigation
    # was replaced by per-page disable; `_emit_selection_for` is now a
    # no-op, so this never fires. Kept so legacy host wiring binds cleanly.
    selection_changed = Signal(int, str, int)
    # Emits (scan_id, branch_label, hidden) when the user toggles a
    # layout's eye button — host writes `branches.trashed_at`.
    visibility_changed = Signal(int, str, bool)
    # Emits (scan_id, node_id) when the round stage-toggle is clicked —
    # host flips the step's per-page disable + reruns the page.
    step_toggle_requested = Signal(int, int)

    def __init__(self, *, scan_id: int, idx: int, raw_node_id: int, raw_image_id: int,
                 raw_filestem: str, pipeline_steps: list[str],
                 thumb_loader: ThumbLoader,
                 raw_dpi: Optional[float] = None,
                 max_card_width_px: int = 150,
                 zoom_tolerance: float = 0.2,
                 parent: Optional[QWidget] = None):
        # IMPORTANT: pass `parent` to QWidget early. A parentless QWidget
        # on macOS briefly becomes a top-level native window (traffic-light
        # chrome + all), which the user sees as a flashing popup while the
        # widget is constructed and then reparented into the flow grid.
        super().__init__(parent)
        self.scan_id = scan_id
        self.idx = idx
        self.raw_filestem = raw_filestem
        self.thumb_loader = thumb_loader
        self.raw_dpi = raw_dpi
        self.pipeline_steps = list(pipeline_steps)
        self.max_steps = len(pipeline_steps)
        self.max_card_w = int(max_card_width_px)
        self.zoom_tol = float(zoom_tolerance)
        # Parent broadcasts via `set_global_zoom`; None until any card sees a final-step layout.
        self.global_zoom: Optional[float] = None
        # WA_StyledBackground: QWidget subclasses need this for the QSS border + bg to paint.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Step rail traversed left/right by the nav buttons.
        self.global_history: list[str] = ["raw"] + list(pipeline_steps)
        self.current_history_idx = 0
        self.interacted_stem: Optional[str] = None
        self.dimmed = False
        # Host-supplied: scan_id → {node_id: (toggleable, disabled)} for the
        # per-page disable round-toggle + the disabled-state band.
        self.step_states_provider = None

        # filestem -> {
        #   "history":   [step_name, ...],
        #   "nodes":     {step_name: {"node_id": int, "image_id": int, "meta": dict}},
        #   "node_to_step": {node_id: step_name},
        #   "parent":    Optional[str],
        #   "children":  [str, ...],
        #   "current_idx": int,
        #   "trashed":   bool,
        # }
        self.items: dict[str, dict] = {}
        # OCR badge state: "none" | "fresh" | "stale". MainWindow flips
        # this via `set_ocr_state` after the OCR worker finishes or after
        # initial DB scan. Badge sits in the final-step thumb's bottom-right.
        self._ocr_state: str = "none"
        # Sticky "user navigated since last OCR sync" flag. Set on
        # prev/next button presses, demotes a "fresh" badge to "stale" so
        # the user knows the OCR ran against a different stage than what
        # they're currently looking at. Cleared by `set_ocr_state` (DB
        # resync after OCR run or after a pipeline rerun re-baselines).
        # Per-stem OCR state pushed from MainWindow (DB-derived,
        # single source of truth). Render-time lookup falls back to the
        # scan-level `_ocr_state` when a stem isn't in this map.
        self._ocr_state_per_stem: dict[str, str] = {}
        # Quick lookup: node_id -> filestem (so events for nested branches resolve fast).
        self._stem_for_node: dict[int, str] = {}
        # image_ids this card displays — so an async thumb-ready signal only
        # repaints the card that actually owns the freshly-built thumbnail.
        self._image_ids: set[int] = set()
        self._ensure_item(raw_filestem, parent_stem=None)
        self._register_node(raw_filestem, "raw", node_id=raw_node_id,
                            image_id=raw_image_id, meta={})
        # The thumb loader builds thumbnails off the GUI thread; refresh this
        # card when one it owns becomes available.
        ready = getattr(thumb_loader, "ready", None)
        if ready is not None:
            ready.connect(self._on_thumb_ready)

        self.setStyleSheet(f"""
            ScanItemWidget {{
                background-color: {COLOR_BG_HINT};
                border: 1px solid {COLOR_OUTLINE};
                border-radius: 8px;
                margin-bottom: 5px;
            }}
        """)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(2)
        self.setLayout(self.layout)

        self.header_widget = _DragHandle(self)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(6, 4, 6, 4)
        header_layout.setSpacing(6)
        from lib.gui.widgets import make_icon_button
        # Scan-level delete — keep the trash icon (this really removes the
        # scan). Layout-level hiding uses eye / eye-off elsewhere.
        self.del_btn = make_icon_button(
            "trash-2", size=20, icon_size=14, color=COLOR_ERROR,
            hover_bg=COLOR_ERROR_BG_SOFT,
        )
        self.del_btn.setToolTip(self.tr("Discard scan"))
        self.del_btn.clicked.connect(lambda: self.delete_requested.emit(self.scan_id))

        self.name_label = ElidedLabel(self.tr("Scan #{idx}").format(idx=idx))
        self.name_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.path_label = ElidedLabel(raw_filestem)
        self.path_label.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 9px;")
        self.path_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header_layout.addWidget(self.del_btn)
        header_layout.addWidget(self.name_label, 1)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.path_label, 1)
        self.layout.addWidget(self.header_widget)

        self.thumbs_container = QWidget()
        self.thumbs_layout = QHBoxLayout(self.thumbs_container)
        self.thumbs_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbs_layout.setSpacing(5)
        self.thumbs_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.thumbs_container)

        # Throttle composite rebuilds: during processing flurries a card
        # can receive several node events per second; rebuilding thumbs on
        # each one makes the card width jitter and the flow grid reshuffle.
        # Trailing-edge: first call starts the timer, the rebuild runs once
        # when it fires, so the last event is always rendered.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self.refresh_composite)

        self.refresh_composite()
        self.update_header()

        # Spinner overlay + dim effect — toggled via `set_processing`.
        # Initial state: not processing. MainWindow flips it on for scans
        # whose pipeline is still in flight (new captures, reprocess).
        self._opacity_effect = QGraphicsOpacityEffect(self.thumbs_container)
        self._opacity_effect.setOpacity(1.0)
        self.thumbs_container.setGraphicsEffect(self._opacity_effect)
        self._spinner = _SpinnerOverlay(self.thumbs_container)
        self._is_processing = False

    # ──────────────────────── processing state ────────────────────────────
    def set_processing(self, processing: bool):
        """Toggle the in-flight visual: dim thumbs to ~50 % opacity and
        overlay a centred braille spinner. Idempotent."""
        if processing == self._is_processing:
            return
        self._is_processing = processing
        if processing:
            self._opacity_effect.setOpacity(0.45)
            self._spinner.start()
        else:
            self._opacity_effect.setOpacity(1.0)
            self._spinner.stop()

    def set_ocr_state(self, state: str) -> None:
        """Update the scan-level OCR badge state (`none` | `fresh` |
        `stale`). Per-stem state takes precedence at render time; this
        scalar is the fallback when no per-stem mapping is supplied."""
        if state not in ("none", "fresh", "stale"):
            state = "none"
        if state == self._ocr_state:
            return
        self._ocr_state = state
        self.refresh_composite()

    def set_ocr_state_per_stem(self, mapping: dict[str, str]) -> None:
        """Replace the per-stem OCR state map. Keys are stems (e.g.
        `"raw_001_A"`). Render uses this lookup first, falling back to
        the scan-level `_ocr_state` when a stem is absent."""
        clean = {k: v for k, v in (mapping or {}).items()
                 if v in ("none", "fresh", "stale")}
        if clean == self._ocr_state_per_stem:
            return
        self._ocr_state_per_stem = clean
        self.refresh_composite()

    def is_processing(self) -> bool:
        return self._is_processing

    # ───────────────────────── internal bookkeeping ─────────────────────────

    def _ensure_item(self, stem: str, *, parent_stem: Optional[str]) -> dict:
        if stem not in self.items:
            self.items[stem] = {
                "history": [],
                "nodes": {},
                "node_to_step": {},
                "parent": parent_stem,
                "children": [],
                "current_idx": self.current_history_idx,
                "trashed": False,
            }
            if parent_stem and parent_stem in self.items:
                if stem not in self.items[parent_stem]["children"]:
                    self.items[parent_stem]["children"].append(stem)
        return self.items[stem]

    def _register_node(self, stem: str, step_name: str, *, node_id: Optional[int],
                       image_id: Optional[int], meta: Optional[dict]):
        entry = self.items[stem]
        if step_name not in entry["history"]:
            entry["history"].append(step_name)
        entry["nodes"][step_name] = {
            "node_id": node_id,
            "image_id": image_id,
            "meta": meta or {},
        }
        if image_id is not None:
            self._image_ids.add(int(image_id))
        if node_id is not None:
            entry["node_to_step"][node_id] = step_name
            self._stem_for_node[node_id] = stem
        if step_name not in self.global_history:
            self.global_history.append(step_name)

    def _resolve_parent_stem(self, parent_node_id: Optional[int]) -> Optional[str]:
        if parent_node_id is None:
            return None
        return self._stem_for_node.get(parent_node_id)

    # ───────────────────────── event entry points ───────────────────────────

    def handle_event(self, *, node_id: Optional[int], parent_node_id: Optional[int],
                     image_id: Optional[int], step_name: str, filestem: str,
                     meta: Optional[dict] = None):
        """Called when a worker emits a new node for this scan."""
        if not step_name or not filestem:
            return
        parent_stem = self._resolve_parent_stem(parent_node_id)
        # Child branches get their own item; same-stem repeats reuse the existing one.
        if filestem != self.raw_filestem and filestem not in self.items:
            self._ensure_item(filestem, parent_stem=parent_stem or self.raw_filestem)
        else:
            self._ensure_item(filestem, parent_stem=parent_stem)
        self._register_node(filestem, step_name, node_id=node_id,
                            image_id=image_id, meta=meta)
        self._auto_advance(filestem, step_name)
        self.schedule_refresh()
        self.update_header()

    def restore_node(self, *, node_id: int, parent_node_id: Optional[int],
                     image_id: int, step_name: str, filestem: str,
                     meta: Optional[dict] = None):
        """Replay a node already persisted in the DB (boot-time load)."""
        self.handle_event(
            node_id=node_id, parent_node_id=parent_node_id,
            image_id=image_id, step_name=step_name or "raw",
            filestem=filestem, meta=meta,
        )

    def _auto_advance(self, stem: str, step_name: str):
        if step_name not in self.global_history:
            return
        new_idx = self.global_history.index(step_name)
        # "Following the stream": advance the cursor when we land at or past the tip.
        if new_idx > self.current_history_idx:
            self.current_history_idx = new_idx
            for v in self.items.values():
                v["current_idx"] = new_idx

    # ───────────────────────── rendering ────────────────────────────────────

    # Thumbnail size knobs. `target_h` sets row height; `max_w` caps
    # individual thumb width so a portrait page doesn't push the row off
    # screen when one branch is much wider than the others.
    def set_max_card_width(self, px: int):
        """Live slider override. Re-renders thumbs at the new max width
        and re-emits the final-step fit_zoom so the parent can recompute
        the global zoom band against the new ceiling."""
        if int(px) == self.max_card_w:
            return
        self.max_card_w = int(px)
        self.schedule_refresh()

    def set_global_zoom(self, g: Optional[float]):
        """Parent broadcasts the latest global min fit_zoom. Triggers a
        re-render so any final-step thumb that's now above (1+d)*g gets
        clamped."""
        if g == self.global_zoom:
            return
        self.global_zoom = g
        self.schedule_refresh()

    def schedule_refresh(self):
        """Coalesced refresh_composite — at most one rebuild per timer
        interval. Direct calls remain for user-interaction paths that
        need instant feedback (nav buttons, trash toggle)."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _on_thumb_ready(self, image_id: int):
        """A background thumbnail finished building. Repaint only if this
        card shows that image (coalesced via the refresh timer)."""
        if int(image_id) in self._image_ids:
            self.schedule_refresh()

    def refresh_composite(self):
        max_w = self.max_card_w
        final_step = self.pipeline_steps[-1] if self.pipeline_steps else None

        while self.thumbs_layout.count():
            item = self.thumbs_layout.takeAt(0)
            w = item.widget()
            if w:
                # `takeAt` only unlinks from the layout — the widget keeps
                # its parent until `deleteLater` is processed by the event
                # loop. During reprocess flurries this code re-runs before
                # that, leaving stale containers parented to thumbs_container
                # → ghost-duplicated thumbnails. setParent(None) detaches
                # synchronously so the next addWidget paints over an empty
                # container.
                # hide() FIRST: Qt re-shows a reparented widget that was
                # visible without WA_WState_ExplicitShowHide — on
                # setParent(None) that means a bare top-level window
                # flashing the thumbnail until deleteLater runs.
                w.hide()
                w.setParent(None)
                w.deleteLater()

        # Per-page disable state for every node of this scan, one query.
        step_states: dict = {}
        if self.step_states_provider is not None:
            try:
                step_states = self.step_states_provider(int(self.scan_id)) or {}
            except Exception:
                step_states = {}

        visible_stems = self._collect_visible_stems(self.raw_filestem)
        for stem in visible_stems:
            data = self.items[stem]
            l_idx = data.get("current_idx", self.current_history_idx)
            if l_idx >= len(self.global_history):
                l_idx = len(self.global_history) - 1
            target_step = self.global_history[l_idx]
            actual_step = target_step
            node_info = data["nodes"].get(target_step)
            if node_info is None:
                # Fall back to the latest available earlier step.
                for i in range(l_idx - 1, -1, -1):
                    prev = self.global_history[i]
                    if prev in data["nodes"]:
                        node_info = data["nodes"][prev]
                        actual_step = prev
                        break

            is_final = (final_step is not None and actual_step == final_step)
            pix, new_w, new_h, fit_zoom = self._build_pixmap_w(
                node_info, data, actual_step, max_w, is_final
            )
            # Notify parent of the final-step fit_zoom so it can update
            # the global min. Parent re-emits via `set_global_zoom`.
            if is_final and fit_zoom is not None:
                self.final_zoom_observed.emit(self.scan_id, float(fit_zoom))

            lbl = _ClickableLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(new_w, new_h)
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            # Hover tooltip = the stage this thumbnail shows.
            lbl.setToolTip(str(actual_step))
            if node_info and node_info.get("node_id") is not None:
                node_id = int(node_info["node_id"])
                lbl_text = f"{stem} | {actual_step}"
                lbl.clicked.connect(
                    lambda nid=node_id, txt=lbl_text:
                    self.debug_requested.emit(nid, txt))

            is_trashed = data.get("trashed", False)
            if is_trashed:
                # Repaint the thumb with a 55 % white wash + a large
                # faint trash glyph centred on top — reads as "muted /
                # deleted" without QGraphicsEffect, which interacted
                # badly with the parent thumbs_container opacity effect.
                washed = QPixmap(pix.size())
                washed.fill(Qt.GlobalColor.transparent)
                _wp = QPainter(washed)
                _wp.drawPixmap(0, 0, pix)
                _wp.fillRect(washed.rect(), qcolor(COLOR_FONT_DIM))
                # Filigrane "hidden" glyph at ~55 % of the smaller thumb edge.
                # Eye-off (not trash) — this layout is hidden, not deleted.
                from lib.gui.theme import lucide_pixmap as _lp
                fil_size = max(24, int(min(pix.width(), pix.height()) * 0.55))
                # Hex token (not rgba): the SVG stroke attribute drops
                # rgba() values, so a muted rgba glyph used to render
                # transparent. Brand-accent hex with explicit painter
                # opacity gives a visible "hidden" filigrane on both
                # themes.
                fil = _lp("eye-off", color=COLOR_PRIMARY, size=fil_size)
                if not fil.isNull():
                    dpr = max(int(fil.devicePixelRatio()), 1)
                    fw = fil.width() // dpr
                    fh = fil.height() // dpr
                    _wp.setOpacity(0.55)
                    _wp.drawPixmap(
                        (pix.width() - fw) // 2,
                        (pix.height() - fh) // 2,
                        fw, fh, fil,
                    )
                _wp.end()
                lbl.setPixmap(washed)
                lbl.setStyleSheet(f"border: 1px solid {COLOR_OUTLINE_STRONG}; "
                                  f"background-color: {COLOR_BG_HINT}; "
                                  "border-radius: 4px;")
            else:
                border_color, thickness = COLOR_OUTLINE, 1
                meta = node_info["meta"] if node_info else {}
                status_val = meta.get("status")
                if status_val is not None:
                    border_color = STATUS_COLORS.get(int(status_val), "gray")
                    if int(status_val) in (Status.SUCCESS, Status.WARNING,
                                            Status.ERROR, Status.REVIEW):
                        thickness = 3
                lbl.setStyleSheet(f"border: {thickness}px solid {border_color}; "
                                  f"background-color: {COLOR_BG_HINT}; border-radius: 4px;")

            # Floor container at 60px tall so the bottom-left trash button
            # never overlaps the top-row nav arrows on thin landscape
            # crops (e.g. a single text line ~30px high).
            container_h = max(new_h, 60)
            # Parent to thumbs_container right away. A parentless QWidget
            # whose child button gets .show()'d by `place_overlay` would
            # briefly appear as a top-level native window on macOS (the
            # tiny "popup" with traffic-light chrome the user reported
            # during processing).
            container = QWidget(self.thumbs_container)
            container.setFixedSize(new_w, container_h)
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(0, 0, 0, 0)
            cont_layout.setSpacing(0)
            cont_layout.addWidget(lbl)
            cont_layout.addStretch(1)

            is_root = (stem == self.raw_filestem)
            if len(self.global_history) > 1:
                self._add_nav_buttons(container, new_w,
                                      stem=stem if not is_root else None,
                                      is_global=is_root)

            # Per-page disable: round toggle for the displayed stage +
            # the disabled-state band (mini-map of the layout's skip set).
            self._add_disable_toggle(container, new_w, node_info,
                                     actual_step, step_states)
            self._add_disabled_band(container, new_w, data, step_states)

            from lib.gui.widgets import make_icon_button, place_overlay
            # Layout-level hide/show: eye-off when hidden, eye when
            # visible. Icon shows CURRENT state; click toggles. Trash
            # icon reserved for scan-level destructive delete in header.
            btn_trash = make_icon_button(
                "eye-off" if is_trashed else "eye",
                size=24, icon_size=14, color=COLOR_FONT_PRIMARY,
                bg=COLOR_SCRIM_MEDIUM, hover_bg=COLOR_SCRIM_STRONG,
                border=f"1px solid {COLOR_OUTLINE}",
            )
            btn_trash.setToolTip(self.tr("Show page") if is_trashed else self.tr("Hide page"))
            btn_trash.clicked.connect(lambda _checked, s=stem: self.toggle_trash(s))
            place_overlay(btn_trash, container, 4, container_h - 28)

            # OCR state badge — per-stem state lookup (DB-derived,
            # pushed from MainWindow). Falls back to the scan-level
            # `_ocr_state` if the stem isn't in the per-stem map.
            is_non_root = (stem != self.raw_filestem)
            stem_state = self._ocr_state_per_stem.get(stem, self._ocr_state)
            if is_non_root and stem_state != "none" and not is_trashed:
                effective = stem_state
                color = COLOR_SUCCESS if effective == "fresh" else COLOR_WARNING
                tooltip = (self.tr("OCR up to date") if effective == "fresh"
                           else self.tr(
                               "OCR is stale — selected page changed since OCR ran. "
                               "Re-run OCR."))
                # Informative-only — flat colored glyph, no background,
                # no border, no click handler. Bottom-right, mirroring
                # the trash button's bottom-left.
                from lib.gui.theme import lucide_pixmap as _lp
                pix = _lp("scan-text", color=color, size=20)
                ocr_badge = QLabel(container)
                ocr_badge.setPixmap(pix)
                ocr_badge.setFixedSize(20, 20)
                ocr_badge.setScaledContents(True)
                ocr_badge.setStyleSheet("background: transparent; border: none;")
                ocr_badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                ocr_badge.setToolTip(tooltip)
                place_overlay(ocr_badge, container,
                              new_w - 24, container_h - 24)

            self.thumbs_layout.addWidget(container)

    def _build_pixmap_w(self, node_info: Optional[dict], data: dict, actual_step: str,
                        max_w: int, is_final: bool
                        ) -> tuple[QPixmap, int, int, Optional[float]]:
        """Width-driven thumb build.

        Returns (pixmap, displayed_width, displayed_height, fit_zoom).

        `fit_zoom` is the natural (cap-at-native) zoom needed to fit the
        full-resolution image into `max_w`. For intermediate steps the
        thumb is always sized to `min(fit_zoom*orig_w, max_w)` (hard clamp).
        For the final step, if `self.global_zoom` is set, the displayed
        zoom is clamped down to `(1 + tol) * global_zoom`. The smallest
        observed `fit_zoom` (across all cards' final thumbs) becomes the
        new global — parent decides and broadcasts back.
        """
        pix = None
        # Fallback aspect for the "Pending image" placeholder.
        new_w = max_w
        new_h = int(max_w * 1.4)
        fit_zoom: Optional[float] = None

        if node_info and node_info.get("image_id") is not None:
            blob = self.thumb_loader(node_info["image_id"], 512)
            if blob:
                img = QImage.fromData(blob)
                if not img.isNull():
                    orig_w = max(img.width(), 1)
                    orig_h = max(img.height(), 1)
                    # The cap-at-native fit_zoom. Never upscale a thumb
                    # beyond its source pixels.
                    fit_zoom = min(1.0, max_w / orig_w)
                    displayed_zoom = fit_zoom
                    if is_final and self.global_zoom is not None:
                        upper = (1.0 + self.zoom_tol) * self.global_zoom
                        if displayed_zoom > upper:
                            displayed_zoom = upper
                    new_w = max(int(round(orig_w * displayed_zoom)), 1)
                    new_h = max(int(round(orig_h * displayed_zoom)), 1)
                    img = img.scaled(new_w, new_h,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                    new_w = img.width()
                    new_h = img.height()
                    pix = QPixmap.fromImage(img)

        if pix is None:
            pix = QPixmap(new_w, new_h)
            pix.fill(QColor(COLOR_BG_BUTTON))
            p = QPainter(pix)
            p.setPen(QColor(COLOR_OUTLINE_STRONG))
            p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, self.tr("Pending\nImage…"))
            p.end()
            # Retry once shortly in case the thumb is still being written.
            if node_info and node_info.get("image_id") is not None:
                QTimer.singleShot(800, self.refresh_composite)

        # Skew overlay (from processor meta)
        meta = node_info["meta"] if node_info else {}
        skew = float(meta.get("skew", 0.0) or 0.0)
        if abs(skew) > 0.1:
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            p.setPen(qcolor(COLOR_PRIMARY))
            p.drawText(pix.rect().adjusted(0, 0, -8, -5),
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                       f"{skew:.1f}°")
            p.end()

        return pix, new_w, new_h, fit_zoom

    # ───────────────────────── navigation ───────────────────────────────────

    def _collect_visible_stems(self, stem: str) -> list[str]:
        data = self.items.get(stem)
        if not data:
            return []
        children = data.get("children", [])
        if not children:
            return [stem]
        current_idx = data.get("current_idx", self.current_history_idx)
        current_type = (self.global_history[current_idx]
                        if 0 <= current_idx < len(self.global_history) else None)
        if current_type and current_type in data["nodes"]:
            return [stem]
        out = []
        for c in children:
            out.extend(self._collect_visible_stems(c))
        return out or [stem]

    def _find_nearest_valid_index(self, start_idx: int, delta: int,
                                  stem: Optional[str] = None) -> int:
        curr = start_idx + delta
        while 0 <= curr < len(self.global_history):
            h_type = self.global_history[curr]
            if stem is not None:
                if stem in self.items and h_type in self.items[stem]["nodes"]:
                    return curr
            else:
                if any(h_type in v["nodes"] for v in self.items.values()):
                    return curr
            curr += delta
        return -1

    def _add_nav_buttons(self, container, width, *, stem, is_global):
        from lib.gui.widgets import make_icon_button, place_overlay
        from lib.gui.colors import COLOR_FONT_INVERSE as _CFI
        nav_kwargs = dict(
            size=24, icon_size=16, color=_CFI,
            bg=COLOR_PRIMARY_BG_STRONG, hover_bg=COLOR_PRIMARY,
        )
        btn_back = make_icon_button("chevron-left", **nav_kwargs)
        btn_fwd = make_icon_button("chevron-right", **nav_kwargs)
        # Disabled-state colour can't go through the helper (no extra
        # selector slot) — append it inline.
        for b in (btn_back, btn_fwd):
            b.setStyleSheet(
                b.styleSheet()
                + f"QPushButton:disabled {{ background-color: {COLOR_BG_OVERLAY_HOVER}; }}"
            )
        place_overlay(btn_back, container, 4, 4)
        place_overlay(btn_fwd, container, width - 28, 4)

        if is_global or stem is None:
            back_ok = self._find_nearest_valid_index(self.current_history_idx, -1) != -1
            fwd_ok = self._find_nearest_valid_index(self.current_history_idx, 1) != -1
            btn_back.setEnabled(back_ok)
            btn_fwd.setEnabled(fwd_ok)
            btn_back.clicked.connect(self.navigate_backward)
            btn_fwd.clicked.connect(self.navigate_forward)
        else:
            l_idx = self.items[stem].get("current_idx", self.current_history_idx)
            back_ok = self._find_nearest_valid_index(l_idx, -1, stem) != -1
            fwd_ok = self._find_nearest_valid_index(l_idx, 1, stem) != -1
            btn_back.setEnabled(back_ok)
            btn_fwd.setEnabled(fwd_ok)
            btn_back.clicked.connect(lambda: self.navigate_stem(stem, -1))
            btn_fwd.clicked.connect(lambda: self.navigate_stem(stem, 1))

    def _add_disable_toggle(self, container, width, node_info,
                            actual_step, states) -> None:
        """Round overlay on the displayed stage showing its pipeline index
        (or "R" for replay). Always shown for pipeline steps so the
        affordance is discoverable while chevroning through stages:
          - toggleable step (COORDINATE/PIXEL_VALUE): blue, click to
            disable → red "✕"; click again to re-enable;
          - locked step (raw / PageDetector / replay): dimmed grey, no
            action (tooltip explains)."""
        if not node_info:
            return
        nid = node_info.get("node_id")
        if nid is None:
            return
        step = (actual_step or "")
        if step == "raw" or not step:
            return  # raw capture isn't a pipeline step — nothing to index
        toggleable, disabled = states.get(int(nid), (False, False))
        from lib.gui.widgets import place_overlay
        from lib.gui.colors import COLOR_FONT_INVERSE as _CFI
        from lib.gui.colors import COLOR_BG_OVERLAY_HOVER as _CLOCK
        if "replay" in step.lower():
            label = "R"
        else:
            head = step.split("_", 1)[0]
            label = str(int(head)) if head.isdigit() else "•"
        text = "✕" if disabled else label
        btn = QPushButton(text, container)
        btn.setFixedSize(22, 22)
        if toggleable:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            bg = COLOR_ERROR if disabled else COLOR_PRIMARY_BG_STRONG
            hover = COLOR_ERROR if disabled else COLOR_PRIMARY
            btn.setStyleSheet(
                f"QPushButton{{border-radius:11px; background:{bg}; color:{_CFI}; "
                "font-size:11px; font-weight:700; border:none;}"
                f"QPushButton:hover{{background:{hover};}}"
            )
            btn.setToolTip(
                self.tr("Step disabled for this page — click to re-enable")
                if disabled else
                self.tr("Click to disable this step for this page")
            )
            btn.clicked.connect(
                lambda _checked=False, n=int(nid):
                self.step_toggle_requested.emit(int(self.scan_id), n)
            )
        else:
            # Locked: shown for orientation, but this step can't be skipped
            # (it would restructure the page tree, or it's raw/replay).
            btn.setEnabled(False)
            btn.setStyleSheet(
                f"QPushButton{{border-radius:11px; background:{_CLOCK}; "
                f"color:{COLOR_FONT_PLACEHOLDER}; font-size:11px; "
                "font-weight:700; border:none;}"
            )
            btn.setToolTip(self.tr("This step can't be disabled per page"))
        place_overlay(btn, container, width // 2 - 11, 4)

    def _add_disabled_band(self, container, width, data, states) -> None:
        """Top-edge mini-map of the layout's disabled steps. Hidden when
        nothing is disabled."""
        slots = []
        nodes = data.get("nodes", {})
        for step in self.global_history:
            ni = nodes.get(step)
            if not ni:
                continue
            nid = ni.get("node_id")
            if nid is None:
                continue
            toggleable, disabled = states.get(int(nid), (False, False))
            if toggleable:
                slots.append((int(nid), bool(disabled)))
        if not any(dis for _n, dis in slots):
            return
        from lib.gui.widgets import place_overlay
        band = _DisabledBand(slots, width, parent=container)
        place_overlay(band, container, 0, 0)

    def navigate_stem(self, stem: str, delta: int):
        data = self.items.get(stem)
        if not data:
            return
        c_idx = data.get("current_idx", self.current_history_idx)
        target = self._find_nearest_valid_index(c_idx, delta, stem)
        if target == -1:
            return
        data["current_idx"] = target
        self.interacted_stem = stem
        self._emit_selection_for(stem)
        self.refresh_composite()
        self.update_header()

    def navigate_backward(self):
        new_idx = self._find_nearest_valid_index(self.current_history_idx, -1)
        if new_idx == -1:
            return
        self.current_history_idx = new_idx
        for stem, v in self.items.items():
            v["current_idx"] = new_idx
            self._emit_selection_for(stem)
        self.refresh_composite()
        self.update_header()

    def navigate_forward(self):
        new_idx = self._find_nearest_valid_index(self.current_history_idx, 1)
        if new_idx == -1:
            return
        self.current_history_idx = new_idx
        for stem, v in self.items.items():
            v["current_idx"] = new_idx
            self._emit_selection_for(stem)
        self.refresh_composite()
        self.update_header()

    def _emit_selection_for(self, stem: str) -> None:
        """No-op. Chevron navigation used to write `chosen_node_id`
        (exit-stage selection); that model was replaced by per-page
        processor disable. The chevrons now only change which stage the
        card displays — they no longer mutate the DB."""
        return

    # ───────────────────────── trash / header ───────────────────────────────

    def toggle_trash(self, stem: str):
        if stem not in self.items:
            return
        new_state = not self.items[stem].get("trashed", False)
        self.items[stem]["trashed"] = new_state
        self.refresh_composite()
        # Notify host so `branches.trashed_at` follows the UI state and
        # the gallery / table see the same visibility.
        if stem == self.raw_filestem:
            branch_label = ""
        else:
            suffix = (stem[len(self.raw_filestem) + 1:]
                      if stem.startswith(self.raw_filestem + "_") else stem)
            branch_label = suffix.split("_")[-1] if "_" in suffix else suffix
        self.visibility_changed.emit(int(self.scan_id),
                                      str(branch_label), bool(new_state))

    def set_stem_trashed(self, stem: str, trashed: bool) -> None:
        """Apply a page's persisted hidden state on load. Unlike
        `toggle_trash`, this does NOT emit `visibility_changed` — it
        reflects the DB into the UI, not the other way round."""
        item = self.items.get(stem)
        if item is None or item.get("trashed", False) == bool(trashed):
            return
        item["trashed"] = bool(trashed)
        self.refresh_composite()

    def update_header(self):
        status_text = self.tr("Scan #{idx}").format(idx=self.idx)
        target_idx = self.current_history_idx

        root_data = self.items.get(self.raw_filestem)
        if root_data and root_data.get("children"):
            max_child_idx = -1
            for child_stem in root_data["children"]:
                c_data = self.items.get(child_stem)
                if c_data:
                    max_child_idx = max(max_child_idx, c_data.get("current_idx", 0))
            if max_child_idx > target_idx:
                target_idx = max_child_idx

        if self.interacted_stem and self.interacted_stem in self.items:
            target_idx = self.items[self.interacted_stem].get("current_idx", target_idx)

        if 0 <= target_idx < len(self.global_history):
            current_type = self.global_history[target_idx]
        else:
            current_type = "raw"

        if current_type == "raw":
            suffix = self.tr(" (Input)")
        else:
            parts = current_type.split("_")
            try:
                step_idx = int(parts[0])
                suffix = self.tr(" [{idx}/{total}]").format(
                    idx=step_idx, total=self.max_steps,
                )
            except (ValueError, IndexError):
                suffix = self.tr(" ({step})").format(step=current_type)

        # Source-DPI tag — surfaces inconsistencies between the
        # importer's claim and what the chain receives (e.g. a PDF
        # extracted at 72 dpi when the user expected 300).
        dpi_tag = (self.tr(" · {dpi} dpi").format(dpi=int(round(self.raw_dpi)))
                   if self.raw_dpi else "")
        final_text = f"{status_text}{dpi_tag}{suffix}"
        if self.dimmed:
            final_text += self.tr(" (No page detected)")
        self.name_label.setFullText(final_text)

        # Palette dispatch happens in lib.gui.colors at import time —
        # the tokens below already resolve to the right shade for the
        # active theme. No per-widget palette probe needed.
        if self.dimmed:
            bg = COLOR_BG_HINT
            fg = COLOR_FONT_DISABLED
            border = COLOR_OUTLINE
        else:
            bg = COLOR_BG_HINT
            fg = COLOR_FONT_PRIMARY
            border = COLOR_OUTLINE_BUTTON

        # Top bar + content area share the same surface — the previous
        # split palette read as two stacked panes glued together. One
        # cohesive ``COLOR_BG_HINT`` slab from the header through the
        # thumbs row gives the card a single-piece feel.
        self.setStyleSheet(f"""
            ScanItemWidget {{
                background-color: {bg};
                border: 1.5px solid {border};
                border-radius: 10px;
            }}
            _DragHandle {{
                background-color: {bg};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            QWidget#ScanThumbs {{
                background-color: {bg};
            }}
        """)
        # WA_StyledBackground turns the QSS bg from "decorative" to
        # "actually painted under children". Without it the thumbs
        # container shows whatever the parent's default palette window
        # role is, leaving a visible seam against the styled header.
        self.thumbs_container.setObjectName("ScanThumbs")
        self.thumbs_container.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
        self.name_label.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {fg};")
