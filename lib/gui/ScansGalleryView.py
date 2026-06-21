# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Bi-dimensional gallery view of scans.

Axes:
  - Horizontal (chevron left / right): pipeline stages of the focused
    scan. Each stage shows ALL its layouts (branches) side-by-side.
    Changing stage = setting the chosen_node_id (per-branch) → committed
    via `chosen_writer` callback so the selection survives reload.
  - Vertical (chevron up / down): step across scans (discrete next/prev).

On reload, auto-positions to the latest scan and the scan's currently
chosen stage (NOT raw — defaults to whatever the user last picked).

Chevron buttons are absolute-positioned children of the host frame to
sidestep QStackedLayout(StackAll) hit-test quirks.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget,
)

from lib.gui.colors import (
    COLOR_BG,
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_ON_BUTTON,
    COLOR_FONT_PLACEHOLDER,
    COLOR_MEDAL_GOLD,
    COLOR_OUTLINE_GHOST,
    COLOR_OUTLINE_STRONG,
    COLOR_OUTLINE_SUBTLE,
    COLOR_SCRIM_LIGHT,
    COLOR_SCRIM_MEDIUM,
    COLOR_SCRIM_STRONG,
)


PRELOAD_AHEAD = 1
KEEP_BEHIND = 1
GALLERY_THUMB_PX = 1600
CHEVRON_SIZE = 54


# image_id, max_dim → bytes
ThumbLoader = Callable[[int, int], Optional[bytes]]
# scan_id, stage → ordered list[(label, image_id, node_id)]
StageResolver = Callable[[int, str], list[tuple[str, Optional[int], Optional[int]]]]
# scan_id → stage_name (sensible default stage on entry, e.g. last
# stage). May return None.
DefaultStageProvider = Callable[[int], Optional[str]]
# scan_id, branch_label → current chosen node_id for that branch (None
# if no chosen yet). Used to colour the star.
BranchChosenProvider = Callable[[int, str], Optional[int]]
# scan_id, branch_label, node_id → write chosen_node_id for this branch
# only. Persisted.
BranchChosenWriter = Callable[[int, str, int], None]
# scan_id, branch_label → True if this branch is currently trashed.
BranchTrashedProvider = Callable[[int, str], bool]
# scan_id, branch_label, trashed: bool → persist the trashed state.
BranchTrashedWriter = Callable[[int, str, bool], None]
# scan_id → ordered list[(label, image_id, node_id)] — one entry per
# branch, each showing the branch's CHOSEN stage (potentially different
# per branch). Used by the "Show selected" toggle.
SelectedResolver = Callable[[int], list[tuple[str, Optional[int], Optional[int]]]]


class _ChevronButton(QToolButton):
    def __init__(self, glyph: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from lib.gui.theme import lucide_pixmap as _lp
        self.setIcon(QPixmap(_lp(glyph, color=COLOR_FONT_INVERSE, size=36)))
        self.setIconSize(QSize(36, 36))
        self.setFixedSize(CHEVRON_SIZE, CHEVRON_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QToolButton{"
            f"  background:{COLOR_SCRIM_MEDIUM};"
            f"  border:1px solid {COLOR_OUTLINE_STRONG};"
            "  border-radius:27px;"
            f"  color:{COLOR_FONT_INVERSE};"
            "}"
            f"QToolButton:hover{{background:{COLOR_SCRIM_STRONG};}}"
            f"QToolButton:disabled{{background:{COLOR_SCRIM_LIGHT}; "
            f" border-color:{COLOR_OUTLINE_SUBTLE};}}"
        )


class ScansGalleryView(QWidget):
    scan_changed = Signal(int)              # scan_id
    stage_changed = Signal(int, str)        # scan_id, stage_name
    # Open the per-leaf debug viewer (same signal contract as the
    # table view) — emitted on left-click of a cell's miniature.
    debug_requested = Signal(int, str)      # node_id, label

    # Decoded-pixmap cache budget (gallery pixmaps are ~8-14 MB each).
    _CACHE_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self, *,
                 scans_provider: Callable[[], list[tuple[int, str]]],
                 stages_provider: Callable[[], list[str]],
                 stage_resolver: StageResolver,
                 thumb_loader: ThumbLoader,
                 default_stage_provider: Optional[DefaultStageProvider] = None,
                 branch_chosen_provider: Optional[BranchChosenProvider] = None,
                 branch_chosen_writer: Optional[BranchChosenWriter] = None,
                 branch_trashed_provider: Optional[BranchTrashedProvider] = None,
                 branch_trashed_writer: Optional[BranchTrashedWriter] = None,
                 selected_resolver: Optional[SelectedResolver] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._scans_provider = scans_provider
        self._stages_provider = stages_provider
        self._stage_resolver = stage_resolver
        self._thumb_loader = thumb_loader
        self._default_stage_provider = default_stage_provider
        self._branch_chosen_provider = branch_chosen_provider
        self._branch_chosen_writer = branch_chosen_writer
        self._branch_trashed_provider = branch_trashed_provider
        self._branch_trashed_writer = branch_trashed_writer
        self._selected_resolver = selected_resolver
        # When True, each cell shows the branch's CHOSEN stage (which may
        # differ across branches). Stage axis is meaningless in this
        # mode → left/right chevrons hidden.
        self._show_selected: bool = False

        self._scans: list[tuple[int, str]] = []
        self._stages: list[str] = ["raw"]
        self._scan_idx: int = 0
        self._stage_idx: int = 0
        # Strong focus so arrow keys / WASD route here when the user
        # tabs in or clicks the view.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Cache (scan_id, stage_name) → list[(label, QPixmap|None)].
        # Byte-budgeted LRU (see _evict_lru) instead of a raw count cap.
        self._cache: "OrderedDict[tuple[int, str], list[tuple[str, Optional[QPixmap]]]]" = OrderedDict()

        # The thumb loader builds large gallery previews off the GUI thread.
        # Coalesce its `ready` bursts into one reload (only while visible).
        self._thumb_ready_timer = QTimer(self)
        self._thumb_ready_timer.setSingleShot(True)
        self._thumb_ready_timer.setInterval(120)
        self._thumb_ready_timer.timeout.connect(self._on_thumb_ready_tick)
        ready = getattr(thumb_loader, "ready", None)
        if ready is not None:
            ready.connect(lambda _img_id: self._thumb_ready_timer.start())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._host = QFrame()
        self._host.setStyleSheet(f"background:{COLOR_BG};")
        hv = QVBoxLayout(self._host)
        hv.setContentsMargins(20, 20, 20, 6)
        hv.setSpacing(8)
        self._strip = QWidget()
        self._strip_l = QHBoxLayout(self._strip)
        self._strip_l.setContentsMargins(0, 0, 0, 0)
        self._strip_l.setSpacing(12)
        self._strip_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl = QLabel(self.tr("No scans"))
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet(f"color:{COLOR_FONT_PLACEHOLDER}; font-size:14px;")
        self._strip_l.addWidget(self._empty_lbl)
        hv.addWidget(self._strip, 1)
        self._caption = QLabel("")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setStyleSheet(f"color:{COLOR_FONT_MUTED}; font-size:12px; padding:6px;")
        hv.addWidget(self._caption)
        outer.addWidget(self._host, 1)

        # Chevrons: absolute-positioned children of host. Re-laid out in
        # `resizeEvent`. StackedLayout-on-top stops mouse events from
        # reaching siblings cleanly, so just float them.
        self._btn_up = _ChevronButton("chevron-up", self._host)
        self._btn_down = _ChevronButton("chevron-down", self._host)
        self._btn_left = _ChevronButton("chevron-left", self._host)
        self._btn_right = _ChevronButton("chevron-right", self._host)
        self._btn_up.clicked.connect(self.go_up)
        self._btn_down.clicked.connect(self.go_down)
        self._btn_left.clicked.connect(self.go_left)
        self._btn_right.clicked.connect(self.go_right)
        for b in (self._btn_up, self._btn_down, self._btn_left, self._btn_right):
            b.raise_()
        self._refresh_chevrons()

    # ── public API ─────────────────────────────────────────────────
    def reload(self, *, scan_to_focus: Optional[int] = None,
               jump_to_latest: bool = False) -> None:
        prev_scan = self._scans[self._scan_idx][0] if self._scans else None
        prev_stage = self._stages[self._stage_idx] if self._stages else None
        self._scans = list(self._scans_provider() or [])
        self._stages = list(self._stages_provider() or []) or ["raw"]
        # Scan focus.
        if jump_to_latest and self._scans:
            self._scan_idx = len(self._scans) - 1
        elif scan_to_focus is not None:
            ids = [s[0] for s in self._scans]
            self._scan_idx = ids.index(scan_to_focus) if scan_to_focus in ids \
                else max(0, len(self._scans) - 1)
        elif prev_scan is not None and self._scans:
            ids = [s[0] for s in self._scans]
            self._scan_idx = ids.index(prev_scan) if prev_scan in ids \
                else max(0, len(self._scans) - 1)
        else:
            self._scan_idx = 0
        # Stage focus: prefer DB-recorded chosen, else preserve prev, else
        # default to last stage (== final pipeline output, what the user
        # normally wants to see).
        self._stage_idx = self._pick_default_stage(prev_stage)
        # Evict cache entries for now-missing scans.
        valid = {s[0] for s in self._scans}
        for key in list(self._cache.keys()):
            if key[0] not in valid:
                self._cache.pop(key, None)
        self._present()

    def _pick_default_stage(self, prev_stage: Optional[str]) -> int:
        if not self._scans or not self._stages:
            return 0
        # 1. DB-recorded chosen stage for the focused scan.
        if self._default_stage_provider is not None:
            try:
                default = self._default_stage_provider(self._scans[self._scan_idx][0])
            except Exception:
                default = None
            if default and default in self._stages:
                return self._stages.index(default)
        # 2. Preserve previously focused stage.
        if prev_stage is not None and prev_stage in self._stages:
            return self._stages.index(prev_stage)
        # 3. Last stage that actually has a node for THIS scan. The global
        #    final stage (e.g. replay) isn't present on every scan, and
        #    landing there shows an empty "(no node for step …)" frame.
        return self._last_stage_with_node(self._scans[self._scan_idx][0])

    def _last_stage_with_node(self, scan_id: int) -> int:
        """Highest stage index that resolves to a node for `scan_id`, else 0."""
        for i in range(len(self._stages) - 1, -1, -1):
            try:
                if self._stage_resolver(scan_id, self._stages[i]):
                    return i
            except Exception:
                continue
        return 0

    def go_left(self) -> None:
        if self._stage_idx > 0:
            self._stage_idx -= 1
            self._present()

    def go_right(self) -> None:
        if self._stage_idx + 1 < len(self._stages):
            self._stage_idx += 1
            self._present()

    def go_up(self) -> None:
        if self._scan_idx > 0:
            self._scan_idx -= 1
            self._stage_idx = self._carry_stage_across_scan(self._stage_idx)
            self._present()

    def go_down(self) -> None:
        if self._scan_idx + 1 < len(self._scans):
            self._scan_idx += 1
            self._stage_idx = self._carry_stage_across_scan(self._stage_idx)
            self._present()

    def _carry_stage_across_scan(self, prev_idx: int) -> int:
        """Hold the current stage when scrolling scan-to-scan if the new
        scan actually has data at that stage. Otherwise fall back to the
        last stage available (final pipeline output)."""
        if not self._stages:
            return 0
        if not (0 <= prev_idx < len(self._stages)):
            return self._pick_default_stage(None)
        prev_stage = self._stages[prev_idx]
        focused = self._scans[self._scan_idx][0]
        try:
            items = self._stage_resolver(focused, prev_stage) or []
        except Exception:
            items = []
        if items:
            return prev_idx
        return self._last_stage_with_node(focused)

    def focused_scan_id(self) -> Optional[int]:
        if not self._scans:
            return None
        return self._scans[self._scan_idx][0]

    def set_show_selected(self, on: bool) -> None:
        """Toggle 'show only the chosen stage per page' mode. Hides
        the left/right stage chevrons since the stage axis no longer
        applies."""
        on = bool(on)
        if on == self._show_selected:
            return
        self._show_selected = on
        self._present()

    # ── internals ──────────────────────────────────────────────────
    def _on_trash_clicked(self, branch_label: str) -> None:
        """Toggle trashed state for `(focused scan, branch_label)`."""
        if self._branch_trashed_writer is None or not self._scans:
            return
        scan_id = self._scans[self._scan_idx][0]
        current = False
        if self._branch_trashed_provider is not None:
            try:
                current = bool(self._branch_trashed_provider(int(scan_id), str(branch_label)))
            except Exception:
                current = False
        try:
            self._branch_trashed_writer(int(scan_id), str(branch_label), not current)
        except Exception:
            return
        self._present()

    def _install_cell_click(self, widget, node_id: int, label: str) -> None:
        """Wire a left-click on ``widget`` to emit ``debug_requested``.
        The filter is parented to ``widget`` → lifetime auto-managed.
        Distinguish click from drag by requiring < 8 px movement
        between press and release."""
        from PySide6.QtCore import QEvent, QObject

        view = self

        class _CellClickFilter(QObject):
            def __init__(self, parent):
                super().__init__(parent)
                self._press_pos = None

            def eventFilter(self, obj, ev):  # noqa: N802 — Qt API
                t = ev.type()
                if t == QEvent.Type.MouseButtonPress:
                    if ev.button() == Qt.MouseButton.LeftButton:
                        self._press_pos = ev.position().toPoint()
                elif t == QEvent.Type.MouseButtonRelease:
                    if (ev.button() == Qt.MouseButton.LeftButton
                            and self._press_pos is not None):
                        moved = (ev.position().toPoint()
                                  - self._press_pos).manhattanLength()
                        self._press_pos = None
                        if moved < 8:
                            view.debug_requested.emit(node_id, label)
                            return True
                return False

        shim = _CellClickFilter(widget)
        widget.installEventFilter(shim)

    def _on_star_clicked(self, branch_label: str, node_id: Optional[int]) -> None:
        """Star toggle: marks (scan, branch, node) as chosen. The DB
        writer is per-branch — different branches can pick different
        stages without interfering with each other."""
        import sys
        if self._branch_chosen_writer is None:
            print("[gallery] star click ignored — no writer wired", file=sys.stderr)
            return
        if not self._scans:
            print("[gallery] star click ignored — empty scan list", file=sys.stderr)
            return
        if node_id is None:
            print(f"[gallery] star click ignored — no node_id for {branch_label!r}",
                  file=sys.stderr)
            return
        scan_id = self._scans[self._scan_idx][0]
        try:
            self._branch_chosen_writer(int(scan_id), str(branch_label), int(node_id))
        except Exception as e:
            print(f"[gallery] branch_chosen_writer raised: {e}", file=sys.stderr)
            return
        print(
            f"[gallery] chosen written: scan={scan_id} branch={branch_label!r} "
            f"node={node_id}", file=sys.stderr,
        )
        self.stage_changed.emit(int(scan_id), self._stages[self._stage_idx])
        # Repaint stars — `branch_chosen_provider` is re-queried on every
        # `_build_cell` call so the gold fill flips immediately.
        self._present()

    def _on_thumb_ready_tick(self) -> None:
        """A batch of background thumbnails finished. Re-render the current
        view (only when visible) so the now-warm thumbs appear; `_cache_get`
        skips memoising pending entries, so this re-resolve is a cache hit."""
        if self.isVisible():
            self.reload()

    def _cache_get(self, scan_id: int, stage: str
                   ) -> list[tuple[str, Optional[QPixmap], Optional[int]]]:
        key = (scan_id, stage)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            items = self._stage_resolver(scan_id, stage) or []
        except Exception:
            items = []
        out: list[tuple[str, Optional[QPixmap], Optional[int]]] = []
        pending = False  # a thumb is still building off-thread
        for tup in items:
            # Resolver tuple: (label, image_id, node_id).
            label = tup[0]
            image_id = tup[1] if len(tup) > 1 else None
            node_id = tup[2] if len(tup) > 2 else None
            pix: Optional[QPixmap] = None
            if image_id is not None:
                try:
                    blob = self._thumb_loader(int(image_id), GALLERY_THUMB_PX)
                except Exception:
                    blob = None
                if blob:
                    img = QImage.fromData(blob)
                    if not img.isNull():
                        pix = QPixmap.fromImage(img)
                else:
                    pending = True
            out.append((label, pix, node_id))
        # Don't memoise a half-built result — the async loader will fire
        # `ready` and we re-resolve, hitting the now-warm thumb cache.
        if not pending:
            self._cache[key] = out
            self._evict_lru()
        return out

    @staticmethod
    def _entry_bytes(entry) -> int:
        n = 0
        for _label, pix, _node in entry:
            if pix is not None and not pix.isNull():
                n += pix.width() * pix.height() * max(1, pix.depth()) // 8
        return n

    def _evict_lru(self) -> None:
        keep = set()
        if self._scans:
            lo = max(0, self._scan_idx - KEEP_BEHIND)
            hi = min(len(self._scans), self._scan_idx + PRELOAD_AHEAD + 1)
            for j in range(lo, hi):
                sid = self._scans[j][0]
                for stage in self._stages:
                    keep.add((sid, stage))
        # Byte-budgeted LRU. Each 1600 px gallery pixmap is ~8-14 MB, so the
        # old count cap (16 × branches) could retain 200-450 MB. Evict oldest
        # non-visible entries until the decoded pixmaps fit the budget.
        total = sum(self._entry_bytes(v) for v in self._cache.values())
        if total <= self._CACHE_BUDGET_BYTES:
            return
        for k in list(self._cache.keys()):   # OrderedDict LRU: oldest first
            if total <= self._CACHE_BUDGET_BYTES:
                break
            if k in keep:
                continue
            total -= self._entry_bytes(self._cache[k])
            self._cache.pop(k, None)

    def _clear_strip(self) -> None:
        while self._strip_l.count() > 0:
            it = self._strip_l.takeAt(0)
            w = it.widget()
            if w is not None and w is not self._empty_lbl:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent; with no parent that's a bare
                # top-level window flash.
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _present(self) -> None:
        self._clear_strip()
        if not self._scans:
            self._strip_l.addWidget(self._empty_lbl)
            self._empty_lbl.show()
            self._caption.setText("")
            self._refresh_chevrons()
            self._position_chevrons()
            return
        self._empty_lbl.hide()
        scan_id, raw_stem = self._scans[self._scan_idx]
        stage = self._stages[self._stage_idx]
        if self._show_selected and self._selected_resolver is not None:
            items = self._build_selected_items(scan_id)
            stage_label = "selected"
        else:
            items = self._cache_get(scan_id, stage)
            stage_label = stage
        if not items:
            placeholder = QLabel(self.tr("(no node for stage '{stage}')").format(stage=stage_label))
            placeholder.setStyleSheet(f"color:{COLOR_FONT_PLACEHOLDER}; font-size:13px;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._strip_l.addWidget(placeholder)
        else:
            avail_w = max(200, self._host.width() - 2 * CHEVRON_SIZE - 80)
            avail_h = max(200, self._host.height() - 120)
            per_w = max(160, avail_w // max(1, len(items)))
            for (label, pix, node_id) in items:
                is_chosen = False
                if self._branch_chosen_provider is not None and node_id is not None:
                    try:
                        chosen_nid = self._branch_chosen_provider(scan_id, label)
                        is_chosen = (chosen_nid is not None
                                     and int(chosen_nid) == int(node_id))
                    except Exception:
                        is_chosen = False
                is_trashed = False
                if self._branch_trashed_provider is not None:
                    try:
                        is_trashed = bool(
                            self._branch_trashed_provider(scan_id, label)
                        )
                    except Exception:
                        is_trashed = False
                cell = self._build_cell(label, pix, per_w, avail_h,
                                          node_id=node_id,
                                          is_chosen=is_chosen,
                                          is_trashed=is_trashed)
                self._strip_l.addWidget(cell)
        n_scans = len(self._scans)
        n_stages = len(self._stages)
        if self._show_selected:
            self._caption.setText(self.tr(
                "#{scan}  {stem}   ·   "
                "scan {idx} / {total}   ·   "
                "showing selected stage per page"
            ).format(
                scan=scan_id, stem=raw_stem,
                idx=self._scan_idx + 1, total=n_scans,
            ))
        else:
            self._caption.setText(self.tr(
                "#{scan}  {stem}   ·   "
                "scan {scan_idx} / {snap_total}   ·   "
                "stage '{stage}'  ({stage_idx} / {stage_total})"
            ).format(
                scan=scan_id, stem=raw_stem,
                scan_idx=self._scan_idx + 1, snap_total=n_scans,
                stage=stage,
                stage_idx=self._stage_idx + 1, stage_total=n_stages,
            ))
        # Warm adjacent.
        for off in (-1, 1):
            j = self._stage_idx + off
            if 0 <= j < len(self._stages):
                self._cache_get(scan_id, self._stages[j])
        for off in (-1, 1):
            j = self._scan_idx + off
            if 0 <= j < len(self._scans):
                self._cache_get(self._scans[j][0], stage)
        self.scan_changed.emit(scan_id)
        self._refresh_chevrons()
        self._position_chevrons()

    def _build_cell(self, label: str, pix: Optional[QPixmap],
                    per_w: int, avail_h: int, *,
                    node_id: Optional[int] = None,
                    is_chosen: bool = False,
                    is_trashed: bool = False) -> QWidget:
        from lib.gui.theme import lucide_pixmap as _lp
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Image container with star overlay in top-left.
        img_host = QWidget()
        img_host.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        # Use a manual layout via a child label + parented star button so
        # the star floats on top of the image.
        img_lbl = QLabel(img_host)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet(
            f"background:transparent; border:1px solid {COLOR_OUTLINE_SUBTLE}; "
            "border-radius:6px;"
        )
        # Image label must not eat star clicks — let mouse events fall
        # through to the parent so the star (z-raised sibling) receives them.
        img_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if pix is None or pix.isNull():
            img_lbl.setText(self.tr("(no image)"))
            img_lbl.setStyleSheet(img_lbl.styleSheet() + f" color:{COLOR_FONT_DIM};")
            img_lbl.setMinimumSize(160, 200)
            cell_w, cell_h = max(per_w, 160), 200
        else:
            target = QSize(per_w, avail_h)
            scaled = pix.scaled(target,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            img_lbl.setPixmap(scaled)
            cell_w, cell_h = scaled.width(), scaled.height()
        # Resize the image container to match the actual scaled image
        # so the star can be positioned relative to the visible bounds.
        img_host.setFixedSize(cell_w, cell_h)
        img_lbl.setGeometry(0, 0, cell_w, cell_h)
        # Click on the miniature → open the per-leaf debug viewer.
        # Skipped when the cell has no node_id (i.e. the chain hasn't
        # produced this stage yet for the scan).
        if node_id is not None:
            img_host.setCursor(Qt.CursorShape.PointingHandCursor)
            # PySide6 doesn't honour `widget.mousePressEvent = fn`
            # — Qt's virtual dispatch goes through the C++ vtable, so
            # the assignment never fires. Use an event filter on the
            # host widget instead; child buttons (star, trash) still
            # eat their own clicks first via standard child hit-test.
            self._install_cell_click(img_host, int(node_id), label)
        # Star button — rating-style, SOLID fill. Gray when not chosen,
        # gold when chosen. No backdrop / circle / outline — flat fill.
        # SVG `stroke=` doesn't accept rgba() so we keep plain hex.
        star_size = 28
        star_color = COLOR_MEDAL_GOLD if is_chosen else COLOR_FONT_PLACEHOLDER
        spix = _lp("star-filled", color=star_color, size=star_size)
        spix.setDevicePixelRatio(2.0)
        star = QToolButton(img_host)
        star.setIcon(QPixmap(spix))
        star.setIconSize(QSize(star_size, star_size))
        star.setFixedSize(star_size + 4, star_size + 4)
        star.setCursor(Qt.CursorShape.PointingHandCursor)
        star.setAutoRaise(False)
        star.setToolTip(
            self.tr("Chosen output for this page") if is_chosen
            else self.tr("Click to mark this stage as the chosen output for this page")
        )
        star.setStyleSheet(
            "QToolButton{background:transparent; border:none; padding:0; margin:0;}"
            f"QToolButton:hover{{background:{COLOR_OUTLINE_GHOST}; border-radius:4px;}}"
        )
        # Semi-opaque so the underlying image breathes through but the
        # selected-gold still pops vs. the gray unselected variant.
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        seff = QGraphicsOpacityEffect(star)
        seff.setOpacity(0.85 if is_chosen else 0.55)
        star.setGraphicsEffect(seff)
        star.move(6, 6)
        star.raise_()
        if node_id is not None:
            star.clicked.connect(
                lambda _, lbl=label, nid=node_id: self._on_star_clicked(lbl, nid)
            )
        else:
            star.setEnabled(False)

        # Layout hide/show toggle — flat icon, bottom-left. Shows the
        # CURRENT state (eye-off when hidden, eye when visible). Trash
        # is reserved for scan-level destructive delete elsewhere.
        trash_size = 24
        trash_glyph = "eye-off" if is_trashed else "eye"
        trash_color = COLOR_FONT_PLACEHOLDER
        tpix = _lp(trash_glyph, color=trash_color, size=trash_size)
        tpix.setDevicePixelRatio(2.0)
        trash_btn = QToolButton(img_host)
        trash_btn.setIcon(QPixmap(tpix))
        trash_btn.setIconSize(QSize(trash_size, trash_size))
        trash_btn.setFixedSize(trash_size + 4, trash_size + 4)
        trash_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        trash_btn.setAutoRaise(False)
        trash_btn.setToolTip(
            self.tr("Show page") if is_trashed else self.tr("Hide page")
        )
        trash_btn.setStyleSheet(
            "QToolButton{background:transparent; border:none; padding:0; margin:0;}"
            f"QToolButton:hover{{background:{COLOR_OUTLINE_GHOST}; border-radius:4px;}}"
        )
        # Bottom-left.
        trash_btn.move(6, cell_h - (trash_size + 4) - 6)
        trash_btn.raise_()
        trash_btn.clicked.connect(
            lambda _, lbl=label: self._on_trash_clicked(lbl)
        )

        if is_trashed:
            # Dim the image AND drop a big trash overlay glyph centered.
            from PySide6.QtWidgets import QGraphicsOpacityEffect
            eff = QGraphicsOpacityEffect(img_lbl)
            eff.setOpacity(0.30)
            img_lbl.setGraphicsEffect(eff)
            overlay_size = min(96, max(48, int(cell_h * 0.35)))
            opix = _lp("eye-off", color=COLOR_FONT_ON_BUTTON, size=overlay_size)
            opix.setDevicePixelRatio(2.0)
            overlay_lbl = QLabel(img_host)
            overlay_lbl.setPixmap(QPixmap(opix))
            overlay_lbl.setFixedSize(overlay_size, overlay_size)
            overlay_lbl.setStyleSheet("background:transparent; border:none;")
            overlay_lbl.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
            # Semi-opaque via effect — SVG fill colors don't take rgba.
            from PySide6.QtWidgets import QGraphicsOpacityEffect
            oeff = QGraphicsOpacityEffect(overlay_lbl)
            oeff.setOpacity(0.45)
            overlay_lbl.setGraphicsEffect(oeff)
            overlay_lbl.move((cell_w - overlay_size) // 2,
                              (cell_h - overlay_size) // 2)
            overlay_lbl.raise_()
            # Star stays interactive on top — re-raise after overlay.
            star.raise_()
            trash_btn.raise_()

        v.addWidget(img_host, 1, Qt.AlignmentFlag.AlignCenter)
        if label:
            cap = QLabel(label)
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap.setStyleSheet(f"color:{COLOR_FONT_PLACEHOLDER}; font-size:11px;")
            v.addWidget(cap)
        return host

    def _refresh_chevrons(self) -> None:
        # Up/down always control scan navigation.
        self._btn_up.setEnabled(self._scan_idx > 0)
        self._btn_down.setEnabled(self._scan_idx + 1 < len(self._scans))
        # Always visible — when show-selected mode is on the stage axis
        # is meaningless, so they stay disabled (= dimmed) instead of
        # disappearing (avoids the layout-shift on every toggle).
        self._btn_left.show()
        self._btn_right.show()
        if self._show_selected:
            self._btn_left.setEnabled(False)
            self._btn_right.setEnabled(False)
        else:
            self._btn_left.setEnabled(self._stage_idx > 0)
            self._btn_right.setEnabled(self._stage_idx + 1 < len(self._stages))
        # Visible dim when disabled — QSS only tints the background; we
        # want the icon to fade too. One reusable opacity effect per btn.
        for btn in (self._btn_up, self._btn_down,
                    self._btn_left, self._btn_right):
            self._set_button_dim(btn, not btn.isEnabled())

    @staticmethod
    def _set_button_dim(btn: QToolButton, dim: bool) -> None:
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        eff = btn.graphicsEffect()
        if eff is None:
            eff = QGraphicsOpacityEffect(btn)
            btn.setGraphicsEffect(eff)
        eff.setOpacity(0.25 if dim else 1.0)

    def _build_selected_items(self, scan_id: int
                                ) -> list[tuple[str, Optional[QPixmap], Optional[int]]]:
        """For show-selected mode: pull one (label, pix, node_id) per
        branch using the chosen_node_id resolver. Bypasses the
        stage cache because the per-cell stages may differ across
        branches and aren't keyed by stage."""
        if self._selected_resolver is None:
            return []
        try:
            items = self._selected_resolver(scan_id) or []
        except Exception:
            return []
        out: list[tuple[str, Optional[QPixmap], Optional[int]]] = []
        for tup in items:
            label = tup[0]
            image_id = tup[1] if len(tup) > 1 else None
            node_id = tup[2] if len(tup) > 2 else None
            pix: Optional[QPixmap] = None
            if image_id is not None:
                try:
                    blob = self._thumb_loader(int(image_id), GALLERY_THUMB_PX)
                except Exception:
                    blob = None
                if blob:
                    img = QImage.fromData(blob)
                    if not img.isNull():
                        pix = QPixmap.fromImage(img)
            out.append((label, pix, node_id))
        return out

    def _position_chevrons(self) -> None:
        """Float chevrons absolutely so they sit ABOVE the strip but
        receive clicks (QStackedLayout(StackAll) eats events that pass
        through transparent regions of a top widget on some platforms)."""
        margin = 16
        w = self._host.width()
        h = self._host.height()
        if w <= 0 or h <= 0:
            return
        cx = (w - CHEVRON_SIZE) // 2
        cy = (h - CHEVRON_SIZE) // 2
        self._btn_up.move(cx, margin)
        self._btn_down.move(cx, h - CHEVRON_SIZE - margin - 28)  # 28 = caption strip
        self._btn_left.move(margin, cy)
        self._btn_right.move(w - CHEVRON_SIZE - margin, cy)
        for b in (self._btn_up, self._btn_down, self._btn_left, self._btn_right):
            b.show()
            b.raise_()

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self._position_chevrons()
        if self._scans:
            # Cheap on cached pixmaps; re-fit cells to the new width.
            self._present()

    def showEvent(self, ev):  # noqa: N802
        super().showEvent(ev)
        self._position_chevrons()
        # Grab focus so arrow keys route here right away.
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    # ── keyboard nav ──────────────────────────────────────────────
    def keyPressEvent(self, ev):  # noqa: N802
        k = ev.key()
        if k in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self.go_left()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self.go_right()
        elif k in (Qt.Key.Key_Up, Qt.Key.Key_W):
            self.go_up()
        elif k in (Qt.Key.Key_Down, Qt.Key.Key_S):
            self.go_down()
        else:
            super().keyPressEvent(ev)
