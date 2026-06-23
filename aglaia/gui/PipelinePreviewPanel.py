# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Live preview panel for the pipeline editor.

Pipeline edits → debounced re-run of the pipeline on one chosen scan's
source image → preview swap. No DB writes; the worker runs the
processors in-process on a QThread.

Layout:

    ┌─ Preview ───────────────────────────────────────────────────┐
    │  [Scan ▼]  [Layout ▼]  Zoom  ▬▬○  [✓autoupdate]  [↻]        │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  preview image (fit)                                 │   │
    │  │  ┌────────┐                                           │   │
    │  │  │  PiP   │  (anchored bottom-left, shows zoomed      │   │
    │  │  │ zoomed │   crop centred on cursor)                 │   │
    │  │  └────────┘                                           │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

The Stage dropdown lists every intermediate output from the run —
`NN_processor` for linear steps and `NN_processor · branch_path` for
branches. Final-leaf entries get a trailing ★. The panel caches every
frame, so swapping stages doesn't rerun the pipeline — only edits do.
"""

from __future__ import annotations

import copy
import dataclasses as _dc
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.processors import registry as _proc_registry
from aglaia.gui.ZoomCanvas import ZoomCanvas, ZoomToolbar
from aglaia.gui.theme import icon
from aglaia.gui.widgets import Card
from aglaia.gui.colors import (
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_SECONDARY,
    COLOR_OUTLINE,
    COLOR_OUTLINE_STRONG,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BORDER,
    COLOR_SCRIM_MEDIUM,
)


# ── source loading ───────────────────────────────────────────────────

def _list_snaps(db_path: str) -> list[tuple[int, str]]:
    """Return [(scan_id, display_label)] for every active scan."""
    from aglaia.storage.db import db_session
    with db_session(db_path) as conn:
        rows = conn.execute(
            "SELECT id, idx, source_ref FROM scans "
            "WHERE deleted_at IS NULL "
            "ORDER BY page_order ASC, idx ASC"
        ).fetchall()
    out = []
    for r in rows:
        ref = (r["source_ref"] or "").rsplit("/", 1)[-1]
        label = f"#{int(r['idx']):03d}"
        if ref:
            label += f" — {ref}"
        out.append((int(r["id"]), label))
    return out


def _load_source(db_path: str, scan_id: int) -> tuple[np.ndarray, float]:
    """Decode the scan's root-node blob. Returns (BGR ndarray, dpi)."""
    from aglaia.storage.db import db_session
    with db_session(db_path) as conn:
        row = conn.execute(
            "SELECT s.capture_dpi, n.image_id "
            "FROM scans s LEFT JOIN nodes n ON n.id = s.root_node_id "
            "WHERE s.id = ?",
            (scan_id,),
        ).fetchone()
        if row is None or row["image_id"] is None:
            raise ValueError(f"scan {scan_id} has no root node")
        img_row = conn.execute(
            "SELECT blob, dpi FROM images WHERE id = ?", (row["image_id"],),
        ).fetchone()
        if img_row is None:
            raise ValueError(f"scan {scan_id} root image missing")
        buf = cv2.imdecode(
            np.frombuffer(bytes(img_row["blob"]), np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        dpi = float(img_row["dpi"] or row["capture_dpi"] or 120.0)
    return buf, dpi


# ── worker ───────────────────────────────────────────────────────────

class _PreviewWorker(QThread):
    """Runs a pipeline document on one source buffer; emits a dict
    `{branch_path: ndarray}` of leaf outputs, plus an error string."""

    done = Signal(dict, str)

    def __init__(self, src_buf: np.ndarray, src_dpi: float, doc: dict,
                 parent=None):
        super().__init__(parent)
        self._src = src_buf
        self._dpi = float(src_dpi)
        self._doc = copy.deepcopy(doc)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 — Qt API
        try:
            results = self._run()
            self.done.emit(results, "")
        except Exception as e:
            self.done.emit({}, f"{type(e).__name__}: {e}")
        finally:
            self._drop_mlx_thread_state()

    @staticmethod
    def _drop_mlx_thread_state() -> None:
        """MLX's CompilerCache is C++ thread_local; entries hold Python
        objects (the traced callables). If they survive until this QThread
        exits, the pthread TSD destructor deallocates them WITHOUT the GIL
        → SIGSEGV in tupledealloc. Dropping the compiled closures here —
        on this thread, with the GIL — erases the entries safely."""
        try:
            import gc

            from aglaia.processors import page_dewarp_mlx
            page_dewarp_mlx.clear_caches()
            gc.collect()
        except Exception:
            pass

    def _build_elements(self) -> list:
        from aglaia.workers.chain_abstraction import SimpleChainElement
        pipeline = self._doc.get("pipeline", []) or []
        out = []
        for idx, step in enumerate(pipeline, 1):
            proc_name = step.get("processor")
            if not proc_name:
                continue
            info = _proc_registry.get_processor(proc_name)
            if info is None:
                continue
            valid = {f.name for f in _dc.fields(info.option_cls)}
            opts_in = step.get("options", {}) or {}
            opts_clean = {k: v for k, v in opts_in.items() if k in valid}
            # Force-disable per-processor debug — the preview never
            # persists anything, no debug folders should leak.
            opts_clean["debug"] = False
            try:
                opts = info.option_cls(**opts_clean)
            except Exception:
                # Fall back to defaults if option parsing fails.
                opts = info.option_cls()
            out.append(SimpleChainElement(
                processor_name=proc_name, options=opts,
                instance_name=f"{idx:02d}_{proc_name}",
            ))
        return out

    def _run(self) -> dict[str, np.ndarray]:
        if self._src is None or self._src.size == 0:
            raise ValueError("empty source image")
        elements = self._build_elements()
        registry = _proc_registry.processor_classes()
        # Cache processor instances so OPTION_CLASS state (loaded models,
        # cached kernels) doesn't repeatedly rebuild during a single run.
        processors = [registry[el.processor_name](el.options) for el in elements]

        buf_type = ImageType.GRAY if self._src.ndim == 2 else ImageType.COLOR
        root = ImageBuffer(
            buffer=self._src.copy(), type=buf_type, dpi=self._dpi,
            filestem="preview", branch_path="", branch_label=None, depth=0,
        )
        frames: dict[str, np.ndarray] = {"00_source": self._src.copy()}
        self._walk(root, processors, elements, 0, frames)
        return frames

    def _walk(self, buf: ImageBuffer, processors: list, elements: list,
              idx: int, frames: dict[str, np.ndarray],
              lineage: Optional[list] = None) -> None:
        """Run forward, dropping every intermediate output into `frames`.

        Key format: ``NN_processor`` for linear steps, ``NN_processor · X``
        for branches (X = branch_path). Final-leaf entries get a trailing
        ``★`` so the dropdown can distinguish them at a glance. A
        synthetic ``NN+1_replay … ★`` frame is appended per leaf — same
        engine as the production replay pass, just driven off the
        in-memory meta trail instead of DB rows.

        BW palette preservation mirrors the runtime worker: when a BW
        buffer flows into a non-binarizing step (skew, dewarp, margin)
        the result must stay BW or downstream binarized previews look
        smudged into greyscale.
        """
        if self._cancel:
            return
        if lineage is None:
            lineage = []
        if idx >= len(processors):
            return
        proc = processors[idx]
        el = elements[idx]
        try:
            result = proc.process(buf)
        except Exception as e:
            raise RuntimeError(
                f"step {idx + 1} ({el.processor_name}): {e}"
            ) from e
        if result is None:
            return
        if isinstance(result, list):
            outputs = result
        elif getattr(result, "children", None):
            outputs = list(result.children)
        else:
            outputs = [result]

        # BW palette preservation — same rule as IntegratedProcessingChain.
        # Without this, MarginSetter / PageDewarper / SkewFinder on a BW
        # input return interpolated greys; the preview shows them as the
        # non-binarised replay intermediate.
        if buf.type == ImageType.BW:
            from aglaia.processors.utils import binarize_fixed
            for t in outputs:
                if t.type != ImageType.BW or not t.check_binary():
                    t.buffer = binarize_fixed(t.buffer, 127)
                    t.type = ImageType.BW

        branched = len(outputs) > 1
        is_last = (idx == len(processors) - 1)
        for j, out in enumerate(outputs):
            out.depth = buf.depth + 1
            if branched:
                label = out.branch_label or chr(ord("A") + j)
                out.branch_label = label
                out.branch_path = (
                    f"{buf.branch_path}.{label}" if buf.branch_path else label
                )
            else:
                out.branch_path = buf.branch_path
                out.branch_label = buf.branch_label
            key = f"{idx + 1:02d}_{el.processor_name}"
            if out.branch_path:
                key += f" · {out.branch_path}"
            if is_last:
                key += " ★"
            frames[key] = out.buffer
            # Snapshot buffer + meta now: most processors mutate the
            # input buffer in place, so reading `out.meta` / `out.buffer`
            # at the end of the walk would give every step the FINAL
            # state, not its own stamp. Replay would then anchor on the
            # wrong node and replay arbitrary later transforms onto
            # what's already a post-MarginSetter result.
            meta_snapshot = dict(out.meta) if out.meta else {}
            buffer_snapshot = out.buffer.copy()
            sub_lineage = lineage + [{
                "step_idx": idx + 1,
                "meta": meta_snapshot,
                "buffer": buffer_snapshot,
            }]
            if is_last:
                if self._doc.get("replay", True):
                    self._add_replay_frame(sub_lineage, len(processors),
                                           out.branch_path, frames)
            else:
                self._walk(out, processors, elements, idx + 1, frames,
                           sub_lineage)

    def _add_replay_frame(self, lineage: list, n_steps: int,
                          branch_path: str,
                          frames: dict[str, np.ndarray]) -> None:
        """Apply the same replay pass the runtime adds to every leaf:
        fuse the geometric transforms back onto the latest non-replay
        buffer (typically PageDetector output), then binarize once at
        the end. Stamps the result as `NN+1_replay [· branch] ★`."""
        try:
            from aglaia.workers.Replay import (
                _apply_binarize, _apply_dewarp, _apply_margin,
                _apply_perspective, _apply_rotate, _ordered_replay_steps,
            )
        except Exception:
            return
        nodes: list[dict] = []
        for entry in lineage:
            nodes.append({
                "step_idx": int(entry["step_idx"]),
                "meta": entry["meta"],
                "_buffer": entry["buffer"],
            })
        latest_non_replay_idx = -1
        for i, n in enumerate(nodes):
            if n["meta"].get("replay_kind") is None:
                latest_non_replay_idx = i
        if latest_non_replay_idx < 0:
            return  # no anchor — every step participated in replay
        candidate = nodes[latest_non_replay_idx + 1:]
        steps = _ordered_replay_steps(candidate)
        if not steps:
            return
        source = nodes[latest_non_replay_idx]
        src_buf = source["_buffer"].copy()
        if src_buf.ndim == 2:
            mask_shape = src_buf.shape
        else:
            mask_shape = src_buf.shape[:2]
        roi = source["meta"].get("roi")
        if roi:
            mask = np.zeros(mask_shape, dtype=np.uint8)
            pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask = np.full(mask_shape, 255, dtype=np.uint8)
        for n in steps:
            kind = n["meta"]["replay_kind"]
            params = n["meta"]["replay_params"]
            if kind == "rotate":
                src_buf, mask = _apply_rotate(src_buf, mask, params)
            elif kind == "perspective":
                src_buf, mask = _apply_perspective(src_buf, mask, params)
            elif kind == "dewarp":
                src_buf, mask = _apply_dewarp(src_buf, mask, params)
            elif kind == "margin":
                src_buf, mask = _apply_margin(src_buf, mask, params)
            elif kind == "binarize":
                src_buf, mask = _apply_binarize(
                    src_buf, mask, params, dpi=self._dpi,
                )
        key = f"{n_steps + 1:02d}_replay"
        if branch_path:
            key += f" · {branch_path}"
        key += " ★"
        frames[key] = src_buf


# ── the panel itself ─────────────────────────────────────────────────

class PipelinePreviewPanel(QWidget):
    """Right-hand panel embedded in the pipeline editor.

    Public API:
      * `set_pipeline_doc(doc)` — store the current pipeline document
        and (if autoupdate is on) schedule a debounced re-run.
      * `request_refresh()` — kick a re-run immediately.
      * `set_db_path(path)` / `set_default_snap(scan_id)` — late binders
        when the host doesn't have these at construction.
    """

    DEBOUNCE_MS = 1000

    def __init__(self, db_path: Optional[str] = None,
                 default_scan_id: Optional[int] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._db_path = db_path
        self._snap_id: Optional[int] = None
        self._src: Optional[np.ndarray] = None
        self._src_dpi: float = 120.0
        self._doc: dict = {}
        # Every intermediate output from the last forward run, keyed by
        # `NN_processor [· branch_path] [★]`. Drives the layout dropdown.
        self._frames: dict[str, np.ndarray] = {}
        self._worker: Optional[_PreviewWorker] = None
        self._pending: bool = False
        self._branch_pills: list = []
        # Last user-picked stage key. Survives auto-refresh runs so a
        # pipeline edit doesn't yank the user back to the default leaf
        # mid-iteration. Reset when the scan changes.
        self._preferred_key: Optional[str] = None

        self._build()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self._run_now)

        self._reload_snaps()
        if default_scan_id is not None:
            self._select_snap_id(default_scan_id)
        # Force a source load even when setCurrentIndex didn't change the
        # row (e.g. default scan already at index 0): the combo may have
        # been populated under blockSignals(), so currentIndexChanged
        # never fired.
        if self._snap_combo.count() > 0:
            self._on_snap_changed(self._snap_combo.currentIndex())

    # ── construction ──────────────────────────────────────────────
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        card = Card(elevated=True)
        body = card.layout()
        body.setSpacing(6)

        # Top bar — only source selectors so it stays narrow and the
        # splitter can be dragged far to the right. Stage combo has an
        # expanding size policy so it eats whatever slack remains.
        top = QHBoxLayout()
        top.setSpacing(8)

        self._snap_combo = QComboBox()
        self._snap_combo.setMinimumWidth(160)
        self._snap_combo.setSizePolicy(QSizePolicy.Policy.Preferred,
                                       QSizePolicy.Policy.Fixed)
        self._snap_combo.currentIndexChanged.connect(self._on_snap_changed)
        top.addWidget(QLabel(self.tr("Scan")))
        top.addWidget(self._snap_combo, 1)

        self._layout_combo = QComboBox()
        self._layout_combo.setMinimumWidth(140)
        self._layout_combo.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Fixed)
        self._layout_combo.currentIndexChanged.connect(self._on_page_changed)
        top.addWidget(QLabel(self.tr("Stage")))
        top.addWidget(self._layout_combo, 2)

        self._refresh_btn = QToolButton()
        self._refresh_btn.setIcon(icon("refresh-cw"))
        self._refresh_btn.setToolTip(self.tr("Refresh preview"))
        self._refresh_btn.clicked.connect(self._run_now)
        top.addWidget(self._refresh_btn)

        body.addLayout(top)

        # Canvas takes the middle (largest) chunk.
        self._canvas = ZoomCanvas(placeholder=self.tr("Preview will appear here"))
        body.addWidget(self._canvas, 1)

        # Bottom bar — zoom + autoupdate + status. Status right-aligned
        # so it doesn't push the controls around as its width changes.
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._zoom_bar = ZoomToolbar(self._canvas, default=2.0)
        bottom.addWidget(self._zoom_bar)

        self._auto = QCheckBox(self.tr("Auto-update"))
        self._auto.setChecked(True)
        self._auto.toggled.connect(self._on_auto_toggled)
        bottom.addWidget(self._auto)

        bottom.addStretch(1)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {COLOR_FONT_MUTED}; padding: 2px 4px;")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        bottom.addWidget(self._status, 1)

        body.addLayout(bottom)

        outer.addWidget(card, 1)

    # ── data binders ──────────────────────────────────────────────
    def set_db_path(self, db_path: str) -> None:
        self._db_path = db_path
        self._reload_snaps()

    def set_default_snap(self, scan_id: int) -> None:
        self._select_snap_id(scan_id)

    def set_pipeline_doc(self, doc: dict) -> None:
        self._doc = copy.deepcopy(doc) if doc else {}
        if self._auto.isChecked():
            self._debounce.start()
        else:
            self._status.setText(self.tr("Edits pending — press ↻ to refresh."))

    def request_refresh(self) -> None:
        self._run_now()

    # ── slots ─────────────────────────────────────────────────────
    def _reload_snaps(self) -> None:
        self._snap_combo.blockSignals(True)
        self._snap_combo.clear()
        if not self._db_path:
            self._snap_combo.addItem(self.tr("(no project)"), -1)
            self._snap_combo.blockSignals(False)
            return
        try:
            scans = _list_snaps(self._db_path)
        except Exception as e:
            self._snap_combo.addItem(self.tr("(db error: {err})").format(err=e), -1)
            self._snap_combo.blockSignals(False)
            return
        if not scans:
            self._snap_combo.addItem(self.tr("(no scans yet)"), -1)
        else:
            for sid, label in scans:
                self._snap_combo.addItem(label, sid)
        self._snap_combo.blockSignals(False)

    def _select_snap_id(self, scan_id: int) -> None:
        for i in range(self._snap_combo.count()):
            if self._snap_combo.itemData(i) == scan_id:
                self._snap_combo.setCurrentIndex(i)
                return

    def _on_snap_changed(self, _idx: int) -> None:
        sid = self._snap_combo.currentData()
        if not isinstance(sid, int) or sid <= 0:
            self._snap_id = None
            self._src = None
            self._canvas.set_image(None)
            return
        try:
            buf, dpi = _load_source(self._db_path, sid)
        except Exception as e:
            self._status.setText(self.tr("Failed to load scan: {err}").format(err=e))
            return
        self._snap_id = sid
        self._src = buf
        self._src_dpi = dpi
        # New scan → drop the previous pick. A stage chosen on scan N
        # has no meaningful counterpart on scan N+1.
        self._preferred_key = None
        self._frames = {"00_source": self._dsp(buf)}
        self._refresh_page_combo(["00_source"])
        # Show the raw source immediately so the user sees the scan they
        # picked even when the pipeline run is gated by autoupdate / will
        # take a while to complete.
        self._canvas.set_image(buf)
        if self._auto.isChecked():
            self._run_now()
        else:
            self._status.setText(self.tr("Scan loaded — press ↻ to run pipeline."))

    # Preview frames are display-only (canvas fit-to-widget + 240 px PiP), so
    # storing full-res 12 MP ndarrays per stage × branch is pure retention —
    # historically the top GUI memory path here. Downscale to ~1600 px on the
    # way into self._frames; the lineage buffers (replay) stay full-res.
    _PREVIEW_MAX = 1600

    @classmethod
    def _dsp(cls, arr):
        if arr is None:
            return None
        h, w = arr.shape[:2]
        longest = max(h, w)
        if longest <= cls._PREVIEW_MAX:
            return arr
        s = cls._PREVIEW_MAX / float(longest)
        return cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA)

    @classmethod
    def _dsp_frames(cls, frames: dict) -> dict:
        return {k: cls._dsp(v) for k, v in (frames or {}).items()}

    def _on_page_changed(self, _idx: int) -> None:
        key = self._layout_combo.currentData() or ""
        if key:
            # Remember the user pick so the next auto-refresh restores it.
            self._preferred_key = key
        img = self._frames.get(key)
        self._canvas.set_image(img if img is not None else None)
        self._refresh_branch_overlay()

    # ── per-stage branch overlay ─────────────────────────────────
    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        """Return `(step_prefix, branch_path)` for a frame key.

        Step prefix = everything before the ` · ` separator (or before
        ` ★` when no branch is present); branch_path = the part after
        ` · ` minus the optional trailing ` ★`. Used to group peer
        branches under the overlay toggle row."""
        k = key
        if k.endswith(" ★"):
            k = k[:-2]
        if " · " in k:
            stem, branch = k.split(" · ", 1)
            return stem, branch
        return k, ""

    def _peer_branches(self, current_key: str) -> list[tuple[str, str]]:
        """All frame keys that share the current key's step prefix.
        Returns `[(branch_path, full_key)]` sorted alphabetically by
        branch_path. The empty-branch case (single linear step) yields
        an empty list so no overlay is drawn."""
        stem, _ = self._split_key(current_key)
        peers: list[tuple[str, str]] = []
        for k in self._frames:
            s, b = self._split_key(k)
            if s == stem and b:
                peers.append((b, k))
        peers.sort(key=lambda t: (len(t[0]), t[0]))
        return peers

    def _refresh_branch_overlay(self) -> None:
        """Render the branch-selector pills at the top-left of the
        canvas. Selected pill is filled blue and disabled; siblings are
        outlined and clickable. Re-built from scratch on every layout
        change so the pill set always matches the current peer group."""
        # Clear any existing pills.
        existing = getattr(self, "_branch_pills", []) or []
        for b in existing:
            # hide() first — Qt re-shows a visible, implicitly-shown
            # widget after reparent (bare top-level window flash).
            b.hide()
            b.setParent(None)
            b.deleteLater()
        self._branch_pills = []

        current_key = self._layout_combo.currentData() or ""
        if not current_key:
            return
        peers = self._peer_branches(current_key)
        if not peers:
            return
        from aglaia.gui.widgets import make_icon_button
        # We need a text-only round button — reuse `make_icon_button`'s
        # styling pass for the round shape, then drop the icon and set
        # text. `text` flag bypasses the icon setter entirely.
        from PySide6.QtWidgets import QPushButton

        size = 26
        margin = 8
        gap = 6
        _cur_stem, cur_branch = self._split_key(current_key)
        for i, (branch, full_key) in enumerate(peers):
            btn = QPushButton(branch, self._canvas)
            btn.setFixedSize(size, size)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            radius = size // 2
            is_selected = (branch == cur_branch)
            if is_selected:
                btn.setEnabled(False)
                style = (
                    f"QPushButton {{ background-color: {COLOR_PRIMARY};"
                    f"  color: {COLOR_FONT_INVERSE}; font-weight: 700;"
                    f"  border: 1px solid {COLOR_OUTLINE_STRONG};"
                    f"  border-radius: {radius}px; }}"
                    f"QPushButton:disabled {{ color: {COLOR_FONT_INVERSE}; }}"
                )
            else:
                style = (
                    f"QPushButton {{ background-color: {COLOR_SCRIM_MEDIUM};"
                    f"  color: {COLOR_FONT_SECONDARY};"
                    f"  border: 1px solid {COLOR_OUTLINE};"
                    f"  border-radius: {radius}px; }}"
                    f"QPushButton:hover {{ background-color: {COLOR_PRIMARY_BORDER};"
                    f"  color: {COLOR_FONT_INVERSE}; }}"
                )
                btn.clicked.connect(
                    lambda _=False, k=full_key: self._select_page_key(k)
                )
            btn.setStyleSheet(style)
            btn.move(margin + i * (size + gap), margin)
            btn.show()
            self._branch_pills.append(btn)

    def _select_page_key(self, key: str) -> None:
        """Set the Stage combo to a specific frame key. Triggers
        `_on_page_changed` via `currentIndexChanged`, which also
        refreshes the overlay so the pill highlight follows the swap."""
        for i in range(self._layout_combo.count()):
            if self._layout_combo.itemData(i) == key:
                self._layout_combo.setCurrentIndex(i)
                return

    def _on_auto_toggled(self, on: bool) -> None:
        if on:
            self._debounce.start()
        else:
            self._debounce.stop()

    # ── runner ────────────────────────────────────────────────────
    def _run_now(self) -> None:
        self._debounce.stop()
        if self._src is None:
            self._status.setText(self.tr("Pick a scan first."))
            return
        if not self._doc.get("pipeline"):
            self._frames = {"00_source": self._dsp(self._src)}
            self._refresh_page_combo(["00_source"])
            self._canvas.set_image(self._src)
            self._status.setText(self.tr("Pipeline is empty — showing source."))
            return
        if self._is_worker_alive():
            # Mark a re-run, let the current one finish (cancel flag stops the next step).
            self._worker.cancel()
            self._pending = True
            return
        self._spawn_worker()

    def _is_worker_alive(self) -> bool:
        """True only if the Python ref points at a still-valid C++ QThread.
        Calling `isRunning()` on a deleteLater-freed object raises
        `RuntimeError: Internal C++ object already deleted`."""
        if self._worker is None:
            return False
        try:
            return bool(self._worker.isRunning())
        except RuntimeError:
            self._worker = None
            return False

    def _spawn_worker(self) -> None:
        self._pending = False
        self._status.setText(self.tr("Running pipeline…"))
        self._worker = _PreviewWorker(self._src, self._src_dpi, self._doc, self)
        self._worker.done.connect(self._on_worker_done)
        # No `finished → deleteLater`: the queued delete leaves a Python
        # ref to a freed C++ object, and the next `_run_now` blows up on
        # `isRunning()`. Letting the panel hold the ref until the next
        # spawn — Python GC then reaps it cleanly.
        self._worker.start()

    def _on_worker_done(self, frames: dict, error: str) -> None:
        if error:
            self._status.setText(self.tr("Error: {err}").format(err=error))
        else:
            self._frames = self._dsp_frames(frames)
            # Stages in pipeline order, branches grouped together. The
            # leading 2-digit step prefix already gives natural sort.
            keys = sorted(frames.keys())
            self._refresh_page_combo(keys)
            n_leaves = sum(1 for k in keys if k.endswith(" ★"))
            if n_leaves == 1:
                self._status.setText(self.tr(
                    "OK — {n_stages} stage(s), 1 final leaf."
                ).format(n_stages=len(keys)))
            else:
                self._status.setText(self.tr(
                    "OK — {n_stages} stage(s), {n_leaves} final leaves."
                ).format(n_stages=len(keys), n_leaves=n_leaves))
        if self._pending:
            # A new edit came in mid-run; chain another pass.
            QTimer.singleShot(0, self._spawn_worker)

    # ── teardown ─────────────────────────────────────────────────
    def closeEvent(self, ev) -> None:  # noqa: N802
        """If a worker is still grinding when the host dialog closes,
        signal it to stop and wait briefly so we don't free its source
        ndarray out from under it."""
        if self._is_worker_alive():
            self._worker.cancel()
            self._worker.wait(2000)
        # Free the retained stage previews + source when the editor closes.
        self._frames = {}
        self._src = None
        super().closeEvent(ev)

    def _refresh_page_combo(self, keys: list[str]) -> None:
        self._layout_combo.blockSignals(True)
        self._layout_combo.clear()
        if not keys:
            self._layout_combo.addItem("—", "")
            self._layout_combo.setEnabled(False)
            self._layout_combo.blockSignals(False)
            return
        for k in keys:
            self._layout_combo.addItem(k, k)
        self._layout_combo.setEnabled(True)
        # Default to the last final-leaf entry (★) when present, else the
        # very last stage. Users almost always want the pipeline output,
        # not the source.
        default_idx = len(keys) - 1
        for i in range(len(keys) - 1, -1, -1):
            if keys[i].endswith(" ★"):
                default_idx = i
                break
        # When the user has explicitly picked a stage, try to keep them
        # on it (or its closest equivalent) across auto-refreshes.
        match_idx = self._match_preferred_index(keys)
        if match_idx is not None:
            default_idx = match_idx
        self._layout_combo.setCurrentIndex(default_idx)
        self._layout_combo.blockSignals(False)
        img = self._frames.get(keys[default_idx])
        if img is not None:
            self._canvas.set_image(img)
        # The combo update was made under blockSignals, so
        # `_on_page_changed` didn't fire. Push the overlay refresh
        # manually so the A/B/C pills appear right after a re-run.
        self._refresh_branch_overlay()

    def _match_preferred_index(self, keys: list[str]) -> Optional[int]:
        """Resolve `_preferred_key` against the new key list.

        Match order:
            1. exact key (same prefix + name + branch + ★)
            2. same step_name (no NN_ prefix) + same branch_path
            3. same step_name only
            4. same branch_path on any final-leaf (★)
        Returns the matching index or None when no preference is set
        / nothing matches. The caller falls back to its default-leaf
        heuristic in that case (which also covers the "settle on last
        one" rule via the existing default_idx walk).
        """
        pref = self._preferred_key
        if not pref or not keys:
            return None
        if pref in keys:
            return keys.index(pref)

        def _strip_prefix(stem: str) -> str:
            # "08_pages_dewarp" → "pages_dewarp"
            if len(stem) >= 3 and stem[2] == "_" and stem[:2].isdigit():
                return stem[3:]
            return stem

        pref_stem, pref_branch = self._split_key(pref)
        pref_name = _strip_prefix(pref_stem)

        # Pass 2: same name + same branch.
        for i, k in enumerate(keys):
            s, b = self._split_key(k)
            if _strip_prefix(s) == pref_name and b == pref_branch:
                return i
        # Pass 3: same name only.
        for i, k in enumerate(keys):
            s, _b = self._split_key(k)
            if _strip_prefix(s) == pref_name:
                return i
        # Pass 4: same branch on any final leaf.
        if pref_branch:
            for i in range(len(keys) - 1, -1, -1):
                k = keys[i]
                if not k.endswith(" ★"):
                    continue
                _s, b = self._split_key(k)
                if b == pref_branch:
                    return i
        return None
