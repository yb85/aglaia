# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Beginner-friendly capture-mode picker for the new-project page.

Layout — three columns:
  ┌────────────┬──────────────┬──────────────────────────────┐
  │ mode list  │  big icon    │  pipeline steps outline       │
  │ [card]     │  "good for…" │  01 step … (expand on click)  │
  │ [card]     │   blurb      │  02 step …                    │
  │ [card]     │              │  03 step …                    │
  │ [+ New]    │              │                               │
  │ (scrolls)  │              │                               │
  ├────────────┴──────────────┴──────────────────────────────┤
  │ [ Properties… ]                                           │
  └───────────────────────────────────────────────────────────┘

List cards = the curated modes (aglaia/app_data/modes.py) + any user-created
pipelines (yamls in APP_DATA/pipelines that are neither a mode nor a shipped
extra) + a trailing "+ New" card. Bundled non-mode variants stay reachable
through the Properties editor. Emits `pipelineChanged(yaml_text, path)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from aglaia.app_data import bundled_pipelines_dir, pipelines_dir
from aglaia.app_data.modes import PipelineMode, mode_for_pipeline, modes
from aglaia.gui.colors import (
    COLOR_BG_HINT,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_FAINT,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_PRIMARY_BORDER,
    COLOR_PRIMARY_BORDER_STRONG,
)
from aglaia.gui.PipelineEditorWidget import (
    PipelineEditorDialog, load_yaml_text, parse_yaml,
)
from aglaia.gui.PipelineStepsOutline import PipelineStepsOutline

_NEW_KEY = "__new__"


class _ModeCard(QFrame):
    """Compact selectable card: icon + name, side by side."""

    clicked = Signal(str)

    def __init__(self, key: str, name: str, icon_path: Optional[Path],
                 *, lucide: Optional[str] = None, lucide_size: int = 20,
                 accent: bool = False, parent=None):
        super().__init__(parent)
        self._key = key
        self._selected = False
        self._accent = accent  # primary-tinted action card (the "+ New" tile)
        self.setObjectName("ModeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # click-select; no focus ring
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 7, 12, 7)
        h.setSpacing(8)

        icon_color = COLOR_PRIMARY if accent else COLOR_FONT_MUTED
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(26, 26)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet("background: transparent; border: none;")
        if icon_path is not None:
            icon_lbl.setPixmap(QIcon(str(icon_path)).pixmap(26, 26))
        elif lucide is not None:
            from aglaia.gui.theme import lucide_pixmap
            icon_lbl.setPixmap(lucide_pixmap(lucide, color=icon_color, size=lucide_size))
        h.addWidget(icon_lbl)

        name_lbl = QLabel(name)
        name_lbl.setWordWrap(True)
        name_color = COLOR_PRIMARY if accent else COLOR_FONT_PRIMARY
        name_lbl.setStyleSheet(
            f"color: {name_color}; font-size: 12px; font-weight: 600;"
            "background: transparent; border: none;")
        h.addWidget(name_lbl, 1)  # fill width in the vertical list
        self._apply_style()

    def key(self) -> str:
        return self._key

    def set_selected(self, on: bool) -> None:
        if on != self._selected:
            self._selected = on
            self._apply_style()

    def _apply_style(self) -> None:
        if self._accent:
            border, bg, bw = COLOR_PRIMARY_BORDER_STRONG, COLOR_PRIMARY_BG_SOFT, "1px"
        elif self._selected:
            border, bg = COLOR_PRIMARY, COLOR_PRIMARY_BG_SOFT
            bw = "1.5px"
        else:
            border, bg = COLOR_OUTLINE_FAINT, COLOR_BG_OVERLAY_SOFT
            bw = "1px"
        self.setStyleSheet(
            "QFrame#ModeCard {"
            f"  background: {bg};"
            f"  border: {bw} solid {border};"
            "  border-radius: 10px;"
            "}"
            "QFrame#ModeCard:hover { border-color: " + COLOR_PRIMARY + "; }"
        )

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        self.clicked.emit(self._key)
        super().mousePressEvent(event)


class ModePickerPanel(QWidget):
    """Mode strip + selected-mode detail + steps outline + Properties."""

    pipelineChanged = Signal(str, object)  # (yaml_text, Path|None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, _ModeCard] = {}
        self._current_key: Optional[str] = None
        self._yaml: str = ""
        self._path: Optional[Path] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # Three columns: [mode list] [icon + blurb] [steps outline].
        body = QHBoxLayout()
        body.setSpacing(14)

        # ── col 1: vertical mode list (scrolls when it overflows) ──────
        self._list_scroll = QScrollArea()
        self._list_scroll.setWidgetResizable(True)
        self._list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_scroll.setFixedWidth(210)
        self._list_host = QWidget()
        self._list = QVBoxLayout(self._list_host)
        self._list.setContentsMargins(2, 2, 2, 2)
        self._list.setSpacing(8)
        self._list.addStretch(1)
        self._list_scroll.setWidget(self._list_host)
        body.addWidget(self._list_scroll)

        # Grouped panel: the icon+blurb (col 2) and steps (col 3) describe
        # the SAME selected pipeline, so they share a subtly raised surface
        # (COLOR_BG_HINT) to read as one unit, distinct from the mode list.
        group = QFrame()
        group.setObjectName("PipelineGroup")
        group.setStyleSheet(
            "QFrame#PipelineGroup {"
            f"  background: {COLOR_BG_HINT};"
            f"  border: 1px solid {COLOR_OUTLINE_SUBTLE};"
            "  border-radius: 12px;"
            "}"
            "QFrame#PipelineGroup QLabel { background: transparent; }"
        )
        g = QHBoxLayout(group)
        g.setContentsMargins(14, 14, 14, 14)
        g.setSpacing(14)

        # ── col 2: big mode icon + "good for…" blurb + Properties ──────
        left = QVBoxLayout()
        left.setSpacing(10)
        self._big_icon = QLabel()
        self._big_icon.setFixedSize(96, 96)
        self._big_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._big_icon.setStyleSheet("background: transparent;")
        left.addWidget(self._big_icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._blurb = QLabel()
        self._blurb.setWordWrap(True)
        self._blurb.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 12px;")
        left.addWidget(self._blurb)
        # Properties edits THIS pipeline → sits with its description.
        self._props_btn = QPushButton(self.tr("Properties…"))
        self._props_btn.setToolTip(self.tr("Open the pipeline editor"))
        self._props_btn.clicked.connect(self._open_properties)
        left.addWidget(self._props_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(180)
        g.addWidget(left_w)

        # ── col 3: pipeline steps outline ──────────────────────────────
        self._steps = PipelineStepsOutline()
        g.addWidget(self._steps, 1)

        body.addWidget(group, 1)
        root.addLayout(body, 1)

        self._rebuild_strip()

    # ── public ─────────────────────────────────────────────────────────
    def current_yaml(self) -> str:
        return self._yaml

    def current_path(self) -> Optional[Path]:
        return self._path

    # ── strip construction ─────────────────────────────────────────────
    def _rebuild_strip(self, select_key: Optional[str] = None) -> None:
        # remember selection
        select_key = select_key or self._current_key
        while self._list.count() > 1:  # keep trailing stretch
            item = self._list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards.clear()

        mode_pipelines = {m.pipeline for m in modes()}
        for m in modes():
            self._add_card(m.key, m.name, m.icon_path(),
                           pipelines_dir() / m.pipeline)

        # user-created pipelines: in APP_DATA but neither a mode nor a
        # shipped extra (those stay reachable via the Properties editor).
        bundled = {p.name for p in bundled_pipelines_dir().glob("*.yaml")}
        for p in sorted(pipelines_dir().glob("*.yaml")):
            if p.name in mode_pipelines or p.name in bundled:
                continue
            try:
                nm = parse_yaml(p.read_text(encoding="utf-8")).get("name") or p.stem
            except OSError:
                nm = p.stem
            self._add_card(f"file:{p.name}", str(nm), None, p, lucide="layers")

        # trailing "+ New" card
        new_card = _ModeCard(_NEW_KEY, self.tr("New"), None, lucide="plus",
                             lucide_size=15, accent=True)
        new_card.clicked.connect(lambda *_: self._on_new())
        self._cards[_NEW_KEY] = new_card
        self._list.insertWidget(self._list.count() - 1, new_card)

        # (re)select
        keys = [k for k in self._cards if k != _NEW_KEY]
        if select_key not in keys:
            select_key = keys[0] if keys else None
        if select_key is not None:
            self._select(select_key)

    def _add_card(self, key: str, name: str, icon_path: Optional[Path],
                  path: Path, *, lucide: Optional[str] = None) -> None:
        card = _ModeCard(key, name, icon_path, lucide=lucide)
        card.clicked.connect(self._select)
        card._path = path  # type: ignore[attr-defined]
        self._cards[key] = card
        self._list.insertWidget(self._list.count() - 1, card)

    # ── selection ──────────────────────────────────────────────────────
    def _select(self, key: str) -> None:
        if key == _NEW_KEY or key not in self._cards:
            return
        self._current_key = key
        for k, c in self._cards.items():
            c.set_selected(k == key)
        path: Path = getattr(self._cards[key], "_path")
        try:
            self._yaml = load_yaml_text(path)
            self._path = path
        except OSError:
            return
        self._update_detail(key, path)
        self.pipelineChanged.emit(self._yaml, path)

    def _update_detail(self, key: str, path: Path) -> None:
        mode: Optional[PipelineMode] = mode_for_pipeline(path.name)
        # big icon
        icon_path = mode.icon_path() if mode else None
        if icon_path is not None:
            self._big_icon.setPixmap(QIcon(str(icon_path)).pixmap(96, 96))
        else:
            from aglaia.gui.theme import lucide_pixmap
            self._big_icon.setPixmap(lucide_pixmap("layers", color=COLOR_FONT_MUTED, size=72))
        # blurb
        if mode is not None:
            self._blurb.setText(mode.description)
        else:
            try:
                nm = parse_yaml(self._yaml).get("name") or path.stem
            except Exception:
                nm = path.stem
            self._blurb.setText(self.tr("Custom pipeline “{name}”.").format(name=nm))
        # steps
        try:
            self._steps.set_pipeline(parse_yaml(self._yaml))
        except Exception:
            self._steps.set_pipeline({})

    # ── editor hooks ────────────────────────────────────────────────────
    def _open_properties(self) -> None:
        dlg = PipelineEditorDialog(
            initial_yaml=self._yaml, parent=self,
            allow_reprocess=False, title=self.tr("Edit pipeline"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._yaml = dlg.yaml_text()
            name = (parse_yaml(self._yaml).get("name") or "").strip()
            # A Save-as creates a new yaml in PIPELINES_DIR — rebuild + try
            # to reselect it; otherwise just refresh the current detail.
            self._rebuild_strip()
            for k, c in self._cards.items():
                p = getattr(c, "_path", None)
                if p is not None and name and parse_yaml_name(p) == name:
                    self._select(k)
                    break
            else:
                if self._current_key:
                    self._update_detail(self._current_key, self._path or Path())
                self.pipelineChanged.emit(self._yaml, self._path)

    def _on_new(self) -> None:
        dlg = PipelineEditorDialog(
            initial_yaml=self._yaml, parent=self,
            allow_reprocess=False, title=self.tr("New pipeline"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_yaml = dlg.yaml_text()
            name = (parse_yaml(new_yaml).get("name") or "").strip()
            self._rebuild_strip()
            for k, c in self._cards.items():
                p = getattr(c, "_path", None)
                if p is not None and name and parse_yaml_name(p) == name:
                    self._select(k)
                    break


def parse_yaml_name(path: Path) -> str:
    try:
        return (parse_yaml(path.read_text(encoding="utf-8")).get("name") or path.stem).strip()
    except OSError:
        return path.stem
