# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""SidebarPanel — ActivityBar + QStackedWidget content pane.

Hosts the ActivityBar on the right and the per-tab content widgets
on the left of the strip (so the panel reads left-to-right as
*content → icons*, with the icons clipped to the window edge — the
VS Code idiom flipped horizontally since Aglaïa's bar lives on the
right side of the main window).

The panel owns:
  * one stacked content widget per registered activity,
  * the collapsed/expanded state,
  * the currently-active activity key.

MainWindow constructs the per-tab widgets (CaptureTab, ImportTab,
PipelineTab, OcrTab, ExportTab), registers them via
``add_tab(name, widget)``, then calls ``set_active(name)`` to pick
the startup tab. Persistence (KEY_SIDEBAR_TAB / KEY_SIDEBAR_COLLAPSED)
is wired by MainWindow — the panel just emits ``state_changed`` on
every change so the host can debounce-write.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from .ActivityBar import ActivityBar, BAR_WIDTH


CONTENT_W = 360  # design-doc value


class SidebarPanel(QWidget):
    """Right-side panel with an icon strip + a content stack.

    Signals:
      * ``tab_changed(name)`` — emitted when the visible tab swaps.
      * ``collapse_changed(collapsed: bool)`` — content pane hidden /
        shown via toggle or active-icon click.
      * ``state_changed`` — fires whenever the persistent state should
        be written (tab or collapsed delta).
    """

    tab_changed = Signal(str)
    collapse_changed = Signal(bool)
    state_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tabs: dict[str, QWidget] = {}

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Each tab is wrapped in its own QScrollArea inside ``add_tab``
        # so the scrollbar only appears for tabs whose content actually
        # exceeds the viewport — a single outer QScrollArea around the
        # QStackedWidget would key off the tallest tab and force a
        # scrollbar on every tab.
        self.stack = QStackedWidget()
        self.stack.setMinimumWidth(CONTENT_W)
        self.stack.setMaximumWidth(CONTENT_W)
        outer.addWidget(self.stack, 1)

        self.bar = ActivityBar(self)
        outer.addWidget(self.bar)

        self.bar.activated.connect(self._on_activated)
        self.bar.collapse_toggled.connect(self._on_collapse)

    # ── tab registration ───────────────────────────────────────────

    def add_tab(self, name: str, widget: QWidget, *,
                icon_name: str, tooltip: str, scrollable: bool = True) -> None:
        if name in self._tabs:
            return
        # By default wrap each tab in its own scroll area so the scrollbar
        # shows only when *that* tab's content overflows. `scrollable=False`
        # adds the tab directly so it fills the viewport and owns its own
        # internal scrolling (e.g. PipelineTab scrolls just its steps list,
        # keeping the Backends footer pinned — no whole-tab scrollbar).
        widget.setAutoFillBackground(False)
        if scrollable:
            container: QWidget = QScrollArea()
            container.setWidgetResizable(True)
            container.setFrameShape(QScrollArea.Shape.NoFrame)
            container.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            container.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Inner viewport also needs the transparent rule — QScrollArea's
            # selector alone doesn't propagate to it, leaving a faint
            # palette-Base shade visible against the surrounding window.
            container.setStyleSheet(
                "QScrollArea, QScrollArea > QWidget > QWidget "
                "{ background: transparent; }"
            )
            container.viewport().setAutoFillBackground(False)
            container.setWidget(widget)
        else:
            container = widget
        self._tabs[name] = container
        self.stack.addWidget(container)
        self.bar.add_activity(name, icon_name, tooltip)

    def add_bottom_action(self, name: str, icon_name: str, tooltip: str,
                          on_click) -> None:
        """Settings (icon-only) lives at the bottom and doesn't swap
        the stack — it invokes ``on_click``."""
        self.bar.add_activity(name, icon_name, tooltip,
                              bottom=True, on_click=on_click)

    def add_tip_button(self, on_click) -> None:
        """Rosy-red heart + 'Tip' label — distinct visual style from
        the icon-only bottom actions."""
        self.bar.add_tip_button(on_click)

    # ── selection / collapse ───────────────────────────────────────

    def set_active(self, name: Optional[str]) -> None:
        if name is None:
            return
        w = self._tabs.get(name)
        if w is None:
            return
        self.stack.setCurrentWidget(w)
        self.bar.set_active(name)
        if self.bar.collapsed():
            self.set_collapsed(False)

    def active(self) -> Optional[str]:
        return self.bar.active()

    def set_collapsed(self, collapsed: bool) -> None:
        if self.bar.collapsed() == collapsed:
            self.stack.setVisible(not collapsed)
            return
        self.bar.set_collapsed(collapsed)
        self.stack.setVisible(not collapsed)
        self.collapse_changed.emit(collapsed)
        self.state_changed.emit()

    def collapsed(self) -> bool:
        return self.bar.collapsed()

    # ── internals ──────────────────────────────────────────────────

    def _on_activated(self, name: str) -> None:
        w = self._tabs.get(name)
        if w is None:
            return
        self.stack.setCurrentWidget(w)
        self.stack.setVisible(True)
        self.tab_changed.emit(name)
        self.state_changed.emit()

    def _on_collapse(self, collapsed: bool) -> None:
        self.stack.setVisible(not collapsed)
        self.collapse_changed.emit(collapsed)
        self.state_changed.emit()
