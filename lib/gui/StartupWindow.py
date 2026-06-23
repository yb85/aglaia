# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""
Startup window — first thing the user sees.

Two-page wizard:

1. **Recent projects grid** — large cards for the most-recently-opened
   projects. The last card is accented "New project". Clicking a recent
   card opens that project; clicking New goes to page 2.

2. **New project** — name + parent folder on top, two radio cards
   (Capture / Files), then a pipeline selector with a Properties button
   that opens the full editor in a modal. The mode field of the returned
   `StartupChoice` is inferred from the radio choice + selected files
   (pdf vs image).

The last-used parent folder is remembered across sessions via QSettings.
Returns a `StartupChoice` describing what should happen next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from lib.gui.CameraEnum import list_cameras
from lib.gui.PipelineEditorWidget import (
    DEFAULT_PIPELINE_PATH, PIPELINES_DIR, PipelineEditorDialog,
    load_yaml_text, parse_yaml,
)
from lib.gui.sidebar.widgets.RadioCardGroup import RadioCardGroup
from lib.gui.ModePicker import ModePickerPanel
from lib.gui.colors import (
    COLOR_BG_OVERLAY_SOFT,
    COLOR_BG_SURFACE,
    COLOR_BG_SURFACE_ALT,
    COLOR_ERROR,
    COLOR_ERROR_STRONG,
    COLOR_FONT_INVERSE,
    COLOR_FONT_LINK_HOVER,
    COLOR_FONT_LINK,
    COLOR_FONT_MUTED,
    COLOR_FONT_PLACEHOLDER,
    COLOR_FONT_PRIMARY,
    COLOR_FONT_SECTION_LABEL,
    COLOR_OUTLINE_FAINT,
    COLOR_OUTLINE_GHOST,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_PRIMARY_BORDER,
    COLOR_PRIMARY_BORDER_STRONG,
    active_palette_name,
)


SETTINGS_ORG = "aglaia"
SETTINGS_APP = "StartupWindow"

TIPPING_URL = "https://ko-fi.com/yb_85"
HOMEPAGE_URL = "https://aglaia.bibli.cc"
GIT_REPO = "https://github.com/yb85/aglaia"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
_PDF_EXTS = {".pdf"}


@dataclass
class StartupChoice:
    """What the user wants to do once the startup wizard closes."""

    mode: str  # "open" | "pdf" | "images" | "capture"
    project_dir: Optional[Path] = None
    project_name: Optional[str] = None
    project_slug: Optional[str] = None
    parent_dir: Optional[Path] = None
    input_files: list[Path] = field(default_factory=list)
    camera_index: Optional[int] = None
    pipeline_yaml: str = ""
    # File imports: assumed DPI for images / render DPI for PDFs.
    input_dpi: Optional[float] = None


# ── recent / new card ──────────────────────────────────────────────────


class _RecentCard(QFrame):
    """Large clickable project card. Click → `clicked` signal.

    Renders project name (bold) + parent folder path (muted) + small
    last-modified hint. `accent=True` paints it as the "New project"
    call-to-action: primary border + bigger plus icon.
    """

    clicked = Signal()
    remove_requested = Signal()

    CARD_W = 240
    CARD_H = 140

    def __init__(self, *, title: str, subtitle: str = "",
                 hint: str = "", icon_name: str = "folder-open",
                 accent: bool = False, missing: bool = False,
                 removable: bool = False, scan_count: int | None = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("RecentCard")
        # Belt + suspenders: setFixedSize alone is sometimes overridden by
        # parent layout (QGridLayout in a Resizable QScrollArea host); pin
        # both axes to Fixed so the card never inherits its cell's width.
        self.setFixedSize(self.CARD_W, self.CARD_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Qt 6's `enterEvent` / `leaveEvent` only fire on the widget the
        # mouse actually lands on. With child QLabels filling the card,
        # the labels eat the enter and the QFrame never hears about it.
        # `WA_Hover` makes Qt synthesise `HoverEnter` / `HoverLeave` for
        # the boundary of *this* widget regardless of which descendant
        # is under the cursor — that's the signal we route into the
        # X-button visibility toggle (see `event()` override).
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._accent = accent
        self._missing = missing
        self._hover = False
        if missing:
            # Don't `setEnabled(False)` — Qt cascades that to all child
            # widgets and the X (remove-from-recents) button would also
            # go dead. Use `_missing` as a soft gate inside the click
            # handler instead; visual dim is the opacity effect in
            # `_refresh`.
            self.setCursor(Qt.CursorShape.ArrowCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(6)

        # Icon row — main icon, plus a scan-count badge for recent projects
        # (cached in recent_projects.scan_count so we never reopen the DB).
        from lib.gui.theme import lucide_pixmap
        icon_color = COLOR_PRIMARY if accent else COLOR_FONT_MUTED
        icon_lbl = QLabel()
        icon_lbl.setPixmap(lucide_pixmap(icon_name, color=icon_color, size=28))
        icon_lbl.setStyleSheet("background: transparent;")
        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_row.setSpacing(6)
        icon_row.addWidget(icon_lbl)
        if scan_count is not None and not accent and not missing:
            cnt_lbl = QLabel(
                self.tr("1 scan") if scan_count == 1
                else self.tr("{n} scans").format(n=scan_count))
            cnt_lbl.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: 11px; "
                "font-weight: 600; background: transparent;")
            icon_row.addWidget(cnt_lbl)
        icon_row.addStretch(1)
        v.addLayout(icon_row)

        v.addStretch(1)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-size: 14px; font-weight: 700;"
            " background: transparent;"
        )
        title_lbl.setWordWrap(False)
        # Truncate long titles so they don't overflow.
        fm = title_lbl.fontMetrics()
        elided = fm.elidedText(title, Qt.TextElideMode.ElideRight, self.CARD_W - 32)
        title_lbl.setText(elided)
        v.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: 11px;"
                " background: transparent;"
            )
            sub_elided = sub_lbl.fontMetrics().elidedText(
                subtitle, Qt.TextElideMode.ElideMiddle, self.CARD_W - 32,
            )
            sub_lbl.setText(sub_elided)
            v.addWidget(sub_lbl)

        if missing:
            err_lbl = QLabel(self.tr("Missing project file"))
            err_lbl.setStyleSheet(
                f"color: {COLOR_ERROR}; font-size: 11px; font-weight: 600;"
                " background: transparent;"
            )
            v.addWidget(err_lbl)
        elif hint:
            hint_lbl = QLabel(hint)
            hint_lbl.setStyleSheet(
                f"color: {COLOR_FONT_PLACEHOLDER}; font-size: 10px;"
                " background: transparent;"
            )
            v.addWidget(hint_lbl)

        # Subtle "forget" X in the top-right corner. Only on recent
        # cards (`removable=True`) — the accent Open/New tiles aren't
        # removable. Positioned via absolute geometry so it floats over
        # the card content (no layout reflow when shown / hidden).
        self._remove_btn: Optional[QPushButton] = None
        if removable:
            from lib.gui.theme import lucide as _lucide
            btn = QPushButton(self)
            btn.setFlat(True)
            # Lucide SVGs set `stroke=color` literally — rgba(...) gets
            # rejected by Qt's SVG renderer (badge tokens hit the same
            # bug). Use a solid hex (`COLOR_FONT_PLACEHOLDER` = #71717a
            # on light, similar mid-grey on dark).
            btn.setIcon(_lucide("x", color=COLOR_FONT_PLACEHOLDER, size=14))
            btn.setIconSize(QSize(14, 14))
            btn.setFixedSize(22, 22)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(self.tr("Remove from recent"))
            btn.setStyleSheet(
                "QPushButton {"
                "  background: transparent;"
                "  border: none;"
                "  border-radius: 11px;"
                "}"
                f"QPushButton:hover {{"
                f"  background: {COLOR_PRIMARY_BG_SOFT};"
                f"}}"
            )
            btn.clicked.connect(self._on_remove_clicked)
            btn.move(self.CARD_W - 22 - 6, 6)
            btn.raise_()
            btn.hide()  # only on hover
            self._remove_btn = btn

        self._refresh()

    def setEnabled(self, on: bool) -> None:  # noqa: N802 — Qt API
        super().setEnabled(on)
        if on:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._refresh()

    def _refresh(self) -> None:
        if self._accent:
            bg = COLOR_PRIMARY_BG_SOFT
            border = COLOR_PRIMARY_BORDER_STRONG
            if self._hover:
                bg = COLOR_BG_SURFACE_ALT
                border = COLOR_PRIMARY
        else:
            bg = COLOR_BG_SURFACE if self._hover else COLOR_BG_OVERLAY_SOFT
            border = COLOR_PRIMARY_BORDER if self._hover else COLOR_OUTLINE_FAINT
        if self._missing:
            border = COLOR_OUTLINE_GHOST
        self.setStyleSheet(
            f"#RecentCard {{ background: {bg};"
            f" border: 1.5px solid {border}; border-radius: 12px; }}"
            "#RecentCard QLabel { background: transparent; }"
        )
        # Belt + suspenders dimming. setEnabled(False) alone leaves the
        # card looking essentially identical to an active one on light
        # theme (palette barely shifts text colour); a 0.45 opacity
        # effect makes the unusable state read at a glance.
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        if self._missing:
            if not isinstance(self.graphicsEffect(), QGraphicsOpacityEffect):
                eff = QGraphicsOpacityEffect(self)
                eff.setOpacity(0.45)
                self.setGraphicsEffect(eff)
        else:
            if self.graphicsEffect() is not None:
                self.setGraphicsEffect(None)

    def event(self, ev) -> bool:  # noqa: N802 — Qt API
        # WA_Hover-synthesised enter/leave for the QFrame boundary even
        # when child labels are under the cursor (Qt 6's enterEvent only
        # fires on the topmost widget). Used to swap the hover bg/border
        # via `_refresh` AND to toggle the remove-X button's visibility.
        t = ev.type()
        if t == QEvent.Type.HoverEnter:
            self._hover = True
            if self._remove_btn is not None:
                self._remove_btn.show()
                self._remove_btn.raise_()
            self._refresh()
        elif t == QEvent.Type.HoverLeave:
            self._hover = False
            if self._remove_btn is not None:
                self._remove_btn.hide()
            self._refresh()
        return super().event(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        # Forward only when the cursor is over the card AND not over the
        # remove-button hit box AND the project isn't missing on disk.
        # Without the X-rect check, clicking the X also opens the project
        # the user just removed. `_missing` gates the open path while
        # leaving the X button live (we don't `setEnabled(False)` because
        # that cascades to all children).
        pos = ev.position().toPoint()
        if (ev.button() == Qt.MouseButton.LeftButton
                and not self._missing
                and self.isEnabled()
                and self.rect().contains(pos)
                and not (self._remove_btn is not None
                         and self._remove_btn.geometry().contains(pos))):
            self.clicked.emit()
        super().mouseReleaseEvent(ev)

    def _on_remove_clicked(self) -> None:
        self.remove_requested.emit()


# ── files drop zone ────────────────────────────────────────────────────


class _DropZone(QFrame):
    """Click-or-drop file picker. Mixed PDF + image lists are rejected;
    `files_changed` carries the canonical extension family ("pdf" /
    "images") so callers can route into the existing import paths.

    Shows the file count only — never the file list (per UX brief). Empty
    state shows the prompt text. Drag-over state highlights the border.
    """

    files_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(72)
        self._files: list[Path] = []
        self._kind: Optional[str] = None  # "pdf" | "images"
        self._drag_over = False

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 11, 20, 11)
        v.setSpacing(3)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        from lib.gui.theme import lucide_pixmap
        self._icon_lbl = QLabel()
        self._icon_lbl.setPixmap(lucide_pixmap(
            "upload", color=COLOR_FONT_MUTED, size=22))
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet("background: transparent;")
        v.addWidget(self._icon_lbl)

        self._headline = QLabel(self.tr("Click to choose, or drop PDFs / images here"))
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._headline.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 600; font-size: 13px;"
            " background: transparent;"
        )
        v.addWidget(self._headline)

        self._sub = QLabel(self.tr("PDF (rendered per page) or JPG / PNG / TIFF"))
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px;"
            " background: transparent;"
        )
        v.addWidget(self._sub)

        self._refresh_style()

    def files(self) -> list[Path]:
        return list(self._files)

    def kind(self) -> Optional[str]:
        return self._kind

    def clear(self) -> None:
        self._files = []
        self._kind = None
        self._headline.setText(self.tr("Click to choose, or drop PDFs / images here"))
        self._sub.setText(self.tr("PDF (rendered per page) or JPG / PNG / TIFF"))
        self.files_changed.emit()

    # ── styling ────────────────────────────────────────────────────────
    def _refresh_style(self) -> None:
        border = COLOR_PRIMARY if self._drag_over else COLOR_OUTLINE_FAINT
        style = "dashed" if self._drag_over else "dashed"
        bg = COLOR_PRIMARY_BG_SOFT if self._drag_over else COLOR_BG_OVERLAY_SOFT
        self.setStyleSheet(
            f"#DropZone {{ background: {bg};"
            f" border: 2px {style} {border}; border-radius: 12px; }}"
            "#DropZone QLabel { background: transparent; }"
        )

    # ── click → file dialog ────────────────────────────────────────────
    def mouseReleaseEvent(self, ev):  # noqa: N802
        if (ev.button() == Qt.MouseButton.LeftButton
                and self.rect().contains(ev.position().toPoint())):
            from PySide6.QtCore import QStandardPaths
            start = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.HomeLocation
            )
            paths, _ = QFileDialog.getOpenFileNames(
                self, self.tr("Select PDFs or images"), str(start),
                self.tr("PDF or images (*.pdf *.jpg *.jpeg *.png *.tif *.tiff)"),
            )
            if paths:
                self._set_files([Path(p) for p in paths])
        super().mouseReleaseEvent(ev)

    # ── drag-drop ──────────────────────────────────────────────────────
    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self._drag_over = True
            self._refresh_style()

    def dragLeaveEvent(self, ev):  # noqa: N802
        self._drag_over = False
        self._refresh_style()
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev):  # noqa: N802
        if not ev.mimeData().hasUrls():
            return
        ev.acceptProposedAction()
        self._drag_over = False
        self._refresh_style()
        paths: list[Path] = []
        for u in ev.mimeData().urls():
            if not u.isLocalFile():
                continue
            p = Path(u.toLocalFile())
            if p.is_file():
                paths.append(p)
        if paths:
            self._set_files(paths)

    # ── ingest selected paths ─────────────────────────────────────────
    def _set_files(self, paths: list[Path]) -> None:
        pdfs = [p for p in paths if p.suffix.lower() in _PDF_EXTS]
        imgs = [p for p in paths if p.suffix.lower() in _IMG_EXTS]
        if pdfs and imgs:
            QMessageBox.warning(
                self, self.tr("Mixed file types"),
                self.tr(
                    "Pick either PDFs OR images, not both — they take different "
                    "import paths."
                ),
            )
            return
        if not pdfs and not imgs:
            QMessageBox.warning(
                self, self.tr("Unsupported files"),
                self.tr("No PDFs or images detected in the dropped selection."),
            )
            return
        if pdfs:
            self._files = pdfs
            self._kind = "pdf"
            if len(pdfs) == 1:
                self._headline.setText(self.tr("{n} PDF selected").format(n=len(pdfs)))
            else:
                self._headline.setText(self.tr("{n} PDFs selected").format(n=len(pdfs)))
        else:
            self._files = imgs
            self._kind = "images"
            if len(imgs) == 1:
                self._headline.setText(self.tr("{n} image selected").format(n=len(imgs)))
            else:
                self._headline.setText(self.tr("{n} images selected").format(n=len(imgs)))
        self._sub.setText(self.tr("Click to change selection"))
        self.files_changed.emit()


# ── footer link strip (reused from old design) ─────────────────────────


class _LinkItem(QWidget):
    def __init__(self, icon_name: str, label: str, url: str,
                 *, glow_color: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._url = url
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(8)
        h.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        from lib.gui.theme import lucide_pixmap
        icon_color = glow_color or COLOR_FONT_LINK
        pix = lucide_pixmap(icon_name, color=icon_color, size=20)
        pix.setDevicePixelRatio(2.0)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(pix)
        icon_lbl.setFixedSize(20, 20)
        icon_lbl.setStyleSheet("background:transparent; border:none;")
        if glow_color is not None:
            glow = QGraphicsDropShadowEffect(icon_lbl)
            glow.setOffset(0, 0)
            glow.setBlurRadius(90)
            glow.setColor(QColor(glow_color))
            icon_lbl.setGraphicsEffect(glow)
        h.addWidget(icon_lbl)
        text_lbl = QLabel(label)
        text_color = glow_color or COLOR_FONT_LINK
        text_weight = "600" if glow_color else "normal"
        text_lbl.setStyleSheet(
            f"QLabel{{color:{text_color}; font-size:13px; font-weight:{text_weight}; "
            f"background:transparent;}}"
            f"QLabel:hover{{color:{COLOR_FONT_LINK_HOVER}; text-decoration:underline;}}"
        )
        if glow_color is not None:
            text_glow = QGraphicsDropShadowEffect(text_lbl)
            text_glow.setOffset(0, 0)
            text_glow.setBlurRadius(60)
            text_glow.setColor(QColor(glow_color))
            text_lbl.setGraphicsEffect(text_glow)
        h.addWidget(text_lbl)
        self.setToolTip(url)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(self._url))
        super().mouseReleaseEvent(ev)


# ── main window ────────────────────────────────────────────────────────


class StartupWindow(QDialog):
    """Project launcher: recent-grid landing → new-project form."""

    MODE_OPEN = "open"
    MODE_PDF = "pdf"
    MODE_IMAGES = "images"
    MODE_CAPTURE = "capture"

    GRID_COLS = 3
    # Tile geometry copied from `_RecentCard.CARD_W/H`; centralised here
    # so the window sizes exactly to a 3×3 of those cards plus margins
    # and the surrounding chrome (header, footer, cancel row).
    _CARD_W = 240
    _CARD_H = 140
    _GRID_HGAP = 18
    _GRID_VGAP = 18
    _MAX_RECENTS = 7  # 9 tiles total = 1 Open + 7 recents + 1 New

    def __init__(self, parent=None, *, initial_action: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Aglaïa — new session"))
        self.setModal(True)
        # Pre-navigation requested by a ⌘N / ⌘O round-trip from MainWindow:
        # "new" lands on the new-project form, "open" fires the file dialog.
        self._initial_action = initial_action

        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._cameras = list_cameras()
        self._choice: Optional[StartupChoice] = None

        # Pipeline state for the new-project page. Picking a pipeline pill
        # loads its yaml; opening Properties launches the full editor.
        self._pipeline_yaml: str = load_yaml_text(DEFAULT_PIPELINE_PATH)
        self._pipeline_path: Optional[Path] = DEFAULT_PIPELINE_PATH

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_landing_page())   # 0
        self._stack.addWidget(self._build_new_page())       # 1
        outer.addWidget(self._stack, 1)

        # Size the dialog to exactly fit the actually-populated grid +
        # chrome. Cap recents at MAX, total tiles = 1 Open + N recents
        # + 1 New, rounded up to whole rows so the bottom edge sits flush
        # against the footer (no tall blank pane below sparse grids).
        n_recents = min(len(self._load_recents()), self._MAX_RECENTS)
        n_tiles = 1 + n_recents + 1
        n_rows = max(1, (n_tiles + self.GRID_COLS - 1) // self.GRID_COLS)
        grid_w = self.GRID_COLS * self._CARD_W + (self.GRID_COLS - 1) * self._GRID_HGAP
        grid_h = n_rows * self._CARD_H + (n_rows - 1) * self._GRID_VGAP
        # Chrome budget = title row (~46) + post-title spacing (16) +
        # outer dialog margins (20+20) + outer dialog spacing (12) +
        # footer link strip (~56) + footer top spacing (12). Bottom
        # margin matches the side margin so the window reads as
        # uniformly padded.
        # Extra 14 px below the grid so the bottom row's 1.5 px border
        # doesn't bleed under the footer link strip (regression at small
        # window heights — looked like the cards were cropped).
        chrome_h = 46 + 16 + 40 + 12 + 56 + 12 + 14 + 18 + 30
        # +48 (not the bare 20+20 margins) gives the edge cards' 1.5 px
        # selected/accent borders a few px of slack so they don't clip.
        win_w = grid_w + 48
        win_h = grid_h + chrome_h
        self.setFixedSize(win_w, win_h)

        self._install_shortcuts()
        # Honour the ⌘N / ⌘O pre-navigation once the dialog is up.
        if self._initial_action == "new":
            self._stack.setCurrentIndex(1)
        elif self._initial_action == "open":
            QTimer.singleShot(0, self._open_existing)

    def _install_shortcuts(self) -> None:
        """⌘Q quit · ⌘N new project · ⌘O open project — so the launcher
        feels native even before a project window exists."""
        from PySide6.QtGui import QShortcut, QKeySequence
        defs = [
            (QKeySequence.StandardKey.Quit, self.reject),
            (QKeySequence.StandardKey.New, self._go_new),
            (QKeySequence.StandardKey.Open, self._open_existing),
        ]
        for seq, fn in defs:
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)

    # ── page 0: recents grid + new ────────────────────────────────────
    def _build_landing_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        title = QLabel()
        title.setAccessibleName(self.tr("Aglaïa Scanner"))
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        # Wordmark image instead of styled text — pick the variant that
        # matches the active colour scheme; fall back to text if the asset
        # is missing.
        from lib.assets import asset_path
        scheme = "dark" if active_palette_name() == "dark" else "light"
        logo_path = asset_path("brand", f"aglaia-{scheme}.png")
        pm = QPixmap(str(logo_path))
        if not pm.isNull():
            dpr = self.devicePixelRatioF() or 1.0
            target_h = 84  # logical px
            scaled = pm.scaledToHeight(
                int(target_h * dpr), Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(dpr)
            title.setPixmap(scaled)
        else:
            title.setText(self.tr("Aglaïa Scanner"))
            title.setStyleSheet(
                f"color: {COLOR_FONT_PRIMARY};"
                " font-size: 26px; font-weight: 700;"
            )
        v.addWidget(title)

        v.addSpacing(16)

        # Direct grid (no scroll area): tile count is capped at 9 so no
        # scroll is needed, and the resizable scroll host was stretching
        # the grid columns past `CARD_W` because `setWidgetResizable=True`
        # grew the host to viewport width. Wrap the grid in an outer
        # HBox + stretches so it stays horizontally centred while
        # columns stay pinned to `CARD_W`.
        grid_wrap = QHBoxLayout()
        grid_wrap.setContentsMargins(0, 0, 0, 0)
        grid_wrap.addStretch(1)
        host = QWidget()
        host.setStyleSheet("background: transparent;")
        host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._recents_grid = QGridLayout(host)
        self._recents_grid.setContentsMargins(0, 0, 0, 0)
        self._recents_grid.setHorizontalSpacing(self._GRID_HGAP)
        self._recents_grid.setVerticalSpacing(self._GRID_VGAP)
        # No row/col stretch or min-size pinning here — would reserve
        # blank rows in sparse grids (2 recents → 3rd row empty but
        # 140 px tall). Cards' own setFixedSize handles dimensions.
        grid_wrap.addWidget(host, 0, Qt.AlignmentFlag.AlignTop)
        grid_wrap.addStretch(1)
        v.addLayout(grid_wrap, 0)
        # Breathing room between the last card row and the footer strip.
        v.addSpacing(18)

        # Bottom row: tip / homepage / repo links centred, Cancel
        # right-anchored on the same baseline. `_LinkItem` packs a slim
        # 6 px vertical padding — tightened from 18 so the taller wordmark
        # logo fits without growing the window.
        footer = QHBoxLayout()
        # Top-only inset; the dialog's outer 20 px bottom margin already
        # gives the strip its symmetric breathing room vs. the 20 px
        # side margins.
        footer.setContentsMargins(0, 4, 0, 0)
        footer.setSpacing(12)
        footer.addStretch(1)
        from lib.app_data import db as _cfg
        try:
            tips_off = _cfg.tip_buttons_disabled()
        except Exception:
            tips_off = False
        for icon_name, label, url, glow, is_tip in (
            ("heart",         self.tr("Tip the developer!"), TIPPING_URL,  COLOR_ERROR_STRONG, True),
            ("link",          self.tr("Aglaïa homepage"),    HOMEPAGE_URL, None,      False),
            ("folder-git-2",  self.tr("Git code repo"),      GIT_REPO,     None,      False),
        ):
            if is_tip and tips_off:
                continue
            footer.addWidget(_LinkItem(icon_name, label, url, glow_color=glow))
        footer.addStretch(1)
        quit_btn = QPushButton(self.tr("Quit"))
        quit_btn.clicked.connect(self.reject)
        footer.addWidget(quit_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(footer)

        self._populate_recents_grid()
        return w

    def _populate_recents_grid(self) -> None:
        """Fill the 3×3 grid:

            [Open project]  [recent 1] [recent 2]
            [recent 3]      [recent 4] [recent 5]
            [recent 6]      [recent 7] [New project]

        Open + New are both accented (primary border / soft tint) so the
        action tiles bookend the recent history."""
        while self._recents_grid.count():
            it = self._recents_grid.takeAt(0)
            wgt = it.widget()
            if wgt is not None:
                # hide() first — Qt re-shows a visible, implicitly-shown
                # widget after reparent (bare top-level window flash).
                wgt.hide()
                wgt.setParent(None)

        items: list[_RecentCard] = []

        # First tile: Open existing project (primary-styled).
        open_card = _RecentCard(
            title=self.tr("Open project"),
            subtitle=self.tr("Pick an existing .agl file"),
            icon_name="folder-open",
            accent=True,
        )
        open_card.clicked.connect(self._open_existing)
        items.append(open_card)

        # Recents — capped so the 3×3 grid stays uniform with bookends.
        for rec in self._load_recents()[: self._MAX_RECENTS]:
            name, path_str, exists, scan_count = rec
            p = Path(path_str)
            sub = str(p.parent)
            hint = ""
            try:
                if exists and p.exists():
                    from datetime import datetime
                    ts = datetime.fromtimestamp(p.stat().st_mtime)
                    hint = ts.strftime("%Y-%m-%d %H:%M")
            except OSError:
                hint = ""
            card = _RecentCard(
                title=name or p.stem,
                subtitle=sub,
                hint=hint,
                icon_name="collections",
                missing=not exists,
                removable=True,
                scan_count=scan_count,
            )
            if exists:
                card.clicked.connect(
                    lambda _=False, ps=path_str: self._open_recent(ps)
                )
            card.remove_requested.connect(
                lambda ps=path_str: self._on_remove_recent(ps)
            )
            items.append(card)

        # Last tile: New project (primary-styled).
        new_card = _RecentCard(
            title=self.tr("New project"),
            subtitle=self.tr("Capture from camera or import files"),
            icon_name="plus",
            accent=True,
        )
        new_card.clicked.connect(self._go_new)
        items.append(new_card)

        for i, card in enumerate(items):
            row, col = divmod(i, self.GRID_COLS)
            # AlignCenter on the cell prevents QGridLayout from stretching
            # the QFrame past its setFixedSize when the grid host widget
            # is wider than `3 × CARD_W + 2 × gap` (which happens because
            # scrollarea's setWidgetResizable=True grows the host to the
            # viewport size).
            self._recents_grid.addWidget(card, row, col,
                                         Qt.AlignmentFlag.AlignCenter)

    def _load_recents(self) -> list[tuple[str, str, bool, int | None]]:
        """Return list of (name, path, exists, scan_count) tuples — newest
        first.

        Pulled from the per-user config DB. Existing-but-renamed paths
        are surfaced as disabled cards so the user knows what's gone.
        scan_count is cached in recent_projects (None until first open)."""
        try:
            from lib.app_data import db as cfg
            with cfg.session() as conn:
                rows = cfg.list_recent_projects(conn, limit=16)
        except Exception:
            rows = []
        from lib.storage import resolve_existing_project_db, is_project_file
        out: list[tuple[str, str, bool, int | None]] = []
        for r in rows:
            try:
                p = Path(r["path"])
            except Exception:
                continue
            if p.is_file() and is_project_file(p):
                exists = p.exists()
            else:
                exists = (resolve_existing_project_db(p, p.name) is not None)
            try:
                sc = r["scan_count"]
            except (IndexError, KeyError):
                sc = None
            out.append((r["name"] or p.name, str(p), exists, sc))
        return out

    # ── page 1: new project ──────────────────────────────────────────
    def _build_new_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        # Top bar: back + heading.
        head = QHBoxLayout()
        back_btn = QPushButton(self.tr("← Back"))
        back_btn.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        head.addWidget(back_btn)
        title = QLabel(self.tr("New project"))
        title.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-size: 18px; font-weight: 700;"
        )
        head.addWidget(title)
        head.addStretch(1)
        v.addLayout(head)

        # Name + parent folder row.
        nf_row = QHBoxLayout()
        nf_row.setSpacing(10)
        col_name = QVBoxLayout()
        col_name.setSpacing(3)
        col_name.addWidget(QLabel(self.tr("Name")))
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText(self.tr("my-book"))
        col_name.addWidget(self._new_name)
        nf_row.addLayout(col_name, 2)

        col_parent = QVBoxLayout()
        col_parent.setSpacing(3)
        col_parent.addWidget(QLabel(self.tr("Parent folder")))
        self._new_parent = QLineEdit(self._remembered_parent("default"))
        self._attach_folder_action(self._new_parent,
                                   lambda: self._browse_dir_into(self._new_parent))
        col_parent.addWidget(self._new_parent)
        nf_row.addLayout(col_parent, 3)
        v.addLayout(nf_row)

        # Source radio cards.
        v.addSpacing(8)
        self._src_group = RadioCardGroup(orientation="horizontal")

        # Capture extras: camera combo.
        cam_extras = QWidget()
        cam_h = QHBoxLayout(cam_extras)
        cam_h.setContentsMargins(0, 4, 0, 0)
        cam_h.setSpacing(8)
        cam_lbl = QLabel(self.tr("Camera"))
        cam_lbl.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        cam_h.addWidget(cam_lbl)
        self._cam_combo = QComboBox()
        for idx, name in self._cameras:
            self._cam_combo.addItem(f"{name} (#{idx})", userData=idx)
        cam_h.addWidget(self._cam_combo, 1)
        self._src_group.add_card(
            "capture", self.tr("From capture"),
            self.tr("Live webcam / Continuity Camera"),
            icon_name="camera",
            extras=cam_extras,
        )

        # Files extras: drop zone + input-DPI row.
        self._drop_zone = _DropZone()
        files_extras = QWidget()
        files_v = QVBoxLayout(files_extras)
        files_v.setContentsMargins(0, 0, 0, 0)
        files_v.setSpacing(6)
        files_v.addWidget(self._drop_zone)

        dpi_row = QHBoxLayout()
        dpi_row.setSpacing(8)
        dpi_lbl = QLabel(self.tr("Input DPI (images only)"))
        dpi_lbl.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        dpi_row.addWidget(dpi_lbl)
        self._files_dpi_spin = QDoubleSpinBox()
        self._files_dpi_spin.setRange(50.0, 1200.0)
        self._files_dpi_spin.setSingleStep(50.0)
        self._files_dpi_spin.setDecimals(0)
        self._files_dpi_spin.setSuffix(" dpi")
        self._files_dpi_spin.setValue(120.0)
        self._files_dpi_spin.setToolTip(self.tr(
            "Assumed resolution for imported images. PDFs ignore this — "
            "each page's DPI is estimated from its paper size."))
        dpi_row.addWidget(self._files_dpi_spin)
        dpi_row.addStretch(1)
        files_v.addLayout(dpi_row)

        self._src_group.add_card(
            "files", self.tr("From files"),
            self.tr("Click or drop PDFs / images"),
            icon_name="file-text",
            extras=files_extras,
        )
        self._src_group.set_current_key("capture")
        v.addWidget(self._src_group)

        # Pipeline mode picker (fills the remaining space).
        v.addSpacing(8)
        self._mode_picker = ModePickerPanel()
        self._mode_picker.pipelineChanged.connect(self._on_mode_pipeline_changed)
        v.addWidget(self._mode_picker, 1)
        # Capture the picker's initial selection (its constructor emits
        # before we connected, so pull the current value explicitly).
        if self._mode_picker.current_yaml():
            self._pipeline_yaml = self._mode_picker.current_yaml()
            self._pipeline_path = self._mode_picker.current_path()

        # Actions row.
        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        actions.addWidget(cancel_btn)
        self._continue_btn = QPushButton(self.tr("Create"))
        self._continue_btn.setDefault(True)
        self._continue_btn.clicked.connect(self._on_create)
        actions.addWidget(self._continue_btn)
        v.addLayout(actions)

        return w

    def _on_mode_pipeline_changed(self, yaml_text: str, path) -> None:
        self._pipeline_yaml = yaml_text
        self._pipeline_path = path

    # ── landing actions ────────────────────────────────────────────────
    def _go_new(self) -> None:
        self._stack.setCurrentIndex(1)

    def _open_existing(self) -> None:
        from lib.storage import (
            PROJECT_DIALOG_FILTER, is_project_file, slug_from_project_file,
        )
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open project"), str(Path.home()), PROJECT_DIALOG_FILTER,
        )
        if not path:
            return
        p = Path(path).expanduser()
        if not (p.is_file() and is_project_file(p)):
            self._warn(self.tr(
                "Pick a .agl (or legacy .scanproj.sqlite) file: {path}"
            ).format(path=p))
            return
        choice = StartupChoice(mode=self.MODE_OPEN, pipeline_yaml="")
        choice.project_dir = p.parent
        choice.project_slug = slug_from_project_file(p)
        choice.project_name = choice.project_slug
        self._choice = choice
        self.accept()

    def _on_remove_recent(self, path: str) -> None:
        """Forget `path` from the recent-projects table and rebuild the
        grid. Does NOT delete the project file on disk."""
        try:
            from lib.app_data import db as cfg
            with cfg.session() as conn:
                cfg.forget_project(conn, path)
        except Exception:
            return
        self._populate_recents_grid()
        # Window size depends on remaining row count — re-pin via a
        # fresh setFixedSize call. Unfix first so Qt accepts the new
        # size.
        n_recents = min(len(self._load_recents()), self._MAX_RECENTS)
        n_tiles = 1 + n_recents + 1
        n_rows = max(1, (n_tiles + self.GRID_COLS - 1) // self.GRID_COLS)
        grid_w = self.GRID_COLS * self._CARD_W + (self.GRID_COLS - 1) * self._GRID_HGAP
        grid_h = n_rows * self._CARD_H + (n_rows - 1) * self._GRID_VGAP
        # Extra 14 px below the grid so the bottom row's 1.5 px border
        # doesn't bleed under the footer link strip (regression at small
        # window heights — looked like the cards were cropped).
        chrome_h = 46 + 16 + 40 + 12 + 56 + 12 + 14 + 18 + 30
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.setFixedSize(grid_w + 48, grid_h + chrome_h)

    def _open_recent(self, path: str) -> None:
        from lib.storage import (
            is_project_file, resolve_existing_project_db, slug_from_project_file,
        )
        p = Path(path).expanduser()
        proj_file: Optional[Path] = None
        if p.is_file() and is_project_file(p):
            proj_file = p
        elif p.is_dir():
            proj_file = resolve_existing_project_db(p, p.name)
        if proj_file is None:
            self._warn(self.tr("Project no longer exists: {path}").format(path=p))
            return
        choice = StartupChoice(mode=self.MODE_OPEN, pipeline_yaml="")
        choice.project_dir = proj_file.parent
        choice.project_slug = slug_from_project_file(proj_file)
        choice.project_name = choice.project_slug
        self._choice = choice
        self.accept()

    # ── settings helpers ───────────────────────────────────────────────
    def _settings_key_parent(self, kind: str) -> str:
        return f"last_parent_dir/{kind}"

    def _remembered_parent(self, kind: str) -> str:
        v = self._settings.value(self._settings_key_parent(kind), "")
        return str(v) if v else str(Path.home())

    def _remember_parent(self, kind: str, value: str) -> None:
        self._settings.setValue(self._settings_key_parent(kind), value)

    # ── file pickers ───────────────────────────────────────────────────
    def _attach_folder_action(self, edit: QLineEdit, on_click) -> None:
        from lib.gui.theme import icon as _icon
        act = edit.addAction(
            _icon("folder-open", color=COLOR_FONT_PLACEHOLDER),
            QLineEdit.ActionPosition.TrailingPosition,
        )
        act.setToolTip(self.tr("Choose folder…"))
        act.triggered.connect(on_click)

    def _browse_dir_into(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, self.tr("Choose folder"), line_edit.text())
        if d:
            line_edit.setText(d)

    # ── finalize new ───────────────────────────────────────────────────
    def _on_create(self) -> None:
        name = self._new_name.text().strip()
        parent = Path(self._new_parent.text().strip()).expanduser()
        if not name:
            self._warn(self.tr("Pick a project name first."))
            return
        if not parent.exists():
            self._warn(self.tr("Parent folder does not exist: {path}").format(path=parent))
            return

        src = self._src_group.current_key() or "capture"
        from slugify import slugify
        slug = slugify(name) or "project"
        choice = StartupChoice(
            mode=self.MODE_CAPTURE if src == "capture" else "",
            project_name=name,
            project_slug=slug,
            parent_dir=parent,
            project_dir=parent,
            pipeline_yaml=self._pipeline_yaml,
        )

        if src == "capture":
            choice.camera_index = int(self._cam_combo.currentData() or 0)
        else:
            files = self._drop_zone.files()
            kind = self._drop_zone.kind()
            if not files or kind is None:
                self._warn(self.tr("Drop at least one PDF or image first."))
                return
            choice.mode = self.MODE_PDF if kind == "pdf" else self.MODE_IMAGES
            choice.input_files = list(files)
            choice.input_dpi = float(self._files_dpi_spin.value())

        self._remember_parent("default", str(parent))
        self._choice = choice
        self.accept()

    def _warn(self, msg: str):
        QMessageBox.warning(self, self.tr("Incomplete selection"), msg)

    def choice(self) -> Optional[StartupChoice]:
        return self._choice
