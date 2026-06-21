# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Modern dark/light theme + icon helpers for the Qt capture GUI.

Wraps `pyqtdarktheme` (palette + base QSS) behind a single
`apply_modern_theme(app)` call so the entry script doesn't have to know
which library we use. Icons come from bundled Lucide SVGs via `icon()`.

The QSS appended on top of qdarktheme's stylesheet pushes a few things our
custom widgets need that the base theme doesn't ship:

  * `QFrame#Card` / `QFrame#CardHeader` — rounded, slightly elevated card
    containers used to group settings (PipelineEditorWidget, StartupWindow).
  * `QLabel#SectionTitle` / `QLabel#Subtle` — typography hooks for the
    section headers + help text in the params form.
  * `Switch` (lib.gui.widgets.Switch) — toggle pill rendered via QSS, used
    in place of QCheckBox for boolean options.
  * Bigger default spinbox / combo / line-edit height (32 px) so the form
    breathes instead of cramming 12 inputs into a single screen.

Call `apply_modern_theme(app, mode="auto")` once, right after
`QApplication(sys.argv)`, then use `icon(name)` to fetch FontAwesome glyphs
that follow the active palette.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import sys

import qdarktheme
from PySide6.QtCore import Qt, QByteArray, QEvent, QObject, QSize
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QComboBox, QScrollArea,
)

from lib.gui.colors import (
    COLOR_ACCENT_BADGE_BG,
    COLOR_ACCENT_BADGE_BORDER,
    COLOR_ACCENT_BADGE_FG,
    COLOR_BG_HINT,
    COLOR_BG_INPUT,
    COLOR_BG_INPUT_DISABLED,
    COLOR_BG_INPUT_FOCUS,
    COLOR_BG_OVERLAY_HOVER,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_BG_SURFACE,
    COLOR_BG_SURFACE_ALT,
    COLOR_FONT_DISABLED,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY,
    COLOR_FONT_SECONDARY,
    COLOR_OUTLINE,
    COLOR_OUTLINE_FAINT,
    COLOR_OUTLINE_GHOST,
    COLOR_OUTLINE_STRONG,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
)

# Bundled Lucide SVGs (https://lucide.dev). Each file uses
# `stroke="currentColor"` so we swap that in for the requested color before
# handing the SVG to QSvgRenderer — gives us palette-aware icons without
# shipping multiple colour variants.
_ICONS_DIR = Path(__file__).parent / "icons"


# Extra QSS layered on top of qdarktheme's base stylesheet. Keeps things
# local to this module so the editor file is just layout code.
_EXTRA_QSS = f"""
/* ── Card containers ─────────────────────────────────────────────── */
QFrame#Card {{
    background-color: {COLOR_BG_SURFACE};
    border: 1px solid {COLOR_OUTLINE_SUBTLE};
    border-radius: 10px;
}}
QFrame#Card[elevated="true"] {{
    background-color: {COLOR_BG_SURFACE_ALT};
}}

QFrame#CardHeader {{
    background: transparent;
    border: none;
    border-bottom: 1px solid {COLOR_OUTLINE_FAINT};
    padding: 6px 12px;
}}

/* Inside styled cards / form panels, kill the palette-window bg on
   text-bearing controls so they inherit the card's painted bg. Scoped
   by objectName so tooltips (QTipLabel — bare QLabel with no card
   ancestor) are NOT swept up by a global QLabel rule. Add new card
   class names here when introducing more bg-painted containers. */
QFrame#RadioCard QLabel,
QFrame#RadioCard QCheckBox,
QFrame#RadioCard QRadioButton,
QFrame#RecentCard QLabel,
QFrame#RecentCard QCheckBox,
QFrame#PipelineRow QLabel,
QFrame#PipelineRow QCheckBox,
QFrame#Card QLabel,
QFrame#Card QCheckBox,
QFrame#Card QRadioButton,
QFrame#FieldCell QLabel,
QFrame#FieldCell QCheckBox,
QFrame#FieldCell QRadioButton,
QFrame#DropZone QLabel {{
    background-color: transparent;
}}

QLabel#SectionTitle {{
    font-size: 12px;
    font-weight: 700;
    color: palette(text);
    letter-spacing: 0.4px;
    text-transform: uppercase;
}}

QLabel#FieldLabel {{
    font-size: 12px;
    font-weight: 600;
    color: palette(text);
}}

QLabel#Subtle, QLabel#HelpText {{
    color: {COLOR_FONT_MUTED};
    font-size: 11px;
}}

QLabel#Badge {{
    background-color: {COLOR_ACCENT_BADGE_BG};
    color: {COLOR_ACCENT_BADGE_FG};
    border: 1px solid {COLOR_ACCENT_BADGE_BORDER};
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}}

/* ── Inputs ──────────────────────────────────────────────────────── */
/* Material-style outlined inputs: subtle filled bg + 1px border that
   crisps to accent on focus. Distinct enough from `QFrame#Card` /
   buttons that the user can scan a 2-col form without confusing a
   label with the input on its left. */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 30px;
    padding: 2px 10px;
    border-radius: 6px;
    background-color: {COLOR_BG_INPUT};
    border: 1px solid {COLOR_OUTLINE};
    color: palette(text);
    selection-background-color: {COLOR_PRIMARY};
    selection-color: {COLOR_FONT_INVERSE};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {COLOR_OUTLINE_STRONG};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1.5px solid {COLOR_PRIMARY};
    background-color: {COLOR_BG_INPUT_FOCUS};
}}
QLineEdit:read-only {{
    background-color: {COLOR_BG_INPUT_DISABLED};
    color: {COLOR_FONT_MUTED};
}}

/* Combo popup — qdarktheme leaves QAbstractItemView at the dark
   palette regardless of light/dark mode, so the dropdown turned into a
   black slab on light theme. Pin bg / fg / selection to palette tokens
   so the list tracks the active scheme. */
QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_INPUT};
    color: palette(text);
    border: 1px solid {COLOR_OUTLINE};
    border-radius: 0px;
    padding: 4px;
    outline: 0;
    selection-background-color: {COLOR_PRIMARY};
    selection-color: {COLOR_FONT_INVERSE};
}}
QComboBox QAbstractItemView::item {{
    padding: 4px 8px;
    border-radius: 4px;
    min-height: 22px;
}}
QComboBox QAbstractItemView::item:hover {{
    background-color: {COLOR_BG_OVERLAY_HOVER};
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {COLOR_PRIMARY};
    color: {COLOR_FONT_INVERSE};
}}

/* Pill buttons used by the pipeline picker. Marked via dynamic property
   `pill="true"` so the style only hits opt-in buttons. */
QPushButton[pill="true"] {{
    background-color: {COLOR_BG_OVERLAY_SOFT};
    color: {COLOR_FONT_SECONDARY};
    border: 1px solid {COLOR_OUTLINE};
    border-radius: 14px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton[pill="true"]:hover {{
    background-color: {COLOR_BG_OVERLAY_HOVER};
    color: {COLOR_FONT_INVERSE};
}}
QPushButton[pill="true"][active="true"] {{
    background-color: {COLOR_PRIMARY_BG_SOFT};
    border: 1.5px solid {COLOR_PRIMARY};
    color: {COLOR_FONT_INVERSE};
}}
QPushButton[pill="true"]:disabled {{
    background-color: {COLOR_BG_OVERLAY_SOFT};
    color: {COLOR_FONT_DISABLED};
    border: 1px dashed {COLOR_OUTLINE_GHOST};
}}

QListWidget {{
    border-radius: 8px;
    padding: 4px;
}}
QListWidget::item {{
    padding: 8px 10px;
    border-radius: 6px;
    margin: 2px 0;
}}

/* ── Context / popup menus ───────────────────────────────────────── */
/* qdarktheme's QMenu picks up a transparent surface on some Qt
   versions — the items then render directly on top of whatever sits
   under the cursor, illegible on light theme. Pin opaque tokens. */
QMenu {{
    background-color: {COLOR_BG_INPUT};
    color: palette(text);
    border: 1px solid {COLOR_OUTLINE};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 16px 6px 12px;
    border-radius: 4px;
    margin: 1px 2px;
}}
QMenu::item:selected {{
    background-color: {COLOR_PRIMARY};
    color: {COLOR_FONT_INVERSE};
}}
QMenu::item:disabled {{
    color: {COLOR_FONT_DISABLED};
}}
QMenu::separator {{
    height: 1px;
    background: {COLOR_OUTLINE_SUBTLE};
    margin: 4px 6px;
}}

/* ── Tooltips ─────────────────────────────────────────────────────
   qdarktheme's QToolTip stylesheet sets `opacity: 200` + dark surface;
   on macOS the system tooltip palette then blends through, leaving
   dark-text-on-dark-overlay. Force solid bg + explicit text color so
   tooltips track the active scheme on both themes. */
QToolTip,
QTipLabel {{
    background-color: {COLOR_BG_HINT};
    color: {COLOR_FONT_PRIMARY};
    border: 1px solid {COLOR_OUTLINE_SUBTLE};
    border-radius: 0;
    padding: 1px 4px;
    font-size: 11px;
    font-weight: 400;
}}

/* ── Tool buttons (icon-only) ────────────────────────────────────── */
QToolButton {{
    border-radius: 6px;
    padding: 6px;
}}
QToolButton:hover {{
    background-color: {COLOR_BG_OVERLAY_HOVER};
}}

/* ── Custom Switch (lib.gui.widgets.Switch) ──────────────────────── */
Switch {{
    qproperty-trackOff: {COLOR_OUTLINE};
    qproperty-trackOn:  {COLOR_PRIMARY};
    qproperty-thumb:    {COLOR_FONT_INVERSE};
}}
"""


def _pick_system_font() -> QFont:
    """Resolve a real system font family instead of letting Qt fall back
    to the generic "Sans Serif" alias. The alias triggers an expensive
    family-alias resolution on every QFont() construction (visible in
    the `qt.qpa.fonts: Populating font family aliases took … ms` log)
    and on macOS isn't even a registered family — so QFont("Sans Serif")
    paths through fontconfig-style globbing.

    macOS: ".AppleSystemUIFont" (system font, used by every native app).
    Other: first available of [Inter, Segoe UI, Helvetica Neue, Arial].
    """
    db = QFontDatabase
    families = set(db.families())
    if sys.platform == "darwin":
        # ".AppleSystemUIFont" is the official handle for the macOS UI
        # font (San Francisco). It's not in `families()` but Qt resolves
        # it at paint time, no alias lookup penalty.
        return QFont(".AppleSystemUIFont", 13)
    for cand in ("Inter", "Segoe UI", "Helvetica Neue", "Arial"):
        if cand in families:
            return QFont(cand, 10)
    return QFont(db.systemFont(QFontDatabase.SystemFont.GeneralFont).family(), 10)


class _NoScrollWheelFilter(QObject):
    """Application-wide event filter: drops wheel events delivered to
    `QSpinBox / QDoubleSpinBox / QComboBox`. Without it, scrolling the
    page inside a `QScrollArea` would land on whatever spin box happens
    to be under the cursor and change its value silently — the user
    only meant to scroll the form.

    When a scroll-area ancestor exists, the wheel event is forwarded to
    its viewport so the page still scrolls.
    """

    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        if event.type() == QEvent.Type.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)):
            w = obj.parentWidget()
            while w is not None:
                if isinstance(w, QScrollArea):
                    QApplication.sendEvent(w.viewport(), event)
                    return True
                w = w.parentWidget()
            return True
        return False


class _CustomTooltip(QObject):
    """App-wide replacement for Qt's native tooltip.

    Qt's `QTipLabel` on macOS lives inside a translucent `NSPanel` whose
    alpha buffer qdarktheme requests at construction time. After the
    native surface is built, neither `WA_TranslucentBackground = False`
    nor `autoFillBackground = True` nor a per-instance stylesheet nor
    `WA_OpaquePaintEvent` get bg pixels onto the screen — the bg paints
    into the alpha-channeled buffer and the underlying scene shows
    through. Tried all four; none reliably win.

    So we don't fight Qt — we replace it. Returning True from
    `QEvent.Type.ToolTip` suppresses Qt's tooltip entirely; in the same
    handler we pop a `QLabel` of our own, parented to the screen as a
    `Qt.ToolTip` window. It's a normal top-level QLabel, so opaque
    `background-color` from QSS + `autoFillBackground` from palette
    Work As Expected. Show / hide on a single shared QLabel keeps
    the cost flat regardless of how many widgets have tooltips."""

    _SHOW_DELAY_MS = 500
    _HIDE_DELAY_MS = 100

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        from PySide6.QtCore import Qt as _Qt, QTimer
        from PySide6.QtWidgets import QLabel
        self._popup = QLabel(
            None,
            _Qt.WindowType.ToolTip
            | _Qt.WindowType.FramelessWindowHint
            | _Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self._popup.setAttribute(_Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._popup.setAttribute(_Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._popup.setAutoFillBackground(True)
        self._popup.setStyleSheet(
            f"QLabel {{"
            f" background-color: {COLOR_BG_HINT};"
            f" color: {COLOR_FONT_PRIMARY};"
            f" border: 1px solid {COLOR_OUTLINE_SUBTLE};"
            f" border-radius: 0;"
            f" padding: 1px 4px;"
            f" font-size: 11px;"
            f" font-weight: 400;"
            f"}}"
        )
        self._popup.setContentsMargins(0, 0, 0, 0)
        font = self._popup.font()
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.Normal)
        self._popup.setFont(font)
        self._pending_text = ""
        self._pending_global_pos = None
        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.timeout.connect(self._do_show)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._popup.hide)

    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        from PySide6.QtWidgets import QWidget
        t = event.type()
        if t == QEvent.Type.ToolTip and isinstance(obj, QWidget):
            text = (obj.toolTip() or "").strip()
            if not text:
                return False
            self._pending_text = text
            try:
                self._pending_global_pos = event.globalPos()
            except AttributeError:
                self._pending_global_pos = event.globalPosition().toPoint()
            # Suppress Qt's native tooltip; ours fires after the delay.
            self._show_timer.start(self._SHOW_DELAY_MS)
            return True
        if t in (
            QEvent.Type.Leave,
            QEvent.Type.HoverLeave,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.FocusOut,
            QEvent.Type.WindowDeactivate,
        ):
            self._show_timer.stop()
            if self._popup.isVisible():
                self._hide_timer.start(self._HIDE_DELAY_MS)
        return False

    def _do_show(self):
        if not self._pending_text or self._pending_global_pos is None:
            return
        self._popup.setText(self._pending_text)
        self._popup.adjustSize()
        # Position 14 px below and 10 px right of the cursor by default,
        # then keep it ON-SCREEN: flip to the left / above the cursor when
        # it would overflow the right / bottom edge (the activity bar sits
        # on the window's right edge, so its tooltips overflowed right),
        # and finally clamp inside the screen.
        from PySide6.QtGui import QGuiApplication
        pos = self._pending_global_pos
        w, h = self._popup.width(), self._popup.height()
        scr = (QGuiApplication.screenAt(pos)
               or QGuiApplication.primaryScreen())
        geo = scr.availableGeometry()
        x = pos.x() + 10
        y = pos.y() + 14
        if x + w > geo.right():
            x = pos.x() - w - 10
        if y + h > geo.bottom():
            y = pos.y() - h - 14
        x = max(geo.left(), min(x, geo.right() - w))
        y = max(geo.top(), min(y, geo.bottom() - h))
        self._popup.move(x, y)
        self._hide_timer.stop()
        self._popup.show()


class _ComboPopupSizer(QObject):
    """Resize ``QComboBox`` popup to fit its widest entry on every
    ``showPopup``. Qt's default popup width is the combo's own width,
    which clips long items in narrow side panels (Settings, Capture
    sidebar). Caller installs this on the application, and every
    ``QComboBox.view()`` paint-time width grows enough to host the
    longest visible item plus a small breathing pad."""

    _PAD_PX = 36  # scrollbar + checkmark margin + padding inside view

    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        if event.type() == QEvent.Type.Show and isinstance(obj, QAbstractItemView):
            combo = obj.parentWidget()
            # qdarktheme wraps the view inside a QFrame container; walk
            # up one more level when that's the case.
            if combo is not None and not isinstance(combo, QComboBox):
                combo = combo.parentWidget()
            if isinstance(combo, QComboBox):
                fm = combo.fontMetrics()
                widest = 0
                for i in range(combo.count()):
                    w = fm.horizontalAdvance(combo.itemText(i))
                    if w > widest:
                        widest = w
                target = max(combo.width(), widest + self._PAD_PX)
                obj.setMinimumWidth(target)
                # Pin popup bg/fg to the palette tokens. The global
                # ``QComboBox QAbstractItemView`` QSS doesn't reach
                # every popup (qdarktheme reparents the view to a
                # popup-window QFrame on some platforms), so set it
                # directly on the view each time it opens.
                # Square, fully-opaque popup. A rounded radius leaves the
                # corners transparent on the dark theme (the popup window is
                # WA_TranslucentBackground there) — you'd see the desktop /
                # card ghosting through. Force the window opaque + 0 radius.
                obj.setAutoFillBackground(True)
                win = obj.window()
                if win is not None and win is not obj:
                    win.setAttribute(
                        Qt.WidgetAttribute.WA_TranslucentBackground, False)
                    win.setAutoFillBackground(True)
                obj.setStyleSheet(
                    f"QAbstractItemView {{"
                    f"  background-color: {COLOR_BG_INPUT};"
                    f"  color: palette(text);"
                    f"  border: 1px solid {COLOR_OUTLINE};"
                    f"  border-radius: 0px;"
                    f"  padding: 4px;"
                    f"  outline: 0;"
                    f"  selection-background-color: {COLOR_PRIMARY};"
                    f"  selection-color: {COLOR_FONT_INVERSE};"
                    f"}}"
                    f"QAbstractItemView::item {{"
                    f"  padding: 4px 8px;"
                    f"  border-radius: 4px;"
                    f"  min-height: 22px;"
                    f"}}"
                    f"QAbstractItemView::item:selected {{"
                    f"  background-color: {COLOR_PRIMARY};"
                    f"  color: {COLOR_FONT_INVERSE};"
                    f"}}"
                )
        return False


# Kept as a module-level singleton so the GC doesn't reap the filter
# right after `apply_modern_theme` returns.
_NO_SCROLL_FILTER: Optional[_NoScrollWheelFilter] = None
_COMBO_POPUP_SIZER: Optional["_ComboPopupSizer"] = None
_TOOLTIP_STYLER: Optional["_CustomTooltip"] = None


def apply_modern_theme(app: QApplication, mode: str = "auto") -> None:
    """Apply the modern theme to the running Qt app.

    `mode` is forwarded to `qdarktheme.setup_theme` ("auto" follows the OS
    appearance, "dark" / "light" force a specific palette). Extra QSS is
    appended afterwards so it overrides the base stylesheet predictably.
    Also pins a proper system font (see `_pick_system_font`) so Qt
    doesn't spend ~500 ms resolving the "Sans Serif" family alias on
    the first paint.
    """
    app.setFont(_pick_system_font())
    qdarktheme.setup_theme(
        mode,
        corner_shape="rounded",
        custom_colors={
            # Slightly warmer accent than qdarktheme's default cyan —
            # plays better with the orange/blue debug overlays.
            "primary": COLOR_PRIMARY,
        },
    )
    base = app.styleSheet() or ""
    app.setStyleSheet(base + "\n" + _EXTRA_QSS)
    # QToolTip ignores stylesheet bg in some Qt versions on macOS and
    # falls back to the palette ToolTipBase/ToolTipText roles. qdarktheme
    # leaves these on dark values regardless of mode, producing dark text
    # on dark bg in light theme. Patch the app palette so both the QSS
    # and palette paths show readable tooltips.
    from PySide6.QtGui import QColor, QPalette
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLOR_BG_HINT))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(COLOR_FONT_PRIMARY))
    app.setPalette(pal)
    # Wheel-eats-input prevention. Installed once per app — re-applying
    # the theme doesn't double-install because we stash the filter on
    # the module.
    global _NO_SCROLL_FILTER, _COMBO_POPUP_SIZER, _TOOLTIP_STYLER
    if _NO_SCROLL_FILTER is None:
        _NO_SCROLL_FILTER = _NoScrollWheelFilter()
        app.installEventFilter(_NO_SCROLL_FILTER)
    if _COMBO_POPUP_SIZER is None:
        _COMBO_POPUP_SIZER = _ComboPopupSizer()
        app.installEventFilter(_COMBO_POPUP_SIZER)
    if _TOOLTIP_STYLER is None:
        _TOOLTIP_STYLER = _CustomTooltip()
        app.installEventFilter(_TOOLTIP_STYLER)


@lru_cache(maxsize=256)
def _read_svg(name: str) -> Optional[str]:
    p = _ICONS_DIR / f"{name}.svg"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def _svg_color_opacity(color: str) -> tuple[str, float]:
    """Normalise a colour for the SVG renderer. QSvgRenderer can't parse
    CSS ``rgba()`` (the alpha form) — it silently falls back to BLACK,
    which made every palette ``rgba(255,255,255,0.55)``-tinted icon render
    black on the dark theme. Convert ``rgb()``/``rgba()`` → ``#rrggbb`` +
    a separate opacity (applied via the painter); hex / named colours pass
    through untouched."""
    import re
    m = re.match(
        r"\s*rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*"
        r"(?:,\s*([\d.]+)\s*)?\)\s*$", color or "")
    if not m:
        return (color, 1.0)
    r, g, b = (max(0, min(255, int(float(m.group(i))))) for i in (1, 2, 3))
    op = float(m.group(4)) if m.group(4) is not None else 1.0
    return (f"#{r:02x}{g:02x}{b:02x}", max(0.0, min(1.0, op)))


def _render_svg_tinted(name: str, color: Optional[str], size: int):
    """Shared loader: read SVG, retint currentColor, render to a QPixmap
    at 2× with the (possibly rgba) colour's opacity baked in. Returns None
    when the SVG is missing."""
    if color is None:
        pal = QApplication.palette()
        color = pal.color(QPalette.ColorRole.Text).name()
    svg_src = _read_svg(name)
    if svg_src is None:
        return None
    svg_color, opacity = _svg_color_opacity(color)
    tinted = (svg_src
              .replace('stroke="currentColor"', f'stroke="{svg_color}"')
              .replace('fill="currentColor"', f'fill="{svg_color}"'))
    renderer = QSvgRenderer(QByteArray(tinted.encode("utf-8")))
    px = max(16, int(size) * 2)
    pix = QPixmap(px, px)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    if opacity < 1.0:
        p.setOpacity(opacity)
    renderer.render(p)
    p.end()
    return pix


def lucide(name: str, *, color: Optional[str] = None, size: int = 20) -> QIcon:
    """Lucide-style icon (https://lucide.dev) loaded from the bundled
    `lib/gui/icons/*.svg`. ``currentColor`` is retinted to `color` (or
    palette text). Returns an empty QIcon when the SVG is missing."""
    pix = _render_svg_tinted(name, color, size)
    return QIcon(pix) if pix is not None else QIcon()


def lucide_pixmap(name: str, *, color: Optional[str] = None, size: int = 64) -> QPixmap:
    """Like `lucide` but returns a QPixmap — useful for the big startup
    cards that paint the icon directly into a QLabel."""
    pix = _render_svg_tinted(name, color, size)
    return pix if pix is not None else QPixmap()


def icon(name: str, *, color: Optional[str] = None, size: int = 16) -> QIcon:
    """Resolve a Lucide icon name from the bundled SVGs."""
    return lucide(name, color=color, size=size)


def palette_color(role: QPalette.ColorRole, group: QPalette.ColorGroup = QPalette.ColorGroup.Active) -> QColor:
    return QApplication.palette().color(group, role)
