# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR sidebar tab.

Replaces the old ``OcrFrame`` group box. Engine combo uses
``ComboBoxWithDescription`` (≤2 entries today: Apple Vision + Surya).
Languages reuse the existing ``LanguageTagInput``. Run split-button
and download-prompt button preserve the prior behaviour.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.LanguageTagInput import LanguageTagInput
from lib.gui.colors import (
    COLOR_BG,
    COLOR_BG_BUTTON,
    COLOR_BG_BUTTON_HOVER,
    COLOR_BG_BUTTON_PRESSED,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_ERROR,
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_ON_BUTTON,
    COLOR_FONT_PRIMARY,
    COLOR_FONT_SECTION_LABEL,
    COLOR_MEDAL_BRONZE,
    COLOR_MEDAL_GOLD,
    COLOR_MEDAL_SILVER,
    COLOR_OUTLINE_BUTTON,
    COLOR_PRIMARY,
    COLOR_SECONDARY_BG_HOVER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_SECONDARY_BG_SOFT,
    COLOR_SECONDARY_BORDER,
    qcolor,
)
from lib.gui.sidebar.widgets import RadioCardGroup


class _BusyOverlay(QWidget):
    """Scrim + spinner + caption painted on top of OcrTab while a long
    op (OCR run / pipeline run) is in flight. Blocks input — the
    underlying controls are also ``setEnabled(False)`` for keyboard
    focus + a11y, but the overlay communicates the lock visually."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # Eats clicks (no WA_TransparentForMouseEvents) — disabled
        # widgets underneath wouldn't react anyway, but blocking here
        # avoids misleading hover cues.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._idx = 0
        self._caption = self.tr("Working…")
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self.hide()
        if parent is not None:
            parent.installEventFilter(self)

    def set_caption(self, text: str) -> None:
        self._caption = text
        self.update()

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self.show()
        self.raise_()
        self._resize_to_parent()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._idx = (self._idx + 1) % len(self._FRAMES)
        self.update()

    def _resize_to_parent(self) -> None:
        p = self.parentWidget()
        if p is not None:
            self.setGeometry(0, 0, p.width(), p.height())

    def eventFilter(self, obj, ev):  # noqa: N802 — Qt API
        if obj is self.parentWidget() and ev.type() in (
            ev.Type.Resize, ev.Type.Show, ev.Type.Move,
        ):
            self._resize_to_parent()
        return False

    def paintEvent(self, _ev):  # noqa: N802
        self._resize_to_parent()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Match `_SpinnerOverlay` (ScanItemWidget): same scrim alpha so
        # the busy state reads identically across the app.
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        font = QFont()
        font.setPixelSize(40)
        font.setBold(True)
        p.setFont(font)
        p.setPen(qcolor(COLOR_PRIMARY))
        spinner_rect = self.rect().adjusted(0, 0, 0, -32)
        p.drawText(spinner_rect, int(Qt.AlignmentFlag.AlignCenter),
                   self._FRAMES[self._idx])
        font.setPixelSize(13)
        font.setBold(True)
        p.setFont(font)
        p.setPen(qcolor(COLOR_FONT_INVERSE))
        caption_rect = self.rect().adjusted(0, 32, 0, 0)
        p.drawText(caption_rect, int(Qt.AlignmentFlag.AlignCenter),
                   self._caption)
        p.end()


class OcrTab(QWidget):
    """OCR controls — engine + languages + run/download.

    Signals:
      * ``run_requested(engine_name, languages, mode, complement)``
    """

    # engine, languages, mode, complement. ``complement`` is only
    # meaningful for the ``apple_docs`` engine ("surya" / "paddle_vl" /
    # "none"); for every other engine it is "".
    run_requested = Signal(str, list, str, str)
    # bool — emitted whenever the user flips the Live-OCR toggle so
    # MainWindow can install / tear down its branch_ready hook.
    live_ocr_toggled = Signal(bool)
    # str — emitted with the new engine key whenever the user picks a
    # different engine in the radio card group. MainWindow uses it to
    # mark all done OCR runs of the OLD engine as stale.
    engine_changed = Signal(str)

    MODE_DEFAULT = "default"
    MODE_FORCE = "force"
    MODE_MISSING = "missing"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel(self.tr("OCR"))
        title.setObjectName("SectionTitle")
        outer.addWidget(title)

        engine_label = QLabel(self.tr("Engine"))
        engine_label.setObjectName("FieldLabel")
        outer.addWidget(engine_label)
        outer.addWidget(self._build_badge_legend())
        self.engine_group = RadioCardGroup()
        # Scroll just the engine cards — not the whole tab. The card stack
        # can be tall (5 engines, some with a complement sub-picker); a
        # single tab-wide scrollbar buried the bottom controls (languages,
        # DPI, Run). Capping the card region with its own scrollbar keeps
        # those controls pinned and always visible. `stretch=1` lets it eat
        # the slack space and shrink first when the viewport is short.
        self._cards_scroll = QScrollArea()
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cards_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._cards_scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget "
            "{ background: transparent; }")
        self._cards_scroll.viewport().setAutoFillBackground(False)
        self.engine_group.setAutoFillBackground(False)
        self._cards_scroll.setWidget(self.engine_group)
        # Keep at least ~1.5 cards visible before scrolling kicks in. No
        # max height — the region is the sole stretch item below, so it
        # grows to fill all free vertical space (pinning the bottom
        # controls to the bottom) and only scrolls when the cards exceed
        # the available room.
        self._cards_scroll.setMinimumHeight(150)
        outer.addWidget(self._cards_scroll, 1)
        self._populate_engines()
        # Track the active engine so we can detect REAL switches (the
        # currentChanged signal also fires on initial seed / refresh).
        self._last_engine = self.engine_group.current_key()
        self.engine_group.currentChanged.connect(
            lambda *_: self._on_engine_picked()
        )

        outer.addSpacing(4)
        self._lang_label = QLabel(self.tr("Languages (Apple only)"))
        self._lang_label.setObjectName("FieldLabel")
        outer.addWidget(self._lang_label)
        self.lang_input = LanguageTagInput()
        # The picker only affects Apple engines — restrict it to Vision's
        # ACTUAL supported set (queried at runtime; no Greek/Latin).
        try:
            from lib.workers.ocr.apple_vision import supported_languages
            self.lang_input.set_allowed_languages(supported_languages())
        except Exception:
            pass
        outer.addWidget(self.lang_input)
        # Shown instead of the picker for engines that auto-detect language
        # (Cloud OCR / Surya / Paddle) — they ignore the language list.
        self._lang_auto = QLabel(self.tr("Auto — detected by the engine"))
        self._lang_auto.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-style: italic; padding: 2px 0;")
        self._lang_auto.setVisible(False)
        outer.addWidget(self._lang_auto)
        # Seed visibility now that the picker exists (default engine was
        # already chosen in _populate_engines, before this widget existed).
        self._refresh_language_state()

        outer.addSpacing(4)
        dpi_label = QLabel(self.tr("Resize to (DPI)"))
        dpi_label.setObjectName("FieldLabel")
        outer.addWidget(dpi_label)
        outer.addWidget(self._build_dpi_picker())

        outer.addSpacing(8)  # decouple Run row from Languages chips
        # Live-OCR toggle — when on, MainWindow auto-OCRs each branch
        # after a 10 s grace window once it finishes processing.
        outer.addWidget(self._build_live_ocr_toggle())
        run_row = QHBoxLayout()
        run_row.setSpacing(6)

        self.run_btn = QToolButton()
        self.run_btn.setText(self.tr("Run OCR"))
        self.run_btn.setToolTip(
            self.tr("OCR all branches with missing or stale text")
        )
        self.run_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        self.run_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.run_btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Fixed)
        self.run_btn.setMinimumHeight(36)
        # Match the sizing of the Pipeline tab's Edit / Force buttons +
        # the Export tab's PDF / Markdown / Slim buttons so every
        # sidebar action button reads the same.
        self.run_btn.setStyleSheet(
            f"QToolButton {{"
            f"  background-color: {COLOR_BG_BUTTON}; color: {COLOR_FONT_ON_BUTTON};"
            f"  border: 1px solid {COLOR_OUTLINE_BUTTON}; border-radius: 6px;"
            f"  padding: 8px 14px; padding-right: 30px;"
            f"  font-weight: bold; font-size: 13px;"
            f"}}"
            f"QToolButton:hover {{ background-color: {COLOR_BG_BUTTON_HOVER}; }}"
            f"QToolButton:pressed {{ background-color: {COLOR_BG_BUTTON_PRESSED}; }}"
            f"QToolButton:disabled {{ background-color: {COLOR_BG}; color: {COLOR_FONT_DIM};"
            f"  border-color: {COLOR_BG_BUTTON_PRESSED}; }}"
            f"QToolButton::menu-button {{"
            f"  subcontrol-origin: padding;"
            f"  subcontrol-position: right center;"
            f"  background: transparent; width: 24px;"
            f"  border: none; border-left: 1px solid {COLOR_OUTLINE_BUTTON};"
            f"}}"
            f"QToolButton::menu-button:hover {{ background-color: {COLOR_BG_OVERLAY_SOFT}; }}"
            f"QToolButton:focus {{ outline: none; }}"
        )
        try:
            from lib.gui.theme import icon as _icon
            self.run_btn.setIcon(_icon("scan-text", color=COLOR_FONT_ON_BUTTON, size=16))
        except Exception:
            pass
        self.run_btn.setIconSize(QSize(16, 16))

        menu = QMenu(self.run_btn)
        self._act_force = QAction(self.tr("Force OCR on all"), self)
        self._act_force.setToolTip(
            self.tr("Re-run OCR on every branch, including up-to-date ones")
        )
        self._act_missing = QAction(
            self.tr("Only OCR missing content (keep stale)"), self
        )
        self._act_missing.setToolTip(
            self.tr("Skip branches with any prior OCR, even if stale")
        )
        menu.addAction(self._act_force)
        menu.addAction(self._act_missing)
        self.run_btn.setMenu(menu)

        self.run_btn.clicked.connect(self._on_default_run)
        self._act_force.triggered.connect(self._on_force_run)
        self._act_missing.triggered.connect(self._on_missing_run)

        run_row.addWidget(self.run_btn, 1)

        # No sibling "Download model" button — the inline amber pill
        # inside the engine card handles install. When the picked
        # engine is unavailable the Run OCR button is simply disabled
        # with a clarifying tooltip (see ``_refresh_engine_state``).
        # Kept as ``None`` so legacy attribute references stay safe.
        self.dl_btn = None

        outer.addLayout(run_row)

        # Cloud cost estimate — red + bold, directly under the Run button so
        # it can't be missed before money is spent. Hidden unless the Cloud
        # engine is the active card.
        self._cost_lbl = QLabel()
        self._cost_lbl.setStyleSheet(
            f"color: {COLOR_ERROR}; font-size: 11px; font-weight: 700;")
        self._cost_lbl.setWordWrap(True)
        self._cost_lbl.setVisible(False)
        outer.addWidget(self._cost_lbl)

        self.status_lbl = QLabel(
            self.tr("Pipeline running — OCR will unlock when done.")
        )
        self.status_lbl.setStyleSheet(
            f"color: {COLOR_FONT_SECTION_LABEL}; font-size: 11px;"
        )
        self.status_lbl.setWordWrap(True)
        outer.addWidget(self.status_lbl)

        # No trailing stretch — the engine-card scroll area above is the
        # sole expander, so it absorbs all free space and keeps these
        # bottom controls pinned to the bottom.

        # Busy overlay — shown while OCR / pipeline are in flight so the
        # disabled state reads as "busy" rather than "broken".
        self._busy_overlay = _BusyOverlay(self)

        self.setEnabled(False)
        self._refresh_engine_state()

    # ── engine combo ────────────────────────────────────────────────

    # Medal palette — Olympic-style tints for accuracy ranking.
    _MEDAL_GOLD = COLOR_MEDAL_GOLD
    _MEDAL_SILVER = COLOR_MEDAL_SILVER
    _MEDAL_BRONZE = COLOR_MEDAL_BRONZE
    # Visual cues rendered next to the card title — speed (rabbit /
    # turtle) and accuracy (medal = best on OmniDocBench-class
    # benchmarks). Each entry is a list of either a Lucide icon name
    # (default slate tint) or a ``(name, color)`` tuple for tinted
    # icons. RadioCardGroup forwards both forms to ``lib.gui.theme.icon``.
    # 3-tier speed cue: rabbit (fast, green), turtle-amber (medium),
    # turtle-red (slow). Same glyph for medium/slow — only the tint
    # differs — keeps the legend compact.
    _ENGINE_BADGES: dict[str, tuple] = {
        # apple_docs = Vision-fast for Latin + a complement only on the
        # few non-Latin lines, so it reads near-Vision speed with
        # gold-tier accuracy on mixed-script pages.
        "apple_docs":   (("rabbit", COLOR_SUCCESS), ("medal", _MEDAL_GOLD)),
        "apple_vision": (("rabbit", COLOR_SUCCESS), ("medal", _MEDAL_BRONZE)),
        "paddle_vl":    (("turtle", COLOR_WARNING), ("medal", _MEDAL_SILVER)),
        "surya":        (("turtle", COLOR_ERROR), ("medal", _MEDAL_GOLD)),
        # Cloud — network glyph (latency depends on the wire) + gold-tier
        # accuracy. Reads any script, off-device.
        "mistral_cloud": (("cloud", COLOR_PRIMARY), ("medal", _MEDAL_GOLD)),
    }

    # The Apple Document engine is the default pick on a capable Mac.
    _DEFAULT_ENGINE = "apple_docs"
    # Card order: Apple Document leads, then Mistral cloud, the VLMs
    # (paddle, surya), with the legacy Apple Vision engine last.
    _ENGINE_ORDER = ("apple_docs", "mistral_cloud", "paddle_vl", "surya",
                     "apple_vision")
    # Per-engine card icon (bundled SVGs in lib/gui/icons/). Apple engines →
    # apple mark, Mistral → Mistral mark, generic VLMs → an OCR glyph.
    _ENGINE_ICONS = {
        "apple_docs": "apple",
        "apple_vision": "apple",
        "surya": "ocr",
        "paddle_vl": "ocr",
        "mistral_cloud": "mistral",
    }
    # Complement choices offered inside the Apple Document card.
    _COMPLEMENT_CHOICES = (
        ("surya", "Surya (default)"),
        ("paddle_vl", "Paddle"),
        ("none", "None"),
    )

    def _populate_engines(self) -> None:
        from lib.workers.ocr import ENGINE_REGISTRY

        # Card order per _ENGINE_ORDER: Apple doc, Mistral, paddle, surya,
        # then legacy Apple Vision. Unknown engines append after.
        names = [n for n in self._ENGINE_ORDER if n in ENGINE_REGISTRY]
        names += [n for n in ENGINE_REGISTRY if n not in names]

        self._complement_combo = None  # rebuilt below if apple_docs present
        for name in names:
            cls = ENGINE_REGISTRY[name]
            try:
                eng = cls()
            except Exception:
                continue
            title = eng.display if eng.available else self.tr("{name} (not installed)").format(name=eng.display)
            badges = list(self._ENGINE_BADGES.get(name, ()))
            desc = getattr(eng, "description", "") or ""
            extras = None
            extras_always = False
            if name == "apple_docs":
                # Complement dropdown lives inside the Document card; it
                # stays visible whenever the card is selected.
                extras = self._build_complement_picker()
                extras_always = False
            elif name == "mistral_cloud":
                # API-key field + status line live inside the Cloud card —
                # shown only while the card is selected (like the Apple
                # Document complement picker), not permanently.
                extras = self._build_cloud_key_widget(eng.available)
                extras_always = False
            elif not eng.available:
                extras = self._build_install_button(eng.display)
                extras_always = True
            self.engine_group.add_card(name, title, desc,
                                        icon_name=self._ENGINE_ICONS.get(
                                            name, "scan-text"),
                                        title_badges=badges,
                                        extras=extras,
                                        extras_always_visible=extras_always)

        self._apply_engine_gating()
        self._select_default_engine()

    def _select_default_engine(self) -> None:
        """Pick the default engine, skipping disabled / unavailable cards.

        Preference order: the configured default (apple_docs) if enabled →
        first enabled available engine → first enabled card → first card."""
        from lib.workers.ocr import ENGINE_REGISTRY

        def _enabled(key: str) -> bool:
            c = self.engine_group._cards.get(key)
            return bool(c and c.frame.isEnabled())

        # 1. configured default, if enabled.
        if (self._DEFAULT_ENGINE in self.engine_group.keys()
                and _enabled(self._DEFAULT_ENGINE)):
            self.engine_group.set_current_key(self._DEFAULT_ENGINE)
            return
        # 2. first enabled + available engine.
        for key in self.engine_group.keys():
            if not _enabled(key):
                continue
            cls = ENGINE_REGISTRY.get(key)
            try:
                if cls is not None and cls().available:
                    self.engine_group.set_current_key(key)
                    return
            except Exception:
                continue
        # 3. first enabled card, else first card.
        for key in self.engine_group.keys():
            if _enabled(key):
                self.engine_group.set_current_key(key)
                return
        keys = self.engine_group.keys()
        if keys:
            self.engine_group.set_current_key(keys[0])

    def _apply_engine_gating(self) -> None:
        """Disable the Apple cards per platform capability.

        * not macOS → both Apple cards disabled, tooltip "macOS only".
        * macOS, pre-26 (no VNRecognizeDocumentsRequest) → only the
          Document card disabled, tooltip "Requires macOS 26+".
        * macOS 26+ → both enabled.
        """
        from lib.workers.ocr.apple_caps import (
            probe_apple_caps, TOOLTIP_NON_MACOS, TOOLTIP_NEEDS_26,
        )
        caps = probe_apple_caps()
        keys = set(self.engine_group.keys())

        def _gate(key: str, enabled: bool, tooltip: str) -> None:
            if key not in keys:
                return
            self.engine_group.set_card_enabled(key, enabled)
            self.engine_group.set_card_tooltip(
                key, "" if enabled else tooltip)

        if not caps.is_macos:
            _gate("apple_docs", False, self.tr(TOOLTIP_NON_MACOS))
            _gate("apple_vision", False, self.tr(TOOLTIP_NON_MACOS))
        elif not caps.has_documents:
            _gate("apple_docs", False, self.tr(TOOLTIP_NEEDS_26))
            _gate("apple_vision", True, "")
        else:
            _gate("apple_docs", True, "")
            _gate("apple_vision", True, "")

    def _build_complement_picker(self) -> QWidget:
        """The 'Complement engine' dropdown shown inside the Apple
        Document card. Persists to ``KEY_OCR_DEFAULTS["complement"]``."""
        from PySide6.QtWidgets import QComboBox
        from lib.app_data import db as _cfg

        wrap = QWidget()
        wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrap.setAutoFillBackground(False)
        wrap.setStyleSheet("background: transparent;")
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(4)

        # Compact caption: the "(non-Latin scripts)" hint folds in here as
        # a dim suffix so the card can drop the longer help line below.
        cap = QLabel()
        cap.setTextFormat(Qt.TextFormat.RichText)
        cap_title = self.tr("Complement engine")
        cap_hint = self.tr("(non-Latin scripts)")
        cap.setText(
            f"{cap_title} "
            f"<span style='color:{COLOR_FONT_DIM}; font-weight:400;'>"
            f"{cap_hint}</span>"
        )
        cap.setStyleSheet(
            f"color: {COLOR_FONT_SECTION_LABEL}; font-size: 11px; "
            "font-weight: 600;"
        )
        col.addWidget(cap)

        combo = QComboBox()
        combo.setMinimumHeight(26)
        # Only offer complements whose model is actually usable — a missing
        # Surya/Paddle download shouldn't be selectable. "None" always is.
        from lib.workers.ocr import get_engine

        def _avail(k: str) -> bool:
            if k == "none":
                return True
            try:
                return bool(get_engine(k).available)
            except Exception:
                return False
        for key, label in self._COMPLEMENT_CHOICES:
            if _avail(key):
                combo.addItem(self.tr(label), key)
        try:
            with _cfg.session() as _conn:
                defaults = _cfg.get(_conn, _cfg.KEY_OCR_DEFAULTS, {}) or {}
            stored = defaults.get("complement") or "surya"
        except Exception:
            stored = "surya"
        idx = combo.findData(stored)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(lambda _i: self._on_complement_changed())
        # Keep the rationale on hover rather than as a permanent help line —
        # the card stays compact.
        combo.setToolTip(self.tr(
            "Recognises regions Apple Vision can't (non-Latin scripts "
            "like Greek). Surya is most accurate."
        ))
        col.addWidget(combo)

        self._complement_combo = combo
        return wrap

    def current_complement(self) -> str:
        """The picked complement for apple_docs, or "" when the active
        engine isn't apple_docs."""
        if self.engine_group.current_key() != "apple_docs":
            return ""
        combo = getattr(self, "_complement_combo", None)
        if combo is None:
            return "surya"
        return str(combo.currentData() or "surya")

    def _on_complement_changed(self) -> None:
        from lib.app_data import db as _cfg
        combo = getattr(self, "_complement_combo", None)
        if combo is None:
            return
        value = str(combo.currentData() or "surya")
        try:
            with _cfg.session() as conn:
                defaults = _cfg.get(conn, _cfg.KEY_OCR_DEFAULTS, {}) or {}
                defaults["complement"] = value
                _cfg.set(conn, _cfg.KEY_OCR_DEFAULTS, defaults)
        except Exception:
            pass

    def _build_cloud_key_widget(self, sdk_available: bool) -> QWidget:
        """API-key controls inside the Cloud OCR card: a status line plus a
        'Set API key…' button. When the SDK isn't installed, show the pip
        hint instead. The key itself never appears here — it's typed into a
        masked dialog and stored in the OS keychain (or APP_DATA/.env)."""
        wrap = QWidget()
        wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrap.setAutoFillBackground(False)
        wrap.setStyleSheet("background: transparent;")
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(4)

        if not sdk_available:
            hint = QLabel(self.tr(
                "Cloud OCR needs the 'cloud' extra:\n"
                "uv sync --extra cloud"))
            hint.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 10px;")
            hint.setWordWrap(True)
            col.addWidget(hint)
            return wrap

        self._cloud_key_status = QLabel()
        self._cloud_key_status.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 10px;")
        self._cloud_key_status.setWordWrap(True)
        col.addWidget(self._cloud_key_status)

        btn = QPushButton(self.tr("Set API key…"))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {COLOR_SECONDARY_BG_SOFT};"
            f"  color: {COLOR_FONT_PRIMARY};"
            f"  border: 1px solid {COLOR_SECONDARY_BORDER};"
            f"  border-radius: 6px; padding: 4px 12px;"
            f"  font-weight: 600; font-size: 11px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {COLOR_SECONDARY_BG_HOVER}; }}"
        )
        try:
            from lib.gui.theme import icon as _icon
            btn.setIcon(_icon("key-round", color=COLOR_FONT_PRIMARY, size=12))
        except Exception:
            pass
        btn.clicked.connect(self._prompt_cloud_key)
        col.addWidget(btn)

        self._refresh_cloud_key_status()
        return wrap

    def _refresh_cloud_key_status(self, *, probe_keychain: bool = False) -> None:
        """Update the cloud-key status line.

        ``probe_keychain`` defaults to False so the common path (startup,
        engine seeding) never reads the OS keychain — that read pops a system
        password prompt the user hasn't asked for. We probe the keychain only
        when the user actively engages Cloud OCR (picks the cloud engine, or
        opens the key dialog). Until then, a key that lives only in the
        keychain shows as a neutral 'click to check' hint rather than a prompt."""
        lbl = getattr(self, "_cloud_key_status", None)
        if lbl is None:
            return
        try:
            from lib.app_data.secrets import mistral_key_location
            where = mistral_key_location(include_keychain=probe_keychain)
        except Exception:
            where = ""
        msgs = {
            "env": self.tr("Key set (from MISTRAL_API_KEY env var)."),
            "keychain": self.tr("Key stored in your OS keychain."),
            "env_file": self.tr("Key stored in APP_DATA/.env (less secure)."),
        }
        if where:
            lbl.setText("✓ " + msgs.get(where, self.tr("API key set.")))
            lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 10px;")
        elif not probe_keychain:
            # Didn't look in the keychain (no prompt). A key may still be there.
            lbl.setText(self.tr("Select Cloud OCR to use a key from your keychain, "
                                "or add one below."))
            lbl.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 10px;")
        else:
            lbl.setText(self.tr("No API key set — pages can't be sent yet."))
            lbl.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 10px;")

    def _prompt_cloud_key(self) -> None:
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout,
        )
        try:
            from lib.app_data import app_data_dir
            env_path = str(app_data_dir() / ".env")
        except Exception:
            env_path = "<APP_DATA>/.env"

        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Mistral API key"))
        v = QVBoxLayout(dlg)
        v.setSpacing(8)
        main = QLabel(self.tr(
            "Paste your Mistral API key. It is stored in your OS keychain. "
            "Leave empty to remove."))
        main.setWordWrap(True)
        v.addWidget(main)
        edit = QLineEdit()
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        edit.setMinimumWidth(440)
        v.addWidget(edit)
        # Smaller, dimmed description — the supersede order.
        note = QLabel(self.tr(
            "The keychain key is superseded by the MISTRAL_API_KEY "
            "environment variable (shell), or by {env_path}."
        ).format(env_path=env_path))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        v.addWidget(note)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        key = edit.text()
        try:
            from lib.app_data.secrets import set_mistral_api_key
            where = set_mistral_api_key(key)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, self.tr("Key storage failed"), str(e))
            return
        self._refresh_cloud_key_status(probe_keychain=True)
        if where == "env_file":
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, self.tr("API key saved"),
                self.tr("No OS keychain was available — the key was saved "
                        "to {env_path} instead.").format(env_path=env_path))

    def _build_install_button(self, engine_display: str) -> QWidget:
        """Compact amber 'Install' pill placed inside an engine card
        when its weights aren't downloaded yet. Clicking opens the
        model downloader (same target as the bigger ``dl_btn`` in the
        run row below)."""
        wrap = QWidget()
        # Transparent wrap so the card's tinted bg shows through —
        # default QWidget palette would paint a Base-colored slab here.
        wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrap.setAutoFillBackground(False)
        wrap.setStyleSheet("background: transparent;")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addStretch(1)
        btn = QPushButton(
            self.tr("Install {name} model").format(name=engine_display)
        )
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {COLOR_SECONDARY_BG_SOFT};"
            f"  color: {COLOR_MEDAL_GOLD};"
            f"  border: 1px solid {COLOR_SECONDARY_BORDER};"
            f"  border-radius: 6px;"
            f"  padding: 4px 12px;"
            f"  font-weight: 600;"
            f"  font-size: 11px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {COLOR_SECONDARY_BG_HOVER}; }}"
        )
        try:
            from lib.gui.theme import icon as _icon
            btn.setIcon(_icon("download", color=COLOR_MEDAL_GOLD, size=12))
        except Exception:
            pass
        btn.clicked.connect(self._open_model_downloader)
        row.addWidget(btn)
        return wrap

    # Candidate DPIs offered by the OCR-DPI picker. Caller filters out
    # any entry above the project's chosen-layout DPI cap (upscaling
    # beyond the source resolution gains no information for OCR).
    _OCR_DPI_CHOICES = (72, 100, 150, 200, 300)

    def _build_badge_legend(self) -> QWidget:
        """Mini legend rendered at the bottom of the OCR sidebar tab —
        same Lucide icons as the engine cards, labelled."""
        try:
            from lib.gui.theme import icon as _icon
        except Exception:
            _icon = None

        wrap = QWidget()
        wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrap.setAutoFillBackground(False)
        wrap.setStyleSheet("background: transparent;")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(10)
        row.addStretch(1)

        def _add_icon(icon_name: str, color: str = COLOR_FONT_SECTION_LABEL) -> None:
            ico_lbl = QLabel()
            ico_lbl.setFixedSize(18, 18)
            ico_lbl.setStyleSheet("background: transparent;")
            if _icon is not None:
                try:
                    ico_lbl.setPixmap(
                        _icon(icon_name, color=color, size=16)
                        .pixmap(16, 16)
                    )
                except Exception:
                    pass
            row.addWidget(ico_lbl)

        def _add_text(text: str) -> None:
            txt = QLabel(text)
            txt.setStyleSheet(
                f"color: {COLOR_FONT_DIM}; font-size: 11px;"
            )
            row.addWidget(txt)

        # 3-tier speed legend reads slow → fast (left → right).
        _add_icon("turtle", COLOR_ERROR)
        _add_icon("turtle", COLOR_WARNING)
        _add_icon("rabbit", COLOR_SUCCESS)
        _add_text(self.tr("speed"))
        _add_text("  ·  ")
        # 3-tier accuracy: bronze → silver → gold (low → high).
        _add_icon("medal", self._MEDAL_BRONZE)
        _add_icon("medal", self._MEDAL_SILVER)
        _add_icon("medal", self._MEDAL_GOLD)
        _add_text(self.tr("accuracy"))

        row.addStretch(1)
        return wrap

    def _build_live_ocr_toggle(self) -> QWidget:
        """Compact toggle row: '⚡ Live OCR' check + tiny help label.
        Persists to ``KEY_LIVE_OCR``; emits ``live_ocr_toggled`` so the
        MainWindow can install (or tear down) the branch_ready hook."""
        from PySide6.QtWidgets import QCheckBox
        from lib.app_data import db as _cfg

        wrap = QWidget()
        wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrap.setAutoFillBackground(False)
        wrap.setStyleSheet("background: transparent;")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        cb = QCheckBox(self.tr("Live OCR"))
        cb.setToolTip(
            self.tr(
                "Auto-OCR every page ~10 s after it finishes processing. "
                "The 10 s grace window leaves time to delete a scan before "
                "OCR cost is spent on it."
            )
        )
        cb.setStyleSheet(
            f"QCheckBox {{ color: {COLOR_FONT_MUTED}; font-size: 12px; }}"
        )
        try:
            with _cfg.session() as _conn:
                stored = bool(_cfg.get(_conn, _cfg.KEY_LIVE_OCR, False))
        except Exception:
            stored = False
        cb.setChecked(stored)

        def _on_toggled(checked: bool) -> None:
            try:
                with _cfg.session() as conn:
                    _cfg.set(conn, _cfg.KEY_LIVE_OCR, bool(checked))
            except Exception:
                pass
            self.live_ocr_toggled.emit(bool(checked))

        cb.toggled.connect(_on_toggled)
        row.addWidget(cb)
        row.addStretch(1)
        self._live_ocr_cb = cb
        return wrap

    def is_live_ocr_on(self) -> bool:
        return bool(getattr(self, "_live_ocr_cb", None)
                     and self._live_ocr_cb.isChecked())

    def _build_dpi_picker(self) -> QWidget:
        """Single OCR-DPI dropdown applied to every engine. Persists to
        ``KEY_OCR_DPI``; each engine's ``recognize`` calls
        ``engine.resolve_ocr_dpi()`` to read it back. Choices are capped
        at ``min(project_layout_dpi, 200)`` so we never offer a DPI
        that would up-sample the input."""
        from PySide6.QtWidgets import QComboBox
        from lib.app_data import db as _cfg

        wrap = QWidget()
        wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrap.setAutoFillBackground(False)
        wrap.setStyleSheet("background: transparent;")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        cap = self._project_page_dpi_cap()  # ≤ 200 — see helper
        combo = QComboBox()
        combo.setMinimumHeight(28)
        for dpi in self._OCR_DPI_CHOICES:
            if dpi <= cap:
                combo.addItem(self.tr("{dpi} DPI").format(dpi=dpi), dpi)
        try:
            with _cfg.session() as _conn:
                stored = int(_cfg.get(_conn, _cfg.KEY_OCR_DPI, 150) or 150)
        except Exception:
            stored = 150
        # Clamp stored value to the offered range.
        stored = min(stored, cap)
        idx = combo.findData(stored)
        if idx < 0 and combo.count() > 0:
            idx = combo.count() - 1
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.setToolTip(
            self.tr(
                "Pages get downsampled to this DPI before being sent to "
                "the engine. 72 = fastest, 150 = balanced default, 200 = "
                "sharper. Capped at the project's layout DPI — upsampling "
                "above the source DPI gains no information."
            )
        )

        def _on_changed(_i: int) -> None:
            try:
                value = int(combo.currentData() or 150)
                with _cfg.session() as conn:
                    _cfg.set(conn, _cfg.KEY_OCR_DPI, value)
            except Exception:
                pass

        combo.currentIndexChanged.connect(_on_changed)
        row.addWidget(combo, 1)
        return wrap

    def _project_page_dpi_cap(self) -> int:
        """Returns the upper bound for the OCR-DPI dropdown, clamped
        to 300. Reads the highest chosen-layout image DPI from the
        active project; falls back to 300 when no project is open
        (Settings tab / first launch) so the full picker still shows."""
        cap = 300
        w = self.window()
        db_path = getattr(w, "db_path", None) if w is not None else None
        if not db_path:
            return cap
        try:
            from lib.storage.db import db_session
            with db_session(str(db_path)) as conn:
                row = conn.execute("""
                    SELECT COALESCE(MAX(i.dpi), 0) AS max_dpi
                      FROM images i
                      JOIN nodes n ON n.image_id = i.id
                      JOIN branches b ON b.chosen_node_id = n.id
                      JOIN scans s ON s.id = b.scan_id
                     WHERE s.deleted_at IS NULL
                       AND b.trashed_at IS NULL
                """).fetchone()
            max_dpi = int(row["max_dpi"] or 0) if row else 0
        except Exception:
            max_dpi = 0
        if max_dpi <= 0:
            return cap
        return min(cap, max_dpi)

    def _on_engine_picked(self) -> None:
        """Slot for ``engine_group.currentChanged``. Fires
        ``engine_changed`` only when the active engine key actually
        moves (the signal also fires when ``set_current_key`` runs
        during initial seed)."""
        self._refresh_engine_state()
        self._refresh_cost_estimate()
        self._refresh_language_state()
        new_key = self.engine_group.current_key()
        if getattr(self, "_cloud_key_status", None) is not None:
            # Probe the keychain only when the user actively switches TO the
            # cloud engine — never on the initial seed (avoids a startup
            # password prompt). See _refresh_cloud_key_status.
            engaging_cloud = (new_key == "mistral_cloud"
                              and new_key != self._last_engine)
            self._refresh_cloud_key_status(probe_keychain=engaging_cloud)
        if new_key and new_key != self._last_engine:
            self._last_engine = new_key
            self.engine_changed.emit(new_key)

    _APPLE_ENGINES = ("apple_vision", "apple_docs")

    def _refresh_language_state(self) -> None:
        """Language selection only affects the Apple engines. For the
        auto-detecting engines (Cloud OCR / Surya / Paddle) hide the picker
        and show 'Auto' instead of the language tags."""
        li = getattr(self, "lang_input", None)
        if li is None:
            return
        is_apple = self.engine_group.current_key() in self._APPLE_ENGINES
        li.setVisible(is_apple)
        if getattr(self, "_lang_auto", None) is not None:
            self._lang_auto.setVisible(not is_apple)

    def _refresh_cost_estimate(self) -> None:
        """Show the red cloud-cost estimate when the Cloud engine is active.

        The estimate is ``pending pages × list price`` (mistral-ocr-latest,
        ~1000 pages/$). Mistral has no balance API, so we can't show live
        credit — the tooltip points at the console instead."""
        lbl = getattr(self, "_cost_lbl", None)
        if lbl is None:
            return
        if self.engine_group.current_key() != "mistral_cloud":
            lbl.setVisible(False)
            return
        from lib.workers.ocr.mistral_cloud import (
            PRICE_PER_PAGE_USD, CONSOLE_URL)
        n = int(getattr(self, "_pending_total", 0) or 0)
        cost = n * PRICE_PER_PAGE_USD
        if n <= 0:
            lbl.setText(self.tr("⚠ Cloud OCR — pages are uploaded to Mistral "
                                "(billed per page)."))
        else:
            lbl.setText(self.tr(
                "⚠ ≈ ${cost:.2f} — {n} page(s) uploaded to Mistral cloud"
            ).format(cost=cost, n=n))
        lbl.setToolTip(self.tr(
            "Estimate at ~1000 pages/$ (mistral-ocr-latest). Mistral exposes "
            "no account-balance API — check remaining credit at {url}"
        ).format(url=CONSOLE_URL))
        lbl.setVisible(True)

    def _refresh_engine_state(self) -> None:
        """Toggle the Run OCR button's enabled state based on whether
        the picked engine has its weights ready. When unavailable the
        button stays in place but greyed out — the install affordance
        lives inside the engine card (amber pill)."""
        from lib.workers.ocr import ENGINE_REGISTRY
        name = self.engine_group.current_key()
        available = False
        display = name or self.tr("Engine")
        if name:
            cls = ENGINE_REGISTRY.get(name)
            if cls is not None:
                try:
                    eng = cls()
                    available = bool(eng.available)
                    display = getattr(eng, "display", name)
                except Exception:
                    available = False
        if available:
            self.run_btn.setEnabled(True)
            self.run_btn.setToolTip(
                self.tr("OCR all branches with missing or stale text")
            )
        else:
            self.run_btn.setEnabled(False)
            self.run_btn.setToolTip(
                self.tr(
                    "{name} isn't installed yet — use the inline "
                    "Install button on the card to fetch its weights."
                ).format(name=display)
            )

    def _open_model_downloader(self) -> None:
        w = self.window()
        opener = getattr(w, "_open_model_downloader", None)
        if callable(opener):
            opener()

    def refresh_engines(self) -> None:
        """Rebuild the engine cards in place — re-probes each engine's
        ``available`` flag so the inline 'Install' pill drops as soon
        as the user's download finishes. MainWindow calls this after
        the model downloader dialog closes."""
        current = self.engine_group.current_key()
        # Wipe + rebuild. Cheap — at most 2 cards today.
        parent_layout = self.engine_group.parentWidget().layout()
        idx = parent_layout.indexOf(self.engine_group)
        from lib.gui.sidebar.widgets import RadioCardGroup
        self.engine_group.deleteLater()
        self.engine_group = RadioCardGroup()
        if idx >= 0:
            parent_layout.insertWidget(idx, self.engine_group)
        else:
            parent_layout.addWidget(self.engine_group)
        self._populate_engines()
        # Restore the prior selection only if it survived and stays
        # enabled; otherwise keep the gated default _populate_engines
        # already picked.
        if current and current in self.engine_group.keys():
            c = self.engine_group._cards.get(current)
            if c is not None and c.frame.isEnabled():
                self.engine_group.set_current_key(current)
        self.engine_group.currentChanged.connect(
            lambda *_: self._refresh_engine_state()
        )
        self._refresh_engine_state()

    # ── public API (mirrors OcrFrame) ───────────────────────────────

    def set_pipeline_running(self, running: bool) -> None:
        if running:
            self.setEnabled(False)
            self.status_lbl.setText(
                self.tr("Pipeline running — OCR will unlock when done.")
            )
            self._busy_overlay.set_caption(self.tr("Pipeline running…"))
            self._busy_overlay.start()
        else:
            self.setEnabled(True)
            self.status_lbl.setText(self.tr("Ready."))
            self._busy_overlay.stop()

    def set_ocr_running(self, running: bool) -> None:
        if running:
            self.setEnabled(False)
            self.status_lbl.setText(self.tr("OCR running…"))
            self._busy_overlay.set_caption(self.tr("OCR running…"))
            self._busy_overlay.start()
        else:
            self.setEnabled(True)
            self.status_lbl.setText(self.tr("Ready."))
            self._busy_overlay.stop()

    def set_pending_count(self, missing: int, stale: int) -> None:
        total = int(missing) + int(stale)
        self._pending_total = total
        self._refresh_cost_estimate()
        # Don't disable when nothing's pending — a default run just toasts
        # "OCR already complete"; re-running is via Force all. Enabled state
        # tracks engine availability only (set in _refresh_engine_state).
        if total == 0:
            self.status_lbl.setText(
                self.tr("All branches OCR'd — nothing to do.")
            )
        else:
            bits = []
            if missing:
                bits.append(self.tr("{n} missing").format(n=missing))
            if stale:
                bits.append(self.tr("{n} stale").format(n=stale))
            self.status_lbl.setText(
                self.tr("{summary} · ready.").format(summary=", ".join(bits))
            )

    # ── click handlers ──────────────────────────────────────────────

    def _emit(self, mode: str) -> None:
        engine = self.engine_group.current_key()
        langs = self.lang_input.tags()
        if not engine:
            return
        self.run_requested.emit(engine, langs, mode, self.current_complement())

    def _toast(self, msg: str) -> None:
        fn = getattr(self.window(), "toast", None)
        if callable(fn):
            fn(msg)

    def _existing_ocr_summary(self) -> tuple[int, list[str]]:
        """(# branches with a done OCR run, distinct engine names)."""
        db = getattr(self.window(), "db_path", None)
        if not db:
            return 0, []
        try:
            from lib.storage.db import db_session
            with db_session(str(db)) as conn:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT scan_id || '/' || branch_path) AS n,"
                    " GROUP_CONCAT(DISTINCT engine) AS engs"
                    " FROM ocr_runs WHERE status = 'done'").fetchone()
            engs = [e for e in (row["engs"] or "").split(",") if e]
            return int(row["n"] or 0), engs
        except Exception:
            return 0, []

    def _on_default_run(self) -> None:
        if int(getattr(self, "_pending_total", 0) or 0) <= 0:
            self._toast(self.tr(
                "OCR already complete — use “Force all” to re-run."))
            return
        self._emit(self.MODE_DEFAULT)

    def _on_force_run(self) -> None:
        n, engines = self._existing_ocr_summary()
        if n > 0:
            from PySide6.QtWidgets import QMessageBox
            eng = ", ".join(engines) if engines else self.tr("an engine")
            reply = QMessageBox.warning(
                self, self.tr("Force OCR on all"),
                self.tr("{n} page(s) already OCR'd with {eng} will be erased "
                        "and re-processed with the selected engine. "
                        "Continue?").format(n=n, eng=eng),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._emit(self.MODE_FORCE)

    def _on_missing_run(self) -> None:
        self._emit(self.MODE_MISSING)
