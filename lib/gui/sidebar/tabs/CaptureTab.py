# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Capture sidebar tab.

Hosts the webcam-mode controls: a compact preview slot, calibration
buttons, zoom slider, rotation row, and voice / SIFT toggles. The
widget is *passive*: MainWindow wires its existing handlers
(``calibrate_camera``, ``calibrate_dpi``, ``_on_freehand_clicked``,
``_apply_selected_transform``, voice toggle, zoom slots) onto the
public attributes exposed here. Mirrors the previous OcrFrame /
ExportTab pattern.

A scaled webcam frame may be set on ``preview_label`` from
``WebcamThread.change_pixmap_signal`` when the MainWindow chooses to
mirror the live feed inside the tab. When no camera thread is
running, ``set_no_camera()`` paints the placeholder text.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QActionGroup, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.colors import (
    COLOR_BG_BUTTON,
    COLOR_BG_BUTTON_CHECKED,
    COLOR_BG_BUTTON_HOVER,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_BG_TOGGLE,
    COLOR_BG_TOGGLE_ON,
    COLOR_BG_VIDEO,
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_ON_BUTTON,
    COLOR_FONT_ON_TOGGLE,
    COLOR_FONT_PRIMARY,
    COLOR_OUTLINE_BUTTON,
    COLOR_OUTLINE_BUTTON_STRONG,
    COLOR_OUTLINE_SUBTLE,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SUCCESS_BORDER,
)


_BTN_QSS = f"""
QPushButton {{
    background-color: {COLOR_BG_BUTTON}; color: {COLOR_FONT_ON_BUTTON};
    border: 1px solid {COLOR_OUTLINE_BUTTON}; border-radius: 6px;
    padding: 4px 10px;
}}
QPushButton:hover  {{ background-color: {COLOR_BG_BUTTON_HOVER}; }}
QPushButton:checked {{ background-color: {COLOR_BG_BUTTON_CHECKED}; color: {COLOR_FONT_INVERSE}; border-color: {COLOR_PRIMARY_HOVER}; }}
QPushButton:disabled {{ color: {COLOR_FONT_DIM}; }}
"""

_TOGGLE_QSS = f"""
QPushButton {{
    background-color: {COLOR_BG_TOGGLE}; color: {COLOR_FONT_ON_TOGGLE};
    border: 1px solid {COLOR_OUTLINE_BUTTON_STRONG}; border-radius: 16px;
    padding: 4px 12px; font-weight: 600;
}}
QPushButton:checked {{ background-color: {COLOR_BG_TOGGLE_ON}; border-color: {COLOR_SUCCESS_BORDER}; }}
"""

class _VoiceSplitButton(QFrame):
    """Cohesive split button: one rounded pill holding a flat on/off toggle
    (left) and a flat ⌄ chevron (right) that opens the engine menu. A
    QToolButton + QSS split renders its menu-button as a detached box on
    macOS, so we compose two flat buttons inside a single styled frame.

    Exposes a QAbstractButton-like surface (`toggled`, `isChecked`,
    `setChecked`) so MainWindow drives it like the old QToolButton."""

    toggled = Signal(bool)
    engine_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("VoiceSplit")
        self._active_text = "Deactivate voice control"
        self._inactive_text = "Activate voice control"

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._toggle = QPushButton(self._inactive_text)
        self._toggle.setObjectName("VoiceToggle")
        self._toggle.setCheckable(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)
        self._toggle.toggled.connect(self._on_toggled)

        self._chevron = QToolButton()
        self._chevron.setObjectName("VoiceChevron")
        self._chevron.setText("▾")
        self._chevron.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chevron.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._chevron.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._menu = QMenu(self._chevron)
        self._chevron.setMenu(self._menu)
        self._group = QActionGroup(self)
        self._group.setExclusive(True)

        row.addWidget(self._toggle, 1)
        row.addWidget(self._chevron)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._restyle()

    # — toggle relay —
    def _on_toggled(self, on: bool) -> None:
        self._toggle.setText(self._active_text if on else self._inactive_text)
        self._restyle()
        self.toggled.emit(on)

    def isChecked(self) -> bool:
        return self._toggle.isChecked()

    def setChecked(self, on: bool) -> None:
        self._toggle.setChecked(on)

    def set_labels(self, active: str, inactive: str) -> None:
        self._active_text, self._inactive_text = active, inactive
        self._toggle.setText(active if self._toggle.isChecked() else inactive)

    # — engine menu —
    def set_voice_engines(self, engines, current=None) -> None:
        self._menu.clear()
        for a in list(self._group.actions()):
            self._group.removeAction(a)
        for key, label in engines:
            act = self._menu.addAction(label)
            act.setCheckable(True)
            act.setData(key)
            act.setChecked(key == current)
            self._group.addAction(act)
            act.triggered.connect(
                lambda _checked=False, k=key: self.engine_changed.emit(k))
        # One engine → nothing to pick; drop the chevron so the pill is plain.
        self._chevron.setVisible(len(engines) > 1)
        self._restyle()

    def _restyle(self) -> None:
        on = self._toggle.isChecked()
        bg = COLOR_BG_TOGGLE_ON if on else COLOR_BG_TOGGLE
        border = COLOR_SUCCESS_BORDER if on else COLOR_OUTLINE_BUTTON_STRONG
        divider = (f"#VoiceChevron {{ border-left: 1px solid {border}; }}"
                   if self._chevron.isVisible() else "")
        self.setStyleSheet(
            f"#VoiceSplit {{ background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 16px; }}"
            f"#VoiceToggle {{ background: transparent; border: none; "
            f"color: {COLOR_FONT_ON_TOGGLE}; font-weight: 600; padding: 4px 12px; }}"
            f"#VoiceChevron {{ background: transparent; border: none; "
            f"color: {COLOR_FONT_ON_TOGGLE}; font-size: 15px; padding: 0 12px; }}"
            f"#VoiceChevron::menu-indicator {{ image: none; width: 0; }}"
            + divider
        )


class _ClickableLabel(QLabel):
    """A QLabel that emits ``clicked`` on a left mouse release — used for the
    DPI readout so the user can click it to set the value manually."""
    clicked = Signal()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802 (Qt override)
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(ev)


class CaptureTab(QWidget):
    """Container for capture-mode controls. Public attrs only — wiring
    is done by MainWindow so the chain logic stays in one place."""

    voice_engine_changed = Signal(str)  # engine key picked from the ▾ menu

    PREVIEW_W = 280
    PREVIEW_H = 158   # 16:9

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel(self.tr("Capture"))
        title.setObjectName("SectionTitle")
        outer.addWidget(title)

        # ── Preview slot ────────────────────────────────────────────
        # Fixed width, but the HEIGHT tracks the live frame's aspect ratio
        # (set per-frame in `set_preview_pixmap`) so the full camera image
        # shows with no cropping and no AR distortion, whatever the device
        # delivers (16:9, 4:3, rotated portrait…).
        self.preview_label = QLabel()
        self.preview_label.setObjectName("CapturePreview")
        self.preview_label.setFixedWidth(self.PREVIEW_W)
        self.preview_label.setMinimumHeight(self.PREVIEW_H)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            f"QLabel#CapturePreview {{"
            f"  background: {COLOR_BG_VIDEO}; color: {COLOR_FONT_DIM};"
            f"  border: 1px dashed {COLOR_OUTLINE_BUTTON_STRONG}; border-radius: 6px;"
            f"}}"
        )
        self.preview_label.setText(self.tr("No camera"))
        outer.addWidget(self.preview_label, 0, Qt.AlignmentFlag.AlignHCenter)

        # ── Camera + format selectors ───────────────────────────────
        # MainWindow populates these (`_populate_capture_devices`) and
        # restarts the webcam on change. The format list exposes a
        # Continuity Camera's several modes so the user can pick the full,
        # un-cropped view if the auto-pick (widest FOV / full sensor) is off.
        cam_row = QHBoxLayout()
        cam_row.setSpacing(6)
        cam_lbl = QLabel(self.tr("Camera"))
        cam_lbl.setObjectName("FieldLabel")
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip(self.tr("Capture device"))
        cam_row.addWidget(cam_lbl)
        cam_row.addWidget(self.camera_combo, 1)
        outer.addLayout(cam_row)

        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(6)
        fmt_lbl = QLabel(self.tr("Format"))
        fmt_lbl.setObjectName("FieldLabel")
        self.format_combo = QComboBox()
        self.format_combo.setToolTip(
            self.tr("Resolution / field of view. Widest is listed first."))
        fmt_row.addWidget(fmt_lbl)
        fmt_row.addWidget(self.format_combo, 1)
        outer.addLayout(fmt_row)

        # ── Current DPI readout ─────────────────────────────────────
        # Always-visible scan resolution at the live zoom. MainWindow
        # refreshes it on calibration + zoom changes (`set_dpi`).
        self.dpi_label = _ClickableLabel(self.tr("DPI: —"))
        self.dpi_label.setObjectName("DpiReadout")
        self.dpi_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dpi_label.setToolTip(
            self.tr("Effective scan resolution at the current zoom. "
                    "Click to set it manually, or calibrate with a "
                    "credit-card-sized object."))
        outer.addWidget(self.dpi_label, 0, Qt.AlignmentFlag.AlignHCenter)

        # ── Deactivate camera ───────────────────────────────────────
        # Hidden by default; MainWindow shows + wires it only when the
        # capture session was activated late (open-from-disk projects).
        self.btn_deactivate = QPushButton(self.tr("Deactivate camera"))
        self.btn_deactivate.setStyleSheet(_BTN_QSS)
        self.btn_deactivate.setToolTip(
            self.tr("Stop the camera and return to the picker.")
        )
        self.btn_deactivate.hide()
        outer.addWidget(self.btn_deactivate, 0, Qt.AlignmentFlag.AlignHCenter)

        # ── Calibration buttons ─────────────────────────────────────
        cal_row = QHBoxLayout()
        cal_row.setSpacing(6)
        self.btn_full_calibrate = QPushButton(self.tr("Full Calibration"))
        self.btn_full_calibrate.setStyleSheet(_BTN_QSS)
        self.btn_full_calibrate.setToolTip(
            self.tr("Print A4_chessboard.pdf, hold it in view, run.")
        )
        self.btn_dpi_calibrate = QPushButton(self.tr("Calibrate DPI"))
        self.btn_dpi_calibrate.setStyleSheet(_BTN_QSS)
        self.btn_dpi_calibrate.setToolTip(
            self.tr("Use a credit card to anchor real-world scale.")
        )
        cal_row.addWidget(self.btn_full_calibrate, 1)
        cal_row.addWidget(self.btn_dpi_calibrate, 1)
        outer.addLayout(cal_row)

        # ── Zoom ────────────────────────────────────────────────────
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        zoom_lbl = QLabel(self.tr("Zoom"))
        zoom_lbl.setObjectName("FieldLabel")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(100, 100)   # init 1.00×–1.00×
        self.zoom_slider.setValue(100)
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setDecimals(2)
        self.zoom_spin.setRange(1.0, 1.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setValue(1.0)
        self.zoom_spin.setSuffix("×")
        zoom_row.addWidget(zoom_lbl)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_spin)
        outer.addLayout(zoom_row)

        # ── Rotation row ────────────────────────────────────────────
        outer.addWidget(self._field_label(self.tr("Rotation")))
        rot_row = QHBoxLayout()
        rot_row.setSpacing(6)
        self.transform_combo = QComboBox()
        # MainWindow seeds the items (must match its `_transform_items`
        # tuple so the index → transform mapping stays in sync). We
        # don't pre-populate here on purpose.
        self.btn_apply_transform = QPushButton(self.tr("Apply"))
        self.btn_apply_transform.setStyleSheet(_BTN_QSS)
        self.btn_apply_transform.setEnabled(False)
        self.transform_combo.currentIndexChanged.connect(
            lambda i: self.btn_apply_transform.setEnabled(i > 0)
        )
        rot_row.addWidget(self.transform_combo, 1)
        rot_row.addWidget(self.btn_apply_transform)
        outer.addLayout(rot_row)

        # ── Voice + SIFT toggles ────────────────────────────────────
        outer.addWidget(self._field_label(self.tr("Hands-free triggers")))
        tog_row = QHBoxLayout()
        tog_row.setSpacing(6)
        # Toggle button for voice on/off. The widget keeps a ▾ engine menu for
        # generality, but voice is Vosk-only now, so with one engine the
        # chevron auto-hides and it renders as a plain toggle.
        self.btn_voice = _VoiceSplitButton()
        self.btn_voice.set_labels(self.tr("Deactivate voice control"),
                                  self.tr("Activate voice control"))
        self.btn_voice.setToolTip(
            self.tr("Toggle voice commands (say 'photo'). Use ⌄ to choose the engine.")
        )
        self.btn_voice.engine_changed.connect(self.voice_engine_changed)

        self.btn_freehand = QPushButton(self.tr("Hands-free (SIFT)"))
        self.btn_freehand.setCheckable(True)
        self.btn_freehand.setStyleSheet(_TOGGLE_QSS)
        self.btn_freehand.setToolTip(
            self.tr("Register a small pattern; briefly covering it triggers a capture.")
        )
        # SIFT freehand is still experimental — hidden for now (wiring kept
        # intact so it can be re-enabled by dropping the setVisible call).
        self.btn_freehand.setVisible(False)
        tog_row.addWidget(self.btn_voice, 1)
        tog_row.addWidget(self.btn_freehand, 1)
        outer.addLayout(tog_row)

        # ── Voice command reference — the universal words + what they do.
        # Populated by MainWindow (`set_voice_command_legend`) from the
        # `voicecontrols` config so it always matches the live grammar.
        self.voice_cmd_legend = QLabel("")
        self.voice_cmd_legend.setWordWrap(True)
        self.voice_cmd_legend.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px; padding: 2px 0;"
        )
        outer.addWidget(self.voice_cmd_legend)

        # ── Voice transcript — last ~10 recognized words (live feedback).
        self.voice_transcript = QLabel(self.tr("Heard: —"))
        self.voice_transcript.setWordWrap(True)
        self.voice_transcript.setMinimumHeight(36)
        self.voice_transcript.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.voice_transcript.setStyleSheet(
            f"QLabel {{"
            f"  background: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 1px solid {COLOR_OUTLINE_SUBTLE};"
            f"  border-radius: 4px;"
            f"  padding: 6px 8px;"
            f"  color: {COLOR_FONT_MUTED};"
            f"  font-size: 11px;"
            f"}}"
        )
        self.voice_transcript.setVisible(False)
        outer.addWidget(self.voice_transcript)

        # Push the keyboard section + capture button to the bottom.
        outer.addStretch(1)

        # ── Keyboard shortcuts (bottom) — populated by MainWindow ────
        outer.addWidget(self._field_label(self.tr("Keyboard shortcuts")))
        self.shortcut_legend = QLabel("—")
        self.shortcut_legend.setWordWrap(True)
        self.shortcut_legend.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px;"
        )
        outer.addWidget(self.shortcut_legend)

        # Primary capture button — same action as SPACE / voice "photo".
        self.btn_capture = QPushButton(self.tr("Capture"))
        self.btn_capture.setMinimumHeight(42)
        self.btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_capture.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_PRIMARY}; "
            f"color: {COLOR_FONT_INVERSE}; border: none; border-radius: 8px; "
            f"font-weight: 600; font-size: 14px; padding: 8px; }}"
            f"QPushButton:hover {{ background-color: {COLOR_PRIMARY_HOVER}; }}"
        )
        outer.addWidget(self.btn_capture)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    # ── Preview helpers ────────────────────────────────────────────

    def set_preview_pixmap(self, pix: QPixmap) -> None:
        """Display ``pix`` at the slot's fixed width, with the label height
        driven by the frame's aspect ratio so the whole frame shows — no
        cropping, no aspect-ratio change. Use from a signal hooked up to
        ``WebcamThread.change_pixmap_signal``."""
        if pix is None or pix.isNull() or pix.width() <= 0:
            self.set_no_camera()
            return
        w = self.PREVIEW_W
        target_h = max(1, round(w * pix.height() / pix.width()))
        if self.preview_label.height() != target_h:
            self.preview_label.setFixedHeight(target_h)
        scaled = pix.scaled(
            w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def set_no_camera(self) -> None:
        self.preview_label.clear()
        self.preview_label.setText(self.tr("No camera"))

    def set_dpi(self, dpi: float, calibrated: bool, manual: bool = False) -> None:
        """Update the DPI readout. ``calibrated`` distinguishes a measured
        value from the uncalibrated default (shown dimmer + flagged);
        ``manual`` flags a user-typed value (shown bold like calibrated, but
        tagged ``(manual)`` since no camera calibration backs it)."""
        if calibrated:
            self.dpi_label.setText(self.tr("DPI: {dpi:.0f}").format(dpi=dpi))
            self.dpi_label.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-weight: 600;")
        elif manual:
            self.dpi_label.setText(
                self.tr("DPI: {dpi:.0f} (manual)").format(dpi=dpi))
            self.dpi_label.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-weight: 600;")
        else:
            self.dpi_label.setText(
                self.tr("DPI: {dpi:.0f} (uncalibrated)").format(dpi=dpi))
            self.dpi_label.setStyleSheet(f"color: {COLOR_FONT_MUTED};")

    # ── Voice engine menu ─────────────────────────────────────────
    def set_voice_engines(self, engines, current=None) -> None:
        """Populate the ▾ engine menu. `engines` = [(key, label), …]
        (currently just [("vosk","Vosk — offline")] → chevron auto-hides).
        Emits `voice_engine_changed(key)` on pick."""
        self.btn_voice.set_voice_engines(engines, current)

    # ── Voice transcript helpers ──────────────────────────────────

    def set_voice_transcript(self, text: str) -> None:
        """Show the last few recognized words. MainWindow forwards
        ``VoiceWorker.transcription_update`` here. The worker sends rich-text
        (HTML) with each word colour-coded — green = fired, yellow = debounced,
        red = unrecognized — which we render as-is. Plain text is still
        accepted (trimmed to the last 10 words) for safety."""
        if "<span" in text:
            body = text
        else:
            words = text.split()
            if len(words) > 10:
                text = "… " + " ".join(words[-10:])
            body = text or "—"
        self.voice_transcript.setText(
            self.tr("Heard: {text}").format(text=body)
        )
        self.voice_transcript.setVisible(True)

    def clear_voice_transcript(self) -> None:
        self.voice_transcript.setText(self.tr("Heard: —"))
        self.voice_transcript.setVisible(False)

    # ── Voice command reference ───────────────────────────────────

    def set_voice_command_legend(self, mapping: dict) -> None:
        """``mapping`` = ``{action: [words]}`` (from
        ``args.config['voicecontrols']``). Renders the spoken word → what it
        does, e.g. ``photo → capture  ·  delete → delete last``."""
        human = {
            "scan": self.tr("capture"),
            "trash": self.tr("delete last"),
            "quit": self.tr("quit"),
        }
        bits = []
        for action, words in (mapping or {}).items():
            if action == "debounce_time" or not words:
                continue
            word = str(words[0])
            does = human.get(action, action)
            bits.append(f"<b>{word}</b> → {does}")
        self.voice_cmd_legend.setText("  ·  ".join(bits))
        self.voice_cmd_legend.setVisible(bool(bits))

    # ── Shortcut legend ───────────────────────────────────────────

    def set_shortcut_legend(self, mapping: dict) -> None:
        """``mapping`` is ``{action_name: [keys]}`` (from
        ``args.config['keycontrols']``). Renders ``scan: SPACE
        ·  trash: BACKSPACE  ·  rotate: R``."""
        if not mapping:
            self.shortcut_legend.setText("—")
            return
        bits = []
        for action, keys in mapping.items():
            if not keys:
                continue
            label = " / ".join(str(k).upper() for k in keys)
            bits.append(f"<b>{action}</b>: {label}")
        self.shortcut_legend.setText("  ·  ".join(bits) or "—")
