# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Export sidebar tab.

Three format cards — PDF / Markdown / Slim Aglaïa project. The user
picks one, optionally tweaks its inline knobs, and clicks the single
``Export`` button below. PDF is the default selection.

The PDF card carries two toggles:

* ``Use JBIG2 for monochrome`` — 1-bit pages encoded with JBIG2 (≈30%
  smaller than G4). Off → G4 fallback.
* ``Add invisible OCR text layer`` — only enabled when at least one
  branch has a fresh OCR run. MainWindow flips ``chk_ocr_layer`` via
  ``set_ocr_layer_available()`` whenever OCR state changes.

The Markdown card is disabled until OCR data is available; MainWindow
calls ``set_markdown_available()`` whenever OCR runs land. While both are
disabled, ``lbl_ocr_hint`` says why and what to do about it — a greyed-out
control with no reason given reads as a broken app.

MainWindow wires the single export click handler on ``btn_export`` and
reads the picked format via ``format_group.current_key()``. Compression
hint comes from ``chk_jbig2.isChecked()``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aglaia.gui.colors import (
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_PLACEHOLDER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
)
from aglaia.gui.sidebar.widgets import RadioCardGroup, ToggleSwitch


_PRIMARY_BTN_QSS = f"""
QPushButton {{
    background-color: {COLOR_PRIMARY}; color: {COLOR_FONT_INVERSE};
    border-radius: 4px; padding: 8px; font-weight: bold;
}}
QPushButton:hover {{ background-color: {COLOR_PRIMARY_HOVER}; }}
QPushButton:disabled {{ background-color: {COLOR_FONT_DIM}; color: {COLOR_FONT_PLACEHOLDER}; }}
"""


class ExportTab(QWidget):
    """Format-cards picker + single Export button."""

    #: A Send-to card was pressed. The host does the export and the send —
    #: this tab knows which destination, not how to reach it.
    send_to_requested = Signal(str)
    #: A destination that is not configured yet: open its settings.
    destination_settings_requested = Signal(str)


    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel(self.tr("Export"))
        title.setObjectName("SectionTitle")
        outer.addWidget(title)

        outer.addWidget(self._field_label(self.tr("Format")))

        self.format_group = RadioCardGroup()
        outer.addWidget(self.format_group)

        # ── PDF card with two toggles as extras ────────────────────
        pdf_extras = QWidget()
        pdf_extras_l = QVBoxLayout(pdf_extras)
        pdf_extras_l.setContentsMargins(0, 4, 0, 0)
        pdf_extras_l.setSpacing(4)

        self.chk_jbig2 = ToggleSwitch(self.tr("Use JBIG2 for monochrome"))
        # Only offer JBIG2 when its encoder is actually built — probe the
        # symbol, not the package (the repo's `aglaia_jbig2/` crate dir is a
        # namespace-package false positive). Unavailable → disabled + OFF
        # (so export uses G4); no silent fallback surprise.
        try:
            from aglaia_jbig2 import encode_page_lossless  # noqa: F401
            _jbig2_ok = True
        except Exception:
            _jbig2_ok = False
        self.chk_jbig2.setChecked(_jbig2_ok)
        self.chk_jbig2.setEnabled(_jbig2_ok)
        self.chk_jbig2.setToolTip(
            self.tr(
                "1-bit pages: encode with JBIG2 (≈30% smaller than G4)."
            ) if _jbig2_ok else self.tr(
                "JBIG2 encoder not installed — exports use CCITT G4. "
                "Build it: cd aglaia_jbig2 && uv run maturin develop --release"
            )
        )
        pdf_extras_l.addWidget(self.chk_jbig2)

        self.chk_ocr_layer = ToggleSwitch(self.tr("Add invisible OCR text layer"))
        self.chk_ocr_layer.setChecked(False)
        self.chk_ocr_layer.setEnabled(False)
        self.chk_ocr_layer.setToolTip(
            self.tr(
                "Overlay the OCR result as selectable, invisible text on top "
                "of each page. Enabled once OCR has been run."
            )
        )
        pdf_extras_l.addWidget(self.chk_ocr_layer)

        self.format_group.add_card(
            "pdf", self.tr("PDF"),
            self.tr("Searchable PDF with optional OCR text layer."),
            icon_name="filetype-pdf",
            extras=pdf_extras,
        )

        # ── Markdown card — disabled until OCR data lands ──────────
        # Post-processing knobs for Mistral OCR output (applied at export
        # time, from the stored raw page — no re-OCR needed to change them).
        md_extras = QWidget()
        md_extras_l = QVBoxLayout(md_extras)
        md_extras_l.setContentsMargins(0, 4, 0, 0)
        md_extras_l.setSpacing(4)

        from aglaia.app_data import db as _cfg
        try:
            with _cfg.session() as _c:
                _fn = str(_cfg.get(_c, _cfg.KEY_MISTRAL_FOOTNOTES, "numeric"))
                _hdr = bool(_cfg.get(_c, _cfg.KEY_MISTRAL_HEADERS, True))
        except Exception:
            _fn, _hdr = "numeric", True

        _fn_row = QHBoxLayout()
        _fn_row.setSpacing(6)
        self.chk_footnotes = ToggleSwitch(self.tr("Convert footnotes"))
        self.chk_footnotes.setChecked(_fn in ("numeric", "alphabetic"))
        self.chk_footnotes.setToolTip(self.tr(
            "Footnote refs (LaTeX $^{N}$, Unicode ⁹⁸, or (N)) → GFM [^N]; "
            "line-start entries → [^N]:. Footnote definitions in the extracted "
            "footer are lifted out as real footnotes."))
        self.chk_footnotes.toggled.connect(self._on_footnotes_toggled)
        _fn_row.addWidget(self.chk_footnotes)
        self.combo_footnote_style = QComboBox()
        self.combo_footnote_style.addItems(["numeric", "alphabetic"])
        self.combo_footnote_style.setCurrentText(
            _fn if _fn in ("numeric", "alphabetic") else "numeric")
        self.combo_footnote_style.setEnabled(self.chk_footnotes.isChecked())
        self.combo_footnote_style.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 10px;")
        self.combo_footnote_style.currentTextChanged.connect(
            self._on_footnote_style_changed)
        _fn_row.addWidget(self.combo_footnote_style)
        _fn_row.addStretch(1)
        md_extras_l.addLayout(_fn_row)

        self.chk_wrap_hf = ToggleSwitch(self.tr("Wrap headers / footers"))
        self.chk_wrap_hf.setChecked(_hdr)
        self.chk_wrap_hf.setToolTip(self.tr(
            "Wrap the page's running head / page number in <header>/<footer> "
            "tags. Off → keep them as plain inline text. Either way the text "
            "is preserved."))
        self.chk_wrap_hf.toggled.connect(self._on_wrap_hf_toggled)
        md_extras_l.addWidget(self.chk_wrap_hf)

        self.format_group.add_card(
            "markdown", self.tr("Markdown"),
            self.tr("Plain text extracted from OCR. Needs an OCR run."),
            icon_name="markdown",
            enabled=False,
            extras=md_extras,
        )

        # ── Slim project card ──────────────────────────────────────
        self.format_group.add_card(
            "slim", self.tr("Slim Aglaïa project"),
            self.tr("Pruned project DB — raw + chosen layout only."),
            icon_name="compression",
        )

        self.format_group.set_current_key("pdf")

        # ── Why-is-this-greyed-out hint ────────────────────────────
        # The Markdown card and the PDF OCR-layer toggle both go dead
        # without an OCR run, and a disabled control with no explanation
        # reads as a bug. Shown only while OCR is missing; hidden the
        # moment a layer exists (see `_set_ocr_hint`).
        self.lbl_ocr_hint = QLabel(self.tr(
            "Run OCR to enable Markdown export and the PDF text layer. "
            "A batched cloud OCR only counts once its result is downloaded "
            "— hit “Check result” in the OCR tab."
        ))
        self.lbl_ocr_hint.setWordWrap(True)
        self.lbl_ocr_hint.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px; padding: 2px 0 4px 0;")
        outer.addWidget(self.lbl_ocr_hint)

        # ── OCR-layer selector — which engine's OCR to export ──────
        # Applies to BOTH the PDF text layer and Markdown. Hidden until at
        # least one OCR layer exists; "Latest" = the most recent layer (the
        # back-compat default). MainWindow fills it via `set_ocr_layers`.
        self._ocr_layer_row = QWidget()
        _ocr_row = QHBoxLayout(self._ocr_layer_row)
        _ocr_row.setContentsMargins(0, 4, 0, 0)
        _ocr_row.setSpacing(6)
        _ocr_row.addWidget(self._field_label(self.tr("OCR layer")))
        self.combo_ocr_layer = QComboBox()
        self.combo_ocr_layer.setToolTip(self.tr(
            "Which engine's OCR layer to export (PDF text layer + Markdown). "
            "'Latest' uses the most recently generated layer."
        ))
        _ocr_row.addWidget(self.combo_ocr_layer, 1)
        self._ocr_layer_row.setVisible(False)
        outer.addWidget(self._ocr_layer_row)

        # ── Single Export button ───────────────────────────────────
        self.btn_export = QPushButton(self.tr("Export"))
        self.btn_export.setStyleSheet(_PRIMARY_BTN_QSS)
        self.btn_export.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        try:
            from aglaia.gui.theme import icon as _icon
            self.btn_export.setIcon(_icon("export", color=COLOR_FONT_INVERSE, size=14))
        except Exception:
            pass
        outer.addWidget(self.btn_export)

        # ── Send to (export plugins) ───────────────────────────────
        # An installed export plugin is only useful if it is offered where
        # the export happens. Filtered by the chosen format, so a Markdown
        # export never offers a destination that only takes PDFs — being
        # told "not accepted" after the export ran is a thing to prevent,
        # not to report.
        self._send_label = self._field_label(self.tr("Send to"))
        outer.addWidget(self._send_label)
        self._send_box = QWidget()
        self._send_layout = QVBoxLayout(self._send_box)
        self._send_layout.setContentsMargins(0, 0, 0, 0)
        self._send_layout.setSpacing(6)
        outer.addWidget(self._send_box)
        self.format_group.currentChanged.connect(
            lambda *_: self.refresh_destinations())
        self.refresh_destinations()

        # Character-width normalisation — pipeline-controlled visibility.
        self.chk_norm_widths = QCheckBox(self.tr("Normalize character width"))
        self.chk_norm_widths.setStyleSheet(
            f"padding: 6px; font-weight: bold; color: {COLOR_PRIMARY};"
        )
        self.chk_norm_widths.setVisible(False)
        outer.addWidget(self.chk_norm_widths)

        outer.addStretch(1)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    # ── Markdown post-processing toggles (persisted to config) ──────
    @staticmethod
    def _set_cfg(key: str, value) -> None:
        from aglaia.app_data import db as cfg
        try:
            with cfg.session() as conn:
                cfg.set(conn, key, value)
                conn.commit()
        except Exception:
            pass

    def _on_footnotes_toggled(self, on: bool) -> None:
        from aglaia.app_data import db as cfg
        self.combo_footnote_style.setEnabled(on)
        self._set_cfg(cfg.KEY_MISTRAL_FOOTNOTES,
                      self.combo_footnote_style.currentText() if on else "off")

    def _on_footnote_style_changed(self, style: str) -> None:
        from aglaia.app_data import db as cfg
        if self.chk_footnotes.isChecked():
            self._set_cfg(cfg.KEY_MISTRAL_FOOTNOTES, style)

    def _on_wrap_hf_toggled(self, on: bool) -> None:
        from aglaia.app_data import db as cfg
        self._set_cfg(cfg.KEY_MISTRAL_HEADERS, bool(on))

    # ── MainWindow-facing API ──────────────────────────────────────

    def _set_ocr_hint(self, available: bool) -> None:
        """Show the hint only while the OCR-gated controls are disabled."""
        self.lbl_ocr_hint.setVisible(not available)

    def set_markdown_available(self, available: bool) -> None:
        """Toggle Markdown card. When the active selection was Markdown
        and it just became unavailable, fall back to PDF."""
        self.format_group.set_card_enabled("markdown", available)
        if not available and self.format_group.current_key() == "markdown":
            self.format_group.set_current_key("pdf")
        self._set_ocr_hint(available)

    def set_ocr_layer_available(self, available: bool) -> None:
        """OCR layer checkbox lives inside the PDF card; flip enable +
        default-checked together. Re-checking on enable transition is
        what the user almost always wants."""
        was_enabled = self.chk_ocr_layer.isEnabled()
        self.chk_ocr_layer.setEnabled(available)
        self._set_ocr_hint(available)
        if not available:
            self.chk_ocr_layer.setChecked(False)
        elif not was_enabled:
            self.chk_ocr_layer.setChecked(True)

    def set_ocr_layers(self, layers) -> None:
        """Populate the OCR-layer selector from `OcrRepo.available_ocr_layers()`
        rows (latest-generated first). Hidden when there are none. Preserves the
        current selection if that engine is still present."""
        prev = self.combo_ocr_layer.currentData()
        self.combo_ocr_layer.blockSignals(True)
        self.combo_ocr_layer.clear()
        self.combo_ocr_layer.addItem(self.tr("Latest layer"), None)
        for r in layers or []:
            eng = r["engine"]
            self.combo_ocr_layer.addItem(f"{eng} ({r['n_branches']})", eng)
        if prev is not None:
            i = self.combo_ocr_layer.findData(prev)
            if i >= 0:
                self.combo_ocr_layer.setCurrentIndex(i)
        self.combo_ocr_layer.blockSignals(False)
        self._ocr_layer_row.setVisible(bool(layers))

    def selected_ocr_engine(self) -> Optional[str]:
        """The engine whose OCR layer to export, or None for the latest layer."""
        return self.combo_ocr_layer.currentData()

    # ── export plugins ────────────────────────────────────────────
    def refresh_destinations(self) -> None:
        """Rebuild the Send-to cards for the current format.

        Cheap and idempotent — called on format change and after a plugin is
        installed or removed, so the section never lists something that is no
        longer there."""
        while self._send_layout.count():
            it = self._send_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        fmt = {"pdf": "pdf", "markdown": "md",
               "slim": "agl"}.get(self.format_group.current_key() or "", "")
        try:
            from aglaia.workers import destinations as _dest
            dests = _dest.for_format(fmt) if fmt else []
        except Exception:
            dests = []
        # Nothing installed that takes this format: hide the heading too,
        # rather than leaving an empty label with a promise under it.
        self._send_label.setVisible(bool(dests))
        self._send_box.setVisible(bool(dests))
        for d in dests:
            self._send_layout.addWidget(self._destination_card(d))

    def _destination_card(self, dest) -> QWidget:
        from aglaia.gui.colors import (COLOR_BG_OVERLAY_SOFT, COLOR_ERROR,
                                       COLOR_FONT_PRIMARY, COLOR_OUTLINE_FAINT,
                                       COLOR_SUCCESS)
        card = QFrame()
        card.setObjectName("sendCard")
        card.setStyleSheet(
            f"QFrame#sendCard {{ background: {COLOR_BG_OVERLAY_SOFT}; "
            f"border: 1px solid {COLOR_OUTLINE_FAINT}; border-radius: 8px; }}")
        row = QHBoxLayout(card)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        col = QVBoxLayout()
        col.setSpacing(1)
        name = QLabel(dest.display or dest.name)
        name.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 600; font-size: 12px;")
        col.addWidget(name)
        missing = dest.missing_settings()
        state = QLabel(self.tr("Ready") if not missing
                       else self.tr("Needs: {what}").format(
                           what=", ".join(missing)))
        state.setWordWrap(True)
        state.setStyleSheet(
            f"color: {COLOR_SUCCESS if not missing else COLOR_ERROR}; "
            f"font-size: 10px;")
        col.addWidget(state)
        row.addLayout(col, 1)

        btn = QPushButton(self.tr("Send") if not missing
                          else self.tr("Set up…"))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(
            lambda _=False, d=dest, m=bool(missing):
                (self.destination_settings_requested.emit(d.name) if m
                 else self.send_to_requested.emit(d.name)))
        row.addWidget(btn)
        return card

    def current_format(self) -> Optional[str]:
        return self.format_group.current_key()

    def compression_hint(self) -> str:
        """Returns ``'jbig2'`` or ``'g4'`` based on the toggle."""
        return "jbig2" if self.chk_jbig2.isChecked() else "g4"

    def set_compression(self, profile: str) -> None:
        """Programmatic compression set, used by --auto-run config."""
        self.chk_jbig2.setChecked(profile == "jbig2")
