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
calls ``set_markdown_available()`` whenever OCR runs land.

MainWindow wires the single export click handler on ``btn_export`` and
reads the picked format via ``format_group.current_key()``. Compression
hint comes from ``chk_jbig2.isChecked()``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
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
        self.format_group.add_card(
            "markdown", self.tr("Markdown"),
            self.tr("Plain text extracted from OCR. Needs an OCR run."),
            icon_name="markdown",
            enabled=False,
        )

        # ── Slim project card ──────────────────────────────────────
        self.format_group.add_card(
            "slim", self.tr("Slim Aglaïa project"),
            self.tr("Pruned project DB — raw + chosen layout only."),
            icon_name="compression",
        )

        self.format_group.set_current_key("pdf")

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

    # ── MainWindow-facing API ──────────────────────────────────────

    def set_markdown_available(self, available: bool) -> None:
        """Toggle Markdown card. When the active selection was Markdown
        and it just became unavailable, fall back to PDF."""
        self.format_group.set_card_enabled("markdown", available)
        if not available and self.format_group.current_key() == "markdown":
            self.format_group.set_current_key("pdf")

    def set_ocr_layer_available(self, available: bool) -> None:
        """OCR layer checkbox lives inside the PDF card; flip enable +
        default-checked together. Re-checking on enable transition is
        what the user almost always wants."""
        was_enabled = self.chk_ocr_layer.isEnabled()
        self.chk_ocr_layer.setEnabled(available)
        if not available:
            self.chk_ocr_layer.setChecked(False)
        elif not was_enabled:
            self.chk_ocr_layer.setChecked(True)

    def current_format(self) -> Optional[str]:
        return self.format_group.current_key()

    def compression_hint(self) -> str:
        """Returns ``'jbig2'`` or ``'g4'`` based on the toggle."""
        return "jbig2" if self.chk_jbig2.isChecked() else "g4"

    def set_compression(self, profile: str) -> None:
        """Programmatic compression set, used by --auto-run config."""
        self.chk_jbig2.setChecked(profile == "jbig2")
