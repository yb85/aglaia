# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Import sidebar tab — drop / pick / reorder / import images + PDFs.

Layout
------
* Drop zone at the top — dashed border, lucide ``upload`` glyph,
  hint text. Accepts external drag-drop of files (mix of image
  formats + ``application/pdf``). Click opens a native file picker
  for the same MIME set.
* Reorderable queue list below the drop zone:
  - one row per pending file
  - thumbnail (32 px square, lazy-rendered for images; PDF first-page
    render for PDFs — the PDF render runs sync at add-time; it's tiny)
  - filename (elided middle), size + page-count (for PDFs)
  - grip-vertical handle on the left for reorder
  - eye-off button on the right to drop from queue
* Footer row: DPI for raw imports + Import button + Clear button.

Internal drag-drop uses the MIME type ``application/x-aglaia-import``
so external file drops don't trigger a no-op move and internal
reorders don't accidentally re-trigger external-drop handling.

Signals
-------
* ``import_requested(items, dpi)`` — list of ``(Path, kind)`` tuples
  in queue order. ``kind`` is ``"image"`` or ``"pdf"``. MainWindow
  is expected to fan-out to ``enqueue_image_files`` /
  ``enqueue_pdf_files`` workers.
* ``cleared()`` — emitted when the queue is wiped via the Clear
  button. Purely advisory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QByteArray, QMimeData, QSize, Qt, Signal,
)
from PySide6.QtGui import (
    QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QIcon, QImage,
    QPainter, QPalette, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from lib.gui.colors import (
    COLOR_BG_BUTTON,
    COLOR_BG_OVERLAY_SOFT,
    COLOR_FONT_DIM,
    COLOR_FONT_INVERSE,
    COLOR_FONT_MUTED,
    COLOR_FONT_PLACEHOLDER,
    COLOR_FONT_PRIMARY,
    COLOR_FONT_SECTION_LABEL,
    COLOR_FONT_TIMING_NAME,
    COLOR_OUTLINE,
    COLOR_OUTLINE_BUTTON_STRONG,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG_SOFT,
    COLOR_PRIMARY_HOVER,
)


_INTERNAL_MIME = "application/x-aglaia-import"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff",
                   ".bmp", ".webp", ".pbm", ".pgm", ".ppm"}
_PDF_SUFFIXES = {".pdf"}

_THUMB_PX = 32


def _classify(path: Path) -> Optional[str]:
    suf = path.suffix.lower()
    if suf in _PDF_SUFFIXES:
        return "pdf"
    if suf in _IMAGE_SUFFIXES:
        return "image"
    return None


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _pdf_page_count(path: Path) -> int:
    """Best-effort page count; returns 0 on failure."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _make_thumb(path: Path, kind: str) -> QPixmap:
    """Return a 32×32 thumbnail. For PDFs, render page 0; for images,
    load + scale. Failure → placeholder square."""
    try:
        if kind == "image":
            img = QImage(str(path))
            if not img.isNull():
                pix = QPixmap.fromImage(img).scaled(
                    _THUMB_PX, _THUMB_PX,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                return pix
        elif kind == "pdf":
            import fitz
            doc = fitz.open(str(path))
            try:
                if len(doc) > 0:
                    page = doc[0]
                    matrix = fitz.Matrix(0.2, 0.2)
                    pm = page.get_pixmap(matrix=matrix, alpha=False)
                    img = QImage(pm.samples, pm.width, pm.height,
                                 pm.stride, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(img.copy()).scaled(
                        _THUMB_PX, _THUMB_PX,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    return pix
            finally:
                doc.close()
    except Exception:
        pass

    # Placeholder.
    pix = QPixmap(_THUMB_PX, _THUMB_PX)
    pix.fill(QColor(COLOR_BG_BUTTON))
    return pix


# ── Drop zone ──────────────────────────────────────────────────────


class _DropZone(QFrame):
    """Top drop area — dashed border + lucide upload glyph. Accepts
    external file drops, click to open a file picker."""

    files_dropped = Signal(list)   # list[Path]
    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ImportDropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(96)
        self.setStyleSheet(
            f"QFrame#ImportDropZone {{"
            f"  background: {COLOR_BG_OVERLAY_SOFT};"
            f"  border: 1px dashed {COLOR_OUTLINE};"
            f"  border-radius: 8px;"
            f"}}"
            f"QFrame#ImportDropZone[hot=\"true\"] {{"
            f"  border-color: {COLOR_PRIMARY};"
            f"  background: {COLOR_PRIMARY_BG_SOFT};"
            f"}}"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel()
        try:
            from lib.gui.theme import icon as _icon
            ic = _icon("upload", color=COLOR_FONT_SECTION_LABEL, size=24)
            icon_lbl.setPixmap(ic.pixmap(24, 24))
        except Exception:
            icon_lbl.setText("⬆")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hint = QLabel(self.tr("Drop images or PDFs"))
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color: {COLOR_FONT_SECTION_LABEL}; font-weight: 600;")

        sub = QLabel(self.tr("…or click to pick files"))
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")

        layout.addWidget(icon_lbl)
        layout.addWidget(hint)
        layout.addWidget(sub)

    def mousePressEvent(self, ev) -> None:  # noqa: N802 — Qt API
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)

    def _accepts_external(self, mime: QMimeData) -> bool:
        # External OS drops carry urls; internal reorder uses our
        # private MIME. Reject the internal one so a reorder doesn't
        # touch the drop zone.
        if mime.hasFormat(_INTERNAL_MIME):
            return False
        return bool(mime.hasUrls())

    def dragEnterEvent(self, ev: QDragEnterEvent) -> None:
        if self._accepts_external(ev.mimeData()):
            ev.acceptProposedAction()
            self.setProperty("hot", True)
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            ev.ignore()

    def dragMoveEvent(self, ev: QDragMoveEvent) -> None:
        if self._accepts_external(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev) -> None:  # noqa: N802
        self.setProperty("hot", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev: QDropEvent) -> None:
        self.setProperty("hot", False)
        self.style().unpolish(self)
        self.style().polish(self)
        if not self._accepts_external(ev.mimeData()):
            ev.ignore()
            return
        paths: list[Path] = []
        for url in ev.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            p = Path(local)
            if p.is_dir():
                for child in sorted(p.rglob("*")):
                    if child.is_file() and _classify(child) is not None:
                        paths.append(child)
            elif p.is_file() and _classify(p) is not None:
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
        ev.acceptProposedAction()


# ── Queue list ─────────────────────────────────────────────────────


class _QueueRow(QWidget):
    """One row inside the queue list."""

    def __init__(self, path: Path, kind: str,
                 on_remove, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = path
        self.kind = kind
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(6)

        try:
            from lib.gui.theme import icon as _icon
            grip = QLabel()
            grip.setPixmap(_icon("grip-vertical", color=COLOR_FONT_DIM,
                                 size=14).pixmap(14, 14))
            h.addWidget(grip)
        except Exception:
            pass

        self.thumb = QLabel()
        self.thumb.setFixedSize(_THUMB_PX, _THUMB_PX)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setPixmap(_make_thumb(path, kind))
        h.addWidget(self.thumb)

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        text_col.setContentsMargins(0, 0, 0, 0)

        name_lbl = QLabel(path.name)
        name_lbl.setStyleSheet(f"color: {COLOR_FONT_TIMING_NAME}; font-weight: 600;")
        name_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction
        )

        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        meta_bits = [_human_bytes(size)]
        if kind == "pdf":
            n = _pdf_page_count(path)
            if n > 0:
                meta_bits.append(self.tr("{n} pages").format(n=n))
        meta_lbl = QLabel(" · ".join(meta_bits))
        meta_lbl.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px;"
        )
        text_col.addWidget(name_lbl)
        text_col.addWidget(meta_lbl)
        h.addLayout(text_col, 1)

        rm = QToolButton()
        rm.setAutoRaise(True)
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.setToolTip(self.tr("Remove from queue"))
        try:
            from lib.gui.theme import icon as _icon
            rm.setIcon(_icon("eye-off", color=COLOR_FONT_PLACEHOLDER, size=14))
        except Exception:
            rm.setText("×")
        rm.clicked.connect(lambda: on_remove(self))
        h.addWidget(rm)


class _QueueList(QListWidget):
    """QListWidget with internal-only drag-drop reorder. External
    file drops are caught by the drop zone above this widget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setUniformItemSizes(False)
        self.setSpacing(2)
        self.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; }}"
            f"QListWidget::item:selected {{"
            f"  background: {COLOR_PRIMARY_BG_SOFT};"
            f"  border-radius: 4px;"
            f"}}"
        )

    def mimeTypes(self) -> list[str]:  # noqa: N802
        # Use our private type so the drop zone above ignores reorder
        # drags. Internal items still carry the queue row reference via
        # ``QListWidget.mimeData`` so QListWidget's stock move logic
        # keeps working.
        return [_INTERNAL_MIME]

    def mimeData(self, items):  # noqa: N802
        m = super().mimeData(items)
        # Tag the payload so the drop zone rejects it.
        m.setData(_INTERNAL_MIME, QByteArray(b"1"))
        return m


# ── ImportTab ──────────────────────────────────────────────────────


class ImportTab(QWidget):
    """Import controls: drop zone + reorderable queue + DPI + buttons."""

    import_requested = Signal(list, float)  # [(Path, kind)], dpi
    cleared = Signal()

    DEFAULT_DPI = 300.0

    def __init__(self, default_dpi: Optional[float] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel(self.tr("Import"))
        title.setObjectName("SectionTitle")
        outer.addWidget(title)

        self.drop_zone = _DropZone(self)
        self.drop_zone.files_dropped.connect(self.add_paths)
        self.drop_zone.clicked.connect(self._open_picker)
        outer.addWidget(self.drop_zone)

        self.queue = _QueueList(self)
        outer.addWidget(self.queue, 1)

        self._empty_hint = QLabel(
            self.tr(
                "Nothing queued yet — drop files above or use the button to "
                "pick some."
            )
        )
        self._empty_hint.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px;"
        )
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._empty_hint)

        # Footer row.
        foot = QHBoxLayout()
        foot.setSpacing(6)
        self.dpi_spin = QDoubleSpinBox()
        self.dpi_spin.setDecimals(0)
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.setValue(float(default_dpi or self.DEFAULT_DPI))
        self.dpi_spin.setToolTip(
            self.tr(
                "DPI tag for raw image imports. PDFs render at the page's "
                "embedded resolution; this only affects loose images."
            )
        )
        foot.addWidget(self.dpi_spin)

        self.btn_clear = QPushButton(self.tr("Clear"))
        self.btn_clear.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent; color: {COLOR_FONT_PLACEHOLDER};"
            f"  border: 1px solid {COLOR_OUTLINE_BUTTON_STRONG}; border-radius: 6px;"
            f"  padding: 4px 10px;"
            f"}}"
            f"QPushButton:hover {{ color: {COLOR_FONT_INVERSE}; border-color: {COLOR_FONT_PLACEHOLDER}; }}"
        )
        self.btn_clear.clicked.connect(self.clear_queue)
        foot.addWidget(self.btn_clear)

        self.btn_import = QPushButton(self.tr("Import"))
        self.btn_import.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {COLOR_PRIMARY}; color: {COLOR_FONT_INVERSE};"
            f"  border-radius: 6px; padding: 6px 16px; font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{ background-color: {COLOR_PRIMARY_HOVER}; }}"
            f"QPushButton:disabled {{ background-color: {COLOR_FONT_DIM}; color: {COLOR_FONT_PLACEHOLDER}; }}"
        )
        self.btn_import.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self.btn_import.clicked.connect(self._emit_import)
        foot.addWidget(self.btn_import, 1)

        outer.addLayout(foot)
        self._refresh_state()

    # ── queue manipulation ─────────────────────────────────────────

    def add_paths(self, paths) -> None:
        for p in paths:
            p = Path(p)
            kind = _classify(p)
            if kind is None:
                continue
            row = _QueueRow(p, kind, self._remove_row, self)
            item = QListWidgetItem(self.queue)
            item.setSizeHint(QSize(0, _THUMB_PX + 8))
            self.queue.addItem(item)
            self.queue.setItemWidget(item, row)
        self._refresh_state()

    def _remove_row(self, row_widget: _QueueRow) -> None:
        for i in range(self.queue.count()):
            it = self.queue.item(i)
            if self.queue.itemWidget(it) is row_widget:
                self.queue.takeItem(i)
                break
        self._refresh_state()

    def clear_queue(self) -> None:
        self.queue.clear()
        self._refresh_state()
        self.cleared.emit()

    def _open_picker(self) -> None:
        # Accept the union of image + PDF MIME so the picker matches the
        # drop zone's accept rule.
        filt = self.tr(
            "Images & PDFs (*.png *.jpg *.jpeg *.tif *.tiff *.bmp "
            "*.webp *.pbm *.pgm *.ppm *.pdf);;All files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Pick images or PDFs"), "", filt,
        )
        if paths:
            self.add_paths([Path(p) for p in paths])

    def _refresh_state(self) -> None:
        n = self.queue.count()
        self.btn_import.setEnabled(n > 0)
        self.btn_clear.setEnabled(n > 0)
        self._empty_hint.setVisible(n == 0)

    def _emit_import(self) -> None:
        items: list[tuple[Path, str]] = []
        for i in range(self.queue.count()):
            it = self.queue.item(i)
            w = self.queue.itemWidget(it)
            if isinstance(w, _QueueRow):
                items.append((w.path, w.kind))
        if not items:
            return
        self.import_requested.emit(items, float(self.dpi_spin.value()))
        # MainWindow clears the queue once the enqueue workers have
        # accepted everything — we don't pre-emptively clear to keep
        # the user informed if the host bails out.
