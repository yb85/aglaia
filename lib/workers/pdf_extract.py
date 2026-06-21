# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PDF → image extraction via pypdfium2.

Each PDF page becomes one RGB ndarray at the requested DPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pypdfium2 as pdfium


def open_pdf(path) -> pdfium.PdfDocument:
    return pdfium.PdfDocument(str(path))


def page_count(path) -> int:
    doc = open_pdf(path)
    try:
        return len(doc)
    finally:
        doc.close()


def render_page(doc: pdfium.PdfDocument, page_index: int,
                render_dpi: float) -> np.ndarray:
    """Render one PDF page to an RGB ndarray.

    `render_dpi` is interpreted as user-space density: a 600 dpi target
    on a US-Letter page (8.5 × 11 in) yields a 5100 × 6600 pixmap. The
    function does not return the dpi — callers pass it in and use the
    same value when persisting the image (it's an authoritative tag, not
    a measurement).
    """
    page = doc[page_index]
    try:
        scale = render_dpi / 72.0
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return np.asarray(pil, dtype=np.uint8)
    finally:
        # PdfPage objects pin a reference into libpdfium; closing keeps
        # the per-document footprint flat for multi-page PDFs.
        try:
            page.close()
        except Exception:
            pass


def iter_pages(path, *, render_dpi: float = 200.0
               ) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield `(page_index, rgb_array, dpi)` for every page.

    Convenience generator for ingestors that want a single loop instead
    of opening / iterating the document themselves."""
    doc = open_pdf(path)
    try:
        n = len(doc)
        for i in range(n):
            arr = render_page(doc, i, render_dpi)
            yield i, arr, float(render_dpi)
    finally:
        doc.close()


def render_one(path, page_index: int, render_dpi: float) -> np.ndarray:
    """One-shot helper: open the doc, render one page, close. Avoid for
    large batches (use `iter_pages` instead)."""
    doc = open_pdf(path)
    try:
        return render_page(doc, page_index, render_dpi)
    finally:
        doc.close()
