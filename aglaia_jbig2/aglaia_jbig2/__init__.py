# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""JBIG2 encoder bindings for Aglaïa PDF export.

Lossless, per-page. Wraps the `jbig2enc-rust` crate via PyO3.

Usage:
    from aglaia_jbig2 import encode_page_lossless
    blob = encode_page_lossless(raw_bytes, width, height)
    # blob is an embedded JBIG2 stream (no file header), ready to drop
    # into a PDF image XObject with /Filter /JBIG2Decode.
"""

from ._native import encode_page as _encode_page, encode_chunk as _encode_chunk

__all__ = ["encode_page_lossless", "encode_chunk_lossless"]


def encode_page_lossless(data: bytes, width: int, height: int) -> bytes:
    """Encode one 1-bit page (row-major, 1 byte/pixel, 0=white 1=black).
    Returns the embedded JBIG2 stream (suitable for PDF /JBIG2Decode)."""
    _globals, page = _encode_page(data, width, height, pdf_mode=True, lossless=True)
    return bytes(page)


def encode_chunk_lossless(pages: list[tuple[bytes, int, int]]) -> bytes:
    """Encode several pages as one document with shared symbol dict.
    Use only when callers can decode multi-page JBIG2 — current PDF
    export sticks to encode_page_lossless."""
    return bytes(_encode_chunk(pages, mode="lossless"))
