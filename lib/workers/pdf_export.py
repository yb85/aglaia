# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""PDF assembly — bitonal (G4 / JBIG2 lossless), colour / grayscale
(DCTDecode), and the invisible OCR text overlay.

For 1-bit pages we target JBIG2Decode (~25–37 % smaller than G4 on
Aglaïa-style scans) with CCITTFaxDecode K=-1 (CCITT G4) as a universally-
supported fallback. Colour pages embed the JPEG bytes verbatim as a
DCTDecode XObject; PNG / non-JPEG sources are re-encoded at quality 90.

Native binary dependencies:
  - `pikepdf` (libqpdf wrapper) — assembles the PDF object graph.
  - `aglaia_jbig2` (optional) — PyO3 wrapper around jbig2enc-rust.

If `aglaia_jbig2` isn't installed, JBIG2 mode falls back to G4 with a
console warning.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

try:
    import pikepdf
    from pikepdf import Name, Stream
except Exception as e:  # pragma: no cover
    pikepdf = None
    _PIKEPDF_ERR = e


def _row_to_pil(row) -> Image.Image:
    """Decode an image row to a PIL Image (already-binarised → mode '1')."""
    blob = bytes(row["blob"])
    im = Image.open(io.BytesIO(blob))
    if im.mode != "1":
        im = im.convert("1")
    return im


def _g4_compress(im: Image.Image) -> bytes:
    """CCITT G4 byte stream for the PIL image. Strips TIFF wrapper —
    PDF /CCITTFaxDecode takes the raw G4 codewords."""
    buf = io.BytesIO()
    # Force a single-strip TIFF so we can return the whole codestream
    # in one slice. PIL's default chunks at ~270 rows/strip and per-strip
    # G4 alignment doesn't concatenate cleanly.
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    ifd[278] = im.height  # RowsPerStrip
    im.save(buf, format="TIFF", compression="group4", tiffinfo=ifd)
    tiff_bytes = buf.getvalue()
    return _extract_tiff_strip(tiff_bytes)


def _extract_tiff_strip(tiff: bytes) -> bytes:
    """Single-strip TIFF → raw G4 codestream. PIL writes single-strip
    G4 by default for 1-bit images."""
    import struct
    if tiff[:2] == b"II":
        endian = "<"
    elif tiff[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError("not a TIFF")
    magic, = struct.unpack(f"{endian}H", tiff[2:4])
    if magic != 42:
        raise ValueError(f"unsupported TIFF magic: {magic}")
    ifd_off, = struct.unpack(f"{endian}I", tiff[4:8])
    n_entries, = struct.unpack(f"{endian}H", tiff[ifd_off:ifd_off + 2])
    strip_off = strip_n = None
    for i in range(n_entries):
        entry = tiff[ifd_off + 2 + i * 12: ifd_off + 2 + i * 12 + 12]
        tag, = struct.unpack(f"{endian}H", entry[0:2])
        # type, count, value/offset
        ttype, count, value = struct.unpack(f"{endian}HII", entry[2:12])
        if tag == 273:        # StripOffsets
            strip_off = value
        elif tag == 279:      # StripByteCounts
            strip_n = value
    if strip_off is None or strip_n is None:
        raise ValueError("TIFF missing strip offsets")
    return tiff[strip_off:strip_off + strip_n]


def _jbig2_compress(im: Image.Image) -> bytes:
    """Lossless JBIG2 embedded stream (no file header) for /JBIG2Decode."""
    from aglaia_jbig2 import encode_page_lossless
    import numpy as np
    arr = np.array(im, dtype=bool)
    # PIL '1' has 255=white; we want 1=black, 0=white.
    flat = (~arr).astype("uint8").reshape(-1)
    h, w = arr.shape
    return encode_page_lossless(bytes(flat), w, h)


def _page_dpi(row) -> float:
    try:
        return float(row["dpi"]) or 300.0
    except (KeyError, TypeError):
        return 300.0


def _make_image_xobject(pdf: "pikepdf.Pdf", *, width: int, height: int,
                        stream: bytes, pdf_filter: Name,
                        decode_parms: dict | None = None) -> "pikepdf.Object":
    obj = Stream(pdf, stream)
    obj.Type = Name.XObject
    obj.Subtype = Name.Image
    obj.Width = width
    obj.Height = height
    obj.ColorSpace = Name.DeviceGray
    obj.BitsPerComponent = 1
    obj.Filter = pdf_filter
    if decode_parms is not None:
        obj.DecodeParms = pikepdf.Dictionary(**decode_parms)
    return obj


# ── native (colour / grayscale) PDF assembly via pikepdf ────────────────
#
# JPG blobs embed losslessly as `DCTDecode` (raw stream = the JPEG bytes);
# PNGs and other formats are re-encoded as JPEG quality 90 and embedded
# the same way. Colour space is decided per row (`type` BW / GRAY / COLOR).

def _make_jpeg_xobject(pdf: "pikepdf.Pdf", *, width: int, height: int,
                       jpeg_bytes: bytes, color_space: Name) -> "pikepdf.Object":
    obj = Stream(pdf, jpeg_bytes)
    obj.Type = Name.XObject
    obj.Subtype = Name.Image
    obj.Width = width
    obj.Height = height
    obj.ColorSpace = color_space
    obj.BitsPerComponent = 8
    obj.Filter = Name.DCTDecode
    return obj


def _row_to_jpeg(row) -> tuple[bytes, int, int, Name]:
    """Best-effort: keep the row's bytes verbatim when it's already JPEG.

    For PNG / unknown rows: decode via PIL and re-encode at quality 90.
    Returns `(jpeg_bytes, width, height, color_space_name)`.
    """
    blob = bytes(row["blob"])
    fmt = (row["format"] or "").upper()
    if fmt == "JPG":
        im = Image.open(io.BytesIO(blob))
        if im.mode == "L":
            cs = Name.DeviceGray
        elif im.mode == "1":
            im = im.convert("L")
            cs = Name.DeviceGray
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            blob = buf.getvalue()
        else:
            if im.mode != "RGB":
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=90)
                blob = buf.getvalue()
            cs = Name.DeviceRGB
        return blob, im.width, im.height, cs

    im = Image.open(io.BytesIO(blob))
    if im.mode == "1" or im.mode == "L":
        im = im.convert("L")
        cs = Name.DeviceGray
    else:
        if im.mode != "RGB":
            im = im.convert("RGB")
        cs = Name.DeviceRGB
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), im.width, im.height, cs


def build_native_pdf(rows: Sequence, output_path: Path) -> bool:
    """Build a PDF whose pages embed the row blobs as DCTDecode JPEGs.

    Per row: re-use the JPEG bytes verbatim when possible (lossless),
    otherwise re-encode at quality 90. Page size matches the image's
    pixel dimensions at the row's stored DPI.
    """
    if pikepdf is None:
        raise RuntimeError(f"pikepdf required for native PDF export: {_PIKEPDF_ERR}")
    pdf = pikepdf.Pdf.new()
    pages_added = 0
    for row in rows:
        try:
            jpeg_bytes, w, h, cs = _row_to_jpeg(row)
        except Exception:
            continue
        dpi = _page_dpi(row)
        page_w_pt = 72.0 * w / dpi
        page_h_pt = 72.0 * h / dpi
        img_obj = _make_jpeg_xobject(
            pdf, width=w, height=h,
            jpeg_bytes=jpeg_bytes, color_space=cs,
        )
        contents = pikepdf.Stream(
            pdf,
            (f"q\n{page_w_pt:.4f} 0 0 {page_h_pt:.4f} 0 0 cm\n/Im0 Do\nQ\n").encode(),
        )
        page = pdf.add_blank_page(page_size=(page_w_pt, page_h_pt))
        page.Contents = contents
        page.Resources = pikepdf.Dictionary(
            XObject=pikepdf.Dictionary(Im0=img_obj),
            ProcSet=[Name.PDF, Name.ImageC, Name.ImageI, Name.ImageB],
        )
        pages_added += 1
    if pages_added == 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(output_path))
    return True


# ── invisible OCR text layer (Helvetica + WinAnsi, render mode 3) ──────


def _pdf_escape_winansi(text: str) -> bytes:
    """PDF literal-string body for Helvetica/WinAnsi.

    Non-Latin-1 codepoints are dropped (Helvetica has no glyph for them
    without an embedded TrueType subset). Special chars (\\, (, )) are
    escaped per PDF 7.3.4.2.
    """
    try:
        raw = text.encode("cp1252")
    except UnicodeEncodeError:
        # Drop unencodable characters one by one.
        out = []
        for ch in text:
            try:
                out.append(ch.encode("cp1252"))
            except UnicodeEncodeError:
                continue
        raw = b"".join(out)
    return (raw
            .replace(b"\\", b"\\\\")
            .replace(b"(", b"\\(")
            .replace(b")", b"\\)"))


def inject_ocr_layer(pdf_path: Path, ocr_per_page: list) -> None:
    """Overlay each page with an invisible Helvetica text run (render
    mode 3) so the produced PDF stays selectable / searchable.

    `ocr_per_page[i]` is `None` or a dict with `lines`, `page_w`, `page_h`.
    """
    if pikepdf is None or not any(ocr_per_page):
        return
    with pikepdf.open(str(pdf_path), allow_overwriting_input=True) as pdf:
        font_obj = pdf.make_indirect(pikepdf.Dictionary(
            Type=Name.Font, Subtype=Name.Type1,
            BaseFont=Name("/Helvetica"), Encoding=Name.WinAnsiEncoding,
        ))
        for i, page in enumerate(pdf.pages):
            ocr = ocr_per_page[i] if i < len(ocr_per_page) else None
            if not ocr or not ocr.get("lines"):
                continue
            try:
                rect = page.mediabox
                pw = float(rect[2] - rect[0])
                ph = float(rect[3] - rect[1])
            except Exception:
                continue
            img_w = float(ocr.get("page_w") or 0)
            img_h = float(ocr.get("page_h") or 0)
            if img_w <= 0 or img_h <= 0:
                continue
            sx = pw / img_w
            sy = ph / img_h
            # Ensure the page resources have an /AglaiaOCR font slot.
            try:
                res = page.Resources
            except Exception:
                res = pikepdf.Dictionary()
                page.Resources = res
            if Name.Font not in res:
                res.Font = pikepdf.Dictionary()
            res.Font.AglaiaOCR = font_obj
            # Build the appended content stream.
            buf = io.BytesIO()
            buf.write(b"q\n")
            for line in ocr["lines"]:
                bbox = line.get("bbox")
                text = line.get("text") or ""
                if not bbox or not text:
                    continue
                x0, y0, x1, y1 = bbox
                w = (x1 - x0) * sx
                h = (y1 - y0) * sy
                if w <= 0 or h <= 0:
                    continue
                fs = max(2.0, h * 0.85)
                # PDF coords: origin bottom-left, y up. Image y is from
                # the top. Place baseline near the bottom edge of the
                # bbox so descenders sit naturally.
                pdf_x = x0 * sx
                pdf_y_baseline = ph - (y1 * sy)
                escaped = _pdf_escape_winansi(text)
                if not escaped:
                    continue
                buf.write(b"BT\n")
                buf.write(f"/AglaiaOCR {fs:.2f} Tf\n".encode("ascii"))
                buf.write(b"3 Tr\n")
                buf.write(f"1 0 0 1 {pdf_x:.2f} {pdf_y_baseline:.2f} Tm\n"
                          .encode("ascii"))
                buf.write(b"(")
                buf.write(escaped)
                buf.write(b") Tj\n")
                buf.write(b"ET\n")
            buf.write(b"Q\n")
            page.contents_add(pikepdf.Stream(pdf, buf.getvalue()))
        pdf.save(str(pdf_path))


def build_bitonal_pdf(rows: Sequence, output_path: Path, *, engine: str = "jbig2") -> bool:
    """Build a PDF whose every page is one bitonal image XObject.

    `engine`: "jbig2" (lossless via aglaia_jbig2; falls back to G4 if the
    module is missing) or "g4" (CCITTFaxDecode K=-1).

    Returns True on success. Rows must already be filtered to BW images
    (mixed inputs are skipped silently per row to avoid breaking the
    export when one source isn't bitonal).
    """
    if pikepdf is None:
        raise RuntimeError(f"pikepdf required for bitonal PDF export: {_PIKEPDF_ERR}")
    if engine == "jbig2":
        # Probe the actual encoder symbol, not just `import aglaia_jbig2`:
        # the repo ships an `aglaia_jbig2/` crate dir, so a bare import
        # succeeds as a *namespace package* (when the wheel isn't built)
        # while `encode_page_lossless` is missing — which silently broke
        # JBIG2 export (build raised mid-loop → no file). Importing the
        # function forces a real availability check → clean G4 fallback.
        try:
            from aglaia_jbig2 import encode_page_lossless  # noqa: F401
        except Exception:
            import sys
            print("[pdf_export] aglaia_jbig2 encoder unavailable; "
                  "falling back to G4", file=sys.stderr)
            engine = "g4"

    pdf = pikepdf.Pdf.new()
    pages_added = 0
    for row in rows:
        if row["type"] != "BW":
            continue
        im = _row_to_pil(row)
        w, h = im.width, im.height
        dpi = _page_dpi(row)
        page_w_pt = 72.0 * w / dpi
        page_h_pt = 72.0 * h / dpi

        if engine == "jbig2":
            stream = _jbig2_compress(im)
            img_obj = _make_image_xobject(
                pdf, width=w, height=h, stream=stream,
                pdf_filter=Name.JBIG2Decode,
            )
        else:  # g4
            stream = _g4_compress(im)
            img_obj = _make_image_xobject(
                pdf, width=w, height=h, stream=stream,
                pdf_filter=Name.CCITTFaxDecode,
                decode_parms={
                    "K": -1,
                    "Columns": w,
                    "Rows": h,
                    # PIL writes TIFF G4 with Photometric=0 (white=0,
                    # black=1) — set BlackIs1 to match, otherwise PDF
                    # readers swap fg/bg and the page looks inverted.
                    "BlackIs1": True,
                },
            )

        # Minimal /Contents stream: paint the image at full page size.
        contents = pikepdf.Stream(
            pdf,
            (f"q\n{page_w_pt:.4f} 0 0 {page_h_pt:.4f} 0 0 cm\n/Im0 Do\nQ\n").encode(),
        )
        page = pdf.add_blank_page(page_size=(page_w_pt, page_h_pt))
        page.Contents = contents
        page.Resources = pikepdf.Dictionary(
            XObject=pikepdf.Dictionary(Im0=img_obj),
            ProcSet=[Name.PDF, Name.ImageB],
        )
        pages_added += 1

    if pages_added == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(output_path))
    return True
