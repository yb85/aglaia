# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Cloud OCR engine — Mistral Document AI (``/v1/ocr``).

Unlike the on-device engines (which OCR one page image at a time), this
engine is **whole-document**: the whole selected page set is assembled
into a single PDF, uploaded once, OCR'd by ``mistral-ocr-latest`` and the
per-page markdown spliced back out. That matches how Mistral bills and
performs best (one upload, ``pages[]`` back).

Flow (``mistralai`` 1.x SDK):

    client.files.upload(file={"file_name", "content"}, purpose="ocr")
    client.files.get_signed_url(file_id=...)
    client.ocr.process(model="mistral-ocr-latest",
                       document={"type":"document_url","document_url": url})

The uploaded PDF reuses Aglaïa's own PDF assembly: an all-bitonal page
set is encoded **CCITT G4** (the same codec as our exported scans);
anything with a colour/grey page falls back to a JPEG (DCTDecode) PDF.

The result per page is stored as a normal ``OcrResult`` with the Mistral
markdown under ``meta["markdown"]`` (``md_export`` prefers it verbatim
over the geometric heuristics). ``lines`` carries a single full-page line
so existing per-branch line counts / search still work.

Key handling lives in ``aglaia/app_data/secrets.py`` (env → OS keychain →
``.env`` fallback). The engine never logs or stores the key.

``whole_doc = True`` tells ``OcrWorker`` to hand the engine every selected
page in one ``recognize_batch`` call instead of chunking.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np

from .engine import OcrEngine, OcrResult, register, engine_log

MODEL = "mistral-ocr-latest"

# Mistral OCR list price: ~1000 pages per US dollar for mistral-ocr-latest
# (https://mistral.ai/news/mistral-ocr — "1000 pages / $"). Used only for a
# pre-flight cost *estimate* in the UI; the real bill is per Mistral's
# metering. Update if the published price changes.
PRICE_PER_PAGE_USD = 0.001

# Mistral exposes no public account-balance / remaining-credit API endpoint
# (only per-response rate-limit headers). So the UI can't show live credit —
# it points the user at the console instead.
CONSOLE_URL = "https://console.mistral.ai/"

# Mistral Document AI hard limits per uploaded document: 1000 pages and a
# 50 MB file. When the selected page set exceeds either, we TRUNCATE the
# upload to the leading pages that fit, OCR those, and flag the rest as
# "truncated" so OcrWorker leaves them pending — the user re-runs to
# continue. (We truncate rather than silently splitting so the cost
# estimate / page mapping stay simple and predictable.)
MAX_PAGES = 1000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _mistralai_available() -> bool:
    try:
        import mistralai  # noqa: F401
        return True
    except Exception:
        return False


def _is_bitonal(arr: np.ndarray) -> bool:
    """True when the RGB page is pure black/white (Aglaïa binarizer output)
    — so it can ride the lossless CCITT G4 path. Cheap unique-value probe."""
    try:
        vals = np.unique(arr)
    except Exception:
        return False
    return vals.size <= 2 and set(int(v) for v in vals.tolist()) <= {0, 255}


def _images_to_pdf(images_rgb: list[np.ndarray],
                   dpis: list[float], out_path: Path) -> bool:
    """Assemble the page images into one PDF, reusing pdf_export's builders.

    All-bitonal → CCITT G4 (matches our scans). Otherwise → JPEG/native so
    colour/grey figures survive. Returns False if no page was added."""
    from PIL import Image
    from aglaia.workers import pdf_export

    rows: list[dict] = []
    all_bw = True
    for arr, dpi in zip(images_rgb, dpis):
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        if _is_bitonal(arr):
            im.convert("1").save(buf, format="PNG")
            rows.append({"blob": buf.getvalue(), "dpi": float(dpi or 0),
                         "type": "BW", "format": "PNG",
                         "width": im.width, "height": im.height})
        else:
            all_bw = False
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=90)
            rows.append({"blob": buf.getvalue(), "dpi": float(dpi or 0),
                         "type": "COLOR", "format": "JPG",
                         "width": im.width, "height": im.height})

    if all_bw:
        return pdf_export.build_bitonal_pdf(rows, out_path, engine="g4")
    return pdf_export.build_native_pdf(rows, out_path)


@register
class MistralCloudEngine(OcrEngine):
    name = "mistral_cloud"
    display = "Cloud OCR (Mistral)"
    description = ("Cloud OCR via Mistral. Clean per-page Markdown, "
                   "any script. Needs an API key.")

    # Tell OcrWorker to send every selected page in one call (one upload).
    whole_doc = True

    def __init__(self) -> None:
        # Availability is purely "is the SDK importable"; the key is
        # checked at run time so the card stays selectable (and the user
        # can be pointed at the key field) even before a key is set.
        self.available = _mistralai_available()

    # Single-image path delegates to the batch path so callers that only
    # have one page still work.
    def recognize(self, image_rgb, languages: list[str],
                  *, src_dpi: float | None = None) -> OcrResult:
        return self.recognize_batch(
            [image_rgb], languages,
            src_dpis=[src_dpi] if src_dpi is not None else None)[0]

    def recognize_batch(self, images_rgb, languages: list[str],
                        *, src_dpis: list[float] | None = None
                        ) -> list[OcrResult]:
        if not self.available:
            raise RuntimeError(
                "Cloud OCR unavailable — install the extra: "
                "`uv sync --extra cloud` (mistralai).")

        from aglaia.app_data.secrets import get_mistral_api_key
        api_key = get_mistral_api_key()
        if not api_key:
            raise RuntimeError(
                "No Mistral API key. Set it in the OCR tab's Cloud card "
                "(stored in your OS keychain), or export MISTRAL_API_KEY.")

        images = list(images_rgb)
        dpis = list(src_dpis) if src_dpis is not None else [0.0] * len(images)
        n = len(images)
        if n == 0:
            return []

        # 1. Build one PDF, truncating to fit Mistral's 1000-page / 50 MB
        #    limits. n_sent = how many leading pages actually went up.
        with tempfile.TemporaryDirectory(prefix="aglaia-mistral-") as td:
            pdf_path = Path(td) / "aglaia-ocr.pdf"
            pdf_bytes, n_sent = self._build_capped_pdf(images, dpis, pdf_path)
            if n_sent == 0:
                raise RuntimeError("Failed to assemble PDF for upload.")

            if n_sent < n:
                engine_log(
                    f"[mistral_cloud] TRUNCATED: {n} page(s) exceed Mistral's "
                    f"{MAX_PAGES}-page / {MAX_UPLOAD_BYTES // (1024*1024)} MB "
                    f"limit — sending the first {n_sent}. Re-run OCR to "
                    f"continue the remaining {n - n_sent}.", "warn")
            engine_log(
                f"[mistral_cloud] uploading {n_sent} page(s), "
                f"{len(pdf_bytes) / 1024:.0f} KiB → {MODEL}", "info")
            pages = self._ocr_pdf(api_key, pdf_bytes)

        self._warn_mismatch(n_sent, len(pages))
        dims = [(int(img.shape[1]), int(img.shape[0])) for img in images]
        return self._assemble_results(dims, n_sent, pages, languages)

    # ── low-memory whole-document path (used by OcrWorker) ────────────
    def recognize_rows(self, rows, languages: list[str]) -> list[OcrResult]:
        """Whole-document OCR straight from stored image blobs.

        ``rows``: lightweight dicts (``blob``/``dpi``/``type``/``format``/
        ``width``/``height``) — NOT decoded RGB arrays. The PDF is assembled
        from the blobs a page at a time, so peak memory stays at tens of MB
        on big projects (vs GBs when holding every page as RGB). Returns one
        ``OcrResult`` per row, in order: Mistral page *i* ↔ row *i* (the
        truncated tail is flagged so OcrWorker keeps those pending)."""
        if not self.available:
            raise RuntimeError(
                "Cloud OCR unavailable — install the extra: "
                "`uv sync --extra cloud` (mistralai).")
        from aglaia.app_data.secrets import get_mistral_api_key
        api_key = get_mistral_api_key()
        if not api_key:
            raise RuntimeError(
                "No Mistral API key. Set it in the OCR tab's Cloud card "
                "(stored in your OS keychain), or export MISTRAL_API_KEY.")
        rows = list(rows)
        n = len(rows)
        if n == 0:
            return []

        with tempfile.TemporaryDirectory(prefix="aglaia-mistral-") as td:
            pdf_path = Path(td) / "aglaia-ocr.pdf"
            pdf_bytes, n_sent = self._build_capped_pdf_from_rows(rows, pdf_path)
            if n_sent == 0:
                raise RuntimeError("Failed to assemble PDF for upload.")
            if n_sent < n:
                engine_log(
                    f"[mistral_cloud] TRUNCATED: {n} page(s) exceed Mistral's "
                    f"{MAX_PAGES}-page / {MAX_UPLOAD_BYTES // (1024*1024)} MB "
                    f"limit — sending the first {n_sent}. Re-run OCR to "
                    f"continue the remaining {n - n_sent}.", "warn")
            engine_log(
                f"[mistral_cloud] uploading {n_sent} page(s), "
                f"{len(pdf_bytes) / 1024:.0f} KiB → {MODEL}", "info")
            pages = self._ocr_pdf(api_key, pdf_bytes)

        self._warn_mismatch(n_sent, len(pages))
        dims = [(int(r.get("width") or 0), int(r.get("height") or 0))
                for r in rows]
        return self._assemble_results(dims, n_sent, pages, languages)

    def _warn_mismatch(self, n_sent: int, n_got: int) -> None:
        if n_got != n_sent:
            engine_log(
                f"[mistral_cloud] page count mismatch: sent {n_sent}, "
                f"got {n_got} — aligning by index.", "warn")

    def _assemble_results(self, dims, n_sent: int, pages: list[str],
                          languages: list[str]) -> list[OcrResult]:
        """Per-page → per-result splice. ``dims`` is ``[(w, h), …]`` in the
        upload order; Mistral page *i* maps to ``dims[i]`` (which is the
        i-th selected scan — page 0,1,2 may be scans 12,45,67). Pages beyond
        ``n_sent`` were truncated and get flagged for OcrWorker."""
        results: list[OcrResult] = []
        for i, (w, h) in enumerate(dims):
            base: OcrResult = {
                "engine": self.name, "languages": list(languages),
                "page_w": int(w), "page_h": int(h),
            }
            if i >= n_sent:
                base["lines"] = []
                base["meta"] = {
                    "source": "mistral", "model": MODEL, "truncated": True,
                    "reason": (f"exceeds Mistral {MAX_PAGES}-page / "
                               f"{MAX_UPLOAD_BYTES // (1024*1024)} MB limit"),
                }
            else:
                md = pages[i] if i < len(pages) else ""
                line = {"text": md, "bbox": (0, 0, int(w), int(h)),
                        "confidence": 1.0}
                base["lines"] = [line] if md else []
                base["meta"] = {"source": "mistral", "model": MODEL,
                                "markdown": md}
            results.append(base)
        return results

    def _build_capped_pdf_from_rows(self, rows, pdf_path: Path
                                    ) -> tuple[bytes, int]:
        """Like ``_build_capped_pdf`` but assembles directly from stored
        blobs via ``pdf_export`` — decodes one page at a time (low peak
        memory). All-bitonal rows → CCITT G4 (Mistral-accepted, matches our
        scans); any colour/grey row → native JPEG PDF."""
        from aglaia.workers import pdf_export
        all_bw = all((r.get("type") == "BW") for r in rows)
        keep = min(len(rows), MAX_PAGES)
        while keep > 0:
            sub = rows[:keep]
            ok = (pdf_export.build_bitonal_pdf(sub, pdf_path, engine="g4")
                  if all_bw else pdf_export.build_native_pdf(sub, pdf_path))
            if not ok:
                return b"", 0
            size = pdf_path.stat().st_size
            if size <= MAX_UPLOAD_BYTES or keep == 1:
                return pdf_path.read_bytes(), keep
            guess = int(keep * MAX_UPLOAD_BYTES / size)
            keep = min(keep - 1, max(1, guess))
        return b"", 0

    def _build_capped_pdf(self, images, dpis, pdf_path: Path
                          ) -> tuple[bytes, int]:
        """Assemble the leading pages that fit Mistral's page+size caps.

        Returns ``(pdf_bytes, n_sent)``. First clamps to ``MAX_PAGES``, then
        shrinks the page count until the file is ``<= MAX_UPLOAD_BYTES``
        (ratio-guided so it converges in a couple of rebuilds). ``n_sent``
        is the number of leading pages actually written."""
        keep = min(len(images), MAX_PAGES)
        while keep > 0:
            if not _images_to_pdf(list(images[:keep]),
                                  list(dpis[:keep]), pdf_path):
                return b"", 0
            size = pdf_path.stat().st_size
            if size <= MAX_UPLOAD_BYTES or keep == 1:
                return pdf_path.read_bytes(), keep
            # Over the byte cap — drop pages. Guess by the size ratio but
            # always decrease by at least one so we can't loop forever.
            guess = int(keep * MAX_UPLOAD_BYTES / size)
            keep = min(keep - 1, max(1, guess))
        return b"", 0

    # ── Mistral round-trip ────────────────────────────────────────────
    def _ocr_pdf(self, api_key: str, pdf_bytes: bytes) -> list[str]:
        """Upload the PDF, OCR it, return per-page markdown (page order)."""
        from mistralai import Mistral

        client = Mistral(api_key=api_key)
        uploaded = client.files.upload(
            file={"file_name": "aglaia-ocr.pdf", "content": pdf_bytes},
            purpose="ocr",
        )
        signed = client.files.get_signed_url(file_id=uploaded.id)
        resp = client.ocr.process(
            model=MODEL,
            document={"type": "document_url", "document_url": signed.url},
            include_image_base64=False,
        )
        out: list[str] = []
        for pg in (getattr(resp, "pages", None) or []):
            out.append(getattr(pg, "markdown", "") or "")
        return out
