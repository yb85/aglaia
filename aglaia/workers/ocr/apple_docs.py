# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Apple **Document** OCR engine (macOS 26+, via pyobjc).

Two Vision passes, fused:

  1. ``VNRecognizeDocumentsRequest`` (macOS 26) → a structured document:
     a reading-ordered ``CRDocumentOutputRegion`` tree (blocks → lines →
     words) plus a flat transcript. We walk the tree into typed blocks
     (paragraph / heading / list / table) under ``meta.document`` so
     ``md_export`` renders real Markdown structure instead of guessing it
     from line geometry.
  2. ``VNRecognizeTextRequest`` (the classic flat path) → per-line text +
     bbox + **confidence**. This is the only reliable confidence signal:
     the ``CR*OutputRegion`` ``confidence()`` accessor returns a constant
     sentinel (``2.0`` for every region on macOS 26), so we can't gate on
     it. The flat recognizer, by contrast, cleanly scores good
     French/Latin lines ~1.00 and script it can't read (Greek — Apple
     Vision has no ``el``/``grc`` model) ~0.30–0.50.

**Confidence gate (the whole point).** Every flat line with
``confidence < gate`` is a region Vision mis-read — almost always
non-Latin script. The gate is a system param (default 0.7), resolved per
run via ``resolve_confidence_gate()``: env ``AGLAIA_OCR_CONFIDENCE_GATE``
→ SQLite ``KEY_OCR_CONFIDENCE_GATE`` → default. We crop that line's bbox
from the source
image and re-OCR it with a *complement* engine that *can* read it
(Surya reads polytonic Greek; Paddle-VL also works, less accurately).
The complement's text replaces Vision's garble both in the flat ``lines``
list AND, by bbox match, inside the ``meta.document`` tree.

Geometry note (verified empirically on macOS 26.5.1): unlike
``VNRecognizeTextRequest.boundingBox`` (bottom-left origin), the
``CRDocumentOutputRegion.boundingQuad()`` corners are normalised with a
**top-left** origin — ``topLeft().y`` is small at the top of the page. We
therefore take the quad's axis-aligned hull *without* flipping Y.

Complement selection order: explicit ``recognize(..., complement=)`` arg →
``AGLAIA_OCR_COMPLEMENT`` env → ``"surya"``. ``"none"`` disables the gate.
Fail-open: if the complement engine is unavailable or errors, Vision's
(wrong) text is kept and the failure is logged — never crash the run.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any, Optional

import numpy as np

from .engine import (
    OcrEngine, OcrResult, register, engine_log, get_engine,
    resolve_ocr_dpi, downsample_to_dpi, resolve_confidence_gate,
)

_AVAILABLE: Optional[bool] = None

# Default per-line confidence gate. A flat line below the resolved gate is
# treated as mis-read and offloaded to the complement engine. 0.7 cleanly
# separates good French/Latin (~1.0) from Greek-as-Latin garble (~0.3–0.5)
# on the athanase corpus. The live value is a system param resolved per run
# via ``resolve_confidence_gate()`` (env ``AGLAIA_OCR_CONFIDENCE_GATE`` →
# SQLite ``KEY_OCR_CONFIDENCE_GATE`` → this default).
DEFAULT_CONFIDENCE_GATE = 0.7

# Pixel padding added around a low-confidence line's bbox before cropping
# for the complement, as a fraction of the line height (re-OCR likes a
# little breathing room around the glyphs).
_CROP_PAD_FRAC = 0.15

_VALID_COMPLEMENTS = ("surya", "paddle_vl", "none")


def _check() -> bool:
    """macOS AND the macOS-26 document request is present."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    if sys.platform != "darwin":
        _AVAILABLE = False
        return False
    try:
        import Vision
        import objc  # noqa: F401
        from Foundation import NSData  # noqa: F401
        _AVAILABLE = bool(hasattr(Vision, "VNRecognizeDocumentsRequest"))
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def resolve_complement(explicit: str | None = None) -> str:
    """Pick the complement engine: explicit arg → env → ``surya``."""
    for cand in (explicit, os.environ.get("AGLAIA_OCR_COMPLEMENT")):
        if cand:
            c = str(cand).strip().lower()
            if c in _VALID_COMPLEMENTS:
                return c
    return "surya"


@register
class AppleDocsEngine(OcrEngine):
    name = "apple_docs"
    display = "Apple Document engine"
    description = ("macOS 26 structured OCR. Offloads scripts Vision "
                   "can't read (e.g. Greek) to a complement engine.")

    def __init__(self) -> None:
        self.available = _check()
        # Carried from the GUI / config so each recognize() picks the
        # right complement without re-plumbing the call signature.
        self.complement: Optional[str] = None

    def recognize(self, image_rgb: np.ndarray, languages: list[str],
                   *, src_dpi: float | None = None,
                   complement: str | None = None) -> OcrResult:
        if not self.available:
            raise RuntimeError(
                "Apple Document engine unavailable "
                "(needs macOS 26 + pyobjc-framework-vision)."
            )

        from PIL import Image

        if image_rgb.ndim == 2:
            arr = np.stack([image_rgb] * 3, axis=-1)
        elif image_rgb.shape[2] == 4:
            arr = image_rgb[:, :, :3]
        else:
            arr = image_rgb
        arr = downsample_to_dpi(arr, src_dpi or 0, resolve_ocr_dpi())
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        h, w = arr.shape[:2]

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=92)
        img_bytes = buf.getvalue()

        # ── pass 1: flat lines (the reliable confidence signal) ───────
        lines = _flat_lines(img_bytes, w, h, languages)

        # ── pass 2: structured document tree (reading order) ──────────
        document: list[dict] = []
        try:
            document = _recognize_document(img_bytes, w, h, languages)
        except Exception as e:  # best-effort; flat path stands alone
            engine_log(f"[apple_docs] structured pass failed: {e}", "warn")

        # ── confidence gate → complement engine ───────────────────────
        comp = resolve_complement(complement if complement is not None
                                   else self.complement)
        gate = resolve_confidence_gate(DEFAULT_CONFIDENCE_GATE)
        n_offloaded = 0
        comp_used: Optional[str] = None
        if comp != "none":
            n_offloaded, comp_used = _apply_complement(
                arr, lines, document, comp, languages, gate)

        meta: dict[str, Any] = {
            "recognition_level": "accurate",
            "ocr_dpi": int(resolve_ocr_dpi() or 0),
            "complement": comp,
            "complement_used": comp_used,
            "n_offloaded": n_offloaded,
            "confidence_threshold": gate,
        }
        if document:
            meta["document"] = document

        return {
            "engine": self.name,
            "languages": list(languages),
            "page_w": w,
            "page_h": h,
            "lines": lines,
            "meta": meta,
        }


# ── flat line pass ────────────────────────────────────────────────────

def _flat_lines(img_bytes: bytes, w: int, h: int,
                languages: list[str]) -> list[dict]:
    """``VNRecognizeTextRequest`` → ``[{text, bbox, confidence}]``.

    bbox is top-left-origin pixel ``[x0,y0,x1,y1]`` (Vision's boundingBox
    is bottom-left normalised; we flip Y here)."""
    import Vision
    import objc
    from Foundation import NSData

    lines: list[dict] = []
    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)
        if languages:
            req.setRecognitionLanguages_(list(languages))
        ns = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"VNImageRequestHandler failed: {err}")
        for r in (req.results() or []):
            cands = r.topCandidates_(1)
            if not cands:
                continue
            top = cands[0]
            bbox = r.boundingBox()  # bottom-left origin, normalised
            x0 = int(bbox.origin.x * w)
            y0 = int((1.0 - bbox.origin.y - bbox.size.height) * h)
            x1 = int(x0 + bbox.size.width * w)
            y1 = int(y0 + bbox.size.height * h)
            lines.append({
                "text": top.string(),
                "bbox": [x0, y0, x1, y1],
                "confidence": float(top.confidence()),
            })
    return lines


# ── structured document pass ──────────────────────────────────────────
#
# CRDocumentOutputRegion.type() codes observed on macOS 26 (Tahoe):
_TYPE_DOCUMENT = 1
_TYPE_BLOCK = 2   # reading-order text block (paragraph / heading / line group)
_TYPE_LINE = 8
_TYPE_WORD = 16


def _call(node, sel):
    fn = getattr(node, sel, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _listify(node, sel) -> list:
    v = _call(node, sel)
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _quad_bbox(quad, w: int, h: int) -> Optional[list[int]]:
    """Axis-aligned pixel ``[x0,y0,x1,y1]`` hull of a ``boundingQuad``.

    Verified on macOS 26.5.1: CR* quad corners use a **top-left** origin
    (topLeft.y small at the top of the page), so we do NOT flip Y — unlike
    the bottom-left ``VNRecognizeTextRequest.boundingBox``."""
    if quad is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for corner in ("topLeft", "topRight", "bottomLeft", "bottomRight"):
        fn = getattr(quad, corner, None)
        if fn is None:
            return None
        try:
            p = fn()
        except Exception:
            return None
        xs.append(p.x)
        ys.append(p.y)
    x0 = int(min(xs) * w)
    x1 = int(max(xs) * w)
    y0 = int(min(ys) * h)
    y1 = int(max(ys) * h)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _region_text(region) -> str:
    try:
        return str(region.text() or "").strip()
    except Exception:
        return ""


def _table_markdown(table) -> Optional[dict]:
    """Render a CRTableOutputRegion to a GitHub-flavoured Markdown table."""
    cells = _listify(table, "cells")
    if not cells:
        return None
    grid: dict[tuple[int, int], str] = {}
    max_r = max_c = 0
    for cell in cells:
        rr, cr = _call(cell, "rowRange"), _call(cell, "colRange")
        if rr is None or cr is None:
            continue
        try:
            r, c = int(rr.location), int(cr.location)
        except Exception:
            continue
        grid[(r, c)] = _region_text(cell).replace("\n", " ").replace("|", "\\|")
        max_r = max(max_r, r)
        max_c = max(max_c, c)
    if not grid:
        return None
    rows = [[grid.get((r, c), "") for c in range(max_c + 1)]
            for r in range(max_r + 1)]
    md = ["| " + " | ".join(rows[0]) + " |",
          "| " + " | ".join(["---"] * (max_c + 1)) + " |"]
    for row in rows[1:]:
        md.append("| " + " | ".join(row) + " |")
    return {"type": "table", "markdown": "\n".join(md)}


def _list_block(region, w: int, h: int) -> Optional[dict]:
    """Render a CRListOutputRegion to ``{"type":"list","items":[{text,bbox}]}``.

    Items carry per-item bbox so the complement gate can correct a single
    list item in place."""
    items = _listify(region, "items") or _listify(region, "listRegions")
    out_items: list[dict] = []
    for it in items:
        t = _region_text(it)
        if t:
            out_items.append({"text": t,
                              "bbox": _quad_bbox(_call(it, "boundingQuad"), w, h)})
    if not out_items:
        return None
    return {"type": "list", "items": out_items}


def _block_record(child, w: int, h: int) -> Optional[dict]:
    """A depth-1 text block → ``{type, text, bbox, lines:[{text,bbox}]}``.

    Carrying per-line bbox lets the complement gate replace just the one
    bad line inside an otherwise-good paragraph."""
    text = _region_text(child)
    if not text:
        return None
    lines: list[dict] = []
    for line in _listify(child, "children"):
        if _call(line, "type") != _TYPE_LINE:
            continue
        lt = _region_text(line)
        if not lt:
            continue
        lines.append({"text": lt,
                      "bbox": _quad_bbox(_call(line, "boundingQuad"), w, h)})
    return {
        "type": "block",
        "text": text,
        "bbox": _quad_bbox(_call(child, "boundingQuad"), w, h),
        "lines": lines,
    }


def _recognize_document(img_bytes: bytes, w: int, h: int,
                        languages: list[str]) -> list[dict]:
    """Run VNRecognizeDocumentsRequest and walk its top-level regions into
    a reading-ordered list of typed blocks. ``[]`` when unavailable."""
    import Vision
    import objc
    from Foundation import NSData

    if not hasattr(Vision, "VNRecognizeDocumentsRequest"):
        return []

    blocks: list[dict] = []
    with objc.autorelease_pool():
        req = Vision.VNRecognizeDocumentsRequest.alloc().init()
        ns = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns, None)
        ok, _err = handler.performRequests_error_([req], None)
        if not ok:
            return []
        results = req.results() or []
        if not results:
            return []
        obs = results[0]
        root = (obs.getCRDocumentOutputRegion()
                if hasattr(obs, "getCRDocumentOutputRegion") else None)
        if root is None:
            return []

        for child in _listify(root, "children"):
            cls = type(child).__name__
            if cls == "CRTableOutputRegion":
                tbl = _table_markdown(child)
                if tbl:
                    tbl["bbox"] = _quad_bbox(_call(child, "boundingQuad"), w, h)
                    blocks.append(tbl)
                continue
            if cls == "CRListOutputRegion":
                lst = _list_block(child, w, h)
                if lst:
                    lst["bbox"] = _quad_bbox(_call(child, "boundingQuad"), w, h)
                    blocks.append(lst)
                continue
            rec = _block_record(child, w, h)
            if rec is not None:
                blocks.append(rec)
    return blocks


# ── confidence gate → complement ──────────────────────────────────────

# Minimum crop height (px) handed to the complement engine. A single
# OCR line is often only ~40–60 px tall after downsampling; the Surya /
# Paddle VLM detectors expect a denser region and return *nothing* on a
# 1-px-thin strip. Upscaling the crop so its glyphs are tall enough makes
# the recognition-only pass reliable.
_MIN_CROP_H = 96


def _pad_crop(arr: np.ndarray, bbox, pad_frac: float = _CROP_PAD_FRAC):
    """Crop ``bbox`` (top-left ``[x0,y0,x1,y1]``) with proportional pad,
    then upscale so the line is at least ``_MIN_CROP_H`` px tall. Returns
    ``None`` when the crop is empty/degenerate."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    # Reject a degenerate source box BEFORE padding inflates it.
    if x1 - x0 < 4 or y1 - y0 < 2:
        return None
    pad = int(max(2, (y1 - y0) * pad_frac))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = arr[y0:y1, x0:x1]
    ch = crop.shape[0]
    if ch < _MIN_CROP_H:
        import cv2 as _cv2
        scale = _MIN_CROP_H / ch
        crop = _cv2.resize(
            crop, (int(round(crop.shape[1] * scale)), _MIN_CROP_H),
            interpolation=_cv2.INTER_CUBIC)
    return np.ascontiguousarray(crop)


def _iou_y_overlap(a, b) -> float:
    """Vertical-band overlap ratio of two top-left bboxes (0..1).

    Document lines and flat lines share the same Y scale; matching on the
    vertical band (plus X overlap) is robust to small bbox jitter between
    the two Vision passes."""
    if a is None or b is None:
        return 0.0
    ay0, ay1 = a[1], a[3]
    by0, by1 = b[1], b[3]
    inter = max(0, min(ay1, by1) - max(ay0, by0))
    span = max(1, min(ay1 - ay0, by1 - by0))
    yov = inter / span
    ax0, ax1 = a[0], a[2]
    bx0, bx1 = b[0], b[2]
    xinter = max(0, min(ax1, bx1) - max(ax0, bx0))
    xspan = max(1, min(ax1 - ax0, bx1 - bx0))
    xov = xinter / xspan
    return min(yov, xov)


def _replace_in_document(document: list[dict], bbox, new_text: str) -> None:
    """Find the doc line/list-item/block best matching ``bbox`` and swap
    its text in place. Block text is re-joined from its (corrected) lines
    so the paragraph stays coherent."""
    best = None
    best_ov = 0.4  # require a real overlap, not a stray nearest-neighbour
    for blk in document:
        # list items
        for it in blk.get("items", []) or []:
            ov = _iou_y_overlap(it.get("bbox"), bbox)
            if ov > best_ov:
                best_ov, best = ov, ("item", blk, it)
        # block lines
        for ln in blk.get("lines", []) or []:
            ov = _iou_y_overlap(ln.get("bbox"), bbox)
            if ov > best_ov:
                best_ov, best = ov, ("line", blk, ln)
        # whole-block fallback (single-line blocks have no .lines match)
        ov = _iou_y_overlap(blk.get("bbox"), bbox)
        if ov > best_ov and not blk.get("lines"):
            best_ov, best = ov, ("block", blk, blk)
    if best is None:
        return
    kind, blk, target = best
    target["text"] = new_text
    if kind == "line":
        lines = [ln.get("text", "") for ln in blk.get("lines", [])]
        blk["text"] = " ".join(t for t in lines if t)
    elif kind == "item":
        # nothing else to recompute — list renders items directly
        pass


def _is_degenerate(text: str) -> bool:
    """True for low-conf lines not worth offloading: page numbers, folio
    marks, lone short tokens (e.g. a running header ``THOSTAZIE``). These
    are never the Greek-block case the complement targets, and a tiny
    1-word crop makes the recognition-only VLM hallucinate. Anything with
    ≥2 words OR a space-separated phrase is kept."""
    t = (text or "").strip()
    if not t:
        return True
    # Pure number / very short folio mark.
    if t.replace(".", "").replace(",", "").isdigit():
        return True
    # A single token with no spaces — running header / folio word (e.g.
    # ``THOSTAZIE``), not a line of prose. A genuine Greek *line* the
    # complement targets always spans several space-separated words.
    if " " not in t:
        return True
    return False


# When grouping adjacent low-conf lines into a crop block, two lines join
# the same block if the vertical gap between them is ≤ this multiple of the
# taller line's height (i.e. roughly one blank line). Empirically the
# recognition-only VLM (Surya) reads a multi-line BLOCK crop reliably but
# returns NOTHING on a single thin one-line strip — so we always crop a
# block, never an isolated line.
_BLOCK_GAP_FRAC = 1.6


def _group_low_blocks(low: list[dict]) -> list[list[dict]]:
    """Cluster vertically-contiguous low-conf lines into blocks.

    Sorted by Y; a new block starts when the gap to the previous line
    exceeds ``_BLOCK_GAP_FRAC × line_height``. Each block is cropped and
    re-OCR'd as one region (a thin single-line strip defeats the VLM)."""
    ordered = sorted(low, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    prev_y1 = None
    prev_h = 0
    for ln in ordered:
        y0, y1 = ln["bbox"][1], ln["bbox"][3]
        h = max(1, y1 - y0)
        if cur and prev_y1 is not None:
            gap = y0 - prev_y1
            if gap > _BLOCK_GAP_FRAC * max(prev_h, h):
                blocks.append(cur)
                cur = []
        cur.append(ln)
        prev_y1, prev_h = y1, h
    if cur:
        blocks.append(cur)
    return blocks


def _union_bbox(lines: list[dict]) -> list[int]:
    xs0 = min(ln["bbox"][0] for ln in lines)
    ys0 = min(ln["bbox"][1] for ln in lines)
    xs1 = max(ln["bbox"][2] for ln in lines)
    ys1 = max(ln["bbox"][3] for ln in lines)
    return [xs0, ys0, xs1, ys1]


def _split_block_text(text: str, n_lines: int) -> list[str]:
    """Split a block's complement text across its ``n_lines`` source lines.

    The VLM returns the block as one (possibly newline-joined) string; we
    honour its own line breaks when the count matches, else fall back to
    putting everything on the first line and blanking the rest (the
    document/flat splice then re-joins per block, so wording survives)."""
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    if n_lines <= 1:
        return [text.strip()]
    if len(parts) == n_lines:
        return parts
    # Count mismatch: keep the whole block on the first line.
    return [text.strip()] + [""] * (n_lines - 1)


def _apply_complement(arr: np.ndarray, lines: list[dict],
                      document: list[dict], comp: str,
                      languages: list[str],
                      gate: float = DEFAULT_CONFIDENCE_GATE,
                      ) -> tuple[int, Optional[str]]:
    """Group low-confidence flat lines into vertically-contiguous BLOCKS,
    re-OCR each block crop with ``comp``, and splice the result back into
    both ``lines`` and ``document``. Block (not per-line) cropping is
    essential — the recognition-only VLM returns nothing on a thin
    one-line strip but reads a multi-line block cleanly. Returns
    ``(n_offloaded, complement_used_or_None)``."""
    low = [ln for ln in lines
           if float(ln.get("confidence", 1.0)) < gate
           and ln.get("bbox")
           and not _is_degenerate(ln.get("text", ""))]
    if not low:
        return 0, None

    try:
        engine = get_engine(comp)
    except Exception as e:
        engine_log(f"[apple_docs] complement {comp!r} unavailable: {e}", "warn")
        return 0, None
    if not getattr(engine, "available", False):
        engine_log(
            f"[apple_docs] complement {comp!r} not available "
            f"(weights/deps missing) — keeping Vision text for "
            f"{len(low)} low-conf line(s).", "warn")
        return 0, None

    blocks = _group_low_blocks(low)
    engine_log(
        f"[apple_docs] {len(low)} low-conf line(s) (<{gate:.2f}) "
        f"in {len(blocks)} block(s) → complement={comp}", "info")

    n = 0
    for blk_lines in blocks:
        crop = _pad_crop(arr, _union_bbox(blk_lines))
        if crop is None:
            continue
        try:
            res = engine.recognize(crop, languages)
        except Exception as e:
            engine_log(
                f"[apple_docs] complement recognize failed on a block: {e} "
                f"— keeping Vision text.", "warn")
            continue
        new_text = "\n".join(
            (r.get("text") or "").strip()
            for r in (res.get("lines") or [])
            if (r.get("text") or "").strip()
        ).strip()
        if not new_text:
            continue
        per_line = _split_block_text(new_text, len(blk_lines))
        # Order block lines top→bottom to align with the VLM's reading order.
        blk_sorted = sorted(blk_lines, key=lambda ln: ln["bbox"][1])
        for ln, txt in zip(blk_sorted, per_line):
            if not txt:
                continue
            engine_log(
                f"[apple_docs]   conf={ln['confidence']:.2f} "
                f"{ln['text'][:28]!r} → {txt[:28]!r}", "info")
            ln["text"] = txt
            ln["confidence"] = max(float(ln.get("confidence", 0.0)), 0.99)
            ln["complement"] = comp
            if document:
                _replace_in_document(document, ln["bbox"], txt)
            n += 1
    return n, (comp if n else None)
