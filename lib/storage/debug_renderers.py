# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""On-the-fly debug-image renderers for the GUI debug viewer.

Per-processor renderers operate purely in-memory (no DB writes, no
disk artefacts). Public entry: ``render_chain_overlays(conn,
leaf_node_id)`` walks the node chain and returns one
``{label, url}`` dict per step. ``url`` is a ``data:image/png;…``
payload the GUI can decode straight into a QPixmap.

The renderer table (``_RENDERERS``) is keyed by ``processor_name``.
Each renderer takes ``(this_image, parent_image_or_None, meta_dict)``
and (optionally) ``siblings=[...]``. Default renderer just stamps a
"no debug renderer" caption on the bare image.

Adapted from the now-removed ``lib/web/routes/debug.py``. The two
production processors that actually carry rich meta — ``Trap`` and
``Dewarp`` — recompute spans + baselines on the source ink each time;
heavy (~hundreds of ms per node) but acceptable for a manually-opened
debug view.
"""
from __future__ import annotations

import base64
import json
import sqlite3
from typing import Optional

import cv2
import numpy as np

from lib.storage.repo import ImageRepo


def _png_data_url(arr: np.ndarray, max_dim: int = 2400) -> str:
    """Encode a numpy image to ``data:image/png;base64,…``.

    Downscale to keep ``max(width, height) ≤ max_dim`` so Qt-based
    viewers don't reject the decoded image under their 256 MB
    allocation cap."""
    h, w = arr.shape[:2]
    big = max(h, w)
    if big > max_dim:
        scale = max_dim / big
        new_size = (int(round(w * scale)), int(round(h * scale)))
        arr = cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(bytes(buf)).decode("ascii")


def _decode_image(blob: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _label_bar(canvas: np.ndarray, text: str,
               *, bg=(0, 0, 0), fg=(255, 255, 255)) -> np.ndarray:
    """Prepend a label strip ABOVE the canvas (does not crop content).

    Returns a new array — `(bar + canvas)` stacked vertically. Callers
    that previously mutated in place must now use the returned value.
    OpenCV's Hershey font has no glyph for `°` (renders as `??`), so we
    replace it with `deg` for ASCII-safety."""
    text = text.replace("°", "deg")
    h, w = canvas.shape[:2]
    bar_h = max(48, w // 36)
    font_scale = bar_h / 36.0
    thickness = max(2, int(font_scale * 1.6))
    text_y = int(bar_h * 0.7)
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = bg
    cv2.putText(bar, text, (12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg,
                thickness, cv2.LINE_AA)
    return np.vstack([bar, canvas])


# ── Per-processor renderers ───────────────────────────────────────


def _skew_renderer(img: np.ndarray, parent: Optional[np.ndarray],
                   meta: dict) -> list[dict]:
    angle = meta.get("skew_angle") or meta.get("skew")
    out = _to_bgr(img)
    h, w = out.shape[:2]

    gray = _to_gray(img)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = cv2.bitwise_not(bw)
    profile = bw.sum(axis=1).astype(np.float32)
    if profile.max() > 0:
        profile = profile / profile.max()
    strip_w = 100
    strip = np.full((h, strip_w, 3), 240, dtype=np.uint8)
    for y in range(h):
        x_end = int(profile[y] * (strip_w - 4))
        if x_end > 0:
            cv2.line(strip, (2, y), (2 + x_end, y), (60, 60, 200), 1)
    composite = np.hstack([out, strip])
    composite = _label_bar(
        composite,
        f"SkewFinder | angle={float(angle):+.3f}°" if angle is not None
        else "SkewFinder | angle=?")
    return [{"url": _png_data_url(composite),
             "label": "deskewed + profile"}]


def _page_renderer(img: np.ndarray, parent: Optional[np.ndarray],
                     meta: dict, siblings: Optional[list[dict]] = None) -> list[dict]:
    crop = meta.get("parent_crop_xywh")
    roi = meta.get("roi")
    page_nums = meta.get("page_nums")

    if parent is not None:
        canvas = _to_bgr(parent).copy()
        items: list[tuple[str, list[int], list, np.ndarray]] = []
        sibs = siblings or [{
            "branch_label": meta.get("branch_label") or "?",
            "meta": meta,
            "image": img,
        }]
        per_child_colors = [
            (0, 140, 240), (0, 220, 180), (240, 80, 160), (60, 60, 240),
        ]
        for s in sibs:
            s_meta = s.get("meta") or {}
            s_crop = s_meta.get("parent_crop_xywh")
            s_img = s.get("image")
            s_label = s.get("branch_label") or "?"
            if not s_crop and s_img is not None:
                try:
                    scale = 0.25
                    tpl = cv2.resize(_to_bgr(s_img),
                                     (max(8, int(s_img.shape[1] * scale)),
                                      max(8, int(s_img.shape[0] * scale))),
                                     interpolation=cv2.INTER_AREA)
                    src = cv2.resize(canvas,
                                     (max(8, int(canvas.shape[1] * scale)),
                                      max(8, int(canvas.shape[0] * scale))),
                                     interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
                    _, _, _, mloc = cv2.minMaxLoc(res)
                    fx = int(mloc[0] / scale)
                    fy = int(mloc[1] / scale)
                    ch_h, ch_w = s_img.shape[:2]
                    s_crop = [fx, fy, ch_w, ch_h]
                except Exception:
                    s_crop = None
            if s_crop:
                items.append((s_label, s_crop,
                              s_meta.get("roi") or [], s_img))

        for s_idx, (s_label, s_crop, s_roi, _) in enumerate(items):
            color = per_child_colors[s_idx % len(per_child_colors)]
            fx, fy, cw, ch = s_crop
            overlay = canvas.copy()
            cv2.rectangle(overlay, (fx, fy), (fx + cw, fy + ch),
                          color, -1)
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
            cv2.rectangle(canvas, (fx, fy), (fx + cw, fy + ch),
                          color, 6, cv2.LINE_AA)
            cv2.rectangle(canvas, (fx, fy),
                          (fx + 56, fy + 32), color, -1)
            cv2.putText(canvas, str(s_label),
                        (fx + 6, fy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if s_roi:
                pts = np.array(s_roi, dtype=np.float32).reshape(-1, 2)
                pts = pts + np.array([fx, fy], dtype=np.float32)
                pts = pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, (0, 220, 0),
                              3, cv2.LINE_AA)

        txt = f"PageDetector | parent + {len(items)} layout(s) + ROI"
        if page_nums:
            txt += f" | page#={page_nums}"
        canvas = _label_bar(canvas, txt)
        return [{"url": _png_data_url(canvas),
                 "label": "parent + layouts"}]

    canvas = _to_bgr(img).copy()
    if roi:
        pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 200, 0), 2, cv2.LINE_AA)
    txt = "PageDetector | child (no parent)"
    if page_nums:
        txt += f" | page#={page_nums}"
    canvas = _label_bar(canvas, txt)
    return [{"url": _png_data_url(canvas), "label": "child + ROI"}]


_SPAN_COLORS = [
    (255, 80, 80),   (80, 200, 80),  (80, 160, 240), (240, 200, 60),
    (200, 80, 220),  (80, 220, 220), (240, 140, 60), (160, 100, 200),
]


def _overlay_spans(canvas: np.ndarray, spans: list,
                   used_idxs: Optional[set] = None,
                   alpha: float = 0.45) -> np.ndarray:
    if not spans:
        return canvas
    overlay = canvas.copy()
    for i, span in enumerate(spans):
        if used_idxs is not None and i not in used_idxs:
            color = (160, 160, 160)
        else:
            color = _SPAN_COLORS[i % len(_SPAN_COLORS)]
        for ci in span:
            try:
                cv2.drawContours(overlay, [ci.contour], -1, color, -1)
            except Exception:
                pass
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    return canvas


def _trap_renderer(img: np.ndarray, parent: Optional[np.ndarray],
                   meta: dict) -> list[dict]:
    quad = meta.get("column_quad")
    H = meta.get("H") or (meta.get("replay_params") or {}).get("H")

    if parent is not None:
        src_canvas = _to_bgr(parent).copy()
    else:
        src_canvas = _to_bgr(img).copy()

    try:
        from lib.processors.geometry import (
            baseline_from_ink, select_per_side_anchors,
        )
        from lib.processors.utils import binarize_fixed, to_gray
        from page_dewarp.contours import get_contours
        from page_dewarp.spans import assemble_spans
        from page_dewarp.options import cfg as pd_cfg

        gray_src = to_gray(parent) if parent is not None else to_gray(img)
        bw = binarize_fixed(gray_src, 127)
        ink = cv2.bitwise_not(bw) if bw.mean() > 127 else bw

        H_img, W_img = ink.shape
        dpi = 300
        cc_h_min = max(3, int(round(dpi * 0.04)))
        cc_h_max = max(cc_h_min + 1, int(round(dpi * 0.45)))
        cc_w_min = max(2, int(round(dpi * 0.02)))
        cc_w_max = max(cc_w_min + 1, int(round(dpi * 0.60)))
        n_cc, _, _stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
        _char_h = [int(s[3]) for s in _stats[1:]
                   if cc_h_min <= s[3] <= cc_h_max
                   and cc_w_min <= s[2] <= cc_w_max]
        if len(_char_h) >= 30:
            _h_med = float(np.median(_char_h))
            kw = max(9, int(round(2.0 * _h_med)))
        else:
            _h_med = 0.0
            kw = max(9, int(round(4.0 * dpi / 25.4)))
        vbreak = max(3, int(round(_h_med / 6))) if _h_med > 0 else 3
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vbreak))
        ink_clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        morphed = cv2.morphologyEx(ink_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, morphed = cv2.threshold(morphed, 127, 255, cv2.THRESH_BINARY)
        saved = {k: getattr(pd_cfg, k) for k in (
            "TEXT_MAX_THICKNESS", "TEXT_MIN_WIDTH", "TEXT_MIN_HEIGHT",
            "EDGE_MAX_LENGTH", "EDGE_MAX_OVERLAP", "EDGE_MAX_ANGLE",
            "SPAN_MIN_WIDTH",
        )}
        if _h_med > 0:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(3.0 * _h_med)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(0.5 * _h_med)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(0.5 * _h_med)))
            pd_cfg.EDGE_MAX_LENGTH = max(20, int(round(3.0 * _h_med)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, 0.1 * _h_med)
            pd_cfg.SPAN_MIN_WIDTH = max(30, int(round(10.0 * _h_med)))
        else:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(dpi * 0.25)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(dpi * 0.10)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(dpi * 0.05)))
            pd_cfg.EDGE_MAX_LENGTH = max(100, int(round(dpi * 0.5)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, dpi * 0.02)
            pd_cfg.SPAN_MIN_WIDTH = max(30, W_img // 20)
        pd_cfg.EDGE_MAX_ANGLE = 7.5
        try:
            rgb = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
            cinfos = get_contours("trap", rgb, morphed)
            pagemask = np.full((H_img, W_img), 255, dtype=np.uint8)
            spans = assemble_spans("trap", rgb, pagemask, cinfos)
        finally:
            for k, v in saved.items():
                setattr(pd_cfg, k, v)

        baselines = []
        bbox_by_span = []
        for span in spans:
            xs, ys = [], []
            sm = np.zeros(ink.shape, dtype=np.uint8)
            for ci in span:
                x, y, w, h = ci.rect
                xs += [x, x + w]; ys += [y, y + h]
                sub = sm[y:y + h, x:x + w]
                tm = ci.mask
                if tm.dtype != np.uint8:
                    tm = tm.astype(np.uint8) * 255
                sub |= tm
            bb = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            bbox_by_span.append(bb)
            bl = baseline_from_ink(ink, bb, span_mask=sm)
            baselines.append(bl)

        valid_idxs = [i for i, bl in enumerate(baselines) if bl is not None]
        used_idxs = set()
        if len(valid_idxs) >= 3:
            bls = [baselines[i] for i in valid_idxs]
            l_idxs, r_idxs, _ = select_per_side_anchors(bls)
            for j in set(l_idxs) | set(r_idxs):
                used_idxs.add(valid_idxs[j])

        _overlay_spans(src_canvas, spans, used_idxs)

        if len(valid_idxs) >= 3:
            bls = [baselines[i] for i in valid_idxs]
            l_idxs, r_idxs, fw_idxs = select_per_side_anchors(bls)
            fw_set = set(fw_idxs)
            for j, (pL, pR) in enumerate(bls):
                if j in fw_set:
                    color = (0, 220, 60); thick = 2
                else:
                    color = (220, 120, 0); thick = 1
                cv2.line(src_canvas,
                         (int(pL[0]), int(pL[1])),
                         (int(pR[0]), int(pR[1])),
                         color, thick, cv2.LINE_AA)
    except Exception:
        pass

    if quad is not None:
        pts = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(src_canvas, [pts], True, (0, 0, 220), 3, cv2.LINE_AA)
        for i, p in enumerate(np.array(quad, dtype=int)):
            cv2.circle(src_canvas, tuple(p), 6, (0, 0, 220), -1)
            cv2.putText(src_canvas, ["TL", "TR", "BR", "BL"][i],
                        tuple(p + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 220), 2, cv2.LINE_AA)

    aspect = meta.get("recovered_aspect_w_h")
    asp_str = f" | aspect={aspect:.3f}" if aspect else ""
    src_lbl = meta.get("column_edge_source", "?")
    nb = meta.get("n_baselines", 0)
    nfw = meta.get("n_full_width", 0)
    src_canvas = _label_bar(
        src_canvas,
        f"Trap source | {src_lbl} | baselines={nb} fw={nfw}{asp_str}")

    out_canvas = _to_bgr(img).copy()
    if quad is not None and H is not None:
        try:
            H_arr = np.array(H, dtype=np.float64)
            quad_arr = np.array(quad, dtype=np.float32).reshape(-1, 1, 2)
            out_quad = cv2.perspectiveTransform(quad_arr, H_arr)
            pts = out_quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_canvas, [pts], True, (0, 0, 220), 2, cv2.LINE_AA)
        except Exception:
            pass
    roi = meta.get("roi")
    if roi:
        pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out_canvas, [pts], True, (0, 200, 0), 2, cv2.LINE_AA)
    out_canvas = _label_bar(out_canvas, "Trap output | projected quad + ROI")

    h_src = src_canvas.shape[0]
    h_out = out_canvas.shape[0]
    target_h = max(h_src, h_out)

    def _pad_to(img2, target_h):
        if img2.shape[0] == target_h:
            return img2
        pad = np.full((target_h - img2.shape[0], img2.shape[1], 3),
                      245, dtype=np.uint8)
        return np.vstack([img2, pad])
    src_padded = _pad_to(src_canvas, target_h)
    out_padded = _pad_to(out_canvas, target_h)
    sep_w = src_canvas.shape[1]
    sep = np.full((target_h, max(6, sep_w // 200), 3), 60, dtype=np.uint8)
    composite = np.hstack([src_padded, sep, out_padded])

    return [{"url": _png_data_url(composite),
             "label": "source <-> output"}]


def _dewarp_renderer(img: np.ndarray, parent: Optional[np.ndarray],
                     meta: dict) -> list[dict]:
    rp = meta.get("replay_params") or {}
    params = rp.get("params")
    page_dims = rp.get("page_dims")
    src_shape = rp.get("src_shape")
    pad_px = int(rp.get("pad_px", 0))

    if parent is not None:
        src_canvas = _to_bgr(parent).copy()
    else:
        src_canvas = _to_bgr(img).copy()
    sh, sw = src_canvas.shape[:2]

    n_grid_x = 18
    n_grid_y = 28
    grid_color = (60, 220, 60)
    grid_thickness = 1

    spans_used = []
    spans_out: list = []
    try:
        from page_dewarp.contours import get_contours
        from page_dewarp.spans import assemble_spans
        from page_dewarp.options import cfg as pd_cfg
        from lib.processors.utils import to_gray, binarize_fixed

        gray = to_gray(parent) if parent is not None else to_gray(img)
        bw = binarize_fixed(gray, 127)
        ink = cv2.bitwise_not(bw) if bw.mean() > 127 else bw
        dpi = 300

        cc_h_min = max(3, int(round(dpi * 0.04)))
        cc_h_max = max(cc_h_min + 1, int(round(dpi * 0.45)))
        cc_w_min = max(2, int(round(dpi * 0.02)))
        cc_w_max = max(cc_w_min + 1, int(round(dpi * 0.60)))
        n_cc, _, _stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
        _char_h = [int(s[3]) for s in _stats[1:]
                   if cc_h_min <= s[3] <= cc_h_max
                   and cc_w_min <= s[2] <= cc_w_max]
        if len(_char_h) >= 30:
            _h_med = float(np.median(_char_h))
            kw = max(9, int(round(2.0 * _h_med)))
        else:
            _h_med = 0.0
            kw = max(9, int(round(4.0 * 300.0 / 25.4)))
        vbreak = max(3, int(round(_h_med / 6))) if _h_med > 0 else 3
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vbreak))
        ink_clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        morphed = cv2.morphologyEx(ink_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, morphed = cv2.threshold(morphed, 127, 255, cv2.THRESH_BINARY)

        saved = {k: getattr(pd_cfg, k) for k in (
            "TEXT_MAX_THICKNESS", "TEXT_MIN_WIDTH", "TEXT_MIN_HEIGHT",
            "EDGE_MAX_LENGTH", "EDGE_MAX_OVERLAP", "EDGE_MAX_ANGLE",
            "SPAN_MIN_WIDTH",
        )}
        if _h_med > 0:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(3.0 * _h_med)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(0.5 * _h_med)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(0.5 * _h_med)))
            pd_cfg.EDGE_MAX_LENGTH = max(20, int(round(3.0 * _h_med)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, 0.1 * _h_med)
            pd_cfg.SPAN_MIN_WIDTH = max(30, int(round(10.0 * _h_med)))
        else:
            pd_cfg.TEXT_MAX_THICKNESS = max(10, int(round(dpi * 0.25)))
            pd_cfg.TEXT_MIN_WIDTH = max(8, int(round(dpi * 0.10)))
            pd_cfg.TEXT_MIN_HEIGHT = max(2, int(round(dpi * 0.01)))
            pd_cfg.EDGE_MAX_LENGTH = max(100, int(round(dpi * 0.5)))
            pd_cfg.EDGE_MAX_OVERLAP = max(2.0, dpi * 0.02)
            pd_cfg.SPAN_MIN_WIDTH = max(30, sw // 20)
        pd_cfg.EDGE_MAX_ANGLE = 7.5
        try:
            rgb = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
            cinfos = get_contours("dewarp", rgb, morphed)
            pagemask = np.full((sh, sw), 255, dtype=np.uint8)
            spans = assemble_spans("dewarp", rgb, pagemask, cinfos)
        finally:
            for k, v in saved.items():
                setattr(pd_cfg, k, v)

        spans_used = list(range(len(spans)))
        spans_out = spans
        _overlay_spans(src_canvas, spans,
                       used_idxs=set(spans_used), alpha=0.5)

        from lib.processors.geometry import (xband_baseline_per_col,
                                             span_bottom_series,
                                             fit_span_baseline)
        step_px = max(4, int(pd_cfg.SPAN_PX_PER_STEP))
        baseline_color = (0, 0, 255)
        for sp in spans:
            xs_b, ys_b, hs_b = span_bottom_series(sp)
            poly = fit_span_baseline(xs_b, ys_b, hs_b)
            polypts: list[tuple[int, int]] = []
            for ci in sp:
                tm = ci.mask
                if tm is None or tm.size == 0:
                    continue
                xmin, ymin = ci.rect[0], ci.rect[1]
                if poly is not None:
                    for col in range(0, tm.shape[1], step_px):
                        gx = int(xmin + col)
                        polypts.append((gx, int(round(float(poly(gx))))))
                    continue
                means = xband_baseline_per_col(tm)
                for col in range(0, tm.shape[1], step_px):
                    m = means[col]
                    if not np.isfinite(m):
                        continue
                    polypts.append((int(xmin + col),
                                    int(ymin + round(float(m)))))
            if len(polypts) >= 2:
                cv2.polylines(src_canvas,
                              [np.array(polypts, dtype=np.int32)],
                              False, baseline_color, 2, cv2.LINE_AA)
    except Exception:
        pass

    try:
        if params and page_dims and src_shape:
            # Model-aware projection — the library project_xy is
            # cylindrical-only and reads the global-cfg focal (1.2 in
            # this process): under a twist model / calibrated focal the
            # grid was unrelated to the fitted correction.
            from lib.processors.sheet_models import (arclength_x,
                                                     project_xy_model)
            pw, ph = float(page_dims[0]), float(page_dims[1])
            params_arr = np.asarray(params, dtype=np.float64)
            model = str(rp.get("sheet_model", "cylindrical"))
            n_modes = int(rp.get("spline_modes", 0))
            model_dims = rp.get("model_dims") or [pw, ph]
            focal = float(rp.get("focal_length", 1.2))
            support = rp.get("support_x")
            support_y = rp.get("support_y")
            support_decay = rp.get("support_decay")
            grading = float(rp.get("knot_grading", 1.0))
            flip = bool(rp.get("binding_flip", False))
            if rp.get("arc_len"):
                # Match the remap's arc-length-uniform x sampling.
                axs, asd = arclength_x(params_arr, pw, model=model,
                                       n_modes=n_modes,
                                       model_dims=model_dims,
                                       support=support,
                                       support_decay=support_decay,
                                       grading=grading, flip=flip)
                gx = np.interp(np.linspace(0.0, float(asd[-1]), n_grid_x),
                               asd, axs)
            else:
                gx = np.linspace(0, pw, n_grid_x)
            gy = np.linspace(0, ph, n_grid_y)
            xs, ys = np.meshgrid(gx, gy)
            xy = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
            proj_norm = project_xy_model(
                xy, params_arr, model=model, n_modes=n_modes,
                model_dims=model_dims, focal_length=focal,
                support=support, support_y=support_y,
                support_decay=support_decay,
                grading=grading, flip=flip).reshape(-1, 2)
            src_H, src_W = int(src_shape[0]), int(src_shape[1])
            half_max = max(src_H, src_W) / 2.0
            cx, cy = src_W / 2.0, src_H / 2.0
            pixel = np.empty_like(proj_norm)
            pixel[:, 0] = proj_norm[:, 0] * half_max + cx
            pixel[:, 1] = proj_norm[:, 1] * half_max + cy
            pixel = pixel - np.array([pad_px, pad_px])
            lattice = pixel.reshape(n_grid_y, n_grid_x, 2)
            for row in lattice:
                cv2.polylines(src_canvas, [row.astype(np.int32)],
                              False, grid_color,
                              grid_thickness, cv2.LINE_AA)
            for col in lattice.transpose(1, 0, 2):
                cv2.polylines(src_canvas, [col.astype(np.int32)],
                              False, grid_color,
                              grid_thickness, cv2.LINE_AA)
    except Exception:
        pass

    span_widths = []
    for sp in spans_out:
        xs: list[float] = []
        for c in sp:
            xs.extend(c.local_xrng)
        if xs:
            span_widths.append(max(xs) - min(xs))
    med_w = int(np.median(span_widths)) if span_widths else 0
    src_canvas = _label_bar(
        src_canvas,
        f"Dewarp source | warped | spans={len(spans_used)} "
        f"med_w={med_w}px")

    out_canvas = _to_bgr(img).copy()
    oh, ow = out_canvas.shape[:2]
    gxs = np.linspace(0, ow - 1, n_grid_x).astype(np.int32)
    gys = np.linspace(0, oh - 1, n_grid_y).astype(np.int32)
    for x in gxs:
        cv2.line(out_canvas, (int(x), 0), (int(x), oh - 1),
                 grid_color, grid_thickness, cv2.LINE_AA)
    for y in gys:
        cv2.line(out_canvas, (0, int(y)), (ow - 1, int(y)),
                 grid_color, grid_thickness, cv2.LINE_AA)
    suc = meta.get("dewarp_success") or meta.get("success")
    suc_str = "ok" if suc else "FALLBACK"
    out_canvas = _label_bar(
        out_canvas, f"Dewarp output | unwarped | {suc_str}")

    target_h = max(src_canvas.shape[0], out_canvas.shape[0])

    def _pad_to(arr, target_h):
        if arr.shape[0] == target_h:
            return arr
        pad = np.full((target_h - arr.shape[0], arr.shape[1], 3),
                      245, dtype=np.uint8)
        return np.vstack([arr, pad])
    src_p = _pad_to(src_canvas, target_h)
    out_p = _pad_to(out_canvas, target_h)
    sep_w = src_canvas.shape[1]
    sep = np.full((target_h, max(6, sep_w // 200), 3), 60, dtype=np.uint8)
    side_by_side = np.hstack([src_p, sep, out_p])

    return [{"url": _png_data_url(side_by_side),
             "label": "source <-> output"}]


def _default_renderer(img: np.ndarray, parent: Optional[np.ndarray],
                      meta: dict) -> list[dict]:
    canvas = _to_bgr(img)
    canvas = _label_bar(canvas, "no debug renderer for this step")
    return [{"url": _png_data_url(canvas), "label": "output"}]


_RENDERERS: dict[str, callable] = {
    "SkewFinder": _skew_renderer,
    "PageDetector": _page_renderer,
    "TrapezoidalCorrection": _trap_renderer,
    "PageDewarper": _dewarp_renderer,
}


# ── DB helpers + entry point ──────────────────────────────────────


def _fetch_parent_image(conn: sqlite3.Connection,
                        parent_id: Optional[int]) -> Optional[np.ndarray]:
    if not parent_id:
        return None
    row = conn.execute(
        "SELECT i.blob FROM nodes n JOIN images i ON i.id = n.image_id "
        "WHERE n.id = ?",
        (parent_id,),
    ).fetchone()
    if row is None:
        return None
    return _decode_image(bytes(row["blob"]))


def _fetch_siblings(conn: sqlite3.Connection, parent_id: Optional[int],
                    step_idx: int) -> list[dict]:
    if parent_id is None:
        return []
    rows = conn.execute(
        "SELECT n.id, n.branch_label, n.meta_json, i.blob "
        "FROM nodes n JOIN images i ON i.id = n.image_id "
        "WHERE n.parent_id = ? AND n.step_idx = ? "
        "ORDER BY n.branch_label",
        (parent_id, step_idx),
    ).fetchall()
    out = []
    for r in rows:
        try:
            m = json.loads(r["meta_json"]) if r["meta_json"] else {}
        except Exception:
            m = {}
        out.append({
            "branch_label": r["branch_label"],
            "meta": m,
            "image": _decode_image(bytes(r["blob"])),
        })
    return out


def _walk_chain(conn: sqlite3.Connection, leaf_node_id: int) -> list[dict]:
    chain: list[dict] = []
    cur_id = leaf_node_id
    seen: set[int] = set()
    while cur_id and cur_id not in seen:
        seen.add(int(cur_id))
        row = conn.execute(
            "SELECT n.id, n.processor_name, n.step_name, n.branch_label, "
            "       n.image_id, n.meta_json, n.scan_id, n.parent_id, "
            "       n.step_idx "
            "  FROM nodes n WHERE n.id = ?",
            (cur_id,),
        ).fetchone()
        if row is None:
            break
        chain.append(dict(row))
        cur_id = row["parent_id"]
    chain.reverse()
    return chain


def _render_one(conn: sqlite3.Connection, node: dict) -> list[dict]:
    img_row = ImageRepo(conn).get(node["image_id"])
    if img_row is None:
        return []
    img = _decode_image(bytes(img_row["blob"]))
    parent_img = _fetch_parent_image(conn, node.get("parent_id"))
    siblings = _fetch_siblings(conn, node.get("parent_id"),
                               node.get("step_idx"))
    meta = {}
    if node.get("meta_json"):
        try:
            meta = json.loads(node["meta_json"])
        except Exception:
            meta = {}
    proc = node.get("processor_name") or ""
    renderer = _RENDERERS.get(proc, _default_renderer)
    try:
        try:
            images = renderer(img, parent_img, meta, siblings=siblings)
        except TypeError:
            images = renderer(img, parent_img, meta)
    except Exception as e:
        images = _default_renderer(img, parent_img, meta)
        images.append({"url": "", "label": f"renderer error: {e}"})
    step_label = (node.get("step_name") or proc or "step").lstrip("0123456789_")
    for im in images:
        im["label"] = f"{step_label} | {im['label']}"
    return images


def render_chain_overlays(conn: sqlite3.Connection,
                          leaf_node_id: int) -> list[dict]:
    """Walk ``leaf_node_id`` to root, render per-step debug images.

    Returns list of ``{label, url}`` dicts ordered raw → leaf. Skips
    the root (no processor). Heavy: trap + dewarp recompute spans on
    the source ink each time."""
    chain = _walk_chain(conn, leaf_node_id)
    if not chain:
        return []
    out: list[dict] = []
    for node in chain:
        if node.get("processor_name") is None:
            continue
        out.extend(_render_one(conn, node))
    return out
