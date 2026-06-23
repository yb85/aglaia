# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Per-node debug-image helpers used by ``DebugViewerTab``.

Extracted from the now-removed ``aglaia.web.routes.debug`` module so the
GUI debug viewer still has its chain-walk + per-step image helpers
without dragging the (deleted) FastAPI web layer along.

Intentionally minimal: ``_render_one`` returns the node's stored image
as a single ``data:`` URL. The richer per-processor renderers that
the old web debug panel had (skew overlay, layout boxes, dewarp grid)
are not ported here — they were ~600 lines of side-quest code only the
web UI used. If the GUI grows back a need for them, restore from
commit ``1774dac~1:aglaia/web/routes/debug.py`` and wire the renderer
table the same way.
"""
from __future__ import annotations

import base64
import sqlite3
from typing import Any


def _walk_chain(conn: sqlite3.Connection, leaf_node_id: int) -> list[dict]:
    """Walk ``parent_id`` back to root, return ``[root, …, leaf]``
    list of node-row dicts."""
    chain: list[dict] = []
    cur_id = leaf_node_id
    seen: set[int] = set()
    while cur_id and cur_id not in seen:
        seen.add(int(cur_id))
        row = conn.execute(
            """
            SELECT n.id, n.processor_name, n.step_name, n.branch_label,
                   n.image_id, n.meta_json, n.scan_id, n.parent_id,
                   n.step_idx
              FROM nodes n WHERE n.id = ?
            """,
            (cur_id,),
        ).fetchone()
        if row is None:
            break
        chain.append(dict(row))
        cur_id = row["parent_id"]
    chain.reverse()
    return chain


def _render_one(conn: sqlite3.Connection, node: dict) -> list[dict]:
    """Return ``[{label, url, meta}]`` for ``node``.

    ``url`` is a ``data:image/...;base64,…`` payload of the node's
    stored image. Caller (``DebugViewerTab``) decodes it back to a
    QPixmap for the scrollable strip. ``meta`` is the decoded
    ``meta_json`` dict so the GUI can paint per-processor overlays
    (ROI polygon, skew angle text, …) on top of the bare image."""
    image_id = node.get("image_id")
    if not image_id:
        return []
    img_row = conn.execute(
        "SELECT blob, format FROM images WHERE id = ?",
        (int(image_id),),
    ).fetchone()
    if img_row is None:
        return []
    blob = bytes(img_row["blob"])
    fmt = (img_row["format"] or "png").lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "tif": "image/tiff",
            "tiff": "image/tiff", "webp": "image/webp",
            "pbm": "image/x-portable-bitmap"}.get(fmt, "image/png")
    b64 = base64.b64encode(blob).decode("ascii")
    url = f"data:{mime};base64,{b64}"
    label = (node.get("step_name")
              or node.get("processor_name")
              or "step")
    label = str(label).lstrip("0123456789_")
    meta: dict[str, Any] = {}
    raw = node.get("meta_json")
    if raw:
        try:
            import json as _json
            meta = _json.loads(raw) or {}
        except Exception:
            meta = {}
    return [{
        "url": url,
        "label": label,
        "meta": meta,
        "processor": node.get("processor_name") or "",
    }]
