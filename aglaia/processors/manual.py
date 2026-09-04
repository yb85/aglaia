# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Reading a page's manual parameter overrides inside a processor (M9).

`IntegratedProcessingChain.run_pipeline` puts the layout's payload on the
buffer as ``meta["manual_overrides"]`` for the duration of one `process()`
call. This module is the one place that unpacks it, so every processor
validates and reports an override the same way.

Two rules the callers must not each reinvent:

- **A spatial override is only valid on the frame it was drawn on.** A polygon
  applied to a frame of another size is silently shifted or rescaled, and a
  page that is quietly wrong is worse than one the pipeline decided alone. So
  pass `frame_wh` and a mismatch drops the field.
- **An honoured override is stamped.** `meta["manual"]` lists the fields the
  step took from the user, so the debug view can say so and a reader of the
  node can tell a measurement from a decision.
"""
from __future__ import annotations

from typing import Any, Optional

#: Fields whose meaning depends on the frame they were drawn on.
SPATIAL_FIELDS = ("roi",)


def manual_value(img_buf, field: str, *, frame_wh=None) -> Optional[Any]:
    """The user's override for `field` on this buffer, or ``None``.

    `frame_wh` is the `(w, h)` of the image the caller is about to apply the
    value to. For a spatial field it is checked against the frame the edit was
    made on, and a mismatch returns ``None`` with a note in
    ``meta["manual_dropped"]``.
    """
    meta = getattr(img_buf, "meta", None) or {}
    payload = meta.get("manual_overrides") or {}
    if not isinstance(payload, dict):
        return None
    value = payload.get(field)
    if value is None:
        return None
    if field in SPATIAL_FIELDS and frame_wh is not None:
        stored = payload.get("frame_wh")
        if stored:
            try:
                same = (int(stored[0]) == int(frame_wh[0])
                        and int(stored[1]) == int(frame_wh[1]))
            except Exception:
                same = True          # unreadable provenance — unvalidatable
            if not same:
                meta.setdefault("manual_dropped", []).append(field)
                return None
    return value


def stamp_manual(img_buf, field: str) -> None:
    """Record that `field` came from the user, not from the estimator."""
    if getattr(img_buf, "meta", None) is None:
        img_buf.meta = {}
    fields = img_buf.meta.setdefault("manual", [])
    if field not in fields:
        fields.append(field)
