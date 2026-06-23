# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
"""Human-readable elapsed-time strings ("3 minutes ago") for ISO8601
timestamps — used by the Mistral OCR card and Jobs tab."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _parse(iso: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def time_ago(iso_timestamp: Optional[str], *, now: Optional[datetime] = None
             ) -> str:
    """ISO8601 → "just now" / "5 minutes ago" / "2 hours ago" / "3 days ago".

    ``now`` is injectable for tests. Returns "" for a missing/unparseable
    timestamp."""
    if not iso_timestamp:
        return ""
    dt = _parse(iso_timestamp)
    if dt is None:
        return str(iso_timestamp)[:16]
    ref = now or datetime.now(timezone.utc)
    secs = int((ref - dt).total_seconds())
    if secs < 0:
        secs = 0
    if secs < 45:
        return "just now"
    for limit, div, unit in (
        (3600, 60, "minute"),
        (86400, 3600, "hour"),
        (2592000, 86400, "day"),
        (31536000, 2592000, "month"),
    ):
        if secs < limit:
            n = max(1, secs // div)
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    n = max(1, secs // 31536000)
    return f"{n} year{'s' if n != 1 else ''} ago"
