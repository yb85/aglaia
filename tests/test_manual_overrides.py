# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Per-page manual parameter overrides — repo round-trip + frame validation.

`step_overrides` says "skip this step for this layout"; `manual_overrides`
says "run it, but with THIS value". The payload is edited field by field from
three separate editors, so `set` merges rather than replaces, and clearing one
field must leave the others alone.
"""

import os
import tempfile

from aglaia.storage.db import open_db
from aglaia.storage.repo import ManualOverrideRepo, validate_frame


def _db():
    p = os.path.join(tempfile.mkdtemp(), "t.agl")
    conn = open_db(p)
    # No real scans rows here — exercise the repo's own SQL in isolation.
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def test_round_trip_per_branch():
    r = ManualOverrideRepo(_db())
    r.set(1, "A", {"skew_deg": -1.75})
    r.set(1, "B", {"curl": {"alpha": 0.2, "beta": -0.05, "gamma": 0.03}})
    assert r.get(1, "A") == {"skew_deg": -1.75}
    assert r.get(1, "B")["curl"]["alpha"] == 0.2
    assert r.get(1, "") == {}
    assert set(r.map_for_scan(1)) == {"A", "B"}


def test_set_merges_and_none_clears_one_field():
    """Three editors write into one payload. Each must be able to set or
    clear its own field without reading the others back first."""
    r = ManualOverrideRepo(_db())
    r.set(2, "A", {"skew_deg": 1.0})
    r.set(2, "A", {"roi": [[0, 0], [10, 0], [10, 10]], "frame_wh": [10, 10]})
    assert r.get(2, "A")["skew_deg"] == 1.0        # untouched by the ROI write
    r.set(2, "A", {"skew_deg": None})
    payload = r.get(2, "A")
    assert "skew_deg" not in payload
    assert payload["roi"] == [[0, 0], [10, 0], [10, 10]]


def test_empty_payload_deletes_the_row():
    """"No override" is the absence of a row, as with `step_overrides` —
    otherwise `map_for_scan` would hand the worker empty dicts to interpret."""
    r = ManualOverrideRepo(_db())
    r.set(3, "A", {"skew_deg": 2.0})
    assert r.set(3, "A", {"skew_deg": None}) == {}
    assert r.map_for_scan(3) == {}


def test_clear_scan_removes_every_branch():
    r = ManualOverrideRepo(_db())
    r.set(4, "A", {"skew_deg": 1.0})
    r.set(4, "B", {"skew_deg": 2.0})
    r.clear_scan(4)
    assert r.map_for_scan(4) == {}


def test_frame_mismatch_drops_the_spatial_field():
    """A polygon means something only against the frame it was drawn on.
    Applying it to another size would shift it silently."""
    payload = {"skew_deg": 1.0, "roi": [[0, 0], [10, 10]], "frame_wh": [800, 600]}
    kept, dropped = validate_frame(payload, (1600, 1200))
    assert dropped == ["roi"]
    assert "roi" not in kept
    assert kept["skew_deg"] == 1.0        # not spatial — survives


def test_matching_frame_keeps_everything():
    payload = {"roi": [[0, 0], [10, 10]], "frame_wh": [800, 600]}
    kept, dropped = validate_frame(payload, (800, 600))
    assert dropped == []
    assert kept == payload


def test_payload_without_provenance_is_accepted():
    """Written before `frame_wh` existed: unvalidatable, not wrong."""
    payload = {"roi": [[0, 0], [10, 10]]}
    kept, dropped = validate_frame(payload, (1600, 1200))
    assert dropped == []
    assert kept == payload
