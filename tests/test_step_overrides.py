# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Per-page processor disable — StepOverrideRepo round-trip + migration."""

import os
import tempfile

from lib.storage.db import open_db
from lib.storage.repo import StepOverrideRepo


def _db():
    p = os.path.join(tempfile.mkdtemp(), "t.agl")
    conn = open_db(p)
    # No real scans rows here — exercise the repo's own SQL in isolation.
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def test_set_disable_and_map():
    conn = _db()
    r = StepOverrideRepo(conn)
    r.set(1, "", 3, True)       # trunk step
    r.set(1, "A", 4, True)      # layout A
    r.set(1, "B", 4, True)      # layout B
    assert r.map_for_scan(1) == {("", 3), ("A", 4), ("B", 4)}
    assert r.is_disabled(1, "", 3)
    assert r.is_disabled(1, "A", 4)
    assert not r.is_disabled(1, "A", 5)


def test_enable_deletes_row():
    conn = _db()
    r = StepOverrideRepo(conn)
    r.set(7, "A", 2, True)
    r.set(7, "A", 2, False)     # re-enable removes the override
    assert r.map_for_scan(7) == set()
    assert not r.is_disabled(7, "A", 2)


def test_disable_is_idempotent():
    conn = _db()
    r = StepOverrideRepo(conn)
    r.set(2, "", 1, True)
    r.set(2, "", 1, True)       # UNIQUE upsert — no duplicate row, no raise
    assert r.map_for_scan(2) == {("", 1)}


def test_scope_isolation_between_scans():
    conn = _db()
    r = StepOverrideRepo(conn)
    r.set(1, "", 3, True)
    r.set(2, "", 3, True)
    r.clear_scan(1)
    assert r.map_for_scan(1) == set()
    assert r.map_for_scan(2) == {("", 3)}
