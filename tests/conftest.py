# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import os
import sys
from pathlib import Path

# Ensure repo root on sys.path so `import aglaia...` works without install
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from aglaia.storage.db import open_db
from aglaia.storage.repo import ProjectRepo, PipelineRepo


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "proj.sqlite"
    conn = open_db(path)
    yield conn
    conn.close()


@pytest.fixture()
def seeded_db(db):
    ProjectRepo(db).init("Test", "test")
    pid = PipelineRepo(db).upsert("name: stub\npipeline: []\n", "stub", step_count=0)
    return db, pid
