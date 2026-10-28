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

# Widget tests must not paint on the developer's desktop: a local run opened
# real, half-built Capture windows over whatever was on screen. Set before any
# Qt import, and FORCED: a desktop session exports its own value (Omarchy:
# "wayland;xcb"), which a setdefault kept. AGLAIA_TESTS_ON_SCREEN=1 opts out.
if os.environ.get("AGLAIA_TESTS_ON_SCREEN") != "1":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
# A child process (a CLI run a test spawns) does not see the in-memory
# keychain below: give it one that refuses every call, so it falls back to
# its own .env and never reaches the developer's real keychain.
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

import pytest

from aglaia.storage.db import open_db
from aglaia.storage.repo import ProjectRepo, PipelineRepo


@pytest.fixture(autouse=True)
def _private_keychain():
    """Every test gets an empty, in-memory keychain.

    The suite isolated APP_DATA but not the OS keychain: a test of the
    plaintext store called `set_mistral_api_key`, which clears the keychain
    copy — and a local `pytest` deleted the developer's real Mistral key.
    A test that patches `keyring` itself still wins."""
    try:
        import keyring
        from keyring.backend import KeyringBackend
    except ImportError:
        yield
        return

    class _Memory(KeyringBackend):
        priority = 1

        def __init__(self):
            super().__init__()
            self._store: dict[tuple[str, str], str] = {}

        def get_password(self, service, username):
            return self._store.get((service, username))

        def set_password(self, service, username, password):
            self._store[(service, username)] = password

        def delete_password(self, service, username):
            if self._store.pop((service, username), None) is None:
                from keyring.errors import PasswordDeleteError
                raise PasswordDeleteError(username)

    real = keyring.get_keyring()
    keyring.set_keyring(_Memory())
    try:
        yield
    finally:
        keyring.set_keyring(real)


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
