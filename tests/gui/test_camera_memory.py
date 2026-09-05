# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A camera's rotation / mirror / flip survives between projects.

The rig does not move between books, so the correction that makes its feed
upright is a property of the camera, not of the project. Keyed by the
camera's name, not its index — indexes shift when devices come and go.
"""
import importlib

import pytest

# `aglaia.gui.__init__` imports the MainWindow, so even this Qt-free module
# needs Qt on the path; CI without the `gui` extra skips, like every GUI test.
pytest.importorskip("PySide6")


@pytest.fixture()
def cm(tmp_path, monkeypatch):
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.db as db
    importlib.reload(ad); importlib.reload(db)
    from aglaia.gui import camera_memory
    importlib.reload(camera_memory)
    return camera_memory


class TestEncoding:
    """The string is what `WebcamThread.set_transform` parses, so 270° is
    spelled the way that parser reads it."""

    def test_identity(self, cm):
        assert cm.encode(0, False, False) == "0"

    def test_rotation_and_modifiers(self, cm):
        assert cm.encode(90, True, False) == "90+mirror"
        assert cm.encode(180, False, True) == "180+flip"
        assert cm.encode(270, True, True) == "-90+mirror+flip"

    def test_it_round_trips_through_the_parser(self, cm):
        from aglaia.gui.WebcamThread import WebcamThread
        for rot, mir, fl in ((0, 0, 0), (90, 1, 0), (180, 0, 1), (270, 1, 1)):
            w = WebcamThread.__new__(WebcamThread)   # no camera opened
            w.set_transform(cm.encode(rot, bool(mir), bool(fl)))
            assert (w.rotation, w.mirror, w.flip) == (rot, bool(mir), bool(fl))


class TestStore:
    def test_save_then_load(self, cm):
        cm.save("Logitech BRIO", 90, True, False)
        assert cm.load("Logitech BRIO") == "90+mirror"

    def test_unknown_camera_is_none(self, cm):
        assert cm.load("never seen") is None

    def test_cameras_are_independent(self, cm):
        cm.save("A", 90, False, False)
        cm.save("B", 180, False, True)
        assert cm.load("A") == "90" and cm.load("B") == "180+flip"

    def test_resetting_to_identity_forgets_the_entry(self, cm):
        """The default is not stored, so a rig put back straight leaves
        nothing behind to surprise anyone later."""
        cm.save("A", 90, False, False)
        cm.save("A", 0, False, False)
        assert cm.load("A") is None


def test_the_key_falls_back_to_the_index_without_avfoundation(cm, monkeypatch):
    import aglaia.gui.WebcamThread as wt
    monkeypatch.setattr(wt, "_camera_label", lambda i: "?")
    assert cm.camera_key(2) == "index:2"
