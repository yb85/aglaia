# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""An export made only to hand to a plugin is a courier, not a deliverable.

"Send to Calibre" used to run the ordinary export first: a save dialog, a
folder to pick, a filename to confirm — for a file the user will never open,
because the point was to put it in calibre. Worse, the proposed name is derived
from the project, so sending the same book twice offered to overwrite the
first copy.

So a send writes to a fresh private directory, keeps the filename the dialog
would have proposed (Kindle attaches it under that name, calibre reads a title
out of it), and deletes it afterwards. A normal export is untouched: still
asked for, still kept, still revealed in the Finder.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from aglaia.gui.MainWindow import MainWindow                        # noqa: E402


class _Host(SimpleNamespace):
    """Just enough of a MainWindow for the two methods under test.

    Constructing a real one costs a Qt window, a database and a chain; these
    two methods touch three attributes between them, and a stub keeps the test
    honest about that."""
    _export_destination = MainWindow._export_destination
    _discard_if_staged = MainWindow._discard_if_staged


def _host(tmp_path, *, sending: str = ""):
    return _Host(_pending_send=sending, _send_staging=set(),
                 args=SimpleNamespace(workspace_dir=tmp_path))


def test_a_send_never_opens_a_dialog(tmp_path, monkeypatch):
    """If it did, a headless send would hang and a normal one would ask for a
    filename the user has no reason to care about."""
    from PySide6.QtWidgets import QFileDialog

    def boom(*a, **k):
        raise AssertionError("asked the user where to put a courier file")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", boom)

    h = _host(tmp_path, sending="send-to-calibre")
    out = h._export_destination("Mon Livre.pdf", "t", "f")
    assert out is not None and out.name == "Mon Livre.pdf"


def test_the_filename_is_kept_exactly(tmp_path):
    """It is not incidental: Kindle attaches the file under this name and
    calibre takes a book title from it."""
    h = _host(tmp_path, sending="d")
    out = h._export_destination("Grégoire — Homélies_appleOCR.pdf", "t", "f")
    assert out.name == "Grégoire — Homélies_appleOCR.pdf"


def test_two_sends_of_the_same_book_cannot_collide(tmp_path):
    h = _host(tmp_path, sending="d")
    a = h._export_destination("Book.pdf", "t", "f")
    b = h._export_destination("Book.pdf", "t", "f")
    assert a != b and a.parent != b.parent
    assert a.name == b.name == "Book.pdf"


def test_a_normal_export_still_asks(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    seen = {}

    def fake(_self, _title, start, _filt):
        seen["start"] = start
        return (str(tmp_path / "Chosen.pdf"), "")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake))

    h = _host(tmp_path)                       # nothing pending
    out = h._export_destination("Book.pdf", "t", "f")
    assert out == tmp_path / "Chosen.pdf"
    # …and still proposes the workspace + derived name as the starting point.
    assert seen["start"] == str(tmp_path / "Book.pdf")
    assert h._send_staging == set()


def test_a_cancelled_dialog_is_no_export(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert _host(tmp_path)._export_destination("Book.pdf", "t", "f") is None


class TestCleaningUp:
    def test_a_courier_is_removed_with_its_directory(self, tmp_path):
        h = _host(tmp_path, sending="d")
        out = h._export_destination("Book.pdf", "t", "f")
        out.write_bytes(b"%PDF-")
        assert h._discard_if_staged(out) is True
        assert not out.exists() and not out.parent.exists()
        assert h._send_staging == set()

    def test_a_file_the_user_chose_is_never_touched(self, tmp_path):
        """The one thing this must not get wrong."""
        h = _host(tmp_path)
        keeper = tmp_path / "Mon Livre.pdf"
        keeper.write_bytes(b"%PDF-")
        assert h._discard_if_staged(keeper) is False
        assert keeper.exists()

    def test_discarding_twice_is_harmless(self, tmp_path):
        """The failure paths and the send path can both reach it."""
        h = _host(tmp_path, sending="d")
        out = h._export_destination("Book.pdf", "t", "f")
        out.write_bytes(b"x")
        assert h._discard_if_staged(out) is True
        assert h._discard_if_staged(out) is False

    def test_a_send_that_was_never_written_still_cleans_up(self, tmp_path):
        """The export can fail before producing anything — the empty staging
        directory must not survive it."""
        h = _host(tmp_path, sending="d")
        out = h._export_destination("Book.pdf", "t", "f")
        assert h._discard_if_staged(out) is True
        assert not Path(out).parent.exists()
