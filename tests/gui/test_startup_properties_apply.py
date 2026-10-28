# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""New project → Properties → Apply keeps the edits (#188).

The panel stored the dialog's YAML, then reselected the card of the same
name — and `_select` reloaded that card's YAML from its file, which Apply
does not write. The project was created with the pipeline as it was on disk.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog              # noqa: E402

from aglaia.gui import ModePicker                                 # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _fake_dialog(edited: str):
    class _Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def yaml_text(self):
            return edited
    return _Dialog


def test_apply_keeps_the_edited_pipeline(app, monkeypatch):
    panel = ModePicker.ModePickerPanel()
    assert panel._current_key and panel._yaml, "a shipped pipeline is selected"
    key, on_disk = panel._current_key, panel._yaml
    edited = on_disk + "\n# edited in the properties dialog\n"
    seen = []
    panel.pipelineChanged.connect(lambda y, p: seen.append(y))
    monkeypatch.setattr(ModePicker, "PipelineEditorDialog", _fake_dialog(edited))

    panel._open_properties()

    assert panel._yaml == edited
    assert seen and seen[-1] == edited
    assert panel._current_key == key           # the card stays selected
