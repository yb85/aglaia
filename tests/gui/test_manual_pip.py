# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The hand-edited mark in the scan views (M9 #102).

A page carrying manual overrides looked exactly like one the pipeline decided
alone, everywhere outside the debug editor. One quiet dot, the same in all
three views, with a tooltip that names what was touched.

Quiet is the requirement: a hand-edited page is not a warning. The mark must
not appear at all on a page with no override, or every page would carry it and
it would say nothing.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from aglaia.gui.widgets import ManualPip, manual_tooltip     # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_tooltip_names_the_fields_in_pipeline_order(qapp):
    """Order follows the pipeline, not the payload's key order, so the same
    page reads the same way wherever the user meets it."""
    assert manual_tooltip(["curl", "skew_deg", "force"]) == (
        "Hand-tuned: deskew angle, dewarp curl, forced dewarp")
    assert manual_tooltip(["quad", "roi"]) == (
        "Hand-tuned: page ROI, keystone quad")


def test_no_override_means_no_tooltip(qapp):
    assert manual_tooltip([]) == ""
    assert manual_tooltip(None) == ""


def test_the_pip_is_a_glyph_and_carries_its_reason(qapp):
    """It was a 7 px dot, which was too small to notice AND sat exactly where
    the next-page chevron is — the two overlapped and the dot lost. Now a
    `hand` glyph, big enough to read at thumbnail size but still a mark
    rather than a badge: no background, no border, no text."""
    pip = ManualPip(["curl"])
    assert 14 <= pip.width() == pip.SIZE <= 24
    assert not pip.pixmap().isNull(), "the hand glyph did not load"
    assert pip.toolTip() == "Hand-tuned: dewarp curl"
    assert pip.text() == ""


def test_a_table_row_marks_only_an_edited_page(qapp):
    from aglaia.gui.ScansTableView import _RowWidget
    item = {"history": [], "nodes": {}, "trashed": False}

    def row(fields):
        return _RowWidget(stem="s_A", item=item, raw_filestem="s",
                          ocr_state="none", thumb_loader=lambda *a, **k: None,
                          thumb_h=60, global_history=[], manual_fields=fields)

    assert row(None).findChildren(ManualPip) == []
    assert len(row(["curl", "roi"]).findChildren(ManualPip)) == 1
