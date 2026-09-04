# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The OCR tab shows three engines, not six.

Six cards is a wall, and for almost everyone the answer is one of the first
two — Apple Document and Cloud on a Mac, Cloud and GLM elsewhere. The rest
fold behind a "More…" handle.

Which three: the first three that are *usable*. A card whose weights are not
downloaded offers an Install button, which is not what someone opening this
panel came for, so it waits with the others.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                       # noqa: E402

from aglaia.gui.sidebar.tabs.OcrTab import OcrTab                # noqa: E402
from aglaia.gui.sidebar.widgets.RadioCardGroup import (          # noqa: E402
    RadioCardGroup)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ── the card group's own show/hide ───────────────────────────────────

def test_a_hidden_card_keeps_its_place_and_its_state(app):
    """Hidden, not destroyed — folding must not lose the selection or force
    a rebuild."""
    g = RadioCardGroup()
    for k in ("a", "b", "c"):
        g.add_card(k, k.upper())
    g.set_current_key("c")
    g.set_card_visible("c", False)
    assert g.keys() == ["a", "b", "c"]
    assert g.current_key() == "c"
    assert g.is_card_visible("c") is False
    g.set_card_visible("c", True)
    assert g.is_card_visible("c") is True


def test_hiding_an_unknown_card_is_not_an_error(app):
    RadioCardGroup().set_card_visible("nope", False)


# ── which three ──────────────────────────────────────────────────────

def _tab(app, monkeypatch, available, order=None, current=None):
    """An OcrTab with a chosen availability map, without touching the app's
    real engines or config."""
    tab = OcrTab.__new__(OcrTab)
    group = RadioCardGroup()
    for key in (order or list(available)):
        group.add_card(key, key)
    tab.engine_group = group
    tab._engine_available = dict(available)
    tab.VISIBLE_ENGINES = 3
    if current:
        group.set_current_key(current)
    return tab


def test_only_the_first_three_usable_engines_are_shown(app, monkeypatch):
    order = ["apple_docs", "mistral_cloud", "glm", "unlimited", "surya",
             "apple_vision"]
    tab = _tab(app, monkeypatch,
               {k: True for k in order}, order, current="apple_docs")
    assert tab._visible_engine_keys() == order[:3]


def test_on_a_mac_the_first_two_are_apple_then_cloud(app, monkeypatch):
    order = ["apple_docs", "mistral_cloud", "glm", "unlimited", "surya",
             "apple_vision"]
    tab = _tab(app, monkeypatch, {k: True for k in order}, order,
               current="apple_docs")
    assert tab._visible_engine_keys()[:2] == ["apple_docs", "mistral_cloud"]


def test_off_a_mac_the_first_two_are_cloud_then_glm(app, monkeypatch):
    """`_platform_ok` drops the Apple engines before the cards are built, so
    the declared order already lands on cloud + glm."""
    order = ["mistral_cloud", "glm", "surya"]
    tab = _tab(app, monkeypatch, {k: True for k in order}, order,
               current="mistral_cloud")
    assert tab._visible_engine_keys()[:2] == ["mistral_cloud", "glm"]


def test_an_engine_with_no_weights_waits_under_more(app, monkeypatch):
    """An Install button is not what someone opening this panel came for."""
    order = ["apple_docs", "mistral_cloud", "glm", "surya", "apple_vision"]
    tab = _tab(app, monkeypatch,
               {"apple_docs": True, "mistral_cloud": True, "glm": False,
                "surya": False, "apple_vision": True},
               order, current="apple_docs")
    shown = tab._visible_engine_keys()
    assert shown == ["apple_docs", "mistral_cloud", "apple_vision"]
    assert "glm" not in shown


def test_a_disabled_card_is_never_one_of_the_three(app, monkeypatch):
    order = ["apple_docs", "mistral_cloud", "glm"]
    tab = _tab(app, monkeypatch, {k: True for k in order}, order,
               current="mistral_cloud")
    tab.engine_group.set_card_enabled("apple_docs", False)
    assert "apple_docs" not in tab._visible_engine_keys()


def test_the_selected_engine_is_always_shown(app, monkeypatch):
    """Hiding the engine that is about to run would be a panel lying about
    what it will do."""
    order = ["apple_docs", "mistral_cloud", "glm", "surya"]
    tab = _tab(app, monkeypatch, {k: True for k in order}, order,
               current="surya")
    assert "surya" in tab._visible_engine_keys()


def test_with_nothing_usable_the_panel_is_not_empty(app, monkeypatch):
    """A fresh install with no models and no key still has to reach the
    Install buttons."""
    order = ["apple_docs", "mistral_cloud", "glm", "surya"]
    tab = _tab(app, monkeypatch, {k: False for k in order}, order)
    assert tab._visible_engine_keys() == order[:3]


# ── the handle ───────────────────────────────────────────────────────

def _with_button(app, monkeypatch, **kw):
    from PySide6.QtWidgets import QPushButton
    tab = _tab(app, monkeypatch, **kw)
    tab._more_btn = QPushButton()
    tab._engines_expanded = False
    tab.tr = lambda t: t
    return tab


def test_folded_hides_the_rest_and_counts_them(app, monkeypatch):
    order = ["a", "b", "c", "d", "e"]
    tab = _with_button(app, monkeypatch,
                       available={k: True for k in order}, order=order,
                       current="a")
    tab._apply_engine_overflow()
    visible = [k for k in order if tab.engine_group.is_card_visible(k)]
    assert visible == ["a", "b", "c"]
    assert "2 more" in tab._more_btn.text()


def test_expanding_shows_everything(app, monkeypatch):
    order = ["a", "b", "c", "d", "e"]
    tab = _with_button(app, monkeypatch,
                       available={k: True for k in order}, order=order,
                       current="a")
    tab._toggle_more_engines()
    assert all(tab.engine_group.is_card_visible(k) for k in order)
    assert "Fewer" in tab._more_btn.text()


def test_toggling_back_folds_again(app, monkeypatch):
    order = ["a", "b", "c", "d"]
    tab = _with_button(app, monkeypatch,
                       available={k: True for k in order}, order=order,
                       current="a")
    tab._toggle_more_engines()
    tab._toggle_more_engines()
    assert tab.engine_group.is_card_visible("d") is False


def test_no_handle_when_nothing_is_folded(app, monkeypatch):
    order = ["a", "b"]
    tab = _with_button(app, monkeypatch,
                       available={k: True for k in order}, order=order,
                       current="a")
    tab._apply_engine_overflow()
    assert tab._more_btn.isVisible() is False
    assert all(tab.engine_group.is_card_visible(k) for k in order)


def test_a_folded_selection_unfolds_the_panel_by_itself(app, monkeypatch):
    """Someone who works in Surya has that persisted as their engine; the
    panel should open showing it, not make them click More every session."""
    order = ["a", "b", "c", "surya", "e"]
    tab = _with_button(app, monkeypatch,
                       available={k: True for k in order}, order=order,
                       current="surya")
    tab._apply_engine_overflow()
    assert tab.engine_group.is_card_visible("surya") is True


# ── against the real registry ────────────────────────────────────────

def test_the_real_tab_folds_to_three_or_fewer(app):
    tab = OcrTab()
    shown = [k for k in tab.engine_group.keys()
             if tab.engine_group.is_card_visible(k)]
    assert 0 < len(shown) <= tab.VISIBLE_ENGINES + 1   # +1: a folded current
    assert tab.engine_group.current_key() in shown
