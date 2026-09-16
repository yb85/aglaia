# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A plugin can ask for headers nobody anticipated (`Field(kind="headers")`).

Cloudflare Access wants `CF-Access-Client-Id` and `CF-Access-Client-Secret`
in front of a corpus; the next proxy will want something else. A field per
proxy would be a plugin release per proxy — so the user adds them by name,
the names are visible and removable, and only the values are secret.
"""
import json
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (QApplication, QLabel,            # noqa: E402
                               QLineEdit, QPushButton, QToolButton)

from aglaia.gui.PluginsTab import PluginSettingsDialog          # noqa: E402
from aglaia.plugin_api import Destination, Field                # noqa: E402


class _Store(dict):
    available = True

    def get(self, key, default=None):
        return dict.get(self, key, default)

    def set(self, key, value):
        self[key] = value

    def delete(self, key):
        self.pop(key, None)


class _Ctx:
    def __init__(self):
        self.config = _Store()
        self.secrets = _Store()


class _Dest(Destination):
    name = "probe"
    display = "Probe"
    CONFIG_FIELDS = (Field("base_url", "URL", "str", ""),)
    SECRET_FIELDS = (Field("extra_headers", "Additional headers", "headers",
                           help="Sent with every request."),)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def dest():
    d = _Dest()
    d.ctx = _Ctx()
    return d


def _headers_widget(dlg):
    return dlg._widgets["extra_headers"][1]


def _inputs(w):
    edits = w.findChildren(QLineEdit)
    add = [b for b in w.findChildren(QPushButton) if b.text() == "Add"][0]
    return edits[0], edits[1], add


def _tag_names(w):
    return [lb.text() for lb in w.findChildren(QLabel)]


def test_a_header_is_added_by_name_and_stored_as_a_secret(app, dest):
    dlg = PluginSettingsDialog(dest)
    w = _headers_widget(dlg)
    name, value, add = _inputs(w)
    name.setText("CF-Access-Client-Id")
    value.setText("id-123")
    add.click()
    # The boxes clear so the second header can be typed straight away —
    # Cloudflare Access always needs two.
    assert name.text() == "" and value.text() == ""
    name.setText("CF-Access-Client-Secret")
    value.setText("secret-456")
    add.click()
    dlg._collect()

    assert json.loads(dest.ctx.config["extra_headers"]) == [
        "CF-Access-Client-Id", "CF-Access-Client-Secret"]
    assert dest.ctx.secrets["extra_headers.CF-Access-Client-Id"] == "id-123"
    assert dest.headers() == {"CF-Access-Client-Id": "id-123",
                              "CF-Access-Client-Secret": "secret-456"}
    assert dest.header_names() == ["CF-Access-Client-Id",
                                   "CF-Access-Client-Secret"]


def test_the_value_never_comes_back_to_the_screen(app, dest):
    dest.ctx.config.set("extra_headers", json.dumps(["CF-Access-Client-Id"]))
    dest.ctx.secrets.set("extra_headers.CF-Access-Client-Id", "id-123")
    dlg = PluginSettingsDialog(dest)
    w = _headers_widget(dlg)
    name, value, _add = _inputs(w)
    assert value.text() == "" and name.text() == ""
    # The NAME is shown — a header the user cannot see is one they cannot fix.
    assert "CF-Access-Client-Id" in _tag_names(w)
    assert "id-123" not in " ".join(_tag_names(w))
    assert value.echoMode() == QLineEdit.EchoMode.Password


def test_removing_a_tag_deletes_its_secret(app, dest):
    dest.ctx.config.set("extra_headers", json.dumps(["CF-Access-Client-Id"]))
    dest.ctx.secrets.set("extra_headers.CF-Access-Client-Id", "id-123")
    dlg = PluginSettingsDialog(dest)
    w = _headers_widget(dlg)
    crosses = [b for b in w.findChildren(QToolButton) if b.text() == "✕"]
    assert crosses, "the tag must carry a visible remove button"
    cross = crosses[0]
    cross.click()
    dlg._collect()
    assert json.loads(dest.ctx.config["extra_headers"]) == []
    assert "extra_headers.CF-Access-Client-Id" not in dest.ctx.secrets
    assert dest.headers() == {}


def test_a_half_filled_header_is_refused_with_a_reason(app, dest):
    dlg = PluginSettingsDialog(dest)
    w = _headers_widget(dlg)
    name, _value, add = _inputs(w)
    name.setText("CF-Access-Client-Id")
    add.click()
    dlg._collect()
    assert json.loads(dest.ctx.config["extra_headers"]) == []
    assert "name and a value" in dlg._status.text()


def test_a_required_headers_field_counts_as_missing_until_one_is_added(dest):
    class Needy(_Dest):
        SECRET_FIELDS = (Field("extra_headers", "Additional headers",
                               "headers", required=True),)
    d = Needy()
    d.ctx = _Ctx()
    assert d.missing_settings() == ["Additional headers"]
    d.ctx.config.set("extra_headers", json.dumps(["X-Token"]))
    d.ctx.secrets.set("extra_headers.X-Token", "v")
    assert d.missing_settings() == []
