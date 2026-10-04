# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Export / Import in the plugin settings dialog (#165).

The part worth a test is not the file dialog: it is that the form shows what
was imported. It does not re-read itself on its own, so without the reload the
next Save writes the pre-import values straight back over the import — which
looks exactly like an import that silently did nothing.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (                                   # noqa: E402
    QApplication, QLineEdit, QPushButton, QSpinBox,
)

from aglaia.app_data.plugin_ctx import PluginConfig               # noqa: E402
from aglaia.gui.PluginsTab import PluginSettingsDialog            # noqa: E402
from aglaia.plugin_api import Field                               # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Secrets:
    def __init__(self):
        self._v: dict[str, str] = {}

    available = True

    def keys(self):
        return sorted(self._v)

    def get(self, key):
        return self._v.get(key)

    def set(self, key, value):
        self._v[key] = value


class _Ctx:
    def __init__(self, slug, config, secrets):
        self.slug, self.config, self.secrets = slug, config, secrets
        self.version = "1.0.0"


class _Dest:
    """The surface `PluginSettingsDialog` reads: fields, values, a context."""

    name = "send-to-corpus"
    display = "Export to Corpus library"
    description = ""
    version = "1.3.1"
    CONFIG_FIELDS = (
        Field("base_url", "Corpus URL", "str", "", required=True),
        Field("timeout_s", "Timeout (seconds)", "int", 300),
    )
    SECRET_FIELDS = (Field("api_key", "API key", "secret", ""),)

    def __init__(self, ctx):
        self.ctx = ctx

    def conf(self, key, default=None):
        return self.ctx.config.get(key, default)

    def secret(self, key):
        return self.ctx.secrets.get(key)

    def header_names(self, key):
        return []


def _dest(tmp_path, **settings):
    cfg = PluginConfig("send-to-corpus", tmp_path / f"{len(settings)}.db")
    for k, v in settings.items():
        cfg.set(k, v)
    return _Dest(_Ctx("send-to-corpus", cfg, _Secrets()))


def _value(dlg, key):
    _f, w, _s = dlg._widgets[key]
    if isinstance(w, QSpinBox):
        return w.value()
    if isinstance(w, QLineEdit):
        return w.text()
    return None


def test_the_footer_offers_both_errands(qapp, tmp_path):
    dlg = PluginSettingsDialog(_dest(tmp_path, base_url="https://a.example"))
    labels = [b.text() for b in dlg.findChildren(QPushButton)]
    assert "Export…" in labels and "Import…" in labels


def test_the_form_shows_what_an_import_wrote(qapp, tmp_path):
    """The regression this guards: the dialog opened on the old values, the
    import changed the store underneath it, and Save put the old ones back."""
    from aglaia.app_data import plugin_transfer as xfer

    source = _dest(tmp_path, base_url="https://new.example", timeout_s=900)
    bundle = tmp_path / "b.json"
    xfer.write(source.ctx, bundle)

    target = _dest(tmp_path, base_url="https://old.example", timeout_s=300)
    dlg = PluginSettingsDialog(target)
    assert _value(dlg, "base_url") == "https://old.example"

    xfer.apply(target.ctx, xfer.read(bundle))
    dlg._reload_values()

    assert _value(dlg, "base_url") == "https://new.example"
    assert _value(dlg, "timeout_s") == 900
    # And a Save now writes the imported values, not the ones on screen before.
    dlg._collect()
    assert target.conf("base_url") == "https://new.example"


def test_a_secret_is_never_put_back_on_screen(qapp, tmp_path):
    """Reload must not round-trip the password into the box — a settings
    dialog that hands one back leaks it to the next screenshot."""
    d = _dest(tmp_path, base_url="https://a.example")
    d.ctx.secrets.set("api_key", "k-123")
    dlg = PluginSettingsDialog(d)
    dlg._reload_values()
    _f, w, _s = dlg._widgets["api_key"]
    assert w.text() == ""
    assert "stored" in w.placeholderText()
