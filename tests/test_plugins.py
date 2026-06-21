# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Drop-in plugin trust gate + discovery (no Qt)."""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest


@pytest.fixture()
def app_data(tmp_path, monkeypatch):
    """Point APP_DATA at a temp dir so the config DB + plugin dirs are
    isolated per test."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    # Re-import fresh so module-level caches (if any) bind to the temp dir.
    import lib.app_data as _ad
    importlib.reload(_ad)
    from lib.app_data import plugins as _pl
    importlib.reload(_pl)
    yield _pl
    # Drop any plugin modules + sys.path entries the test imported.
    for kind in ("processors", "ocr"):
        d = str(_ad.plugins_dir(kind))
        if d in sys.path:
            sys.path.remove(d)


def _drop(plugins, kind: str, name: str, body: str):
    p = plugins.plugins_dir(kind) / f"{name}.py"
    p.write_text(textwrap.dedent(body))
    return p.resolve()


def test_plugins_dir_typed(app_data):
    pl = app_data
    root = pl.plugins_dir()
    procs = pl.plugins_dir("processors")
    assert procs.parent == root
    assert procs.is_dir()


def test_scan_pending_new_then_acknowledge(app_data):
    pl = app_data
    _drop(pl, "ocr", "foo_plugin", "x = 1\n")
    pending = pl.scan_pending()
    assert len(pending) == 1
    cand = pending[0]
    assert cand.kind == "ocr"
    assert cand.reason == "new"
    # Acknowledged → no longer pending, and available for load.
    pl.acknowledge(cand)
    assert pl.scan_pending() == []
    assert cand.path in pl.accepted_for_load("ocr")


def test_changed_file_reverts_to_pending(app_data):
    pl = app_data
    path = _drop(pl, "processors", "bar_plugin", "x = 1\n")
    pl.acknowledge(pl.scan_pending()[0])
    assert pl.scan_pending() == []
    # Mutate content → sha mismatch → pending again, excluded from load.
    path.write_text("x = 2\n")
    pending = pl.scan_pending()
    assert len(pending) == 1
    assert pending[0].reason == "changed"
    assert path not in pl.accepted_for_load("processors")


def test_private_files_skipped(app_data):
    pl = app_data
    _drop(pl, "ocr", "_helper", "x = 1\n")
    assert pl.scan_pending() == []


def test_reject_deletes_file(app_data):
    pl = app_data
    path = _drop(pl, "ocr", "evil", "x = 1\n")
    pl.reject(pl.scan_pending()[0], delete_file=True)
    assert not path.exists()
    assert pl.scan_pending() == []


def test_import_accepted_only_acknowledged(app_data):
    pl = app_data
    # Two engine plugins; only one is acknowledged.
    _drop(pl, "ocr", "eng_yes", """
        MARKER_YES = True
    """)
    _drop(pl, "ocr", "eng_no", """
        MARKER_NO = True
    """)
    yes = next(c for c in pl.scan_pending() if c.path.stem == "eng_yes")
    pl.acknowledge(yes)
    imported = pl.import_accepted("ocr")
    assert "eng_yes" in imported
    assert "eng_no" not in imported
    assert "eng_yes" in sys.modules
    assert "eng_no" not in sys.modules


def test_ocr_plugin_registers_engine(app_data):
    pl = app_data
    _drop(pl, "ocr", "myengine_plugin", """
        from lib.workers.ocr.engine import OcrEngine, register

        @register
        class MyTestEngine(OcrEngine):
            name = "mytest_engine"
            display = "My Test Engine"
            available = True
    """)
    pl.acknowledge(pl.scan_pending()[0])
    pl.import_accepted("ocr")
    from lib.workers.ocr.engine import ENGINE_REGISTRY
    assert "mytest_engine" in ENGINE_REGISTRY


def test_processor_registry_picks_up_plugin(app_data, monkeypatch):
    pl = app_data
    _drop(pl, "processors", "MyProcPlugin", """
        from lib.processors.abstraction import AbstractImageProcessor

        class MyPluginProc(AbstractImageProcessor):
            SUMMARY = "test plugin processor"
            OPTIONS = {}
    """)
    pl.acknowledge(pl.scan_pending()[0])

    # Reset registry discovery so it re-runs with our temp plugin dir.
    from lib.processors import registry as reg
    monkeypatch.setattr(reg, "_DISCOVERED", False)
    monkeypatch.setattr(reg, "_REGISTRY", {})
    procs = reg.all_processors()
    assert "MyPluginProc" in procs
    assert procs["MyPluginProc"].summary == "test plugin processor"
