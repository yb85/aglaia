# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""`aglaia run … --export pdf --send-to send-to-kindle`: the CLI reaches the
export plugins.

It did not. `--export` wrote files into the project directory and stopped;
`aglaia list destinations` could list plugins and nothing could use one. A
headless batch is exactly where "process, then put it in my library" matters.
"""
import importlib
from pathlib import Path

import pytest

from aglaia.cli.shared import run_config


def test_send_to_is_parsed_into_the_config(tmp_path):
    cfg = run_config([tmp_path / "x.agl"], None, None, False, None, "auto",
                     "pdf", None, None, None, None, False,
                     send_to="send-to-kindle+send-to-corpus")
    assert cfg.send_to == ["send-to-kindle", "send-to-corpus"]


def test_no_send_to_means_none(tmp_path):
    cfg = run_config([tmp_path / "x.agl"], None, None, False, None, "auto",
                     "pdf", None, None, None, None, False)
    assert cfg.send_to == []


@pytest.fixture()
def dests(tmp_path, monkeypatch):
    """A clean APP_DATA with one stub destination that accepts pdf only."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "appdata"))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as reg
    for m in (ad, pc, reg):
        importlib.reload(m)
    from aglaia.workers import destinations as d
    d.reset_for_tests()
    p = reg.installed_root("destinations") / "pdf-only"
    p.mkdir(parents=True)
    (p / "aglaia-plugin.toml").write_text(
        '[plugin]\nslug = "pdf-only"\nname = "PDF only"\nversion = "1.0.0"\n'
        'entry = "pdf_only.py"\nlicense = "MIT"\n[requires]\napi = 1\n'
        '[capabilities]\nconfig = true\n', encoding="utf-8")
    (p / "pdf_only.py").write_text(
        "from aglaia.plugin_api import Destination, SendResult, register_destination\n"
        "SENT = []\n"
        "@register_destination\n"
        "class P(Destination):\n"
        "    name = 'pdf-only'\n    display = 'PDF only'\n    accepts = ('pdf',)\n"
        "    def send(self, path, meta):\n"
        "        SENT.append((path.name, meta.title))\n"
        "        return SendResult(True, f'took {path.name}')\n", encoding="utf-8")
    d.reset_for_tests()
    yield d
    d.forget("pdf-only")
    d.reset_for_tests()


def test_each_plugin_gets_only_the_formats_it_accepts(dests, tmp_path):
    from aglaia.workers.headless import _send_exports
    import sys
    pdf, md = tmp_path / "book.pdf", tmp_path / "book.md"
    pdf.write_bytes(b"%PDF"); md.write_text("# x")
    rc = _send_exports(["pdf-only"], [pdf, md], project_file=tmp_path / "book.agl", slug="book")
    assert rc == 0
    assert sys.modules["aglaia_plugin_pdf_only"].SENT == [("book.pdf", "book")]


def test_a_plugin_that_is_not_installed_fails_and_names_the_fix(dests, tmp_path, capsys):
    from aglaia.workers.headless import _send_exports
    pdf = tmp_path / "b.pdf"; pdf.write_bytes(b"%PDF")
    rc = _send_exports(["send-to-nowhere"], [pdf], project_file=tmp_path / "b.agl", slug="b")
    assert rc == 1
    err = capsys.readouterr().err
    assert "not installed" in err and "aglaia plugins install send-to-nowhere" in err
    assert "pdf-only" in err                     # says what IS installed


def test_a_plugin_with_nothing_to_take_is_a_failure_not_a_silent_success(dests, tmp_path, capsys):
    """Exporting only Markdown to a PDF-only destination "succeeding" is the
    expensive kind of success."""
    from aglaia.workers.headless import _send_exports
    md = tmp_path / "b.md"; md.write_text("# x")
    rc = _send_exports(["pdf-only"], [md], project_file=tmp_path / "b.agl", slug="b")
    assert rc == 1
    assert "none of the exported files" in capsys.readouterr().err


def test_nothing_to_send_is_not_an_error(tmp_path):
    from aglaia.workers.headless import _send_exports
    assert _send_exports([], [tmp_path / "b.pdf"], project_file=tmp_path / "b.agl", slug="b") == 0
