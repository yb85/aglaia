"""Subcommand-CLI dispatch tests (docs/subcommand-cli.md, step 1).

The GUI / headless handlers are stubbed so we test *routing + arg parsing*
without launching Qt or the processing chain.
"""

from __future__ import annotations

import pytest

import aglaia.app as app
from aglaia.cli import _prepare_args, run as cli_run


@pytest.fixture
def captured(monkeypatch):
    """Capture which handler fired and with what CliConfig."""
    calls: dict[str, object] = {}
    monkeypatch.setattr(app, "launch_gui", lambda cfg: (calls.__setitem__("gui", cfg), 0)[1])
    monkeypatch.setattr(app, "_run_headless", lambda cfg: (calls.__setitem__("run", cfg), 0)[1])
    return calls


# ── default command (no subcommand → gui) ──────────────────────────────

def test_no_args_opens_gui(captured):
    assert cli_run([]) == 0
    assert "gui" in captured and captured["gui"].source == "none"


def test_bare_project_path_opens_gui(captured):
    assert cli_run(["/tmp/book.agl"]) == 0
    assert captured["gui"].source == "project"


def test_bare_image_path_opens_gui(captured):
    assert cli_run(["/tmp/page.png"]) == 0
    assert captured["gui"].source == "images"


# ── run ─────────────────────────────────────────────────────────────────

def test_run_pdf_is_headless(captured):
    assert cli_run(["run", "/tmp/x.pdf"]) == 0
    cfg = captured["run"]
    assert cfg.headless and cfg.source == "pdfs"


def test_run_ocr_and_export_specs(captured):
    assert cli_run(["run", "/tmp/a.png", "--ocr", "surya:lang=fr-FR", "--export", "pdf:g4+md"]) == 0
    cfg = captured["run"]
    assert cfg.do_ocr and cfg.ocr_engine == "surya"
    assert cfg.ocr_languages == ["fr-FR"]
    assert {e.kind for e in cfg.exports} == {"pdf", "md"}
    assert any(e.kind == "pdf" and e.profile == "g4" for e in cfg.exports)


def test_run_requires_inputs(captured):
    assert cli_run(["run"]) == 2          # variadic-but-empty → our guard
    assert "run" not in captured


# ── gui options ──────────────────────────────────────────────────────────

def test_gui_camera_id(captured):
    assert cli_run(["gui", "--camera-id", "2"]) == 0
    assert captured["gui"].camera_id == 2


# ── clean break: old flat flags are gone ────────────────────────────────

def test_old_headless_flag_errors(captured):
    # `--headless` no longer exists; it falls to `gui` as an unknown option.
    assert cli_run(["--headless", "/tmp/x.pdf"]) == 2
    assert "gui" not in captured and "run" not in captured


# ── version / list ───────────────────────────────────────────────────────

def test_version_command(capsys):
    assert cli_run(["version"]) == 0
    assert "Aglaïa" in capsys.readouterr().out


def test_version_flag(capsys):
    assert cli_run(["--version"]) == 0
    assert "Aglaïa" in capsys.readouterr().out


def test_list_pipelines(capsys):
    assert cli_run(["list", "pipelines"]) == 0
    assert "Pipelines:" in capsys.readouterr().out


# ── pre-parse unit ─────────────────────────────────────────────────────

def test_prepare_args_injects_gui():
    assert _prepare_args([]) == ["gui"]
    assert _prepare_args(["x.pdf"]) == ["gui", "x.pdf"]
    assert _prepare_args(["run", "x"]) == ["run", "x"]
    assert _prepare_args(["--version"]) == ["--version"]
    assert _prepare_args(["--help"]) == ["--help"]


# ── usage errors must come back as exit codes, not tracebacks ────────────

def test_usage_errors_are_caught_from_typers_own_click():
    """`run()` must catch the usage errors THIS typer raises.

    Typer >= 0.26 vendors click as `typer._click` and drops the dependency on
    the standalone package, so a hard-coded `click.ClickException` catches
    nothing it throws — and `import click` may not even resolve. Both existing
    exit-code-2 cases already cover the behaviour; this pins the mechanism, so
    that "simplifying" the handler back to `import click` fails here instead of
    only on a machine that happens to have a newer typer.
    """
    import aglaia.cli as cli

    assert issubclass(cli._ClickException, Exception)
    assert cli._ClickException.__module__.startswith(("typer.", "click."))
    # Whatever typer's parser raises for a bad option must be catchable.
    from typer.main import get_command
    with pytest.raises(cli._ClickException):
        get_command(cli.app)(args=["run", "--no-such-option"],
                             standalone_mode=False)
