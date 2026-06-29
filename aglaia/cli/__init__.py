# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""The `aglaia` subcommand CLI (Typer).

`aglaia [global opts] <command> [opts] [args]`. `gui` is the default command:
`aglaia` and `aglaia ~/book.agl` both open the GUI. See docs/subcommand-cli.md.

:func:`run` is the process entry point (called by ``aglaia/__main__.py`` after
the multiprocessing-spawn wiring).
"""

from __future__ import annotations

import sys
from typing import Annotated, Optional

import click
import typer

from aglaia.cli.commands import gui as _gui
from aglaia.cli.commands import list_cmd as _list
from aglaia.cli.commands import run as _run
from aglaia.cli.commands import setup as _setup
from aglaia.cli.commands import version as _version

#: First-token names that are real commands (not the default `gui`).
KNOWN_COMMANDS = {"gui", "run", "setup", "list", "version"}


def _version_callback(value: bool) -> None:
    if value:
        from aglaia.version import get_version
        typer.echo(f"Aglaïa {get_version()}")
        raise typer.Exit()


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
    help="Aglaïa — webcam book scanner: capture, dewarp, binarize, OCR to PDF/Markdown.",
)


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True,
                     help="Print the version and exit."),
    ] = False,
) -> None:
    """Global options apply to every command."""


app.command("gui")(_gui.gui)
app.command("run")(_run.run)
app.command("setup")(_setup.setup)
app.command("list")(_list.list_)
app.command("version")(_version.version)


def _prepare_args(argv: Optional[list[str]]) -> list[str]:
    """Inject the default command (`gui`) when the first token isn't a command.

    `aglaia` → `gui`; `aglaia book.agl` → `gui book.agl`. `-h`/`--help`/
    `--version` pass through to the root. Old flat flags (`--headless`, …) fall
    to `gui` as unknown options and error out — the intended clean break."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return ["gui"]
    first = args[0]
    if first in KNOWN_COMMANDS or first in ("-h", "--help", "--version"):
        return args
    return ["gui", *args]


def run(argv: Optional[list[str]] = None) -> int:
    """Dispatch the CLI and return a process exit code."""
    args = _prepare_args(argv)
    command = typer.main.get_command(app)
    try:
        command(args=args, standalone_mode=False)
        return 0
    except click.exceptions.Exit as exc:          # typer.Exit / --help / --version
        return int(exc.exit_code)
    except click.exceptions.Abort:                # Ctrl-C
        return 130
    except click.ClickException as exc:           # usage errors, etc.
        exc.show()
        return exc.exit_code
    except SystemExit as exc:                     # spec parsers raise SystemExit
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
