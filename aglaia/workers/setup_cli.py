# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""``aglaia --setup`` — interactive first-run setup for CLI-only installs.

The terminal counterpart of the GUI OnboardingWizard, so a ``--without-gui``
install is viable on its own: permissions note → model picker →
download → bootstrap the config DB + seed pipelines → print where everything
lives. Uses rich (output / progress) + questionary (checkbox picker). No Qt.
"""

from __future__ import annotations

import sys

# A headless/CLI install has no GUI Downloader, so --setup is the only place to
# fetch models — offer everything the registry knows, EXCEPT voice control
# (Vosk), which only drives live capture and is useless headless. dbnet is the
# default detector (required off-macOS) and the only pre-ticked model.
_EXCLUDE_KEYS = {"vosk_en"}        # capture-only, no headless use
_MAC_ONLY_KEYS = {"paddle_vl"}     # MLX 4-bit — Apple Silicon only
_DEFAULT_KEYS = {"dbnet"}          # pre-ticked recommendation
_ORDER = ["dbnet", "east", "surya", "paddle_vl"]   # display order; rest appended


def _offered_specs(is_mac: bool) -> list:
    """Registry specs offered by --setup, in display order, minus excludes
    (Vosk always; MLX-only models off macOS — they can't run there)."""
    from aglaia.app_data.models import _load_model_specs
    skip = _EXCLUDE_KEYS if is_mac else _EXCLUDE_KEYS | _MAC_ONLY_KEYS
    specs = {s.key: s for s in _load_model_specs() if s.key not in skip}
    ordered = [specs[k] for k in _ORDER if k in specs]
    ordered += [s for k, s in specs.items() if k not in _ORDER]
    return ordered


def has_user_config() -> bool:
    """True once setup (or the GUI) has bootstrapped the config DB."""
    try:
        from aglaia.app_data import app_data_dir, db as cfg
        if not (app_data_dir() / "aglaia-config.db").exists():
            return False
        with cfg.session() as conn:
            return bool(cfg.get(conn, cfg.KEY_WELCOME_SEEN, False))
    except Exception:
        return False


def run_setup() -> int:
    """Run the interactive setup. Returns a process exit code."""
    try:
        import questionary
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import (
            BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn,
        )
    except Exception as e:  # pragma: no cover - missing TUI deps
        print(f"aglaia --setup needs the TUI deps (rich, questionary): {e}",
              file=sys.stderr)
        return 2

    from aglaia.app_data import (
        app_data_dir, db as cfg, log_dir, models_dir, seed_pipelines,
    )
    from aglaia.app_data.models import download_model, is_model_installed, spec_for

    console = Console()
    is_mac = sys.platform == "darwin"

    console.print(Panel.fit(
        "[bold]Set up Aglaïa[/bold]\nTake a minute to configure your install.",
        border_style="cyan"))

    # (No language step — UI language only affects the Qt GUI; the CLI is
    # English-only, and the GUI prompts for it on its own first run.)

    # 1 ─ Permissions note
    console.print(Panel(
        "Aglaïa runs offline by default — your pages stay on this machine.\n"
        "  • Camera / microphone — only for live capture or voice control.\n"
        "  • System keychain — only if you save a Cloud OCR API key.\n"
        "  • Files — projects, settings and models live in your app-data folder.",
        title="Permissions", border_style="grey50"))

    # 2 ─ Models
    choices = []
    for spec in _offered_specs(is_mac):
        installed = is_model_installed(spec.key)
        checked = spec.key in _DEFAULT_KEYS
        meta = f"~{spec.approx_size_mb} MB · {spec.purpose}"
        suffix = "  [already installed]" if installed else f"  ({meta})"
        choices.append(questionary.Choice(
            spec.title + suffix, value=spec.key,
            checked=checked and not installed,
            disabled="installed" if installed else None))
    if not is_mac:
        console.print("[grey50]DBnet is required off macOS (no Apple Vision "
                      "fallback); it will be fetched even if unticked.[/grey50]")
    picked = questionary.checkbox(
        "Models to download (space to toggle, enter to confirm)",
        choices=choices).ask()
    if picked is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return 1
    picked = set(picked)
    if not is_mac and not is_model_installed("dbnet"):
        picked.add("dbnet")   # required off-macOS

    to_fetch = [s for s in (spec_for(k) for k in picked)
                if s is not None and not is_model_installed(s.key)]

    # 3 ─ Download
    failures = []
    for spec in to_fetch:
        with Progress(TextColumn("[cyan]{task.description}"), BarColumn(),
                      DownloadColumn(), TransferSpeedColumn(),
                      console=console) as prog:
            task = prog.add_task(spec.title, total=None)

            def cb(done: int, total: int, _t=task, _p=prog) -> None:
                _p.update(_t, completed=done, total=total or None)
            try:
                download_model(spec, cb)
            except Exception as e:
                failures.append((spec.title, str(e)))
                console.print(f"[red]✗ {spec.title}: {e}[/red]")
    if failures:
        console.print("[yellow]Some downloads failed — re-run "
                      "`aglaia --setup` to retry.[/yellow]")

    # 4 ─ Persist + bootstrap config, seed pipelines
    try:
        with cfg.session() as conn:
            cfg.bootstrap(conn)
            cfg.set(conn, cfg.KEY_WELCOME_SEEN, True)
            cfg.set(conn, cfg.KEY_MODELS_PROMPT_DISMISSED, True)
            conn.commit()
    except Exception as e:
        console.print(f"[red]config bootstrap failed: {e}[/red]")
        return 1
    try:
        seed_pipelines()
    except Exception:
        pass

    # 5 ─ Where things live
    console.print(Panel(
        f"Config DB : {app_data_dir() / 'aglaia-config.db'}\n"
        f"Pipelines : {app_data_dir() / 'pipelines'}  (edit the *.yaml by hand)\n"
        f"Models    : {models_dir()}\n"
        f"Logs      : {log_dir()}",
        title="[green]Setup complete[/green] — where things live",
        border_style="green"))
    console.print("Process a batch:  "
                  "[bold]aglaia <project.agl | images… | file.pdf> --headless[/bold]")
    return 0
