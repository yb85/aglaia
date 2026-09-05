# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""`aglaia plugins …` — the Plugins tab, from a terminal.

    aglaia plugins list                       what is installed, and what needs setting up
    aglaia plugins search [TERM]              what the registry offers
    aglaia plugins install SLUG | FILE.zip    from the registry, or a local archive (--trust)
    aglaia plugins update [SLUG | --all]
    aglaia plugins toggle SLUG                enable / disable
    aglaia plugins remove SLUG [--yes]        and its settings, files and secrets
    aglaia plugins config SLUG [--set k=v …]  a small TUI, or scripted with --set

Everything here calls the same functions the GUI does (`plugin_registry`,
`destinations`), so a plugin set up on a headless box and one set up in the
window are the same plugin. The one deliberate difference is consent for an
unreviewed archive: the window makes you type a sentence; the terminal makes
you pass `--trust`, which is the same act in the medium's own idiom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

plugins_app = typer.Typer(
    help="Install, configure and manage plugins from the terminal.",
    no_args_is_help=True,
)


def _console():
    """A console that does not wrap when the output is not a terminal.

    Rich assumes 80 columns for a pipe. Output going to a file, a script or an
    agent is better long than folded — a table cell split across lines is not
    something a caller can parse."""
    import sys
    from rich.console import Console
    if sys.stdout.isatty():
        return Console()
    return Console(width=200)


def _err(msg: str) -> None:
    typer.echo(msg, err=True)


# ── list ──────────────────────────────────────────────────────────────

@plugins_app.command("list")
def list_() -> None:
    """Installed plugins: version, state, and what each still needs."""
    from rich.table import Table
    from aglaia.app_data import plugin_registry as reg
    from aglaia.workers import destinations as dest

    installed = reg.list_installed()
    if not installed:
        typer.echo("No plugins installed. `aglaia plugins search` shows what "
                   "the registry offers.")
        return
    index = reg.fetch_index()
    loaded = dest.load_all()
    t = Table(show_header=True, header_style="bold")
    for col in ("plugin", "kind", "version", "state", "source", "needs"):
        t.add_column(col, overflow="fold")
    for it in installed:
        slug, man, rec = it["slug"], it["manifest"], it["record"] or {}
        version = man.version if man else "?"
        state = []
        if it["error"]:
            state.append("[red]broken[/red]")
        if reg.is_disabled(slug):
            state.append("[dim]disabled[/dim]")
        entry = index.get(slug) if index else None
        if entry is not None and reg.update_available(slug, entry.version):
            state.append(f"[cyan]update → {entry.version}[/cyan]")
        if not state:
            state.append("[green]enabled[/green]")
        needs = ""
        d = loaded.get(slug)
        if d is not None:
            missing = d.missing_settings()
            needs = ", ".join(missing) if missing else "[green]ready[/green]"
        elif it["kind"] == "destinations" and not it["error"]:
            needs = dest.load_error(slug) or ""
        src = str(rec.get("source") or "")
        src = {"zip": "[red]archive (unreviewed)[/red]", "registry": "registry"}.get(src, src)
        t.add_row(man.name if man else slug, it["kind"], version,
                  " · ".join(state), src, needs)
    _console().print(t)


# ── search ────────────────────────────────────────────────────────────

@plugins_app.command()
def search(term: Annotated[Optional[str], typer.Argument(help="Filter by slug, name or summary.")] = None) -> None:
    """What the registry offers."""
    from rich.table import Table
    from aglaia.app_data import plugin_registry as reg

    index = reg.fetch_index()
    if index.error and not index.entries:
        _err(f"Could not reach the registry: {index.error}")
        raise typer.Exit(1)
    installed = {it["slug"]: it for it in reg.list_installed()}
    q = (term or "").lower()
    rows = [e for e in index.entries
            if not q or q in e.slug.lower() or q in e.name.lower()
            or q in (e.summary or "").lower()]
    if not rows:
        typer.echo("Nothing matches." if q else "The registry is empty.")
        return
    t = Table(show_header=True, header_style="bold")
    for col in ("slug", "kind", "version", "by", "summary", ""):
        t.add_column(col, overflow="fold")
    for e in sorted(rows, key=lambda e: e.slug):
        have = installed.get(e.slug)
        mark = ""
        if have and have["manifest"]:
            mark = ("[cyan]update available[/cyan]"
                    if reg.update_available(e.slug, e.version) else "[green]installed[/green]")
        by = "Aglaïa" if e.first_party else (e.author.split("<")[0].strip() or "?")
        t.add_row(e.slug, e.kind, e.version, by, e.summary or "", mark)
    _console().print(t)


# ── install / update ──────────────────────────────────────────────────

@plugins_app.command()
def install(
    what: Annotated[str, typer.Argument(help="A registry slug, or a local .aglplugin / .zip.")],
    trust: Annotated[bool, typer.Option("--trust", help="Required for a local archive: nobody has reviewed it.")] = False,
    kind: Annotated[Optional[str], typer.Option("--kind", help="For an archive: processors | ocr | destinations (default: from its manifest).")] = None,
) -> None:
    """Install from the registry, or from a local archive."""
    from aglaia.app_data import plugin_registry as reg

    path = Path(what).expanduser()
    if path.suffix.lower() in (".zip", ".aglplugin") and path.is_file():
        if not trust:
            _err("This archive did not come from the Aglaïa registry and nobody "
                 "has reviewed it. Once installed it runs with the same access "
                 "to your files as Aglaïa itself.\n"
                 "If you trust the author, run again with --trust.")
            raise typer.Exit(2)
        res = reg.install_from_archive(path, kind or "")
    else:
        index = reg.fetch_index()
        entry = index.get(what) if index else None
        if entry is None:
            _err(f"No plugin called {what!r} in the registry."
                 + (f" ({index.error})" if index and index.error else "")
                 + "\n`aglaia plugins search` lists what there is.")
            raise typer.Exit(1)
        typer.echo(f"Installing {entry.name} {entry.version} — reviewed and "
                   f"merged into the registry; written by "
                   f"{'Aglaïa' if entry.first_party else entry.author or 'its author'}.")
        res = reg.install_from_registry(
            entry, on_progress=lambda i, n, rel: typer.echo(f"  {rel} ({i}/{n})"))
    if not res.ok:
        _err(f"Not installed: {res.message}")
        raise typer.Exit(1)
    typer.echo(res.message)
    _post_install_hint(res.slug)


def _post_install_hint(slug: str) -> None:
    from aglaia.workers import destinations as dest
    dest.reset_for_tests()
    d = dest.load_all().get(slug)
    if d is not None and d.missing_settings():
        typer.echo(f"It still needs: {', '.join(d.missing_settings())}. "
                   f"Set it up with `aglaia plugins config {slug}`.")


@plugins_app.command()
def update(
    slug: Annotated[Optional[str], typer.Argument(help="One plugin, or omit with --all.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Every plugin the registry has a newer version of.")] = False,
) -> None:
    """Update to the registry's version, keeping settings, files and secrets."""
    from aglaia.app_data import plugin_registry as reg

    if not slug and not all_:
        _err("Say which plugin, or --all.")
        raise typer.Exit(2)
    index = reg.fetch_index()
    if index.error and not index.entries:
        _err(f"Could not reach the registry: {index.error}")
        raise typer.Exit(1)
    targets = [it["slug"] for it in reg.list_installed()] if all_ else [slug]
    done = 0
    for s in targets:
        e = index.get(s)
        if e is None:
            if not all_:
                _err(f"{s} is not in the registry.")
                raise typer.Exit(1)
            continue
        if not reg.update_available(s, e.version):
            if not all_:
                typer.echo(f"{s} is already at {e.version}.")
            continue
        res = reg.update_from_registry(e)
        typer.echo(("✓ " if res.ok else "! ") + res.message)
        done += int(res.ok)
    if all_ and not done:
        typer.echo("Everything is up to date.")
    if done:
        typer.echo("Restart Aglaïa for the new code to take effect.")


# ── toggle / remove ───────────────────────────────────────────────────

@plugins_app.command()
def toggle(slug: Annotated[str, typer.Argument()]) -> None:
    """Disable an enabled plugin, or enable a disabled one."""
    from aglaia.app_data import plugin_registry as reg
    if not reg.installed_record(slug):
        _err(f"{slug} is not installed.")
        raise typer.Exit(1)
    now = not reg.is_disabled(slug)
    reg.set_disabled(slug, now)
    typer.echo(f"{slug}: {'disabled' if now else 'enabled'}. "
               f"Takes effect at the next launch.")


@plugins_app.command()
def remove(
    slug: Annotated[str, typer.Argument()],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask.")] = False,
) -> None:
    """Uninstall — with its settings, its files and any password it stored."""
    from aglaia.app_data import plugin_registry as reg
    from aglaia.workers import destinations as dest
    if not reg.installed_record(slug) and not any(
            it["slug"] == slug for it in reg.list_installed()):
        _err(f"{slug} is not installed.")
        raise typer.Exit(1)
    if not yes:
        import sys
        if not sys.stdin.isatty():
            _err("This deletes the plugin, its settings, its files and any "
                 "password it stored in your keychain. Pass --yes to confirm.")
            raise typer.Exit(2)
        import questionary
        ok = questionary.confirm(
            f"Remove {slug}? This deletes the plugin, its settings, its files "
            f"and any password it stored in your keychain.", default=False).ask()
        if not ok:
            typer.echo("Kept.")
            return
    res = reg.uninstall(slug)
    dest.forget(slug)
    typer.echo(res.message)
    if not res.ok:
        raise typer.Exit(1)


# ── config ────────────────────────────────────────────────────────────

def _fields(d) -> list:
    return list(d.CONFIG_FIELDS) + list(d.SECRET_FIELDS)


def _current(d, f) -> str:
    """What to show for a field: its value, or that a secret is stored."""
    if f.kind == "secret":
        return "•••• stored" if d.secret(f.key) else "[red]not set[/red]"
    v = d.conf(f.key, f.default)
    return "" if v in (None, "") else str(v)


def _coerce(f, raw: str):
    """A typed value from a string, by the field's kind. Raises ValueError."""
    if f.kind == "int":
        return int(str(raw).strip())
    if f.kind == "bool":
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{f.label}: yes or no")
    if f.kind == "choice" and str(raw) not in tuple(f.choices):
        raise ValueError(f"{f.label}: one of {', '.join(f.choices)}")
    return str(raw)


def _write(d, f, value) -> None:
    if f.kind == "secret":
        d.ctx.secrets.set(f.key, str(value))
    else:
        d.ctx.config.set(f.key, value)


def _load_dest(slug: str):
    from aglaia.workers import destinations as dest
    d = dest.load_all().get(slug)
    if d is None:
        why = dest.load_error(slug)
        _err(f"{slug} has no settings to configure"
             + (f" — {why}" if why else " (not an export plugin, or not installed)."))
        raise typer.Exit(1)
    return d


@plugins_app.command()
def config(
    slug: Annotated[str, typer.Argument()],
    set_: Annotated[Optional[list[str]], typer.Option("--set", help="key=value; repeatable. Scripted alternative to the interactive view.")] = None,
    test: Annotated[bool, typer.Option("--test", help="Test the connection after setting.")] = False,
) -> None:
    """Configure an export plugin: a small interactive view, or --set key=value."""
    d = _load_dest(slug)
    fields = {f.key: f for f in _fields(d)}
    if set_:
        for item in set_:
            if "=" not in item:
                _err(f"--set wants key=value, got {item!r}")
                raise typer.Exit(2)
            k, v = item.split("=", 1)
            f = fields.get(k.strip())
            if f is None:
                _err(f"{slug} has no setting {k.strip()!r}. It has: "
                     f"{', '.join(fields)}")
                raise typer.Exit(2)
            try:
                _write(d, f, _coerce(f, v))
            except ValueError as e:
                _err(str(e))
                raise typer.Exit(2)
            typer.echo(f"{f.label}: {'set' if f.kind == 'secret' else v}")
        if test:
            _run_test(d)
        _show(d)
        return
    import sys
    if not sys.stdin.isatty():
        _show(d)
        typer.echo("\nNot a terminal — use --set key=value to change settings.")
        return
    _tui(d)


def _show(d) -> None:
    from rich.table import Table
    t = Table(show_header=True, header_style="bold", title=d.display or d.name)
    for col in ("setting", "value", ""):
        t.add_column(col, overflow="fold")
    for f in _fields(d):
        req = "[red]required[/red]" if f.required and f.key in d.missing_settings() else ""
        t.add_row(f.label, _current(d, f), req)
    _console().print(t)


def _run_test(d) -> None:
    from aglaia.gui.PluginsTab import KIND_LABEL_UI  # labels only; no Qt widgets
    res = d.check()
    label = KIND_LABEL_UI.get(getattr(res, "kind", ""), "")
    typer.echo(("✓ " if res.ok else "! ") + (f"{label} — " if label and not res.ok else "") + res.message)


def _tui(d) -> None:
    """Pick a setting, change it, repeat. Stored on each change, so quitting
    half-way loses nothing."""
    import questionary
    while True:
        _show(d)
        choices = [questionary.Choice(f"{f.label}" + (" *" if f.required and f.key in d.missing_settings() else ""), value=f.key)
                   for f in _fields(d)]
        choices += [questionary.Separator(),
                    questionary.Choice("Test connection", value="__test"),
                    questionary.Choice("Done", value="__done")]
        pick = questionary.select("Change which setting?", choices=choices).ask()
        if pick in (None, "__done"):
            return
        if pick == "__test":
            _run_test(d)
            continue
        f = next(x for x in _fields(d) if x.key == pick)
        prompt = f.label + (f"  ({f.help})" if f.help else "")
        if f.kind == "bool":
            v = questionary.confirm(prompt, default=bool(d.conf(f.key, f.default))).ask()
        elif f.kind == "choice":
            v = questionary.select(prompt, choices=list(f.choices),
                                   default=str(d.conf(f.key, f.default) or f.choices[0])).ask()
        elif f.kind == "secret":
            v = questionary.password(prompt + "  (stored in your keychain)").ask()
            if v == "":
                continue                       # empty = keep what is stored
        else:
            v = questionary.text(prompt, default=str(d.conf(f.key, f.default) or "")).ask()
        if v is None:
            return
        try:
            _write(d, f, _coerce(f, v) if f.kind != "bool" else bool(v))
        except ValueError as e:
            typer.echo(str(e))
