# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""`aglaia skill` prints the agent skill, and the skill stays true to the CLI.

The skill is prose about commands and options. Prose rots: an option renamed in
`run.py` leaves a sentence recommending a flag that no longer exists, and an
agent reading it will run a command that fails. So the test walks the real
Typer app and demands that every command and every option it finds is named
in the skill — the document cannot fall behind the code without failing here.
"""
from __future__ import annotations

import typer
from typer.testing import CliRunner

from aglaia.assets import asset_path
from aglaia.cli import KNOWN_COMMANDS, app

def _skill() -> str:
    return asset_path("SKILL.md").read_text(encoding="utf-8")


def test_skill_prints_the_asset_verbatim():
    r = CliRunner().invoke(app, ["skill"])
    assert r.exit_code == 0, r.output
    assert r.output == _skill()


def test_skill_is_reachable_through_the_real_entry_point(capsys):
    """`aglaia skill` must not be swallowed by the default-command pre-parse
    (which would open the GUI with "skill" as a project path)."""
    from aglaia.cli import run
    assert run(["skill"]) == 0
    assert capsys.readouterr().out == _skill()


def test_skill_has_frontmatter():
    text = _skill()
    assert text.startswith("---\nname: aglaia-cli\n")
    assert "\ndescription: " in text.split("---", 2)[1]


def _walk(cmd, prefix: str):
    yield prefix, cmd
    # Duck-typed: Typer's group class is not a `click.Group` subclass in
    # every click version, but it always carries `.commands`.
    for name, sub in (getattr(cmd, "commands", None) or {}).items():
            yield from _walk(sub, f"{prefix} {name}")


def test_every_command_and_option_is_named_in_the_skill():
    text = _skill()
    root = typer.main.get_command(app)
    assert set(root.commands) == KNOWN_COMMANDS
    missing = []
    for path, cmd in _walk(root, "aglaia"):
        if path != "aglaia" and path not in text:
            missing.append(path)
        for param in cmd.params:
            # click is typer's dependency, not ours to import — the release
            # gate on Linux/Windows proved it absent. An option is the param
            # kind that spells itself with dashes.
            if getattr(param, "param_type_name", "") != "option":
                continue
            longs = [o for o in param.opts if o.startswith("--")]
            if longs and longs[0] != "--help" and not any(o in text for o in longs):
                missing.append(f"{path} {longs[0]}")
    assert not missing, f"skill does not mention: {missing}"


def test_every_bundled_pipeline_and_export_is_named():
    from aglaia.assets import config_path
    text = _skill()
    for y in sorted(config_path("pipelines").glob("*.yaml")):
        assert f"`{y.stem}`" in text, y.stem
    for tok in ("pdf:g4", "pdf:jbig2", "pdf:native", "md:refine="):
        assert tok in text, tok
