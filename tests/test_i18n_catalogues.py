# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Every locale the Settings combo offers must actually be translatable (#170).

The packaged app shipped for a release with no app catalogue at all: the .qm
files were in the wheel but not in `Aglaia.spec`'s `datas`, so
`install_translator` found nothing and every string came back in English while
Settings still said Français. `QTranslator.load()` returning False was not
checked, so nothing said so.

This is a PACKAGING test, not a GUI one, and it deliberately does not import
`aglaia.i18n` — that module imports PySide6, which the CI test job does not
install, and a guard that skips on CI is not a guard. `SUPPORTED_LOCALES` is
read out of the source with `ast` instead.
"""

from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_I18N = _REPO / "aglaia" / "i18n"
CATALOG = "aglaia"


def _supported_locales() -> list[str]:
    """The locale codes `aglaia/i18n/__init__.py` offers, without importing
    it. The "" entry is auto (follow the system), which may be any language on
    earth and is allowed to have no catalogue."""
    tree = ast.parse((_I18N / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if "SUPPORTED_LOCALES" in names:
            pairs = ast.literal_eval(node.value)
            return [code for code, _label in pairs if code]
    raise AssertionError("SUPPORTED_LOCALES not found in aglaia/i18n")


LOCALES = _supported_locales()


def test_there_is_at_least_one_locale_to_check():
    assert LOCALES


@pytest.mark.parametrize("locale", LOCALES)
def test_every_offered_locale_has_a_compiled_catalogue(locale):
    qm = _I18N / "qm" / f"{CATALOG}_{locale}.qm"
    assert qm.is_file(), (
        f"{qm.relative_to(_REPO)} is missing — run scripts/i18n_compile.sh. "
        f"Settings offers {locale} and the loader would fall back to English.")
    assert qm.stat().st_size > 0


@pytest.mark.parametrize("locale", LOCALES)
def test_every_offered_locale_has_its_source_catalogue(locale):
    assert (_I18N / f"{CATALOG}_{locale}.ts").is_file()


def test_french_is_fully_translated():
    """French is the project's own second language — an untranslated string
    there is an oversight, not a backlog."""
    ts = _I18N / f"{CATALOG}_fr_FR.ts"
    unfinished = [
        m.findtext("source")
        for m in ET.parse(ts).getroot().findall(".//message")
        if (m.find("translation") is not None
            and m.find("translation").get("type") == "unfinished")
    ]
    assert unfinished == [], (
        f"{len(unfinished)} string(s) with no French: {unfinished[:10]}")


def test_the_compiled_catalogues_ship_in_the_macos_bundle():
    """`Aglaia.spec` is a separate list from the wheel's package-data, and it
    is the one that was missing them."""
    spec = (_REPO / "Aglaia.spec").read_text(encoding="utf-8")
    assert re.search(r'"i18n"\s*/\s*"qm"', spec) or "i18n/qm" in spec, (
        "Aglaia.spec does not ship aglaia/i18n/qm — the frozen app would "
        "have no translations at all.")


def test_the_compiled_catalogues_ship_in_the_wheel():
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "i18n/qm/*.qm" in pyproject
