# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Language→script inference + unexpected-script (garbage) detection.

Apple Vision falls back to a confusable known script for scripts it can't read
(ancient Greek → Cyrillic) with HIGH confidence, so a script the chosen
languages don't cover is a near-certain misread. Coverage is script-level via
langcodes + regex \\p{sc}, so plugins / unknown-language-known-script work."""

from __future__ import annotations

from aglaia.workers.ocr.text_scripts import (
    expected_scripts,
    has_unexpected_script,
    scripts_for_language,
)

FR_EL = ["fr-FR", "el-GR"]


def test_language_to_script_forms():
    assert scripts_for_language("el") == frozenset({"Grek"})
    assert scripts_for_language("el-GR") == frozenset({"Grek"})
    assert scripts_for_language("grc") == frozenset({"Grek"})       # CLDR quirk fixed
    assert scripts_for_language("fra") == frozenset({"Latn"})        # 639-2/3
    assert scripts_for_language("ru") == frozenset({"Cyrl"})
    assert scripts_for_language("ar") == frozenset({"Arab"})
    assert scripts_for_language("zh") == frozenset({"Hani"})
    assert scripts_for_language("sr-Cyrl") == frozenset({"Cyrl"})    # explicit subtag
    assert scripts_for_language("sr-Latn") == frozenset({"Latn"})


def test_aggregate_writing_systems_expand():
    assert scripts_for_language("ja") == frozenset({"Hani", "Hira", "Kana"})
    assert scripts_for_language("ko") == frozenset({"Hang", "Hani"})


def test_auto_and_unknown_yield_nothing():
    assert scripts_for_language("auto") == frozenset()
    assert scripts_for_language("") == frozenset()


def test_expected_scripts_union():
    assert expected_scripts(("fr-FR", "el-GR")) == frozenset({"Latn", "Grek"})


def test_cyrillic_misread_flagged_in_fr_el():
    assert has_unexpected_script("Ібо той Лохои, ібок той П", FR_EL) is True


def test_legit_greek_and_french_pass():
    assert has_unexpected_script("Ἰδιον τοῦ Λόγου", FR_EL) is False
    assert has_unexpected_script("propre de la substance du Père", FR_EL) is False


def test_chinese_flagged_unless_language_covers_it():
    assert has_unexpected_script("這是中文", FR_EL) is True
    assert has_unexpected_script("這是中文", ["zh"]) is False


def test_single_stray_char_tolerated():
    assert has_unexpected_script("Café Ж", FR_EL) is False   # 1 < min_chars=2


def test_no_languages_no_judgement():
    # No hidden domain default: with no expected scripts, nothing is flagged.
    assert has_unexpected_script("Ібо той Лохои", []) is False
    assert has_unexpected_script("Ібо той Лохои", ["auto"]) is False
