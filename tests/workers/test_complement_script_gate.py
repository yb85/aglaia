# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Script-anomaly garbage detection for the apple_docs complement gate.

Apple Vision renders ancient Greek as confusable CYRILLIC — often with HIGH
confidence, so the confidence gate alone leaks it. A line carrying a script the
document's languages don't cover is a near-certain misread and must go to the
complement regardless of confidence."""

from __future__ import annotations

from aglaia.workers.ocr.apple_docs import _has_unexpected_script

FR_EL = ["fr-FR", "el-GR"]


def test_cyrillic_misread_flagged():
    # Vision's Greek→Cyrillic garble in a French+Greek document.
    assert _has_unexpected_script("Ібо той Лохои, ібок той П", FR_EL) is True


def test_legit_greek_not_flagged():
    assert _has_unexpected_script("Ἰδιον τοῦ Λόγου, Ἰδιον τ", FR_EL) is False


def test_legit_french_not_flagged():
    assert _has_unexpected_script("propre de la substance du Père", FR_EL) is False


def test_single_stray_char_tolerated():
    # One confusable char is OCR noise, not a misread line (min_chars=2).
    assert _has_unexpected_script("Athanase et Cyrille", FR_EL) is False


def test_cyrillic_allowed_when_language_covers_it():
    # A Russian document → Cyrillic is expected, not garbage.
    assert _has_unexpected_script("Ибо той Логос", ["ru-RU"]) is False


def test_default_scripts_when_no_languages():
    # No languages → default expected = Latin + Greek; Cyrillic still flagged.
    assert _has_unexpected_script("Ібо той Лохои", []) is True
    assert _has_unexpected_script("Ἰδιον τοῦ Λόγου", []) is False
