# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Runtime translation install for the Qt GUI.

Source language: en-US (strings as written in the code).
Target catalogues: aglaia/i18n/aglaia_<locale>.ts → compiled .qm under
aglaia/i18n/qm/.

Workflow:

* Mark strings via ``self.tr("…")`` inside QObject subclasses, or
  ``QCoreApplication.translate("Ctx", "…")`` elsewhere.
* Run ``scripts/i18n_extract.sh`` to (re)generate the .ts files.
* Translate with Qt Linguist (``pyside6-linguist``).
* Run ``scripts/i18n_compile.sh`` to produce the .qm binaries.
* The loader picks them up at QApplication startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator

# Public catalogue name — matches the .ts / .qm filename stem prefix.
CATALOG = "aglaia"

# Per-locale display names exposed in the Settings combo. The key is the
# canonical Qt locale string (language_TERRITORY) — used as the filename
# suffix and stored as-is in the per-user config DB. "" = auto (follow
# QLocale.system()).
SUPPORTED_LOCALES: list[tuple[str, str]] = [
    ("", "Auto (system)"),
    ("en_US", "English (US)"),
    ("fr_FR", "Français"),
]

_I18N_DIR = Path(__file__).resolve().parent
_QM_DIR = _I18N_DIR / "qm"

# Keep references alive — Qt does not own installed QTranslator objects.
_installed: list[QTranslator] = []


def resolve_locale(preferred: Optional[str]) -> str:
    """Map a user preference to a concrete locale string.

    Empty / unknown → ``QLocale.system().name()`` (falls back to en_US
    when the system locale has no catalogue).
    """
    if preferred:
        return preferred
    sys_name = QLocale.system().name() or "en_US"
    # Strip script tags ("zh_Hans_CN" → keep as-is; we accept what Qt
    # gives us). If we don't have a catalogue, the loader silently falls
    # back to source strings.
    return sys_name


def install_translator(app, preferred: Optional[str] = None) -> str:
    """Install the right ``QTranslator`` for ``preferred`` on ``app``.

    Returns the resolved locale string actually applied (useful for
    logging / Settings status). Safe to call multiple times — drops any
    previously-installed translators first.
    """
    global _installed
    for t in _installed:
        try:
            app.removeTranslator(t)
        except Exception:
            pass
    _installed = []

    locale = resolve_locale(preferred)

    # 1. App catalogue (aglaia_<locale>.qm).
    app_t = QTranslator(app)
    if app_t.load(f"{CATALOG}_{locale}", str(_QM_DIR)):
        app.installTranslator(app_t)
        _installed.append(app_t)

    # 2. Stock Qt strings (qtbase_<lang>.qm) — covers stock dialog
    # buttons (Cancel / OK), QMessageBox titles, etc. Best-effort.
    qt_t = QTranslator(app)
    qt_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if qt_t.load(QLocale(locale), "qtbase", "_", qt_dir):
        app.installTranslator(qt_t)
        _installed.append(qt_t)

    return locale


def tr(ctx: str, text: str, disambiguation: Optional[str] = None,
       n: int = -1) -> str:
    """Translate a string outside a QObject. Shortcut for
    ``QCoreApplication.translate``."""
    return QCoreApplication.translate(ctx, text, disambiguation, n)
