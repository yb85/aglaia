# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Heuristic-fallback warning for PageDetector.

If the pipeline asks for `auto` and `probe_active_backend()` returns
"heuristic" (i.e. no Vision / EAST / DBNet could load), the heuristic
projection method is quite worse than the ML alternatives. Block the
chain start and let the user either open the Model Downloader or accept
the degraded mode — with an opt-out checkbox stored in the per-user
config DB.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)


def _tr(text: str) -> str:
    return QCoreApplication.translate("PageWarningDialog", text)


def maybe_show_heuristic_warning(
    pipeline_def: dict,
    parent: Optional[QWidget] = None,
) -> str:
    """Return one of: "ok" (chain may start), "open_downloader",
    "abort". Reads `KEY_LAYOUT_HEURISTIC_NO_WARN` and skips the dialog
    if the user has silenced it. Walks `pipeline_def["pipeline"]` for a
    PageDetector step; the warning only fires when its backend option
    is "auto" (explicit picks mean the user knows what they want)."""
    backend_opt = _layout_backend_option(pipeline_def)
    if backend_opt is None:
        return "ok"  # no PageDetector in this pipeline
    if backend_opt != "auto":
        return "ok"  # user picked a backend explicitly — don't override

    try:
        from aglaia.app_data import db as cfg
        with cfg.session() as conn:
            silenced = bool(cfg.get(conn, cfg.KEY_LAYOUT_HEURISTIC_NO_WARN, False))
    except Exception:
        silenced = False
    if silenced:
        return "ok"

    try:
        from aglaia.processors.layout_backends.factory import probe_active_backend
        active = probe_active_backend("auto")
    except Exception:
        active = "heuristic"
    if active != "heuristic":
        return "ok"

    return _show_dialog(parent)


def _layout_backend_option(pipeline_def: dict) -> Optional[str]:
    """Extract PageDetector's `backend` option from a parsed pipeline
    YAML. Returns None if no PageDetector step is configured."""
    steps = (pipeline_def or {}).get("pipeline") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        proc = step.get("processor") or step.get("name")
        if proc != "PageDetector":
            continue
        opts = step.get("options") or {}
        return str(opts.get("backend", "auto")).lower()
    return None


def _show_dialog(parent: Optional[QWidget]) -> str:
    dlg = QDialog(parent)
    dlg.setWindowTitle(_tr("No page detection model"))
    dlg.setModal(True)
    dlg.setMinimumWidth(480)

    v = QVBoxLayout(dlg)
    v.setContentsMargins(20, 20, 20, 16)
    v.setSpacing(12)

    title = QLabel(_tr("No model found for PageDetection."))
    title.setStyleSheet("font-weight: 600; font-size: 14px;")
    v.addWidget(title)

    body = QLabel(
        _tr(
            "Aglaïa will fall back to a projection-profile heuristic that is "
            "noticeably worse than EAST, DBNet or Apple Vision — page splits "
            "may be sloppy, especially on dense layouts.\n\n"
            "Open the Model Downloader to fetch a proper detector, or accept "
            "the heuristic for now."
        )
    )
    body.setWordWrap(True)
    v.addWidget(body)

    no_warn = QCheckBox(_tr("Don't show this again"))
    no_warn.setChecked(False)
    v.addWidget(no_warn)

    buttons = QDialogButtonBox(Qt.Orientation.Horizontal)
    btn_dl = QPushButton(_tr("Open downloader"))
    btn_dl.setDefault(True)
    btn_accept = QPushButton(_tr("No thanks, use heuristic"))
    buttons.addButton(btn_dl, QDialogButtonBox.ButtonRole.AcceptRole)
    buttons.addButton(btn_accept, QDialogButtonBox.ButtonRole.RejectRole)
    v.addWidget(buttons)

    result = {"choice": "abort"}

    def _dl():
        result["choice"] = "open_downloader"
        dlg.accept()

    def _accept():
        result["choice"] = "ok"
        dlg.accept()

    btn_dl.clicked.connect(_dl)
    btn_accept.clicked.connect(_accept)
    dlg.exec()

    # Persist "don't show again" regardless of which path the user took.
    if no_warn.isChecked():
        try:
            from aglaia.app_data import db as cfg
            with cfg.session() as conn:
                cfg.set(conn, cfg.KEY_LAYOUT_HEURISTIC_NO_WARN, True)
        except Exception:
            pass

    return result["choice"]
