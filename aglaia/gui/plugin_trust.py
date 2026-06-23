# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Startup trust gate for drop-in plugins.

Shows a modal warning for every plugin file the user hasn't yet
acknowledged (new file, or content changed since acceptance). The user
adds it (becomes trusted + imported on next discovery), deletes it, or
skips it for this session.

This is the *only* place unacknowledged plugin code is surfaced before it
could run — discovery (`aglaia.app_data.plugins.import_accepted`) imports
solely accepted, sha-matching files, so an un-added file never executes.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from aglaia.app_data import plugins as _plugins


def prompt_pending_plugins(parent: QWidget | None = None) -> None:
    """Walk pending plugins and resolve each via a modal dialog.

    Safe to call once at startup before any widget that reads the
    processor / OCR registries is built. No-op when nothing is pending.
    """
    try:
        pending = _plugins.scan_pending()
    except Exception as e:  # noqa: BLE001 — never block startup on the gate
        print(f"[plugin-trust] scan skipped: {e}")
        return

    for cand in pending:
        verb = "changed since you accepted it" if cand.reason == "changed" \
            else "new and has never been run"
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Untrusted plugin")
        box.setText(
            f"A {cand.kind[:-1] if cand.kind.endswith('s') else cand.kind} "
            f"plugin is {verb}:\n\n{cand.path.name}"
        )
        box.setInformativeText(
            "Plugins run as untrusted Python code with full access to your "
            "files. Only add it if you trust the source.\n\n"
            f"Location: {cand.path}"
        )
        add_btn = box.addButton("Add (trust && run)",
                                QMessageBox.ButtonRole.AcceptRole)
        del_btn = box.addButton("Delete file",
                                QMessageBox.ButtonRole.DestructiveRole)
        skip_btn = box.addButton("Skip for now",
                                 QMessageBox.ButtonRole.RejectRole)
        # macOS sizes message-box buttons too narrow for these labels, clipping
        # them ("d (trust & r…"). Widen each to fit its own text.
        for _b in (add_btn, del_btn, skip_btn):
            _b.setMinimumWidth(_b.sizeHint().width() + 28)
        box.setDefaultButton(skip_btn)  # Skip — safe default
        box.exec()

        clicked = box.clickedButton()
        if clicked is add_btn:
            _plugins.acknowledge(cand)
        elif clicked is del_btn:
            _plugins.reject(cand, delete_file=True)
        # Skip → leave pending; re-prompted next startup. Not imported.
