# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The Plugins tab — browse the registry, install, configure, remove (#130).

Two lists and two install paths, and the difference between the paths is the
whole design:

* **From the registry** — reviewed, hash-pinned. One confirm, above a
  disclaimer naming the person who actually wrote it and linking to the code
  that was reviewed. Reviewing something does not make it ours, and the button
  says so: *I trust the code and/or its author*. The "and/or" is load-bearing —
  a reader who checked the diff and a reader who knows the author are both
  consenting truthfully, and neither should have to pretend to the other's
  grounds.
* **From a local archive** — reviewed by nobody. Red frame, the plugin's
  declared capabilities next to what it *actually* imports with the undeclared
  ones called out, and a sentence the user must type. No paste shortcut, no
  case-insensitive match; Cancel is the default button.

A plugin installed from an archive stays marked **UNREVIEWED** for as long as
it is installed. A one-time warning that vanishes is a warning the user forgets
he accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)

from aglaia.gui.colors import (
    COLOR_BG_OVERLAY_SOFT, COLOR_ERROR, COLOR_FONT_DIM, COLOR_FONT_MUTED,
    COLOR_FONT_PRIMARY, COLOR_OUTLINE_FAINT, COLOR_PRIMARY, COLOR_SUCCESS,
    COLOR_WARNING,
)
from aglaia.gui.theme import lucide

#: The sentence. Typed exactly, or the button stays dead.
#: Fallback only. The dialog asks for `trust_sentence()`, which is
#: translated: a French user made to transcribe an English sentence is not
#: affirming anything, they are copying shapes, and the whole point of typing
#: it is that it cannot be done absently.
TRUST_SENTENCE = "I TRUST THE AUTHOR OF THIS PLUGIN"


def trust_sentence() -> str:
    """The sentence the user must type to install an unreviewed plugin.

    Read through one function so the label and the comparison can never drift
    apart — two `tr()` calls with the same source would translate identically
    today and be a locked dialog the day one of them is edited."""
    from PySide6.QtCore import QCoreApplication
    return QCoreApplication.translate(
        "ArchiveInstallDialog", "I TRUST THE AUTHOR OF THIS PLUGIN")


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {COLOR_OUTLINE_FAINT};")
    return f


#: A card's fixed width. The list is browsed, not read line by line, so a
#: column of full-width rows wastes the pane and makes six plugins look like
#: sixty.
CARD_W = 330


def _card(border: str = "") -> QFrame:
    """A plugin card.

    The stylesheet is scoped to `QFrame#pluginCard`, NOT to `QFrame` —
    **`QLabel` is a subclass of `QFrame`**, so a bare `QFrame { … }` rule
    paints a background and a border on every label inside the card too. That
    is what turned each card into a stack of grey stripes."""
    f = QFrame()
    f.setObjectName("pluginCard")
    f.setFixedWidth(CARD_W)
    f.setStyleSheet(
        f"QFrame#pluginCard {{ background: {COLOR_BG_OVERLAY_SOFT}; "
        f"border: 1px solid {border or COLOR_OUTLINE_FAINT}; "
        f"border-radius: 10px; }}")
    return f


def _pill(text: str, colour: str) -> QLabel:
    """A small tag — a version, a kind. Scoped by object name for the same
    reason `_card` is."""
    lbl = QLabel(text)
    lbl.setObjectName("pluginPill")
    lbl.setStyleSheet(
        f"QLabel#pluginPill {{ color: {colour}; font-size: 10px; "
        f"font-weight: 600; border: 1px solid {colour}; border-radius: 7px; "
        f"padding: 1px 6px; background: transparent; }}")
    lbl.setSizePolicy(lbl.sizePolicy().horizontalPolicy().Fixed,
                      lbl.sizePolicy().verticalPolicy().Fixed)
    return lbl


#: What a plugin kind is CALLED. "destinations" is what the code calls the
#: folder; "export" is what the thing does, which is what a user is looking
#: for in a list.
KIND_LABEL = {
    "destinations": "export",
    "processors": "processing",
    "ocr": "OCR",
}


#: One word for each failure kind a plugin can report. Deliberately not the
#: enum's own name: "auth" is our vocabulary, "Wrong credentials" is theirs.
KIND_LABEL_UI = {
    "network": "Cannot connect",
    "auth": "Wrong credentials",
    "permission": "Not allowed",
    "server": "Server problem",
    "config": "Missing setting",
    "unknown": "Failed",
}


class _IndexJob(QThread):
    """Fetch the index off the GUI thread — it is a network call, and a tab
    that freezes while it opens is a tab people stop opening."""

    done = Signal(object)

    def run(self) -> None:
        from aglaia.app_data import plugin_registry as reg
        try:
            self.done.emit(reg.fetch_index())
        except Exception as e:  # noqa: BLE001
            from aglaia.app_data.plugin_registry import IndexResult
            self.done.emit(IndexResult(error=f"{type(e).__name__}: {e}"))


class _InstallJob(QThread):
    """Download and install off the GUI thread.

    The index fetch was already threaded; the install was not, and it is the
    slower of the two — one request per file in the plugin, each of which can
    take as long as the index did. On a slow link that was a minute of frozen
    window with no way to tell a slow install from a hung one."""

    done = Signal(object)
    progress = Signal(str)

    def __init__(self, entry, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry

    def run(self) -> None:
        from aglaia.app_data import plugin_registry as reg
        try:
            self.done.emit(reg.install_from_registry(
                self.entry,
                on_progress=lambda i, n, rel: self.progress.emit(
                    f"{self.entry.name}: {rel} ({i}/{n})")))
        except Exception as e:  # noqa: BLE001
            from aglaia.app_data.plugin_registry import InstallResult
            self.done.emit(InstallResult(
                False, f"{type(e).__name__}: {e}"))


class PluginSettingsDialog(QDialog):
    """A settings form built from a plugin's declared `Field`s.

    The host has never heard of this plugin and does not need to: `kind` says
    how to render, `required` says what to insist on, and `secret` says which
    box is masked and which store it goes to. Same idea as the OCR tab reading
    engine capability flags instead of hard-coding engine names."""

    def __init__(self, dest, parent=None) -> None:
        super().__init__(parent)
        self.dest = dest
        #: False once this dialog is gone, so a test that answers late does
        #: not write into a deleted widget.
        self._alive = True
        self._test_job = None
        self.setWindowTitle(self.tr("{name} — settings").format(
            name=dest.display or dest.name))
        self.setMinimumWidth(520)
        v = QVBoxLayout(self)
        v.setSpacing(10)

        if getattr(dest, "description", ""):
            top = QLabel(dest.description)
            top.setWordWrap(True)
            top.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
            v.addWidget(top)

        self._widgets: dict[str, tuple] = {}
        for field in dest.CONFIG_FIELDS:
            v.addWidget(self._field_row(field, False))

        if dest.SECRET_FIELDS:
            # Where a secret goes is the HOST's fact and it is the same for
            # every plugin, so it is said once here rather than repeated in
            # each plugin's field help. It also has to be TRUE: on a headless
            # Linux box with no Secret Service the value is written as plain
            # text, and the user is entitled to know that before typing a
            # password into the box.
            secrets = getattr(getattr(dest, "ctx", None), "secrets", None)
            in_keychain = bool(getattr(secrets, "available", False))
            note = QLabel(
                self.tr("Stored in your keychain.") if in_keychain else
                self.tr("No keychain available — these are stored as plain "
                        "text in the app's data folder."))
            note.setWordWrap(True)
            note.setStyleSheet(
                f"color: {COLOR_FONT_MUTED if in_keychain else COLOR_ERROR}; "
                f"font-size: 11px; padding-top: 6px;")
            v.addWidget(note)
            for field in dest.SECRET_FIELDS:
                v.addWidget(self._field_row(field, True))

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        v.addWidget(self._status)

        row = QHBoxLayout()
        self._test_btn = QPushButton(self.tr("Test connection"))
        self._test_btn.clicked.connect(self._on_test)
        row.addWidget(self._test_btn)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_save)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        v.addLayout(row)

    def _field_row(self, field, is_secret: bool) -> QWidget:
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)
        label = field.label + (" *" if field.required else "")
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-size: 12px; font-weight: 600;")
        col.addWidget(lbl)

        if field.kind == "bool":
            w: QWidget = QCheckBox()
            w.setChecked(bool(self.dest.conf(field.key, field.default)))
        elif field.kind == "int":
            w = QSpinBox()
            w.setRange(0, 10_000_000)
            try:
                w.setValue(int(self.dest.conf(field.key, field.default) or 0))
            except (TypeError, ValueError):
                w.setValue(0)
        elif field.kind == "choice":
            w = QComboBox()
            w.addItems(list(field.choices))
            cur = str(self.dest.conf(field.key, field.default) or "")
            if cur in field.choices:
                w.setCurrentText(cur)
        else:
            w = QLineEdit()
            if is_secret:
                w.setEchoMode(QLineEdit.EchoMode.Password)
                # Never round-trip the value through the form. Show that one
                # is stored; a settings dialog that hands a password back to
                # the screen is a settings dialog that leaks it to a
                # screenshot.
                if self.dest.secret(field.key):
                    w.setPlaceholderText(
                        self.tr("•••• stored — type to replace"))
                elif field.placeholder:
                    w.setPlaceholderText(field.placeholder)
            else:
                w.setText(str(self.dest.conf(field.key, field.default) or ""))
                if field.placeholder:
                    w.setPlaceholderText(field.placeholder)
        col.addWidget(w)

        if field.help:
            h = QLabel(field.help)
            h.setWordWrap(True)
            h.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 10px;")
            col.addWidget(h)
        self._widgets[field.key] = (field, w, is_secret)
        return wrap

    def _collect(self) -> None:
        for key, (field, w, is_secret) in self._widgets.items():
            if isinstance(w, QCheckBox):
                value = w.isChecked()
            elif isinstance(w, QSpinBox):
                value = w.value()
            elif isinstance(w, QComboBox):
                value = w.currentText()
            else:
                value = w.text()
            if is_secret:
                # An empty secret box means "leave what is stored", not
                # "delete it" — the box is empty by design on every open.
                if value:
                    self.dest.ctx.secrets.set(key, value)
            else:
                self.dest.ctx.config.set(key, value)

    def _on_test(self) -> None:
        """Test the connection off the GUI thread.

        `check()` reaches the network — an SMTP login, or an HTTP round trip
        to a server that may be asleep. On the GUI thread that froze the whole
        window, modal dialog included, for as long as the far end took to
        answer or time out."""
        from aglaia.gui.plugin_jobs import DestinationJob
        self._collect()
        self._status.setText(self.tr("Testing…"))
        self._status.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        self._test_btn.setEnabled(False)

        def _done(outcome) -> None:
            # The dialog can be closed while the test is in flight; the C++
            # object is then gone and touching a label raises out of a slot
            # nobody can catch.
            if not self._alive:
                return
            msg = outcome.message or self.tr("The server did not answer.")
            # A one-word label for WHICH kind of failure, when the plugin can
            # tell them apart. "Could not connect" and "wrong password" have
            # nothing in common except the word "failed", and a user who reads
            # only the first two words should already be looking in the right
            # place.
            label = KIND_LABEL_UI.get(getattr(outcome.result, "kind", ""), "")
            if label and not outcome.ok:
                msg = f"{self.tr(label)} — {msg}"
            self._status.setText(msg)
            self._status.setStyleSheet(
                f"color: {COLOR_SUCCESS if outcome.ok else COLOR_ERROR}; "
                f"font-size: 11px;")
            self._test_btn.setEnabled(True)
            # The machine detail never reaches the label. It is the thing that
            # makes a bug report useful and the thing that makes a dialog
            # unreadable.
            detail = getattr(outcome.result, "detail", None)
            if detail or outcome.error:
                print(f"[plugins] {self.dest.name} check: "
                      f"{outcome.error or detail}")

        job = DestinationJob(self.dest.check)
        job.done.connect(_done)
        job.finished.connect(job.deleteLater)
        self._test_job = job
        job.start()

    def closeEvent(self, ev):  # noqa: N802 — Qt API
        self._alive = False
        super().closeEvent(ev)

    def _on_save(self) -> None:
        self._collect()
        self.accept()


class RegistryInstallDialog(QDialog):
    """Consent for a reviewed plugin. One confirm, and a disclaimer that names
    whose code it is."""

    def __init__(self, entry, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle(self.tr("Install {name}").format(name=entry.name))
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.setSpacing(10)

        title = QLabel(f"{entry.name} {entry.version}")
        title.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-size: 16px; font-weight: 700;")
        v.addWidget(title)
        if entry.summary:
            s = QLabel(entry.summary)
            s.setWordWrap(True)
            s.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 12px;")
            v.addWidget(s)

        caps = entry.declared()
        if caps:
            c = QLabel(self.tr("It declares: ") + " · ".join(caps))
            c.setWordWrap(True)
            c.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 11px;")
            v.addWidget(c)
        if entry.imports:
            i = QLabel(self.tr("It imports: ") + ", ".join(entry.imports))
            i.setWordWrap(True)
            i.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
            v.addWidget(i)

        v.addWidget(_hline())
        if entry.first_party:
            # "Submitted by Aglaïa, not by Aglaïa" is not a disclaimer; it is
            # a sentence that makes a reader distrust the rest of the dialog.
            # The part worth keeping is the part that is still true of our own
            # code: it runs with your access, and being ours does not change
            # that.
            text = self.tr(
                "Written and maintained by Aglaïa, and installed through the "
                "same reviewed registry as everything else. Like any plugin, "
                "it runs with the same access to your files as Aglaïa itself.")
        else:
            text = self.tr(
                "Reviewed and merged into the Aglaïa plugin registry. It was "
                "written and submitted by <b>{who}</b>, not by Aglaïa, and it "
                "runs with the same access to your files as Aglaïa itself."
            ).format(who=entry.author or self.tr("its author"))
        disc = QLabel(text)
        disc.setWordWrap(True)
        disc.setTextFormat(Qt.TextFormat.RichText)
        disc.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        v.addWidget(disc)

        src = QPushButton(self.tr("Read the source on GitHub"))
        src.setFlat(True)
        src.setCursor(Qt.CursorShape.PointingHandCursor)
        src.setStyleSheet(
            f"QPushButton {{ color: {COLOR_PRIMARY}; border: none; "
            f"text-align: left; font-size: 11px; padding: 0; }}")
        src.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(entry.web_url)))
        v.addWidget(src)

        meta = QLabel(f"{entry.slug} · {entry.license or '—'}")
        meta.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 10px;")
        v.addWidget(meta)

        bb = QDialogButtonBox()
        self._ok = bb.addButton(self.tr("I trust the code and/or its author"),
                                QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


class ArchiveInstallDialog(QDialog):
    """The red gate. Nobody reviewed this one."""

    def __init__(self, man, scan, sha: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Unreviewed plugin"))
        self.setMinimumWidth(600)
        v = QVBoxLayout(self)
        v.setSpacing(10)

        head = QLabel("⚠  " + self.tr("UNREVIEWED PLUGIN"))
        head.setStyleSheet(
            f"color: {COLOR_ERROR}; font-size: 16px; font-weight: 800;")
        v.addWidget(head)

        warn = QLabel(self.tr(
            "This plugin did not come from the Aglaïa registry. Nobody has "
            "reviewed it. Once installed it runs with the same access to your "
            "files as Aglaïa itself."))
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-size: 12px;")
        v.addWidget(warn)

        who = man.author or self.tr("(no author given)")
        ident = QLabel(f"{man.name} {man.version} — {who}\nsha256 {sha[:16]}…")
        ident.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        v.addWidget(ident)

        caps = man.declared()
        v.addWidget(self._row(self.tr("It declares:"),
                              " · ".join(caps) or self.tr("nothing"),
                              COLOR_WARNING if caps else COLOR_FONT_MUTED))
        allowed = sorted({a for a in (scan.allowed if scan else [])
                          if not a.startswith(("aglaia", "__future__", "."))})
        v.addWidget(self._row(self.tr("It imports:"),
                              ", ".join(allowed) or self.tr("nothing beyond "
                                                            "the plugin API"),
                              COLOR_FONT_MUTED))
        undeclared = sorted(set(scan.undeclared)) if scan else []
        if undeclared:
            v.addWidget(self._row(
                self.tr("Undeclared:"),
                ", ".join(undeclared) + "   ← " + self.tr("not in its manifest"),
                COLOR_ERROR))
        if scan and scan.review:
            v.addWidget(self._row(
                self.tr("Worth a look:"), ", ".join(sorted(set(scan.review))),
                COLOR_WARNING))

        v.addWidget(_hline())
        ask = QLabel(self.tr("Type the sentence below to install it."))
        ask.setStyleSheet(f"color: {COLOR_FONT_PRIMARY}; font-size: 12px;")
        v.addWidget(ask)
        sentence = QLabel(trust_sentence())
        sentence.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 12px; "
            f"font-family: monospace;")
        v.addWidget(sentence)

        self._entry = QLineEdit()
        # No paste: the point of typing it is that it cannot be done absently.
        self._entry.setStyleSheet(
            f"QLineEdit {{ border: 2px solid {COLOR_ERROR}; border-radius: 6px; "
            f"padding: 6px; font-family: monospace; }}")
        self._entry.textChanged.connect(self._sync)
        v.addWidget(self._entry)

        bb = QDialogButtonBox()
        self._ok = bb.addButton(self.tr("Install"),
                                QDialogButtonBox.ButtonRole.AcceptRole)
        self._ok.setEnabled(False)
        self._ok.setStyleSheet(
            f"QPushButton:enabled {{ background: {COLOR_ERROR}; "
            f"color: white; font-weight: 700; border-radius: 6px; "
            f"padding: 6px 14px; }}")
        cancel = bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        cancel.setDefault(True)          # Return dismisses, never installs
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.setStyleSheet(f"QDialog {{ border: 2px solid {COLOR_ERROR}; }}")

    def _row(self, label: str, value: str, colour: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        lab.setFixedWidth(110)
        lab.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        val = QLabel(value)
        val.setWordWrap(True)
        val.setStyleSheet(f"color: {colour}; font-size: 11px;")
        h.addWidget(lab)
        h.addWidget(val, 1)
        return w

    def _sync(self, text: str) -> None:
        # Exact match. Not stripped, not case-folded: a ritual that accepts an
        # approximation is not a ritual.
        self._ok.setEnabled(text == trust_sentence())


class PluginsTab(QWidget):
    """Installed plugins, and what the registry has to offer."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = None
        self._job: Optional[_IndexJob] = None
        self._elapsed = 0
        self._installing = ""
        self._install_job: Optional[_InstallJob] = None
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel(self.tr("Plugins"))
        title.setObjectName("SectionTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {COLOR_FONT_DIM}; font-size: 11px;")
        bar.addWidget(self._status)
        self._file_btn = QPushButton(self.tr("Install from file…"))
        self._file_btn.setIcon(lucide("folder-open", color=COLOR_FONT_MUTED, size=13))
        self._file_btn.clicked.connect(self._install_from_file)
        bar.addWidget(self._file_btn)
        self._refresh_btn = QPushButton(self.tr("Refresh"))
        self._refresh_btn.setIcon(lucide("refresh-cw", color=COLOR_PRIMARY,
                                         size=13))
        self._refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self._refresh_btn)
        root.addLayout(bar)

        # No standing warning banner here: the install dialogs carry it, at
        # the moment it is a decision. A permanent one is read once and then
        # never again, which is worse than none — it trains the eye to skip
        # the place warnings appear.

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        scroll.setWidget(self._body)
        root.addWidget(scroll, 1)

        self.refresh()

    # ── data ──────────────────────────────────────────────────────────
    def refresh(self) -> None:
        self._index = None
        self._rebuild()
        if self._job is not None and self._job.isRunning():
            return
        self._elapsed = 0
        self._status.setText(self.tr("Checking the registry…"))
        self._refresh_btn.setEnabled(False)
        # A static label on a fetch that can take half a minute reads as a
        # hang. Count up so it is visibly alive, and say what it is waiting on.
        self._tick.start()
        self._job = _IndexJob(self)
        self._job.done.connect(self._on_index)
        self._job.start()

    def _on_tick(self) -> None:
        self._elapsed += 1
        self._status.setText(
            self.tr("Checking the registry… {n}s").format(n=self._elapsed))
        if self._elapsed == 10:
            self._status.setToolTip(self.tr(
                "Fetching index.json from GitHub. Slow here is usually the "
                "network, not the registry — it will fall back to the last "
                "copy it saw."))

    def _on_index(self, index) -> None:
        self._tick.stop()
        self._refresh_btn.setEnabled(True)
        self._index = index
        if index.error:
            self._status.setText(index.error)
        else:
            # Just the count. Whether the index is signed is a property of
            # the distribution, not something a user standing here can act
            # on — it belongs in the docs, not in a status line they read
            # every time they open the tab.
            self._status.setText(
                self.tr("{n} in the registry").format(n=len(index.entries)))
        self._rebuild()

    # ── rendering ─────────────────────────────────────────────────────
    def _clear(self) -> None:
        while self._body_layout.count():
            it = self._body_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def _heading(self, text: str) -> QLabel:
        h = QLabel(text)
        h.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: 11px; font-weight: 700; "
            f"text-transform: uppercase; letter-spacing: .08em;")
        return h

    def _rebuild(self) -> None:
        from aglaia.app_data import plugin_registry as reg
        self._clear()

        installed = reg.list_installed()
        self._body_layout.addWidget(self._heading(self.tr("Installed")))
        if not installed:
            empty = QLabel(self.tr("Nothing installed yet."))
            empty.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
            self._body_layout.addWidget(empty)
        self._body_layout.addWidget(
            self._grid([self._installed_card(i) for i in installed]))

        have = {i["slug"] for i in installed}
        entries = [e for e in (self._index.entries if self._index else [])
                   if e.slug not in have]
        self._body_layout.addWidget(self._heading(self.tr("Available")))
        if not entries:
            if self._index is None:
                # Still in flight. Saying "not reachable" here was a lie the
                # user had no way to tell from the truth — and a slow fetch
                # then read as a broken one.
                msg = self.tr("Checking the registry…")
            elif self._index.error:
                msg = self._index.error
            else:
                msg = self.tr("Everything in the registry is installed.")
            lbl = QLabel(msg)
            lbl.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
            self._body_layout.addWidget(lbl)
        self._body_layout.addWidget(
            self._grid([self._available_card(e) for e in entries]))
        self._body_layout.addStretch(1)

    def _grid(self, cards: list) -> QWidget:
        """Cards that wrap to the pane's width.

        `FlowLayout` is the app's own — the scans grid uses it — so the
        wrapping behaves the way the rest of Aglaïa does instead of being a
        second, subtly different one."""
        from aglaia.gui.FlowLayout import FlowContentWidget, FlowLayout
        host = FlowContentWidget()
        lay = FlowLayout(host, h_spacing=10, v_spacing=10)
        for c in cards:
            lay.insertWidget(-1, c)
        return host

    def _installed_card(self, item: dict) -> QWidget:
        from aglaia.app_data import plugin_registry as reg
        man = item.get("manifest")
        rec = item.get("record") or {}
        source = str(rec.get("source") or "")
        unreviewed = source == "zip"
        card = _card(COLOR_ERROR if unreviewed else "")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        lbl = QLabel(man.name if man else item["slug"])
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 700; font-size: 14px;")
        v.addWidget(lbl)

        tags = QHBoxLayout()
        tags.setSpacing(5)
        if man:
            tags.addWidget(_pill(man.version, COLOR_FONT_MUTED))
        tags.addWidget(_pill(KIND_LABEL.get(item["kind"], item["kind"]),
                             COLOR_PRIMARY))
        if unreviewed:
            tags.addWidget(_pill(self.tr("UNREVIEWED"), COLOR_ERROR))
        newer = self._newer_version(item["slug"])
        if newer:
            tags.addWidget(_pill(self.tr("UPDATE"), COLOR_PRIMARY))
        tags.addStretch(1)
        v.addLayout(tags)

        if item.get("error"):
            err = QLabel(item["error"])
            err.setWordWrap(True)
            err.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 11px;")
            v.addWidget(err)
        elif man and man.summary:
            s = QLabel(man.summary)
            s.setWordWrap(True)
            s.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
            v.addWidget(s)

        if man:
            caps = man.declared()
            if caps:
                c = QLabel(" · ".join(caps))
                c.setWordWrap(True)
                c.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 10px;")
                v.addWidget(c)

        row = QHBoxLayout()
        slug = item["slug"]
        dis = QCheckBox(self.tr("Disabled"))
        dis.setChecked(reg.is_disabled(slug))
        dis.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 11px;")
        dis.toggled.connect(lambda on, s=slug: self._set_disabled(s, on))
        row.addWidget(dis)
        row.addStretch(1)
        if item["kind"] == "destinations":
            cfg = QPushButton(self.tr("Settings…"))
            cfg.clicked.connect(lambda _=False, s=slug: self._configure(s))
            row.addWidget(cfg)
        if newer:
            up = QPushButton(self.tr("Update to {v}").format(v=newer.version))
            up.setStyleSheet("font-weight: 600;")
            up.clicked.connect(lambda _=False, e=newer: self._update(e))
            row.addWidget(up)
        rm = QPushButton(self.tr("Uninstall"))
        rm.clicked.connect(lambda _=False, s=slug: self._uninstall(s))
        row.addWidget(rm)
        v.addLayout(row)
        return card

    def _newer_version(self, slug: str):
        """The registry entry for `slug` if it is newer than what is
        installed, else None. Reads the index already fetched — an update
        check must not become a second network round trip per card."""
        from aglaia.app_data import plugin_registry as reg
        index = getattr(self, "_index", None)
        entry = index.get(slug) if index is not None else None
        if entry is None:
            return None
        return entry if reg.update_available(slug, entry.version) else None

    def _update(self, entry) -> None:
        """Update in place, keeping everything the plugin owns.

        Not uninstall-then-install: uninstall deletes the plugin's data
        directory, which for the stamp remover is every hand-traced stamp and
        for a destination is the stored password. Nobody expects an update to
        cost them that."""
        from aglaia.app_data import plugin_registry as reg
        from aglaia.workers import destinations as dest
        res = reg.update_from_registry(entry)
        if not res.ok:
            QMessageBox.warning(
                self, self.tr("Could not update {name}").format(
                    name=entry.name), res.message)
            return
        # The old module is still imported; the new code only takes effect on
        # the next launch, and saying so is better than the user wondering why
        # a fixed plugin still misbehaves.
        dest.forget(entry.slug)
        dest.reset_for_tests()
        self.refresh()
        QMessageBox.information(
            self, self.tr("{name} updated").format(name=entry.name),
            self.tr("Updated to {v}. Restart Aglaïa for it to take effect.")
            .format(v=entry.version))

    def _available_card(self, entry) -> QWidget:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)
        lbl = QLabel(entry.name)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {COLOR_FONT_PRIMARY}; font-weight: 700; font-size: 14px;")
        v.addWidget(lbl)

        tags = QHBoxLayout()
        tags.setSpacing(5)
        tags.addWidget(_pill(entry.version, COLOR_FONT_MUTED))
        tags.addWidget(_pill(KIND_LABEL.get(entry.kind, entry.kind),
                             COLOR_PRIMARY))
        tags.addStretch(1)
        v.addLayout(tags)

        top = QHBoxLayout()
        top.addStretch(1)
        busy = (self._installing == entry.slug)
        btn = QPushButton(self.tr("Installing…") if busy
                          else self.tr("Install…"))
        # While one install is in flight every other Install is dead too:
        # two concurrent installs would race on the same registry client and
        # the same status line for no benefit.
        btn.setEnabled(not self._installing)
        btn.clicked.connect(lambda _=False, e=entry: self._install(e))

        if entry.summary:
            sm = QLabel(entry.summary)
            sm.setWordWrap(True)
            sm.setStyleSheet(f"color: {COLOR_FONT_DIM}; font-size: 11px;")
            v.addWidget(sm)
        byline = QLabel(
            self.tr("by Aglaïa") if entry.first_party
            else self.tr("by {who}").format(
                who=entry.author or self.tr("an unnamed author")))
        byline.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: 10px;")
        v.addWidget(byline)
        caps = entry.declared()
        if caps:
            c = QLabel(" · ".join(caps))
            c.setWordWrap(True)
            c.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 10px;")
            v.addWidget(c)
        v.addStretch(1)
        top.addWidget(btn)
        v.addLayout(top)
        return card

    # ── actions ───────────────────────────────────────────────────────
    def _set_disabled(self, slug: str, on: bool) -> None:
        from aglaia.app_data import plugin_registry as reg
        from aglaia.workers import destinations as dest
        reg.set_disabled(slug, on)
        dest.reset_for_tests()

    def _install(self, entry) -> None:
        if RegistryInstallDialog(entry,
                                 self).exec() != QDialog.DialogCode.Accepted:
            return
        if self._install_job is not None and self._install_job.isRunning():
            return
        # Off the GUI thread: one request per file, and on a slow link that is
        # a minute of beach ball if it runs here.
        self._installing = entry.slug
        self._status.setText(
            self.tr("Installing {name}…").format(name=entry.name))
        self._rebuild()
        job = _InstallJob(entry, self)
        job.progress.connect(self._status.setText)
        job.done.connect(lambda res, e=entry: self._on_installed(e, res))
        self._install_job = job
        job.start()

    def _on_installed(self, entry, res) -> None:
        from aglaia.workers import destinations as dest
        self._installing = ""
        self._install_job = None
        if not res.ok:
            self._status.setText(res.message)
            QMessageBox.warning(self, self.tr("Install failed"), res.message)
            self._rebuild()
            return
        dest.reset_for_tests()
        self._status.setText(res.message)
        self._rebuild()
        self._notify_export_tab()

    def _notify_export_tab(self) -> None:
        """The Export tab lists installed destinations; a new or removed one
        must show up there without a restart."""
        w = self.window()
        tab = getattr(w, "_export_tab", None)
        if tab is not None and hasattr(tab, "refresh_destinations"):
            try:
                tab.refresh_destinations()
            except Exception:
                pass

    def _install_from_file(self) -> None:
        from aglaia.app_data import plugin_registry as reg
        from aglaia.workers import destinations as dest
        import hashlib
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Install a plugin archive"), "",
            self.tr("Aglaïa plugin (*.aglplugin *.zip)"))
        if not path:
            return
        man, files, scan, err = reg.stage_archive(Path(path))
        if err or man is None:
            QMessageBox.warning(self, self.tr("Not a usable plugin"),
                                err or self.tr("unreadable archive"))
            return
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if ArchiveInstallDialog(man, scan, sha,
                                self).exec() != QDialog.DialogCode.Accepted:
            return
        res = reg.install_from_archive(Path(path), man.kind or "destinations")
        if not res.ok:
            QMessageBox.warning(self, self.tr("Install failed"), res.message)
            return
        dest.reset_for_tests()
        self._status.setText(res.message)
        self._rebuild()
        self._notify_export_tab()

    def _uninstall(self, slug: str) -> None:
        from aglaia.app_data import plugin_registry as reg
        from aglaia.workers import destinations as dest
        if QMessageBox.question(
                self, self.tr("Remove {slug}?").format(slug=slug),
                self.tr("This deletes the plugin, its settings, its files and "
                        "any password it stored in your keychain.")
        ) != QMessageBox.StandardButton.Yes:
            return
        res = reg.uninstall(slug)
        # Files gone is not enough: the class is still in the registry and
        # the module still in sys.modules, so the destination would go on
        # being listed and offered until the app restarted.
        dest.forget(slug)
        dest.reset_for_tests()
        self._status.setText(res.message)
        self._rebuild()
        self._notify_export_tab()
        self._notify_export_tab()

    def _configure(self, slug: str) -> None:
        from aglaia.workers import destinations as dest
        d = dest.load_all().get(slug)
        if d is None:
            # Two audiences, two sentences. The user gets what it means for
            # them and what to do; the log gets the Python reason, which is
            # for whoever wrote the plugin.
            detail = dest.load_detail(slug)
            if detail:
                print(f"[plugins] {slug}: {detail}")
            QMessageBox.warning(
                self, self.tr("{slug} cannot be used").format(slug=slug),
                self.tr("This plugin is damaged. Remove it below, or report "
                        "it to whoever wrote it."))
            return
        PluginSettingsDialog(d, self).exec()
