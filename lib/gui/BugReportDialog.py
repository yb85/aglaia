# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Bug-report generator.

A small dialog that collects opt-in diagnostics into a per-project folder
``aglaia_debug_report_<timestamp>/`` (a Markdown report + optional scan
debug images), reveals it in the file manager, and shows how to file it as
a GitHub issue. Wired from MainWindow's "Report a bug" affordances.

Everything is best-effort: a failing collector contributes an error note,
never aborts the report.
"""

from __future__ import annotations

import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout,
)

GIT_REPO = "https://github.com/yb85/aglaia"
ISSUES_NEW_URL = GIT_REPO + "/issues/new"
ISSUES_SEARCH_URL = GIT_REPO + "/issues"
APP_VERSION = "0.1.0"

# Config keys whose values we never write to the report.
_SECRET_HINT = ("key", "token", "secret", "password", "api")


RELEASES_URL = GIT_REPO + "/releases/latest"

_UPDATE_TTL_S = 6 * 3600      # re-check at most every 6 h
_update_cache: dict = {"at": -1e18, "ok": True}


def _repo_slug() -> str:
    """``owner/repo`` parsed from GIT_REPO (``https://github.com/owner/repo``)."""
    return GIT_REPO.rstrip("/").split("github.com/", 1)[-1]


def _ver_tuple(s: str) -> tuple[int, ...]:
    import re
    nums = [int(n) for n in re.findall(r"\d+", s or "")][:3]
    return tuple(nums + [0] * (3 - len(nums)))


def latest_release_version(timeout: float = 3.0) -> Optional[str]:
    """Latest published GitHub release tag (leading 'v' stripped), or None."""
    import json
    import urllib.request
    url = f"https://api.github.com/repos/{_repo_slug()}/releases/latest"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "aglaia-update-check"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    tag = (data.get("tag_name") or "").strip().lstrip("vV")
    return tag or None


def is_up_to_date() -> bool:
    """True when this build is at least the latest GitHub release.

    Cached (6 h) and FAIL-OPEN: any error — no releases yet, 404, offline,
    timeout — returns True, so a network hiccup never blocks the bug report
    or nags falsely. Only a confirmed newer release flips it to False."""
    import time
    now = time.monotonic()
    if now - _update_cache["at"] < _UPDATE_TTL_S:
        return _update_cache["ok"]
    ok = True
    try:
        latest = latest_release_version()
        if latest:
            ok = _ver_tuple(APP_VERSION) >= _ver_tuple(latest)
    except Exception:
        ok = True
    _update_cache.update(at=now, ok=ok)
    return ok


def version_tag() -> str:
    """A short provenance tag for the report: ``release/<ver>`` plus
    ``HEAD@<short-commit>`` when running from a git checkout."""
    tag = f"release/{APP_VERSION}"
    try:
        import subprocess
        repo = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo),
            capture_output=True, text=True, timeout=2)
        commit = out.stdout.strip()
        if commit:
            tag += f" · HEAD@{commit}"
    except Exception:
        pass
    return tag


def open_bug_report(main_window) -> None:
    """Entry point. Warns (politely) when the build is out of date before
    letting the user file a report — a stale-version bug wastes everyone's
    time. Upgrade / submit-anyway / cancel."""
    if not is_up_to_date():
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(main_window)
        box.setWindowTitle(QMessageBox.tr("Update available"))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(QMessageBox.tr(
            "You're not running the latest version of Aglaïa.\n\n"
            "Many bugs are already fixed upstream — filing one from an old "
            "build can waste time for you and the maintainers. Please "
            "upgrade and check whether the problem is still there first."))
        up = box.addButton(QMessageBox.tr("Upgrade"),
                           QMessageBox.ButtonRole.AcceptRole)
        anyway = box.addButton(
            QMessageBox.tr("I understand, submit anyway"),
            QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.tr("Cancel"),
                      QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is up:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(RELEASES_URL))
            return
        if clicked is not anyway:
            return
    BugReportDialog(main_window).exec()


def open_diagnostics(main_window) -> None:
    """Read-only diagnostics window: app/OS, live memory, and recent logs —
    the same data the bug report collects, surfaced without writing a folder.
    Copy-to-clipboard for pasting into an issue or a support chat."""
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QPlainTextEdit

    src = BugReportDialog(main_window)  # built, never shown — used as collector
    text = (
        f"# Aglaïa diagnostics\n\n_{version_tag()}_\n\n"
        f"## System\n{src._md_machine()}\n\n"
        f"## Memory\n{src._md_memory()}\n\n"
        f"## Pipeline\n{src._md_pipeline()}\n\n"
        f"## Logs\n{src._md_logs()}\n"
    )
    src.deleteLater()

    dlg = QDialog(main_window)
    dlg.setWindowTitle(QDialog.tr("Diagnostics"))
    dlg.resize(720, 560)
    lay = QVBoxLayout(dlg)
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setPlainText(text)
    view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    lay.addWidget(view, 1)
    bb = QDialogButtonBox()
    copy_btn = bb.addButton(QDialogButtonBox.tr("Copy"),
                            QDialogButtonBox.ButtonRole.ActionRole)
    bb.addButton(QDialogButtonBox.StandardButton.Close)
    copy_btn.clicked.connect(
        lambda: QGuiApplication.clipboard().setText(text))
    bb.rejected.connect(dlg.reject)
    bb.accepted.connect(dlg.accept)
    lay.addWidget(bb)
    dlg.exec()


class BugReportDialog(QDialog):
    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent or main_window)
        self._mw = main_window
        self.setWindowTitle(self.tr("Report a bug"))
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        v.setSpacing(8)

        v.addWidget(QLabel(self.tr("Include in the report:")))
        self._chk: dict[str, QCheckBox] = {}
        for key, label, default in (
            ("machine", self.tr("Machine + OS info"), True),
            ("backends", self.tr("Enabled backends"), True),
            ("pipeline", self.tr("Current pipeline"), True),
            ("settings", self.tr("Current settings"), True),
            ("memory", self.tr("Memory usage"), True),
            ("stacks", self.tr("Thread stack traces"), False),
            ("logs", self.tr("Console logs"), True),
        ):
            c = QCheckBox(label)
            c.setChecked(default)
            v.addWidget(c)
            self._chk[key] = c

        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Problematic scan:")))
        self._scan_combo = QComboBox()
        self._scan_combo.addItem(self.tr("— none —"), None)
        self._populate_scans()
        row.addWidget(self._scan_combo, 1)
        v.addLayout(row)
        hint = QLabel(self.tr(
            "Selecting a scan attaches its stage images (raw → output)."))
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        gen = QPushButton(self.tr("Generate report in project folder"))
        gen.clicked.connect(self._generate)
        v.addWidget(gen)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    # ── scan list ─────────────────────────────────────────────────────
    def _populate_scans(self) -> None:
        try:
            from lib.storage.db import db_session
            from lib.storage.repo import ScanRepo
            with db_session(str(self._mw.db_path)) as conn:
                for s in ScanRepo(conn).list_active(newest_first=False):
                    self._scan_combo.addItem(f"#{s['idx']}", int(s["id"]))
        except Exception:
            pass

    # ── generate ──────────────────────────────────────────────────────
    def _generate(self) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            project_dir = Path(self._mw.args.workspace_dir)
        except Exception:
            project_dir = Path(self._mw.db_path).parent
        out_dir = project_dir / f"aglaia_debug_report_{ts}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Report failed"),
                                self.tr("Could not create the report folder: "
                                        "{e}").format(e=e))
            return

        scan_id = self._scan_combo.currentData()
        parts: list[str] = [
            "# Aglaïa bug report\n",
            f"_Generated {ts} · {version_tag()}_\n",
            "## Describe the bug\n\n"
            "_<!-- replace this: what happened, what you expected, steps to "
            "reproduce -->_\n",
        ]
        sections = [
            ("machine", self.tr("Machine & OS"), self._md_machine),
            ("backends", self.tr("Backends"), self._md_backends),
            ("pipeline", self.tr("Pipeline"), self._md_pipeline),
            ("settings", self.tr("Settings"), self._md_settings),
            ("memory", self.tr("Memory"), self._md_memory),
            ("stacks", self.tr("Thread stacks"), self._md_stacks),
            ("logs", self.tr("Console logs"), self._md_logs),
        ]
        for key, title, fn in sections:
            if not self._chk[key].isChecked():
                continue
            try:
                body = fn()
            except Exception as e:
                body = f"_collector failed: {e}_"
            parts.append(f"## {title}\n\n{body}\n")

        n_imgs = 0
        if scan_id is not None:
            try:
                n_imgs = self._dump_scan_images(int(scan_id), out_dir)
                parts.append(f"## Scan images\n\n{n_imgs} image(s) attached in "
                             f"this folder (see `images/`).\n")
            except Exception as e:
                parts.append(f"## Scan images\n\n_failed: {e}_\n")

        report = out_dir / "report.md"
        report.write_text("\n".join(parts), encoding="utf-8")
        (out_dir / "HOW_TO_SUBMIT.md").write_text(
            self._submit_text(), encoding="utf-8")

        try:
            from lib.gui.path_reveal import reveal_path
            reveal_path(report)
        except Exception:
            pass
        self._show_submit_procedure(out_dir, n_imgs)

    # ── collectors ────────────────────────────────────────────────────
    def _md_machine(self) -> str:
        lines = [
            f"- App: Aglaïa {APP_VERSION}",
            f"- OS: {platform.platform()}",
            f"- Machine: {platform.machine()}",
            f"- Python: {platform.python_version()} ({sys.platform})",
        ]
        try:
            import psutil
            vm = psutil.virtual_memory()
            lines.append(f"- CPU cores: {psutil.cpu_count(logical=True)}")
            lines.append(f"- RAM: {vm.total / 1e9:.1f} GB "
                         f"({vm.percent:.0f}% used)")
        except Exception:
            pass
        return "\n".join(lines)

    def _md_pipeline(self) -> str:
        import json
        import yaml as _yaml
        yaml_text = None
        # 1. Project DB — the authoritative active pipeline.
        try:
            from lib.storage.db import db_session
            from lib.storage.repo import PipelineRepo
            with db_session(str(self._mw.db_path)) as conn:
                row = PipelineRepo(conn).get_active()
            if row is not None:
                yaml_text = row["yaml_text"]
        except Exception:
            pass
        # 2. The per-project pipeline file.
        if not yaml_text:
            path = getattr(self._mw, "pipeline_yaml_path", None)
            if path:
                try:
                    yaml_text = Path(path).read_text(encoding="utf-8")
                except Exception:
                    pass
        if yaml_text:
            try:
                pdef = _yaml.safe_load(yaml_text)
                return "```json\n" + json.dumps(pdef, indent=2,
                                                default=str) + "\n```"
            except Exception:
                return "```yaml\n" + yaml_text + "\n```"
        names = getattr(self._mw, "pipeline_proc_names", []) or []
        return "Steps: " + " → ".join(names) if names else "_unavailable_"

    def _md_settings(self) -> str:
        try:
            from lib.app_data import db as cfg
            with cfg.session() as conn:
                items = cfg.items(conn)
        except Exception as e:
            return f"_unavailable: {e}_"
        out = []
        for k in sorted(items):
            v = items[k]
            if any(h in k.lower() for h in _SECRET_HINT):
                v = "«redacted»"
            out.append(f"- `{k}` = `{v}`")
        return "\n".join(out)

    def _md_memory(self) -> str:
        # Prefer the same numbers the status bar shows (last [RSS-poll]),
        # which include every worker — psutil from scratch only sees workers
        # that happen to be alive right now (none when the pipeline is idle).
        try:
            vals = dict(self._mw.status_bar_widget.rss.last_values)
        except Exception:
            vals = {}
        if vals:
            gui = vals.get("gui", 0.0)
            lines = [f"- GUI: {gui:.0f} MB"]
            tot = gui
            for k in sorted(k for k in vals if k != "gui"):
                lines.append(f"- {k}: {vals[k]:.0f} MB")
                tot += vals[k]
            lines.append(f"- **Total: {tot:.0f} MB**")
            lines.append("\n_(last sampled values — workers may have exited "
                         "if the pipeline is idle)_")
            return "\n".join(lines)
        # Fallback: live process tree.
        try:
            import os
            import psutil
            p = psutil.Process(os.getpid())
            tot = p.memory_info().rss
            lines = [f"- GUI: {tot / 1e6:.0f} MB"]
            for c in p.children(recursive=True):
                try:
                    rss = c.memory_info().rss
                    tot += rss
                    lines.append(f"- {c.name()} (pid {c.pid}): "
                                 f"{rss / 1e6:.0f} MB")
                except Exception:
                    continue
            lines.append(f"- **Total: {tot / 1e6:.0f} MB**")
            return "\n".join(lines)
        except Exception as e:
            return f"_unavailable: {e}_"

    def _md_stacks(self) -> str:
        out = []
        frames = sys._current_frames()
        for tid, frame in frames.items():
            out.append(f"### Thread {tid}\n```\n"
                       + "".join(traceback.format_stack(frame))
                       + "```")
        return "\n".join(out) if out else "_none_"

    def _md_backends(self) -> str:
        try:
            from lib.workers.Initializer import probe_capabilities
            caps = probe_capabilities()
        except Exception as e:
            return f"_unavailable: {e}_"
        return "\n".join(
            f"- {'✅' if ok else '❌'} **{name}** — {detail}"
            for name, ok, detail in caps) or "_none_"

    def _md_logs(self) -> str:
        chunks = []
        # In-app session log — the real console output the user saw
        # (ProcessMonitor's rolling buffer + Qt-thread worker prints).
        try:
            buf = list(self._mw.monitor_thread.log_buffer)
            if buf:
                chunks.append("### Session log\n```\n"
                              + "\n".join(buf[-500:]) + "\n```")
        except Exception:
            pass
        # Launch trace file.
        try:
            from lib.app_data import log_dir
            p = log_dir() / "aglaia-launch.log"
            if p.exists():
                lines = p.read_text(encoding="utf-8",
                                    errors="replace").splitlines()
                chunks.append("### Launch log\n```\n"
                              + "\n".join(lines[-200:]) + "\n```")
        except Exception:
            pass
        return "\n\n".join(chunks) if chunks else "_no logs captured_"

    def _dump_scan_images(self, scan_id: int, out_dir: Path) -> int:
        """Render the per-processor DEBUG overlays (spans, baselines, quad,
        grid …) for each branch of the scan — the same views the pipeline
        Inspect window shows — and write them to ``images/``. Reuses
        ``render_chain_overlays`` so the report matches the GUI exactly."""
        import base64
        import re
        from lib.storage.db import db_session
        from lib.storage.repo import BranchRepo
        from lib.storage.debug_renderers import render_chain_overlays
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        with db_session(str(self._mw.db_path)) as conn:
            for b in BranchRepo(conn).by_scan(scan_id):
                leaf = b["chosen_node_id"]
                if leaf is None:
                    continue
                try:
                    steps = render_chain_overlays(conn, int(leaf))
                except Exception:
                    continue
                bp = (b["branch_path"] or "").replace(".", "-")
                for i, step in enumerate(steps):
                    url = step.get("url") or ""
                    if not url.startswith("data:image"):
                        continue
                    try:
                        data = base64.b64decode(url.split(",", 1)[1])
                    except Exception:
                        continue
                    label = re.sub(r"[^\w.-]+", "_",
                                   str(step.get("label") or f"step{i}"))
                    pre = f"{bp}_" if bp else ""
                    (img_dir / f"{i:02d}_{pre}{label}.png").write_bytes(data)
                    n += 1
        return n

    # ── submission procedure ──────────────────────────────────────────
    def _submit_text(self) -> str:
        return (
            "# How to submit this report\n\n"
            f"1. **Check it isn't already known** — search {ISSUES_SEARCH_URL}\n"
            "2. **Review `report.md` for any sensitive information** you'd "
            "like to remove before sharing.\n"
            "3. Open / sign in to a GitHub account.\n"
            f"4. Go to **{ISSUES_NEW_URL}** (New issue).\n"
            "5. Paste the contents of `report.md` into the issue body.\n"
            "6. Drag the images from `images/` into the issue (if attached).\n"
            "7. Replace the _Describe the bug_ placeholder with your own "
            "description + any extra comments.\n"
            "8. Submit. Thanks!\n"
        )

    def _show_submit_procedure(self, out_dir: Path, n_imgs: int) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(self.tr("Report generated"))
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(self.tr(
            "Saved to <b>{folder}</b>"
            "{imgs}.<br><br>"
            "To submit:<br>"
            "1. <a href=\"{search}\">Check it isn't already reported</a>.<br>"
            "2. <b>Review report.md</b> and remove anything sensitive.<br>"
            "3. Sign in to GitHub, open <a href=\"{url}\">a new issue</a>.<br>"
            "4. Paste <code>report.md</code> into the body.<br>"
            "5. Drag the images from <code>images/</code> in.<br>"
            "6. Add your description + comments, then submit.<br><br>"
            "<i>See HOW_TO_SUBMIT.md in the folder.</i>"
        ).format(folder=out_dir.name,
                 imgs=(f" · {n_imgs} image(s)" if n_imgs else ""),
                 search=ISSUES_SEARCH_URL, url=ISSUES_NEW_URL))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
