# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Hand-offs from a worker thread to the GUI thread (#204).

`apply_pipeline_callback` rebuilt the chain on a plain `threading.Thread`
and gave the window the new queue with `QTimer.singleShot(0, _swap)`. With
no context object, that timer lives on the calling thread, which has no Qt
event loop, so `_swap` never ran: the window kept the stopped chain's queue
and every capture after a pipeline Apply was lost.
"""
from __future__ import annotations

import ast
import os
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

APP_PY = Path(__file__).resolve().parents[2] / "aglaia" / "app.py"


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _fire_from_thread(qapp, *args) -> list[str]:
    hit: list[str] = []

    def cb():
        hit.append(threading.current_thread().name)

    t = threading.Thread(target=lambda: QTimer.singleShot(0, *args, cb))
    t.start()
    t.join()
    for _ in range(50):
        qapp.processEvents()
        if hit:
            break
        threading.Event().wait(0.01)
    return hit


def test_context_singleshot_runs_on_the_gui_thread(qapp):
    ctx = QObject()
    assert _fire_from_thread(qapp, ctx) == [threading.main_thread().name]


def test_contextless_singleshot_from_a_thread_is_lost(qapp):
    """The Qt behaviour the bug came from. If a Qt release changes it this
    test fails, and the guard below can be reconsidered."""
    assert _fire_from_thread(qapp) == []


def _thread_targets(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Thread":
            for kw in node.keywords:
                if kw.arg == "target" and isinstance(kw.value, ast.Name):
                    names.add(kw.value.id)
    return names


def test_app_thread_targets_pass_a_singleshot_context():
    """Every function app.py runs on a `threading.Thread` must give
    `QTimer.singleShot` a context object (3 positional args)."""
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    targets = _thread_targets(tree)
    assert {"_worker", "_catchup"} <= targets
    bad = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in targets):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "singleShot"
                    and len(node.args) < 3):
                bad.append(f"{fn.name}:{node.lineno}")
    assert not bad, f"context-less singleShot on a worker thread: {bad}"
