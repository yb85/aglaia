# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The import scan skips a plugin's tests.

CONTRIBUTING asks every submission to ship `tests/test_<slug>.py`. A test
necessarily imports `pytest`, `sys` and whatever it exercises — so scanning
them refused every plugin that followed the instructions, including the
first-party one, at the moment it grew a test suite.

Tests are also never imported by the app: only the manifest's entry module and
its private siblings are. This scan exists to say what the RUNNING code can
reach.
"""
from pathlib import Path

from aglaia.app_data.plugin_manifest import scan_plugin_dir

ENTRY = "from aglaia.plugin_api import to_gray\n"
TEST = "import importlib\nimport sys\nimport pytest\n"


def _plugin(tmp_path, with_tests=True):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "demo.py").write_text(ENTRY, encoding="utf-8")
    if with_tests:
        (d / "tests").mkdir()
        (d / "tests" / "test_demo.py").write_text(TEST, encoding="utf-8")
    return d


def test_a_plugin_with_tests_still_scans_clean(tmp_path):
    assert scan_plugin_dir(_plugin(tmp_path)).refused == []


def test_the_same_imports_in_the_ENTRY_module_are_still_refused(tmp_path):
    """The exclusion is about where the code lives, not about the words in
    it — a plugin cannot smuggle `sys` in by naming its module `test_`… it
    can, and that is why the entry module is decided by the manifest and
    scanned regardless."""
    d = _plugin(tmp_path, with_tests=False)
    (d / "demo.py").write_text("import sys\n" + ENTRY, encoding="utf-8")
    assert "sys" in scan_plugin_dir(d).refused


def test_a_plugin_that_is_only_tests_has_nothing_to_scan(tmp_path):
    d = tmp_path / "demo"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_demo.py").write_text(TEST, encoding="utf-8")
    assert scan_plugin_dir(d).error
