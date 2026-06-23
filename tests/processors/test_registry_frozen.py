# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Guard the frozen-build processor discovery fallback.

In a PyInstaller bundle the `aglaia/processors` directory does not exist on disk,
so `pkgutil.iter_modules` finds nothing and the registry falls back to a
hard-coded list of built-in module names (`_BUILTIN_PROCESSOR_MODULES`). If a
new built-in processor is added but not listed there, the bundled app would
silently skip it ("Unknown processor … Skipping") while the source build works
fine. This test fails loudly in that case."""

from aglaia.processors import registry


def test_hardcoded_builtin_list_matches_discovery():
    import os
    import sys

    import aglaia.processors as _pkg

    registry.all_processors()  # force discovery
    hardcoded = set(registry._BUILTIN_PROCESSOR_MODULES)
    pkg_dir = _pkg.__path__[0]

    # Module (short) names of every processor whose source lives directly in
    # aglaia/processors/ — i.e. the built-ins (plugins live under APP_DATA and are
    # excluded). The frozen app can only find these via the hard-coded list.
    builtin_module_names = set()
    for info in registry.all_processors().values():
        mod = sys.modules.get(info.processor_cls.__module__)
        mod_file = getattr(mod, "__file__", None)
        if mod_file and os.path.dirname(os.path.abspath(mod_file)) == pkg_dir:
            builtin_module_names.add(info.processor_cls.__module__.rsplit(".", 1)[-1])

    # Every hard-coded name must be a real built-in module…
    assert hardcoded <= builtin_module_names, (
        f"_BUILTIN_PROCESSOR_MODULES has stale entries: {hardcoded - builtin_module_names}")
    # …and every built-in module must be hard-coded, or the frozen app skips it.
    missing = builtin_module_names - hardcoded
    assert not missing, (
        f"built-in processors missing from _BUILTIN_PROCESSOR_MODULES "
        f"(frozen app will skip them): {missing}")
