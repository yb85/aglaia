# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The Mistral SDK surface this code depends on.

A dependency upgrade to `mistralai` 2.x broke cloud OCR at runtime with

    ImportError: cannot import name 'Mistral' from 'mistralai' (unknown location)

The 2.x SDK is a restructure, not an API bump: the top-level `mistralai`
became a NAMESPACE package (no `__init__.py`, hence "unknown location") and
`Mistral` moved to `mistralai.client`. Nothing in the suite noticed, because
cloud OCR needs a paid API key and is never exercised in CI.

These tests need no key and no network — they check only that the SDK is
shaped the way the code calls it. That is enough to turn a silent runtime
failure into a red test.
"""
import importlib.util

import pytest

#: The SDK-shape tests need the `cloud` extra; the PIN test deliberately does
#: not — CI does not install that extra, so the pin guard is the only one of
#: these that runs there, and it is the one that catches a lock upgrade.
needs_sdk = pytest.mark.skipif(
    importlib.util.find_spec("mistralai") is None,
    reason="cloud extra not installed")

#: Every attribute path `mistral_cloud.py` / `mistral_batch.py` reach for.
USED_PATHS = [
    "files.upload",
    "files.get_signed_url",
    "files.download",
    "ocr.process",
    "batch.jobs.create",
    "batch.jobs.get",
    "batch.jobs.cancel",
    "batch.jobs.list",
]


@needs_sdk
def test_mistralai_is_a_real_package_not_a_namespace():
    """The 2.x namespace layout is exactly what produced the runtime
    ImportError, and it is invisible to `import mistralai` alone."""
    spec = importlib.util.find_spec("mistralai")
    assert spec is not None and spec.origin, (
        "mistralai resolved as a namespace package (no __init__.py) — this is "
        "the 2.x layout, in which `from mistralai import Mistral` fails")


@needs_sdk
def test_mistral_is_importable_from_the_top_level():
    from mistralai import Mistral
    assert Mistral is not None


@needs_sdk
@pytest.mark.parametrize("path", USED_PATHS)
def test_the_client_exposes_every_call_site_we_use(path):
    """No key, no network — attribute lookup only."""
    from mistralai import Mistral
    obj = Mistral(api_key="not-a-real-key")
    for part in path.split("."):
        assert hasattr(obj, part), f"client.{path} is missing (broke at {part!r})"
        obj = getattr(obj, part)


def test_the_pin_that_keeps_this_working_is_still_in_place():
    """The ceiling is load-bearing, not tidiness — without it a routine
    `uv lock --upgrade` silently re-breaks cloud OCR."""
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = tomllib.loads((root / "pyproject.toml").read_text())
    reqs = [r for group in
            [spec["project"]["dependencies"],
             *spec["project"].get("optional-dependencies", {}).values()]
            for r in group if r.startswith("mistralai")]
    assert reqs, "mistralai requirement vanished"
    assert any("<2" in r for r in reqs), (
        f"the mistralai <2 ceiling is gone ({reqs}) — 2.x moves `Mistral` to "
        f"mistralai.client and breaks cloud OCR at runtime")
