"""Safety net for the server tests: the **paid** Mistral API must never be hit.

The tests stub the heavy operations at a higher level
(`processor.run_pipeline` / `check_batch` / `cancel_batch`), so they never reach
the Mistral network calls. This autouse fixture replaces those calls with
raisers anyway — if a test path ever slips through, it fails loudly instead of
spending money. (If the optional `cloud` extra / `mistralai` isn't installed,
there's nothing to call, so the guard is a no-op.)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def block_mistral_api():
    try:
        import aglaia.workers.ocr.mistral_batch as mb
    except Exception:
        yield
        return

    def _forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Mistral API was called in a test — it is a paid API. "
            "Stub run_pipeline / check_batch / cancel_batch instead."
        )

    names = ("submit", "poll", "fetch_pages", "cancel")
    originals = {n: getattr(mb, n, None) for n in names}
    for n in names:
        if originals[n] is not None:
            setattr(mb, n, _forbidden)
    try:
        yield
    finally:
        for n, fn in originals.items():
            if fn is not None:
                setattr(mb, n, fn)
