# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Capability probes for the Apple OCR engines — UI gating lives here.

The OCR tab disables / tooltips the two Apple cards based on three states:

* **not macOS** → both Apple cards disabled ("macOS only").
* **macOS, pre-26** (no ``VNRecognizeDocumentsRequest``) → only the
  *document* card disabled ("Requires macOS 26+"); the legacy Vision card
  stays usable.
* **macOS 26+** → both enabled.

There is deliberately **no Apple-Intelligence gate**: the document
request runs without Apple Intelligence (verified on macOS 26.5.1).
"""

from __future__ import annotations

import sys
from typing import NamedTuple


class AppleCaps(NamedTuple):
    is_macos: bool
    has_vision: bool          # pyobjc Vision import OK
    has_documents: bool       # VNRecognizeDocumentsRequest present (macOS 26)


def probe_apple_caps() -> AppleCaps:
    if sys.platform != "darwin":
        return AppleCaps(False, False, False)
    try:
        import Vision  # noqa: WPS433
    except Exception:
        return AppleCaps(True, False, False)
    has_docs = bool(hasattr(Vision, "VNRecognizeDocumentsRequest"))
    return AppleCaps(True, True, has_docs)


# Tooltip strings — kept here so the UI and any headless caller agree.
TOOLTIP_NON_MACOS = "macOS only"
TOOLTIP_NEEDS_26 = "Requires macOS 26+"
