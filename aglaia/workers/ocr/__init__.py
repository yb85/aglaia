# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""OCR engines for Aglaïa.

Each engine implements `OcrEngine.recognize(image_rgb, languages) -> OcrResult`.
The result dict is what lands in `ocr_runs.result_json`.
"""

from .engine import OcrEngine, OcrResult, OcrLine, ENGINE_REGISTRY, get_engine
# Side-effect imports register each engine in ENGINE_REGISTRY. Keep last
# so the registry is populated before any caller pulls from it.
from . import apple_vision as _apple_vision  # noqa: F401
from . import apple_docs as _apple_docs       # noqa: F401
from . import surya as _surya                 # noqa: F401
from . import paddle_vl as _paddle_vl         # noqa: F401
from . import glm_ocr as _glm_ocr             # noqa: F401
from . import unlimited_ocr as _unlimited_ocr  # noqa: F401
from . import mistral_cloud as _mistral_cloud  # noqa: F401

# User drop-in OCR plugins (<APP_DATA>/plugins/ocr). Each plugin module
# decorates its engine with @register, which populates ENGINE_REGISTRY as
# a side effect of import — so importing the accepted, sha-matching files
# is all that's needed. Only acknowledged files are imported (the trust
# gate ran at startup); unacknowledged code never executes here.
try:
    from aglaia.app_data import plugins as _plugins
    _plugins.import_accepted(_plugins.KIND_OCR)
except Exception as _e:  # noqa: BLE001 — plugin layer optional / best-effort
    print(f"[ocr] plugin discovery skipped: {_e}")

__all__ = ["OcrEngine", "OcrResult", "OcrLine", "ENGINE_REGISTRY", "get_engine"]
