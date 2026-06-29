# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Aglaïa server — a warm-pool HTTP job API (`aglaia server`, see issue #52).

Slice 1 (this module set): the scaffold — DB (`api_keys`/`jobs`/`config`),
API-key + admin auth, file intake, and the `/run` `/list` `/check` `/get`
`/delete` `/admin` endpoints. Job *processing* (chain + OCR + export), Mistral
batch polling with exponential backoff, and completion email land in later
slices.
"""

DEFAULT_PORT = 4674
