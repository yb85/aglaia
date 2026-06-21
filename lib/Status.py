# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

from enum import IntEnum

class Status(IntEnum):
    PENDING = 0
    SUCCESS = 1
    WARNING = 2
    ERROR = 3
    REVIEW = 4

STATUS_COLORS = {
    Status.PENDING: "gray",
    Status.SUCCESS: "green",
    Status.WARNING: "yellow",
    Status.ERROR: "red",
    Status.REVIEW: "purple"
}
