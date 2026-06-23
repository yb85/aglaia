# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Single chain-element dataclass shared by the active chain.

The earlier ABC + two flavours of element + dead StaticProcessingChain
shipped scaffolding for a second backend that never materialised. The
SimpleChainElement alias is kept (most callsites still import it) and
forwards to ChainElement.
"""

from dataclasses import dataclass
from typing import Optional

from aglaia.processors.abstraction import AbstractProcessorOption


@dataclass
class ChainElement:
    """Configuration for one pipeline step."""

    processor_name: str
    options: AbstractProcessorOption
    instance_name: Optional[str] = None  # If set, output writes to this subfolder.


# Back-compat: existing callsites import SimpleChainElement; keep the alias.
SimpleChainElement = ChainElement
