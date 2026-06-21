# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import pickle

import numpy as np

from lib.ImageBuffer import ImageBuffer, ImageType


def test_pickle_strips_parent_and_children():
    parent = ImageBuffer(
        np.zeros((4000, 3000, 3), dtype=np.uint8), ImageType.COLOR,
        dpi=300, filestem="page_001",
    )
    child = ImageBuffer(
        np.zeros((200, 300), dtype=np.uint8), ImageType.GRAY,
        dpi=300, parent=parent, filestem="page_001_layout0",
    )
    parent.children.append(child)

    blob = pickle.dumps(child, protocol=pickle.HIGHEST_PROTOCOL)
    # Child crop is 60 kB; parent frame is 36 MB. Without __getstate__
    # the parent (and its children list, recursively) ride along.
    assert len(blob) < 1_000_000

    restored = pickle.loads(blob)
    assert restored.parent is None
    assert restored.children == []
    assert restored.parent_stem == "page_001"
    assert restored.filestem == "page_001_layout0"
    # Original object untouched by pickling.
    assert child.parent is parent
    assert parent.children == [child]
