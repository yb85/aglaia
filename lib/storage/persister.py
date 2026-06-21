# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import io
import sqlite3
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from lib.storage.repo import ImageRepo, ThumbRepo, NodeRepo, BranchRepo


def encode_image(buffer: np.ndarray, image_type: str) -> tuple[bytes, str, int, int]:
    """Encode numpy image as JPG (COLOR/GRAY) or PNG (BW). Returns (bytes, fmt, width, height).

    No `optimize=True`: the extra Huffman / PNG-filter trials cost
    hundreds of ms per 12 MP frame on the hot persist path for a few
    percent of blob size."""
    if buffer is None:
        raise ValueError("encode_image: buffer is None")
    h, w = buffer.shape[:2]
    if image_type == "BW":
        # 1-bit PNG via PIL
        if buffer.ndim == 3:
            buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2GRAY)
        pil = Image.fromarray(buffer).convert("1")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue(), "PNG", w, h
    # JPG for COLOR/GRAY
    if image_type == "GRAY" and buffer.ndim == 3:
        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2GRAY)
    if image_type == "COLOR" and buffer.ndim == 2:
        buffer = cv2.cvtColor(buffer, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(buffer)
    if image_type == "GRAY":
        pil = pil.convert("L")
    else:
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), "JPG", w, h


def make_thumb(blob: bytes, max_dim: int) -> tuple[bytes, int, int]:
    pil = Image.open(io.BytesIO(blob))
    pil.thumbnail((max_dim, max_dim))
    w, h = pil.size
    out = io.BytesIO()
    if pil.mode in ("1", "L"):
        pil = pil.convert("L")
    else:
        pil = pil.convert("RGB")
    pil.save(out, format="JPEG", quality=80)
    return out.getvalue(), w, h


class Persister:
    """
    Inserts an ImageBuffer-equivalent payload (bytes + metadata) into the DB
    as image + thumb + node + branches upsert.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.images = ImageRepo(conn)
        self.thumbs = ThumbRepo(conn)
        self.nodes = NodeRepo(conn)
        self.branches = BranchRepo(conn)

    def persist_image(self, buffer: np.ndarray, image_type: str, dpi: float) -> int:
        # Thumbnails are NOT generated here: the pipeline never reads
        # them, and the GUI's ThumbLoader lazily builds + caches a thumb
        # on first request. Eager generation cost a blob re-decode + two
        # resamples + two JPEG encodes per step.
        blob, fmt, w, h = encode_image(buffer, image_type)
        return self.images.insert(blob, fmt, image_type, w, h, dpi)

    def persist_node(self, *, scan_id: int, parent_id: Optional[int], pipeline_version_id: int,
                     step_idx: int, step_name: Optional[str], processor_name: Optional[str],
                     branch_label: Optional[str], depth: int, filestem: str,
                     image_id: Optional[int], status_int: int = 1, elapsed_ms: Optional[float] = None,
                     meta: Optional[dict] = None, is_branch_point: bool = False) -> int:
        return self.nodes.insert(
            scan_id=scan_id, parent_id=parent_id, pipeline_version_id=pipeline_version_id,
            step_idx=step_idx, step_name=step_name, processor_name=processor_name,
            branch_label=branch_label, depth=depth, filestem=filestem, image_id=image_id,
            status_int=status_int, elapsed_ms=elapsed_ms, meta=meta,
            is_branch_point=is_branch_point,
        )

    def upsert_branch(self, scan_id: int, branch_path: str, terminal_node_id: int) -> int:
        return self.branches.upsert(scan_id, branch_path, terminal_node_id)
