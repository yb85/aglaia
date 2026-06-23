# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Force-rerun must trash the images it regenerates, not keep the old ones.

`reprocess_active_scans` wipes a scan's pipeline subtree. The `images` table is
content-addressed (`ImageRepo.insert` dedups on sha256), so unless the wipe also
prunes the now-orphaned blobs, a re-run silently re-adopts a stale image row and
"regenerate" never actually regenerates the pixels. This pins the GC.
"""
import io
import numpy as np
from PIL import Image

from aglaia.storage.db import open_db
from aglaia.storage.repo import (
    ProjectRepo, PipelineRepo, ImageRepo, ThumbRepo, ScanRepo, NodeRepo,
)
from aglaia.workers.ImportHelpers import reprocess_active_scans


def _png(color: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeChain:
    """Captures enqueued buffers; the GC runs before enqueue, so we don't
    need to process anything."""
    def __init__(self):
        self.enqueued = []

    def enqueue(self, buf):
        self.enqueued.append(buf)


def test_reprocess_prunes_orphaned_step_images(tmp_path):
    path = tmp_path / "proj.sqlite"
    conn = open_db(path)
    ProjectRepo(conn).init("T", "t")
    pid = PipelineRepo(conn).upsert("name: s\npipeline: []\n", "s", step_count=1)
    images, thumbs, scans, nodes = (
        ImageRepo(conn), ThumbRepo(conn), ScanRepo(conn), NodeRepo(conn))

    raw_id = images.insert(_png(0), "PNG", "COLOR", 8, 8, 300.0)
    step_id = images.insert(_png(128), "PNG", "COLOR", 8, 8, 300.0)   # intermediate
    thumbs.upsert(step_id, 256, 8, 8, _png(128))

    sid = scans.create("import", pid, source_ref="x.jpg")
    root = nodes.insert(scan_id=sid, parent_id=None, pipeline_version_id=pid,
                        step_idx=0, step_name=None, processor_name=None,
                        branch_label=None, depth=0, filestem="x", image_id=raw_id)
    scans.set_root(sid, root)
    child = nodes.insert(scan_id=sid, parent_id=root, pipeline_version_id=pid,
                         step_idx=1, step_name="01_x", processor_name="P",
                         branch_label="A", depth=1, filestem="x_A", image_id=step_id)
    conn.commit()
    conn.close()

    n = reprocess_active_scans(db_path=str(path), pipeline_version_id=pid,
                               chain=_FakeChain())
    assert n == 1

    conn = open_db(path)
    try:
        # The intermediate image + its thumb are gone; the raw root image stays.
        assert conn.execute("SELECT 1 FROM images WHERE id=?", (step_id,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM thumbs WHERE image_id=?", (step_id,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM images WHERE id=?", (raw_id,)).fetchone() is not None
    finally:
        conn.close()


def test_reprocess_keeps_image_shared_with_another_scan(tmp_path):
    """A content-addressed blob referenced by a second scan's surviving node
    must NOT be pruned when the first scan is wiped."""
    path = tmp_path / "proj.sqlite"
    conn = open_db(path)
    ProjectRepo(conn).init("T", "t")
    pid = PipelineRepo(conn).upsert("name: s\npipeline: []\n", "s", step_count=1)
    images, scans, nodes = ImageRepo(conn), ScanRepo(conn), NodeRepo(conn)

    raw1 = images.insert(_png(0), "PNG", "COLOR", 8, 8, 300.0)
    raw2 = images.insert(_png(10), "PNG", "COLOR", 8, 8, 300.0)
    shared = images.insert(_png(200), "PNG", "COLOR", 8, 8, 300.0)  # same content both scans

    s1 = scans.create("import", pid, source_ref="a.jpg")
    r1 = nodes.insert(scan_id=s1, parent_id=None, pipeline_version_id=pid, step_idx=0,
                      step_name=None, processor_name=None, branch_label=None, depth=0,
                      filestem="a", image_id=raw1)
    scans.set_root(s1, r1)
    nodes.insert(scan_id=s1, parent_id=r1, pipeline_version_id=pid, step_idx=1,
                 step_name="01_x", processor_name="P", branch_label="A", depth=1,
                 filestem="a_A", image_id=shared)

    s2 = scans.create("import", pid, source_ref="b.jpg")
    r2 = nodes.insert(scan_id=s2, parent_id=None, pipeline_version_id=pid, step_idx=0,
                      step_name=None, processor_name=None, branch_label=None, depth=0,
                      filestem="b", image_id=raw2)
    scans.set_root(s2, r2)
    nodes.insert(scan_id=s2, parent_id=r2, pipeline_version_id=pid, step_idx=1,
                 step_name="01_x", processor_name="P", branch_label="A", depth=1,
                 filestem="b_A", image_id=shared)  # scan 2 still references it
    conn.commit()
    conn.close()

    # Wipe ONLY scan 1.
    reprocess_active_scans(db_path=str(path), pipeline_version_id=pid,
                           chain=_FakeChain(), scan_ids={s1})

    conn = open_db(path)
    try:
        assert conn.execute("SELECT 1 FROM images WHERE id=?", (shared,)).fetchone() is not None
    finally:
        conn.close()
