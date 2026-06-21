# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import io
import numpy as np
from PIL import Image

from lib.storage.repo import (
    ProjectRepo, PipelineRepo, CalibrationRepo, ImageRepo, ThumbRepo,
    ScanRepo, NodeRepo,
)


def _png_bytes(size=8, color=0):
    img = Image.new("L", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_project_singleton(db):
    repo = ProjectRepo(db)
    assert repo.get() is None
    repo.init("My Book", "my-book")
    p = repo.get()
    assert p["name"] == "My Book"
    repo.init("Other", "other-slug")
    # still id=1 row
    rows = db.execute("SELECT COUNT(*) AS c FROM project").fetchone()
    assert rows["c"] == 1
    assert repo.get()["slug"] == "other-slug"


def test_pipeline_upsert_dedup(db):
    repo = PipelineRepo(db)
    a = repo.upsert("yaml-a\n", "a", step_count=1)
    b = repo.upsert("yaml-a\n", "a", step_count=1)
    assert a == b
    c = repo.upsert("yaml-b\n", "b", step_count=2)
    assert c != a
    assert repo.get_active()["id"] == c  # last upsert with make_active=True wins


def test_calibration_active(db):
    repo = CalibrationRepo(db)
    a = repo.insert([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0, 0, 0]], dpi=300.0,
                    resolution=(720, 1280), sample_count=10)
    b = repo.insert([[2, 0, 0], [0, 2, 0], [0, 0, 1]], [[0, 0, 0, 0, 0]], dpi=305.0,
                    resolution=(720, 1280), sample_count=10)
    assert repo.get_active()["id"] == b
    repo.set_active(a)
    assert repo.get_active()["id"] == a


def test_image_dedup(db):
    repo = ImageRepo(db)
    blob = _png_bytes()
    a = repo.insert(blob, "PNG", "BW", 8, 8, 72.0)
    b = repo.insert(blob, "PNG", "BW", 8, 8, 72.0)
    assert a == b
    assert repo.get(a)["sha256"]


def test_snap_create_is_atomic_under_concurrent_writers(tmp_path):
    """
    Two connections to the same SQLite file create scans in parallel.
    Atomic-INSERT idx assignment must produce distinct idx values without
    raising UNIQUE constraint violations.
    """
    import threading
    from lib.storage.db import open_db
    from lib.storage.repo import ProjectRepo, PipelineRepo, ScanRepo

    db_path = tmp_path / "race.sqlite"
    main = open_db(db_path)
    ProjectRepo(main).init("race", "race")
    pid = PipelineRepo(main).upsert("name: r\npipeline: []\n", "r", step_count=0)
    main.close()

    N_WORKERS = 8
    N_PER_WORKER = 25
    errors: list[str] = []
    ids: list[int] = []
    lock = threading.Lock()

    def worker(worker_id):
        try:
            conn = open_db(db_path)
            s = ScanRepo(conn)
            local = []
            for _ in range(N_PER_WORKER):
                sid = s.create("import", pid, source_ref=f"w{worker_id}.jpg")
                local.append(sid)
            conn.close()
            with lock:
                ids.extend(local)
        except Exception as e:
            with lock:
                errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"writer errors: {errors[:3]}"
    assert len(ids) == N_WORKERS * N_PER_WORKER

    # Every scan must have a distinct idx.
    main = open_db(db_path)
    rows = main.execute("SELECT idx FROM scans").fetchall()
    main.close()
    idxs = [r["idx"] for r in rows]
    assert len(idxs) == len(set(idxs)), \
        f"duplicate idx: {len(idxs)} rows, {len(set(idxs))} unique"
    assert min(idxs) == 1 and max(idxs) == len(idxs), \
        f"idx range not contiguous: {min(idxs)}..{max(idxs)} for {len(idxs)} rows"


def test_snap_next_idx_and_root(seeded_db):
    db, pid = seeded_db
    img_id = ImageRepo(db).insert(_png_bytes(), "PNG", "BW", 8, 8, 72.0)
    s = ScanRepo(db)
    assert s.next_idx() == 1
    sid = s.create("capture", pid, transform="0", capture_dpi=72.0)
    assert s.next_idx() == 2
    nid = NodeRepo(db).insert(scan_id=sid, parent_id=None, pipeline_version_id=pid,
                              step_idx=0, step_name=None, processor_name=None,
                              branch_label=None, depth=0, filestem="t_001", image_id=img_id)
    s.set_root(sid, nid)
    assert s.get(sid)["root_node_id"] == nid


def test_node_subtree_and_delete(seeded_db):
    db, pid = seeded_db
    iid = ImageRepo(db).insert(_png_bytes(), "PNG", "BW", 8, 8, 72.0)
    sid = ScanRepo(db).create("import", pid)
    n = NodeRepo(db)

    def add(parent, step_idx, depth, stem, label=None):
        return n.insert(scan_id=sid, parent_id=parent, pipeline_version_id=pid,
                        step_idx=step_idx, step_name=f"{step_idx:02d}_s",
                        processor_name="X", branch_label=label, depth=depth,
                        filestem=stem, image_id=iid)

    root = add(None, 0, 0, "t_001")
    s1 = add(root, 1, 1, "t_001a")
    s2 = add(s1, 2, 2, "t_001b")
    # tree: root -> s1 -> s2
    tree = n.subtree(root)
    assert [r["id"] for r in tree] == [root, s1, s2]
    # delete from s1 downward (include_self)
    n.delete_subtree(s1, include_self=True)
    assert n.get(s1) is None
    assert n.get(s2) is None
    assert n.get(root) is not None
