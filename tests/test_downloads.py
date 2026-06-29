# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Central download registry: registration, on-disk presence, and the
config-DB lifecycle state reconciled against disk (no Qt)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    """Isolate APP_DATA (config DB + models dir) per test and hand back a fresh
    `downloads` module with the core targets registered."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as _ad

    importlib.reload(_ad)
    from aglaia.app_data import db as _db

    importlib.reload(_db)
    from aglaia.app_data import downloads as _dl

    importlib.reload(_dl)  # re-runs _register_core_targets() against the temp dir
    return _dl


def _materialise(dl, key):
    """Create the on-disk files a target expects, at their recorded sizes."""
    from aglaia.app_data import models_dir

    t = dl.target_for(key)
    dest = models_dir() / t.filename
    if t.kind == "hf-snapshot":
        dest.mkdir(parents=True, exist_ok=True)
        for rel, size in t.required_files or (("blob.bin", 4096),):
            with open(dest / rel, "wb") as fh:
                fh.truncate(max(size, 1))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.truncate(t.approx_size_mb * 1024 * 1024 or 4096)
    return t


def test_core_targets_registered(reg):
    keys = {t.key for t in reg.registry()}
    assert {"vosk_en", "east", "surya", "paddle_vl", "dbnet"} <= keys
    paddle = reg.target_for("paddle_vl")
    assert paddle.platform == "darwin-arm64"
    assert len(paddle.required_files) == 4


def test_shim_aliases_resolve(reg):
    from aglaia.app_data import models as M

    importlib.reload(M)
    assert M.ModelSpec is reg.DownloadTarget
    assert {s.key for s in M._load_model_specs()} == {t.key for t in reg.registry()}
    assert M.is_model_installed("paddle_vl") is False


def test_plugin_can_register(reg):
    reg.register_download(
        reg.DownloadTarget(
            key="glm_ocr_mlx",
            title="GLM-OCR (MLX)",
            filename="GLM-OCR-mlx",
            url="mlx-community/GLM-OCR-8bit",
            approx_size_mb=900,
            kind="hf-snapshot",
            purpose="OCR",
            platform="darwin-arm64",
            registered_by="GlmOcr",
            required_files=(("model.safetensors", 4096),),
        )
    )
    assert reg.target_for("glm_ocr_mlx").registered_by == "GlmOcr"
    assert reg.is_downloaded("glm_ocr_mlx") is False


def test_not_downloaded_initially(reg):
    assert reg.is_downloaded("surya") is False
    assert reg.download_status("surya") == reg.STATUS_NONE


def test_presence_marks_downloaded_in_db(reg):
    _materialise(reg, "paddle_vl")
    assert reg.is_downloaded("paddle_vl") is True
    # download_status records the reconciled 'downloaded' row.
    assert reg.download_status("paddle_vl") == reg.STATUS_DOWNLOADED
    from aglaia.app_data import db

    with db.session() as conn:
        assert db.get_download_status(conn, "paddle_vl") == "downloaded"


def test_deleting_files_reconciles_status_down(reg):
    t = _materialise(reg, "paddle_vl")
    assert reg.download_status("paddle_vl") == reg.STATUS_DOWNLOADED  # row written
    from aglaia.app_data import models_dir

    (models_dir() / t.filename / t.required_files[0][0]).unlink()
    # Stale 'downloaded' row must be cleared and the verdict drop.
    assert reg.is_downloaded("paddle_vl") is False
    assert reg.download_status("paddle_vl") == reg.STATUS_NONE
    from aglaia.app_data import db

    with db.session() as conn:
        assert db.get_download_status(conn, "paddle_vl") is None


def test_failed_status_persists_without_disk(reg):
    reg.record_status("surya", reg.STATUS_FAILED)
    assert reg.download_status("surya") == reg.STATUS_FAILED


def test_truncated_snapshot_is_not_downloaded(reg):
    # A required file present but well under its recorded size = incomplete pull.
    from aglaia.app_data import models_dir

    t = reg.target_for("surya")
    dest = models_dir() / t.filename
    dest.mkdir(parents=True, exist_ok=True)
    for rel, _size in t.required_files:
        (dest / rel).write_bytes(b"truncated")
    assert reg.is_downloaded("surya") is False
