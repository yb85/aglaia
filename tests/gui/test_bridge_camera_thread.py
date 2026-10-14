# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""BridgeCameraThread (#49): zoom crop/upscale, transform parity, the full-res
still freshness cache, and graceful degradation on timeout / lost session.

No thread is started — the synchronous methods (get_frame / set_zoom / _zoom_crop)
are exercised directly. Signals are checked via DirectConnection so they fire
inline without an event loop."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from aglaia.gui.BridgeCameraThread import BridgeCameraThread  # noqa: E402
from aglaia.workers.bridge_live import BridgeSessionLost  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class StubServer:
    """Stands in for BridgeLiveServer; counts still requests."""

    def __init__(self, still: np.ndarray | None = None) -> None:
        self._still = still if still is not None else np.full((300, 400, 3), 128, np.uint8)
        self.still_calls = 0
        self.alive = True
        self.raise_on_still: Exception | None = None
        self.device_name = "Stub"
        self.still_dims = (400, 300)

    @property
    def session_alive(self) -> bool:
        return self.alive

    def latest_preview(self):
        return None

    def request_still(self, timeout: float = 6.0) -> np.ndarray:
        self.still_calls += 1
        if self.raise_on_still is not None:
            raise self.raise_on_still
        return self._still.copy()


def _thread(qapp, **kw) -> BridgeCameraThread:
    return BridgeCameraThread(StubServer(**kw))


def test_zoom_crop_keeps_dims_and_magnifies_center(qapp) -> None:
    t = _thread(qapp)
    ramp = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))  # 0→255 L→R
    ramp = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    t.set_zoom(2.0)
    out = t._zoom_crop(ramp, cv2.INTER_LINEAR)
    assert out.shape == ramp.shape                      # dims unchanged
    # 2× crops to the center columns (~64..191) then stretches → the extreme
    # ends (0, 255) are gone.
    assert out[:, 0, 0].mean() > 30
    assert out[:, -1, 0].mean() < 225


def test_zoom_identity_at_1x(qapp) -> None:
    t = _thread(qapp)
    img = np.random.randint(0, 255, (40, 60, 3), np.uint8)
    assert np.array_equal(t._zoom_crop(img, cv2.INTER_LINEAR), img)


def test_set_zoom_clamps_to_max(qapp) -> None:
    t = _thread(qapp)
    assert t.set_zoom(0.2) == 1.0
    assert t.set_zoom(99.0) == t.max_zoom == 2.0 + 1.0  # BRIDGE_MAX_ZOOM == 3.0


def test_get_frame_applies_transform(qapp) -> None:
    server = StubServer(still=np.zeros((300, 400, 3), np.uint8))
    t = BridgeCameraThread(server)
    t.set_transform("rotate 90")
    frame = t.get_frame()
    assert frame is not None
    assert frame.shape[:2] == (400, 300)  # 90° rotation flips H/W


def test_get_frame_freshness_cache(qapp) -> None:
    server = StubServer()
    t = BridgeCameraThread(server)
    t.get_frame()
    t.get_frame()  # within STILL_CACHE_SECONDS → served from cache
    assert server.still_calls == 1
    t._still_at = 0.0  # force the cache stale
    t.get_frame()
    assert server.still_calls == 2


def test_get_frame_timeout_signals_and_returns_none(qapp) -> None:
    server = StubServer()
    server.raise_on_still = TimeoutError("slow phone")
    t = BridgeCameraThread(server)
    fired = []
    t.still_failed.connect(lambda: fired.append(True), Qt.ConnectionType.DirectConnection)
    assert t.get_frame() is None
    assert fired == [True]


def test_get_frame_lost_session_signals(qapp) -> None:
    server = StubServer()
    server.raise_on_still = BridgeSessionLost("gone")
    t = BridgeCameraThread(server)
    reasons = []
    t.session_lost.connect(reasons.append, Qt.ConnectionType.DirectConnection)
    assert t.get_frame() is None
    assert len(reasons) == 1
