# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Remember a camera's rotation / mirror / flip between projects.

A capture rig does not move between books: the camera is clamped where it is,
pointing the way it points, and the correction that makes its feed upright is a
property of the rig, not of the project. Re-applying it by hand at every new
project was three clicks nobody should have to remember.

Keyed by the camera's own name (AVFoundation `localizedName`) rather than its
cv2 index — indexes shift when a device is plugged in or removed, names do
not. Stored in the app-data config DB, not the project.
"""

from __future__ import annotations

from typing import Optional

from aglaia.app_data import db as cfg


def camera_key(cam_id: int) -> str:
    """A stable identity for the device at `cam_id`."""
    try:
        from aglaia.gui.WebcamThread import _camera_label
        name = _camera_label(int(cam_id))
        if name and name != "?":
            return name
    except Exception:
        pass
    return f"index:{int(cam_id)}"


def encode(rotation: int, mirror: bool, flip: bool) -> str:
    """The `--transform` string `WebcamThread.set_transform` parses.
    270° is written `-90`, which is how that parser spells it."""
    rot = {0: "0", 90: "90", 180: "180", 270: "-90"}.get(int(rotation) % 360, "0")
    parts = [rot] + (["mirror"] if mirror else []) + (["flip"] if flip else [])
    return "+".join(parts)


def load(cam_key: str) -> Optional[str]:
    """The remembered transform string for this camera, or None."""
    try:
        with cfg.session() as conn:
            table = cfg.get(conn, cfg.KEY_CAMERA_TRANSFORMS, {}) or {}
    except Exception:
        return None
    v = table.get(cam_key) if isinstance(table, dict) else None
    return str(v) if v else None


def save(cam_key: str, rotation: int, mirror: bool, flip: bool) -> None:
    value = encode(rotation, mirror, flip)
    try:
        with cfg.session() as conn:
            table = cfg.get(conn, cfg.KEY_CAMERA_TRANSFORMS, {}) or {}
            if not isinstance(table, dict):
                table = {}
            if value == "0":
                table.pop(cam_key, None)      # the identity is the default; do not store it
            else:
                table[cam_key] = value
            cfg.set(conn, cfg.KEY_CAMERA_TRANSFORMS, table)
            conn.commit()
    except Exception as e:  # noqa: BLE001 — a failed remember must not stop capture
        print(f"[camera] could not remember the transform: {e}")
