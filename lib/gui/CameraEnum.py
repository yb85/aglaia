# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""macOS camera enumeration via AVFoundation.

Returns a list of (cv2_index, friendly_name) tuples in the same order that
`cv2.VideoCapture(index)` would expose them on macOS.
"""

from typing import List, Tuple


def _discovery_devices():
    """The AVCaptureDevice list in the same order cv2's AVFoundation
    backend indexes them. Empty list off-macOS / on failure."""
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    device_types = [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    for opt in ("AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeContinuityCamera"):
        if hasattr(AV, opt):
            device_types.append(getattr(AV, opt))
    session = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types, AV.AVMediaTypeVideo, AV.AVCaptureDevicePositionUnspecified,
    )
    return list(session.devices())


def list_cameras() -> List[Tuple[int, str]]:
    cameras: list[tuple[int, str]] = []
    for i, dev in enumerate(_discovery_devices()):
        try:
            cameras.append((i, str(dev.localizedName())))
        except Exception:
            continue
    if not cameras:
        cameras = [(0, "Default camera")]
    return cameras


def list_camera_formats(camera_id: int) -> List[dict]:
    """List a device's landscape video formats, widest field-of-view first.

    Each entry: ``{index, width, height, fov, max_zoom, label}`` where
    ``index`` is the position in ``device.formats()`` (what WebcamThread
    selects on), ``fov`` is the horizontal field of view in degrees, and
    ``label`` is a UI-ready string. Returns ``[]`` off-macOS / on failure.

    Continuity Cameras (iPhone) advertise several formats with different
    crops/zoom; the widest-FOV one matches what Photo Booth shows.
    """
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    devs = _discovery_devices()
    if not (0 <= camera_id < len(devs)):
        return []
    out: list[dict] = []
    for i, fmt in enumerate(devs[camera_id].formats()):
        try:
            dim = AV.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
            fw, fh = int(dim.width), int(dim.height)
            # Keep only landscape formats at least as wide as 4:3 — this
            # drops square / portrait still-modes (e.g. 1552×1552) that are
            # centre crops of the sensor, not the full wide view.
            if fw <= 0 or fh <= 0 or fw < fh * 4 // 3:
                continue
            try:
                fov = float(fmt.videoFieldOfView())   # often 0 on UVC / Continuity
            except Exception:
                fov = 0.0
            try:
                mz = float(fmt.videoMaxZoomFactor())
            except Exception:
                mz = 1.0
            ar = _aspect_label(fw, fh)
            label = f"{fw}×{fh}  ({ar})"
            if fov > 0:
                label += f"  ·  {fov:.0f}° FOV"
            out.append({"index": i, "width": fw, "height": fh,
                        "fov": fov, "max_zoom": mz, "label": label})
        except Exception:
            continue
    # FOV is unreliable (frequently reported as 0), so rank by sensor
    # coverage: widest FOV when known, then largest area (the full-sensor
    # format, not a crop).
    out.sort(key=lambda f: (f["fov"], f["width"] * f["height"]), reverse=True)
    return out


def _aspect_label(w: int, h: int) -> str:
    from math import gcd
    g = gcd(w, h) or 1
    return f"{w // g}:{h // g}"
