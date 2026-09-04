# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The editable handle layer of the debug view (M9 #97).

`ZoomCanvas` shows a picture. The per-processor renderers burn their rich
diagnostics — span masks, baselines, the fitted grid — into that picture,
which is right for reading and useless for grabbing. `EditCanvas` adds a
VECTOR layer on top, fed by the `geom` those renderers now return beside the
raster (#96), and hands drags back as edited values.

Two shapes are enough for the three tunable stages:

* a **polygon** whose vertices drag — the layout ROI;
* a **rotation handle** — a bar through the image centre whose free end
  drags, for the deskew angle.

Both are drawn in SOURCE-image coordinates and mapped through the canvas's own
fit rect, so they stay registered with the picture at any window size. The
dewarp needs no handle: its three curl coefficients are sliders.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from aglaia.gui.ZoomCanvas import ZoomCanvas

#: Grab radius around a vertex, in view pixels. Generous: the vertices of a
#: text-tight hull sit close together, and a miss that pans the PiP instead of
#: moving the point reads as the handle being broken.
GRAB_PX = 12

_POLY_LINE = QColor(60, 220, 60)
_POLY_VERTEX = QColor(255, 255, 255)
_POLY_ACTIVE = QColor(255, 190, 60)
_ROT_LINE = QColor(255, 190, 60)


class EditCanvas(ZoomCanvas):
    """A `ZoomCanvas` that can also hand back an edited overlay.

    `edited` fires on every drag step, not only on release: with
    auto-process off the user wants to see the handle follow the cursor, and
    with it on the host is the one that decides to debounce a rerun.
    """

    #: ``(kind, value)`` — ``("roi", [[x, y], …])`` or ``("skew_deg", float)``.
    edited = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None, **kw):
        super().__init__(parent, **kw)
        self._poly: Optional[list[list[float]]] = None
        self._rot_deg: Optional[float] = None
        # Where the stage frame sits inside the displayed composite (the
        # renderer's `geom["origin"]`): the label bar above it, the crop
        # offset of a child drawn on its parent.
        self._origin: tuple[float, float] = (0.0, 0.0)
        self._frame_wh: Optional[tuple[int, int]] = None
        self._drag_vertex: Optional[int] = None
        self._drag_rot = False

    # ── public API ────────────────────────────────────────────────
    def set_editable(self, *, polygon=None, rotation_deg=None,
                     origin=(0, 0), frame_wh=None) -> None:
        """Install (or clear, with both ``None``) the editable overlay."""
        self._origin = (float(origin[0]), float(origin[1]))
        self._frame_wh = (tuple(int(v) for v in frame_wh)
                          if frame_wh else None)
        self._poly = ([[float(x), float(y)] for x, y in polygon]
                      if polygon else None)
        self._rot_deg = (float(rotation_deg) if rotation_deg is not None
                         else None)
        self._drag_vertex = None
        self._drag_rot = False
        self.update()

    def polygon(self) -> Optional[list[list[float]]]:
        return None if self._poly is None else [list(p) for p in self._poly]

    def rotation_deg(self) -> Optional[float]:
        return self._rot_deg

    def set_rotation_deg(self, deg: float) -> None:
        """Move the handle without emitting — for a slider driving the same
        value, which would otherwise echo back and fight the user."""
        self._rot_deg = float(deg)
        self.update()

    def is_editing(self) -> bool:
        return self._drag_vertex is not None or self._drag_rot

    def _frame(self) -> tuple[int, int]:
        """Size of the STAGE frame the geometry lives in. Falls back to the
        displayed pixmap when a renderer sent no `frame_wh`."""
        if self._frame_wh:
            return self._frame_wh
        if self._pix is not None:
            return self._pix.width(), self._pix.height()
        return (1, 1)

    # ── coordinate mapping ────────────────────────────────────────
    def _to_view(self, x: float, y: float) -> QPointF:
        fit = self._fit_rect()
        if self._pix is None or fit.isEmpty():
            return QPointF(0, 0)
        cx = x + self._origin[0]
        cy = y + self._origin[1]
        return QPointF(fit.x() + cx * fit.width() / self._pix.width(),
                       fit.y() + cy * fit.height() / self._pix.height())

    def _to_src(self, pos: QPoint) -> Optional[tuple[float, float]]:
        fit = self._fit_rect()
        if self._pix is None or fit.isEmpty():
            return None
        return ((pos.x() - fit.x()) * self._pix.width() / fit.width()
                - self._origin[0],
                (pos.y() - fit.y()) * self._pix.height() / fit.height()
                - self._origin[1])

    def _rot_end(self) -> Optional[QPointF]:
        """Free end of the rotation bar, in view coordinates.

        The bar runs through the image centre at the current angle, its arm
        two thirds of the half-width — long enough that a degree of rotation
        is a visible arc, short enough to stay inside a portrait page."""
        if self._pix is None or self._rot_deg is None:
            return None
        fw, fh = self._frame()
        cx, cy = fw / 2.0, fh / 2.0
        arm = fw / 3.0
        a = math.radians(self._rot_deg)
        return self._to_view(cx + arm * math.cos(a), cy + arm * math.sin(a))

    # ── events ────────────────────────────────────────────────────
    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if ev.button() == Qt.MouseButton.LeftButton and self._pix is not None:
            end = self._rot_end()
            if end is not None and (end - QPointF(pos)).manhattanLength() <= GRAB_PX * 2:
                self._drag_rot = True
                return
            if self._poly:
                for i, (x, y) in enumerate(self._poly):
                    if (self._to_view(x, y) - QPointF(pos)).manhattanLength() <= GRAB_PX * 2:
                        self._drag_vertex = i
                        self.update()
                        return
        # Not on a handle — let the base class do its pin/unpin PiP thing.
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if self._drag_rot:
            src = self._to_src(pos)
            if src is not None and self._pix is not None:
                fw, fh = self._frame()
                cx, cy = fw / 2.0, fh / 2.0
                self._rot_deg = math.degrees(math.atan2(src[1] - cy,
                                                        src[0] - cx))
                self.edited.emit("skew_deg", self._rot_deg)
                self.update()
            return
        if self._drag_vertex is not None and self._poly:
            src = self._to_src(pos)
            if src is not None and self._pix is not None:
                # Clamp into the image: the ROI is intersected with the crop
                # downstream anyway, and a vertex dragged off-canvas would
                # vanish under the cursor.
                fw, fh = self._frame()
                self._poly[self._drag_vertex] = [
                    max(0.0, min(float(fw - 1), src[0])),
                    max(0.0, min(float(fh - 1), src[1]))]
                self.edited.emit("roi", self.polygon())
                self.update()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self._drag_rot or self._drag_vertex is not None:
            self._drag_rot = False
            self._drag_vertex = None
            self.update()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Double-click an edge to insert a vertex there.

        A hull that follows the text usually needs a point ADDED, not moved —
        the detector's polygon is convex and the page rarely is."""
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if (ev.button() != Qt.MouseButton.LeftButton or not self._poly
                or self._pix is None):
            super().mouseDoubleClickEvent(ev)
            return
        n = len(self._poly)
        best_i, best_d, best_pt = None, float(GRAB_PX * 2), None
        target = QPointF(pos)
        for i in range(n):
            a = self._to_view(*self._poly[i])
            b = self._to_view(*self._poly[(i + 1) % n])
            ab = b - a
            L2 = ab.x() ** 2 + ab.y() ** 2
            if L2 <= 1e-9:
                continue
            t = max(0.0, min(1.0, (QPointF.dotProduct(target - a, ab)) / L2))
            foot = a + ab * t
            d = (foot - target).manhattanLength()
            if d < best_d:
                best_i, best_d, best_pt = i, d, foot
        if best_i is None or best_pt is None:
            super().mouseDoubleClickEvent(ev)
            return
        src = self._to_src(best_pt.toPoint())
        if src is None:
            return
        self._poly.insert(best_i + 1, [src[0], src[1]])
        self.edited.emit("roi", self.polygon())
        self.update()

    # ── paint ─────────────────────────────────────────────────────
    def paintEvent(self, ev) -> None:  # noqa: N802
        super().paintEvent(ev)
        if self._pix is None or (self._poly is None and self._rot_deg is None):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._poly:
            poly = QPolygonF([self._to_view(x, y) for x, y in self._poly])
            p.setPen(QPen(_POLY_LINE, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(poly)
            for i, (x, y) in enumerate(self._poly):
                c = _POLY_ACTIVE if i == self._drag_vertex else _POLY_VERTEX
                p.setPen(QPen(_POLY_LINE, 2))
                p.setBrush(c)
                p.drawEllipse(self._to_view(x, y), 5, 5)
        if self._rot_deg is not None:
            end = self._rot_end()
            if end is not None:
                fw, fh = self._frame()
                centre = self._to_view(fw / 2.0, fh / 2.0)
                p.setPen(QPen(_ROT_LINE, 2, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(centre, end)
                p.setPen(QPen(_ROT_LINE, 2))
                p.setBrush(_ROT_LINE if self._drag_rot else _POLY_VERTEX)
                p.drawEllipse(end, 7, 7)
        p.end()
