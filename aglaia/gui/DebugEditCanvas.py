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

* a **polygon** whose vertices drag — the layout ROI, and the keystone
  column quad (which refuses insertion: four points, no more);
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
#: The live dewarp preview. Deliberately NOT the renderer's green: that grid
#: is what was fitted, this one is what the sliders are asking for.
_PREVIEW_LINE = QColor(255, 90, 200)
#: Per-layout outline colours, cycled — they have to be told apart, and the
#: composite underneath already tints each crop from its own list.
_LAYOUT_LINES = (QColor(60, 220, 60), QColor(80, 190, 255),
                 QColor(255, 190, 60), QColor(240, 120, 220))
_BADGE_BG = QColor(20, 20, 24, 150)
_BADGE_FG = QColor(255, 255, 255, 230)
#: Radius of the barycentre trash badge and the top-right add badge, in view
#: pixels. Big enough to hit without magnifying, small enough not to cover
#: the text it sits on.
BADGE_R = 15


class EditCanvas(ZoomCanvas):
    """A `ZoomCanvas` that can also hand back an edited overlay.

    `edited` fires on every drag step, not only on release, so the handle
    follows the cursor. It is NOT a commit signal: persisting and rerunning
    per step meant hundreds of chain reruns stacked up inside one drag, which
    exhausted memory and killed the app (#116). `edit_finished` is the commit
    — emitted once on release, and immediately for an atomic edit like a
    double-click vertex insert.
    """

    #: ``(kind, value)`` — ``("roi", [[x, y], …])`` or ``("skew_deg", float)``.
    #: Fires on every drag STEP so the handle tracks the cursor.
    edited = Signal(str, object)
    #: The drag ended. This is the one to persist and rerun on: `edited`
    #: alone had the host writing SQLite and launching a chain rerun per
    #: mouse-move event, hundreds deep into a single drag, until memory ran
    #: out and the app died (#116).
    edit_finished = Signal()
    #: Layout SET changed by a badge press — ``("delete", index)`` or
    #: ``("add", None)``. Separate from `edited` because the host answers it
    #: by rewriting the whole set, not by moving a point (#118).
    layout_action = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None, **kw):
        super().__init__(parent, **kw)
        self._poly: Optional[list[list[float]]] = None
        self._rot_deg: Optional[float] = None
        # Where the stage frame sits inside the displayed composite (the
        # renderer's `geom["origin"]`): the label bar above it, the crop
        # offset of a child drawn on its parent.
        self._origin: tuple[float, float] = (0.0, 0.0)
        self._frame_wh: Optional[tuple[int, int]] = None
        # What the renderer shrank the composite by on its way to a data URL.
        # `origin` and the geometry are in FULL-resolution composite pixels;
        # the picture on screen is not. Miss this and the error grows with the
        # coordinate — the handles walk away down and right.
        self._scale: float = 1.0
        # A keystone is a projective map from FOUR points; a fifth would have
        # no meaning, so its polygon refuses insertion (M9 #100).
        self._allow_insert = True
        # Read-only polylines in stage coordinates, drawn through the same
        # mapping as the handles — the dewarp's live grid preview.
        self._preview: list = []
        self._drag_vertex: Optional[int] = None
        self._drag_rot = False
        # Layout-set mode (#118): several polygons at once, each with a
        # delete badge, plus one add badge. `_poly` stays the single-shape
        # path (the keystone quad, and PageDetector rows on a node with no
        # parent to draw on).
        self._layouts: list = []
        self._layout_labels: list = []
        self._drag_layout: Optional[int] = None

    # ── public API ────────────────────────────────────────────────
    def set_editable(self, *, polygon=None, rotation_deg=None,
                     origin=(0, 0), frame_wh=None, scale=1.0,
                     allow_insert=True) -> None:
        """Install (or clear, with both ``None``) the editable overlay."""
        self._allow_insert = bool(allow_insert)
        self._origin = (float(origin[0]), float(origin[1]))
        self._scale = float(scale) or 1.0
        self._frame_wh = (tuple(int(v) for v in frame_wh)
                          if frame_wh else None)
        self._poly = ([[float(x), float(y)] for x, y in polygon]
                      if polygon else None)
        self._rot_deg = (float(rotation_deg) if rotation_deg is not None
                         else None)
        self._layouts = []
        self._layout_labels = []
        self._drag_layout = None
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

    def set_layouts(self, polygons, *, labels=None, origin=(0, 0),
                    frame_wh=None, scale=1.0) -> None:
        """Install the editable layout SET — several polygons in one frame.

        The single-polygon path clamps a dragged vertex into the frame it was
        given, and for a PageDetector row that frame was ONE child's crop: the
        orange box in the debug view. So a vertex could not be moved to where
        the page actually is, only to where the detector had already decided
        it was. Here the frame is the PARENT, every layout is reachable, and
        the set itself can grow and shrink (#118)."""
        self._origin = (float(origin[0]), float(origin[1]))
        self._scale = float(scale) or 1.0
        self._frame_wh = (tuple(int(v) for v in frame_wh)
                          if frame_wh else None)
        self._layouts = [[[float(x), float(y)] for x, y in poly]
                         for poly in (polygons or [])]
        self._layout_labels = [str(v) for v in (labels or [])]
        self._poly = None
        self._rot_deg = None
        self._drag_vertex = None
        self._drag_layout = None
        self._drag_rot = False
        self.update()

    def layouts(self) -> list:
        return [[list(pt) for pt in poly] for poly in self._layouts]

    def _add_badge_centre(self) -> Optional[QPointF]:
        """Top-right corner of the PICTURE, inset by a badge radius."""
        fit = self._fit_rect()
        if fit.isEmpty() or not self._layouts and self._frame_wh is None:
            return None
        return QPointF(fit.right() - BADGE_R - 6, fit.top() + BADGE_R + 6)

    @staticmethod
    def _barycentre(poly) -> tuple[float, float]:
        n = len(poly) or 1
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)

    def _trash_badge_centre(self, i: int) -> Optional[QPointF]:
        if not (0 <= i < len(self._layouts)):
            return None
        cx, cy = self._barycentre(self._layouts[i])
        return self._to_view(cx, cy)

    def _badge_hit(self, pos) -> Optional[tuple]:
        """``("add", None)`` / ``("delete", i)`` under `pos`, else None."""
        target = QPointF(pos)
        add = self._add_badge_centre()
        if add is not None and (add - target).manhattanLength() <= BADGE_R * 2:
            return ("add", None)
        # A single layout keeps no trash badge: deleting the last one would
        # leave the page with nothing to process.
        if len(self._layouts) > 1:
            for i in range(len(self._layouts)):
                c = self._trash_badge_centre(i)
                if c is not None and (c - target).manhattanLength() <= BADGE_R * 2:
                    return ("delete", i)
        return None

    def set_preview(self, lines) -> None:
        """Install read-only polylines (stage coords), or clear with None.

        The dewarp has no handle — its shape is three numbers — so the only
        way to know what a slider did was to reprocess and look. This draws
        the sheet the sliders describe, live, before anything is rerun."""
        self._preview = [list(pl) for pl in (lines or [])]
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
        cx = (x + self._origin[0]) * self._scale
        cy = (y + self._origin[1]) * self._scale
        return QPointF(fit.x() + cx * fit.width() / self._pix.width(),
                       fit.y() + cy * fit.height() / self._pix.height())

    def _to_src(self, pos: QPoint) -> Optional[tuple[float, float]]:
        fit = self._fit_rect()
        if self._pix is None or fit.isEmpty():
            return None
        sc = self._scale or 1.0
        return ((pos.x() - fit.x()) * self._pix.width() / fit.width() / sc
                - self._origin[0],
                (pos.y() - fit.y()) * self._pix.height() / fit.height() / sc
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
            if self._layouts:
                hit = self._badge_hit(pos)
                if hit is not None:
                    # A badge press is the whole gesture — never also the
                    # start of a drag on whatever sits under it.
                    self.layout_action.emit(hit[0], hit[1])
                    return
                for li, poly in enumerate(self._layouts):
                    for i, (x, y) in enumerate(poly):
                        if (self._to_view(x, y)
                                - QPointF(pos)).manhattanLength() <= GRAB_PX * 2:
                            self._drag_layout = li
                            self._drag_vertex = i
                            self.update()
                            return
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
        if self._drag_vertex is not None and self._drag_layout is not None:
            src = self._to_src(pos)
            if src is not None and self._pix is not None:
                fw, fh = self._frame()
                poly = self._layouts[self._drag_layout]
                poly[self._drag_vertex] = [
                    max(0.0, min(float(fw - 1), src[0])),
                    max(0.0, min(float(fh - 1), src[1]))]
                self.edited.emit("layouts", self.layouts())
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
            self._drag_layout = None
            self.update()
            # AFTER clearing the flags: the host checks `is_editing()` to
            # tell a drag step from a commit.
            self.edit_finished.emit()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Double-click an edge to insert a vertex there.

        A hull that follows the text usually needs a point ADDED, not moved —
        the detector's polygon is convex and the page rarely is."""
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if (ev.button() == Qt.MouseButton.LeftButton and self._layouts
                and self._pix is not None):
            if self._insert_into_layouts(pos):
                return
            super().mouseDoubleClickEvent(ev)
            return
        if (ev.button() != Qt.MouseButton.LeftButton or not self._poly
                or self._pix is None or not self._allow_insert):
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
        # Atomic — there is no drag to wait for, so it commits at once.
        self.edit_finished.emit()
        self.update()

    def _insert_into_layouts(self, pos) -> bool:
        """Add a vertex on the nearest edge of the nearest layout."""
        target = QPointF(pos)
        best = None                      # (dist, layout, edge, foot)
        for li, poly in enumerate(self._layouts):
            n = len(poly)
            for i in range(n):
                a = self._to_view(*poly[i])
                b = self._to_view(*poly[(i + 1) % n])
                ab = b - a
                L2 = ab.x() ** 2 + ab.y() ** 2
                if L2 <= 1e-9:
                    continue
                t = max(0.0, min(1.0,
                                 QPointF.dotProduct(target - a, ab) / L2))
                foot = a + ab * t
                d = (foot - target).manhattanLength()
                if d <= GRAB_PX * 2 and (best is None or d < best[0]):
                    best = (d, li, i, foot)
        if best is None:
            return False
        _d, li, edge, foot = best
        src = self._to_src(foot.toPoint())
        if src is None:
            return False
        self._layouts[li].insert(edge + 1, [src[0], src[1]])
        self.edited.emit("layouts", self.layouts())
        self.edit_finished.emit()
        self.update()
        return True

    def _draw_badge(self, p: QPainter, centre: QPointF, glyph: str) -> None:
        """Semi-transparent disc + glyph. Deliberately translucent: it sits on
        top of the page the user is reading to decide whether to keep it."""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_BADGE_BG)
        p.drawEllipse(centre, BADGE_R, BADGE_R)
        pen = QPen(_BADGE_FG, 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        x, y, r = centre.x(), centre.y(), BADGE_R
        if glyph == "+":
            p.drawLine(QPointF(x - r * 0.45, y), QPointF(x + r * 0.45, y))
            p.drawLine(QPointF(x, y - r * 0.45), QPointF(x, y + r * 0.45))
            return
        # Trash: lid, can, two ribs.
        w, h = r * 0.46, r * 0.52
        p.drawLine(QPointF(x - w, y - h * 0.6), QPointF(x + w, y - h * 0.6))
        p.drawLine(QPointF(x - w * 0.45, y - h * 0.6),
                   QPointF(x - w * 0.45, y - h))
        p.drawLine(QPointF(x + w * 0.45, y - h * 0.6),
                   QPointF(x + w * 0.45, y - h))
        p.drawLine(QPointF(x - w * 0.75, y - h * 0.6),
                   QPointF(x - w * 0.6, y + h))
        p.drawLine(QPointF(x + w * 0.75, y - h * 0.6),
                   QPointF(x + w * 0.6, y + h))
        p.drawLine(QPointF(x - w * 0.6, y + h), QPointF(x + w * 0.6, y + h))
        p.drawLine(QPointF(x, y - h * 0.25), QPointF(x, y + h * 0.6))

    # ── paint ─────────────────────────────────────────────────────
    def paintEvent(self, ev) -> None:  # noqa: N802
        super().paintEvent(ev)
        if self._pix is None or (self._poly is None and self._rot_deg is None
                                 and not self._preview and not self._layouts):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._preview:
            p.setPen(QPen(_PREVIEW_LINE, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for line in self._preview:
                pts = QPolygonF([self._to_view(x, y) for x, y in line])
                p.drawPolyline(pts)
        for li, lpoly in enumerate(self._layouts):
            col = _LAYOUT_LINES[li % len(_LAYOUT_LINES)]
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(QPolygonF([self._to_view(x, y) for x, y in lpoly]))
            for i, (x, y) in enumerate(lpoly):
                active = (li == self._drag_layout and i == self._drag_vertex)
                p.setPen(QPen(col, 2))
                p.setBrush(_POLY_ACTIVE if active else _POLY_VERTEX)
                p.drawEllipse(self._to_view(x, y), 5, 5)
            if len(self._layouts) > 1:
                c = self._trash_badge_centre(li)
                if c is not None:
                    self._draw_badge(p, c, "trash")
        if self._layouts:
            add = self._add_badge_centre()
            if add is not None:
                self._draw_badge(p, add, "+")
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
