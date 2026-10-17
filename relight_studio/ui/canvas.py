"""Central canvas: image display, zoom/pan, light gizmos, brushes and before/after.

Light positions are stored in normalized image coordinates, so gizmos stay
attached to the same scene point under any zoom, pan or window size.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QRadialGradient,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..core.color import rgb_to_hex
from ..core.lights import Light, LightType
from . import theme
from .controller import EditorController

GIZMO_R = 13.0
HIT_R = 17.0


class Canvas(QWidget):
    zoomChanged = Signal(float)
    toolChanged = Signal(str)
    cursorMoved = Signal(object)  # (u, v) or None

    def __init__(self, controller: EditorController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctl = controller
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self.result: QImage | None = None
        self.original: QImage | None = None
        self.mask_overlay: QImage | None = None
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.fit_mode = True
        self.tool = "select"
        self.compare_split = False
        self.split_pos = 0.5
        self.hold_original = False
        self.show_gizmos = True
        self.busy_text = ""

        self._drag: str | None = None  # light | pan | split | brush
        self._drag_light: int | None = None
        self._drag_offset = QPointF()
        self._last_pos = QPointF()
        self._space = False
        self._mouse = QPointF(-1000, -1000)

    # ---------------------------------------------------------------- geometry
    def image_size(self) -> tuple[int, int]:
        doc = self.ctl.doc
        return (doc.width, doc.height) if doc else (1, 1)

    def image_rect(self) -> QRectF:
        w, h = self.image_size()
        return QRectF(self.pan.x(), self.pan.y(), w * self.zoom, h * self.zoom)

    def to_norm(self, pos: QPointF) -> tuple[float, float]:
        r = self.image_rect()
        return (pos.x() - r.x()) / r.width(), (pos.y() - r.y()) / r.height()

    def to_widget(self, u: float, v: float) -> QPointF:
        r = self.image_rect()
        return QPointF(r.x() + u * r.width(), r.y() + v * r.height())

    def fit(self) -> None:
        if not self.ctl.doc:
            return
        w, h = self.image_size()
        margin = 28
        z = min((self.width() - 2 * margin) / w, (self.height() - 2 * margin) / h)
        self.zoom = max(z, 0.01)
        self.pan = QPointF((self.width() - w * self.zoom) / 2, (self.height() - h * self.zoom) / 2)
        self.fit_mode = True
        self.zoomChanged.emit(self.zoom)
        self.update()

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        zoom = min(max(zoom, 0.02), 16.0)
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        u, v = self.to_norm(anchor)
        self.zoom = zoom
        w, h = self.image_size()
        self.pan = QPointF(anchor.x() - u * w * zoom, anchor.y() - v * h * zoom)
        self.fit_mode = False
        self.zoomChanged.emit(self.zoom)
        self.update()

    def zoom_step(self, factor: float) -> None:
        self.set_zoom(self.zoom * factor)

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self.fit_mode:
            self.fit()
        super().resizeEvent(event)

    # ----------------------------------------------------------------- content
    def set_result(self, image: QImage) -> None:
        self.result = image
        self.update()

    def set_original(self, image: QImage | None) -> None:
        self.original = image
        self.result = None
        self.update()

    def set_mask_overlay(self, mask: np.ndarray | None) -> None:
        if mask is None:
            self.mask_overlay = None
        else:
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = 70
            rgba[..., 2] = 90
            rgba[..., 3] = ((255 - mask.astype(np.uint16)) * 150 // 255).astype(np.uint8)
            self.mask_overlay = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self.update()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._update_cursor()
        self.toolChanged.emit(tool)
        self.update()

    def _update_cursor(self) -> None:
        if self._space or self.tool == "pan":
            self.setCursor(Qt.CursorShape.ClosedHandCursor if self._drag == "pan" else Qt.CursorShape.OpenHandCursor)
        elif self.tool in ("mask", "depth"):
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif self.tool.startswith("place:"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.tool == "zoom":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # -------------------------------------------------------------------- paint
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        doc = self.ctl.doc
        if doc is None or self.original is None:
            self._paint_empty(p)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2.0)
        rect = self.image_rect()
        # Soft drop shadow behind the photo.
        p.fillRect(rect.adjusted(-1, -1, 1, 1), QColor(0, 0, 0, 120))
        shown = self.original if (self.hold_original or self.result is None) else self.result
        p.drawImage(rect, shown)
        if self.compare_split and self.result is not None and not self.hold_original:
            split_x = rect.x() + rect.width() * self.split_pos
            p.save()
            p.setClipRect(QRectF(rect.x(), rect.y(), split_x - rect.x(), rect.height()))
            p.drawImage(rect, self.original)
            p.restore()
            self._paint_split(p, split_x, rect)
        if self.mask_overlay is not None and doc.mask_settings.show_overlay:
            p.drawImage(rect, self.mask_overlay)
        if self.hold_original:
            self._badge(p, "ORIGINAL")
        if self.show_gizmos and self.ctl.view in ("result", "lighting", "depth", "normals") and not self.hold_original:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            for light in doc.lights:
                self._paint_gizmo(p, light, light.id == self.ctl.selected)
        if self.tool in ("mask", "depth") and self.rect().contains(self._mouse.toPoint()):
            self._paint_brush(p)
        if self.busy_text:
            self._badge(p, self.busy_text, bottom=True)

    def _paint_empty(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2 - 20)
        grad = QRadialGradient(c, 120)
        grad.setColorAt(0, QColor(255, 154, 60, 40))
        grad.setColorAt(1, QColor(255, 154, 60, 0))
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(c, 120, 120)
        p.setPen(QColor(theme.TEXT))
        f = QFont(self.font())
        f.setPointSizeF(13)
        p.setFont(f)
        p.drawText(QRectF(0, c.y() - 10, self.width(), 30), Qt.AlignmentFlag.AlignCenter, "Open a photograph to start relighting")
        f.setPointSizeF(9)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(0, c.y() + 22, self.width(), 24), Qt.AlignmentFlag.AlignCenter,
                   "Ctrl+O  ·  or drop a JPEG, PNG, WebP or TIFF here")

    def _paint_split(self, p: QPainter, x: float, rect: QRectF) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        cy = rect.center().y()
        p.setBrush(QColor(20, 20, 22, 230))
        p.drawEllipse(QPointF(x, cy), 13, 13)
        p.setPen(QPen(QColor(255, 255, 255), 1.6))
        for s in (-1, 1):
            p.drawPolyline([QPointF(x + s * 3, cy - 4), QPointF(x + s * 7, cy), QPointF(x + s * 3, cy + 4)])
        self._label(p, QPointF(rect.left() + 10, rect.top() + 10), "BEFORE")
        self._label(p, QPointF(rect.right() - 58, rect.top() + 10), "AFTER")

    def _label(self, p: QPainter, pos: QPointF, text: str) -> None:
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        f.setBold(True)
        p.setFont(f)
        r = QRectF(pos.x(), pos.y(), 50, 18)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _badge(self, p: QPainter, text: str, bottom: bool = False) -> None:
        f = QFont(self.font())
        f.setPointSizeF(8)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 24
        y = self.height() - 40 if bottom else 14
        r = QRectF((self.width() - w) / 2, y, w, 24)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(24, 24, 27, 230))
        p.drawRoundedRect(r, 12, 12)
        p.setPen(QColor(theme.TEXT))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _gizmo_scale(self, light: Light) -> float:
        # Lights closer to the camera (negative z) are drawn slightly larger.
        return float(np.clip(1.0 - 0.35 * light.z, 0.65, 1.5))

    def _paint_gizmo(self, p: QPainter, light: Light, selected: bool) -> None:
        c = self.to_widget(light.x, light.y)
        id_color = QColor(light.gizmo_color)
        fill = QColor(rgb_to_hex(light.color))
        r = GIZMO_R * self._gizmo_scale(light)
        alpha = 255 if light.enabled else 90

        if selected and light.type in (LightType.POINT, LightType.AREA, LightType.RIM):
            rr = light.radius * self.image_rect().height()
            pen = QPen(QColor(id_color.red(), id_color.green(), id_color.blue(), 110), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rr, rr)
        if light.type is LightType.DIRECTIONAL:
            center = self.to_widget(0.5, 0.5)
            direction = QPointF(center.x() - c.x(), center.y() - c.y())
            length = math.hypot(direction.x(), direction.y())
            if length > r * 2:
                d = QPointF(direction.x() / length, direction.y() / length)
                tip = QPointF(c.x() + d.x() * min(length - 4, 90), c.y() + d.y() * min(length - 4, 90))
                p.setPen(QPen(QColor(id_color.red(), id_color.green(), id_color.blue(), alpha), 1.6))
                p.drawLine(QPointF(c.x() + d.x() * r, c.y() + d.y() * r), tip)
                n = QPointF(-d.y(), d.x())
                p.drawPolyline([tip - d * 7 + n * 4, tip, tip - d * 7 - n * 4])

        # Glow
        glow = QRadialGradient(c, r * 2.4)
        gc = QColor(fill)
        gc.setAlpha(110 if light.enabled else 30)
        glow.setColorAt(0, gc)
        gc.setAlpha(0)
        glow.setColorAt(1, gc)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, r * 2.4, r * 2.4)

        # Body
        p.setBrush(QColor(18, 18, 20, 200))
        ring = QColor(id_color)
        ring.setAlpha(alpha)
        p.setPen(QPen(ring, 2.2 if selected else 1.6))
        if light.type is LightType.AREA:
            p.drawRoundedRect(QRectF(c.x() - r, c.y() - r * 0.7, 2 * r, 1.4 * r), 4, 4)
        else:
            p.drawEllipse(c, r, r)
        core = QColor(fill)
        core.setAlpha(alpha)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(core)
        if light.type is LightType.RIM:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(core, 3))
            p.drawArc(QRectF(c.x() - r * 0.55, c.y() - r * 0.55, r * 1.1, r * 1.1), 30 * 16, 300 * 16)
        elif light.type is LightType.AMBIENT:
            p.drawEllipse(c, r * 0.55, r * 0.55)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(core, 1))
            p.drawEllipse(c, r * 0.8, r * 0.8)
        else:
            p.drawEllipse(c, r * 0.45, r * 0.45)
        if selected:
            p.setPen(QPen(QColor(255, 255, 255, 230), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r + 4, r + 4)

        # Id tag
        f = QFont(self.font())
        f.setPointSizeF(7)
        f.setBold(True)
        p.setFont(f)
        tag = QRectF(c.x() + r * 0.55, c.y() - r * 1.35, 15, 13)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ring)
        p.drawRoundedRect(tag, 4, 4)
        p.setPen(QColor(15, 15, 15))
        p.drawText(tag, Qt.AlignmentFlag.AlignCenter, str(light.id))

        # Depth indicator: a small bar showing z from near (top) to far (bottom).
        if light.is_positional and selected:
            bar = QRectF(c.x() - r - 11, c.y() - r, 4, 2 * r)
            p.setBrush(QColor(255, 255, 255, 40))
            p.drawRoundedRect(bar, 2, 2)
            t = float(np.clip((light.z + 1.0) / 2.5, 0, 1))
            p.setBrush(ring)
            p.drawEllipse(QPointF(bar.center().x(), bar.top() + t * bar.height()), 3, 3)

    def _paint_brush(self, p: QPainter) -> None:
        rad = self.ctl.brush.size * self.image_rect().height()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 160), 2.5))
        p.drawEllipse(self._mouse, rad, rad)
        p.setPen(QPen(QColor(255, 255, 255, 220), 1))
        p.drawEllipse(self._mouse, rad, rad)
        inner = rad * self.ctl.brush.hardness
        if inner > 2:
            p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DotLine))
            p.drawEllipse(self._mouse, inner, inner)
        p.drawPoint(self._mouse)

    # -------------------------------------------------------------- hit testing
    def light_at(self, pos: QPointF) -> Light | None:
        doc = self.ctl.doc
        if not doc:
            return None
        best, best_d = None, HIT_R
        for light in reversed(doc.lights):
            c = self.to_widget(light.x, light.y)
            d = math.hypot(c.x() - pos.x(), c.y() - pos.y())
            if d <= best_d * self._gizmo_scale(light):
                best, best_d = light, d
        return best

    def _near_split(self, pos: QPointF) -> bool:
        if not self.compare_split:
            return False
        rect = self.image_rect()
        x = rect.x() + rect.width() * self.split_pos
        return abs(pos.x() - x) < 8 and rect.top() <= pos.y() <= rect.bottom()

    # -------------------------------------------------------------------- mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self.setFocus()
        pos = e.position()
        self._last_pos = pos
        if not self.ctl.doc:
            return
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and (self._space or self.tool == "pan")):
            self._drag = "pan"
            self._update_cursor()
            return
        if e.button() == Qt.MouseButton.RightButton and self.tool in ("mask", "depth"):
            self._start_brush(pos, alt=True)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._near_split(pos):
            self._drag = "split"
            return
        if self.tool == "zoom":
            self.set_zoom(self.zoom * (0.5 if e.modifiers() & Qt.KeyboardModifier.AltModifier else 2.0), pos)
            return
        if self.tool in ("mask", "depth"):
            self._start_brush(pos, alt=bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
            return
        if self.tool.startswith("place:"):
            u, v = self.to_norm(pos)
            self.ctl.add_light(LightType(self.tool.split(":", 1)[1]), x=u, y=v)
            self.set_tool("select")
            return
        light = self.light_at(pos)
        if light is not None:
            self.ctl.select(light.id)
            self._drag_offset = self.to_widget(light.x, light.y) - pos
            self._drag = "light"
            self._drag_light = light.id
            self.ctl.interactionChanged.emit(True)
        else:
            self.ctl.select(None)
        self.update()

    def _start_brush(self, pos: QPointF, alt: bool) -> None:
        u, v = self.to_norm(pos)
        self._drag = "brush"
        self._brush_alt = alt
        self.ctl.begin_stroke(self.tool, u, v, alt)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position()
        self._mouse = pos
        if self.ctl.doc:
            u, v = self.to_norm(pos)
            self.cursorMoved.emit((u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None)
        if self._drag == "pan":
            self.pan += pos - self._last_pos
            self.fit_mode = False
        elif self._drag == "split":
            rect = self.image_rect()
            self.split_pos = float(np.clip((pos.x() - rect.x()) / rect.width(), 0.0, 1.0))
        elif self._drag == "light" and self._drag_light is not None:
            u, v = self.to_norm(pos + self._drag_offset)
            u, v = float(np.clip(u, -0.5, 1.5)), float(np.clip(v, -0.5, 1.5))
            self.ctl.update_light(self._drag_light, merge_key=f"drag{self._drag_light}", x=u, y=v)
        elif self._drag == "brush":
            u, v = self.to_norm(pos)
            self.ctl.stroke_to(u, v, self._brush_alt)
        elif self._near_split(pos):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif self.tool == "select":
            hover = self.light_at(pos)
            self.setCursor(Qt.CursorShape.SizeAllCursor if hover else Qt.CursorShape.ArrowCursor)
        self._last_pos = pos
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._drag == "light":
            self.ctl.interactionChanged.emit(False)
        elif self._drag == "brush":
            self.ctl.end_stroke()
        self._drag = None
        self._drag_light = None
        self._update_cursor()
        self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._mouse = QPointF(-1000, -1000)
        self.cursorMoved.emit(None)
        self.update()

    def wheelEvent(self, e: QWheelEvent) -> None:  # noqa: N802
        if not self.ctl.doc:
            return
        delta = e.angleDelta()
        steps = (delta.y() or delta.x()) / 120.0
        mods = e.modifiers()
        light = self.ctl.selected_light()
        if light is not None and mods & Qt.KeyboardModifier.ControlModifier:
            self.ctl.update_light(light.id, merge_key=f"wheelz{light.id}", z=float(np.clip(light.z - 0.03 * steps, -1.0, 1.5)))
        elif light is not None and mods & Qt.KeyboardModifier.ShiftModifier:
            self.ctl.update_light(light.id, merge_key=f"wheelr{light.id}", radius=float(np.clip(light.radius * (1.08 ** steps), 0.05, 3.0)))
        elif light is not None and mods & Qt.KeyboardModifier.AltModifier:
            self.ctl.update_light(light.id, merge_key=f"wheeli{light.id}", intensity=float(np.clip(light.intensity * (1.06 ** steps), 0.0, 10.0)))
        elif self.tool in ("mask", "depth") and mods & Qt.KeyboardModifier.ShiftModifier:
            self.ctl.brush.size = float(np.clip(self.ctl.brush.size * (1.1 ** steps), 0.005, 0.5))
        else:
            self.set_zoom(self.zoom * (1.15 ** steps), e.position())
        self.update()

    # ----------------------------------------------------------------- keyboard
    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        key = e.key()
        if key == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor()
        elif key == Qt.Key.Key_Backslash and not e.isAutoRepeat():
            self.hold_original = True
            self.update()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.ctl.remove_light(self.ctl.selected)
        elif key == Qt.Key.Key_BracketLeft:
            self.ctl.brush.size = max(0.005, self.ctl.brush.size / 1.15)
            self.update()
        elif key == Qt.Key.Key_BracketRight:
            self.ctl.brush.size = min(0.5, self.ctl.brush.size * 1.15)
            self.update()
        elif key == Qt.Key.Key_H:
            self.show_gizmos = not self.show_gizmos
            self.update()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down) and self.ctl.selected_light():
            light = self.ctl.selected_light()
            step = 0.05 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.01
            dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(key, 0.0)
            dy = {Qt.Key.Key_Up: -step, Qt.Key.Key_Down: step}.get(key, 0.0)
            self.ctl.update_light(light.id, merge_key=f"nudge{light.id}", x=light.x + dx, y=light.y + dy)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if e.isAutoRepeat():
            return
        if e.key() == Qt.Key.Key_Space:
            self._space = False
            self._update_cursor()
        elif e.key() == Qt.Key.Key_Backslash:
            self.hold_original = False
            self.update()
        else:
            super().keyReleaseEvent(e)
