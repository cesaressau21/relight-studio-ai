"""Side view of the scene: where the selected light sits in depth.

Horizontal axis: depth, from the camera (left) to the farthest surface (right).
Vertical axis: image rows. The filled silhouette is the estimated surface along
the image column under the selected light. Dragging the light marker changes
its Z (horizontally) and Y (vertically).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ..core.color import rgb_to_hex
from . import theme
from .controller import EditorController

Z_MIN, Z_MAX = -1.0, 1.5


class DepthProfile(QWidget):
    def __init__(self, controller: EditorController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctl = controller
        self.depth_map: np.ndarray | None = None
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setToolTip("Side view: drag the light horizontally to change depth (Z), vertically for Y")
        self._dragging = False

    def set_depth(self, depth: np.ndarray | None) -> None:
        self.depth_map = depth
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(26, 8, self.width() - 34, self.height() - 26)

    def _z_to_x(self, z: float, r: QRectF) -> float:
        return r.left() + (z - Z_MIN) / (Z_MAX - Z_MIN) * r.width()

    def _x_to_z(self, x: float, r: QRectF) -> float:
        return Z_MIN + (x - r.left()) / r.width() * (Z_MAX - Z_MIN)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.SURFACE))
        r = self._plot()
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor(theme.BG))
        p.drawRoundedRect(r, 6, 6)

        # Near / far guides of the normalized scene depth.
        for z, text in ((0.0, "near"), (1.0, "far")):
            x = self._z_to_x(z, r)
            p.setPen(QPen(QColor(theme.FAINT), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QColor(theme.FAINT))
            f = QFont(self.font())
            f.setPointSizeF(7)
            p.setFont(f)
            p.drawText(QRectF(x - 20, r.bottom() + 2, 40, 14), Qt.AlignmentFlag.AlignCenter, text)

        # Camera glyph on the left.
        cy = r.center().y()
        p.setPen(QPen(QColor(theme.MUTED), 1.3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(4, cy - 6, 12, 12), 2, 2)
        p.drawPolyline([QPointF(16, cy - 3), QPointF(21, cy - 6), QPointF(21, cy + 6), QPointF(16, cy + 3)])

        light = self.ctl.selected_light()
        if self.depth_map is not None and light is not None:
            h, w = self.depth_map.shape
            col = int(np.clip(light.x, 0.0, 0.999) * w)
            profile = self.depth_map[:, max(0, col - 1) : col + 2].mean(axis=1)
            path = QPainterPath()
            path.moveTo(r.right(), r.top())
            for i, d in enumerate(profile):
                path.lineTo(self._z_to_x(float(d), r), r.top() + (i + 0.5) / h * r.height())
            path.lineTo(r.right(), r.bottom())
            path.closeSubpath()
            grad = QLinearGradient(r.topLeft(), r.topRight())
            grad.setColorAt(0, QColor("#5a4a3c"))
            grad.setColorAt(1, QColor("#2a2a30"))
            p.save()
            p.setClipRect(r)
            p.setPen(QPen(QColor("#a08066"), 1.2))
            p.setBrush(grad)
            p.drawPath(path)
            p.restore()

        if self.ctl.doc is not None:
            for other in self.ctl.doc.lights:
                if not other.is_positional:
                    continue
                pos = QPointF(self._z_to_x(other.z, r), r.top() + np.clip(other.y, 0, 1) * r.height())
                sel = light is not None and other.id == light.id
                p.setPen(QPen(QColor(other.gizmo_color), 2 if sel else 1))
                p.setBrush(QColor(rgb_to_hex(other.color)) if sel else QColor(0, 0, 0, 0))
                p.drawEllipse(pos, 6 if sel else 4, 6 if sel else 4)
                if sel:
                    p.setPen(QPen(QColor(other.gizmo_color), 1, Qt.PenStyle.DashLine))
                    p.drawLine(QPointF(r.left(), pos.y()), pos)

        if light is None:
            p.setPen(QColor(theme.FAINT))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Select a light")

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.ctl.selected_light():
            self._dragging = True
            self.ctl.interactionChanged.emit(True)
            self._apply(e.position())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._apply(e.position())

    def mouseReleaseEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            self.ctl.interactionChanged.emit(False)

    def _apply(self, pos: QPointF) -> None:
        light = self.ctl.selected_light()
        if light is None:
            return
        r = self._plot()
        z = float(np.clip(self._x_to_z(pos.x(), r), Z_MIN, Z_MAX))
        y = float(np.clip((pos.y() - r.top()) / r.height(), -0.5, 1.5))
        self.ctl.update_light(light.id, merge_key=f"profile{light.id}", z=z, y=y)
