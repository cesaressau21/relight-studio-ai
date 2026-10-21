"""Original line-icon set drawn with QPainter (no external icon assets)."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from . import theme

Draw = Callable[[QPainter], None]


def _rays(p: QPainter, cx: float, cy: float, r0: float, r1: float, n: int = 8, start: float = 0.0) -> None:
    for i in range(n):
        a = start + i * 2 * math.pi / n
        p.drawLine(QPointF(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), QPointF(cx + r1 * math.cos(a), cy + r1 * math.sin(a)))


def _open(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(3, 7); path.lineTo(9, 7); path.lineTo(11, 9); path.lineTo(21, 9)
    path.lineTo(21, 19); path.lineTo(3, 19); path.closeSubpath()
    p.drawPath(path)


def _save(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
    p.drawRect(QRectF(8, 4, 8, 5))
    p.drawRect(QRectF(7, 13, 10, 7))


def _export(p: QPainter) -> None:
    p.drawLine(QPointF(12, 15), QPointF(12, 3))
    p.drawPolyline([QPointF(7, 8), QPointF(12, 3), QPointF(17, 8)])
    p.drawPolyline([QPointF(4, 13), QPointF(4, 20), QPointF(20, 20), QPointF(20, 13)])


def _undo(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(5, 10); path.lineTo(15, 10)
    path.arcTo(QRectF(10, 10, 10, 10), 90, -180)
    path.lineTo(9, 20)
    p.drawPath(path)
    p.drawPolyline([QPointF(9, 6), QPointF(5, 10), QPointF(9, 14)])


def _redo(p: QPainter) -> None:
    p.translate(24, 0); p.scale(-1, 1); _undo(p)


def _compare(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3, 5, 18, 14), 2, 2)
    p.drawLine(QPointF(12, 3), QPointF(12, 21))
    p.save(); p.setBrush(p.pen().color()); p.setOpacity(0.35)
    p.drawRect(QRectF(3, 5, 9, 14)); p.restore()


def _select(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(6, 3); path.lineTo(18, 14); path.lineTo(12.5, 14.5); path.lineTo(15.5, 20.5)
    path.lineTo(13, 21.5); path.lineTo(10, 15.5); path.lineTo(6, 19); path.closeSubpath()
    p.drawPath(path)


def _point(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 3.2, 3.2)
    _rays(p, 12, 12, 6, 9)


def _directional(p: QPainter) -> None:
    p.drawEllipse(QPointF(7, 7), 3, 3)
    _rays(p, 7, 7, 4.8, 6.4, 8)
    for off in (0, 4):
        p.drawLine(QPointF(11 + off, 13 - off + 4), QPointF(18 + off - 2, 20 - off + 2))
    p.drawPolyline([QPointF(14, 21), QPointF(19, 21), QPointF(19, 16)])


def _ambient(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 8, 8)
    p.save(); p.setOpacity(0.5)
    p.drawEllipse(QPointF(12, 12), 4.5, 4.5)
    p.restore()


def _area(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 5, 16, 9), 2, 2)
    for x in (7, 12, 17):
        p.drawLine(QPointF(x, 16.5), QPointF(x, 20))


def _rim(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 7, 7)
    pen = QPen(p.pen()); pen.setWidthF(3.0); p.save(); p.setPen(pen)
    p.drawArc(QRectF(5, 5, 14, 14), -60 * 16, 120 * 16); p.restore()


def _mask(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(17, 3); path.lineTo(21, 7); path.lineTo(11, 17); path.lineTo(7, 13); path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(14, 6), QPointF(18, 10))
    brush = QPainterPath()
    brush.moveTo(7, 15); brush.cubicTo(3, 15, 4, 20, 3, 21); brush.cubicTo(7, 21, 10, 20, 9, 17)
    p.drawPath(brush)


def _depth(p: QPainter) -> None:
    for i, y in enumerate((7, 12, 17)):
        p.save(); p.setOpacity(1.0 - i * 0.28)
        p.drawLine(QPointF(4 + i * 2, y), QPointF(20 - i * 2, y)); p.restore()
    p.drawPolyline([QPointF(12, 2.5), QPointF(12, 21.5)])
    p.drawPolyline([QPointF(9.5, 19), QPointF(12, 21.5), QPointF(14.5, 19)])


def _hand(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(8, 13); path.lineTo(8, 5.5); path.arcTo(QRectF(8, 4, 2.6, 3), 180, -180)
    path.lineTo(10.6, 11); path.lineTo(10.6, 3.8); path.arcTo(QRectF(10.6, 2.5, 2.6, 3), 180, -180)
    path.lineTo(13.2, 11); path.lineTo(13.2, 5); path.arcTo(QRectF(13.2, 3.5, 2.6, 3), 180, -180)
    path.lineTo(15.8, 12); path.lineTo(15.8, 7.5); path.arcTo(QRectF(15.8, 6, 2.6, 3), 180, -180)
    path.lineTo(18.4, 15); path.cubicTo(18.4, 19, 16, 21, 13, 21)
    path.cubicTo(10, 21, 8.5, 19.5, 7, 17); path.lineTo(4.8, 13.5)
    path.cubicTo(4, 12, 6.5, 11, 8, 13)
    p.drawPath(path)


def _zoom(p: QPainter) -> None:
    p.drawEllipse(QPointF(10.5, 10.5), 6, 6)
    p.drawLine(QPointF(15, 15), QPointF(20.5, 20.5))
    p.drawLine(QPointF(8, 10.5), QPointF(13, 10.5)); p.drawLine(QPointF(10.5, 8), QPointF(10.5, 13))


def _fit(p: QPainter) -> None:
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        cx, cy = 12 - sx * 8, 12 - sy * 8
        p.drawPolyline([QPointF(cx, cy + sy * 5), QPointF(cx, cy), QPointF(cx + sx * 5, cy)])


def _eye(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(2.5, 12); path.cubicTo(7, 5, 17, 5, 21.5, 12); path.cubicTo(17, 19, 7, 19, 2.5, 12)
    p.drawPath(path); p.drawEllipse(QPointF(12, 12), 3, 3)


def _eye_off(p: QPainter) -> None:
    _eye(p); p.drawLine(QPointF(4, 4), QPointF(20, 20))


def _plus(p: QPainter) -> None:
    p.drawLine(QPointF(12, 5), QPointF(12, 19)); p.drawLine(QPointF(5, 12), QPointF(19, 12))


def _trash(p: QPainter) -> None:
    p.drawLine(QPointF(4, 7), QPointF(20, 7))
    p.drawPolyline([QPointF(9, 7), QPointF(9.5, 4), QPointF(14.5, 4), QPointF(15, 7)])
    p.drawPolyline([QPointF(6, 7), QPointF(7, 20), QPointF(17, 20), QPointF(18, 7)])


def _duplicate(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(8, 8, 12, 12), 2, 2)
    p.drawPolyline([QPointF(5, 15), QPointF(4, 15), QPointF(4, 4), QPointF(15, 4), QPointF(15, 5)])


def _sparkle(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(12, 3); path.cubicTo(12.8, 9, 15, 11.2, 21, 12); path.cubicTo(15, 12.8, 12.8, 15, 12, 21)
    path.cubicTo(11.2, 15, 9, 12.8, 3, 12); path.cubicTo(9, 11.2, 11.2, 9, 12, 3)
    p.drawPath(path)


def _reset(p: QPainter) -> None:
    p.drawArc(QRectF(5, 5, 14, 14), 100 * 16, 290 * 16)
    p.drawPolyline([QPointF(5, 4), QPointF(9, 6.3), QPointF(6.5, 10)])


_ICONS: dict[str, Draw] = {
    "open": _open, "save": _save, "export": _export, "undo": _undo, "redo": _redo, "compare": _compare,
    "select": _select, "point": _point, "directional": _directional, "ambient": _ambient, "area": _area,
    "rim": _rim, "mask": _mask, "depth": _depth, "hand": _hand, "zoom": _zoom, "fit": _fit, "eye": _eye,
    "eye_off": _eye_off, "plus": _plus, "trash": _trash, "duplicate": _duplicate, "sparkle": _sparkle,
    "reset": _reset,
}


def _render(name: str, color: str, size: int) -> QPixmap:
    ratio = 2.0
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(ratio)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 24.0, size / 24.0)
    pen = QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    _ICONS[name](p)
    p.end()
    return pm


@lru_cache(maxsize=None)
def icon(name: str, size: int = 20) -> QIcon:
    ic = QIcon()
    ic.addPixmap(_render(name, theme.TEXT, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(_render(name, theme.ACCENT, size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(_render(name, theme.ACCENT, size), QIcon.Mode.Active, QIcon.State.On)
    ic.addPixmap(_render(name, theme.FAINT, size), QIcon.Mode.Disabled, QIcon.State.Off)
    return ic


def logo_pixmap(size: int = 22) -> QPixmap:
    """Brand mark: the subject and a light orbiting it (see assets/brand, 56-unit grid).

    Drawn at the small-icon optics of the logo kit: heavier orbit, larger light.
    """
    ratio = 2.0
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(ratio)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 56.0, size / 56.0)
    center, orbit, stroke, subject, light, halo = 28.0, 22.0, 5.4, 13.5, 7.2, 3.6
    angle = math.radians(225)
    lx, ly = center + orbit * math.cos(angle), center + orbit * math.sin(angle)
    # Orbit with a gap around the light: the arc spans the rest of the circle.
    gap = math.degrees(2 * math.asin(min(1.0, (light + halo) / (2 * orbit))))
    p.setPen(QPen(QColor(theme.ACCENT), stroke, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    rect = QRectF(center - orbit, center - orbit, 2 * orbit, 2 * orbit)
    start = 135 + gap  # Qt angles run counterclockwise from 3 o'clock; the light sits at 135°
    p.drawArc(rect, int(start * 16), int((360 - 2 * gap) * 16))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(theme.TEXT))
    p.drawEllipse(QPointF(center, center), subject, subject)
    p.setBrush(QColor(theme.ACCENT))
    p.drawEllipse(QPointF(lx, ly), light, light)
    p.end()
    return pm
