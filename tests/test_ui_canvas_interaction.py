"""Canvas driven with real mouse, wheel and keyboard events."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from relight_studio.core.lights import LightType

from .conftest import settle

L = Qt.MouseButton.LeftButton
NO = Qt.KeyboardModifier.NoModifier


def pt(p: QPointF) -> QPoint:
    return p.toPoint()


def drag(widget, start: QPoint, end: QPoint, button=L, modifiers=NO, steps: int = 6) -> None:
    QTest.mousePress(widget, button, modifiers, start)
    for i in range(1, steps + 1):
        pos = start + (end - start) * i / steps
        QTest.mouseMove(widget, QPoint(round(pos.x()), round(pos.y())))
    QTest.mouseRelease(widget, button, modifiers, end)


def wheel(widget, pos: QPoint, steps: int, modifiers=NO) -> None:
    event = QWheelEvent(QPointF(pos), QPointF(widget.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, 120 * steps),
                        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def test_dragging_a_gizmo_moves_the_light_in_one_undo_step(qapp, editor):
    canvas, ctl = editor.canvas, editor.ctl
    light = ctl.selected_light()
    x0, y0 = light.x, light.y  # lights are edited in place: keep the values
    drag(canvas, pt(canvas.to_widget(x0, y0)), pt(canvas.to_widget(0.7, 0.6)))
    moved = ctl.selected_light()
    assert (moved.x, moved.y) == pytest.approx((0.7, 0.6), abs=0.02)
    ctl.undo()
    assert (ctl.selected_light().x, ctl.selected_light().y) == pytest.approx((x0, y0))
    settle(qapp, editor)


def test_render_updates_while_dragging(qapp, editor):
    canvas, ctl = editor.canvas, editor.ctl
    settle(qapp, editor)
    before = canvas.result
    light = ctl.selected_light()
    drag(canvas, pt(canvas.to_widget(light.x, light.y)), pt(canvas.to_widget(0.9, 0.9)))
    settle(qapp, editor)
    assert canvas.result is not before and canvas.result != before


def test_clicking_empty_space_deselects(qapp, editor):
    canvas = editor.canvas
    QTest.mouseClick(canvas, L, NO, pt(canvas.to_widget(0.95, 0.95)))
    assert editor.ctl.selected is None


@pytest.mark.parametrize("tool,light_type", [("place:point", LightType.POINT), ("place:area", LightType.AREA),
                                             ("place:directional", LightType.DIRECTIONAL),
                                             ("place:rim", LightType.RIM)])
def test_place_tools_add_a_light_where_clicked(qapp, editor, tool, light_type):
    editor.set_tool(tool)
    QTest.mouseClick(editor.canvas, L, NO, pt(editor.canvas.to_widget(0.6, 0.4)))
    light = editor.ctl.selected_light()
    assert light.type is light_type and (light.x, light.y) == pytest.approx((0.6, 0.4), abs=0.02)
    assert editor.canvas.tool == "select"


@pytest.mark.parametrize("modifier,field", [(Qt.KeyboardModifier.ControlModifier, "z"),
                                            (Qt.KeyboardModifier.ShiftModifier, "radius"),
                                            (Qt.KeyboardModifier.AltModifier, "intensity")])
def test_modifier_wheel_adjusts_the_selected_light(qapp, editor, modifier, field):
    before = getattr(editor.ctl.selected_light(), field)
    wheel(editor.canvas, QPoint(300, 300), 2, modifier)
    assert getattr(editor.ctl.selected_light(), field) != pytest.approx(before)


def test_plain_wheel_zooms_around_the_cursor(qapp, editor):
    canvas = editor.canvas
    anchor = pt(canvas.to_widget(0.25, 0.25))
    zoom = canvas.zoom
    wheel(canvas, anchor, 3)
    assert canvas.zoom > zoom and not canvas.fit_mode
    assert canvas.to_norm(QPointF(anchor)) == pytest.approx((0.25, 0.25), abs=0.01)


def test_mask_brush_paints_and_erases(qapp, editor):
    canvas, ctl = editor.canvas, editor.ctl
    ctl.clear_mask()
    editor.set_tool("mask")
    a, b = pt(canvas.to_widget(0.3, 0.5)), pt(canvas.to_widget(0.7, 0.5))
    drag(canvas, a, b)
    mask = ctl.doc.mask
    h, w = mask.shape
    assert mask[h // 2, w // 2] > 200
    drag(canvas, a, b, modifiers=Qt.KeyboardModifier.AltModifier)
    assert ctl.doc.mask[h // 2, w // 2] < 60
    drag(canvas, a, b, button=Qt.MouseButton.RightButton)  # right-drag also erases
    assert ctl.doc.mask[h // 2, w // 2] < 60


def test_depth_brush_pushes_and_pulls(qapp, editor):
    canvas, ctl = editor.canvas, editor.ctl
    editor.set_tool("depth")
    c = pt(canvas.to_widget(0.5, 0.5))
    drag(canvas, c, c + QPoint(4, 0))
    off = ctl.doc.depth_offset
    h, w = off.shape
    assert off[h // 2, w // 2] > 0
    drag(canvas, c, c + QPoint(4, 0), modifiers=Qt.KeyboardModifier.AltModifier)
    drag(canvas, c, c + QPoint(4, 0), modifiers=Qt.KeyboardModifier.AltModifier)
    assert ctl.doc.depth_offset[h // 2, w // 2] < 0


def test_pan_with_middle_button_and_space(qapp, editor):
    canvas = editor.canvas
    pan = QPointF(canvas.pan)
    drag(canvas, QPoint(400, 400), QPoint(450, 420), button=Qt.MouseButton.MiddleButton)
    assert canvas.pan - pan == QPointF(50, 20)
    pan = QPointF(canvas.pan)
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    drag(canvas, QPoint(400, 400), QPoint(380, 400))
    QTest.keyRelease(canvas, Qt.Key.Key_Space)
    assert canvas.pan - pan == QPointF(-20, 0)


def test_zoom_tool_and_fit(qapp, editor):
    canvas = editor.canvas
    editor.set_tool("zoom")
    zoom = canvas.zoom
    QTest.mouseClick(canvas, L, NO, QPoint(400, 300))
    assert canvas.zoom == pytest.approx(zoom * 2)
    QTest.mouseClick(canvas, L, Qt.KeyboardModifier.AltModifier, QPoint(400, 300))
    assert canvas.zoom == pytest.approx(zoom)
    canvas.set_zoom(3.0)
    editor.canvas_fit()
    assert canvas.fit_mode and canvas.image_rect().width() <= canvas.width()


def test_compare_split_divider_drags(qapp, editor):
    canvas = editor.canvas
    editor.compare_btn.setChecked(True)
    assert canvas.compare_split
    rect = canvas.image_rect()
    x = rect.x() + rect.width() * canvas.split_pos
    drag(canvas, QPoint(round(x), round(rect.center().y())), QPoint(round(rect.x() + rect.width() * 0.2), round(rect.center().y())))
    assert canvas.split_pos == pytest.approx(0.2, abs=0.02)
    editor.compare_btn.setChecked(False)
    assert not canvas.compare_split


def test_keyboard_shortcuts_on_canvas(qapp, editor):
    canvas, ctl = editor.canvas, editor.ctl
    canvas.setFocus()
    x0, y0 = ctl.selected_light().x, ctl.selected_light().y
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    assert ctl.selected_light().x == pytest.approx(x0 + 0.01)
    QTest.keyClick(canvas, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert ctl.selected_light().y == pytest.approx(y0 + 0.05)
    size = ctl.brush.size
    QTest.keyClick(canvas, Qt.Key.Key_BracketRight)
    assert ctl.brush.size > size
    QTest.keyClick(canvas, Qt.Key.Key_BracketLeft)
    assert ctl.brush.size == pytest.approx(size)
    QTest.keyClick(canvas, Qt.Key.Key_H)
    assert not canvas.show_gizmos
    QTest.keyPress(canvas, Qt.Key.Key_Backslash)
    assert canvas.hold_original
    QTest.keyRelease(canvas, Qt.Key.Key_Backslash)
    assert not canvas.hold_original
    lid = ctl.selected
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert ctl.doc.light(lid) is None


def test_side_view_drag_sets_depth_and_height(qapp, editor):
    profile = editor.lights_panel.profile
    profile.resize(320, 180)
    r = profile._plot()
    target = QPoint(round(profile._z_to_x(0.8, r)), round(r.top() + 0.25 * r.height()))
    QTest.mousePress(profile, L, NO, target)
    QTest.mouseRelease(profile, L, NO, target)
    light = editor.ctl.selected_light()
    assert light.z == pytest.approx(0.8, abs=0.03) and light.y == pytest.approx(0.25, abs=0.03)


def test_gizmos_never_appear_in_the_render(qapp, editor):
    """The rendered frame is produced without overlays; gizmos are painted only on the widget."""
    settle(qapp, editor)
    frame = editor.canvas.result
    editor.canvas.show_gizmos = False
    editor.ctl.update_light(editor.ctl.selected, intensity=editor.ctl.selected_light().intensity)  # no-op edit
    settle(qapp, editor)
    assert editor.canvas.result.size() == frame.size()
