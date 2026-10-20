"""Removing the original light's color from the editor: controls, neutral point, history and projects."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from relight_studio.core.presets import load_presets

from .conftest import settle
from .test_background_editor import write_background


def test_light_color_slider_is_undoable(qapp, editor):
    ctl, form = editor.ctl, editor.scene_panel.relight_form
    form.rows["light_color"].spin.setValue(0.7)
    assert ctl.doc.relight.light_color == pytest.approx(0.7)
    ctl.undo()
    assert ctl.doc.relight.light_color == 0.0
    settle(qapp, editor)


def test_pick_neutral_on_the_canvas(qapp, editor):
    ctl, canvas, panel = editor.ctl, editor.canvas, editor.scene_panel
    assert not panel.clear_neutral.isEnabled()
    panel.pick_neutral.click()
    assert canvas.tool == "neutral"
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.to_widget(0.25, 0.75).toPoint())
    x, y = ctl.doc.relight.neutral_point
    assert x == pytest.approx(0.25, abs=0.02) and y == pytest.approx(0.75, abs=0.02)
    assert ctl.doc.relight.light_color == 1.0  # picking turns the removal on
    assert canvas.tool == "select" and panel.clear_neutral.isEnabled()
    assert "neutral point" in panel.neutral_hint.text().lower()
    ctl.undo()
    assert ctl.doc.relight.neutral_point is None and ctl.doc.relight.light_color == 0.0
    settle(qapp, editor)


def test_escape_cancels_picking(qapp, editor):
    editor.scene_panel.pick_neutral.click()
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    assert editor.canvas.tool == "select" and editor.ctl.doc.relight.neutral_point is None


def test_clear_neutral_point(qapp, editor):
    ctl = editor.ctl
    ctl.set_neutral_point(0.4, 0.4)
    editor.scene_panel.clear_neutral.click()
    assert ctl.doc.relight.neutral_point is None
    assert ctl.doc.relight.light_color == 1.0  # the removal itself stays on
    assert not editor.scene_panel.clear_neutral.isEnabled()
    settle(qapp, editor)


def test_points_outside_the_image_are_clamped(qapp, editor):
    editor.ctl.set_neutral_point(-0.3, 1.4)
    assert editor.ctl.doc.relight.neutral_point == (0.0, 1.0)
    settle(qapp, editor)


def test_project_keeps_light_color_and_neutral_point(qapp, editor, tmp_path):
    from relight_studio.core.project_io import save_project

    ctl = editor.ctl
    ctl.update_relight(light_color=0.6)
    ctl.set_neutral_point(0.3, 0.6)
    project = save_project(ctl.doc, tmp_path / "scene.rsproj")
    ctl.update_relight(light_color=0.0)
    ctl.clear_neutral_point()
    editor.open_project(project)
    assert ctl.doc.relight.light_color == pytest.approx(0.6)
    assert ctl.doc.relight.neutral_point == pytest.approx((0.3, 0.6))
    settle(qapp, editor)


def test_presets_keep_the_photo_correction(qapp, editor):
    ctl = editor.ctl
    ctl.set_neutral_point(0.5, 0.2)
    ctl.update_relight(light_color=0.8)
    ctl.apply_preset(next(p for p in load_presets() if p.name == "Neon Cyberpunk"))
    assert ctl.doc.relight.light_color == pytest.approx(0.8)
    assert ctl.doc.relight.neutral_point == (0.5, 0.2)
    settle(qapp, editor)


def test_matching_a_background_removes_the_old_light_color(qapp, editor, tmp_path):
    ctl = editor.ctl
    ctl.set_background(str(write_background(tmp_path / "beach.jpg")))
    ctl.match_lighting()
    assert ctl.doc.relight.light_color == 1.0
    settle(qapp, editor)
