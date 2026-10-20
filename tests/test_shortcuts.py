"""Every window shortcut the README lists, pressed for real."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from relight_studio.core.lights import LightType

from .conftest import settle

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


def press(qapp, editor, key, modifiers=Qt.KeyboardModifier.NoModifier):
    editor.activateWindow()
    editor.canvas.setFocus()
    QTest.keyClick(editor.canvas, key, modifiers)
    qapp.processEvents()


@pytest.mark.parametrize(("key", "tool"), [
    (Qt.Key.Key_P, "place:point"), (Qt.Key.Key_A, "place:area"), (Qt.Key.Key_L, "place:directional"),
    (Qt.Key.Key_R, "place:rim"), (Qt.Key.Key_B, "mask"), (Qt.Key.Key_D, "depth"), (Qt.Key.Key_Z, "zoom"),
])
def test_tool_shortcuts(qapp, editor, key, tool):
    press(qapp, editor, key)
    assert editor.canvas.tool == tool
    press(qapp, editor, Qt.Key.Key_Escape)
    assert editor.canvas.tool == "select"
    press(qapp, editor, key)
    press(qapp, editor, Qt.Key.Key_V)
    assert editor.canvas.tool == "select"


def test_compare_shortcut(qapp, editor):
    press(qapp, editor, Qt.Key.Key_C)
    assert editor.compare_btn.isChecked() and editor.canvas.compare_split
    press(qapp, editor, Qt.Key.Key_C)
    assert not editor.compare_btn.isChecked()


def test_undo_redo_shortcuts(qapp, editor):
    ctl = editor.ctl
    count = len(ctl.doc.lights)
    ctl.add_light(LightType.POINT)
    press(qapp, editor, Qt.Key.Key_Z, CTRL)
    assert len(ctl.doc.lights) == count
    press(qapp, editor, Qt.Key.Key_Y, CTRL)
    assert len(ctl.doc.lights) == count + 1
    press(qapp, editor, Qt.Key.Key_Z, CTRL)
    press(qapp, editor, Qt.Key.Key_Z, CTRL | SHIFT)
    assert len(ctl.doc.lights) == count + 1
    settle(qapp, editor)


def test_zoom_shortcuts(qapp, editor):
    canvas = editor.canvas
    press(qapp, editor, Qt.Key.Key_1, CTRL)
    assert canvas.zoom == pytest.approx(1.0)
    press(qapp, editor, Qt.Key.Key_Equal, CTRL)
    assert canvas.zoom == pytest.approx(1.25)
    press(qapp, editor, Qt.Key.Key_Minus, CTRL)
    assert canvas.zoom == pytest.approx(1.0)
    press(qapp, editor, Qt.Key.Key_0, CTRL)
    rect = canvas.image_rect()
    assert rect.width() <= canvas.width() + 1 and rect.height() <= canvas.height() + 1


def test_file_shortcuts_open_their_dialogs(qapp, editor, monkeypatch):
    opened = []
    monkeypatch.setattr("relight_studio.ui.main_window.QFileDialog.getOpenFileName",
                        lambda *a, **k: (opened.append("open"), ("", ""))[1])
    monkeypatch.setattr("relight_studio.ui.main_window.QFileDialog.getSaveFileName",
                        lambda *a, **k: (opened.append("save"), ("", ""))[1])
    monkeypatch.setattr("relight_studio.ui.main_window.ExportDialog.exec", lambda self: (opened.append("export"), 0)[1])
    press(qapp, editor, Qt.Key.Key_O, CTRL)
    press(qapp, editor, Qt.Key.Key_S, CTRL)
    press(qapp, editor, Qt.Key.Key_S, CTRL | SHIFT)
    press(qapp, editor, Qt.Key.Key_E, CTRL)
    assert opened == ["open", "save", "save", "export"]
