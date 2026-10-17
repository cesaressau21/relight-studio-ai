"""Main window workflows: shortcuts, files, export, rendering queue."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QMessageBox

from relight_studio.core.image_io import ExportOptions, load_image
from relight_studio.core.lights import LightType
from relight_studio.core.project_io import read_project, save_project

from .conftest import settle, wait_until, write_photo


def shortcut(win, keys: str) -> QShortcut:
    seq = QKeySequence(keys)
    return next(s for s in win.findChildren(QShortcut) if s.key() == seq)


@pytest.mark.parametrize("keys,tool", [("V", "select"), ("P", "place:point"), ("A", "place:area"),
                                       ("L", "place:directional"), ("R", "place:rim"), ("B", "mask"),
                                       ("D", "depth"), ("Z", "zoom"), ("Escape", "select")])
def test_tool_shortcuts(qapp, editor, keys, tool):
    shortcut(editor, keys).activated.emit()
    assert editor.canvas.tool == tool
    assert editor.tool_buttons[tool].isChecked()


def test_tool_rail_buttons(qapp, editor):
    editor.tool_buttons["place:rim"].click()
    assert editor.canvas.tool == "place:rim"
    editor.ambient_btn.click()
    assert editor.ctl.selected_light().type is LightType.AMBIENT
    editor.tool_buttons["mask"].click()
    assert editor.tabs.currentWidget() is editor.layers_panel
    editor.canvas.set_zoom(3.0)
    editor.actual_btn.click()
    assert editor.canvas.zoom == 1.0
    editor.fit_btn.click()
    assert editor.canvas.fit_mode


def test_undo_redo_buttons_and_shortcuts(qapp, editor):
    assert not editor.undo_btn.isEnabled()
    editor.ctl.update_relight(ambient=0.5)
    assert editor.undo_btn.isEnabled()
    shortcut(editor, "Ctrl+Z").activated.emit()
    assert editor.ctl.doc.relight.ambient == 0.0 and editor.redo_btn.isEnabled()
    shortcut(editor, "Ctrl+Y").activated.emit()
    assert editor.ctl.doc.relight.ambient == 0.5
    editor.undo_btn.click()
    shortcut(editor, "Ctrl+Shift+Z").activated.emit()
    assert editor.ctl.doc.relight.ambient == 0.5


def test_compare_shortcut_and_zoom_shortcuts(qapp, editor):
    shortcut(editor, "C").activated.emit()
    assert editor.canvas.compare_split
    zoom = editor.canvas.zoom
    shortcut(editor, "Ctrl+=").activated.emit()
    assert editor.canvas.zoom > zoom
    shortcut(editor, "Ctrl+1").activated.emit()
    assert editor.canvas.zoom == 1.0
    shortcut(editor, "Ctrl+0").activated.emit()
    assert editor.canvas.fit_mode


def test_title_marks_unsaved_changes(qapp, editor):
    assert "•" not in editor.title_label.text()
    editor.ctl.update_adjustments(exposure=0.3)
    assert "•" in editor.title_label.text() and "•" in editor.windowTitle()


def test_status_bar_shows_resolution_render_time_and_device(qapp, editor):
    settle(qapp, editor)
    assert "180×120" in editor.res_label.text()
    assert "Render" in editor.render_label.text() and "ms" in editor.render_label.text()
    assert editor.gpu_label.text()


def test_save_and_reopen_project_through_the_window(qapp, editor, tmp_path, monkeypatch):
    ctl = editor.ctl
    ctl.update_light(ctl.selected, x=0.66, intensity=2.2)
    ctl.fill_mask()
    target = tmp_path / "scene.rsproj"
    monkeypatch.setattr("relight_studio.ui.main_window.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(target), ""))
    assert editor.save_project()
    assert not ctl.dirty and target.exists()

    runs = []
    monkeypatch.setattr(type(editor), "run_depth", lambda self: runs.append(1))
    ctl.update_light(ctl.selected, x=0.1)
    editor.open_project(target)
    assert ctl.selected_light().x == pytest.approx(0.66)
    assert ctl.doc.mask is not None and ctl.project_path == target
    assert runs == []  # cached depth was reused, no new inference
    settle(qapp, editor)


def test_open_project_with_missing_photo_asks_to_locate_it(qapp, editor, tmp_path, monkeypatch):
    photo = write_photo(tmp_path / "moved.png")
    doc = editor.ctl.doc
    project = save_project(doc, tmp_path / "p.rsproj")
    doc.image_path.unlink()
    monkeypatch.setattr("relight_studio.ui.main_window.QMessageBox.warning", lambda *a, **k: None)
    monkeypatch.setattr("relight_studio.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(photo), ""))
    editor.open_project(project)
    assert editor.ctl.doc.image_path == photo
    settle(qapp, editor)


def test_export_through_the_window(qapp, editor, tmp_path, monkeypatch):
    from relight_studio.ui.export_dialog import ExportDialog

    out = tmp_path / "out.webp"
    monkeypatch.setattr(ExportDialog, "exec", lambda self: True)
    monkeypatch.setattr(ExportDialog, "target_path", lambda self: out)
    monkeypatch.setattr(ExportDialog, "options", lambda self: ExportOptions(format="WEBP", quality=80, long_side=90))
    editor.ctl.update_light(editor.ctl.selected, intensity=4.0)
    editor.export_image()
    assert wait_until(qapp, lambda: editor._export_worker is None, timeout=60)
    with Image.open(out) as im:
        assert im.format == "WEBP" and im.size == (90, 60)
    assert "Exported" in editor.status_label.text()


def test_export_dialog_options(qapp):
    from relight_studio.ui.export_dialog import ExportDialog

    dialog = ExportDialog(4000, 3000, Path("C:/x/photo_relit.png"))
    dialog.format.setCurrentIndex(dialog.format.findData("JPEG"))
    assert dialog.target_path().suffix == ".jpg" and dialog.quality.isEnabled()
    dialog.format.setCurrentIndex(dialog.format.findData("PNG"))
    assert dialog.target_path().suffix == ".png" and not dialog.quality.isEnabled()
    dialog.size.setCurrentIndex(dialog.size.findData(2048))
    assert dialog.options().target_size(4000, 3000) == (2048, 1536)
    dialog.icc.setChecked(False)
    assert dialog.options().embed_srgb is False
    assert "2048 × 1536" in dialog.info.text()


def test_closing_with_unsaved_changes_can_be_cancelled(qapp, editor, monkeypatch):
    editor.ctl.update_relight(ambient=0.4)
    monkeypatch.setattr("relight_studio.ui.main_window.QMessageBox.question",
                        lambda *a, **k: QMessageBox.StandardButton.Cancel)
    assert not editor.close()
    assert editor.isVisible()
    monkeypatch.setattr("relight_studio.ui.main_window.QMessageBox.question",
                        lambda *a, **k: QMessageBox.StandardButton.Discard)
    assert editor._confirm_discard()


def test_drag_and_drop_accepts_photos_and_projects(qapp, editor, tmp_path):
    for name, accepted in (("a.jpg", True), ("b.rsproj", True), ("c.txt", False)):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(tmp_path / name))])
        event = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                Qt.KeyboardModifier.NoModifier)
        event.setAccepted(False)
        editor.dragEnterEvent(event)
        assert event.isAccepted() == accepted, name


def test_opening_a_bad_file_shows_an_error(qapp, editor, tmp_path, monkeypatch):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    shown = []
    monkeypatch.setattr("relight_studio.ui.main_window.QMessageBox.critical", lambda *a, **k: shown.append(a))
    editor.open_image(bad)
    assert shown and editor.ctl.doc.image_path.name == "photo.png"


def test_render_queue_keeps_only_the_latest_request(qapp, editor):
    ctl = editor.ctl
    settle(qapp, editor)
    frames = []
    editor.render_service.frameReady.connect(lambda img, stats: frames.append(stats))
    for i in range(10):
        ctl.update_light(ctl.selected, merge_key="burst", intensity=1.0 + i * 0.2)
    settle(qapp, editor)
    assert 1 <= len(frames) <= 3  # intermediate requests were coalesced


def test_depth_edits_toggle_controls_what_the_renderer_receives(qapp, editor, monkeypatch):
    jobs = []
    monkeypatch.setattr(editor.render_service, "request", jobs.append)
    editor.ctl.begin_stroke("depth", 0.5, 0.5, False)
    editor.ctl.end_stroke()
    assert jobs[-1].offset is not None
    editor.ctl.update_mask_settings(depth_edits_enabled=False)
    assert jobs[-1].offset is None


def test_original_photo_file_is_never_written(qapp, editor):
    path = editor.ctl.doc.image_path
    before = path.read_bytes()
    editor.ctl.update_light(editor.ctl.selected, intensity=5.0)
    editor.ctl.fill_mask()
    settle(qapp, editor)
    assert path.read_bytes() == before
    assert np.array_equal(load_image(path), editor.ctl.doc.image)


def test_ai_matte_is_scheduled_and_applied(qapp, editor, monkeypatch):
    import relight_studio.ai.matting as matting

    calls = []

    def fake_matte(image, disparity, settings, offset=None, matter=None):
        calls.append(settings.edge_band)
        return np.full(image.shape[:2], 255, np.uint8)

    monkeypatch.setattr(matting, "matte_subject", fake_matte)
    ctl = editor.ctl
    ctl.update_depth(edge_mode="matting", edge_band=0.03)
    assert wait_until(qapp, lambda: ctl.current_alpha() is not None, timeout=10)
    assert calls[-1] == pytest.approx(0.03)
    jobs = []
    monkeypatch.setattr(editor.render_service, "request", jobs.append)
    ctl.update_light(ctl.selected, intensity=2.0)
    assert jobs[-1].alpha is not None
    ctl.update_depth(edge_band=0.05)  # settings changed: the old matte no longer applies
    assert ctl.current_alpha() is None
    assert wait_until(qapp, lambda: ctl.current_alpha() is not None, timeout=10)


def test_fast_and_off_edge_modes_skip_ai_matting(qapp, editor, monkeypatch):
    import relight_studio.ai.matting as matting

    calls = []
    monkeypatch.setattr(matting, "matte_subject", lambda *a, **k: calls.append(1))
    for mode in ("guided", "off"):
        editor.ctl.update_depth(edge_mode=mode)
        editor.run_matte()
    assert calls == []
