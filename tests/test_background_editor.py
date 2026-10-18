"""Background replacement in the editor: controls, history, rendering, projects and export."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from relight_studio.core.image_io import ExportOptions
from relight_studio.core.lights import LightTarget

from .conftest import settle, wait_until

BG_COLOR = (30, 60, 200)


def write_background(path, color=BG_COLOR, sun=True):
    img = np.empty((90, 160, 3), np.uint8)
    img[:] = color
    if sun:
        img[5:30, 5:40] = (255, 190, 90)  # warm source upper left
    Image.fromarray(img).save(path)
    return path


@pytest.fixture()
def bg_path(tmp_path):
    return write_background(tmp_path / "beach.jpg")


# ------------------------------------------------------------------ controller
def test_set_background_enables_and_is_undoable(qapp, editor, bg_path):
    ctl = editor.ctl
    ctl.set_background(str(bg_path))
    assert ctl.doc.background.enabled and ctl.doc.background.path == str(bg_path)
    assert ctl.preview_background() is not None
    assert ctl.preview_background().shape == ctl.preview_image.shape
    ctl.undo()
    assert not ctl.doc.background.enabled
    settle(qapp, editor)


def test_unreadable_background_is_rejected(qapp, editor, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(Exception):
        editor.ctl.set_background(str(bad))
    assert not editor.ctl.doc.background.enabled


def test_match_lighting_replaces_the_rig(qapp, editor, bg_path):
    ctl = editor.ctl
    ctl.set_background(str(bg_path))
    before = [l.name for l in ctl.doc.lights]
    ctl.match_lighting()
    lights = ctl.doc.lights
    assert lights and all(l.target is LightTarget.SUBJECT for l in lights)
    assert lights[0].x < 0.5 and lights[0].color[0] > lights[0].color[2]  # warm key from the upper left
    assert ctl.doc.relight.original_light == pytest.approx(0.25)
    ctl.undo()
    assert [l.name for l in ctl.doc.lights] == before
    settle(qapp, editor)


# ------------------------------------------------------------------ rendering
def test_render_shows_the_new_background(qapp, editor, bg_path):
    ctl = editor.ctl
    settle(qapp, editor)
    ctl.update_relight(original_light=1.0)
    ctl.doc.lights.clear()
    ctl.set_background(str(bg_path))
    settle(qapp, editor)
    frame = editor.canvas.result
    corner = frame.pixelColor(frame.width() - 3, frame.height() - 3)
    assert abs(corner.blue() - BG_COLOR[2]) < 12 and abs(corner.red() - BG_COLOR[0]) < 12
    subject = frame.pixelColor(frame.width() // 2, frame.height() // 2)
    assert subject.red() > 150  # the photo's subject (skin-colored block) is still there


def test_background_status_and_hint(qapp, editor, bg_path, monkeypatch):
    panel = editor.background_panel
    editor.ctl.set_background(str(bg_path))
    settle(qapp, editor)
    assert wait_until(qapp, lambda: "behind" in panel.status.text().lower())
    panel.set_status(applied=False)
    assert "subject" in panel.status.text().lower()


# ------------------------------------------------------------------ panel
def test_background_panel_controls(qapp, editor, bg_path, monkeypatch):
    panel, ctl = editor.background_panel, editor.ctl
    assert not panel.match_btn.isEnabled()
    monkeypatch.setattr("relight_studio.ui.panels.QFileDialog.getOpenFileName", lambda *a, **k: (str(bg_path), ""))
    panel.choose_btn.click()
    assert ctl.doc.background.path == str(bg_path) and panel.match_btn.isEnabled()
    assert not panel.thumbnail.pixmap().isNull()
    panel.blur.spin.setValue(0.4)
    panel.brightness.spin.setValue(-0.5)
    assert ctl.doc.background.blur == pytest.approx(0.4) and ctl.doc.background.brightness == pytest.approx(-0.5)
    panel.enabled.setChecked(False)
    assert not ctl.doc.background.enabled
    panel.enabled.setChecked(True)
    panel.match_btn.click()
    assert ctl.doc.lights[0].name.endswith("(matched)")
    panel.remove_btn.click()
    assert ctl.doc.background.path == "" and not ctl.doc.background.enabled
    settle(qapp, editor)


def test_cancelling_the_file_dialog_changes_nothing(qapp, editor, monkeypatch):
    monkeypatch.setattr("relight_studio.ui.panels.QFileDialog.getOpenFileName", lambda *a, **k: ("", ""))
    editor.background_panel.choose_btn.click()
    assert not editor.ctl.doc.background.enabled


# ------------------------------------------------------------------ export and projects
def test_export_includes_the_new_background(qapp, editor, bg_path, tmp_path):
    from relight_studio.ui.workers import ExportWorker

    ctl = editor.ctl
    ctl.update_relight(original_light=1.0)
    ctl.set_background(str(write_background(tmp_path / "plain.png", sun=False)))
    out = tmp_path / "composite.png"
    worker = ExportWorker(editor._export_snapshot(), editor._params(view="result"), ExportOptions(format="PNG"), out)
    worker.start()
    assert wait_until(qapp, worker.isFinished, timeout=60)
    with Image.open(out) as im:
        arr = np.asarray(im.convert("RGB")).astype(int)
    assert np.abs(arr[2, 2] - np.array(BG_COLOR)).max() < 12
    settle(qapp, editor)


def test_project_keeps_the_background(qapp, editor, bg_path, tmp_path):
    from relight_studio.core.project_io import save_project

    ctl = editor.ctl
    ctl.set_background(str(bg_path))
    ctl.update_background(blur=0.3)
    project = save_project(ctl.doc, tmp_path / "scene.rsproj")
    ctl.clear_background()
    editor.open_project(project)
    assert ctl.doc.background.enabled and ctl.doc.background.path == str(bg_path)
    assert ctl.doc.background.blur == pytest.approx(0.3)
    assert ctl.preview_background() is not None
    settle(qapp, editor)


def test_project_with_a_missing_background_opens_without_it(qapp, editor, tmp_path):
    from relight_studio.core.project_io import save_project

    path = write_background(tmp_path / "gone.jpg")
    editor.ctl.set_background(str(path))
    project = save_project(editor.ctl.doc, tmp_path / "scene.rsproj")
    path.unlink()
    editor.open_project(project)
    assert not editor.ctl.doc.background.enabled
    assert "background" in editor.status_label.text().lower()
    settle(qapp, editor)


def test_tab_bar_fits_the_side_panel(qapp, editor):
    bar = editor.tabs.tabBar()
    assert bar.sizeHint().width() <= editor.side_panel.width()
