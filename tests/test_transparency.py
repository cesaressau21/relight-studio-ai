"""Images with transparency (PNG/WebP alpha): the transparent area is not part of the scene."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from relight_studio.core.image_io import ExportOptions, load_image_with_alpha, save_image
from relight_studio.core.lights import LightType, new_light
from relight_studio.core.settings import DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

from .conftest import settle, wait_until

H, W = 160, 120


def cutout(background=(0, 0, 0)):
    """A character-like cutout: opaque ellipse with a soft edge on a transparent canvas."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - 60) / 40.0) ** 2 + ((yy - 80) / 65.0) ** 2)
    alpha = (np.clip((1.0 - r) * 12, 0, 1) * 255).astype(np.uint8)
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = (230, 120, 90)
    rgb[:, :, 1] = (120 + 60 * yy / H).astype(np.uint8)  # some shading variation
    rgb[alpha == 0] = background  # whatever the file stores under fully transparent pixels
    return rgb, alpha


def cutout_disparity():
    yy, xx = np.mgrid[0:H // 2, 0:W // 2]
    r = np.sqrt(((xx - 30) / 20.0) ** 2 + ((yy - 40) / 32.5) ** 2)
    return np.where(r < 1, 1.0 - 0.3 * r, 0.05).astype(np.float32)


def save_rgba(path, rgb, alpha):
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)
    return path


# ------------------------------------------------------------------ loading
def test_png_alpha_is_kept(tmp_path):
    rgb, alpha = cutout()
    loaded_rgb, loaded_alpha = load_image_with_alpha(save_rgba(tmp_path / "c.png", rgb, alpha))
    assert loaded_rgb.shape == (H, W, 3)
    assert np.array_equal(loaded_alpha, alpha)


def test_opaque_images_have_no_alpha(tmp_path):
    path = tmp_path / "o.png"
    Image.fromarray(np.dstack([np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 255, np.uint8)]), "RGBA").save(path)
    assert load_image_with_alpha(path)[1] is None
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "o.jpg")
    assert load_image_with_alpha(tmp_path / "o.jpg")[1] is None


def test_palette_and_gray_alpha_formats(tmp_path):
    la = Image.fromarray(np.dstack([np.full((8, 8), 200, np.uint8), np.tile([0, 255], 32).reshape(8, 8).astype(np.uint8)]), "LA")
    la.save(tmp_path / "la.png")
    rgb, alpha = load_image_with_alpha(tmp_path / "la.png")
    assert rgb.shape == (8, 8, 3) and alpha is not None and set(np.unique(alpha)) == {0, 255}
    pal = Image.new("P", (8, 8), 0)
    pal.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    pal.paste(1, (0, 0, 4, 8))
    pal.info["transparency"] = 0
    pal.save(tmp_path / "p.png", transparency=0)
    assert load_image_with_alpha(tmp_path / "p.png")[1][:, 6].max() == 0


# ------------------------------------------------------------------ scene and lighting
def relit(rgb, alpha, device, **light):
    scene = build_scene(rgb, cutout_disparity(), DepthSettings(), device, opacity=alpha)
    spec = dict(x=0.1, y=0.3, z=-0.3, intensity=2.0, radius=1.5)
    spec.update(light)
    params = RenderParams(lights=[new_light(1, LightType.POINT, **spec)], relight=RelightSettings(original_light=0.3))
    return scene, Renderer().render(scene, params)


def test_hidden_background_color_does_not_affect_the_subject(device):
    """The user's report: a PNG relit as if it had a black background."""
    black, alpha = cutout((0, 0, 0))
    white, _ = cutout((255, 255, 255))
    _, a = relit(black, alpha, device)
    _, b = relit(white, alpha, device)
    visible = alpha > 0
    assert np.abs(a[visible][:, :3].astype(int) - b[visible][:, :3].astype(int)).max() <= 2


def test_render_keeps_the_original_transparency(device):
    rgb, alpha = cutout()
    _, out = relit(rgb, alpha, device)
    assert out.shape == (H, W, 4)
    assert np.array_equal(out[..., 3], alpha)


def test_transparency_is_the_subject_outline(device):
    rgb, alpha = cutout()
    scene, _ = relit(rgb, alpha, device)
    assert scene.alpha is not None
    assert np.allclose(scene.alpha[0, 0].cpu().numpy(), alpha / 255.0, atol=1 / 255)


def test_depth_range_is_spent_on_the_visible_subject(device):
    rgb, alpha = cutout()
    scene, _ = relit(rgb, alpha, device)
    depth = scene.depth01[0, 0].cpu().numpy()
    solid = alpha == 255
    assert depth[solid].max() - depth[solid].min() > 0.6  # the subject's own relief uses the range


def test_edges_of_the_subject_are_not_darkened_by_the_hidden_background(device):
    rgb, alpha = cutout((0, 0, 0))
    _, out = relit(rgb, alpha, device)
    edge = (alpha > 60) & (alpha < 250)
    inner = alpha == 255
    # Semi-transparent edge pixels keep the subject's brightness instead of fading to black.
    assert out[edge][:, :3].mean() > 0.6 * out[inner][:, :3].mean()


def test_opaque_images_render_rgb(device):
    rgb, _ = cutout()
    scene = build_scene(rgb, cutout_disparity(), DepthSettings(), device)
    assert Renderer().render(scene, RenderParams()).shape == (H, W, 3)


# ------------------------------------------------------------------ export
@pytest.mark.parametrize("fmt,ext", [("PNG", ".png"), ("WEBP", ".webp")])
def test_export_keeps_alpha(tmp_path, fmt, ext):
    rgb, alpha = cutout()
    path = save_image(np.dstack([rgb, alpha]), tmp_path / f"o{ext}", ExportOptions(format=fmt, quality=100))
    with Image.open(path) as im:
        assert im.mode == "RGBA"
        assert np.abs(np.asarray(im)[..., 3].astype(int) - alpha).max() <= 1


def test_jpeg_export_flattens_onto_the_background_color(tmp_path):
    rgb, alpha = cutout()
    path = save_image(np.dstack([rgb, alpha]), tmp_path / "o.jpg",
                      ExportOptions(format="JPEG", quality=100, background=(1.0, 1.0, 1.0)))
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB")).astype(int)
    assert arr[2, 2].min() > 245  # transparent corner became white
    assert abs(arr[80, 60, 0] - 230) < 8  # the subject is untouched


def test_export_can_flatten_png_on_request(tmp_path):
    rgb, alpha = cutout()
    path = save_image(np.dstack([rgb, alpha]), tmp_path / "flat.png",
                      ExportOptions(format="PNG", keep_transparency=False, background=(0.0, 0.0, 1.0)))
    with Image.open(path) as im:
        assert im.mode == "RGB" and np.asarray(im)[2, 2].tolist() == [0, 0, 255]


# ------------------------------------------------------------------ editor
def test_editor_workflow_with_a_transparent_png(qapp, editor, tmp_path, monkeypatch):
    rgb, alpha = cutout()
    path = save_rgba(tmp_path / "character.png", rgb, alpha)
    matte_calls = []
    monkeypatch.setattr("relight_studio.ai.matting.matte_subject", lambda *a, **k: matte_calls.append(1))
    editor.open_path(path)
    ctl = editor.ctl
    assert ctl.doc.opacity is not None and np.array_equal(ctl.doc.opacity, alpha)
    settle(qapp, editor)
    assert wait_until(qapp, lambda: editor.canvas.result is not None and editor.canvas.result.hasAlphaChannel())
    assert editor.canvas.original.hasAlphaChannel()
    editor.run_matte()
    assert matte_calls == []  # the file's own transparency is the outline: no AI matte needed
    assert "transparency" in editor.scene_panel.edge_status.text().lower()

    from relight_studio.ui.workers import ExportWorker

    out = tmp_path / "relit.png"
    worker = ExportWorker(ctl.doc, editor._params(view="result"), ExportOptions(format="PNG"), out)
    worker.start()
    assert wait_until(qapp, worker.isFinished, timeout=60)
    with Image.open(out) as im:
        assert im.mode == "RGBA" and np.array_equal(np.asarray(im)[..., 3], alpha)


def test_project_reopens_with_transparency(qapp, editor, tmp_path, monkeypatch):
    from relight_studio.core.project_io import save_project

    rgb, alpha = cutout()
    path = save_rgba(tmp_path / "character.png", rgb, alpha)
    editor.open_path(path)
    project = save_project(editor.ctl.doc, tmp_path / "c.rsproj")
    editor.open_project(project)
    assert editor.ctl.doc.opacity is not None
    settle(qapp, editor)


def test_export_dialog_transparency_options(qapp):
    from pathlib import Path

    from relight_studio.ui.export_dialog import ExportDialog

    plain = ExportDialog(400, 300, Path("C:/x/a.png"))
    assert plain.transparency_row.isHidden()
    dialog = ExportDialog(400, 300, Path("C:/x/a.png"), has_alpha=True)
    assert not dialog.transparency_row.isHidden() and dialog.options().keep_transparency
    assert not dialog.background.isEnabled()  # PNG keeps alpha: no fill needed
    dialog.format.setCurrentIndex(dialog.format.findData("JPEG"))
    assert not dialog.transparent.isEnabled() and dialog.background.isEnabled()
    dialog.background.colorChanged.emit((0.0, 0.0, 0.0))
    dialog.background.set_color((0.0, 0.0, 0.0))
    assert dialog.options().background == (0.0, 0.0, 0.0)
