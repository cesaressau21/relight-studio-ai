"""Per-light targets: a light can affect everything, only the subject, or only the background."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.lights import Light, LightTarget, LightType, new_light
from relight_studio.core.settings import DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

from .conftest import settle

H, W = 120, 160


def subject_scene(device, with_alpha=True, soft_edge=False):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - 80) / 30.0) ** 2 + ((yy - 60) / 40.0) ** 2)
    truth = r < 1.0
    image = np.full((H, W, 3), 120, np.uint8)
    image[truth] = (200, 150, 120)
    disp = np.where(truth, 0.9, 0.2).astype(np.float32)
    alpha = None
    if with_alpha:
        a = np.clip((1.0 - r) * 6, 0, 1) if soft_edge else truth.astype(np.float32)
        alpha = (a * 255).astype(np.uint8)
    settings = DepthSettings(edge_mode="matting" if with_alpha else "off")
    return build_scene(image, disp, settings, device, alpha=alpha), truth, alpha


def render(scene, lights):
    return Renderer().render(scene, RenderParams(lights=lights, relight=RelightSettings(original_light=0.4))).astype(int)


def key(target="all", **kw):
    spec = dict(x=0.2, y=0.4, z=-0.4, intensity=2.5, radius=1.5)
    spec.update(kw)
    return new_light(1, LightType.POINT, target=target, **spec)


def background_far(truth, margin=4):
    import cv2

    return cv2.erode((~truth).astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)


def subject_core(truth, margin=4):
    import cv2

    return cv2.erode(truth.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)


# ------------------------------------------------------------------ model
def test_light_target_defaults_and_serialization():
    light = new_light(1, LightType.RIM, target="subject")
    assert light.target is LightTarget.SUBJECT
    assert Light.from_dict(light.to_dict()).target is LightTarget.SUBJECT
    old = light.to_dict()
    del old["target"]  # projects saved before targets existed
    assert Light.from_dict(old).target is LightTarget.ALL
    assert new_light(2, LightType.POINT).target is LightTarget.ALL


# ------------------------------------------------------------------ engine
def test_subject_only_light_leaves_the_background_untouched(device):
    scene, truth, _ = subject_scene(device)
    dark = render(scene, [])
    lit = render(scene, [key("subject")])
    bg, core = background_far(truth), subject_core(truth)
    assert np.array_equal(lit[bg], dark[bg])
    assert lit[core].mean() > dark[core].mean() + 20


def test_background_only_light_leaves_the_subject_untouched(device):
    scene, truth, _ = subject_scene(device)
    dark = render(scene, [])
    lit = render(scene, [key("background")])
    bg, core = background_far(truth), subject_core(truth)
    assert np.array_equal(lit[core], dark[core])
    assert lit[bg].mean() > dark[bg].mean() + 20


def test_all_target_lights_both(device):
    scene, truth, _ = subject_scene(device)
    dark = render(scene, [])
    lit = render(scene, [key("all")])
    assert lit[subject_core(truth)].mean() > dark[subject_core(truth)].mean() + 20
    assert lit[background_far(truth)].mean() > dark[background_far(truth)].mean() + 20


def test_subject_and_background_split_adds_up_to_all(device):
    """In linear light, subject-only + background-only equals the same light on everything."""
    import torch

    from relight_studio.rendering.engine import _region

    scene, _, _ = subject_scene(device, soft_edge=True)
    region = _region(scene, 0, H)

    def light_of(target):
        diffuse, specular = Renderer().lighting(scene, region, RenderParams(lights=[key(target)]))
        return diffuse + specular

    with torch.inference_mode():
        both = light_of("all")
        split = light_of("subject") + light_of("background")
    assert torch.allclose(both, split, atol=1e-5)  # soft edges are shared, nothing lost or doubled


@pytest.mark.parametrize("light_type", [LightType.AMBIENT, LightType.RIM, LightType.DIRECTIONAL, LightType.AREA])
def test_every_light_type_honors_its_target(device, light_type):
    scene, truth, _ = subject_scene(device)
    dark = render(scene, [])
    light = new_light(1, light_type, target="subject", x=0.9, y=0.4, intensity=3.0)
    lit = render(scene, [light])
    assert np.array_equal(lit[background_far(truth)], dark[background_far(truth)])


def test_subject_still_casts_shadows_from_a_background_light(device):
    scene, truth, _ = subject_scene(device)
    spec = dict(x=0.0, y=0.5, z=0.45, intensity=3.0, radius=3.0, softness=0.1)
    with_shadow = render(scene, [new_light(1, LightType.POINT, target="background", shadows=True, **spec)])
    no_shadow = render(scene, [new_light(1, LightType.POINT, target="background", shadows=False, **spec)])
    behind = (slice(45, 75), slice(118, 140))  # wall on the far side of the subject
    assert no_shadow[behind].mean() - with_shadow[behind].mean() > 15


def test_targets_are_ignored_without_a_separated_subject(device):
    scene, _, _ = subject_scene(device, with_alpha=False)
    assert scene.alpha is None
    assert np.array_equal(render(scene, [key("subject")]), render(scene, [key("all")]))


def test_transparent_png_subject_target(device):
    scene, truth, _ = subject_scene(device)
    image = np.full((H, W, 3), 120, np.uint8)
    opacity = (truth * 255).astype(np.uint8)
    png = build_scene(image, np.where(truth, 0.9, 0.2).astype(np.float32), DepthSettings(), device, opacity=opacity)
    lit = Renderer().render(png, RenderParams(lights=[key("subject")]))
    assert lit.shape[2] == 4 and np.array_equal(lit[..., 3], opacity)


# ------------------------------------------------------------------ editor
def test_affects_control_sets_the_target_and_is_undoable(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    panel.target_combo.setCurrentIndex(panel.target_combo.findData("subject"))
    assert ctl.selected_light().target is LightTarget.SUBJECT
    assert "Subject" in panel.list.item(0).text()
    ctl.undo()
    assert ctl.selected_light().target is LightTarget.ALL
    settle(qapp, editor)


def test_affects_hint_when_no_subject_is_separated(qapp, editor):
    panel = editor.lights_panel
    panel.target_combo.setCurrentIndex(panel.target_combo.findData("background"))
    panel.set_subject_available(False)
    assert not panel.target_hint.isHidden() and "subject" in panel.target_hint.text().lower()
    panel.set_subject_available(True)
    assert panel.target_hint.isHidden()


def test_target_survives_a_project_round_trip(qapp, editor, tmp_path):
    from relight_studio.core.project_io import build_document, read_project, save_project

    editor.ctl.update_light(editor.ctl.selected, target=LightTarget.BACKGROUND)
    data = read_project(save_project(editor.ctl.doc, tmp_path / "t.rsproj"))
    doc = build_document(data, editor.ctl.doc.image, editor.ctl.doc.image_path)
    assert doc.lights[0].target is LightTarget.BACKGROUND
    settle(qapp, editor)


def test_preset_rim_lights_only_touch_the_subject():
    from relight_studio.core.presets import load_presets

    rims = [l for p in load_presets() for l in p.build_lights() if l.type is LightType.RIM]
    assert rims and all(l.target is LightTarget.SUBJECT for l in rims)
