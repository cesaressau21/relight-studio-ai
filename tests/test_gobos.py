"""Light patterns (gobos): blinds, window, leaves, dots or a custom image projected by a light."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from relight_studio.core.lights import GOBO_PATTERNS, Light, LightType, new_light
from relight_studio.core.settings import DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

from .conftest import settle, sphere_disparity

H, W = 120, 160


@pytest.fixture()
def wall(device):
    return build_scene(np.full((H, W, 3), 150, np.uint8), np.ones((H, W), np.float32),
                       DepthSettings(edge_mode="off"), device)


def lit(scene, **light):
    light_type = light.pop("type", LightType.POINT)
    spec = dict(x=0.5, y=0.5, z=-0.8, intensity=2.5, radius=3.0, shadows=False, specular=0.0)
    spec.update(light)
    params = RenderParams(lights=[new_light(1, light_type, **spec)],
                          relight=RelightSettings(original_light=0.0, specular=0.0))
    return Renderer().render(scene, params).astype(np.float32).mean(axis=2)


def roughness(img, axis):
    return np.abs(np.diff(img, axis=axis)).mean()


def crossings(profile):
    centered = profile - profile.mean()
    return int((np.diff(np.sign(centered)) != 0).sum())


# ------------------------------------------------------------------ model
def test_gobo_fields_default_and_round_trip():
    light = new_light(1, LightType.POINT)
    assert light.gobo == "none"
    light = new_light(1, LightType.POINT, gobo="blinds", gobo_size=0.2, gobo_rotation=30.0,
                      gobo_softness=0.4, gobo_strength=0.8)
    assert Light.from_dict(light.to_dict()) == light
    assert set(GOBO_PATTERNS) >= {"none", "blinds", "window", "leaves", "dots", "custom"}


def test_unknown_gobo_falls_back_to_none():
    assert new_light(1, LightType.POINT, gobo="spaceship").gobo == "none"


# ------------------------------------------------------------------ patterns
@pytest.mark.parametrize("pattern", ["blinds", "window", "leaves", "dots"])
def test_each_pattern_breaks_up_the_light(wall, pattern):
    plain = lit(wall)
    patterned = lit(wall, gobo=pattern)
    assert patterned.std() > plain.std() + 10, pattern
    assert patterned.mean() < plain.mean()  # the pattern blocks part of the light


def test_no_gobo_matches_the_plain_light(wall):
    assert np.array_equal(lit(wall, gobo="none"), lit(wall))


def test_zero_strength_disables_the_pattern(wall):
    assert np.abs(lit(wall, gobo="blinds", gobo_strength=0.0) - lit(wall)).max() <= 1


def test_blinds_follow_the_rotation(wall):
    vertical = lit(wall, gobo="blinds", gobo_rotation=0.0)
    horizontal = lit(wall, gobo="blinds", gobo_rotation=90.0)
    assert roughness(vertical, axis=1) > 3 * roughness(vertical, axis=0)
    assert roughness(horizontal, axis=0) > 3 * roughness(horizontal, axis=1)


def test_size_changes_the_number_of_stripes(wall):
    small = lit(wall, gobo="blinds", gobo_size=0.06)[H // 2]
    large = lit(wall, gobo="blinds", gobo_size=0.25)[H // 2]
    assert crossings(small) > 2 * crossings(large)


def test_softness_blurs_the_pattern_edges(wall):
    sharp = lit(wall, gobo="blinds", gobo_softness=0.0)
    soft = lit(wall, gobo="blinds", gobo_softness=1.0)
    assert np.abs(np.diff(soft, axis=1)).max() < 0.6 * np.abs(np.diff(sharp, axis=1)).max()


def test_pattern_moves_with_the_light(wall):
    a = lit(wall, gobo="blinds", x=0.4)[H // 2]
    b = lit(wall, gobo="blinds", x=0.6)[H // 2]
    assert not np.allclose(a, b, atol=3)


def test_pattern_wraps_around_the_geometry(device, wall):
    sphere = build_scene(np.full((H, W, 3), 150, np.uint8), sphere_disparity(H, W),
                         DepthSettings(edge_mode="off"), device)
    on_wall = lit(wall, gobo="blinds")
    on_sphere = lit(sphere, gobo="blinds")
    region = (slice(40, 80), slice(50, 110))
    flat_stripes = (on_wall[region] > on_wall.mean()).astype(int)
    sphere_stripes = (on_sphere[region] > on_sphere[region].mean()).astype(int)
    assert (flat_stripes != sphere_stripes).mean() > 0.1  # stripes bend over the 3D shape


def test_directional_and_area_lights_project_patterns(wall):
    assert lit(wall, type=LightType.DIRECTIONAL, gobo="window", x=0.2, y=0.2).std() > 10
    assert lit(wall, type=LightType.AREA, gobo="leaves").std() > 10


@pytest.mark.parametrize("light_type", [LightType.AMBIENT, LightType.RIM])
def test_patterns_do_not_apply_to_ambient_and_rim(wall, light_type):
    assert np.array_equal(lit(wall, type=light_type, gobo="blinds"), lit(wall, type=light_type))


# ------------------------------------------------------------------ custom image
def test_custom_image_gobo(wall, tmp_path):
    stripes = np.zeros((64, 64), np.uint8)
    stripes[:, ::16] = 255
    stripes[:, 1::16] = 255
    path = tmp_path / "gobo.png"
    Image.fromarray(stripes).save(path)
    custom = lit(wall, gobo="custom", gobo_image=str(path), gobo_size=0.3)
    assert custom.std() > 10
    assert roughness(custom, axis=1) > 3 * roughness(custom, axis=0)  # vertical stripes from the image


def test_missing_custom_image_is_harmless(wall, tmp_path):
    assert np.array_equal(lit(wall, gobo="custom", gobo_image=str(tmp_path / "missing.png")), lit(wall))


# ------------------------------------------------------------------ editor
def test_pattern_controls(qapp, editor, tmp_path, monkeypatch):
    panel, ctl = editor.lights_panel, editor.ctl
    panel.gobo_combo.setCurrentIndex(panel.gobo_combo.findData("window"))
    assert ctl.selected_light().gobo == "window"
    for key, value in (("gobo_size", 0.3), ("gobo_rotation", 45.0), ("gobo_softness", 0.5), ("gobo_strength", 0.7)):
        panel.rows[key].spin.setValue(value)
        assert getattr(ctl.selected_light(), key) == pytest.approx(value, abs=0.01), key
    image = tmp_path / "pattern.png"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(image)
    monkeypatch.setattr("relight_studio.ui.panels.QFileDialog.getOpenFileName", lambda *a, **k: (str(image), ""))
    panel.gobo_combo.setCurrentIndex(panel.gobo_combo.findData("custom"))
    assert ctl.selected_light().gobo == "custom" and ctl.selected_light().gobo_image == str(image)
    ctl.undo()
    assert ctl.selected_light().gobo == "window"
    settle(qapp, editor)


def test_cancelling_the_custom_image_dialog_keeps_the_pattern(qapp, editor, monkeypatch):
    panel, ctl = editor.lights_panel, editor.ctl
    panel.gobo_combo.setCurrentIndex(panel.gobo_combo.findData("dots"))
    monkeypatch.setattr("relight_studio.ui.panels.QFileDialog.getOpenFileName", lambda *a, **k: ("", ""))
    panel.gobo_combo.setCurrentIndex(panel.gobo_combo.findData("custom"))
    assert ctl.selected_light().gobo == "dots"
    assert panel.gobo_combo.currentData() == "dots"


def test_pattern_controls_disable_for_rim_and_ambient(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    ctl.add_light(LightType.RIM)
    assert not panel.gobo_combo.isEnabled()
    ctl.add_light(LightType.POINT)
    assert panel.gobo_combo.isEnabled()
    assert not panel.rows["gobo_size"].isEnabled()  # no pattern selected yet
    panel.gobo_combo.setCurrentIndex(panel.gobo_combo.findData("blinds"))
    assert panel.rows["gobo_size"].isEnabled()


def test_gobo_presets_exist():
    from relight_studio.core.presets import load_presets

    with_patterns = {p.name: [l.gobo for l in p.build_lights() if l.gobo != "none"] for p in load_presets()}
    assert with_patterns.get("Venetian Blinds") and with_patterns.get("Dappled Leaves")


@pytest.mark.parametrize("pattern,expected", [("blinds", 0.55), ("leaves", 0.6), ("dots", 0.28)])
def test_patterns_are_consistent_on_both_sides_of_the_axis(pattern, expected):
    """Pattern coverage must not depend on the sign of the projected coordinates."""
    import torch

    from relight_studio.rendering.gobos import _pattern

    grid = torch.linspace(0, 30, 600)
    s, t = torch.meshgrid(grid, grid, indexing="ij")
    positive = _pattern(pattern, s[None, None], t[None, None], 0.02).mean().item()
    negative = _pattern(pattern, -s[None, None] - 0.37, -t[None, None] - 0.61, 0.02).mean().item()
    assert positive == pytest.approx(expected, abs=0.1)
    assert negative == pytest.approx(expected, abs=0.1)
