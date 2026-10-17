"""Behavior of the relighting engine on synthetic geometry with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.lights import LightType, new_light
from relight_studio.core.settings import Adjustments, DepthSettings, MaskSettings, RelightSettings
from relight_studio.rendering.engine import VIEW_MODES, Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

from .conftest import gray_image, sphere_disparity


def render(scene, lights=(), **relight):
    params = RenderParams(lights=list(lights), relight=RelightSettings(**relight))
    return Renderer().render(scene, params).astype(np.float32)


@pytest.fixture()
def sphere_scene(device):
    return build_scene(gray_image(), sphere_disparity(), DepthSettings(relief=0.6), device)


@pytest.fixture()
def flat_scene(device):
    return build_scene(gray_image(), np.ones((96, 128), np.float32), DepthSettings(), device)


def test_no_lights_and_full_original_light_reproduces_the_photo(device):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    scene = build_scene(image, sphere_disparity(), DepthSettings(), device)
    out = render(scene, original_light=1.0)
    assert np.abs(out - image).max() <= 1.0


def test_original_photo_is_never_modified(device):
    image = gray_image()
    before = image.copy()
    scene = build_scene(image, sphere_disparity(), DepthSettings(), device)
    render(scene, [new_light(1, LightType.POINT, intensity=3.0)], original_light=0.2)
    assert np.array_equal(image, before)


def test_side_light_follows_estimated_geometry(sphere_scene):
    left = new_light(1, LightType.POINT, x=0.0, y=0.5, z=0.0, intensity=2.0, radius=2.0, shadows=False)
    out = render(sphere_scene, [left], original_light=0.0, specular=0.0).mean(axis=2)
    cx, cy, r = 64, 48, 0.35 * 96
    lit_side = out[cy - 4 : cy + 4, int(cx - r * 0.8) : int(cx - r * 0.5)].mean()
    far_side = out[cy - 4 : cy + 4, int(cx + r * 0.5) : int(cx + r * 0.8)].mean()
    assert lit_side > far_side + 20, (lit_side, far_side)


def test_depth_map_changes_the_lighting(sphere_scene, flat_scene):
    light = new_light(1, LightType.POINT, x=0.1, y=0.5, z=0.0, intensity=2.0, radius=2.0, shadows=False)
    sphere = render(sphere_scene, [light], original_light=0.0, specular=0.0)
    flat = render(flat_scene, [light], original_light=0.0, specular=0.0)
    region = (slice(36, 60), slice(70, 90))  # right half of the sphere, facing away from the light
    assert flat[region].mean() - sphere[region].mean() > 15


def test_moving_the_light_moves_the_highlight(flat_scene):
    a = render(flat_scene, [new_light(1, LightType.POINT, x=0.15, y=0.5, z=-0.1, radius=0.3)], original_light=0.0)
    b = render(flat_scene, [new_light(1, LightType.POINT, x=0.85, y=0.5, z=-0.1, radius=0.3)], original_light=0.0)
    assert a[:, :40].mean() > a[:, -40:].mean()
    assert b[:, -40:].mean() > b[:, :40].mean()


def test_light_depth_and_radius_change_the_result(flat_scene):
    near = render(flat_scene, [new_light(1, LightType.POINT, x=0.5, y=0.5, z=0.8, radius=0.3)], original_light=0.0)
    far = render(flat_scene, [new_light(1, LightType.POINT, x=0.5, y=0.5, z=-1.0, radius=0.3)], original_light=0.0)
    wide = render(flat_scene, [new_light(1, LightType.POINT, x=0.5, y=0.5, z=-1.0, radius=2.0)], original_light=0.0)
    assert near.mean() > far.mean()
    assert wide.mean() > far.mean()


def test_intensity_and_color_are_applied(flat_scene):
    base = dict(x=0.5, y=0.5, z=-0.3, shadows=False)
    dim = render(flat_scene, [new_light(1, LightType.POINT, intensity=0.5, **base)], original_light=0.0)
    bright = render(flat_scene, [new_light(1, LightType.POINT, intensity=2.0, **base)], original_light=0.0)
    red = render(flat_scene, [new_light(1, LightType.POINT, color=(1.0, 0.0, 0.0), **base)], original_light=0.0)
    assert bright.mean() > dim.mean()
    assert red[..., 0].mean() > red[..., 1].mean() + 20


def test_foreground_object_casts_a_shadow_on_the_background(device):
    disp = np.full((96, 128), 0.1, np.float32)
    disp[30:66, 54:74] = 1.0  # a box in front of a wall
    scene = build_scene(gray_image(), disp, DepthSettings(relief=0.8, smoothing=0.0), device)
    light = dict(x=0.05, y=0.5, z=0.3, intensity=2.0, radius=3.0, softness=0.1)
    lit = render(scene, [new_light(1, LightType.POINT, shadows=False, **light)], original_light=0.0, specular=0.0)
    shadowed = render(scene, [new_light(1, LightType.POINT, shadows=True, **light)], original_light=0.0,
                      specular=0.0, shadow_strength=1.0)
    behind = (slice(40, 56), slice(80, 92))  # wall area on the far side of the box
    assert lit[behind].mean() - shadowed[behind].mean() > 20
    front = (slice(40, 56), slice(20, 40))  # wall area facing the light, unoccluded
    assert abs(lit[front].mean() - shadowed[front].mean()) < 6


@pytest.mark.parametrize("light_type", list(LightType))
def test_every_light_type_contributes(sphere_scene, light_type):
    dark = render(sphere_scene, [], original_light=0.0)
    lit = render(sphere_scene, [new_light(1, light_type, x=0.7, y=0.3)], original_light=0.0)
    assert lit.mean() > dark.mean() + 1


def test_disabled_light_has_no_effect(sphere_scene):
    off = new_light(1, LightType.POINT, enabled=False, intensity=5.0)
    assert np.array_equal(render(sphere_scene, [off]), render(sphere_scene, []))


def test_rim_light_hits_silhouettes_more_than_the_center(sphere_scene):
    rim = new_light(1, LightType.RIM, x=1.2, y=0.5, z=1.2, intensity=3.0, radius=3.0)
    out = render(sphere_scene, [rim], original_light=0.0).mean(axis=2)
    edge = out[44:52, 88:94].mean()  # right rim of the sphere
    center = out[44:52, 60:68].mean()
    assert edge > center + 10


def test_zero_mask_keeps_the_original(device):
    image = gray_image(value=90)
    mask = np.zeros((48, 64), np.uint8)
    scene = build_scene(image, sphere_disparity(), DepthSettings(), device, mask=mask)
    params = RenderParams(lights=[new_light(1, LightType.POINT, intensity=4.0)], relight=RelightSettings(original_light=0.1),
                          mask_settings=MaskSettings(enabled=True))
    out = Renderer().render(scene, params).astype(np.float32)
    assert np.abs(out - image).max() <= 1.0
    params.mask_settings.invert = True
    assert np.abs(Renderer().render(scene, params).astype(np.float32) - image).max() > 10


def test_chunked_render_matches_single_pass(sphere_scene):
    params = RenderParams(lights=[new_light(1, LightType.AREA), new_light(2, LightType.DIRECTIONAL)])
    full = Renderer().render(sphere_scene, params, chunk_rows=1000)
    chunked = Renderer().render(sphere_scene, params, chunk_rows=17)
    assert np.abs(full.astype(int) - chunked.astype(int)).max() <= 2


@pytest.mark.parametrize("view", VIEW_MODES)
def test_view_modes_render(sphere_scene, view):
    out = Renderer().render(sphere_scene, RenderParams(lights=[new_light(1, LightType.POINT)], view=view))
    assert out.shape == (96, 128, 3) and out.dtype == np.uint8


def test_adjustments(sphere_scene):
    plain = Renderer().render(sphere_scene, RenderParams()).astype(np.float32)
    brighter = Renderer().render(sphere_scene, RenderParams(adjustments=Adjustments(exposure=1.0))).astype(np.float32)
    gray = Renderer().render(sphere_scene, RenderParams(adjustments=Adjustments(saturation=-1.0)))
    warm = Renderer().render(sphere_scene, RenderParams(adjustments=Adjustments(temperature=1.0))).astype(np.float32)
    assert brighter.mean() > plain.mean() + 10
    assert np.abs(gray[..., 0].astype(int) - gray[..., 2].astype(int)).max() <= 1
    assert warm[..., 0].mean() > warm[..., 2].mean()


def test_highlight_rolloff_is_identity_in_range_and_bounded(sphere_scene):
    hot = render(sphere_scene, [new_light(1, LightType.POINT, intensity=10.0, z=-0.2, x=0.5, y=0.5)])
    assert hot.max() <= 255
    calm = render(sphere_scene, [], original_light=1.0)
    assert np.allclose(calm, 140, atol=1.0)
