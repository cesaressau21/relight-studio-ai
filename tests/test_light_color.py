"""Color-aware delighting: the original light's color leaves the albedo, not only its brightness."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.color import linear_to_srgb_np as linear_to_srgb, srgb_to_linear_np as srgb_to_linear
from relight_studio.core.lights import LightType, new_light
from relight_studio.core.settings import DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

from .conftest import sphere_disparity

H, W = 96, 128
TEAL = np.array([0.15, 0.9, 1.0])
ORANGE = np.array([1.0, 0.45, 0.1])


def to_srgb8(lin: np.ndarray) -> np.ndarray:
    return (np.clip(linear_to_srgb(np.clip(lin, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8)


def sphere_mask(radius=0.35):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = (xx + 0.5 - W / 2) / H, (yy + 0.5 - H / 2) / H
    inside = x * x + y * y < (radius * 0.85) ** 2
    return inside, x / radius


def two_color_sphere(albedo=(0.5, 0.5, 0.5)):
    """Gray sphere: teal light from the left, orange light from the right."""
    inside, nx = sphere_mask()
    light = (np.clip(-nx, 0, None)[..., None] * TEAL + np.clip(nx, 0, None)[..., None] * ORANGE + 0.35)
    lin = np.array(albedo) * light
    lin[~inside] = 0.3
    return to_srgb8(lin)


def render(scene, view="result", lights=(), **relight):
    return Renderer().render(scene, RenderParams(lights=list(lights), relight=RelightSettings(**relight), view=view))


def chroma(rgb8: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb8.astype(np.float32) / 255.0)
    return lin / lin.mean(axis=-1, keepdims=True).clip(1e-4)


def color_spread(rgb8: np.ndarray, region) -> float:
    """How much the color (not the brightness) changes over a region."""
    c = chroma(rgb8)[region].reshape(-1, 3)
    return float(c.std(axis=0).mean())


@pytest.fixture()
def sphere(device):
    return build_scene(two_color_sphere(), sphere_disparity(), DepthSettings(relief=0.6), device)


# ------------------------------------------------------------------ invariants
def test_without_color_removal_full_original_light_reproduces_the_photo(device):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    scene = build_scene(image, sphere_disparity(), DepthSettings(), device)
    out = render(scene, original_light=1.0, light_color=0.0, neutral_point=(0.5, 0.5)).astype(int)
    assert np.abs(out - image).max() <= 1


def test_kept_original_light_loses_its_color_but_not_its_brightness(sphere):
    inside, _ = sphere_mask()
    photo = render(sphere, original_light=1.0)
    neutral = render(sphere, original_light=1.0, light_color=1.0)
    assert color_spread(neutral, inside) < 0.5 * color_spread(photo, inside)
    lum = lambda rgb: srgb_to_linear(rgb.astype(np.float32) / 255.0)[inside].mean(axis=-1)  # noqa: E731
    assert np.abs(lum(neutral) - lum(photo)).mean() < 0.03


def test_zero_strength_changes_nothing(sphere):
    a = render(sphere, "albedo", light_color=0.0)
    b = render(sphere, "albedo", light_color=0.0, neutral_point=(0.5, 0.5))
    assert np.array_equal(a, b)


def test_light_color_is_only_estimated_when_used(sphere):
    render(sphere, light_color=0.0)
    assert sphere._tints is None  # no cost for photos that do not use it
    render(sphere, light_color=0.5)
    assert sphere._tints is not None


def test_chunked_render_matches_full_render(sphere):
    params = RenderParams(lights=[new_light(1, LightType.POINT, x=0.2, y=0.4, intensity=2.0)],
                          relight=RelightSettings(original_light=0.3, light_color=1.0, neutral_point=(0.3, 0.5)))
    full = Renderer().render(sphere, params, chunk_rows=1000).astype(int)
    chunked = Renderer().render(sphere, params, chunk_rows=17).astype(int)
    assert np.abs(full - chunked).max() <= 1


# ------------------------------------------------------------------ removing the light color
def test_two_colored_lights_leave_the_albedo(sphere):
    inside, _ = sphere_mask()
    before = color_spread(render(sphere, "albedo"), inside)
    after = color_spread(render(sphere, "albedo", light_color=1.0), inside)
    assert after < 0.5 * before, (before, after)


def test_half_strength_removes_part_of_the_color(sphere):
    inside, _ = sphere_mask()
    spreads = [color_spread(render(sphere, "albedo", light_color=s), inside) for s in (0.0, 0.5, 1.0)]
    assert spreads[0] > spreads[1] > spreads[2]


def test_colored_gradient_on_a_flat_wall_is_removed(device):
    t = np.linspace(0, 1, W)[None, :, None]
    lin = 0.5 * ((1 - t) * TEAL + t * ORANGE)
    image = to_srgb8(np.broadcast_to(lin, (H, W, 3)))
    scene = build_scene(image, np.ones((H, W), np.float32), DepthSettings(edge_mode="off"), device)
    middle = (slice(10, H - 10), slice(10, W - 10))
    before = color_spread(render(scene, "albedo"), middle)
    after = color_spread(render(scene, "albedo", light_color=1.0), middle)
    assert after < 0.5 * before, (before, after)


def test_new_white_light_no_longer_shows_the_old_colors(sphere):
    inside, _ = sphere_mask()
    front = new_light(1, LightType.DIRECTIONAL, x=0.5, y=0.5, z=-1.0, intensity=1.5, shadows=False)
    old = render(sphere, lights=[front], original_light=0.0, specular=0.0)
    new = render(sphere, lights=[front], original_light=0.0, specular=0.0, light_color=1.0)
    assert color_spread(new, inside) < 0.5 * color_spread(old, inside)


# ------------------------------------------------------------------ object color
def test_object_color_under_white_light_is_kept(device):
    image = to_srgb8(np.broadcast_to(np.array([0.6, 0.12, 0.1]), (H, W, 3)))
    scene = build_scene(image, sphere_disparity(), DepthSettings(relief=0.6), device)
    a = render(scene, "albedo").astype(int)
    b = render(scene, "albedo", light_color=1.0).astype(int)
    assert np.abs(a - b).max() <= 3  # a red object stays red


def test_neutral_point_turns_a_global_cast_gray(device):
    teal_card = to_srgb8(np.broadcast_to(0.5 * TEAL, (H, W, 3)))
    scene = build_scene(teal_card, np.ones((H, W), np.float32), DepthSettings(edge_mode="off"), device)
    kept = chroma(render(scene, "albedo", light_color=1.0))[H // 2, W // 2]
    assert kept[2] > 1.5 * kept[0]  # without a reference the average color is object color
    gray = chroma(render(scene, "albedo", light_color=1.0, neutral_point=(0.5, 0.5)))[H // 2, W // 2]
    assert np.abs(gray - 1.0).max() < 0.08, gray


def test_neutral_point_is_saved_with_the_settings():
    rs = RelightSettings(light_color=0.8, neutral_point=(0.25, 0.75))
    restored = RelightSettings.from_dict(rs.to_dict())
    assert restored.light_color == pytest.approx(0.8) and restored.neutral_point == (0.25, 0.75)
    assert RelightSettings.from_dict({}).light_color == 0.0 and RelightSettings.from_dict({}).neutral_point is None
