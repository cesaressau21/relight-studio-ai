"""Background replacement: analysis of a background photo and the lights proposed from it."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.background import analyze_background, fit_cover, suggest_lights
from relight_studio.core.lights import LightTarget, LightType

H, W = 200, 300


def background(base=(30, 40, 90), spot=None, spot_color=(255, 170, 80), radius=40):
    """A dim background with an optional bright light source (sun, window, lamp)."""
    img = np.empty((H, W, 3), np.float32)
    img[:] = base
    if spot is not None:
        yy, xx = np.mgrid[0:H, 0:W]
        glow = np.exp(-(((xx - spot[0] * W) ** 2 + (yy - spot[1] * H) ** 2) / (2 * radius**2)))[..., None]
        img = img * (1 - glow) + np.array(spot_color, np.float32) * glow
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ analysis
def test_bright_region_position_and_color():
    a = analyze_background(background(spot=(0.15, 0.2)))
    assert a.key_x < 0.3 and a.key_y < 0.35
    assert a.key_color[0] > a.key_color[2] + 0.2  # warm light
    assert a.ambient_color[2] > a.ambient_color[0]  # bluish surroundings
    assert a.key_strength > 0.4


def test_light_on_the_right_is_found_on_the_right():
    assert analyze_background(background(spot=(0.85, 0.5))).key_x > 0.7


def test_evenly_lit_background_has_no_strong_key():
    a = analyze_background(background(base=(128, 128, 128)))
    assert a.key_strength < 0.15


def test_brighter_background_has_a_higher_ambient_level():
    dark = analyze_background(background(base=(20, 20, 25)))
    bright = analyze_background(background(base=(200, 200, 190)))
    assert bright.ambient_level > dark.ambient_level + 0.4


def test_backlight_detection():
    assert analyze_background(background(spot=(0.5, 0.35))).backlit
    assert not analyze_background(background(spot=(0.05, 0.5))).backlit


# ------------------------------------------------------------------ suggestion
def test_suggested_key_comes_from_the_bright_side_with_its_color():
    lights, relight = suggest_lights(analyze_background(background(spot=(0.1, 0.25))))
    key = lights[0]
    assert key.type in (LightType.DIRECTIONAL, LightType.AREA)
    assert key.x < 0.5 and key.y < 0.5
    assert key.color[0] > key.color[2]
    assert relight["original_light"] < 0.5  # the subject takes the new lighting


def test_suggested_fill_uses_the_surrounding_color():
    lights, _ = suggest_lights(analyze_background(background(base=(20, 60, 140), spot=(0.8, 0.3))))
    fill = next(l for l in lights if l.type is LightType.AMBIENT)
    assert fill.color[2] > fill.color[0]


def test_backlit_background_adds_a_rim_light():
    lights, _ = suggest_lights(analyze_background(background(spot=(0.5, 0.3))))
    assert any(l.type is LightType.RIM for l in lights)


def test_soft_background_gives_a_soft_key_and_no_rim():
    lights, _ = suggest_lights(analyze_background(background(base=(150, 150, 150))))
    assert lights[0].type is LightType.AREA
    assert not any(l.type is LightType.RIM for l in lights)


def test_suggested_lights_only_touch_the_subject_and_have_ids():
    lights, _ = suggest_lights(analyze_background(background(spot=(0.3, 0.3))), start_id=5)
    assert all(l.target is LightTarget.SUBJECT for l in lights)
    assert [l.id for l in lights] == list(range(5, 5 + len(lights)))


# ------------------------------------------------------------------ fitting
@pytest.mark.parametrize("size", [(400, 200), (200, 400), (300, 300)])
def test_fit_cover_fills_the_frame_and_crops_the_center(size):
    w, h = size
    bg = background()
    bg[:, : W // 2] = (255, 0, 0)  # left half red, right half blue
    bg[:, W // 2 :] = (0, 0, 255)
    out = fit_cover(bg, w, h)
    assert out.shape == (h, w, 3) and out.dtype == np.uint8
    assert out[h // 2, 2, 0] > 200 and out[h // 2, w - 3, 2] > 200  # both halves still visible at the edges
