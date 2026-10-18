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


# ------------------------------------------------------------------ compositing (engine)
import torch  # noqa: E402

from relight_studio.core.settings import Adjustments, BackgroundSettings, DepthSettings, RelightSettings  # noqa: E402
from relight_studio.rendering.engine import Renderer, RenderParams  # noqa: E402
from relight_studio.rendering.scene import build_scene  # noqa: E402

SUBJECT = (210, 150, 120)
OLD_BG = (40, 200, 60)  # green screen-like old background
NEW_BG = (40, 60, 200)


def portrait(soft=False):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - 150) / 60.0) ** 2 + ((yy - 100) / 80.0) ** 2)
    alpha = np.clip((1.0 - r) * (8.0 if soft else 1000.0), 0, 1)
    image = (alpha[..., None] * np.array(SUBJECT) + (1 - alpha[..., None]) * np.array(OLD_BG)).round().astype(np.uint8)
    disp = np.where(r < 1, 0.9, 0.2).astype(np.float32)
    return image, disp, alpha


def composite(image, disp, alpha_f, bg=NEW_BG, settings=None, opacity=None, **relight):
    new_bg = np.empty_like(image)
    new_bg[:] = bg
    alpha = (alpha_f * 255).round().astype(np.uint8)
    scene = build_scene(image, disp, DepthSettings(edge_mode="matting"), torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        alpha=alpha if opacity is None else None, opacity=opacity, background=new_bg)
    params = RenderParams(relight=RelightSettings(**({"original_light": 1.0} | relight)),
                          background=settings or BackgroundSettings(enabled=True))
    return scene, Renderer().render(scene, params).astype(int)


def test_subject_stays_and_background_is_replaced():
    image, disp, alpha = portrait()
    _, out = composite(image, disp, alpha)
    assert np.abs(out[100, 150] - np.array(SUBJECT)).max() <= 2
    assert np.abs(out[5, 5] - np.array(NEW_BG)).max() <= 2


def test_hair_edges_lose_the_old_background_color():
    image, disp, alpha = portrait(soft=True)
    _, out = composite(image, disp, alpha)
    edge = (alpha > 0.25) & (alpha < 0.75)
    ideal = alpha[..., None] * np.array(SUBJECT) + (1 - alpha[..., None]) * np.array(NEW_BG)
    assert np.abs(out[edge] - ideal[edge]).mean() < 12
    assert out[edge][:, 1].mean() < ideal[edge][:, 1].mean() + 12  # no green fringe from the old background


def test_transparent_png_becomes_opaque_on_the_new_background():
    image, disp, alpha = portrait()
    opacity = (alpha * 255).astype(np.uint8)
    _, out = composite(image, disp, alpha, opacity=opacity)
    assert out.shape[2] == 3
    assert np.abs(out[5, 5] - np.array(NEW_BG)).max() <= 2


def test_background_needs_a_separated_subject():
    image, disp, _ = portrait()
    new_bg = np.full_like(image, 90)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = build_scene(image, disp, DepthSettings(edge_mode="off"), device, background=new_bg)
    assert scene.alpha is None
    relight = RelightSettings(original_light=1.0)
    with_bg = Renderer().render(scene, RenderParams(relight=relight, background=BackgroundSettings(enabled=True)))
    without = Renderer().render(scene, RenderParams(relight=relight))
    assert np.array_equal(with_bg, without)  # nothing to cut out: the photo is left as it is


def test_disabled_background_changes_nothing():
    image, disp, alpha = portrait()
    scene, out = composite(image, disp, alpha, settings=BackgroundSettings(enabled=False))
    assert np.abs(out[5, 5] - np.array(OLD_BG)).max() <= 2


def test_blur_and_brightness_of_the_background():
    image, disp, alpha = portrait()
    stripes = np.full_like(image, 50)  # mid tones: brightness can move them, blur can average them
    stripes[:, ::4] = 140
    scene = build_scene(image, disp, DepthSettings(edge_mode="matting"), torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        alpha=(alpha * 255).astype(np.uint8), background=stripes)
    render = lambda **bg: Renderer().render(scene, RenderParams(relight=RelightSettings(original_light=1.0),  # noqa: E731
                                                                 background=BackgroundSettings(enabled=True, **bg))).astype(float)
    sharp, blurred, bright = render(), render(blur=0.5), render(brightness=1.0)
    corner = (slice(0, 30), slice(0, 60))
    assert blurred[corner].std() < 0.3 * sharp[corner].std()
    assert bright[corner].mean() > sharp[corner].mean() + 10


def test_grading_applies_to_the_whole_composite():
    image, disp, alpha = portrait()
    _, plain = composite(image, disp, alpha)
    scene, _ = composite(image, disp, alpha)
    graded = Renderer().render(scene, RenderParams(relight=RelightSettings(original_light=1.0), background=BackgroundSettings(enabled=True),
                                                   adjustments=Adjustments(exposure=1.0))).astype(int)
    assert graded[5, 5].mean() > plain[5, 5].mean() + 10


def test_chunked_export_matches_preview_with_background():
    image, disp, alpha = portrait(soft=True)
    scene, _ = composite(image, disp, alpha)
    params = RenderParams(relight=RelightSettings(original_light=1.0), background=BackgroundSettings(enabled=True, blur=0.4))
    full = Renderer().render(scene, params, chunk_rows=1000).astype(int)
    chunked = Renderer().render(scene, params, chunk_rows=23).astype(int)
    assert np.abs(full - chunked).max() <= 2


def test_original_view_ignores_the_background():
    image, disp, alpha = portrait()
    scene, _ = composite(image, disp, alpha)
    original = Renderer().render(scene, RenderParams(view="original", background=BackgroundSettings(enabled=True)))
    assert np.abs(original[5, 5].astype(int) - np.array(OLD_BG)).max() <= 2
