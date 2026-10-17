"""Every engine parameter has an observable, correctly-directed effect.

These are regression tests: each one pins the direction of a control so a
refactor cannot silently invert or disconnect it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from relight_studio.core.lights import LightType, new_light
from relight_studio.core.settings import Adjustments, DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.filters import box_filter, gaussian_blur, guided_filter
from relight_studio.rendering.scene import build_scene, compute_normals, normalize_disparity

from .conftest import gray_image, sphere_disparity

H, W = 96, 128


def scene_with(device, image=None, disparity=None, offset=None, **depth):
    return build_scene(
        gray_image() if image is None else image,
        sphere_disparity() if disparity is None else disparity,
        DepthSettings(**depth),
        device,
        depth_offset=offset,
    )


def lum(scene, lights=(), view="result", adjustments=None, **relight) -> np.ndarray:
    params = RenderParams(lights=list(lights), relight=RelightSettings(**relight), view=view,
                          adjustments=adjustments or Adjustments())
    return Renderer().render(scene, params).astype(np.float32).mean(axis=2)


@pytest.fixture()
def sphere(device):
    return scene_with(device)


SPHERE_FACING_AWAY = (slice(40, 56), slice(84, 92))  # right side of the sphere
SPHERE_CENTER = (slice(44, 52), slice(60, 68))


# ------------------------------------------------------------------- depth settings
def test_normalized_depth_is_zero_near_and_one_far(device):
    disp = torch.tensor(sphere_disparity(), device=device)[None, None]
    depth = normalize_disparity(disp, DepthSettings())
    assert depth[0, 0, 48, 64] < 0.05  # sphere center is nearest
    assert depth[0, 0, 2, 2] > 0.95  # background corner is farthest


def test_invert_depth_swaps_near_and_far(device):
    disp = torch.tensor(sphere_disparity(), device=device)[None, None]
    depth = normalize_disparity(disp, DepthSettings(invert=True))
    assert depth[0, 0, 48, 64] > 0.95 and depth[0, 0, 2, 2] < 0.05


def test_near_far_remap_and_gamma(device):
    disp = torch.tensor(sphere_disparity(), device=device)[None, None]
    base = normalize_disparity(disp, DepthSettings())
    remapped = normalize_disparity(disp, DepthSettings(near=0.2, far=0.6))
    curved = normalize_disparity(disp, DepthSettings(gamma=2.0))
    mid = (base > 0.3) & (base < 0.5)
    assert torch.all(remapped[base <= 0.2] == 0) and torch.all(remapped[base >= 0.6] == 1)
    assert torch.all(curved[mid] < base[mid])


def test_relief_steepens_normals(device):
    flat = scene_with(device, relief=0.1).normals
    steep = scene_with(device, relief=1.5).normals
    tilt = lambda n: (1 + n[0, 2]).mean().item()  # 0 when facing the camera
    assert tilt(steep) > tilt(flat) * 2


def test_smoothing_reduces_normal_noise(device):
    rng = np.random.default_rng(3)
    noisy = (np.full((H, W), 0.5) + rng.normal(0, 0.02, (H, W))).astype(np.float32)
    rough = scene_with(device, disparity=noisy, smoothing=0.0).normals
    smooth = scene_with(device, disparity=noisy, smoothing=1.0).normals
    assert rough[0, 0].std() > smooth[0, 0].std()


def test_depth_offset_moves_surfaces(device):
    offset = np.zeros((H // 2, W // 2), np.float32)
    offset[:, : W // 4] = 0.3
    plain = scene_with(device).depth01
    pushed = scene_with(device, offset=offset).depth01
    assert (pushed[..., :, : W // 2 - 4] - plain[..., :, : W // 2 - 4]).mean() > 0.2
    assert torch.allclose(pushed[..., :, W // 2 + 4 :], plain[..., :, W // 2 + 4 :], atol=0.02)


def test_normals_are_unit_and_face_the_camera_on_flat_ground():
    z = torch.zeros(1, 1, 16, 16)
    n = compute_normals(z)
    assert torch.allclose(n.norm(dim=1), torch.ones(1, 16, 16))
    assert torch.allclose(n[0, 2], -torch.ones(16, 16))


# ------------------------------------------------------------------- light parameters
POINT = dict(x=0.05, y=0.5, z=0.0, intensity=2.0, radius=2.0, shadows=False)


def test_softness_brightens_the_terminator(sphere):
    hard = lum(sphere, [new_light(1, LightType.POINT, softness=0.0, **POINT)], original_light=0.0, specular=0.0)
    soft = lum(sphere, [new_light(1, LightType.POINT, softness=1.0, **POINT)], original_light=0.0, specular=0.0)
    terminator = (slice(44, 52), slice(66, 72))  # normals just past perpendicular to the light
    assert soft[terminator].mean() > hard[terminator].mean() + 5


def test_diffusion_reduces_directional_contrast(sphere):
    def contrast(diffusion):
        out = lum(sphere, [new_light(1, LightType.POINT, diffusion=diffusion, softness=0.0, **POINT)],
                  original_light=0.0, specular=0.0)
        return out[44:52, 38:46].mean() - out[SPHERE_FACING_AWAY].mean()

    assert contrast(1.0) < contrast(0.0)


def test_falloff_darkens_distant_surfaces(device):
    flat = scene_with(device, disparity=np.ones((H, W), np.float32))
    spec = dict(x=0.1, y=0.5, z=-0.2, intensity=2.0, radius=0.6, shadows=False)
    gentle = lum(flat, [new_light(1, LightType.POINT, attenuation=0.3, **spec)], original_light=0.0)
    steep = lum(flat, [new_light(1, LightType.POINT, attenuation=2.5, **spec)], original_light=0.0)
    assert gentle[:, -30:].mean() > steep[:, -30:].mean() + 5


def test_kelvin_tints_the_light(device):
    flat = scene_with(device, disparity=np.ones((H, W), np.float32))
    spec = dict(x=0.5, y=0.5, z=-0.3, intensity=1.5, shadows=False)
    params = lambda k: RenderParams(lights=[new_light(1, LightType.POINT, kelvin=k, **spec)],
                                    relight=RelightSettings(original_light=0.0))
    warm = Renderer().render(flat, params(2500)).astype(np.float32)
    cool = Renderer().render(flat, params(11000)).astype(np.float32)
    assert warm[..., 0].mean() > warm[..., 2].mean() + 10
    assert cool[..., 2].mean() > cool[..., 0].mean() + 5


def test_specular_and_roughness_shape_the_highlight(sphere):
    spec = dict(x=0.5, y=0.5, z=-1.0, intensity=1.0, radius=3.0, shadows=False)
    matte = lum(sphere, [new_light(1, LightType.POINT, specular=0.0, **spec)], original_light=0.0, specular=1.0)
    glossy = lum(sphere, [new_light(1, LightType.POINT, specular=2.0, **spec)], original_light=0.0, specular=1.0)
    assert glossy[SPHERE_CENTER].mean() > matte[SPHERE_CENTER].mean() + 5
    sharp = lum(sphere, [new_light(1, LightType.POINT, specular=2.0, **spec)], original_light=0.0,
                specular=1.0, roughness=0.15)
    broad = lum(sphere, [new_light(1, LightType.POINT, specular=2.0, **spec)], original_light=0.0,
                specular=1.0, roughness=0.8)
    assert sharp.max() - np.median(sharp) > broad.max() - np.median(broad)


def test_area_size_softens_shadows(device):
    disp = np.full((H, W), 0.1, np.float32)
    disp[30:66, 54:74] = 1.0
    scene = scene_with(device, disparity=disp, relief=0.8, smoothing=0.0)
    spec = dict(x=0.05, y=0.5, z=0.3, intensity=2.0, radius=3.0, softness=0.0)
    small = lum(scene, [new_light(1, LightType.AREA, size=0.0, **spec)], original_light=0.0, specular=0.0)
    large = lum(scene, [new_light(1, LightType.AREA, size=0.5, **spec)], original_light=0.0, specular=0.0)
    edge = (slice(40, 56), slice(76, 96))
    assert np.abs(np.diff(large[edge], axis=1)).max() < np.abs(np.diff(small[edge], axis=1)).max()


def test_directional_light_comes_from_the_gizmo_side(sphere):
    left = lum(sphere, [new_light(1, LightType.DIRECTIONAL, x=0.0, y=0.5, shadows=False)], original_light=0.0)
    right = lum(sphere, [new_light(1, LightType.DIRECTIONAL, x=1.0, y=0.5, shadows=False)], original_light=0.0)
    assert left[44:52, 38:46].mean() > left[SPHERE_FACING_AWAY].mean()
    assert right[SPHERE_FACING_AWAY].mean() > right[44:52, 38:46].mean()


def test_ambient_light_is_uniform_on_flat_ground(device):
    flat = scene_with(device, disparity=np.ones((H, W), np.float32))
    out = lum(flat, [new_light(1, LightType.AMBIENT, intensity=0.8)], original_light=0.0)
    assert out.std() < 1.0 and out.mean() > 50


def test_shadow_toggle_and_strength(device):
    disp = np.full((H, W), 0.1, np.float32)
    disp[30:66, 54:74] = 1.0
    scene = scene_with(device, disparity=disp, relief=0.8, smoothing=0.0)
    spec = dict(x=0.05, y=0.5, z=0.3, intensity=2.0, radius=3.0, softness=0.1)
    off = lum(scene, [new_light(1, LightType.POINT, shadows=False, **spec)], original_light=0.0, specular=0.0)
    zero = lum(scene, [new_light(1, LightType.POINT, shadows=True, **spec)], original_light=0.0, specular=0.0,
               shadow_strength=0.0)
    half = lum(scene, [new_light(1, LightType.POINT, shadows=True, **spec)], original_light=0.0, specular=0.0,
               shadow_strength=0.5)
    full = lum(scene, [new_light(1, LightType.POINT, shadows=True, **spec)], original_light=0.0, specular=0.0,
               shadow_strength=1.0)
    region = (slice(40, 56), slice(80, 92))
    assert np.allclose(off, zero, atol=1.0)
    assert off[region].mean() > half[region].mean() > full[region].mean()


# ------------------------------------------------------------------- lighting model
def test_original_light_scales_captured_illumination(sphere):
    assert lum(sphere, original_light=0.5).mean() < lum(sphere, original_light=1.0).mean() - 20


def test_delighting_changes_the_albedo_estimate(device):
    image = gray_image()
    image[:, : W // 2] = 60  # darker half = captured shading
    scene = scene_with(device, image=image)
    weak = lum(scene, view="albedo", delighting=0.0)
    strong = lum(scene, view="albedo", delighting=1.0)
    assert np.abs(strong[:, :50].mean() - strong[:, -50:].mean()) <= np.abs(weak[:, :50].mean() - weak[:, -50:].mean())


def test_global_ambient_and_color(device):
    flat = scene_with(device, disparity=np.ones((H, W), np.float32))
    dark = lum(flat, original_light=0.0)
    lit = lum(flat, original_light=0.0, ambient=0.8)
    assert lit.mean() > dark.mean() + 40
    params = RenderParams(relight=RelightSettings(original_light=0.0, ambient=0.8, ambient_color=(0.0, 0.0, 1.0)))
    blue = Renderer().render(flat, params).astype(np.float32)
    assert blue[..., 2].mean() > blue[..., 0].mean() + 30


def test_ambient_occlusion_darkens_cavities(device):
    disp = np.ones((H, W), np.float32)
    disp[40:56, 56:72] = 0.2  # a hole
    scene = scene_with(device, disparity=disp)
    no_ao = lum(scene, original_light=0.0, ambient=0.8, ambient_occlusion=0.0)
    ao = lum(scene, original_light=0.0, ambient=0.8, ambient_occlusion=1.0)
    assert no_ao[44:52, 60:68].mean() > ao[44:52, 60:68].mean() + 5


def test_shadow_steps_are_honored(device):
    disp = np.full((H, W), 0.1, np.float32)
    disp[30:66, 54:74] = 1.0
    scene = scene_with(device, disparity=disp, relief=0.8)
    light = [new_light(1, LightType.POINT, x=0.05, y=0.5, z=0.3, intensity=2.0, radius=3.0)]
    few = lum(scene, light, original_light=0.0, shadow_steps=4)
    many = lum(scene, light, original_light=0.0, shadow_steps=48)
    assert not np.array_equal(few, many)


# ------------------------------------------------------------------- adjustments
@pytest.mark.parametrize(
    "field,value,check",
    [
        ("contrast", 0.8, lambda base, out: out.std() > base.std()),
        ("contrast", -0.8, lambda base, out: out.std() < base.std()),
        ("highlights", -1.0, lambda base, out: out.max() < base.max()),
        ("shadows", 1.0, lambda base, out: out.min() > base.min()),
        ("whites", 1.0, lambda base, out: out.mean() > base.mean()),
        ("blacks", 1.0, lambda base, out: out.min() > base.min()),
        ("gamma", 2.0, lambda base, out: out.mean() > base.mean()),
        ("exposure", -1.0, lambda base, out: out.mean() < base.mean()),
    ],
)
def test_each_adjustment_moves_the_image(device, field, value, check):
    ramp = np.repeat(np.linspace(10, 245, W, dtype=np.uint8)[None, :, None], H, 0).repeat(3, 2)
    scene = scene_with(device, image=ramp)
    base = lum(scene)
    out = lum(scene, adjustments=Adjustments(**{field: value}))
    assert check(base, out), field


def test_tint_shifts_green_magenta(sphere):
    green = Renderer().render(sphere, RenderParams(adjustments=Adjustments(tint=-1.0))).astype(np.float32)
    magenta = Renderer().render(sphere, RenderParams(adjustments=Adjustments(tint=1.0))).astype(np.float32)
    assert green[..., 1].mean() > magenta[..., 1].mean() + 5


# ------------------------------------------------------------------- filters
def test_box_filter_matches_a_naive_mean():
    x = torch.rand(1, 1, 20, 30)
    out = box_filter(x, 2)
    assert torch.isclose(out[0, 0, 10, 10], x[0, 0, 8:13, 8:13].mean(), atol=1e-5)
    assert torch.isclose(out[0, 0, 0, 0], x[0, 0, 0:3, 0:3].mean(), atol=1e-5)  # clipped window at the border


def test_guided_filter_preserves_strong_edges_and_gaussian_smooths():
    step = torch.zeros(1, 1, 40, 40)
    step[..., 20:] = 1.0
    kept = guided_filter(step, step, 4, 1e-4)
    assert kept[..., :, :19].max() < 0.02 and kept[..., :, 21:].min() > 0.98  # edge stays sharp
    blurred = gaussian_blur(step, 3.0)
    assert 0.05 < blurred[0, 0, 5, 19] < 0.95  # plain blur spreads it
