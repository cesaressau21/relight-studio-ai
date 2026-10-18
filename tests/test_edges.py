"""Silhouette edges: lighting must follow the true outline, not the blurry depth edge.

Monocular depth is low resolution and soft at object boundaries. These tests
build a scene whose depth edge is blurred and dilated relative to the real
silhouette (as Depth Anything produces) and check that the background next to
the subject is not lit or shaded as if it were part of the subject.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from relight_studio.core.lights import LightType, new_light
from relight_studio.core.settings import DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

H, W = 200, 300


def subject_scene():
    """A disc-shaped subject in front of a wall, plus a depth map that is wrong at the edge."""
    yy, xx = np.mgrid[0:H, 0:W]
    truth = ((xx - 150) ** 2 + (yy - 100) ** 2) < 60**2
    rng = np.random.default_rng(0)
    image = np.empty((H, W, 3), np.uint8)
    image[:] = (70, 140, 200)  # blue wall
    image[truth] = (225, 160, 120)  # skin-colored subject
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    # Depth as a network would see it: bulging subject, edge dilated by 6 px and blurred.
    bulge = np.sqrt(np.clip(1 - ((xx - 150) ** 2 + (yy - 100) ** 2) / 66**2, 0, 1))
    wide = ((xx - 150) ** 2 + (yy - 100) ** 2) < 66**2
    disp = np.where(wide, 0.7 + 0.3 * bulge, 0.15).astype(np.float32)
    disp = cv2.GaussianBlur(disp, (0, 0), 4)
    small = cv2.resize(disp, (W // 3, H // 3), interpolation=cv2.INTER_AREA)  # low-res network output
    return image, small, truth


def band(truth: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Background ring between ``inner`` and ``outer`` pixels outside the silhouette."""
    dist = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 5)
    return (dist > inner) & (dist <= outer)


def wall_only(image: np.ndarray, device) -> np.ndarray:
    """Lighting of the same wall with no subject: what the background next to it should get."""
    flat = np.full((H // 3, W // 3), 0.15, np.float32)
    return lighting(build_scene(image, flat, DepthSettings(edge_mode="off"), device))


def halo(scene, image, truth, device, inner: int = 2, outer: int = 8) -> float:
    ring = band(truth, inner, outer)
    return float(np.abs(lighting(scene)[ring] - wall_only(image, device)[ring]).mean())


def lighting(scene, view="lighting"):
    light = new_light(1, LightType.POINT, x=0.95, y=0.5, z=0.1, intensity=2.0, radius=2.0, shadows=False)
    params = RenderParams(lights=[light], relight=RelightSettings(original_light=0.0, specular=0.0), view=view)
    return Renderer().render(scene, params).astype(np.float32).mean(axis=2)


@pytest.fixture(scope="module")
def data():
    return subject_scene()


def test_background_next_to_subject_is_not_lit_as_subject(data, device):
    image, disp, truth = data
    scene = build_scene(image, disp, DepthSettings(edge_mode="guided"), device)
    error = halo(scene, image, truth, device)
    assert error < 4.0, f"halo of {error:.1f} levels on the background next to the subject"


def test_background_normals_near_the_silhouette_face_the_camera(data, device):
    image, disp, truth = data
    scene = build_scene(image, disp, DepthSettings(edge_mode="guided"), device)
    nz = scene.normals[0, 2].cpu().numpy()
    assert nz[band(truth, 2, 8)].mean() < -0.97


def test_refinement_beats_the_unrefined_edge(data, device):
    image, disp, truth = data
    off = halo(build_scene(image, disp, DepthSettings(edge_mode="off"), device), image, truth, device)
    on = halo(build_scene(image, disp, DepthSettings(edge_mode="guided"), device), image, truth, device)
    assert on < 0.1 * off, (on, off)


def test_subject_rim_is_lit_on_the_side_facing_the_light(data, device):
    image, disp, truth = data
    lit = lighting(build_scene(image, disp, DepthSettings(edge_mode="guided"), device))
    inner_rim = truth & ~cv2.erode(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    right, left = inner_rim.copy(), inner_rim.copy()
    right[:, :150], left[:, 150:] = False, False
    assert lit[right].mean() > lit[left].mean() + 60  # the light is on the right


def test_subject_alpha_follows_the_true_silhouette(data, device):
    image, disp, truth = data
    scene = build_scene(image, disp, DepthSettings(edge_mode="guided"), device)
    alpha = scene.alpha[0, 0].cpu().numpy()
    predicted = alpha > 0.5
    iou = (predicted & truth).sum() / (predicted | truth).sum()
    assert iou > 0.95


def test_subject_keeps_its_own_relief(data, device):
    image, disp, truth = data
    scene = build_scene(image, disp, DepthSettings(edge_mode="guided"), device)
    nz = scene.normals[0, 2].cpu().numpy()
    inner_rim = truth & ~cv2.erode(truth.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert nz[inner_rim].mean() > -0.95  # the subject's edge still curves away


def test_external_alpha_is_used_when_provided(data, device):
    image, disp, truth = data
    alpha = (truth * 255).astype(np.uint8)
    scene = build_scene(image, disp, DepthSettings(edge_mode="matting"), device, alpha=alpha)
    assert np.array_equal(scene.alpha[0, 0].cpu().numpy() > 0.5, truth)


def test_matting_mode_without_alpha_falls_back_to_guided(data, device):
    image, disp, _ = data
    a = build_scene(image, disp, DepthSettings(edge_mode="matting"), device).alpha
    b = build_scene(image, disp, DepthSettings(edge_mode="guided"), device).alpha
    assert torch.allclose(a, b)


def test_edge_mode_off_has_no_alpha(data, device):
    image, disp, _ = data
    assert build_scene(image, disp, DepthSettings(edge_mode="off"), device).alpha is None


def test_scene_without_a_distinct_subject_is_left_alone(device):
    ramp = np.tile(np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None], (1, W))  # receding floor
    image = np.full((H, W, 3), 128, np.uint8)
    scene = build_scene(image, ramp, DepthSettings(edge_mode="guided"), device)
    assert scene.alpha is None


# ------------------------------------------------------------------ helpers
def test_trimap_from_depth(device):
    from relight_studio.rendering.edges import trimap_from_depth

    depth = torch.ones(1, 1, 100, 100, device=device)
    depth[..., 30:70, 30:70] = 0.0
    trimap = trimap_from_depth(depth, threshold=0.5, band_px=5)[0, 0].cpu().numpy()
    assert trimap[50, 50] == 1.0 and trimap[5, 5] == 0.0
    assert trimap[50, 30] == 0.5 and trimap[50, 27] == 0.5  # unknown band straddles the edge


def test_push_pull_fill(device):
    from relight_studio.rendering.edges import fill_from

    values = torch.zeros(1, 1, 64, 64, device=device)
    valid = torch.zeros_like(values)
    values[..., :, :20] = 0.3
    valid[..., :, :20] = 1.0
    filled = fill_from(values, valid)
    assert torch.allclose(filled[..., :, :20], values[..., :, :20])
    assert torch.allclose(filled, torch.full_like(filled, 0.3), atol=1e-4)


def test_subject_threshold_detection(device):
    from relight_studio.rendering.edges import subject_threshold

    two_levels = torch.cat([torch.zeros(500), torch.ones(500)]).view(1, 1, 25, 40).to(device)
    t, separable = subject_threshold(two_levels)
    assert 0.0 < t < 1.0 and separable
    ramp = torch.linspace(0, 1, 1000).view(1, 1, 25, 40).to(device)
    assert not subject_threshold(ramp)[1]


def test_alpha_upsampling_keeps_edges_sharp(device):
    from relight_studio.rendering.edges import upsample_alpha

    image = torch.zeros(1, 3, 200, 200, device=device)
    image[..., 100:] = 1.0
    small = torch.zeros(1, 1, 50, 50, device=device)
    small[..., 25:] = 1.0
    big = upsample_alpha(small, image)[0, 0]
    assert big[:, 97].max() < 0.1 and big[:, 103].min() > 0.9


def test_subject_in_front_of_a_sloped_wall_is_detected(device):
    """Real backgrounds are rarely flat: a receding wall must not hide a clear subject."""
    from relight_studio.rendering.edges import subject_threshold

    yy, xx = np.mgrid[0:H, 0:W]
    wall = 0.55 + 0.45 * xx / W  # depth ramps across the frame
    disc = ((xx - 150) ** 2 + (yy - 110) ** 2) < 70**2
    relief = 0.25 * ((xx - 150) ** 2 + (yy - 110) ** 2) / 70**2
    depth = torch.tensor(np.where(disc, 0.05 + relief, wall), dtype=torch.float32, device=device)[None, None]
    threshold, separable = subject_threshold(depth)
    assert separable and 0.3 < threshold < 0.6


def test_light_pattern_does_not_draw_a_line_along_soft_silhouettes(data, device):
    """A soft matte mixes subject and background; so must the projected pattern.

    Edge pixels have a depth between both layers. Projecting the pattern at that
    in-between depth sweeps through several stripes within 1-2 px and draws a
    thin dark/bright line along the outline.
    """
    image, disp, truth = data
    alpha = cv2.GaussianBlur(truth.astype(np.float32), (0, 0), 1.2)
    scene = build_scene(image, disp, DepthSettings(edge_mode="matting"), device,
                        alpha=(alpha * 255 + 0.5).astype(np.uint8))
    light = new_light(1, LightType.DIRECTIONAL, x=0.05, y=0.1, z=-0.2, intensity=1.3, shadows=False,
                      gobo="blinds", gobo_size=0.07, gobo_rotation=35, gobo_softness=0.12, gobo_strength=0.95)
    params = RenderParams(lights=[light], relight=RelightSettings(original_light=0.0, specular=0.0), view="lighting")
    lit = Renderer().render(scene, params).astype(np.float32).mean(axis=2)
    edge = (alpha > 0.05) & (alpha < 0.95)
    # Each edge pixel must lie between the light received by nearby pure subject and pure background.
    kernel = np.ones((9, 9), np.uint8)
    lo = cv2.erode(np.where(edge, 255.0, lit).astype(np.float32), kernel)
    hi = cv2.dilate(np.where(edge, 0.0, lit).astype(np.float32), kernel)
    outside = np.maximum(lo - lit, lit - hi)[edge]
    assert outside.max() < 25, f"silhouette pixels {outside.max():.0f} levels outside their neighbors' lighting"





def test_light_pattern_does_not_draw_a_line_along_depth_steps_inside_the_subject(data, device):
    """A depth step inside an opaque subject (a scarf over the neck) gets no line either.

    Upsampled monocular depth turns the step into a ramp of a few pixels. Projecting the
    pattern at the in-between depths sweeps through several stripes within 1-2 px and
    draws a thin bright line along the step, although the matte is 1 on both sides.
    """
    image, _, truth = data
    yy, xx = np.mgrid[0:H, 0:W]
    bulge = np.sqrt(np.clip(1 - ((xx - 150) ** 2 + (yy - 100) ** 2) / 66**2, 0, 1))
    wide = ((xx - 150) ** 2 + (yy - 100) ** 2) < 66**2
    step_y = 105  # the lower part of the subject (the scarf) is nearer
    disp = np.where(wide, 0.6 + 0.2 * bulge + 0.2 * (yy >= step_y), 0.15).astype(np.float32)
    # Network depth at half resolution with a crisp step: upsampled, a ramp of ~2 px as at ~1000 px.
    small = cv2.resize(disp, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    scene = build_scene(image, small, DepthSettings(edge_mode="matting"), device, alpha=(truth * 255).astype(np.uint8))

    def linear_lighting(**gobo):
        light = new_light(1, LightType.DIRECTIONAL, x=0.05, y=0.1, z=-0.2, intensity=1.3, shadows=False, **gobo)
        params = RenderParams(lights=[light], relight=RelightSettings(original_light=0.0, specular=0.0),
                              view="lighting")
        srgb = Renderer().render(scene, params).astype(np.float32).mean(axis=2) / 255.0
        return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

    plain = linear_lighting()
    # Stripes cross the step, so on each side the pattern hardly changes across it.
    patterned = linear_lighting(gobo="blinds", gobo_size=0.07, gobo_rotation=0, gobo_softness=0.12,
                                gobo_strength=0.95)
    passed = patterned / np.maximum(plain, 1e-3)  # the ramp's own shading cancels out
    # Each pixel on the ramp must get the pattern of one side of the step, or a mix of both.
    kernel = np.ones((1, 3), np.uint8)
    lo = cv2.erode(passed.astype(np.float32), kernel)
    hi = cv2.dilate(passed.astype(np.float32), kernel)
    far_lo, far_hi, near_lo, near_hi = lo[step_y - 7], hi[step_y - 7], lo[step_y + 6], hi[step_y + 6]
    worst = 0.0
    for y in range(step_y - 3, step_y + 3):
        for x in range(110, 191):
            assert truth[y - 7 : y + 8, x].all() and plain[y, x] > 0.05
            outside = max(min(far_lo[x], near_lo[x]) - passed[y, x], passed[y, x] - max(far_hi[x], near_hi[x]))
            worst = max(worst, outside)
    assert worst < 0.1, f"pattern on the depth step {worst:.2f} outside what either side gets"
