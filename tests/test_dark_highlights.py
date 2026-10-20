"""Highlights on dark materials: smooth depth normals must not paint gray bands over hair."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.color import srgb_to_linear_np
from relight_studio.core.lights import LightType, new_light
from relight_studio.core.settings import DepthSettings, RelightSettings
from relight_studio.rendering.engine import Renderer, RenderParams
from relight_studio.rendering.scene import build_scene

from .conftest import sphere_disparity

H, W = 96, 128
FRONT = new_light(1, LightType.DIRECTIONAL, x=0.5, y=0.5, z=-1.0, intensity=1.5, shadows=False)


def two_tone(dark: float, light: float) -> np.ndarray:
    """Left half a dark material (hair), right half a light one (skin), evenly lit."""
    img = np.empty((H, W, 3), np.uint8)
    img[:, : W // 2] = round(dark * 255)
    img[:, W // 2 :] = round(light * 255)
    return img


def highlight(scene, region) -> float:
    """Linear radiance the specular term adds over a region."""
    def render(spec):
        out = Renderer().render(scene, RenderParams(lights=[FRONT], relight=RelightSettings(
            original_light=0.0, specular=spec, roughness=0.8)))
        return srgb_to_linear_np(out.astype(np.float32) / 255.0)
    return float((render(1.0) - render(0.0))[region].mean())


@pytest.fixture()
def hair_and_skin(device):
    return build_scene(two_tone(0.06, 0.7), sphere_disparity(), DepthSettings(relief=0.6, edge_mode="off"), device)


# Mirrored patches on the sphere: same normals, different material.
LEFT = (slice(42, 54), slice(52, 60))
RIGHT = (slice(42, 54), slice(68, 76))


def test_dark_material_loses_most_of_the_highlight(hair_and_skin):
    dark, light = highlight(hair_and_skin, LEFT), highlight(hair_and_skin, RIGHT)
    assert light > 0.05
    assert dark < 0.2 * light, (dark, light)


def test_evenly_dark_photo_keeps_its_highlights(device):
    # An underexposed photo is not a dark material: darkness is relative to the subject.
    scene = build_scene(two_tone(0.15, 0.15), sphere_disparity(), DepthSettings(relief=0.6, edge_mode="off"), device)
    assert highlight(scene, LEFT) == pytest.approx(highlight(scene, RIGHT), rel=0.1)
    assert highlight(scene, LEFT) > 0.02


def test_gloss_is_a_weight(hair_and_skin):
    g = hair_and_skin.gloss
    assert g.shape[-2:] == (H, W) and float(g.min()) >= 0.0 and float(g.max()) <= 1.0
