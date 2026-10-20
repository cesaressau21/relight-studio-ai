"""Background replacement helpers: fitting a background photo and matching its lighting.

The analysis is deliberately simple and explainable: the brightest region of
the background gives the key light's direction and color, the rest of the
image gives the fill, and a bright source behind the subject adds a rim light.
The proposed lights are a starting point the user can adjust.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import RGB
from .lights import Light, LightTarget, LightType, new_light

ANALYSIS_WIDTH = 160
STRONG_KEY = 0.35  # key strength above which the source reads as a directional light (sun, lamp)


@dataclass(frozen=True)
class LightingAnalysis:
    key_x: float  # normalized position of the dominant light region (0..1)
    key_y: float
    key_color: RGB  # chromaticity of that region (max channel = 1)
    key_strength: float  # 0 = evenly lit, 1 = a single strong source
    ambient_color: RGB  # chromaticity of the surroundings (max channel = 1)
    ambient_level: float  # mean brightness 0..1
    backlit: bool  # the source sits behind the subject (upper middle of the frame)


def fit_cover(background: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``background`` to cover ``width`` x ``height`` and crop the center."""
    bh, bw = background.shape[:2]
    scale = max(width / bw, height / bh)
    sw, sh = max(width, round(bw * scale)), max(height, round(bh * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(np.ascontiguousarray(background[..., :3]), (sw, sh), interpolation=interp)
    x0, y0 = (sw - width) // 2, (sh - height) // 2
    return np.ascontiguousarray(scaled[y0 : y0 + height, x0 : x0 + width])


def _chroma(color: np.ndarray) -> RGB:
    peak = float(color.max())
    if peak <= 1e-6:
        return (1.0, 1.0, 1.0)
    return tuple(float(c) for c in np.clip(color / peak, 0.0, 1.0))  # type: ignore[return-value]


def analyze_background(rgb: np.ndarray) -> LightingAnalysis:
    h, w = rgb.shape[:2]
    small_h = max(1, round(h * ANALYSIS_WIDTH / w))
    small = cv2.resize(np.ascontiguousarray(rgb[..., :3]), (ANALYSIS_WIDTH, small_h), interpolation=cv2.INTER_AREA)
    f = small.astype(np.float32) / 255.0
    lum = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]

    bright_cut = float(np.quantile(lum, 0.9))
    weights = np.clip(lum - bright_cut, 0.0, None)
    ys, xs = np.mgrid[0:small_h, 0:ANALYSIS_WIDTH]
    if weights.sum() > 1e-6:
        key_x = float((weights * (xs + 0.5)).sum() / weights.sum() / ANALYSIS_WIDTH)
        key_y = float((weights * (ys + 0.5)).sum() / weights.sum() / small_h)
        key_color = (weights[..., None] * f).sum(axis=(0, 1)) / weights.sum()
    else:  # evenly lit: no dominant source
        key_x, key_y = 0.5, 0.3
        key_color = f.reshape(-1, 3).mean(axis=0)

    top = lum >= float(np.quantile(lum, 0.97))
    top_mean = float(lum[top].mean())
    strength = float(np.clip((top_mean - float(np.median(lum))) / (top_mean + 1e-3), 0.0, 1.0))
    surroundings = f[lum <= bright_cut] if (lum <= bright_cut).any() else f.reshape(-1, 3)
    ambient_color = surroundings.reshape(-1, 3).mean(axis=0)

    return LightingAnalysis(
        key_x=key_x,
        key_y=key_y,
        key_color=_chroma(key_color),
        key_strength=strength,
        ambient_color=_chroma(ambient_color),
        ambient_level=float(lum.mean()),
        backlit=abs(key_x - 0.5) < 0.2 and key_y < 0.6 and strength > STRONG_KEY,
    )


def suggest_lights(analysis: LightingAnalysis, start_id: int = 1) -> tuple[list[Light], dict]:
    """Lights (all targeting the subject) and lighting-model values that match a background."""
    a = analysis
    kx = float(np.clip(a.key_x, -0.2, 1.2))
    ky = float(np.clip(a.key_y, -0.2, 1.0))
    lights: list[Light] = []
    next_id = start_id

    if a.key_strength > STRONG_KEY:
        key = new_light(next_id, LightType.DIRECTIONAL, name="Key (matched)", x=kx, y=ky, z=-0.35,
                        intensity=0.9 + 1.2 * a.key_strength, color=a.key_color, softness=0.3,
                        target=LightTarget.SUBJECT)
    else:
        key = new_light(next_id, LightType.AREA, name="Key (matched)", x=kx, y=ky, z=-0.45, size=0.35,
                        radius=1.6, softness=0.8, diffusion=0.35, intensity=1.0 + 0.5 * a.ambient_level,
                        color=a.key_color, target=LightTarget.SUBJECT)
    lights.append(key)
    next_id += 1

    if a.backlit:
        lights.append(new_light(next_id, LightType.RIM, name="Rim (matched)", x=kx, y=ky, z=1.0,
                                intensity=1.5 + 1.5 * a.key_strength, radius=1.2, color=a.key_color,
                                target=LightTarget.SUBJECT))
        next_id += 1

    lights.append(new_light(next_id, LightType.AMBIENT, name="Fill (matched)",
                            intensity=0.1 + 0.35 * a.ambient_level, color=a.ambient_color,
                            target=LightTarget.SUBJECT))
    # Keep a little of the photo's own light for detail; the new rig does the rest.
    # The old light's color would tint the subject against the new background: remove it.
    return lights, {"original_light": 0.25, "ambient": 0.0, "light_color": 1.0}
