"""Light patterns (gobos) projected by a light onto the estimated geometry.

A point or area light acts as a projector aimed at the center of the scene:
each surface point is projected through the light onto a pattern plane, so the
pattern scales with distance and bends over the 2.5D shape. Directional lights
project orthographically (sunlight through blinds or leaves).

Patterns are procedural and resolution independent; a custom grayscale image
can be used too (white = light passes, tiled).
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..core.lights import Light, LightType
from .filters import gaussian_blur, smoothstep

log = logging.getLogger(__name__)

PATTERN_LIGHTS = (LightType.POINT, LightType.AREA, LightType.DIRECTIONAL)


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right/down vectors perpendicular to ``axis``; frontal axis maps to image x/y."""
    right = np.cross(np.array([0.0, 1.0, 0.0]), axis)
    if np.linalg.norm(right) < 1e-4:  # aiming straight up or down
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(axis, right)
    return right, down / np.linalg.norm(down)


def _dot(v: torch.Tensor, w: np.ndarray) -> torch.Tensor:
    return v[:, 0:1] * float(w[0]) + v[:, 1:2] * float(w[1]) + v[:, 2:3] * float(w[2])


def _fract(x: torch.Tensor) -> torch.Tensor:
    """Fractional part in [0, 1) also for negative values (torch.frac keeps the sign)."""
    return x - torch.floor(x)


def _hash(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
    return _fract(torch.sin(ix * 127.1 + iy * 311.7) * 43758.5453)


def _value_noise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ix, iy = torch.floor(x), torch.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy), _hash(ix + 1, iy)
    c, d = _hash(ix, iy + 1), _hash(ix + 1, iy + 1)
    return torch.lerp(torch.lerp(a, b, ux), torch.lerp(c, d, ux), uy)


def _band(distance: torch.Tensor, half_width: float, edge: float) -> torch.Tensor:
    """1 inside ``half_width`` of a line, fading over ``edge``."""
    return 1.0 - smoothstep(half_width - edge, half_width + edge, distance)


def _pattern(name: str, s: torch.Tensor, t: torch.Tensor, edge: float) -> torch.Tensor:
    if name == "blinds":
        return _band((_fract(s) - 0.5).abs(), 0.275, edge)  # 55 % open slats
    if name == "window":
        inside = _band(s.abs(), 1.0, edge) * _band(t.abs(), 1.5, edge)
        mullion_s = _band((s - torch.round(s)).abs(), 0.06, edge)
        mullion_t = _band((t - 0.5 - torch.round(t - 0.5)).abs(), 0.06, edge)
        return inside * (1.0 - mullion_s) * (1.0 - mullion_t)
    if name == "leaves":
        n = (0.55 * _value_noise(s, t)
             + 0.30 * _value_noise(2.1 * s + 5.2, 2.1 * t + 1.3)
             + 0.15 * _value_noise(4.3 * s + 11.7, 4.3 * t + 7.1))
        return smoothstep(0.44 - edge, 0.44 + edge, n)  # ~60 % of the light gets through
    if name == "dots":
        fx, fy = _fract(s) - 0.5, _fract(t) - 0.5
        return _band(torch.sqrt(fx * fx + fy * fy), 0.3, edge)
    return torch.ones_like(s)


@lru_cache(maxsize=8)
def _load_custom(path: str, mtime: float, softness: float, device: str) -> torch.Tensor | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail((1024, 1024))
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception as exc:  # unreadable file: the light simply has no pattern
        log.warning("Cannot read gobo image %s: %s", path, exc)
        return None
    tex = torch.from_numpy(arr).to(device)[None, None]
    sigma = softness * 0.06 * max(tex.shape[-2:])
    return gaussian_blur(tex, sigma) if sigma >= 0.5 else tex


def _custom(light: Light, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
    path = light.gobo_image
    if not path or not os.path.isfile(path):
        return None
    tex = _load_custom(path, os.path.getmtime(path), round(light.gobo_softness, 3), str(s.device))
    if tex is None:
        return None
    gx = _fract(s + 0.5) * 2.0 - 1.0  # tiled
    gy = _fract(t + 0.5) * 2.0 - 1.0
    grid = torch.stack([gx[:, 0], gy[:, 0]], dim=-1)
    return F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)


def transmission(pos: torch.Tensor, light: Light, light_pos: tuple[float, float, float],
                 center: tuple[float, float, float]) -> torch.Tensor | None:
    """Fraction of the light that passes the pattern at each surface point (1x1xhxW), or None."""
    if light.gobo == "none" or light.type not in PATTERN_LIGHTS or light.gobo_strength <= 0:
        return None
    lp, c = np.array(light_pos, float), np.array(center, float)
    if light.type is LightType.DIRECTIONAL:
        axis = c - lp
        axis = axis / max(np.linalg.norm(axis), 1e-6)
        right, down = _basis(axis)
        u, v = _dot(pos, right), _dot(pos, down)
        front = torch.ones_like(u)
    else:
        axis = c - lp
        axis = axis / max(np.linalg.norm(axis), 1e-6)
        right, down = _basis(axis)
        d = pos - torch.tensor(lp, dtype=pos.dtype, device=pos.device).view(1, 3, 1, 1)
        depth = _dot(d, axis)
        front = (depth > 1e-3).to(pos.dtype)
        depth = depth.clamp(min=1e-3)
        u, v = _dot(d, right) / depth, _dot(d, down) / depth
    theta = math.radians(light.gobo_rotation)
    size = max(light.gobo_size, 1e-3)
    s = (u * math.cos(theta) + v * math.sin(theta)) / size
    t = (-u * math.sin(theta) + v * math.cos(theta)) / size
    if light.gobo == "custom":
        pattern = _custom(light, s, t)
        if pattern is None:
            return None
    else:
        pattern = _pattern(light.gobo, s, t, 0.01 + 0.3 * float(np.clip(light.gobo_softness, 0, 1)))
    return torch.lerp(torch.ones_like(pattern), pattern * front, float(np.clip(light.gobo_strength, 0, 1)))
