"""Brush dabs for the editable raster layers (relight mask and depth corrections)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BrushSettings:
    size: float = 0.06  # radius in image-height units
    hardness: float = 0.5
    flow: float = 0.6  # mask: opacity per dab; depth: offset per dab scale
    depth_strength: float = 0.02  # normalized-depth change per dab at full flow


def _falloff(shape: tuple[int, int], u: float, v: float, radius: float, hardness: float):
    h, w = shape
    r_px = max(radius * h, 0.5)
    cx, cy = u * w, v * h
    x0, x1 = max(0, int(cx - r_px - 1)), min(w, int(cx + r_px + 2))
    y0, y1 = max(0, int(cy - r_px - 1)), min(h, int(cy + r_px + 2))
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2) / r_px
    edge = min(max(hardness, 0.0), 0.99)
    t = np.clip((d - edge) / (1.0 - edge), 0.0, 1.0)
    weight = 1.0 - t * t * (3.0 - 2.0 * t)
    return (slice(y0, y1), slice(x0, x1)), weight


def paint_mask(mask: np.ndarray, u: float, v: float, brush: BrushSettings, erase: bool) -> bool:
    """Blend toward 255 (paint) or 0 (erase). Returns True when pixels changed."""
    hit = _falloff(mask.shape, u, v, brush.size, brush.hardness)
    if hit is None:
        return False
    region, weight = hit
    target = 0.0 if erase else 255.0
    current = mask[region].astype(np.float32)
    mask[region] = np.clip(np.round(current + (target - current) * weight * brush.flow), 0, 255).astype(np.uint8)
    return True


def paint_depth(offset: np.ndarray, u: float, v: float, brush: BrushSettings, push: bool) -> bool:
    """Push surfaces away from the camera (``push``) or pull them closer."""
    hit = _falloff(offset.shape, u, v, brush.size, brush.hardness)
    if hit is None:
        return False
    region, weight = hit
    sign = 1.0 if push else -1.0
    offset[region] = np.clip(offset[region] + sign * brush.depth_strength * brush.flow * weight, -1.0, 1.0)
    return True


def stroke_samples(u0: float, v0: float, u1: float, v1: float, aspect: float, radius: float) -> list[tuple[float, float]]:
    """Points between two cursor positions spaced at 25% of the brush radius."""
    dx, dy = (u1 - u0) * aspect, v1 - v0
    dist = math.hypot(dx, dy)
    spacing = max(radius * 0.25, 1e-4)
    n = max(1, int(dist / spacing))
    return [(u0 + (u1 - u0) * i / n, v0 + (v1 - v0) * i / n) for i in range(1, n + 1)]
