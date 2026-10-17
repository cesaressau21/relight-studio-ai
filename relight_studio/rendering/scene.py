"""2.5D scene reconstruction from a photograph and its estimated depth.

Coordinate system (shared with :mod:`relight_studio.core.lights`):
- x grows to the right, y grows downwards, both in image-height units, so the
  frame spans ``x in [0, aspect]`` and ``y in [0, 1]``.
- z grows *away* from the camera; ``z = depth01 * relief`` where ``depth01`` is
  0 at the nearest estimated surface and 1 at the farthest.
- Normals point towards the camera (negative z).

The projection is treated as orthographic: a deliberate simplification because
monocular depth gives neither camera intrinsics nor metric scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.settings import DepthSettings
from .filters import gaussian_blur, guided_filter, luminance, resize, srgb_to_linear


@dataclass
class Scene:
    height: int
    width: int
    relief: float
    image_lin: torch.Tensor  # 1x3xHxW linear RGB of the original photo
    shading: torch.Tensor  # 1x1xHxW estimated original illumination (normalized, ~1 = brightest)
    depth01: torch.Tensor  # 1x1xHxW normalized depth (0 = near)
    z: torch.Tensor  # 1x1xHxW depth01 * relief, used for shadows
    normals: torch.Tensor  # 1x3xHxW unit normals (camera-facing has nz < 0)
    cavity: torch.Tensor  # 1x1xHxW screen-space occlusion term in [0, 1]
    mask: torch.Tensor | None  # 1x1xHxW relight mask in [0, 1]

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def device(self) -> torch.device:
        return self.image_lin.device

    def depth_preview(self, max_side: int = 512) -> np.ndarray:
        """Small CPU copy of the normalized depth for UI widgets."""
        scale = min(1.0, max_side / max(self.height, self.width))
        h, w = max(1, round(self.height * scale)), max(1, round(self.width * scale))
        return resize(self.depth01, h, w)[0, 0].float().cpu().numpy()


def normalize_disparity(disparity: torch.Tensor, settings: DepthSettings) -> torch.Tensor:
    """Relative inverse depth -> normalized depth in [0, 1] (0 = near), with user remap."""
    flat = disparity.flatten()
    if flat.numel() > 1_000_000:
        flat = flat[:: flat.numel() // 1_000_000 + 1]
    lo, hi = torch.quantile(flat, torch.tensor([0.01, 0.99], device=flat.device))
    nearness = ((disparity - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    depth = 1.0 - nearness
    if settings.invert:
        depth = 1.0 - depth
    span = max(settings.far - settings.near, 1e-3)
    depth = ((depth - settings.near) / span).clamp(0.0, 1.0)
    if abs(settings.gamma - 1.0) > 1e-4:
        depth = depth.clamp(min=1e-6) ** settings.gamma
    return depth


MAX_SLOPE = 2.5  # ~68 degrees: depth jumps at silhouettes are not true surfaces


def compute_normals(z: torch.Tensor, max_slope: float = MAX_SLOPE) -> torch.Tensor:
    """Normals of the height field z(x, y) with pixel pitch 1/H (image-height units).

    The slope is clamped so that occlusion boundaries (where monocular depth
    jumps between objects) do not produce grazing normals and bright halos.
    """
    h = z.shape[-2]
    pad = torch.nn.functional.pad(z, (1, 1, 1, 1), mode="replicate")
    zx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) * (h / 2.0)
    zy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) * (h / 2.0)
    scale = (max_slope / torch.sqrt(zx * zx + zy * zy).clamp(min=1e-8)).clamp(max=1.0)
    zx, zy = zx * scale, zy * scale
    n = torch.cat([zx, zy, -torch.ones_like(z)], dim=1)
    return n / n.norm(dim=1, keepdim=True)


@torch.inference_mode()
def build_scene(
    image: np.ndarray,
    disparity: np.ndarray,
    settings: DepthSettings,
    device: torch.device,
    depth_offset: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> Scene:
    """Build all cached maps for ``image`` (already at the target resolution).

    Args:
        image: H x W x 3 uint8 sRGB.
        disparity: model output at any resolution.
        depth_offset: user depth corrections at edit resolution (normalized depth units).
        mask: relight mask at edit resolution (uint8, 255 = apply).
    """
    h, w = image.shape[:2]
    img = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).to(device)
    img = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_lin = srgb_to_linear(img)
    guide = luminance(img)  # perceptual luminance as the edge guide

    disp = torch.from_numpy(np.ascontiguousarray(disparity, dtype=np.float32)).to(device)[None, None]
    depth = resize(normalize_disparity(disp, settings), h, w)
    if depth_offset is not None:
        off = torch.from_numpy(np.ascontiguousarray(depth_offset, dtype=np.float32)).to(device)[None, None]
        depth = (depth + resize(off, h, w)).clamp(-0.5, 1.5)

    # Snap the (low-res) depth edges onto image edges: joint guided upsampling.
    s = float(np.clip(settings.smoothing, 0.0, 1.0))
    r_edge = max(1, round(h * 0.003 * (1.0 + 2.0 * s)))
    depth = guided_filter(guide, depth, r_edge, 2e-3 + 6e-3 * s)

    relief = max(float(settings.relief), 1e-3)
    z = depth * relief

    # Normals come from an edge-preserving (self-guided) smoothing of the depth:
    # small undulations such as skin texture are flattened while the jumps
    # between objects stay sharp, which keeps silhouette artifacts thin.
    r_norm = max(2, round(h * (0.004 + 0.012 * s)))
    d_smooth = guided_filter(depth, depth, r_norm, (0.006 + 0.012 * s) ** 2)
    normals = compute_normals(gaussian_blur(d_smooth, 1.0 + h * 0.001) * relief)

    # Cavity: pixels deeper than their neighborhood receive less ambient light.
    d_blur = gaussian_blur(depth, h * 0.025)
    cavity = ((depth - d_blur) * 6.0).clamp(0.0, 1.0)

    # Original illumination estimate: luminance smoothed with the *depth* as guide.
    # Shading follows geometry while albedo detail (texture, freckles, print) does
    # not correlate with depth and is averaged out.
    lum = luminance(image_lin)
    r_shade = max(2, round(h * 0.02))
    shading = guided_filter(depth, lum, r_shade, 1e-3).clamp(min=1e-4)
    flat = shading.flatten()
    if flat.numel() > 1_000_000:
        flat = flat[:: flat.numel() // 1_000_000 + 1]
    ref = torch.quantile(flat, 0.97).clamp(min=1e-3)
    shading = (shading / ref).clamp(0.08, 1.5)

    return Scene(h, w, relief, image_lin, shading, depth, z, normals, cavity, mask_tensor(mask, h, w, device))


def mask_tensor(mask: np.ndarray | None, h: int, w: int, device: torch.device) -> torch.Tensor | None:
    """Edit-resolution uint8 mask -> 1x1xHxW float tensor in [0, 1]."""
    if mask is None:
        return None
    m = torch.from_numpy(np.ascontiguousarray(mask)).to(device)[None, None].float() / 255.0
    return resize(m, h, w).clamp(0.0, 1.0)
