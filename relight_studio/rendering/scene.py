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

from dataclasses import dataclass, field

import numpy as np
import torch

from ..core.settings import DepthSettings
from .edges import decontaminate, extend, fill_from, guided_alpha, layered_depth, subject_threshold, trimap_from_depth, upsample_alpha
from .filters import gaussian_blur, guided_filter, luminance, resize, srgb_to_linear


@dataclass
class Layer:
    """One depth layer (subject or background) extended across the silhouette band."""

    z: torch.Tensor  # 1x1xHxW
    normals: torch.Tensor  # 1x3xHxW


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
    alpha: torch.Tensor | None = None  # 1x1xHxW subject matte (None = no subject separation)
    opacity: torch.Tensor | None = None  # 1x1xHxW transparency of the source image (None = opaque)
    # Subject and background layers that ``z`` and ``normals`` composite with ``alpha``.
    # Edge pixels are shaded once per layer and mixed by the matte. None without separation.
    layers: tuple[Layer, Layer] | None = None
    depth_px: float = 1.0  # output pixels per depth-map pixel: how wide upsampling spreads a depth step
    background_lin: torch.Tensor | None = None  # 1x3xHxW replacement background (linear RGB)
    subject_lin: torch.Tensor | None = None  # subject colors without the old background at the edges
    _background_cache: dict = field(default_factory=dict, repr=False)

    def background_for(self, blur: float, brightness: float) -> torch.Tensor:
        """Replacement background with blur (relative to image height) and exposure applied."""
        assert self.background_lin is not None
        key = (round(float(blur), 3), round(float(brightness), 3))
        if key not in self._background_cache:
            bg = gaussian_blur(self.background_lin, float(blur) * 0.03 * self.height) if blur > 0 else self.background_lin
            self._background_cache = {key: bg * (2.0 ** float(brightness))}  # keep only the latest
        return self._background_cache[key]

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


def normalize_disparity(disparity: torch.Tensor, settings: DepthSettings,
                        valid: torch.Tensor | None = None) -> torch.Tensor:
    """Relative inverse depth -> normalized depth in [0, 1] (0 = near), with user remap.

    ``valid`` (same size, > 0.5 = use) restricts the range estimate to visible pixels,
    so a transparent background does not consume the depth range.
    """
    flat = disparity.flatten()
    if valid is not None and bool((valid > 0.5).any()):
        flat = disparity[valid > 0.5].flatten()
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


def _image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    img = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).to(device)
    return img.permute(2, 0, 1).unsqueeze(0).float() / 255.0


@torch.inference_mode()
def scene_depth(
    height: int,
    width: int,
    disparity: np.ndarray,
    settings: DepthSettings,
    device: torch.device,
    depth_offset: np.ndarray | None = None,
    opacity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized depth (0 = near) at the given size, including the user's depth edits.

    With ``opacity`` (1x1xHxW in [0, 1]) the depth range comes from visible pixels only
    and fully transparent pixels are pushed behind the subject.
    """
    disp = torch.from_numpy(np.ascontiguousarray(disparity, dtype=np.float32)).to(device)[None, None]
    valid = None if opacity is None else resize(opacity, *disp.shape[-2:])
    depth = resize(normalize_disparity(disp, settings, valid), height, width)
    if opacity is not None:
        depth = torch.where(opacity < 0.02, torch.full_like(depth, TRANSPARENT_DEPTH), depth)
    if depth_offset is not None:
        off = torch.from_numpy(np.ascontiguousarray(depth_offset, dtype=np.float32)).to(device)[None, None]
        depth = (depth + resize(off, height, width)).clamp(-0.5, 1.5)
    return depth


TRANSPARENT_DEPTH = 1.25  # behind the farthest visible surface


def opacity_tensor(opacity: np.ndarray | None, h: int, w: int, device: torch.device) -> torch.Tensor | None:
    """uint8 alpha (any resolution) -> 1x1xHxW float tensor in [0, 1]."""
    if opacity is None:
        return None
    a = torch.from_numpy(np.array(opacity, dtype=np.uint8, copy=True)).to(device)[None, None].float() / 255.0  # source is read-only
    return resize(a, h, w).clamp(0.0, 1.0)


def bleed_colors(img: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    """Replace colors under fully transparent pixels by nearby visible colors.

    Files store arbitrary colors (often black) where alpha is 0; without this
    they would bleed into the albedo/shading estimate along the outline.
    """
    visible = (opacity > 0.004).to(img.dtype)
    return torch.where(visible > 0.5, img, fill_from(img, visible))


def edge_band_px(settings: DepthSettings, height: int) -> int:
    return max(2, round(settings.edge_band * height))


def resolve_threshold(depth01: torch.Tensor, settings: DepthSettings) -> tuple[float, bool]:
    """Subject/background split: the user's value, or Otsu when automatic."""
    if settings.subject_threshold >= 0:
        return float(settings.subject_threshold), True
    return subject_threshold(depth01)


def _smoothed_normals(depth: torch.Tensor, relief: float, s: float) -> torch.Tensor:
    # Edge-preserving (self-guided) smoothing: skin texture does not become relief
    # while real depth jumps stay sharp.
    h = depth.shape[-2]
    r_norm = max(2, round(h * (0.004 + 0.012 * s)))
    d_smooth = guided_filter(depth, depth, r_norm, (0.006 + 0.012 * s) ** 2)
    return compute_normals(gaussian_blur(d_smooth, 1.0 + h * 0.001) * relief)


@torch.inference_mode()
def build_scene(
    image: np.ndarray,
    disparity: np.ndarray,
    settings: DepthSettings,
    device: torch.device,
    depth_offset: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    background: np.ndarray | None = None,
) -> Scene:
    """Build all cached maps for ``image`` (already at the target resolution).

    Args:
        image: H x W x 3 uint8 sRGB.
        disparity: model output at any resolution.
        depth_offset: user depth corrections at edit resolution (normalized depth units).
        mask: relight mask at edit resolution (uint8, 255 = apply).
        alpha: subject matte from an AI matting model (uint8, any resolution); used
            when ``settings.edge_mode == "matting"``, otherwise a fast matte is estimated.
        opacity: the image's own transparency (uint8, any resolution). Transparent
            pixels are not part of the scene, and the transparency is the subject outline.
        background: replacement background (uint8 RGB, already fitted to the image's
            aspect ratio); composited behind the subject when enabled at render time.
    """
    h, w = image.shape[:2]
    img = _image_tensor(image, device)
    op = opacity_tensor(opacity, h, w, device)
    if op is not None:
        img = bleed_colors(img, op)
    image_lin = srgb_to_linear(img)
    depth = scene_depth(h, w, disparity, settings, device, depth_offset, op)
    s = float(np.clip(settings.smoothing, 0.0, 1.0))
    relief = max(float(settings.relief), 1e-3)

    alpha_t = None
    layers = None
    if op is not None:
        alpha_t = op  # the file's own transparency is the exact outline
    elif settings.edge_mode in ("guided", "matting"):
        band = edge_band_px(settings, h)
        if settings.edge_mode == "matting" and alpha is not None:
            a = torch.from_numpy(np.ascontiguousarray(alpha)).to(device)[None, None].float() / 255.0
            alpha_t = upsample_alpha(a, img)
        else:
            threshold, separable = resolve_threshold(depth, settings)
            if separable:
                alpha_t = guided_alpha(img, trimap_from_depth(depth, threshold, band), band)

    if alpha_t is not None:
        # Two layers, each free of the silhouette depth jump, composited by the matte.
        band = edge_band_px(settings, h)
        depth_fg, depth_bg = layered_depth(depth, alpha_t, band)
        depth = alpha_t * depth_fg + (1.0 - alpha_t) * depth_bg
        # Past the opaque subject its depth is only a continuation whose normals can face
        # anywhere (even the light). Edge pixels show the subject's rim: carry its normals out.
        n_fg = extend(_smoothed_normals(depth_fg, relief, s), (alpha_t > 0.98).to(depth.dtype), 6 * band)
        n_fg = n_fg / n_fg.norm(dim=1, keepdim=True).clamp(min=1e-8)
        n_bg = _smoothed_normals(depth_bg, relief, s)
        layers = (Layer(depth_fg * relief, n_fg), Layer(depth_bg * relief, n_bg))
        n = alpha_t * n_fg + (1.0 - alpha_t) * n_bg
        normals = n / n.norm(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        # Snap the (low-res) depth edges onto image edges: joint guided upsampling.
        r_edge = max(1, round(h * 0.003 * (1.0 + 2.0 * s)))
        depth = guided_filter(luminance(img), depth, r_edge, 2e-3 + 6e-3 * s)
        normals = _smoothed_normals(depth, relief, s)
    z = depth * relief

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

    background_lin = subject_lin = None
    if background is not None:
        background_lin = srgb_to_linear(resize(_image_tensor(background, device), h, w))
        if alpha_t is not None:
            # Transparent PNGs already store clean subject colors; photos need edge cleanup.
            subject_lin = image_lin if op is not None else decontaminate(image_lin, alpha_t)

    return Scene(h, w, relief, image_lin, shading, depth, z, normals, cavity,
                 mask_tensor(mask, h, w, device), alpha_t, op, layers, h / max(disparity.shape[0], 1),
                 background_lin, subject_lin)


def mask_tensor(mask: np.ndarray | None, h: int, w: int, device: torch.device) -> torch.Tensor | None:
    """Edit-resolution uint8 mask -> 1x1xHxW float tensor in [0, 1]."""
    if mask is None:
        return None
    m = torch.from_numpy(np.ascontiguousarray(mask)).to(device)[None, None].float() / 255.0
    return resize(m, h, w).clamp(0.0, 1.0)
