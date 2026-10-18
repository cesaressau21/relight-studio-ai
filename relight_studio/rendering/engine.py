"""Relighting renderer (PyTorch; CUDA when available, CPU otherwise).

Image formation model (all in linear RGB)::

    I        = A * S                       photo = albedo x original shading
    S_d      = S ** delighting             partial, user-controlled separation
    A        = I / S_d
    relit    = A * (keep * S_d + ambient + sum_i diffuse_i) + sum_i specular_i

``diffuse_i`` uses the estimated normals, distance attenuation and screen-space
shadows marched through the depth map. With no lights and ``keep = 1`` the
output equals the photograph exactly.

This is a plausible 2.5D approximation, not physically based inverse rendering:
the depth is relative, the projection orthographic, occluders are treated as
height fields (infinitely thick) and nothing outside the frame is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..core.color import kelvin_to_linear_gain, srgb_to_linear
from ..core.lights import Light, LightTarget, LightType
from ..core.settings import Adjustments, MaskSettings, RelightSettings
from .filters import box_filter, linear_to_srgb, luminance, smoothstep, turbo
from .gobos import transmission as gobo_transmission
from .scene import Scene

VIEW_MODES = ("result", "original", "depth", "normals", "subject", "albedo", "shading", "lighting", "occlusion")

MAX_SHADOW_REACH = 0.6  # screen-space shadow ray length, in image heights


@dataclass
class RenderParams:
    lights: list[Light] = field(default_factory=list)
    relight: RelightSettings = field(default_factory=RelightSettings)
    adjustments: Adjustments = field(default_factory=Adjustments)
    mask_settings: MaskSettings = field(default_factory=MaskSettings)
    view: str = "result"


# --------------------------------------------------------------------------- helpers
def _vec(values, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=ref.dtype, device=ref.device).view(1, -1, 1, 1)


def light_rgb(light: Light) -> tuple[float, float, float]:
    """Linear radiance multiplier: color x temperature x intensity."""
    lin = srgb_to_linear(light.color)
    gain = kelvin_to_linear_gain(light.kelvin)
    return tuple(c * g * light.intensity for c, g in zip(lin, gain))  # type: ignore[return-value]


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=1, keepdim=True)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _shininess(roughness: float) -> float:
    r = max(roughness, 0.02)
    return float(np.clip(2.0 / (r**4) - 2.0, 2.0, 2000.0))


@dataclass
class _Region:
    """Rows [y0, y1) of the scene, with per-pixel positions."""

    y0: int
    y1: int
    pos: torch.Tensor  # 1x3xhxW surface points
    normals: torch.Tensor
    noise: torch.Tensor  # 1x1xhxW blue-ish noise in [0, 1) for ray jitter


def _region(scene: Scene, y0: int, y1: int) -> _Region:
    dev, h = scene.device, scene.height
    ys = (torch.arange(y0, y1, device=dev, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(scene.width, device=dev, dtype=torch.float32) + 0.5) / h
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.cat([gx[None, None], gy[None, None], scene.z[:, :, y0:y1]], dim=1)
    # Interleaved gradient noise: deterministic, no temporal flicker.
    py, px = torch.meshgrid(
        torch.arange(y0, y1, device=dev, dtype=torch.float32),
        torch.arange(scene.width, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    noise = torch.frac(52.9829189 * torch.frac(0.06711056 * px + 0.00583715 * py))[None, None]
    return _Region(y0, y1, pos, scene.normals[:, :, y0:y1], noise)


# --------------------------------------------------------------------------- shadows
def _shadow(scene: Scene, reg: _Region, ray: torch.Tensor, steps: int, softness: float) -> torch.Tensor:
    """Visibility in [0, 1] along ``ray`` (1x3xhxW, from each surface point).

    Marches the ray in screen space and compares its depth with the depth map;
    a sample is occluded when the depth map is in front of the ray. Rays leaving
    the frame are treated as unoccluded.
    """
    reach = torch.sqrt(ray[:, 0:1] ** 2 + ray[:, 1:2] ** 2)
    ray = ray * (MAX_SHADOW_REACH / reach.clamp(min=1e-6)).clamp(max=1.0)
    length = ray.norm(dim=1, keepdim=True)
    relief = scene.relief
    vis = torch.ones_like(reg.noise)
    zmap = scene.z
    aspect = scene.aspect
    for k in range(steps):
        t = ((k + reg.noise) / steps) ** 1.4  # denser samples near the receiver
        q = reg.pos + ray * t
        gx = q[:, 0] / aspect * 2.0 - 1.0
        gy = q[:, 1] * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        z_at = F.grid_sample(zmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
        inside = ((gx.abs() <= 1.0) & (gy.abs() <= 1.0)).unsqueeze(1).to(vis.dtype)
        dist = length * t
        bias = 0.006 * relief + 0.02 * dist
        penumbra = (0.004 + 0.05 * softness) * relief + softness * 0.15 * dist
        occ = smoothstep(bias, bias + penumbra, q[:, 2:3] - z_at) * inside
        vis = torch.minimum(vis, 1.0 - occ)
    return box_filter(vis, 1)


# --------------------------------------------------------------------------- lights
def _diffuse(ndl: torch.Tensor, softness: float, diffusion: float) -> torch.Tensor:
    wrap = 0.5 * softness
    wrapped = ((ndl + wrap) / (1.0 + wrap)).clamp(min=0.0)
    half = (0.5 * ndl + 0.5) ** 2
    return torch.lerp(wrapped, half, diffusion)


def _attenuation(dist: torch.Tensor, light: Light) -> torch.Tensor:
    radius = max(light.radius, 1e-3)
    att = (1.0 + (dist / radius) ** 2) ** (-max(light.attenuation, 0.0))
    window = (1.0 - (dist / (radius * 4.0)) ** 4).clamp(0.0, 1.0) ** 2
    return att * window


def _light_position(scene: Scene, light: Light) -> tuple[float, float, float]:
    return light.x * scene.aspect, light.y, light.z * scene.relief


def _specular(n: torch.Tensor, l: torch.Tensor, rs: RelightSettings, light: Light, rough_boost: float = 0.0):
    view = _vec([0.0, 0.0, -1.0], n)
    half = _normalize(l + view)
    shin = _shininess(min(rs.roughness + rough_boost, 1.0))
    norm = (shin + 8.0) / (8.0 * math.pi)
    ndh = _dot(n, half).clamp(min=0.0)
    return ndh**shin * norm * rs.specular * light.specular * (_dot(n, l) > 0).to(n.dtype)


def _point_like(scene: Scene, reg: _Region, light: Light, rs: RelightSettings):
    lp = _vec(_light_position(scene, light), reg.pos)
    n = reg.normals
    is_area = light.type is LightType.AREA
    if is_area:
        s = max(light.size, 0.0)
        offsets = [(0, 0), (s, 0), (-s, 0), (0, s), (0, -s)]
    else:
        offsets = [(0, 0)]
    diffuse = torch.zeros_like(reg.noise)
    for ox, oy in offsets:
        to_light = lp + _vec([ox, oy, 0.0], reg.pos) - reg.pos
        dist = to_light.norm(dim=1, keepdim=True).clamp(min=1e-6)
        l = to_light / dist
        diffuse = diffuse + _diffuse(_dot(n, l), light.softness, light.diffusion) * _attenuation(dist, light)
    diffuse = diffuse / len(offsets)

    to_light = lp - reg.pos
    dist = to_light.norm(dim=1, keepdim=True).clamp(min=1e-6)
    l = to_light / dist
    vis = None
    if light.shadows and rs.shadow_strength > 0:
        if is_area and light.size > 0:
            # Soft lateral penumbra: average the visibility of points spread over the light.
            steps = max(8, rs.shadow_steps // 2)
            vis = sum(
                _shadow(scene, reg, lp + _vec([ox, oy, 0.0], reg.pos) - reg.pos, steps, light.softness)
                for ox, oy in offsets
            ) / len(offsets)
        else:
            vis = _shadow(scene, reg, to_light, rs.shadow_steps, light.softness)
    spec = _specular(n, l, rs, light, rough_boost=0.3 * light.size if is_area else 0.0) * _attenuation(dist, light)
    diffuse, spec = _apply_gobo(scene, reg, light, diffuse, spec)
    return diffuse, spec, vis


def _scene_center(scene: Scene) -> tuple[float, float, float]:
    return scene.aspect * 0.5, 0.5, 0.5 * scene.relief


def _apply_gobo(scene: Scene, reg: _Region, light: Light, diffuse: torch.Tensor, spec: torch.Tensor):
    """Multiply a light's contribution by its projected pattern, if any."""
    pattern = gobo_transmission(reg.pos, light, _light_position(scene, light), _scene_center(scene))
    if pattern is None:
        return diffuse, spec
    return diffuse * pattern, spec * pattern


def _directional(scene: Scene, reg: _Region, light: Light, rs: RelightSettings):
    lp = np.array(_light_position(scene, light))
    center = np.array([scene.aspect * 0.5, 0.5, 0.5 * scene.relief])
    direction = lp - center
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    l = _vec(direction.tolist(), reg.pos)
    n = reg.normals
    diffuse = _diffuse(_dot(n, l), light.softness, light.diffusion)
    vis = None
    if light.shadows and rs.shadow_strength > 0:
        xy = max(float(np.hypot(direction[0], direction[1])), 0.05)
        ray = l * (MAX_SHADOW_REACH / xy) * torch.ones_like(reg.noise)
        vis = _shadow(scene, reg, ray, rs.shadow_steps, light.softness)
    spec = _specular(n, l, rs, light)
    diffuse, spec = _apply_gobo(scene, reg, light, diffuse, spec)
    return diffuse, spec, vis


def _rim(scene: Scene, reg: _Region, light: Light):
    lx, ly, lz = _light_position(scene, light)
    n = reg.normals
    dx = lx - reg.pos[:, 0:1]
    dy = ly - reg.pos[:, 1:2]
    dist_xy = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    nxy = torch.sqrt(n[:, 0:1] ** 2 + n[:, 1:2] ** 2).clamp(min=1e-6)
    facing = ((n[:, 0:1] * dx + n[:, 1:2] * dy) / (nxy * dist_xy)).clamp(min=0.0)
    edge = (1.0 + n[:, 2:3]).clamp(0.0, 1.0)  # 0 when facing the camera, 1 at silhouettes
    sharpness = 1.0 + 4.0 * (1.0 - min(light.softness, 1.0))
    behind = smoothstep(-0.05 * scene.relief, 0.1 * scene.relief, lz - reg.pos[:, 2:3])
    rim = edge**sharpness * facing ** (1.0 - 0.7 * light.diffusion) * behind
    return rim * _attenuation(dist_xy, light)


# --------------------------------------------------------------------------- tone & grading
def _rolloff(x: torch.Tensor) -> torch.Tensor:
    """Out-of-range highlights burn toward white instead of clipping per channel.

    Identity for every pixel whose channels are all <= 1, so untouched areas
    keep the exact photo values.
    """
    peak = x.amax(dim=1, keepdim=True).clamp(min=1.0)
    toward_white = 1.0 - 1.0 / peak
    return torch.lerp(x / peak, torch.ones_like(x), toward_white)


def apply_adjustments(lin: torch.Tensor, adj: Adjustments, rolloff: bool) -> torch.Tensor:
    """Linear RGB -> graded sRGB in [0, 1]."""
    x = lin * (2.0 ** adj.exposure)
    if adj.temperature or adj.tint:
        gains = _vec([1.0 + 0.3 * adj.temperature, 1.0 - 0.25 * adj.tint, 1.0 - 0.3 * adj.temperature], x)
        gains = gains / (0.2126 * gains[0, 0] + 0.7152 * gains[0, 1] + 0.0722 * gains[0, 2])
        x = x * gains
    if rolloff:
        x = _rolloff(x)
    s = linear_to_srgb(x).clamp(0.0, 1.0)
    if abs(adj.gamma - 1.0) > 1e-4:
        s = s ** (1.0 / max(adj.gamma, 0.05))
    if adj.blacks or adj.whites:
        black = -0.12 * adj.blacks
        white = 1.0 - 0.18 * adj.whites
        s = ((s - black) / max(white - black, 1e-3)).clamp(0.0, 1.0)
    if adj.shadows or adj.highlights:
        lum = luminance(s)
        w_sh = 1.0 - smoothstep(0.0, 0.55, lum)
        w_hi = smoothstep(0.45, 1.0, lum)
        s = s * (1.0 + 0.9 * adj.shadows * w_sh)
        headroom = (1.0 - s) if adj.highlights > 0 else s  # brighten toward white / darken toward black
        s = (s + 0.35 * adj.highlights * w_hi * headroom).clamp(0.0, 1.0)
    if adj.contrast:
        if adj.contrast > 0:
            s = torch.lerp(s, s * s * (3.0 - 2.0 * s), min(adj.contrast, 1.0))
        else:
            s = torch.lerp(s, 0.5 + (s - 0.5) * 0.4, min(-adj.contrast, 1.0))
    if adj.saturation:
        lum = luminance(s)
        s = lum + (s - lum) * (1.0 + adj.saturation)
    return s.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- renderer
class Renderer:
    """Stateless renderer over a :class:`Scene`."""

    @torch.inference_mode()
    def render(
        self,
        scene: Scene,
        params: RenderParams,
        chunk_rows: int | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        """Render to an H x W x 3 uint8 sRGB array (rows processed in chunks).

        Images with transparency return H x W x 4: the relit colors plus the
        original alpha, untouched.
        """
        h, w = scene.height, scene.width
        if chunk_rows is None:
            chunk_rows = max(64, int(4_000_000 / max(w, 1)))  # bound GPU temporaries
        out = np.empty((h, w, 3 if scene.opacity is None else 4), np.uint8)
        if scene.opacity is not None:
            out[..., 3] = (scene.opacity[0, 0] * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        halo = 1  # the 3x3 shadow filter reads one row beyond each chunk
        for y0 in range(0, h, chunk_rows):
            y1 = min(h, y0 + chunk_rows)
            ya, yb = max(0, y0 - halo), min(h, y1 + halo)
            rgb = self.render_region(scene, params, ya, yb)[:, :, y0 - ya : y0 - ya + (y1 - y0)]
            out[y0:y1, :, :3] = (rgb[0].permute(1, 2, 0) * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            if progress:
                progress(y1 / h)
        return out

    def render_region(self, scene: Scene, params: RenderParams, y0: int, y1: int) -> torch.Tensor:
        """Rows [y0, y1) as 1x3xhxW sRGB in [0, 1]."""
        view = params.view
        rows = slice(y0, y1)
        if view == "original":
            return linear_to_srgb(scene.image_lin[:, :, rows]).clamp(0, 1)
        if view == "depth":
            return turbo(1.0 - scene.depth01[:, :, rows].clamp(0, 1))
        if view == "normals":
            n = scene.normals[:, :, rows]
            return torch.cat([n[:, 0:1] * 0.5 + 0.5, -n[:, 1:2] * 0.5 + 0.5, -n[:, 2:3]], dim=1).clamp(0, 1)
        if view == "subject":
            if scene.alpha is None:
                return (linear_to_srgb(scene.image_lin[:, :, rows]) * 0.25).clamp(0, 1)
            a = scene.alpha[:, :, rows]
            # Subject shown in full color, background dimmed: edges are easy to inspect.
            return (linear_to_srgb(scene.image_lin[:, :, rows]) * (0.15 + 0.85 * a)).clamp(0, 1)
        if view == "occlusion":
            return (1.0 - scene.cavity[:, :, rows]).expand(-1, 3, -1, -1).clamp(0, 1)

        rs = params.relight
        shading_d = scene.shading[:, :, rows] ** rs.delighting
        if view == "shading":
            return linear_to_srgb(shading_d.expand(-1, 3, -1, -1) * 0.8).clamp(0, 1)
        image_lin = scene.image_lin[:, :, rows]
        albedo = image_lin / shading_d
        if view == "albedo":
            return linear_to_srgb(albedo).clamp(0, 1)

        reg = _region(scene, y0, y1)
        diffuse, specular = self.lighting(scene, reg, params)
        if view == "lighting":
            return linear_to_srgb((diffuse + specular) * 0.5).clamp(0, 1)

        relit = albedo * (rs.original_light * shading_d + diffuse) + specular
        mask = scene.mask[:, :, rows] if scene.mask is not None and params.mask_settings.enabled else None
        if mask is not None:
            if params.mask_settings.invert:
                mask = 1.0 - mask
            relit = torch.lerp(image_lin, relit, mask)
        return apply_adjustments(relit, params.adjustments, rs.highlight_rolloff)

    def lighting(self, scene: Scene, reg: _Region, params: RenderParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulated diffuse irradiance and specular radiance (linear RGB) for a region."""
        rs = params.relight
        cavity = scene.cavity[:, :, reg.y0 : reg.y1]
        occlusion = 1.0 - rs.ambient_occlusion * cavity
        diffuse = _vec(srgb_to_linear(rs.ambient_color), reg.pos) * rs.ambient * occlusion
        specular = torch.zeros_like(diffuse)
        subject = scene.alpha[:, :, reg.y0 : reg.y1] if scene.alpha is not None else None
        for light in params.lights:
            if not light.enabled or light.intensity <= 0:
                continue
            color = _vec(light_rgb(light), reg.pos)
            # Light targets weight the contribution by the subject matte (soft at hair and edges).
            # Without a separated subject every light affects everything.
            if subject is not None and light.target is LightTarget.SUBJECT:
                color = color * subject
            elif subject is not None and light.target is LightTarget.BACKGROUND:
                color = color * (1.0 - subject)
            if light.type is LightType.AMBIENT:
                diffuse = diffuse + color * occlusion
                continue
            if light.type is LightType.RIM:
                diffuse = diffuse + color * _rim(scene, reg, light)
                continue
            if light.type is LightType.DIRECTIONAL:
                d, s, vis = _directional(scene, reg, light, rs)
            else:
                d, s, vis = _point_like(scene, reg, light, rs)
            if vis is not None:
                vis = 1.0 - rs.shadow_strength * (1.0 - vis)
                d, s = d * vis, s * vis
            diffuse = diffuse + color * d
            specular = specular + color * s
        return diffuse, specular
