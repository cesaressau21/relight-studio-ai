"""GPU-friendly image filters on NCHW tensors (O(1) per pixel box filtering)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _box_1d(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    r = int(min(max(r, 0), n - 1))
    if r == 0:
        return x
    c = torch.cumsum(x, dim=dim)
    zero_shape = list(x.shape)
    zero_shape[dim] = 1
    c = torch.cat([torch.zeros(zero_shape, dtype=x.dtype, device=x.device), c], dim=dim)
    idx = torch.arange(n, device=x.device)
    lo = (idx - r).clamp(min=0)
    hi = (idx + r).clamp(max=n - 1)
    total = c.index_select(dim, hi + 1) - c.index_select(dim, lo)
    count_shape = [1] * x.ndim
    count_shape[dim] = n
    count = (hi - lo + 1).to(x.dtype).view(count_shape)
    return total / count


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, normalized by the valid pixel count at borders."""
    return _box_1d(_box_1d(x, r, -1), r, -2)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian approximation by three successive box filters."""
    if sigma < 0.5:
        return x
    passes = 3
    width = math.sqrt(12.0 * sigma * sigma / passes + 1.0)
    r = max(1, int(round((width - 1.0) / 2.0)))
    for _ in range(passes):
        x = box_filter(x, r)
    return x


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """He et al. guided filter with a single-channel guide (edge-preserving smoothing)."""
    mean_i = box_filter(guide, r)
    mean_p = box_filter(src, r)
    corr_ip = box_filter(guide * src, r)
    var_i = box_filter(guide * guide, r) - mean_i * mean_i
    a = (corr_ip - mean_i * mean_p) / (var_i.clamp(min=0) + eps)
    b = mean_p - a * mean_i
    return box_filter(a, r) * guide + box_filter(b, r)


def resize(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.shape[-2:] == (height, width):
        return x
    shrinking = height < x.shape[-2] and width < x.shape[-1]
    mode = "area" if shrinking else "bilinear"
    kwargs = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(x, size=(height, width), mode=mode, **kwargs)


def smoothstep(e0: float | torch.Tensor, e1: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-8) ** (1.0 / 2.4) - 0.055)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * w).sum(dim=1, keepdim=True)


def turbo(x: torch.Tensor) -> torch.Tensor:
    """Turbo colormap (polynomial fit by Google AI), input N1HW in [0, 1] -> N3HW."""
    x = x.clamp(0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)
