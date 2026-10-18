"""Subject/background separation at silhouettes.

Monocular depth is predicted at ~518 px and is soft at occlusion boundaries:
upsampled, the subject's depth bleeds several pixels into the background. Lit
directly, that band glows like part of the subject and casts ghost shadows.

The fix treats the scene as two layers:

1. Find the subject from the depth histogram (Otsu) and build a trimap whose
   unknown band covers the unreliable depth edge.
2. Estimate an alpha matte inside that band, either from an AI matting model
   (ViTMatte, see :mod:`relight_studio.ai.matting`) or with the fast color
   projection below.
3. Extend each layer's depth across the band from its reliable interior
   (push-pull filling), so neither layer contains a depth jump.
4. Composite depth and normals with the alpha: the silhouette now follows the
   photo's real outline, including semi-transparent hair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .filters import box_filter, guided_filter, luminance, resize

SEPARABILITY_MIN = 0.6  # Otsu effectiveness (between-class / total variance)
VALLEY_MAX = 0.4  # depth histogram density at the split relative to the smaller mode; a ramp scores ~1
SUBJECT_MIN_FRACTION = 0.03
BLEED_TOLERANCE = 0.04  # background pixels pulled nearer than this are treated as subject bleed


def subject_threshold(depth01: torch.Tensor) -> tuple[float, bool]:
    """Otsu threshold on normalized depth and whether a distinct subject exists.

    A subject is "distinct" when the depth histogram has a clear valley at the
    split (a real jump between subject and background), not just a slope.
    """
    flat = depth01.flatten().float().clamp(0.0, 1.0)
    if flat.numel() > 1_000_000:
        flat = flat[:: flat.numel() // 1_000_000 + 1]
    hist = torch.histc(flat, bins=256, min=0.0, max=1.0)
    p = hist / hist.sum().clamp(min=1.0)
    centers = (torch.arange(256, device=p.device, dtype=p.dtype) + 0.5) / 256
    omega = torch.cumsum(p, 0)
    mu = torch.cumsum(p * centers, 0)
    mu_t = mu[-1]
    between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega)).clamp(min=1e-12)
    k = int(torch.argmax(between[:-1]))
    threshold = float(centers[k] + 0.5 / 256)
    total = (p * (centers - mu_t) ** 2).sum()
    effectiveness = float(between[k] / total.clamp(min=1e-12))
    fraction = float(omega[k])
    coarse = torch.histc(flat, bins=64, min=0.0, max=1.0)
    j = min(int(threshold * 64), 63)
    valley = coarse[max(0, j - 3) : j + 4].mean()
    smaller_mode = torch.minimum(coarse[: max(j, 1)].max(), coarse[j:].max()).clamp(min=1.0)
    valley_ratio = float(valley / smaller_mode)
    separable = (
        effectiveness >= SEPARABILITY_MIN
        and valley_ratio <= VALLEY_MAX
        and SUBJECT_MIN_FRACTION <= fraction <= 1 - SUBJECT_MIN_FRACTION
    )
    return threshold, separable


def _max_pool(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return x
    x = F.max_pool2d(x, (1, 2 * r + 1), stride=1, padding=(0, r))
    return F.max_pool2d(x, (2 * r + 1, 1), stride=1, padding=(r, 0))


def erode(mask: torch.Tensor, r: int) -> torch.Tensor:
    return 1.0 - _max_pool(1.0 - mask, r)


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    return _max_pool(mask, r)


def trimap_from_depth(depth01: torch.Tensor, threshold: float, band_px: int) -> torch.Tensor:
    """1 = subject, 0 = background, 0.5 = unknown band around the depth edge."""
    near = (depth01 < threshold).to(depth01.dtype)
    fg = erode(near, band_px)
    bg = erode(1.0 - near, band_px)
    return torch.where(fg > 0.5, torch.ones_like(near), torch.where(bg > 0.5, torch.zeros_like(near), 0.5 * torch.ones_like(near)))


def fill_from(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Push-pull interpolation: keep valid pixels, fill the rest from their surroundings."""
    pyramid: list[tuple[torch.Tensor, torch.Tensor]] = []
    num, den = values * valid, valid
    while True:
        pyramid.append((num, den))
        if min(num.shape[-2:]) <= 2:
            break
        num = F.avg_pool2d(num, 2, ceil_mode=True)
        den = F.avg_pool2d(den, 2, ceil_mode=True)
    num, den = pyramid[-1]
    estimate = num.sum(dim=(-2, -1), keepdim=True) / den.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    estimate = estimate.expand_as(num)
    for num, den in reversed(pyramid):
        up = F.interpolate(estimate, size=num.shape[-2:], mode="bilinear", align_corners=False)
        weight = den.clamp(0.0, 1.0)
        local = num / den.clamp(min=1e-12)
        estimate = weight * local + (1.0 - weight) * up
    return estimate


def guided_alpha(image: torch.Tensor, trimap: torch.Tensor, band_px: int) -> torch.Tensor:
    """Fast matte: project each unknown pixel's color onto its local subject/background colors.

    Args:
        image: 1x3xHxW sRGB in [0, 1].
        trimap: output of :func:`trimap_from_depth`.
    """
    fg_known = (trimap > 0.75).to(image.dtype)
    bg_known = (trimap < 0.25).to(image.dtype)
    fg_color = fill_from(image, fg_known)
    bg_color = fill_from(image, bg_known)
    diff = fg_color - bg_color
    dist2 = (diff * diff).sum(dim=1, keepdim=True)
    projected = ((image - bg_color) * diff).sum(dim=1, keepdim=True) / dist2.clamp(min=1e-6)
    projected = projected.clamp(0.0, 1.0)
    # Where subject and background colors are alike the projection is meaningless:
    # fall back to a soft split in the middle of the unknown band.
    confidence = (dist2.sqrt() / 0.15).clamp(0.0, 1.0)
    soft = F.avg_pool2d(trimap, 2 * band_px + 1, stride=1, padding=band_px, count_include_pad=False)
    alpha = confidence * projected + (1.0 - confidence) * (soft > 0.5).to(image.dtype)
    alpha = guided_filter(luminance(image), alpha, max(1, band_px // 4), 1e-4).clamp(0.0, 1.0)
    return torch.where(trimap > 0.75, torch.ones_like(alpha), torch.where(trimap < 0.25, torch.zeros_like(alpha), alpha))


def upsample_alpha(alpha: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Resize a matte to ``image`` and snap its edges to the image edges."""
    h, w = image.shape[-2:]
    if alpha.shape[-2:] == (h, w):
        return alpha.clamp(0.0, 1.0)
    scale = max(h / alpha.shape[-2], w / alpha.shape[-1])
    up = resize(alpha, h, w)
    if scale <= 1.0:
        return up.clamp(0.0, 1.0)
    r = max(1, round(scale * 1.5))
    return guided_filter(luminance(image), up, r, 1e-4).clamp(0.0, 1.0)


def extend(values: torch.Tensor, valid: torch.Tensor, steps: int) -> torch.Tensor:
    """Grow valid values outward, then push-pull fill the rest.

    Invalid pixels take the mean of valid pixels within a radius that doubles at
    each pass (1, 2, 4, ... up to ``steps``), so pixels next to the valid region
    get their nearest neighbors' values. Unlike plain push-pull, a surface keeps
    its edge depth instead of being pulled toward its interior average.
    """
    values = values * valid
    valid = valid.clone()
    r = 1
    while r <= max(steps, 1):
        num = box_filter(values * valid, r)
        den = box_filter(valid, r)
        grow = (valid < 0.5) & (den > 1e-6)
        if not bool(grow.any()):
            break
        values = torch.where(grow, num / den.clamp(min=1e-12), values)
        valid = torch.where(grow, torch.ones_like(valid), valid)
        r *= 2
    return fill_from(values, valid)


def decontaminate(image_lin: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Estimate the subject's own color where the matte is partial (hair, soft edges).

    The photo stores I = a*F + (1-a)*B. B is extended from the nearby background,
    F is solved for, and pixels with almost no coverage take the nearby subject
    color, so placing F on a new background leaves no fringe of the old one.
    """
    old_background = fill_from(image_lin, (alpha < 0.02).to(image_lin.dtype))
    subject_fill = fill_from(image_lin, (alpha > 0.98).to(image_lin.dtype))
    solved = ((image_lin - (1.0 - alpha) * old_background) / alpha.clamp(min=0.05)).clamp(0.0, 1.0)
    return torch.where(alpha > 0.98, image_lin, torch.where(alpha < 0.05, subject_fill, solved))


def layered_depth(depth01: torch.Tensor, alpha: torch.Tensor, band_px: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Subject and background depth layers, each extended across the silhouette band.

    Depth predicted at the silhouette mixes both layers. Background pixels whose
    depth is pulled toward the subject (bleed) are discarded and re-grown from
    the reliable background; the subject keeps its own rim depth and is grown
    outward under semi-transparent matte regions such as hair.
    """
    bg_mask = (alpha < 0.02).to(depth01.dtype)
    fg_mask = (alpha > 0.98).to(depth01.dtype)
    if fg_mask.sum() < 1 or bg_mask.sum() < 1:
        return depth01, depth01
    far_core = erode(bg_mask, 4 * band_px)
    if far_core.sum() < 1:
        far_core = erode(bg_mask, band_px)
    if far_core.sum() < 1:
        far_core = bg_mask
    reference = fill_from(depth01, far_core)
    bg_reliable = torch.maximum(bg_mask * (depth01 >= reference - BLEED_TOLERANCE).to(depth01.dtype), far_core)
    steps = 6 * band_px
    depth_bg = extend(depth01, erode(bg_reliable, 1), steps)
    depth_fg = extend(depth01, fg_mask, steps)
    # Never let the extended subject sit behind the background.
    depth_fg = torch.minimum(depth_fg, depth_bg)
    return depth_fg, depth_bg
