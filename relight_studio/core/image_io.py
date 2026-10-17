"""Image loading and export (Pillow based, Unicode-path safe)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

SUPPORTED_INPUT = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
EXPORT_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """Load a photograph as H x W x 3 uint8 sRGB, honoring EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float32)
            arr = arr / (65535.0 if arr.max() > 255 else 255.0)
            im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")
        elif im.mode != "RGB":
            if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (0, 0, 0))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                im = bg
            else:
                im = im.convert("RGB")
        if icc:
            im = _to_srgb(im, icc)
        return np.array(im, dtype=np.uint8)


def has_transparency(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (im.mode == "P" and "transparency" in im.info)


def load_image_with_alpha(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image as (H x W x 3 uint8 sRGB, H x W uint8 alpha or None).

    Colors are kept *straight* (not composited onto any background), so what
    the file stores under transparent pixels never leaks into the result. The
    alpha is None when the image has no transparency or is fully opaque.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if not has_transparency(im):
            return load_image(path), None
        icc = im.info.get("icc_profile")
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
    rgb, alpha = rgba[..., :3].copy(), rgba[..., 3].copy()
    if icc:
        rgb = np.array(_to_srgb(Image.fromarray(rgb), icc), dtype=np.uint8)
    if alpha.min() == 255:
        return rgb, None
    return rgb, alpha


def _to_srgb(im: Image.Image, icc: bytes) -> Image.Image:
    """Convert from an embedded ICC profile to sRGB when it is not sRGB already."""
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = ImageCms.getProfileDescription(src).lower()
        if "srgb" in desc:
            return im
        return ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
    except Exception:  # unreadable profiles fall back to treating data as sRGB
        return im


def srgb_icc_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@dataclass
class ExportOptions:
    format: str = "PNG"  # PNG | JPEG | WEBP
    quality: int = 95
    long_side: int | None = None  # None = original resolution
    embed_srgb: bool = True
    keep_transparency: bool = True  # PNG and WebP keep the alpha channel of transparent images
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)  # fill when transparency is not kept

    @property
    def supports_alpha(self) -> bool:
        return self.format.upper() in ("PNG", "WEBP")

    def target_size(self, width: int, height: int) -> tuple[int, int]:
        if not self.long_side or self.long_side >= max(width, height):
            return width, height
        scale = self.long_side / max(width, height)
        return max(1, round(width * scale)), max(1, round(height * scale))


def save_image(rgb: np.ndarray, path: str | Path, options: ExportOptions) -> Path:
    path = Path(path)
    fmt = options.format.upper()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {options.format}")
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        if options.keep_transparency and fmt in ("PNG", "WEBP"):
            im = Image.fromarray(pixels, "RGBA")
        else:
            alpha = pixels[..., 3:4].astype(np.float32) / 255.0
            bg = np.array(options.background, np.float32) * 255.0
            flat = pixels[..., :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
            im = Image.fromarray(np.clip(flat + 0.5, 0, 255).astype(np.uint8), "RGB")
    else:
        im = Image.fromarray(pixels, "RGB")
    kwargs: dict = {}
    if options.embed_srgb:
        kwargs["icc_profile"] = srgb_icc_bytes()
    if fmt == "JPEG":
        kwargs.update(quality=int(options.quality), subsampling=0 if options.quality >= 90 else 2, optimize=True)
    elif fmt == "WEBP":
        kwargs.update(quality=int(options.quality), method=5, exact=True)  # exact: keep colors under alpha
    elif fmt == "PNG":
        kwargs.update(compress_level=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format=fmt, **kwargs)
    return path
