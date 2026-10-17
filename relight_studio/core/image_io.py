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
    im = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")
    kwargs: dict = {}
    if options.embed_srgb:
        kwargs["icc_profile"] = srgb_icc_bytes()
    if fmt == "JPEG":
        kwargs.update(quality=int(options.quality), subsampling=0 if options.quality >= 90 else 2, optimize=True)
    elif fmt == "WEBP":
        kwargs.update(quality=int(options.quality), method=5)
    elif fmt == "PNG":
        kwargs.update(compress_level=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format=fmt, **kwargs)
    return path
