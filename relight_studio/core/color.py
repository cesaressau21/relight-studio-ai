"""Color-space helpers shared by the engine, the UI and the presets."""

from __future__ import annotations

import math

import numpy as np

RGB = tuple[float, float, float]

NEUTRAL_KELVIN = 6500.0


def srgb_to_linear_scalar(c: float) -> float:
    c = min(max(c, 0.0), 1.0)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb_scalar(c: float) -> float:
    c = max(c, 0.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


def srgb_to_linear(rgb: RGB) -> RGB:
    return tuple(srgb_to_linear_scalar(c) for c in rgb)  # type: ignore[return-value]


def srgb_to_linear_np(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_np(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, None)
    return np.where(arr <= 0.0031308, arr * 12.92, 1.055 * np.power(arr, 1.0 / 2.4) - 0.055)


def _kelvin_srgb(kelvin: float) -> RGB:
    """Black-body color approximation (Tanner Helland fit), sRGB in [0, 1]."""
    t = min(max(kelvin, 1000.0), 40000.0) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    return tuple(min(max(v, 0.0), 255.0) / 255.0 for v in (r, g, b))  # type: ignore[return-value]


def kelvin_to_linear_gain(kelvin: float) -> RGB:
    """Linear RGB multiplier for a color temperature, normalized so 6500 K is neutral."""
    ref = srgb_to_linear(_kelvin_srgb(NEUTRAL_KELVIN))
    cur = srgb_to_linear(_kelvin_srgb(kelvin))
    gain = [c / max(r, 1e-6) for c, r in zip(cur, ref)]
    # Keep perceived brightness roughly constant across temperatures.
    lum = 0.2126 * gain[0] + 0.7152 * gain[1] + 0.0722 * gain[2]
    return tuple(g / max(lum, 1e-6) for g in gain)  # type: ignore[return-value]


def kelvin_to_srgb_preview(kelvin: float) -> RGB:
    """Display color for UI swatches."""
    return _kelvin_srgb(kelvin)


def hex_to_rgb(value: str) -> RGB:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: RGB) -> str:
    return "#" + "".join(f"{round(min(max(c, 0.0), 1.0) * 255):02x}" for c in rgb)
