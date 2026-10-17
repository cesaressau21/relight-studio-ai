"""Editable relighting document.

The source photograph is loaded once and never modified. Everything the user
changes lives in parameter objects plus two editable raster layers (relight
mask and depth corrections) stored at a fixed *edit resolution* so projects stay
independent of the preview device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .lights import Light, LightRig, LightType, new_light
from .settings import Adjustments, DepthSettings, MaskSettings, RelightSettings

EDIT_MAX_SIDE = 2048


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


@dataclass
class DocState:
    """Immutable-by-convention snapshot used by the undo history."""

    lights: tuple[dict[str, Any], ...]
    relight: dict[str, Any]
    depth: dict[str, Any]
    adjustments: dict[str, Any]
    mask_settings: dict[str, Any]
    mask: np.ndarray | None
    depth_offset: np.ndarray | None

    def same_params(self, other: "DocState") -> bool:
        return (
            self.lights == other.lights
            and self.relight == other.relight
            and self.depth == other.depth
            and self.adjustments == other.adjustments
            and self.mask_settings == other.mask_settings
            and self.mask is other.mask
            and self.depth_offset is other.depth_offset
        )


@dataclass
class Document:
    image: np.ndarray  # H x W x 3 uint8 sRGB, read-only
    image_path: Path | None = None
    opacity: np.ndarray | None = None  # H x W uint8 transparency of the source (None = opaque), read-only
    rig: LightRig = field(default_factory=LightRig)
    relight: RelightSettings = field(default_factory=RelightSettings)
    depth: DepthSettings = field(default_factory=DepthSettings)
    adjustments: Adjustments = field(default_factory=Adjustments)
    mask_settings: MaskSettings = field(default_factory=MaskSettings)
    disparity: np.ndarray | None = None  # raw model output (relative inverse depth)
    disparity_source: str = ""  # model key + detail used for ``disparity``
    mask: np.ndarray | None = None  # edit-res uint8, 255 = relight applied
    depth_offset: np.ndarray | None = None  # edit-res float32 offsets in normalized depth
    alpha: np.ndarray | None = None  # derived subject matte (uint8, preview resolution)
    alpha_key: tuple = ()  # inputs the matte was computed from

    def __post_init__(self) -> None:
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Document image must be an H x W x 3 uint8 array")
        self.image.setflags(write=False)
        if self.opacity is not None:
            if self.opacity.shape != self.image.shape[:2] or self.opacity.dtype != np.uint8:
                raise ValueError("Document opacity must be an H x W uint8 array matching the image")
            self.opacity.setflags(write=False)

    # ------------------------------------------------------------------ geometry
    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def edit_size(self) -> tuple[int, int]:
        return fit_size(self.width, self.height, EDIT_MAX_SIDE)

    def ensure_mask(self) -> np.ndarray:
        if self.mask is None:
            w, h = self.edit_size
            self.mask = np.full((h, w), 255, np.uint8)
        return self.mask

    def ensure_depth_offset(self) -> np.ndarray:
        if self.depth_offset is None:
            w, h = self.edit_size
            self.depth_offset = np.zeros((h, w), np.float32)
        return self.depth_offset

    # -------------------------------------------------------------------- lights
    @property
    def lights(self) -> list[Light]:
        return self.rig.lights

    def add_light(self, light_type: LightType, **overrides: Any) -> Light:
        light = new_light(self.rig.next_id(), light_type, **overrides)
        self.rig.lights.append(light)
        return light

    def remove_light(self, light_id: int) -> None:
        self.rig.lights = [l for l in self.rig.lights if l.id != light_id]

    def light(self, light_id: int | None) -> Light | None:
        return self.rig.get(light_id)

    # ------------------------------------------------------------------- history
    def snapshot(self) -> DocState:
        return DocState(
            lights=tuple(l.to_dict() for l in self.lights),
            relight=self.relight.to_dict(),
            depth=self.depth.to_dict(),
            adjustments=self.adjustments.to_dict(),
            mask_settings=self.mask_settings.to_dict(),
            mask=self.mask,
            depth_offset=self.depth_offset,
        )

    def restore(self, state: DocState) -> None:
        self.rig = LightRig([Light.from_dict(d) for d in state.lights])
        self.relight = RelightSettings.from_dict(state.relight)
        self.depth = DepthSettings.from_dict(state.depth)
        self.adjustments = Adjustments.from_dict(state.adjustments)
        self.mask_settings = MaskSettings.from_dict(state.mask_settings)
        self.mask = state.mask
        self.depth_offset = state.depth_offset
