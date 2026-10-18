"""Parameter groups of a relighting document."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

from .color import RGB

T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items() if k in known}
    return cls(**clean)


@dataclass
class DepthSettings:
    """How the raw model output becomes scene geometry."""

    model: str = "small"
    detail: str = "standard"
    relief: float = 0.6  # depth range in image-height units
    smoothing: float = 0.5  # edge-preserving smoothing amount (0..1)
    near: float = 0.0  # remap of the normalized depth range
    far: float = 1.0
    gamma: float = 1.0
    invert: bool = False
    edge_mode: str = "matting"  # "off" | "guided" (fast, color based) | "matting" (AI matte)
    subject_threshold: float = -1.0  # normalized depth splitting subject/background; < 0 = automatic
    edge_band: float = 0.02  # width of the uncertain silhouette band, in image heights

    def depth_key(self) -> tuple:
        """Parameters that change the normalized depth itself."""
        return (self.near, self.far, self.gamma, self.invert)

    def matte_key(self) -> tuple:
        """Parameters that change the subject matte."""
        return (*self.depth_key(), self.edge_mode, self.subject_threshold, self.edge_band)

    def geometry_key(self) -> tuple:
        return (self.relief, self.smoothing, *self.matte_key())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DepthSettings":
        return _from_dict(cls, data)


@dataclass
class RelightSettings:
    """Global lighting model parameters."""

    original_light: float = 0.75  # how much of the captured illumination is kept
    delighting: float = 0.6  # strength of the albedo/shading separation
    ambient: float = 0.0
    ambient_color: RGB = (1.0, 1.0, 1.0)
    ambient_occlusion: float = 0.5
    specular: float = 0.2
    roughness: float = 0.5
    shadow_strength: float = 0.85
    shadow_steps: int = 24
    highlight_rolloff: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ambient_color"] = list(self.ambient_color)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RelightSettings":
        return _from_dict(cls, data)


@dataclass
class Adjustments:
    """Photographic adjustments applied after relighting."""

    exposure: float = 0.0  # EV
    contrast: float = 0.0  # -1..1
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    saturation: float = 0.0
    temperature: float = 0.0  # white balance, -1 (cool) .. 1 (warm)
    tint: float = 0.0  # -1 (green) .. 1 (magenta)
    gamma: float = 1.0

    def is_identity(self) -> bool:
        return self == Adjustments()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Adjustments":
        return _from_dict(cls, data)


@dataclass
class BackgroundSettings:
    """A replacement background photo placed behind the separated subject."""

    enabled: bool = False
    path: str = ""
    blur: float = 0.0  # 0..1, depth-of-field style softening
    brightness: float = 0.0  # EV

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BackgroundSettings":
        return _from_dict(cls, data)


@dataclass
class MaskSettings:
    enabled: bool = True
    invert: bool = False
    show_overlay: bool = False
    depth_edits_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MaskSettings":
        return _from_dict(cls, data)
