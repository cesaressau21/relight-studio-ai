"""Light source model.

Coordinates are resolution independent:
- ``x`` and ``y`` are normalized image coordinates (0..1 inside the frame, values
  outside place the light off-frame).
- ``z`` uses the normalized depth axis of the scene: 0 is the nearest estimated
  surface, 1 the farthest one, negative values sit between the camera and the
  scene.
- ``radius`` and ``size`` are expressed in image-height units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from .color import RGB

GIZMO_PALETTE = (
    "#ffa94d",  # amber
    "#4dd4ff",  # cyan
    "#ff5ca8",  # magenta
    "#9be15d",  # lime
    "#b18cff",  # violet
    "#ff7a5c",  # coral
    "#5ce0b8",  # mint
    "#ffd84d",  # yellow
)


class LightType(str, Enum):
    POINT = "point"
    DIRECTIONAL = "directional"
    AMBIENT = "ambient"
    AREA = "area"
    RIM = "rim"

    @property
    def label(self) -> str:
        return {
            LightType.POINT: "Point",
            LightType.DIRECTIONAL: "Directional",
            LightType.AMBIENT: "Ambient",
            LightType.AREA: "Soft Area",
            LightType.RIM: "Rim",
        }[self]


class LightTarget(str, Enum):
    """What a light illuminates, using the scene's subject/background separation."""

    ALL = "all"
    SUBJECT = "subject"
    BACKGROUND = "background"

    @property
    def label(self) -> str:
        return {LightTarget.ALL: "Everything", LightTarget.SUBJECT: "Subject only",
                LightTarget.BACKGROUND: "Background only"}[self]


@dataclass
class Light:
    id: int
    type: LightType = LightType.POINT
    name: str = ""
    enabled: bool = True
    x: float = 0.3
    y: float = 0.35
    z: float = -0.3
    intensity: float = 1.5
    radius: float = 0.7
    color: RGB = (1.0, 1.0, 1.0)
    kelvin: float = 6500.0
    softness: float = 0.35
    diffusion: float = 0.15
    attenuation: float = 1.0
    size: float = 0.2
    specular: float = 1.0
    shadows: bool = True
    target: LightTarget = LightTarget.ALL

    def __post_init__(self) -> None:
        self.type = LightType(self.type)
        try:
            self.target = LightTarget(self.target)
        except ValueError:
            self.target = LightTarget.ALL
        self.color = tuple(float(c) for c in self.color)  # type: ignore[assignment]
        if not self.name:
            self.name = f"{self.type.label} {self.id}"

    @property
    def gizmo_color(self) -> str:
        return GIZMO_PALETTE[(self.id - 1) % len(GIZMO_PALETTE)]

    @property
    def is_positional(self) -> bool:
        return self.type is not LightType.AMBIENT

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        data["target"] = self.target.value
        data["color"] = list(self.color)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Light":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        if "color" in clean:
            clean["color"] = tuple(clean["color"])
        return cls(**clean)

    def copy(self, **changes: Any) -> "Light":
        data = self.to_dict()
        data.update(changes)
        return Light.from_dict(data)


# Sensible starting values per light type.
TYPE_DEFAULTS: dict[LightType, dict[str, Any]] = {
    LightType.POINT: dict(intensity=1.5, radius=0.7, softness=0.35, diffusion=0.15, z=-0.3),
    LightType.DIRECTIONAL: dict(intensity=1.0, softness=0.25, diffusion=0.1, z=-0.4),
    LightType.AMBIENT: dict(intensity=0.25, x=0.92, y=0.08, softness=1.0, diffusion=1.0),
    LightType.AREA: dict(intensity=1.4, radius=0.9, softness=0.7, diffusion=0.3, size=0.25, z=-0.35),
    LightType.RIM: dict(intensity=2.0, radius=0.8, softness=0.4, diffusion=0.0, z=0.9),
}


def new_light(light_id: int, light_type: LightType, **overrides: Any) -> Light:
    params: dict[str, Any] = dict(TYPE_DEFAULTS[light_type])
    params.update(overrides)
    return Light(id=light_id, type=light_type, **params)


@dataclass
class LightRig:
    """Ordered collection of lights with stable ids."""

    lights: list[Light] = field(default_factory=list)

    def next_id(self) -> int:
        return max((l.id for l in self.lights), default=0) + 1

    def get(self, light_id: int | None) -> Light | None:
        return next((l for l in self.lights if l.id == light_id), None)
