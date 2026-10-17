"""Lighting presets: JSON files that set real engine parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .document import Document
from .lights import Light, LightRig, LightType, new_light
from .paths import presets_dir
from .settings import Adjustments, RelightSettings


@dataclass
class Preset:
    key: str
    name: str
    description: str
    lights: list[dict[str, Any]]
    relight: dict[str, Any] = field(default_factory=dict)
    adjustments: dict[str, Any] = field(default_factory=dict)
    swatch: list[str] = field(default_factory=list)
    order: int = 100

    @classmethod
    def from_file(cls, path: Path) -> "Preset":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            key=path.stem,
            name=data["name"],
            description=data.get("description", ""),
            lights=data.get("lights", []),
            relight=data.get("relight", {}),
            adjustments=data.get("adjustments", {}),
            swatch=data.get("swatch", []),
            order=int(data.get("order", 100)),
        )

    def build_lights(self) -> list[Light]:
        lights = []
        for i, spec in enumerate(self.lights, start=1):
            spec = dict(spec)
            light_type = LightType(spec.pop("type", "point"))
            lights.append(new_light(i, light_type, **spec))
        return lights

    def apply(self, doc: Document) -> None:
        """Replace the light rig and apply global parameters (others reset to defaults)."""
        doc.rig = LightRig(self.build_lights())
        relight = RelightSettings().to_dict()
        relight.update(self.relight)
        doc.relight = RelightSettings.from_dict(relight)
        adjustments = Adjustments().to_dict()
        adjustments.update(self.adjustments)
        doc.adjustments = Adjustments.from_dict(adjustments)


def load_presets(directory: Path | None = None) -> list[Preset]:
    directory = directory or presets_dir()
    presets = [Preset.from_file(p) for p in directory.glob("*.json")]
    return sorted(presets, key=lambda p: (p.order, p.name))
