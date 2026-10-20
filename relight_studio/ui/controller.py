"""Editor state controller: the single place where the document is mutated.

Every user edit goes through a method here, which records an undo step and
emits ``changed`` with the kinds of data that were touched so the renderer and
the panels can react.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from ..core.brush import BrushSettings, paint_depth, paint_mask, stroke_samples
from ..core.document import Document, fit_size
from ..core.history import History
from ..core.background import analyze_background, fit_cover, suggest_lights
from ..core.image_io import load_image
from ..core.lights import Light, LightRig, LightType
from ..core.presets import Preset
from ..rendering.device import select_device

log = logging.getLogger(__name__)

ALL_KINDS = frozenset({"lights", "relight", "depth", "geometry", "adjust", "mask", "selection"})


def preview_max_side() -> int:
    return 1600 if select_device().cuda else 1024


class EditorController(QObject):
    changed = Signal(object)  # frozenset of kinds
    documentLoaded = Signal()
    selectionChanged = Signal(object)  # light id or None
    historyChanged = Signal()
    interactionChanged = Signal(bool)  # True while dragging (lower preview cost)

    def __init__(self) -> None:
        super().__init__()
        self.doc: Document | None = None
        self.history = History()
        self.selected: int | None = None
        self.brush = BrushSettings()
        self.view = "result"
        self.project_path: Path | None = None
        self.dirty = False
        self.preview_image: np.ndarray | None = None
        self.preview_opacity: np.ndarray | None = None
        self.mask_version = 0
        self.offset_version = 0
        self.disparity_version = 0
        self.alpha_version = 0
        self._background: tuple[str, np.ndarray | None] = ("", None)
        self._background_preview: tuple[tuple, np.ndarray | None] = ((), None)
        self._stroke: tuple[str, float, float] | None = None

    # ------------------------------------------------------------------ document
    def load_document(self, doc: Document, project_path: Path | None = None) -> None:
        self.doc = doc
        self.project_path = project_path
        w, h = fit_size(doc.width, doc.height, preview_max_side())
        same = (w, h) == (doc.width, doc.height)
        self.preview_image = doc.image if same else cv2.resize(doc.image, (w, h), interpolation=cv2.INTER_AREA)
        self.preview_opacity = None
        if doc.opacity is not None:
            self.preview_opacity = doc.opacity if same else cv2.resize(doc.opacity, (w, h), interpolation=cv2.INTER_AREA)
        self.selected = doc.lights[0].id if doc.lights else None
        self.history.reset(doc.snapshot())
        self.mask_version += 1
        self.offset_version += 1
        self.disparity_version += 1
        self.dirty = False
        self.documentLoaded.emit()
        self.selectionChanged.emit(self.selected)
        self.historyChanged.emit()
        self.changed.emit(ALL_KINDS)

    def analysis_image(self) -> np.ndarray:
        """Preview image for the AI models: transparent areas shown as neutral gray, not black."""
        assert self.preview_image is not None
        if self.preview_opacity is None:
            return self.preview_image
        a = self.preview_opacity[..., None].astype(np.float32) / 255.0
        return (self.preview_image.astype(np.float32) * a + 128.0 * (1.0 - a) + 0.5).astype(np.uint8)

    def preview_rgba(self) -> np.ndarray:
        """Preview for display (RGB, or RGBA when the image has transparency)."""
        assert self.preview_image is not None
        if self.preview_opacity is None:
            return self.preview_image
        return np.dstack([self.preview_image, self.preview_opacity])

    @property
    def has_document(self) -> bool:
        return self.doc is not None

    def set_disparity(self, disparity: np.ndarray, source: str) -> None:
        assert self.doc is not None
        reanalysis = self.doc.disparity is not None  # the first estimate is not a user edit
        self.doc.disparity = disparity
        self.doc.disparity_source = source
        self.disparity_version += 1
        if reanalysis:
            self.dirty = True
        self.changed.emit(frozenset({"geometry"}))

    def matte_key(self) -> tuple:
        """Inputs the subject matte depends on (compared with ``doc.alpha_key``)."""
        assert self.doc is not None
        offset = self.offset_version if self.doc.mask_settings.depth_edits_enabled else -1
        return (self.disparity_version, offset, self.doc.depth.matte_key())

    def current_alpha(self) -> np.ndarray | None:
        """The AI matte if it matches the current depth and edge settings."""
        if self.doc is None or self.doc.alpha is None or self.doc.alpha_key != self.matte_key():
            return None
        return self.doc.alpha

    def set_alpha(self, alpha: np.ndarray | None, key: tuple) -> None:
        assert self.doc is not None
        self.doc.alpha, self.doc.alpha_key = alpha, key
        self.alpha_version += 1
        self.changed.emit(frozenset({"geometry", "matte"}))

    # ------------------------------------------------------------------- history
    def _commit(self, label: str, kinds: set[str], merge_key: str | None = None) -> None:
        assert self.doc is not None
        if self.history.push(self.doc.snapshot(), label, merge_key):
            self.dirty = True
            self.historyChanged.emit()
        self.changed.emit(frozenset(kinds))

    def _restore(self, state) -> None:
        assert self.doc is not None
        old_mask, old_offset = self.doc.mask, self.doc.depth_offset
        self.doc.restore(state)
        if self.doc.mask is not old_mask:
            self.mask_version += 1
        if self.doc.depth_offset is not old_offset:
            self.offset_version += 1
        if self.doc.light(self.selected) is None:
            self.selected = self.doc.lights[-1].id if self.doc.lights else None
            self.selectionChanged.emit(self.selected)
        self.dirty = True
        self.historyChanged.emit()
        self.changed.emit(ALL_KINDS)

    def undo(self) -> None:
        state = self.history.undo() if self.doc else None
        if state is not None:
            self._restore(state)

    def redo(self) -> None:
        state = self.history.redo() if self.doc else None
        if state is not None:
            self._restore(state)

    # -------------------------------------------------------------------- lights
    def select(self, light_id: int | None) -> None:
        if light_id != self.selected:
            self.selected = light_id
            self.selectionChanged.emit(light_id)
            self.changed.emit(frozenset({"selection"}))

    def selected_light(self) -> Light | None:
        return self.doc.light(self.selected) if self.doc else None

    def add_light(self, light_type: LightType, **overrides: Any) -> Light | None:
        if not self.doc:
            return None
        light = self.doc.add_light(light_type, **overrides)
        self.selected = light.id
        self.selectionChanged.emit(light.id)
        self._commit(f"Add {light.type.label} light", {"lights", "selection"})
        return light

    def remove_light(self, light_id: int | None) -> None:
        if not self.doc or self.doc.light(light_id) is None:
            return
        self.doc.remove_light(light_id)  # type: ignore[arg-type]
        if self.selected == light_id:
            self.selected = self.doc.lights[-1].id if self.doc.lights else None
            self.selectionChanged.emit(self.selected)
        self._commit("Delete light", {"lights", "selection"})

    def duplicate_light(self, light_id: int | None) -> None:
        src = self.doc.light(light_id) if self.doc else None
        if src is None:
            return
        new_id = self.doc.rig.next_id()
        clone = src.copy(id=new_id, name=f"{src.name} copy", x=src.x + 0.04, y=src.y + 0.04)
        self.doc.lights.append(clone)
        self.selected = new_id
        self.selectionChanged.emit(new_id)
        self._commit("Duplicate light", {"lights", "selection"})

    def update_light(self, light_id: int | None, merge_key: str | None = None, **changes: Any) -> None:
        light = self.doc.light(light_id) if self.doc else None
        if light is None:
            return
        for key, value in changes.items():
            setattr(light, key, value)
        light.__post_init__()
        key = merge_key or f"light{light_id}." + ",".join(sorted(changes))
        self._commit(f"Edit {light.name}", {"lights"}, merge_key=key)

    # ---------------------------------------------------------- parameter groups
    def _update_group(self, group: str, kinds: set[str], label: str, merge_key: str | None, changes: dict) -> None:
        if not self.doc:
            return
        target = getattr(self.doc, group)
        for key, value in changes.items():
            setattr(target, key, value)
        self._commit(label, kinds, merge_key or f"{group}." + ",".join(sorted(changes)))

    def update_relight(self, merge_key: str | None = None, **changes: Any) -> None:
        self._update_group("relight", {"relight"}, "Lighting settings", merge_key, changes)

    def set_neutral_point(self, x: float, y: float) -> None:
        """Mark an image point (normalized) as neutral gray; turns color removal on if it was off."""
        if not self.doc:
            return
        rs = self.doc.relight
        rs.neutral_point = (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0))
        if rs.light_color <= 0.0:
            rs.light_color = 1.0
        self._commit("Pick neutral point", {"relight"})

    def clear_neutral_point(self) -> None:
        if self.doc and self.doc.relight.neutral_point is not None:
            self.doc.relight.neutral_point = None
            self._commit("Clear neutral point", {"relight"})

    def update_depth(self, merge_key: str | None = None, **changes: Any) -> None:
        self._update_group("depth", {"depth", "geometry"}, "Depth settings", merge_key, changes)

    def update_adjustments(self, merge_key: str | None = None, **changes: Any) -> None:
        self._update_group("adjustments", {"adjust"}, "Adjustments", merge_key, changes)

    def update_mask_settings(self, merge_key: str | None = None, **changes: Any) -> None:
        kinds = {"mask", "geometry"} if "depth_edits_enabled" in changes else {"mask"}
        self._update_group("mask_settings", kinds, "Mask settings", merge_key, changes)

    # ---------------------------------------------------------------- background
    def background_image(self) -> np.ndarray | None:
        """Full-resolution replacement background (cached per path), or None."""
        path = self.doc.background.path if self.doc else ""
        if not path:
            return None
        if self._background[0] != path:
            self._background = (path, load_image(path))
        return self._background[1]

    def preview_background(self) -> np.ndarray | None:
        """Background fitted to the preview image (cover + center crop)."""
        if self.preview_image is None or not (self.doc and self.doc.background.path):
            return None
        h, w = self.preview_image.shape[:2]
        key = (self.doc.background.path, w, h)
        if self._background_preview[0] != key:
            self._background_preview = (key, fit_cover(self.background_image(), w, h))
        return self._background_preview[1]

    def set_background(self, path: str) -> None:
        """Use ``path`` as the new background (raises if the image cannot be read)."""
        if not self.doc:
            return
        image = load_image(path)  # fail before touching the document
        self._background = (path, image)
        self._update_group("background", {"background"}, "Choose background", None,
                           {"path": path, "enabled": True})

    def clear_background(self) -> None:
        self._update_group("background", {"background"}, "Remove background", None, {"path": "", "enabled": False})

    def update_background(self, merge_key: str | None = None, **changes: Any) -> None:
        self._update_group("background", {"background"}, "Background", merge_key, changes)

    def match_lighting(self) -> None:
        """Replace the light rig with lights that match the background's light."""
        image = self.background_image()
        if not self.doc or image is None:
            return
        lights, relight = suggest_lights(analyze_background(image))
        self.doc.rig = LightRig(lights)
        for key, value in relight.items():
            setattr(self.doc.relight, key, value)
        self.selected = lights[0].id
        self.selectionChanged.emit(self.selected)
        self._commit("Match lighting to background", {"lights", "relight", "selection"})

    def reset_adjustments(self) -> None:
        if not self.doc:
            return
        from ..core.settings import Adjustments

        self.doc.adjustments = Adjustments()
        self._commit("Reset adjustments", {"adjust"})

    def apply_preset(self, preset: Preset) -> None:
        if not self.doc:
            return
        preset.apply(self.doc)
        self.selected = self.doc.lights[0].id if self.doc.lights else None
        self.selectionChanged.emit(self.selected)
        self._commit(f"Preset: {preset.name}", {"lights", "relight", "adjust", "selection"})

    def set_view(self, view: str) -> None:
        if view != self.view:
            self.view = view
            self.changed.emit(frozenset({"view"}))

    # -------------------------------------------------------------------- layers
    def _replace_layer(self, kind: str, fn: Callable[[np.ndarray], np.ndarray], label: str) -> None:
        if not self.doc:
            return
        if kind == "mask":
            self.doc.mask = fn(self.doc.ensure_mask().copy())
            self.mask_version += 1
            self._commit(label, {"mask"})
        else:
            self.doc.depth_offset = fn(self.doc.ensure_depth_offset().copy())
            self.offset_version += 1
            self._commit(label, {"geometry"})

    def fill_mask(self) -> None:
        self._replace_layer("mask", lambda m: np.full_like(m, 255), "Fill mask")

    def clear_mask(self) -> None:
        self._replace_layer("mask", np.zeros_like, "Clear mask")

    def invert_mask(self) -> None:
        self._replace_layer("mask", lambda m: 255 - m, "Invert mask")

    def clear_depth_edits(self) -> None:
        self._replace_layer("depth", np.zeros_like, "Clear depth edits")

    def begin_stroke(self, kind: str, u: float, v: float, alt: bool) -> None:
        """Start a brush stroke; the layer is copied so history keeps the old one."""
        if not self.doc or kind not in ("mask", "depth"):
            return
        if kind == "mask":
            self.doc.mask = self.doc.ensure_mask().copy()
        else:
            self.doc.depth_offset = self.doc.ensure_depth_offset().copy()
        self._stroke = (kind, u, v)
        self.interactionChanged.emit(True)
        self._dab(kind, u, v, alt)

    def stroke_to(self, u: float, v: float, alt: bool) -> None:
        if not self._stroke or not self.doc:
            return
        kind, u0, v0 = self._stroke
        for su, sv in stroke_samples(u0, v0, u, v, self.doc.width / self.doc.height, self.brush.size):
            self._dab(kind, su, sv, alt, emit=False)
        self._stroke = (kind, u, v)
        self._emit_layer(kind)

    def end_stroke(self) -> None:
        if not self._stroke:
            return
        kind = self._stroke[0]
        self._stroke = None
        self.interactionChanged.emit(False)
        self._commit("Paint mask" if kind == "mask" else "Edit depth", {"mask"} if kind == "mask" else {"geometry"})

    def _dab(self, kind: str, u: float, v: float, alt: bool, emit: bool = True) -> None:
        assert self.doc is not None
        if kind == "mask":
            paint_mask(self.doc.mask, u, v, self.brush, erase=alt)  # type: ignore[arg-type]
        else:
            paint_depth(self.doc.depth_offset, u, v, self.brush, push=not alt)  # type: ignore[arg-type]
        if emit:
            self._emit_layer(kind)

    def _emit_layer(self, kind: str) -> None:
        if kind == "mask":
            self.mask_version += 1
            self.changed.emit(frozenset({"mask"}))
        else:
            self.offset_version += 1
            self.changed.emit(frozenset({"geometry"}))
