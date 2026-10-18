"""Right-hand property panels."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..ai.depth import DETAIL_SIZES, MODELS
from ..core.color import rgb_to_hex
from ..core.lights import GOBO_PATTERNS, Light, LightTarget, LightType
from ..core.presets import Preset
from ..rendering.engine import VIEW_MODES
from . import icons, theme
from .controller import EditorController
from .depth_profile import DepthProfile
from .widgets import ColorButton, SliderRow, divider, hint, labeled, section_title

LIGHT_ICON = {
    LightType.POINT: "point",
    LightType.DIRECTIONAL: "directional",
    LightType.AMBIENT: "ambient",
    LightType.AREA: "area",
    LightType.RIM: "rim",
}


def scroll_page() -> tuple[QScrollArea, QVBoxLayout]:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget()
    lay = QVBoxLayout(body)
    lay.setContentsMargins(14, 10, 14, 14)
    lay.setSpacing(4)
    area.setWidget(body)
    return area, lay


def light_swatch(light: Light, size: int = 18) -> QIcon:
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(light.gizmo_color), 2))
    p.setBrush(QColor(rgb_to_hex(light.color)))
    p.drawEllipse(2, 2, size - 4, size - 4)
    p.end()
    return QIcon(pm)


# ============================================================================ Lights
class LightsPanel(QWidget):
    LIGHT_FIELDS: list[tuple[str, str, float, float, int, str, str]] = [
        ("x", "Position X", -0.5, 1.5, 3, "", "Horizontal position (0 = left edge, 1 = right edge)"),
        ("y", "Position Y", -0.5, 1.5, 3, "", "Vertical position (0 = top edge, 1 = bottom edge)"),
        ("z", "Depth Z", -1.0, 1.5, 3, "", "0 = nearest surface, 1 = farthest; negative = toward the camera.\nCtrl + wheel on the canvas"),
        ("intensity", "Intensity", 0.0, 10.0, 2, "", "Alt + wheel on the canvas"),
        ("radius", "Radius", 0.05, 3.0, 2, "", "Distance where the light falls to half (image heights).\nShift + wheel on the canvas"),
        ("kelvin", "Temperature", 1500.0, 12000.0, 0, " K", "Color temperature, multiplied with the color"),
        ("softness", "Softness", 0.0, 1.0, 2, "", "Terminator wrap and shadow penumbra"),
        ("diffusion", "Diffusion", 0.0, 1.0, 2, "", "Fraction of light that ignores surface orientation"),
        ("attenuation", "Falloff", 0.0, 3.0, 2, "", "Distance attenuation exponent (1 ≈ inverse square)"),
        ("size", "Area size", 0.0, 0.6, 2, "", "Soft area light extent (image heights)"),
        ("specular", "Specular", 0.0, 3.0, 2, "", "Per-light highlight strength"),
    ]

    GOBO_FIELDS: list[tuple[str, str, float, float, int, str, str]] = [
        ("gobo_size", "Pattern size", 0.02, 0.6, 3, "", "Size of one pattern repeat"),
        ("gobo_rotation", "Rotation", -180.0, 180.0, 0, "°", "Pattern rotation"),
        ("gobo_softness", "Focus blur", 0.0, 1.0, 2, "", "0 = sharp pattern, 1 = very soft"),
        ("gobo_strength", "Pattern strength", 0.0, 1.0, 2, "", "How much of the light the pattern blocks"),
    ]

    def __init__(self, controller: EditorController) -> None:
        super().__init__()
        self.ctl = controller
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        area, lay = scroll_page()
        root.addWidget(area)

        head = QHBoxLayout()
        head.addWidget(section_title("Light sources"))
        head.addStretch(1)
        self.add_btn = QToolButton()
        self.add_btn.setIcon(icons.icon("plus", 16))
        self.add_btn.setToolTip("Add light")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.add_btn)
        for lt in LightType:
            act = menu.addAction(icons.icon(LIGHT_ICON[lt], 16), f"{lt.label} light")
            act.triggered.connect(lambda _=False, t=lt: self._add(t))
        self.add_btn.setMenu(menu)
        self.dup_btn = QToolButton()
        self.dup_btn.setIcon(icons.icon("duplicate", 16))
        self.dup_btn.setToolTip("Duplicate light")
        self.dup_btn.clicked.connect(lambda: self.ctl.duplicate_light(self.ctl.selected))
        self.del_btn = QToolButton()
        self.del_btn.setIcon(icons.icon("trash", 16))
        self.del_btn.setToolTip("Delete light (Del)")
        self.del_btn.clicked.connect(lambda: self.ctl.remove_light(self.ctl.selected))
        for b in (self.add_btn, self.dup_btn, self.del_btn):
            head.addWidget(b)
        lay.addLayout(head)

        self.list = QListWidget()
        self.list.setIconSize(QSize(18, 18))
        self.list.setFixedHeight(128)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.list)

        lay.addWidget(section_title("Side view"))
        self.profile = DepthProfile(controller)
        lay.addWidget(self.profile)

        lay.addWidget(section_title("Properties"))
        self.form = QWidget()
        form = QVBoxLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(3)
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set(name=self.name.text().strip() or "Light"))
        form.addWidget(labeled("Name", self.name))
        self.type_combo = QComboBox()
        for lt in LightType:
            self.type_combo.addItem(icons.icon(LIGHT_ICON[lt], 16), lt.label, lt.value)
        self.type_combo.currentIndexChanged.connect(lambda _: self._set(type=LightType(self.type_combo.currentData())))
        form.addWidget(labeled("Type", self.type_combo))
        self.target_combo = QComboBox()
        for target in LightTarget:
            self.target_combo.addItem(target.label, target.value)
        self.target_combo.setToolTip("Light only the subject (e.g. a hair light) or only the background "
                                     "(e.g. a wall wash), using the subject/background separation.")
        self.target_combo.currentIndexChanged.connect(
            lambda _: self._set(target=LightTarget(self.target_combo.currentData())))
        form.addWidget(labeled("Affects", self.target_combo))
        self.target_hint = hint("No subject is separated in this image, so this light affects everything. "
                                "See Scene → Subject edges.")
        self.target_hint.setVisible(False)
        form.addWidget(self.target_hint)
        self._subject_available = True

        color_row = QWidget()
        cl = QHBoxLayout(color_row)
        cl.setContentsMargins(0, 0, 0, 0)
        self.color = ColorButton()
        self.color.colorChanged.connect(lambda c: self._set(color=tuple(c)))
        self.enabled = QCheckBox("On")
        self.enabled.toggled.connect(lambda v: self._set(enabled=v))
        self.shadows = QCheckBox("Shadows")
        self.shadows.toggled.connect(lambda v: self._set(shadows=v))
        cl.addWidget(self.color)
        cl.addSpacing(6)
        cl.addWidget(self.enabled)
        cl.addWidget(self.shadows)
        cl.addStretch(1)
        form.addWidget(labeled("Color", color_row))

        self.rows: dict[str, SliderRow] = {}
        for key, label, lo, hi, dec, suffix, tip in self.LIGHT_FIELDS:
            row = SliderRow(label, lo, hi, 0.0, dec, suffix, tip)
            row.valueChanged.connect(lambda v, k=key: self._set(**{k: v}))
            self.rows[key] = row
            form.addWidget(row)

        form.addWidget(section_title("Pattern"))
        self.gobo_combo = QComboBox()
        for key, label in GOBO_PATTERNS.items():
            self.gobo_combo.addItem(label, key)
        self.gobo_combo.setToolTip("Project a pattern through the light (gobo): blinds, window panes, leaves…")
        self.gobo_combo.currentIndexChanged.connect(lambda _: self._gobo_chosen())
        form.addWidget(labeled("Gobo", self.gobo_combo))
        for key, label, lo, hi, dec, suffix, tip in self.GOBO_FIELDS:
            row = SliderRow(label, lo, hi, 0.0, dec, suffix, tip)
            row.valueChanged.connect(lambda v, k=key: self._set(**{k: v}))
            self.rows[key] = row
            form.addWidget(row)
        lay.addWidget(self.form)
        lay.addWidget(hint("Drag gizmos on the photo. Ctrl+wheel: depth · Shift+wheel: radius · "
                           "Alt+wheel: intensity · arrows: nudge · H: hide gizmos"))
        lay.addStretch(1)

        controller.changed.connect(self._on_changed)
        controller.selectionChanged.connect(lambda _: self.refresh())
        controller.documentLoaded.connect(self.refresh)
        self.refresh()

    def _gobo_chosen(self) -> None:
        light = self.ctl.selected_light()
        if light is None:
            return
        gobo = self.gobo_combo.currentData()
        if gobo == "custom":
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose a pattern image (white lets light through)",
                str(Path(light.gobo_image).parent) if light.gobo_image else "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
            if not path:
                with QSignalBlocker(self.gobo_combo):  # cancelled: keep the current pattern
                    self.gobo_combo.setCurrentIndex(self.gobo_combo.findData(light.gobo))
                return
            self.ctl.update_light(light.id, gobo="custom", gobo_image=path)
        else:
            self._set(gobo=gobo)

    def set_subject_available(self, available: bool) -> None:
        """Called after each render: whether the scene has a subject/background separation."""
        self._subject_available = available
        self._update_target_hint()

    def _update_target_hint(self) -> None:
        light = self.ctl.selected_light()
        targeted = light is not None and light.target is not LightTarget.ALL
        self.target_hint.setVisible(targeted and not self._subject_available)

    def _add(self, light_type: LightType) -> None:
        self.ctl.add_light(light_type)

    def _set(self, **changes) -> None:
        light = self.ctl.selected_light()
        if light is None:
            return
        if all(getattr(light, k) == v for k, v in changes.items()):
            return
        self.ctl.update_light(light.id, **changes)

    def _on_current(self, item: QListWidgetItem | None, _prev) -> None:
        if item is not None:
            self.ctl.select(item.data(Qt.ItemDataRole.UserRole))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        light_id = item.data(Qt.ItemDataRole.UserRole)
        light = self.ctl.doc.light(light_id) if self.ctl.doc else None
        enabled = item.checkState() == Qt.CheckState.Checked
        if light is not None and light.enabled != enabled:
            self.ctl.update_light(light_id, enabled=enabled)

    def _on_changed(self, kinds) -> None:
        if kinds & {"lights", "selection"}:
            self.refresh()
        elif "geometry" in kinds:
            self.profile.update()

    def refresh(self) -> None:
        doc = self.ctl.doc
        lights = doc.lights if doc else []
        with QSignalBlocker(self.list):
            self.list.clear()
            for light in lights:
                text = f"{light.name}   ·   {light.type.label}"
                if light.target is not LightTarget.ALL:
                    text += f"   ·   {light.target.label.replace(' only', '')}"
                item = QListWidgetItem(light_swatch(light), text)
                item.setData(Qt.ItemDataRole.UserRole, light.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if light.enabled else Qt.CheckState.Unchecked)
                item.setSizeHint(QSize(0, 28))
                self.list.addItem(item)
                if light.id == self.ctl.selected:
                    self.list.setCurrentItem(item)
        light = self.ctl.selected_light()
        self.form.setEnabled(light is not None)
        self.dup_btn.setEnabled(light is not None)
        self.del_btn.setEnabled(light is not None)
        self.add_btn.setEnabled(doc is not None)
        if light is not None:
            with QSignalBlocker(self.target_combo):
                self.target_combo.setCurrentIndex(self.target_combo.findData(light.target.value))
            self._update_target_hint()
            with QSignalBlocker(self.name), QSignalBlocker(self.type_combo), QSignalBlocker(self.enabled), \
                    QSignalBlocker(self.shadows):
                self.name.setText(light.name)
                self.type_combo.setCurrentIndex(self.type_combo.findData(light.type.value))
                self.enabled.setChecked(light.enabled)
                self.shadows.setChecked(light.shadows)
            self.color.set_color(light.color)
            for key, row in self.rows.items():
                row.set_value(getattr(light, key))
            positional = light.is_positional
            for key in ("x", "y", "z", "radius", "attenuation", "softness"):
                self.rows[key].setEnabled(positional)
            self.rows["size"].setEnabled(light.type is LightType.AREA)
            self.rows["specular"].setEnabled(light.type in (LightType.POINT, LightType.AREA, LightType.DIRECTIONAL))
            self.rows["radius"].setEnabled(light.type in (LightType.POINT, LightType.AREA, LightType.RIM))
            self.rows["attenuation"].setEnabled(light.type in (LightType.POINT, LightType.AREA, LightType.RIM))
            self.shadows.setEnabled(light.type in (LightType.POINT, LightType.AREA, LightType.DIRECTIONAL))
            patterned = light.type in (LightType.POINT, LightType.AREA, LightType.DIRECTIONAL)
            with QSignalBlocker(self.gobo_combo):
                self.gobo_combo.setCurrentIndex(self.gobo_combo.findData(light.gobo))
            self.gobo_combo.setEnabled(patterned)
            for key, *_ in self.GOBO_FIELDS:
                self.rows[key].setEnabled(patterned and light.gobo != "none")
        self.profile.update()


# ============================================================================ Generic form
class ParamForm(QWidget):
    """Slider rows bound to the fields of one document parameter group."""

    def __init__(self, fields: list[tuple], getter: Callable[[], object | None], setter: Callable[..., None]):
        super().__init__()
        self.getter, self.setter = getter, setter
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.rows: dict[str, SliderRow] = {}
        for key, label, lo, hi, dec, suffix, tip, default in fields:
            row = SliderRow(label, lo, hi, default, dec, suffix, tip)
            row.valueChanged.connect(lambda v, k=key: self._changed(k, v))
            self.rows[key] = row
            lay.addWidget(row)

    def _changed(self, key: str, value: float) -> None:
        target = self.getter()
        if target is None:
            return
        current = getattr(target, key)
        value = int(round(value)) if isinstance(current, int) and not isinstance(current, bool) else value
        if current != value:
            self.setter(**{key: value})

    def refresh(self) -> None:
        target = self.getter()
        self.setEnabled(target is not None)
        if target is not None:
            for key, row in self.rows.items():
                row.set_value(getattr(target, key))


# ============================================================================ Scene
class ScenePanel(QWidget):
    """Global lighting model + depth analysis controls."""

    reanalyzeRequested = Signal()

    RELIGHT_FIELDS = [
        ("original_light", "Original light", 0.0, 1.0, 2, "", "Mix of the captured illumination that is kept", 0.75),
        ("delighting", "Delighting", 0.0, 1.0, 2, "", "How strongly the original shading is removed from the albedo estimate", 0.6),
        ("ambient", "Ambient", 0.0, 1.5, 2, "", "Uniform fill light", 0.0),
        ("ambient_occlusion", "Occlusion", 0.0, 1.0, 2, "", "Screen-space cavity darkening of ambient light", 0.5),
        ("specular", "Specular", 0.0, 1.5, 2, "", "Global highlight strength (Blinn-Phong)", 0.2),
        ("roughness", "Roughness", 0.05, 1.0, 2, "", "Highlight spread", 0.5),
        ("shadow_strength", "Shadows", 0.0, 1.0, 2, "", "Opacity of screen-space cast shadows", 0.85),
        ("shadow_steps", "Shadow quality", 4, 64, 0, "", "Ray-march samples per light (higher = slower, smoother)", 24),
    ]
    DEPTH_FIELDS = [
        ("relief", "Relief", 0.05, 2.0, 2, "", "Depth range relative to the image height: steeper normals and longer shadows", 0.6),
        ("smoothing", "Smoothing", 0.0, 1.0, 2, "", "Edge-preserving smoothing of depth and normals", 0.5),
        ("near", "Near clip", 0.0, 0.9, 2, "", "Remap: depth treated as nearest", 0.0),
        ("far", "Far clip", 0.1, 1.0, 2, "", "Remap: depth treated as farthest", 1.0),
        ("gamma", "Depth gamma", 0.3, 3.0, 2, "", "Non-linear depth distribution", 1.0),
    ]

    def __init__(self, controller: EditorController) -> None:
        super().__init__()
        self.ctl = controller
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        area, lay = scroll_page()
        root.addWidget(area)

        lay.addWidget(section_title("View"))
        self.view = QComboBox()
        labels = {"result": "Result", "original": "Original", "depth": "Depth map", "normals": "Normals",
                  "albedo": "Albedo estimate", "shading": "Original shading", "lighting": "New lighting only",
                  "occlusion": "Ambient occlusion", "subject": "Subject matte"}
        for mode in VIEW_MODES:
            self.view.addItem(labels[mode], mode)
        self.view.currentIndexChanged.connect(lambda _: self.ctl.set_view(self.view.currentData()))
        lay.addWidget(labeled("Display", self.view))

        lay.addWidget(section_title("Lighting model"))
        self.ambient_color = ColorButton()
        self.ambient_color.colorChanged.connect(lambda c: self.ctl.update_relight(ambient_color=tuple(c)))
        lay.addWidget(labeled("Ambient color", self.ambient_color, stretch=False))
        self.relight_form = ParamForm(self.RELIGHT_FIELDS, lambda: self.ctl.doc and self.ctl.doc.relight,
                                      self.ctl.update_relight)
        lay.addWidget(self.relight_form)
        self.rolloff = QCheckBox("Highlight roll-off")
        self.rolloff.setToolTip("Compress values above white instead of clipping them")
        self.rolloff.toggled.connect(lambda v: self.ctl.update_relight(highlight_rolloff=v))
        lay.addWidget(self.rolloff)

        lay.addSpacing(6)
        lay.addWidget(divider())
        lay.addWidget(section_title("Depth analysis"))
        self.model = QComboBox()
        for spec in MODELS.values():
            note = "" if spec.commercial_use else "  (non-commercial)"
            self.model.addItem(f"{spec.label.replace('Depth Anything V2', 'DA-V2')} · {spec.license}{note}", spec.key)
        self.detail = QComboBox()
        for key, size in DETAIL_SIZES.items():
            self.detail.addItem(f"{key.title()} ({size}px{', flip TTA' if key == 'high' else ''})", key)
        lay.addWidget(labeled("Model", self.model))
        lay.addWidget(labeled("Detail", self.detail))
        self.model.currentIndexChanged.connect(lambda _: self._depth_source_changed())
        self.detail.currentIndexChanged.connect(lambda _: self._depth_source_changed())
        self.reanalyze = QPushButton("Re-analyze depth")
        self.reanalyze.setIcon(icons.icon("sparkle", 16))
        self.reanalyze.clicked.connect(self.reanalyzeRequested)
        lay.addWidget(self.reanalyze)
        self.source = hint("")
        lay.addWidget(self.source)
        self.depth_form = ParamForm(self.DEPTH_FIELDS, lambda: self.ctl.doc and self.ctl.doc.depth,
                                    self.ctl.update_depth)
        lay.addWidget(self.depth_form)
        self.invert = QCheckBox("Invert depth")
        self.invert.toggled.connect(lambda v: self.ctl.update_depth(invert=v))
        lay.addWidget(self.invert)
        lay.addWidget(hint("Monocular depth is relative: it ranks what is near or far but has no metric scale. "
                           "Use Relief and the depth brush to correct implausible geometry."))

        lay.addSpacing(6)
        lay.addWidget(divider())
        lay.addWidget(section_title("Subject edges"))
        self.edge_mode = QComboBox()
        for label, mode in (("AI matting (ViTMatte)", "matting"), ("Fast (color based)", "guided"), ("Off", "off")):
            self.edge_mode.addItem(label, mode)
        self.edge_mode.setToolTip("How lighting follows the outline of the subject. "
                                  "AI matting resolves hair strands; Fast uses colors only.")
        self.edge_mode.currentIndexChanged.connect(lambda _: self.ctl.update_depth(edge_mode=self.edge_mode.currentData()))
        lay.addWidget(labeled("Refinement", self.edge_mode))
        self.auto_threshold = QCheckBox("Detect subject automatically")
        self.auto_threshold.toggled.connect(self._auto_threshold_toggled)
        lay.addWidget(self.auto_threshold)
        self.edge_form = ParamForm(
            [("subject_threshold", "Subject depth", 0.02, 0.98, 2, "",
              "Normalized depth separating the subject (nearer) from the background", 0.5),
             ("edge_band", "Edge band", 0.005, 0.08, 3, "",
              "Width of the uncertain zone around the outline (image heights)", 0.02)],
            lambda: self.ctl.doc and self.ctl.doc.depth, self.ctl.update_depth)
        lay.addWidget(self.edge_form)
        self.edge_status = hint("")
        lay.addWidget(self.edge_status)
        lay.addStretch(1)

        controller.changed.connect(self._on_changed)
        controller.documentLoaded.connect(self.refresh)
        self.refresh()

    def _depth_source_changed(self) -> None:
        if not self.ctl.doc:
            return
        model, detail = self.model.currentData(), self.detail.currentData()
        if (model, detail) != (self.ctl.doc.depth.model, self.ctl.doc.depth.detail):
            self.ctl.update_depth(model=model, detail=detail)
            if not MODELS[model].commercial_use:
                self.source.setText(f"{MODELS[model].label} is licensed {MODELS[model].license}: "
                                    "non-commercial use only. Press Re-analyze to apply.")
            else:
                self.source.setText("Press Re-analyze to apply.")

    def _auto_threshold_toggled(self, auto: bool) -> None:
        if not self.ctl.doc:
            return
        value = -1.0 if auto else max(self.edge_form.rows["subject_threshold"].value(), 0.02)
        self.ctl.update_depth(subject_threshold=value)

    def set_edge_status(self, subject_found: bool, ai_matte: bool) -> None:
        doc = self.ctl.doc
        if doc is not None and doc.opacity is not None:
            self.edge_status.setText("Using the image's transparency as the subject outline: transparent "
                                     "areas are ignored by depth and lighting.")
        elif doc is None or doc.depth.edge_mode == "off":
            self.edge_status.setText("Edge refinement is off: lighting follows the raw depth edges.")
        elif not subject_found:
            self.edge_status.setText("No distinct subject found in the depth map. Set the subject depth manually "
                                     "if the scene has one.")
        elif doc.depth.edge_mode == "matting" and not ai_matte:
            self.edge_status.setText("Subject separated with the fast matte (AI matte pending).")
        else:
            source = "AI matte" if ai_matte else "fast matte"
            self.edge_status.setText(f"Subject separated with the {source}. Check it in Display → Subject matte.")

    def _on_changed(self, kinds) -> None:
        if kinds & {"relight", "depth", "geometry", "view"}:
            self.refresh()

    def refresh(self) -> None:
        doc = self.ctl.doc
        self.relight_form.refresh()
        self.depth_form.refresh()
        self.edge_form.refresh()
        for w in (self.ambient_color, self.rolloff, self.model, self.detail, self.reanalyze, self.invert,
                  self.edge_mode, self.auto_threshold):
            w.setEnabled(doc is not None)
        with QSignalBlocker(self.view):
            self.view.setCurrentIndex(self.view.findData(self.ctl.view))
        if doc is None:
            return
        self.ambient_color.set_color(doc.relight.ambient_color)
        with QSignalBlocker(self.rolloff), QSignalBlocker(self.invert), QSignalBlocker(self.model), \
                QSignalBlocker(self.detail):
            self.rolloff.setChecked(doc.relight.highlight_rolloff)
            self.invert.setChecked(doc.depth.invert)
            self.model.setCurrentIndex(self.model.findData(doc.depth.model))
            self.detail.setCurrentIndex(self.detail.findData(doc.depth.detail))
        with QSignalBlocker(self.edge_mode), QSignalBlocker(self.auto_threshold):
            self.edge_mode.setCurrentIndex(self.edge_mode.findData(doc.depth.edge_mode))
            self.auto_threshold.setChecked(doc.depth.subject_threshold < 0)
        refine = doc.depth.edge_mode != "off"
        self.auto_threshold.setEnabled(refine)
        self.edge_form.rows["subject_threshold"].setEnabled(refine and doc.depth.subject_threshold >= 0)
        self.edge_form.rows["edge_band"].setEnabled(refine)
        if doc.disparity_source:
            self.source.setText(f"Current depth: {doc.disparity_source}")


# ============================================================================ Adjust
class AdjustPanel(QWidget):
    FIELDS = [
        ("exposure", "Exposure", -3.0, 3.0, 2, " EV", "", 0.0),
        ("contrast", "Contrast", -1.0, 1.0, 2, "", "", 0.0),
        ("highlights", "Highlights", -1.0, 1.0, 2, "", "", 0.0),
        ("shadows", "Shadows", -1.0, 1.0, 2, "", "", 0.0),
        ("whites", "Whites", -1.0, 1.0, 2, "", "", 0.0),
        ("blacks", "Blacks", -1.0, 1.0, 2, "", "", 0.0),
        ("saturation", "Saturation", -1.0, 1.0, 2, "", "", 0.0),
        ("temperature", "Temperature", -1.0, 1.0, 2, "", "White balance: cool ↔ warm", 0.0),
        ("tint", "Tint", -1.0, 1.0, 2, "", "White balance: green ↔ magenta", 0.0),
        ("gamma", "Gamma", 0.4, 2.5, 2, "", "Midtone gamma correction", 1.0),
    ]

    def __init__(self, controller: EditorController) -> None:
        super().__init__()
        self.ctl = controller
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        area, lay = scroll_page()
        root.addWidget(area)
        head = QHBoxLayout()
        head.addWidget(section_title("Tone & color"))
        head.addStretch(1)
        reset = QToolButton()
        reset.setIcon(icons.icon("reset", 16))
        reset.setToolTip("Reset all adjustments")
        reset.clicked.connect(self.ctl.reset_adjustments)
        head.addWidget(reset)
        lay.addLayout(head)
        self.form = ParamForm(self.FIELDS, lambda: self.ctl.doc and self.ctl.doc.adjustments,
                              self.ctl.update_adjustments)
        lay.addWidget(self.form)
        lay.addWidget(hint("Applied after relighting. Exposure and white balance act in linear light; "
                           "tone controls act on display values."))
        lay.addStretch(1)
        controller.changed.connect(lambda k: self.form.refresh() if "adjust" in k else None)
        controller.documentLoaded.connect(self.form.refresh)
        self.form.refresh()


# ============================================================================ Layers
class LayersPanel(QWidget):
    def __init__(self, controller: EditorController, set_tool: Callable[[str], None]) -> None:
        super().__init__()
        self.ctl = controller
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        area, lay = scroll_page()
        root.addWidget(area)

        lay.addWidget(section_title("Brush"))
        self.size = SliderRow("Size", 0.005, 0.5, controller.brush.size, 3, "", "Radius in image heights ( [ / ] )")
        self.hardness = SliderRow("Hardness", 0.0, 0.99, controller.brush.hardness, 2)
        self.flow = SliderRow("Flow", 0.05, 1.0, controller.brush.flow, 2)
        self.size.valueChanged.connect(lambda v: setattr(self.ctl.brush, "size", v))
        self.hardness.valueChanged.connect(lambda v: setattr(self.ctl.brush, "hardness", v))
        self.flow.valueChanged.connect(lambda v: setattr(self.ctl.brush, "flow", v))
        for w in (self.size, self.hardness, self.flow):
            lay.addWidget(w)

        lay.addSpacing(6)
        lay.addWidget(divider())
        lay.addWidget(section_title("Relight mask"))
        lay.addWidget(hint("Where the new lighting applies. Paint (B) adds, Alt/right-drag erases. "
                           "Unpainted areas keep the original photo."))
        row = QHBoxLayout()
        self.mask_enabled = QCheckBox("Enabled")
        self.mask_invert = QCheckBox("Invert")
        self.mask_overlay = QCheckBox("Show overlay")
        for w in (self.mask_enabled, self.mask_invert, self.mask_overlay):
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)
        self.mask_enabled.toggled.connect(lambda v: self.ctl.update_mask_settings(enabled=v))
        self.mask_invert.toggled.connect(lambda v: self.ctl.update_mask_settings(invert=v))
        self.mask_overlay.toggled.connect(lambda v: self.ctl.update_mask_settings(show_overlay=v))
        grid = QGridLayout()
        for i, (text, fn) in enumerate((("Paint", lambda: set_tool("mask")), ("Fill", self.ctl.fill_mask),
                                         ("Clear", self.ctl.clear_mask), ("Invert", self.ctl.invert_mask))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            grid.addWidget(b, 0, i)
        lay.addLayout(grid)

        lay.addSpacing(6)
        lay.addWidget(divider())
        lay.addWidget(section_title("Depth corrections"))
        lay.addWidget(hint("Depth brush (D): drag pushes surfaces away, Alt/right-drag pulls them closer. "
                           "Check the result in View → Depth map."))
        self.depth_strength = SliderRow("Strength", 0.002, 0.1, controller.brush.depth_strength, 3)
        self.depth_strength.valueChanged.connect(lambda v: setattr(self.ctl.brush, "depth_strength", v))
        lay.addWidget(self.depth_strength)
        row2 = QHBoxLayout()
        self.depth_enabled = QCheckBox("Enabled")
        self.depth_enabled.toggled.connect(lambda v: self.ctl.update_mask_settings(depth_edits_enabled=v))
        row2.addWidget(self.depth_enabled)
        row2.addStretch(1)
        b1 = QPushButton("Brush")
        b1.clicked.connect(lambda: set_tool("depth"))
        b2 = QPushButton("Clear")
        b2.clicked.connect(self.ctl.clear_depth_edits)
        row2.addWidget(b1)
        row2.addWidget(b2)
        lay.addLayout(row2)
        lay.addStretch(1)

        controller.changed.connect(lambda k: self.refresh() if k & {"mask", "geometry"} else None)
        controller.documentLoaded.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        doc = self.ctl.doc
        self.setEnabled(doc is not None)
        if doc is None:
            return
        ms = doc.mask_settings
        with QSignalBlocker(self.mask_enabled), QSignalBlocker(self.mask_invert), \
                QSignalBlocker(self.mask_overlay), QSignalBlocker(self.depth_enabled):
            self.mask_enabled.setChecked(ms.enabled)
            self.mask_invert.setChecked(ms.invert)
            self.mask_overlay.setChecked(ms.show_overlay)
            self.depth_enabled.setChecked(ms.depth_edits_enabled)
        self.size.set_value(self.ctl.brush.size)


# ============================================================================ Presets
class PresetCard(QFrame):
    clicked = Signal()

    def __init__(self, preset: Preset) -> None:
        super().__init__()
        self.setObjectName("Card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preset = preset
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        sw = QLabel()
        sw.setPixmap(self._swatch())
        lay.addWidget(sw)
        text = QVBoxLayout()
        text.setSpacing(0)
        name = QLabel(preset.name)
        name.setStyleSheet("font-weight: 600;")
        desc = QLabel(preset.description)
        desc.setObjectName("Hint")
        desc.setWordWrap(True)
        text.addWidget(name)
        text.addWidget(desc)
        lay.addLayout(text, 1)
        self.setToolTip(f"Apply “{preset.name}” ({len(preset.lights)} lights). Replaces the current light rig.")

    def _swatch(self) -> QPixmap:
        size = 34
        pm = QPixmap(size * 2, size * 2)
        pm.setDevicePixelRatio(2)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        from PySide6.QtGui import QConicalGradient

        colors = self.preset.swatch or ["#ffffff"]
        grad = QConicalGradient(size / 2, size / 2, 90)
        for i, c in enumerate(colors + colors[:1]):
            grad.setColorAt(i / len(colors), QColor(c))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(0, 0, size, size, 8, 8)
        p.end()
        return pm

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.clicked.emit()


class PresetsPanel(QWidget):
    def __init__(self, controller: EditorController, presets: list[Preset]) -> None:
        super().__init__()
        self.ctl = controller
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        area, lay = scroll_page()
        root.addWidget(area)
        lay.addWidget(section_title("Lighting presets"))
        lay.addWidget(hint("Presets set real engine parameters: light rig, lighting model and grading. "
                           "Every value stays editable and the change is undoable."))
        self.cards: list[PresetCard] = []
        for preset in presets:
            card = PresetCard(preset)
            card.clicked.connect(lambda p=preset: self.ctl.apply_preset(p))
            lay.addWidget(card)
            self.cards.append(card)
        lay.addStretch(1)
        controller.documentLoaded.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        for card in self.cards:
            card.setEnabled(self.ctl.has_document)
