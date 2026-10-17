"""Main application window."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QIcon, QDragEnterEvent, QDropEvent, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..ai.depth import MODELS
from ..core.document import Document
from ..core.image_io import SUPPORTED_INPUT, load_image
from ..core.lights import LightType
from ..core.paths import assets_dir
from ..core.presets import load_presets
from ..core.project_io import PROJECT_EXTENSION, ProjectError, build_document, read_project, save_project
from ..rendering.device import select_device, vram_usage
from ..rendering.engine import RenderParams
from . import icons, theme
from .canvas import Canvas
from .controller import EditorController
from .export_dialog import ExportDialog
from .panels import AdjustPanel, LayersPanel, LightsPanel, PresetsPanel, ScenePanel
from .workers import DepthWorker, ExportWorker, MatteWorker, RenderJob, RenderService, to_qimage

log = logging.getLogger(__name__)

IMAGE_FILTER = "Images (" + " ".join(f"*{e}" for e in SUPPORTED_INPUT) + ")"
PROJECT_FILTER = f"Relight Studio project (*{PROJECT_EXTENSION})"


def tool_button(icon_name: str, tip: str, checkable: bool = False, size: int = 20) -> QToolButton:
    b = QToolButton()
    b.setIcon(icons.icon(icon_name, size))
    b.setIconSize(QSize(size, size))
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    return b


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        app_icon = assets_dir() / "icon.png"
        self.setWindowIcon(QIcon(str(app_icon)) if app_icon.exists() else icons.icon("sparkle", 32))
        self.resize(1600, 980)
        self.setAcceptDrops(True)
        # INI file under the user profile (relocatable with QSettings.setPath, e.g. in tests).
        self.settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "RelightStudioAI", "RelightStudioAI")

        self.device_info = select_device()
        self.ctl = EditorController()
        self.render_service = RenderService(self)
        self._depth_worker: DepthWorker | None = None
        self._depth_rerun = False
        self._matte_worker: MatteWorker | None = None
        self._matte_rerun = False
        self._export_worker: ExportWorker | None = None
        self._image_key = 0
        self._offset_copy: tuple[int, np.ndarray | None] = (-1, None)
        self._mask_copy: tuple[int, np.ndarray | None] = (-1, None)
        self._interactive = False
        self._last_full_ms = 0.0
        self._model_ready = False

        self._build_ui()
        self._connect()
        self._update_actions()
        self._update_gpu()
        self._start_warmup()

    # ================================================================== layout
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_topbar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_tool_rail())
        self.canvas = Canvas(self.ctl)
        body.addWidget(self.canvas, 1)
        body.addWidget(self._build_side_panel())
        root.addLayout(body, 1)
        self.setCentralWidget(central)
        self._build_statusbar()

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(48)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(4)
        logo = QLabel()
        logo.setPixmap(icons.logo_pixmap(22))
        lay.addWidget(logo)
        lay.addSpacing(6)
        brand = QLabel("Relight Studio")
        brand.setObjectName("Brand")
        brand_ai = QLabel("AI")
        brand_ai.setObjectName("BrandAccent")
        lay.addWidget(brand)
        lay.addWidget(brand_ai)
        lay.addSpacing(18)

        self.open_btn = self._top_action("open", "Open", "Open photo or project (Ctrl+O)")
        self.save_btn = self._top_action("save", "Save", "Save project (Ctrl+S)")
        lay.addWidget(self.open_btn)
        lay.addWidget(self.save_btn)
        sep = QFrame()
        sep.setFixedSize(1, 22)
        sep.setStyleSheet(f"background: {theme.BORDER};")
        lay.addSpacing(6)
        lay.addWidget(sep)
        lay.addSpacing(6)
        self.undo_btn = tool_button("undo", "Undo (Ctrl+Z)", size=18)
        self.redo_btn = tool_button("redo", "Redo (Ctrl+Y)", size=18)
        lay.addWidget(self.undo_btn)
        lay.addWidget(self.redo_btn)
        lay.addStretch(1)
        self.title_label = QLabel("")
        self.title_label.setObjectName("Value")
        lay.addWidget(self.title_label)
        lay.addStretch(1)
        self.compare_btn = self._top_action("compare", "Compare", "Before / after split view (C).\nHold \\ to see the original")
        self.compare_btn.setCheckable(True)
        lay.addWidget(self.compare_btn)
        lay.addSpacing(6)
        self.export_btn = QToolButton()
        self.export_btn.setObjectName("Primary")
        self.export_btn.setText("  Export")
        self.export_btn.setIcon(icons.icon("export", 16))
        self.export_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.export_btn.setToolTip("Export the relit image (Ctrl+E)")
        lay.addWidget(self.export_btn)
        return bar

    def _top_action(self, icon_name: str, text: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("TopAction")
        b.setIcon(icons.icon(icon_name, 18))
        b.setText(text)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        b.setToolTip(tip)
        return b

    def _build_tool_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("ToolRail")
        rail.setFixedWidth(52)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(6, 10, 6, 10)
        lay.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict[str, QToolButton] = {}

        def add_tool(tool: str, icon_name: str, tip: str) -> None:
            b = tool_button(icon_name, tip, checkable=True)
            b.clicked.connect(lambda: self.set_tool(tool))
            self.tool_group.addButton(b)
            self.tool_buttons[tool] = b
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)

        def label(text: str) -> None:
            lbl = QLabel(text)
            lbl.setObjectName("Hint")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)

        add_tool("select", "select", "Select & move lights (V)")
        lay.addSpacing(6)
        label("LIGHT")
        add_tool("place:point", "point", "Place point light (P)")
        add_tool("place:area", "area", "Place soft area light (A)")
        add_tool("place:directional", "directional", "Place directional light (L)")
        add_tool("place:rim", "rim", "Place rim light (R)")
        self.ambient_btn = tool_button("ambient", "Add ambient light")
        self.ambient_btn.clicked.connect(lambda: self.ctl.add_light(LightType.AMBIENT))
        lay.addWidget(self.ambient_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(6)
        label("PAINT")
        add_tool("mask", "mask", "Relight mask brush (B)\nAlt / right-drag erases · [ ] size")
        add_tool("depth", "depth", "Depth correction brush (D)\nDrag pushes away · Alt / right-drag pulls closer")
        lay.addSpacing(6)
        label("VIEW")
        add_tool("pan", "hand", "Pan (hold Space)")
        add_tool("zoom", "zoom", "Zoom (Z) · Alt-click zooms out")
        self.fit_btn = tool_button("fit", "Fit to window (Ctrl+0)")
        self.fit_btn.clicked.connect(self.canvas_fit)
        lay.addWidget(self.fit_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self.actual_btn = QToolButton()
        self.actual_btn.setText("1:1")
        self.actual_btn.setToolTip("Actual pixels (Ctrl+1)")
        self.actual_btn.clicked.connect(lambda: self.canvas.set_zoom(1.0))
        lay.addWidget(self.actual_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(1)
        self.tool_buttons["select"].setChecked(True)
        return rail

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("Panel")
        panel.setFixedWidth(360)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.lights_panel = LightsPanel(self.ctl)
        self.scene_panel = ScenePanel(self.ctl)
        self.adjust_panel = AdjustPanel(self.ctl)
        self.layers_panel = LayersPanel(self.ctl, self.set_tool)
        try:
            presets = load_presets()
        except Exception:
            log.exception("Could not load presets")
            presets = []
        self.presets_panel = PresetsPanel(self.ctl, presets)
        self.tabs.addTab(self.lights_panel, "Lights")
        self.tabs.addTab(self.scene_panel, "Scene")
        self.tabs.addTab(self.adjust_panel, "Adjust")
        self.tabs.addTab(self.layers_panel, "Masks")
        self.tabs.addTab(self.presets_panel, "Presets")
        lay.addWidget(self.tabs)
        return panel

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.res_label = QLabel("")
        self.zoom_label = QLabel("")
        self.render_label = QLabel("")
        self.gpu_label = QLabel("")
        sb.addWidget(self.status_label, 1)
        sb.addWidget(self.progress)
        for w in (self.res_label, self.zoom_label, self.render_label, self.gpu_label):
            sb.addPermanentWidget(w)
        self.setStatusBar(sb)

    # ================================================================== wiring
    def _connect(self) -> None:
        self.open_btn.clicked.connect(self.open_dialog)
        self.save_btn.clicked.connect(self.save_project)
        self.export_btn.clicked.connect(self.export_image)
        self.undo_btn.clicked.connect(self.ctl.undo)
        self.redo_btn.clicked.connect(self.ctl.redo)
        self.compare_btn.toggled.connect(self._set_compare)
        self.scene_panel.reanalyzeRequested.connect(self.run_depth)

        self.ctl.changed.connect(self._on_changed)
        self.ctl.historyChanged.connect(self._update_actions)
        self.ctl.documentLoaded.connect(self._on_document_loaded)
        self.ctl.interactionChanged.connect(self._on_interaction)
        self.canvas.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.canvas.toolChanged.connect(self._sync_tool_buttons)

        self.render_service.frameReady.connect(self._on_frame)
        self.render_service.failed.connect(lambda m: self.set_status(m, error=True))
        self.render_service.depthPreview.connect(self.lights_panel.profile.set_depth)

        shortcuts = {
            QKeySequence.StandardKey.Open: self.open_dialog,
            QKeySequence.StandardKey.Save: self.save_project,
            QKeySequence("Ctrl+Shift+S"): lambda: self.save_project(save_as=True),
            QKeySequence("Ctrl+E"): self.export_image,
            QKeySequence.StandardKey.Undo: self.ctl.undo,
            QKeySequence("Ctrl+Y"): self.ctl.redo,
            QKeySequence("Ctrl+Shift+Z"): self.ctl.redo,
            QKeySequence("Ctrl+0"): self.canvas_fit,
            QKeySequence("Ctrl+1"): lambda: self.canvas.set_zoom(1.0),
            QKeySequence("Ctrl+="): lambda: self.canvas.zoom_step(1.25),
            QKeySequence("Ctrl+-"): lambda: self.canvas.zoom_step(0.8),
            QKeySequence("C"): self.compare_btn.toggle,
            QKeySequence("V"): lambda: self.set_tool("select"),
            QKeySequence("P"): lambda: self.set_tool("place:point"),
            QKeySequence("A"): lambda: self.set_tool("place:area"),
            QKeySequence("L"): lambda: self.set_tool("place:directional"),
            QKeySequence("R"): lambda: self.set_tool("place:rim"),
            QKeySequence("B"): lambda: self.set_tool("mask"),
            QKeySequence("D"): lambda: self.set_tool("depth"),
            QKeySequence("Z"): lambda: self.set_tool("zoom"),
            QKeySequence("Escape"): lambda: self.set_tool("select"),
        }
        for seq, fn in shortcuts.items():
            sc = QShortcut(seq, self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)

        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self._update_gpu)
        self.gpu_timer.start(2000)
        self.interaction_timer = QTimer(self)
        self.interaction_timer.setSingleShot(True)
        self.interaction_timer.setInterval(150)
        self.interaction_timer.timeout.connect(self._request_render)
        self.matte_timer = QTimer(self)
        self.matte_timer.setSingleShot(True)
        self.matte_timer.setInterval(300)
        self.matte_timer.timeout.connect(self.run_matte)

    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)

    def _sync_tool_buttons(self, tool: str) -> None:
        button = self.tool_buttons.get(tool)
        if button:
            button.setChecked(True)
        if tool in ("mask", "depth"):
            self.tabs.setCurrentWidget(self.layers_panel)

    def canvas_fit(self) -> None:
        self.canvas.fit()

    def _set_compare(self, on: bool) -> None:
        self.canvas.compare_split = on
        self.canvas.update()

    def set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {theme.DANGER};" if error else "")
        (log.error if error else log.info)(text)

    def _update_actions(self) -> None:
        has = self.ctl.has_document
        self.undo_btn.setEnabled(self.ctl.history.can_undo())
        self.redo_btn.setEnabled(self.ctl.history.can_redo())
        self.undo_btn.setToolTip(f"Undo {self.ctl.history.undo_label()} (Ctrl+Z)")
        self.redo_btn.setToolTip(f"Redo {self.ctl.history.redo_label()} (Ctrl+Y)")
        self.save_btn.setEnabled(has)
        self.export_btn.setEnabled(has and self.ctl.doc.disparity is not None)
        self.compare_btn.setEnabled(has)
        for name, b in self.tool_buttons.items():
            b.setEnabled(has or name == "select")
        self.ambient_btn.setEnabled(has)
        self._update_title()

    def _update_title(self) -> None:
        doc = self.ctl.doc
        if doc is None:
            self.setWindowTitle(APP_NAME)
            self.title_label.setText("")
            return
        name = (self.ctl.project_path or doc.image_path or Path("Untitled")).name
        mark = " •" if self.ctl.dirty else ""
        self.title_label.setText(f"{name}{mark}")
        self.setWindowTitle(f"{name}{mark} — {APP_NAME}")

    def _update_gpu(self) -> None:
        info = self.device_info
        usage = vram_usage()
        if usage:
            used, total = usage
            self.gpu_label.setText(f"GPU {info.name.replace('NVIDIA GeForce ', '')} · {used / 2**30:.1f}/{total / 2**30:.0f} GB")
        else:
            self.gpu_label.setText("CPU mode")
            self.gpu_label.setStyleSheet(f"color: {theme.ACCENT};")
            self.gpu_label.setToolTip(info.warning)

    # ================================================================== model
    def _start_warmup(self) -> None:
        if self.device_info.warning:
            self.set_status(self.device_info.warning, error=not self.device_info.cuda)
        worker = DepthWorker(None, "small", "standard", self)
        worker.done.connect(self._on_warmup)
        worker.failed.connect(lambda m: self.set_status(m.splitlines()[0], error=True))
        self._warmup_worker = worker
        if not self.device_info.warning:
            self.set_status("Loading depth model…")
        worker.start()

    def _on_warmup(self, _disp, _key, seconds: float) -> None:
        self._model_ready = True
        if self._depth_worker is None:
            self.set_status(f"Depth model ready ({seconds:.1f}s) — {MODELS['small'].label}, {MODELS['small'].license}")

    def run_depth(self) -> None:
        doc = self.ctl.doc
        if doc is None:
            return
        if self._depth_worker is not None:
            self._depth_rerun = True  # restart for the current document when the running job ends
            return
        self._depth_rerun = False
        model, detail = doc.depth.model, doc.depth.detail
        spec = MODELS[model]
        self.set_status(f"Analyzing depth with {spec.label} ({detail})…")
        self.canvas.busy_text = "Estimating depth…"
        self.canvas.update()
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        worker = DepthWorker(self.ctl.preview_image, model, detail, self)
        worker.done.connect(lambda disp, src, sec, target=doc: self._on_depth(target, disp, src, sec))
        worker.failed.connect(self._on_depth_failed)
        worker.finished.connect(self._depth_finished)
        self._depth_worker = worker
        worker.start()

    def _depth_finished(self) -> None:
        self._depth_worker = None
        if self._depth_rerun:
            self.run_depth()

    def _on_depth(self, target: Document, disparity, source: str, seconds: float) -> None:
        if target is not self.ctl.doc:
            return  # result belongs to a document that is no longer open
        self.progress.setVisible(False)
        self.canvas.busy_text = ""
        self.ctl.set_disparity(disparity, source)
        self.set_status(f"Depth estimated in {seconds:.2f}s ({source}, {disparity.shape[1]}×{disparity.shape[0]})")
        self._update_actions()

    def _on_depth_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.canvas.busy_text = ""
        self.canvas.update()
        self.set_status(message.splitlines()[0], error=True)
        QMessageBox.critical(self, "Depth estimation failed", message)

    # ================================================================== render
    def _on_document_loaded(self) -> None:
        self._image_key += 1
        self.canvas.set_original(to_qimage(self.ctl.preview_image))
        self.canvas.fit()
        doc = self.ctl.doc
        ph, pw = self.ctl.preview_image.shape[:2]
        self.res_label.setText(f"{doc.width}×{doc.height} · preview {pw}×{ph}")
        self.render_service.drop_cache()
        self._update_actions()

    def _on_changed(self, kinds) -> None:
        self._update_title()
        if kinds & {"mask"} and self.ctl.doc is not None:
            self.canvas.set_mask_overlay(self.ctl.doc.mask if self.ctl.doc.mask_settings.show_overlay else None)
        if kinds - {"selection"}:
            self._request_render()
        if kinds & {"geometry", "depth"} and "matte" not in kinds:
            self.matte_timer.start()  # debounced: sliders fire many changes
        self.canvas.update()

    # ================================================================== matte
    def run_matte(self) -> None:
        """Compute the AI subject matte when the edge mode needs it and it is out of date."""
        doc = self.ctl.doc
        if doc is None or doc.disparity is None or doc.depth.edge_mode != "matting":
            return
        key = self.ctl.matte_key()
        if doc.alpha_key == key:
            return
        if self._matte_worker is not None:
            self._matte_rerun = True
            return
        self._matte_rerun = False
        offset = doc.depth_offset.copy() if (doc.depth_offset is not None and doc.mask_settings.depth_edits_enabled) else None
        worker = MatteWorker(self.ctl.preview_image, doc.disparity, copy.deepcopy(doc.depth), offset, key, self)
        worker.done.connect(lambda alpha, k, sec, target=doc: self._on_matte(target, alpha, k, sec))
        worker.failed.connect(lambda m: self.set_status(m, error=True))
        worker.finished.connect(self._matte_finished)
        self._matte_worker = worker
        self.set_status("Refining subject edges…")
        worker.start()

    def _matte_finished(self) -> None:
        self._matte_worker = None
        if self._matte_rerun:
            self.run_matte()

    def _on_matte(self, target: Document, alpha, key: tuple, seconds: float) -> None:
        if target is not self.ctl.doc or key != self.ctl.matte_key():
            self._matte_rerun = True  # settings changed meanwhile: compute again
            return
        self.ctl.set_alpha(alpha, key)
        if alpha is None:
            self.set_status("No distinct subject in depth: edge refinement inactive")
        else:
            self.set_status(f"Subject edges refined with AI matting ({seconds * 1000:.0f} ms)")

    def _on_interaction(self, active: bool) -> None:
        # Drop to half resolution while dragging only when full frames are slow.
        self._interactive = active and self._last_full_ms > 45.0
        if not active:
            self.interaction_timer.start()

    def _layer_copy(self, cache: tuple[int, np.ndarray | None], version: int, layer: np.ndarray | None):
        if cache[0] == version:
            return cache
        return version, (layer.copy() if layer is not None else None)

    def _request_render(self) -> None:
        doc = self.ctl.doc
        if doc is None or doc.disparity is None:
            return
        self._offset_copy = self._layer_copy(self._offset_copy, self.ctl.offset_version, doc.depth_offset)
        self._mask_copy = self._layer_copy(self._mask_copy, self.ctl.mask_version, doc.mask)
        use_offset = doc.mask_settings.depth_edits_enabled
        job = RenderJob(
            image=self.ctl.preview_image,
            image_key=self._image_key,
            disparity=doc.disparity,
            disparity_version=self.ctl.disparity_version,
            depth=copy.copy(doc.depth),
            params=self._params(),
            offset=self._offset_copy[1] if use_offset else None,
            offset_version=self._offset_copy[0],
            mask=self._mask_copy[1],
            mask_version=self._mask_copy[0],
            alpha=self.ctl.current_alpha(),
            alpha_version=self.ctl.alpha_version,
            interactive=self._interactive,
        )
        self.render_service.request(job)

    def _params(self, view: str | None = None) -> RenderParams:
        doc = self.ctl.doc
        return RenderParams(
            lights=[l.copy() for l in doc.lights],
            relight=copy.deepcopy(doc.relight),
            adjustments=copy.deepcopy(doc.adjustments),
            mask_settings=copy.deepcopy(doc.mask_settings),
            view=view or self.ctl.view,
        )

    def _on_frame(self, image: QImage, stats: dict) -> None:
        self.canvas.set_result(image)
        ms = stats.get("render_ms", 0.0)
        if not stats.get("interactive"):
            self._last_full_ms = ms - stats.get("scene_ms", 0.0)
        w, h = stats.get("size", (0, 0))
        extra = f" · scene {stats['scene_ms']:.0f} ms" if "scene_ms" in stats else ""
        mode = " · draft" if stats.get("interactive") else ""
        self.render_label.setText(f"Render {ms:.0f} ms ({w}×{h}{mode}){extra}")
        if "subject" in stats and not stats.get("interactive"):
            self.scene_panel.set_edge_status(stats["subject"], self.ctl.current_alpha() is not None)

    # ================================================================== files
    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        start = self.settings.value("last_dir", str(Path.home() / "Pictures"))
        path, _ = QFileDialog.getOpenFileName(self, "Open photo or project", start,
                                              f"{IMAGE_FILTER};;{PROJECT_FILTER};;All files (*)")
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        self.settings.setValue("last_dir", str(path.parent))
        if path.suffix.lower() == PROJECT_EXTENSION:
            self.open_project(path)
        else:
            self.open_image(path)

    def open_image(self, path: Path) -> None:
        try:
            start = time.perf_counter()
            image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open image", f"{path}\n\n{exc}")
            return
        doc = Document(image=image, image_path=path)
        doc.add_light(LightType.POINT, name="Key", x=0.28, y=0.3, z=-0.35)
        self.ctl.load_document(doc)
        self.set_status(f"Opened {path.name} ({doc.width}×{doc.height}) in {time.perf_counter() - start:.2f}s")
        self.run_depth()

    def open_project(self, path: Path) -> None:
        try:
            data = read_project(path)
        except ProjectError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))
            return
        image_path = next((p for p in data.candidate_image_paths(path) if p.exists()), None)
        if image_path is None:
            QMessageBox.warning(self, "Source photo not found",
                                f"The project references:\n{data.manifest['image']['path']}\n\nPlease locate it.")
            located, _ = QFileDialog.getOpenFileName(self, "Locate source photo", str(path.parent), IMAGE_FILTER)
            if not located:
                return
            image_path = Path(located)
        try:
            image = load_image(image_path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open image", f"{image_path}\n\n{exc}")
            return
        expected = (data.manifest["image"].get("width"), data.manifest["image"].get("height"))
        if expected != (image.shape[1], image.shape[0]):
            QMessageBox.warning(self, "Different photo", "The photo dimensions differ from the project; "
                                "masks and depth edits were reset.")
            data.mask = data.depth_offset = data.disparity = None
        doc = build_document(data, image, image_path)
        self.ctl.load_document(doc, project_path=path)
        self.set_status(f"Opened project {path.name}")
        wanted = f"{doc.depth.model}/{doc.depth.detail}"
        if doc.disparity is None or doc.disparity_source != wanted:
            self.run_depth()

    def save_project(self, save_as: bool = False) -> bool:
        doc = self.ctl.doc
        if doc is None:
            return False
        path = self.ctl.project_path
        if path is None or save_as:
            default = (doc.image_path.with_suffix(PROJECT_EXTENSION) if doc.image_path else Path("untitled" + PROJECT_EXTENSION))
            chosen, _ = QFileDialog.getSaveFileName(self, "Save project", str(default), PROJECT_FILTER)
            if not chosen:
                return False
            path = Path(chosen).with_suffix(PROJECT_EXTENSION)
        try:
            save_project(doc, path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot save project", str(exc))
            return False
        self.ctl.project_path = path
        self.ctl.dirty = False
        self._update_title()
        self.set_status(f"Saved project {path}")
        return True

    def export_image(self) -> None:
        doc = self.ctl.doc
        if doc is None or doc.disparity is None or self._export_worker is not None:
            return
        base = doc.image_path or Path.home() / "relit"
        default = base.with_name(base.stem + "_relit.png")
        dialog = ExportDialog(doc.width, doc.height, Path(self.settings.value("export_dir", str(default.parent))) / default.name, self)
        if not dialog.exec():
            return
        path, options = dialog.target_path(), dialog.options()
        self.settings.setValue("export_dir", str(path.parent))
        snapshot = Document(
            image=doc.image, image_path=doc.image_path, relight=copy.deepcopy(doc.relight),
            depth=copy.deepcopy(doc.depth), adjustments=copy.deepcopy(doc.adjustments),
            mask_settings=copy.deepcopy(doc.mask_settings), disparity=doc.disparity,
            mask=None if doc.mask is None else doc.mask.copy(),
            depth_offset=None if doc.depth_offset is None else doc.depth_offset.copy(),
            alpha=self.ctl.current_alpha(),
        )
        worker = ExportWorker(snapshot, self._params(view="result"), options, path, self)
        worker.progress.connect(self.progress.setValue)
        worker.done.connect(self._on_exported)
        worker.failed.connect(lambda m: (self.set_status(m, error=True), QMessageBox.critical(self, "Export failed", m)))
        worker.finished.connect(self._export_finished)
        self._export_worker = worker
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.set_status(f"Exporting {path.name}…")
        self.export_btn.setEnabled(False)
        worker.start()

    def _on_exported(self, path: str, seconds: float) -> None:
        self.set_status(f"Exported {path} in {seconds:.1f}s")

    def _export_finished(self) -> None:
        self._export_worker = None
        self.progress.setVisible(False)
        self._update_actions()

    def _confirm_discard(self) -> bool:
        if not (self.ctl.doc and self.ctl.dirty):
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", "Save the current project before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    # ================================================================== events
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802
        urls = e.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in (*SUPPORTED_INPUT, PROJECT_EXTENSION):
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        path = Path(e.mimeData().urls()[0].toLocalFile())
        if self._confirm_discard():
            self.open_path(path)

    def closeEvent(self, e: QCloseEvent) -> None:  # noqa: N802
        if not self._confirm_discard():
            e.ignore()
            return
        for worker in (self._depth_worker, self._export_worker, self._matte_worker, getattr(self, "_warmup_worker", None)):
            if worker is not None:
                worker.wait(10000)
        self.render_service.shutdown()
        e.accept()
