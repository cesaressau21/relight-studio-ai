"""Background workers: interactive rendering, depth inference and export.

All heavy work runs off the UI thread. Rendering requests are coalesced: while
a frame is being rendered only the most recent request is kept.
"""

from __future__ import annotations

import copy
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from ..ai.depth import DepthEstimator
from ..core.document import Document
from ..core.image_io import ExportOptions, save_image
from ..core.settings import DepthSettings
from ..rendering.device import select_device
from ..rendering.engine import Renderer, RenderParams
from ..rendering.scene import Scene, build_scene, mask_tensor

log = logging.getLogger(__name__)


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


@dataclass
class RenderJob:
    image: np.ndarray
    image_key: int
    disparity: np.ndarray
    disparity_version: int
    depth: DepthSettings
    params: RenderParams
    offset: np.ndarray | None  # None = depth edits disabled / absent
    offset_version: int
    mask: np.ndarray | None
    mask_version: int
    interactive: bool = False
    stats: dict = field(default_factory=dict)


class RenderWorker(QObject):
    finished = Signal(QImage, dict)
    sceneBuilt = Signal(object)  # depth preview (np.ndarray)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.device = select_device().device
        self.renderer = Renderer()
        self._scenes: dict[float, tuple[tuple, Scene]] = {}
        self._mask_versions: dict[float, int] = {}

    def _scene(self, job: RenderJob, scale: float) -> Scene:
        key = (
            job.image_key, job.disparity_version, job.depth.geometry_key(),
            job.offset_version if job.offset is not None else -1,
        )
        cached = self._scenes.get(scale)
        if cached is None or cached[0] != key:
            image = job.image
            if scale != 1.0:
                h, w = image.shape[:2]
                image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            scene = build_scene(image, job.disparity, job.depth, self.device, job.offset, job.mask)
            job.stats["scene_ms"] = (time.perf_counter() - start) * 1000
            self._scenes[scale] = (key, scene)
            self._mask_versions[scale] = job.mask_version
            if scale == 1.0:
                self.sceneBuilt.emit(scene.depth_preview())
        else:
            scene = cached[1]
        if self._mask_versions.get(scale) != job.mask_version:
            scene.mask = mask_tensor(job.mask, scene.height, scene.width, self.device)
            self._mask_versions[scale] = job.mask_version
        return scene

    @Slot(object)
    def render(self, job: RenderJob) -> None:
        try:
            start = time.perf_counter()
            scale = 0.5 if job.interactive else 1.0
            scene = self._scene(job, scale)
            params = job.params
            if job.interactive:
                params = copy.copy(params)
                params.relight = copy.copy(params.relight)
                params.relight.shadow_steps = max(8, params.relight.shadow_steps // 2)
            rgb = self.renderer.render(scene, params)
            job.stats.update(
                render_ms=(time.perf_counter() - start) * 1000,
                size=(scene.width, scene.height),
                interactive=job.interactive,
            )
            self.finished.emit(to_qimage(rgb), job.stats)
        except Exception as exc:  # keep the UI alive, report the error
            log.exception("Render failed")
            self.failed.emit(f"Render failed: {exc}")

    @Slot()
    def warmup(self) -> None:
        """Compile/load the GPU kernels once so the first real frame is fast."""
        try:
            from ..core.lights import LightType, new_light

            start = time.perf_counter()
            image = np.full((96, 128, 3), 128, np.uint8)
            disparity = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (48, 1))
            scene = build_scene(image, disparity, DepthSettings(), self.device, mask=np.full((48, 64), 255, np.uint8))
            lights = [new_light(i + 1, t) for i, t in enumerate(LightType)]
            self.renderer.render(scene, RenderParams(lights=lights))
            log.info("Renderer warm-up %.1fs", time.perf_counter() - start)
        except Exception:
            log.exception("Renderer warm-up failed")

    @Slot()
    def drop_cache(self) -> None:
        self._scenes.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class RenderService(QObject):
    """GUI-side front end of :class:`RenderWorker` with request coalescing."""

    frameReady = Signal(QImage, dict)
    depthPreview = Signal(object)
    failed = Signal(str)
    _dispatch = Signal(object)
    _drop = Signal()
    _warm = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("RenderThread")
        self._worker = RenderWorker()
        self._worker.moveToThread(self._thread)
        self._dispatch.connect(self._worker.render)
        self._drop.connect(self._worker.drop_cache)
        self._warm.connect(self._worker.warmup)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.sceneBuilt.connect(self.depthPreview)
        self._thread.start()
        self._busy = False
        self._pending: RenderJob | None = None
        self._warm.emit()

    def request(self, job: RenderJob) -> None:
        if self._busy:
            self._pending = job
        else:
            self._busy = True
            self._dispatch.emit(job)

    def drop_cache(self) -> None:
        self._drop.emit()

    def _next(self) -> None:
        if self._pending is not None:
            job, self._pending = self._pending, None
            self._dispatch.emit(job)
        else:
            self._busy = False

    def _on_finished(self, image: QImage, stats: dict) -> None:
        self.frameReady.emit(image, stats)
        self._next()

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)
        self._next()

    def shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(3000)


class DepthWorker(QThread):
    """Runs depth inference (or a model warm-up when ``image`` is None)."""

    done = Signal(object, str, float)  # disparity, source key, seconds
    failed = Signal(str)

    def __init__(self, image: np.ndarray | None, model: str, detail: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.image, self.model, self.detail = image, model, detail

    def run(self) -> None:
        try:
            start = time.perf_counter()
            estimator = DepthEstimator.shared()
            if self.image is None:
                estimator.warmup(self.model)
                self.done.emit(None, self.model, time.perf_counter() - start)
                return
            disparity = estimator.predict(self.image, self.model, self.detail)
            self.done.emit(disparity, f"{self.model}/{self.detail}", time.perf_counter() - start)
        except Exception as exc:
            log.exception("Depth estimation failed")
            self.failed.emit(f"Depth estimation failed: {exc}\n{traceback.format_exc(limit=2)}")


def render_export(
    doc: Document,
    params: RenderParams,
    options: ExportOptions,
    device: torch.device,
    progress=None,
) -> np.ndarray:
    """Full-quality render of ``doc`` at the export resolution (no UI overlays)."""
    if doc.disparity is None:
        raise RuntimeError("Depth has not been estimated yet")
    w, h = options.target_size(doc.width, doc.height)
    image = doc.image if (w, h) == (doc.width, doc.height) else cv2.resize(doc.image, (w, h), interpolation=cv2.INTER_AREA)
    offset = doc.depth_offset if doc.mask_settings.depth_edits_enabled else None
    scene = build_scene(image, doc.disparity, doc.depth, device, offset, doc.mask)
    if progress:
        progress(0.1)
    renderer = Renderer()
    return renderer.render(scene, params, progress=(lambda f: progress(0.1 + 0.85 * f)) if progress else None)


class ExportWorker(QThread):
    progress = Signal(int)
    done = Signal(str, float)
    failed = Signal(str)

    def __init__(self, doc: Document, params: RenderParams, options: ExportOptions, path: Path,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.doc, self.params, self.options, self.path = doc, params, options, path

    def run(self) -> None:
        start = time.perf_counter()
        report = lambda f: self.progress.emit(int(f * 100))  # noqa: E731
        device = select_device().device
        try:
            try:
                rgb = render_export(self.doc, self.params, self.options, device, report)
            except torch.cuda.OutOfMemoryError:
                log.warning("Export ran out of GPU memory; retrying on the CPU")
                torch.cuda.empty_cache()
                rgb = render_export(self.doc, self.params, self.options, torch.device("cpu"), report)
            save_image(rgb, self.path, self.options)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            self.progress.emit(100)
            self.done.emit(str(self.path), time.perf_counter() - start)
        except Exception as exc:
            log.exception("Export failed")
            self.failed.emit(f"Export failed: {exc}")
