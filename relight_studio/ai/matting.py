"""Subject matting with ViTMatte (Hugging Face Transformers).

``hustvl/vitmatte-small-composition-1k`` is Apache-2.0 and loads natively in
Transformers (no remote code). ViTMatte needs a trimap; it is derived from the
estimated depth (see :func:`relight_studio.rendering.edges.trimap_from_depth`),
so no user input is required.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ..core.paths import models_dir
from ..rendering.device import select_device

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MattingModelSpec:
    repo_id: str
    label: str
    license: str
    size_mb: int


MATTING_MODEL = MattingModelSpec("hustvl/vitmatte-small-composition-1k", "ViTMatte Small", "Apache-2.0", 100)


class SubjectMatter:
    """Process-wide ViTMatte holder (loaded once, on first use)."""

    _instance: "SubjectMatter | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self.device = select_device().device

    @classmethod
    def shared(cls) -> "SubjectMatter":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SubjectMatter()
            return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, VitMatteForImageMatting

        spec, cache = MATTING_MODEL, str(models_dir())
        start = time.perf_counter()
        try:
            processor = AutoImageProcessor.from_pretrained(spec.repo_id, cache_dir=cache, local_files_only=True)
            model = VitMatteForImageMatting.from_pretrained(spec.repo_id, cache_dir=cache, local_files_only=True)
        except OSError:
            log.info("Downloading %s (~%d MB) from Hugging Face", spec.repo_id, spec.size_mb)
            processor = AutoImageProcessor.from_pretrained(spec.repo_id, cache_dir=cache)
            model = VitMatteForImageMatting.from_pretrained(spec.repo_id, cache_dir=cache)
        self._processor, self._model = processor, model.to(self.device).eval()
        log.info("Loaded %s on %s in %.1fs", spec.repo_id, self.device, time.perf_counter() - start)

    @torch.inference_mode()
    def predict(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Alpha matte in [0, 1] (float32, H x W) for ``image`` (uint8 RGB) and ``trimap``.

        The trimap is uint8 with 0 = background, 128 = unknown, 255 = subject.
        """
        h, w = image.shape[:2]
        with self._lock:
            self._load()
            start = time.perf_counter()
            inputs = self._processor(images=Image.fromarray(image), trimaps=Image.fromarray(trimap, "L"),
                                     return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    alpha = self._model(pixel_values=pixels).alphas
            else:
                alpha = self._model(pixel_values=pixels).alphas
            result = alpha[0, 0, :h, :w].float().clamp(0, 1).cpu().numpy()
        # Keep the trimap's certain regions exact; the model only decides the band.
        result[trimap >= 255] = 1.0
        result[trimap == 0] = 0.0
        log.info("Matting %dx%d in %.0f ms", w, h, (time.perf_counter() - start) * 1000)
        return result


def matte_subject(
    image: np.ndarray,
    disparity: np.ndarray,
    settings,
    depth_offset: np.ndarray | None = None,
    matter: "SubjectMatter | None" = None,
) -> np.ndarray | None:
    """Subject matte (uint8, image resolution) from depth + ViTMatte, or None without a distinct subject."""
    from ..rendering.edges import trimap_from_depth
    from ..rendering.scene import edge_band_px, resolve_threshold, scene_depth

    h, w = image.shape[:2]
    matter = matter or SubjectMatter.shared()
    depth = scene_depth(h, w, disparity, settings, matter.device, depth_offset)
    threshold, separable = resolve_threshold(depth, settings)
    if not separable:
        return None
    trimap = trimap_from_depth(depth, threshold, edge_band_px(settings, h))[0, 0]
    trimap_u8 = (trimap * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    trimap_u8[trimap_u8 == 127] = 128
    alpha = matter.predict(image, trimap_u8)
    return (alpha * 255 + 0.5).astype(np.uint8)
