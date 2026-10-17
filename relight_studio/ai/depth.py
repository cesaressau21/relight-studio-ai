"""Monocular depth estimation with Depth Anything V2 (Hugging Face Transformers).

Only the *Small* checkpoint is Apache-2.0 and therefore the default. *Base* and
*Large* are CC-BY-NC-4.0 (non-commercial) and must be selected explicitly.

The model predicts *relative inverse depth* (disparity): larger values are
closer to the camera and the scale/shift are unknown. The scene builder turns
it into a normalized depth range.
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
class DepthModelSpec:
    key: str
    repo_id: str
    label: str
    license: str
    commercial_use: bool
    size_mb: int


MODELS: dict[str, DepthModelSpec] = {
    "small": DepthModelSpec(
        "small", "depth-anything/Depth-Anything-V2-Small-hf",
        "Depth Anything V2 Small", "Apache-2.0", True, 99,
    ),
    "base": DepthModelSpec(
        "base", "depth-anything/Depth-Anything-V2-Base-hf",
        "Depth Anything V2 Base", "CC-BY-NC-4.0", False, 390,
    ),
    "large": DepthModelSpec(
        "large", "depth-anything/Depth-Anything-V2-Large-hf",
        "Depth Anything V2 Large", "CC-BY-NC-4.0", False, 1340,
    ),
}

# Network input resolution (multiples of 14, the ViT patch size).
DETAIL_SIZES = {"standard": 518, "high": 756}


class DepthEstimator:
    """Process-wide model holder: each checkpoint is loaded at most once."""

    _instance: "DepthEstimator | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: str | None = None
        self._model = None
        self._processor = None
        self.device = select_device().device

    @classmethod
    def shared(cls) -> "DepthEstimator":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = DepthEstimator()
            return cls._instance

    @property
    def loaded_key(self) -> str | None:
        return self._key

    def _load(self, key: str) -> None:
        if self._key == key:
            return
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        spec = MODELS[key]
        self._model = None  # release the previous checkpoint before loading
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        start = time.perf_counter()
        cache = str(models_dir())
        try:
            processor = AutoImageProcessor.from_pretrained(spec.repo_id, cache_dir=cache, local_files_only=True)
            model = AutoModelForDepthEstimation.from_pretrained(spec.repo_id, cache_dir=cache, local_files_only=True)
        except OSError:
            log.info("Downloading %s (~%d MB) from Hugging Face", spec.repo_id, spec.size_mb)
            processor = AutoImageProcessor.from_pretrained(spec.repo_id, cache_dir=cache)
            model = AutoModelForDepthEstimation.from_pretrained(spec.repo_id, cache_dir=cache)
        model = model.to(self.device).eval()
        self._processor, self._model, self._key = processor, model, key
        log.info("Loaded %s on %s in %.1fs", spec.repo_id, self.device, time.perf_counter() - start)

    @torch.inference_mode()
    def _forward(self, pil: Image.Image, size: int) -> torch.Tensor:
        inputs = self._processor(images=pil, return_tensors="pt", size={"height": size, "width": size})
        pixels = inputs["pixel_values"].to(self.device)
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = self._model(pixel_values=pixels).predicted_depth
        else:
            out = self._model(pixel_values=pixels).predicted_depth
        return out[0].float()

    def predict(self, image: np.ndarray, model: str = "small", detail: str = "standard") -> np.ndarray:
        """Return relative inverse depth (float32) at network resolution.

        ``detail="high"`` runs a larger input plus a horizontal-flip pass that
        is averaged in, reducing noise at the cost of ~3x the compute.
        """
        if model not in MODELS:
            raise ValueError(f"Unknown depth model: {model}")
        size = DETAIL_SIZES.get(detail, DETAIL_SIZES["standard"])
        with self._lock:
            self._load(model)
            pil = Image.fromarray(image)
            start = time.perf_counter()
            disp = self._forward(pil, size)
            if detail == "high":
                flipped = self._forward(pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT), size)
                disp = 0.5 * (disp + torch.flip(flipped, dims=[-1]))
            result = disp.cpu().numpy().astype(np.float32)
            log.info("Depth inference %s/%s %s in %.0f ms", model, detail, result.shape,
                     (time.perf_counter() - start) * 1000)
            return result

    def warmup(self, model: str = "small") -> None:
        """Load weights and compile kernels with a tiny dummy image."""
        self.predict(np.zeros((64, 64, 3), np.uint8), model)
