from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture(scope="session")
def device() -> torch.device:
    from relight_studio.rendering.device import select_device

    return select_device().device


def sphere_disparity(h: int = 96, w: int = 128, radius: float = 0.35) -> np.ndarray:
    """Disparity of a hemisphere in front of a flat background (larger = nearer)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx + 0.5 - w / 2) / h
    y = (yy + 0.5 - h / 2) / h
    r2 = x * x + y * y
    bulge = np.sqrt(np.clip(radius * radius - r2, 0.0, None)) / radius
    return (0.2 + 0.8 * bulge).astype(np.float32)


def gray_image(h: int = 96, w: int = 128, value: int = 140) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def real_sample() -> Path | None:
    """A real photograph for integration tests (samples/ or RELIGHT_TEST_IMAGE)."""
    env = os.environ.get("RELIGHT_TEST_IMAGE")
    if env and Path(env).exists():
        return Path(env)
    candidates = sorted(SAMPLES.glob("*.webp")) + sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))
    return candidates[-1] if candidates else None


def model_available() -> bool:
    from relight_studio.ai.depth import MODELS
    from relight_studio.core.paths import models_dir

    folder = "models--" + MODELS["small"].repo_id.replace("/", "--")
    return (models_dir() / folder).exists() or bool(os.environ.get("RELIGHT_ALLOW_DOWNLOAD"))
