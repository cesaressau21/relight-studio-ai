from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

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


def _cached(repo_id: str) -> bool:
    from relight_studio.core.paths import models_dir

    return (models_dir() / ("models--" + repo_id.replace("/", "--"))).exists()


def model_available() -> bool:
    from relight_studio.ai.depth import MODELS

    return _cached(MODELS["small"].repo_id) or bool(os.environ.get("RELIGHT_ALLOW_DOWNLOAD"))


# ----------------------------------------------------------------------------- Qt
@pytest.fixture(scope="session")
def qapp(tmp_path_factory):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from relight_studio.ui.theme import apply_theme

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path_factory.mktemp("qsettings")))
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def wait_until(qapp, condition: Callable[[], bool], timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def write_photo(path: Path, h: int = 120, w: int = 180) -> Path:
    from PIL import Image

    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([(xx * 255 // w), (yy * 255 // h), np.full_like(xx, 128)], -1).astype(np.uint8)
    rgb[30:90, 60:120] = (220, 170, 140)  # a "subject"
    Image.fromarray(rgb).save(path)
    return path


def fake_disparity(h: int = 60, w: int = 90) -> np.ndarray:
    disp = np.full((h, w), 0.2, np.float32)
    disp[15:45, 30:60] = 1.0
    return disp


@pytest.fixture()
def editor(qapp, tmp_path, monkeypatch):
    """Main window with a synthetic photo loaded and depth injected (no model needed)."""
    from relight_studio.ui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_start_warmup", lambda self: None)

    def fake_run_depth(self):
        if self.ctl.doc is not None:
            self.ctl.set_disparity(fake_disparity(), f"{self.ctl.doc.depth.model}/{self.ctl.doc.depth.detail}")
            self._update_actions()

    monkeypatch.setattr(MainWindow, "run_depth", fake_run_depth)
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    win.open_path(write_photo(tmp_path / "photo.png"))
    assert wait_until(qapp, lambda: win.canvas.result is not None), "first frame not rendered"
    yield win
    wait_until(qapp, lambda: not win.render_service._busy, timeout=10)
    win.ctl.dirty = False
    win.close()
    win.deleteLater()
    qapp.processEvents()


def settle(qapp, win, timeout: float = 15.0) -> None:
    """Wait until queued renders are done."""
    assert wait_until(qapp, lambda: not win.render_service._busy and win.render_service._pending is None, timeout)
