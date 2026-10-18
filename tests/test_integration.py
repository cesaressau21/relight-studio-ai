"""End-to-end checks with the real depth model and the real UI (offscreen).

Skipped when the model weights are not cached locally (set RELIGHT_ALLOW_DOWNLOAD=1
to let the test download them) or when no sample photograph is available.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .conftest import model_available, real_sample

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not model_available(), reason="Depth Anything V2 Small weights not cached"),
]


def test_depth_model_predicts_relative_inverse_depth():
    from relight_studio.ai.depth import DepthEstimator

    # Synthetic scene: a bright disc (object) whose lower half overlaps a ground plane gradient.
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    image = np.zeros((h, w, 3), np.uint8)
    image[..., :] = (60 + 120 * yy / h)[..., None].astype(np.uint8)
    disc = (xx - 160) ** 2 + (yy - 120) ** 2 < 60**2
    image[disc] = (200, 120, 90)
    disp = DepthEstimator.shared().predict(image)
    finite = np.isfinite(disp)
    diagnosis = (
        f"shape={disp.shape} dtype={disp.dtype} nan={int(np.isnan(disp).sum())} "
        f"inf={int(np.isinf(disp).sum())} of {disp.size}; finite min/max/std="
        f"{disp[finite].min() if finite.any() else 'n/a'}/{disp[finite].max() if finite.any() else 'n/a'}/"
        f"{disp[finite].std() if finite.any() else 'n/a'}"
    )
    assert disp.ndim == 2 and disp.dtype == np.float32, diagnosis
    assert finite.all(), f"non-finite depth values: {diagnosis}"
    assert disp.std() > 0, f"flat depth output: {diagnosis}"


@pytest.mark.skipif(real_sample() is None, reason="no sample photograph")
def test_real_photo_subject_is_nearer_than_background():
    from relight_studio.ai.depth import DepthEstimator
    from relight_studio.core.image_io import load_image

    image = load_image(real_sample())
    disp = DepthEstimator.shared().predict(image)
    h, w = disp.shape
    center = disp[int(h * 0.3) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)].mean()
    corner = disp[: int(h * 0.15), : int(w * 0.15)].mean()
    assert center > corner


@pytest.fixture()
def window(qapp):
    from relight_studio.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    yield win
    win.ctl.dirty = False
    win.close()


def pump(qapp, condition, timeout: float = 180.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(real_sample() is None, reason="no sample photograph")
def test_mvp_workflow_in_the_ui(qapp, window, tmp_path):
    """Open → depth → light X/Y/Z, intensity, color, radius → interactive update → export → project."""
    from relight_studio.core.image_io import load_image
    from relight_studio.core.project_io import read_project

    src = real_sample()
    original_bytes = src.read_bytes()
    window.open_path(src)
    ctl = window.ctl
    assert pump(qapp, lambda: ctl.doc.disparity is not None), "depth was not estimated"
    assert pump(qapp, lambda: window.canvas.result is not None), "no frame rendered"

    light = ctl.selected_light()
    assert light is not None
    start_x = light.x
    first = window.canvas.result
    ctl.update_light(light.id, x=0.85, y=0.2, z=-0.5, intensity=3.0, color=(0.2, 0.6, 1.0), radius=1.2)
    assert pump(qapp, lambda: window.canvas.result is not first and window.render_service._busy is False)
    moved = window.canvas.result
    assert moved.size() == first.size() and moved != first

    # Undo restores the previous light state.
    ctl.undo()
    assert ctl.selected_light().x == pytest.approx(start_x)
    ctl.redo()
    assert ctl.selected_light().x == pytest.approx(0.85)

    # Export at full resolution through the same worker the UI uses.
    from relight_studio.core.image_io import ExportOptions
    from relight_studio.ui.workers import ExportWorker

    out = tmp_path / "relit.jpg"
    worker = ExportWorker(ctl.doc, window._params(view="result"), ExportOptions(format="JPEG", quality=92), out)
    worker.start()
    assert pump(qapp, lambda: worker.isFinished())
    with Image.open(out) as im:
        assert im.size == (ctl.doc.width, ctl.doc.height)
        exported = np.asarray(im.convert("RGB"), dtype=np.float32)
    original = load_image(src).astype(np.float32)
    assert np.abs(exported - original).mean() > 3, "export does not contain the relighting"

    # Project save / reopen and the photo stays untouched.
    from relight_studio.core.project_io import save_project

    project = save_project(ctl.doc, tmp_path / "scene.rsproj")
    data = read_project(project)
    assert data.manifest["lights"][0]["x"] == pytest.approx(0.85)
    assert src.read_bytes() == original_bytes


@pytest.mark.skipif(len(list(Path(__file__).resolve().parents[1].glob("samples/*.webp"))) < 2,
                    reason="needs two sample photos")
def test_opening_a_second_photo_during_inference_gets_its_own_depth(qapp, window):
    samples = sorted(Path(__file__).resolve().parents[1].glob("samples/*.webp"))[:2]
    window.open_path(samples[0])
    window.open_path(samples[1])  # while the first inference may still be running
    ctl = window.ctl
    assert pump(qapp, lambda: ctl.doc.disparity is not None and window._depth_worker is None)
    assert ctl.doc.image_path == samples[1]
    ph, pw = ctl.preview_image.shape[:2]
    dh, dw = ctl.doc.disparity.shape
    assert abs(dw / dh - pw / ph) < 0.03


@pytest.mark.skipif(real_sample() is None, reason="no sample photograph")
def test_ai_matte_on_a_real_portrait():
    from relight_studio.ai.depth import DepthEstimator
    from relight_studio.ai.matting import matte_subject
    from relight_studio.core.image_io import load_image
    from relight_studio.core.settings import DepthSettings

    image = load_image(real_sample())
    disparity = DepthEstimator.shared().predict(image)
    alpha = matte_subject(image, disparity, DepthSettings())
    assert alpha is not None and alpha.shape == image.shape[:2] and alpha.dtype == np.uint8
    h, w = alpha.shape
    assert alpha[int(h * 0.45), int(w * 0.5)] == 255  # face
    assert alpha[int(h * 0.1), int(w * 0.05)] == 0  # wall
    soft = ((alpha > 10) & (alpha < 245)).mean()
    assert 0.001 < soft < 0.2  # a genuine matte with partial coverage along the outline
