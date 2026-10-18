"""Depth inference recovers when half-precision (float16) produces non-finite values."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relight_studio.ai.depth import DepthEstimator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="float16 autocast runs only on CUDA")


class FakeProcessor:
    def __call__(self, images, return_tensors, size):
        return {"pixel_values": torch.zeros(1, 3, 14, 14)}


class FakeModel:
    """Overflows under float16 autocast (like a numerically unstable layer), fine in float32."""

    def __init__(self, broken_in_fp32: bool = False):
        self.calls: list[bool] = []
        self.broken_in_fp32 = broken_in_fp32

    def __call__(self, pixel_values):
        half = torch.is_autocast_enabled("cuda")
        self.calls.append(half)
        depth = torch.linspace(0.1, 1.0, 14 * 14, device=pixel_values.device).view(1, 14, 14)
        if half or self.broken_in_fp32:
            depth = depth.clone()
            depth[0, 0, 0] = float("inf")
            depth[0, 1, 1] = float("nan")
        return SimpleNamespace(predicted_depth=depth)


def estimator_with(model: FakeModel) -> DepthEstimator:
    est = DepthEstimator()
    est.device = torch.device("cuda")
    est._processor, est._model, est._key = FakeProcessor(), model, "small"
    return est


def test_non_finite_half_precision_output_is_recomputed_in_float32(caplog):
    model = FakeModel()
    with caplog.at_level(logging.WARNING):
        disp = estimator_with(model).predict(np.zeros((32, 32, 3), np.uint8))
    assert np.isfinite(disp).all()
    assert model.calls == [True, False]  # float16 first, then one float32 retry
    assert "float32" in caplog.text


def test_finite_output_needs_no_retry():
    est = estimator_with(FakeModel())
    est._model = lambda pixel_values: SimpleNamespace(
        predicted_depth=torch.ones(1, 14, 14, device=pixel_values.device))
    disp = est.predict(np.zeros((32, 32, 3), np.uint8))
    assert np.isfinite(disp).all()


def test_still_non_finite_in_float32_raises_a_clear_error():
    est = estimator_with(FakeModel(broken_in_fp32=True))
    with pytest.raises(RuntimeError, match="non-finite"):
        est.predict(np.zeros((32, 32, 3), np.uint8))
