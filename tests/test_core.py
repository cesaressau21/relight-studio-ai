"""Document model, history, brushes, colors and presets."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.brush import BrushSettings, paint_depth, paint_mask, stroke_samples
from relight_studio.core.color import kelvin_to_linear_gain, linear_to_srgb_np, srgb_to_linear_np
from relight_studio.core.document import Document
from relight_studio.core.history import History
from relight_studio.core.lights import Light, LightType
from relight_studio.core.presets import load_presets

REQUIRED_PRESETS = {
    "Studio Portrait", "Golden Hour", "Cinematic Blue & Orange", "Neon Cyberpunk",
    "Soft Natural Light", "Dramatic Side Light", "Sunset", "Moonlight",
}


def make_doc() -> Document:
    return Document(image=np.zeros((300, 400, 3), np.uint8))


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101)
    assert np.allclose(linear_to_srgb_np(srgb_to_linear_np(x)), x, atol=1e-6)


def test_kelvin_gain_is_neutral_at_6500_and_ordered():
    assert np.allclose(kelvin_to_linear_gain(6500), (1, 1, 1), atol=1e-6)
    warm, cool = kelvin_to_linear_gain(3000), kelvin_to_linear_gain(10000)
    assert warm[0] > warm[2] and cool[2] > cool[0]


def test_document_image_is_read_only():
    doc = make_doc()
    with pytest.raises(ValueError):
        doc.image[0, 0, 0] = 1


def test_document_rejects_bad_images():
    with pytest.raises(ValueError):
        Document(image=np.zeros((10, 10), np.uint8))


def test_light_serialization_roundtrip():
    light = Light(id=3, type=LightType.RIM, color=(0.2, 0.4, 0.6), kelvin=3200, shadows=False)
    again = Light.from_dict(light.to_dict())
    assert again == light
    assert Light.from_dict({**light.to_dict(), "unknown_future_field": 1}) == light


def test_at_least_three_independent_lights():
    doc = make_doc()
    ids = [doc.add_light(t).id for t in (LightType.POINT, LightType.AREA, LightType.DIRECTIONAL, LightType.RIM)]
    assert len(set(ids)) == 4
    doc.light(ids[0]).intensity = 7.0
    assert all(doc.light(i).intensity != 7.0 for i in ids[1:])


def test_history_undo_redo_and_merge():
    doc = make_doc()
    history = History(merge_window=10.0)
    history.reset(doc.snapshot())
    light = doc.add_light(LightType.POINT)
    history.push(doc.snapshot(), "add")
    start_x = light.x
    for x in (0.5, 0.6, 0.7):  # a drag collapses into one step
        light.x = x
        history.push(doc.snapshot(), "move", merge_key="drag1")
    doc.restore(history.undo())
    assert doc.lights[0].x == pytest.approx(start_x)
    doc.restore(history.undo())
    assert doc.lights == []
    assert not history.can_undo()
    doc.restore(history.redo())
    assert len(doc.lights) == 1
    doc.restore(history.redo())
    assert doc.lights[0].x == pytest.approx(0.7)


def test_history_ignores_no_op_and_trims():
    doc = make_doc()
    history = History(limit=5)
    history.reset(doc.snapshot())
    assert history.push(doc.snapshot(), "nothing") is False
    for i in range(10):
        doc.relight.ambient = i / 10
        history.push(doc.snapshot(), f"edit {i}")
    count = 0
    while history.can_undo():
        history.undo()
        count += 1
    assert count == 4


def test_mask_brush_paint_and_erase():
    mask = np.zeros((100, 100), np.uint8)
    brush = BrushSettings(size=0.1, hardness=0.8, flow=1.0)
    assert paint_mask(mask, 0.5, 0.5, brush, erase=False)
    assert mask[50, 50] == 255 and mask[0, 0] == 0
    paint_mask(mask, 0.5, 0.5, brush, erase=True)
    assert mask[50, 50] == 0
    assert not paint_mask(mask, 3.0, 3.0, brush, erase=False)  # off-canvas is a no-op


def test_depth_brush_push_and_pull():
    offset = np.zeros((100, 100), np.float32)
    brush = BrushSettings(size=0.1, flow=1.0, depth_strength=0.05)
    paint_depth(offset, 0.5, 0.5, brush, push=True)
    assert offset[50, 50] == pytest.approx(0.05)
    paint_depth(offset, 0.5, 0.5, brush, push=False)
    paint_depth(offset, 0.5, 0.5, brush, push=False)
    assert offset[50, 50] == pytest.approx(-0.05)


def test_stroke_samples_are_spaced():
    pts = stroke_samples(0.0, 0.5, 1.0, 0.5, aspect=1.0, radius=0.1)
    assert len(pts) == 40 and pts[-1] == (1.0, 0.5)


def test_presets_cover_requested_looks_and_set_engine_parameters():
    presets = {p.name: p for p in load_presets()}
    assert REQUIRED_PRESETS <= set(presets)
    for preset in presets.values():
        doc = make_doc()
        preset.apply(doc)
        assert doc.lights, preset.name
        assert doc.relight.original_light == preset.relight.get("original_light", doc.relight.original_light)
        for spec, light in zip(preset.lights, doc.lights):
            assert light.type == LightType(spec["type"])
            for key in ("x", "y", "z", "intensity"):
                if key in spec:
                    assert getattr(light, key) == spec[key]
