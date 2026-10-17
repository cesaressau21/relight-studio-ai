"""EditorController: every edit operation, its undo step and its change notification."""

from __future__ import annotations

import numpy as np
import pytest

from relight_studio.core.document import Document
from relight_studio.core.lights import LightType
from relight_studio.core.presets import load_presets


@pytest.fixture()
def ctl(qapp):
    from relight_studio.ui.controller import EditorController

    c = EditorController()
    c.kinds = []
    c.changed.connect(lambda k: c.kinds.append(set(k)))
    doc = Document(image=np.zeros((200, 300, 3), np.uint8))
    doc.add_light(LightType.POINT)
    c.load_document(doc)
    c.kinds.clear()
    return c


def test_load_document_builds_preview_and_resets_history(ctl):
    assert ctl.preview_image.shape == (200, 300, 3)
    assert not ctl.history.can_undo() and not ctl.dirty
    assert ctl.selected == ctl.doc.lights[0].id


def test_preview_is_downscaled_for_large_photos(qapp):
    from relight_studio.ui.controller import EditorController, preview_max_side

    c = EditorController()
    c.load_document(Document(image=np.zeros((3000, 4500, 3), np.uint8)))
    assert max(c.preview_image.shape[:2]) == preview_max_side()


@pytest.mark.parametrize("light_type", list(LightType))
def test_add_light_selects_it_and_is_undoable(ctl, light_type):
    light = ctl.add_light(light_type, x=0.7, y=0.2)
    assert ctl.selected == light.id and light.type is light_type and light.x == 0.7
    assert {"lights", "selection"} <= ctl.kinds[-1]
    ctl.undo()
    assert ctl.doc.light(light.id) is None


def test_remove_and_duplicate_light(ctl):
    first = ctl.doc.lights[0]
    ctl.duplicate_light(first.id)
    clone = ctl.selected_light()
    assert clone.id != first.id and clone.x == pytest.approx(first.x + 0.04) and clone.color == first.color
    ctl.remove_light(clone.id)
    assert ctl.doc.light(clone.id) is None and ctl.selected == first.id
    ctl.remove_light(999)  # unknown id is ignored
    assert len(ctl.doc.lights) == 1


def test_update_light_changes_values_and_merges_drags(ctl):
    lid = ctl.selected
    for x in (0.4, 0.5, 0.6):
        ctl.update_light(lid, merge_key="drag", x=x)
    assert ctl.doc.light(lid).x == 0.6
    ctl.undo()
    assert ctl.doc.light(lid).x == pytest.approx(0.3)  # the whole drag was one step
    assert ctl.kinds[-1] >= {"lights"}


def test_selection_change_emits_selection_only(ctl):
    other = ctl.add_light(LightType.RIM)
    ctl.kinds.clear()
    ctl.select(ctl.doc.lights[0].id)
    assert ctl.kinds == [{"selection"}]
    ctl.select(None)
    assert ctl.selected_light() is None
    assert other.id != ctl.doc.lights[0].id


@pytest.mark.parametrize(
    "method,field,value,kinds",
    [
        ("update_relight", "original_light", 0.2, {"relight"}),
        ("update_depth", "relief", 1.2, {"depth", "geometry"}),
        ("update_adjustments", "exposure", 1.0, {"adjust"}),
        ("update_mask_settings", "invert", True, {"mask"}),
        ("update_mask_settings", "depth_edits_enabled", False, {"mask", "geometry"}),
    ],
)
def test_parameter_groups_update_notify_and_undo(ctl, method, field, value, kinds):
    group = {"update_relight": "relight", "update_depth": "depth", "update_adjustments": "adjustments",
             "update_mask_settings": "mask_settings"}[method]
    before = getattr(getattr(ctl.doc, group), field)
    getattr(ctl, method)(**{field: value})
    assert getattr(getattr(ctl.doc, group), field) == value
    assert kinds <= ctl.kinds[-1] and ctl.dirty
    ctl.undo()
    assert getattr(getattr(ctl.doc, group), field) == before
    ctl.redo()
    assert getattr(getattr(ctl.doc, group), field) == value


def test_reset_adjustments(ctl):
    ctl.update_adjustments(exposure=1.0, contrast=0.5)
    ctl.reset_adjustments()
    assert ctl.doc.adjustments.is_identity()


def test_apply_preset_replaces_rig_in_one_undo_step(ctl):
    preset = next(p for p in load_presets() if p.name == "Neon Cyberpunk")
    old_ids = [l.id for l in ctl.doc.lights]
    ctl.apply_preset(preset)
    assert [l.name for l in ctl.doc.lights] == [s["name"] for s in preset.lights]
    assert ctl.doc.relight.original_light == preset.relight["original_light"]
    assert ctl.selected == ctl.doc.lights[0].id
    ctl.undo()
    assert [l.id for l in ctl.doc.lights] == old_ids


def test_view_mode_is_not_part_of_history(ctl):
    ctl.set_view("depth")
    assert ctl.view == "depth" and ctl.kinds[-1] == {"view"}
    assert not ctl.history.can_undo()


def test_set_disparity_bumps_geometry(ctl):
    version = ctl.disparity_version
    ctl.set_disparity(np.ones((10, 15), np.float32), "small/standard")
    assert ctl.disparity_version == version + 1 and ctl.kinds[-1] == {"geometry"}
    assert ctl.doc.disparity_source == "small/standard"


def test_mask_stroke_is_one_undo_step_and_keeps_history_arrays_intact(ctl):
    ctl.fill_mask()
    filled = ctl.doc.mask
    ctl.begin_stroke("mask", 0.5, 0.5, alt=True)  # erase
    ctl.stroke_to(0.6, 0.5, alt=True)
    ctl.end_stroke()
    painted = ctl.doc.mask
    assert painted is not filled and filled.min() == 255  # snapshot untouched
    h, w = painted.shape
    assert painted[h // 2, w // 2] < 255
    ctl.undo()
    assert ctl.doc.mask is filled


def test_depth_stroke_push_and_pull(ctl):
    ctl.brush.size, ctl.brush.flow = 0.1, 1.0
    version = ctl.offset_version
    ctl.begin_stroke("depth", 0.5, 0.5, alt=False)
    ctl.end_stroke()
    offset = ctl.doc.depth_offset
    h, w = offset.shape
    assert offset[h // 2, w // 2] > 0 and ctl.offset_version > version
    ctl.begin_stroke("depth", 0.5, 0.5, alt=True)
    ctl.end_stroke()
    assert ctl.doc.depth_offset[h // 2, w // 2] == pytest.approx(0.0, abs=1e-6)


def test_interaction_signal_brackets_strokes(ctl):
    events = []
    ctl.interactionChanged.connect(events.append)
    ctl.begin_stroke("mask", 0.5, 0.5, alt=False)
    ctl.end_stroke()
    assert events == [True, False]


@pytest.mark.parametrize("op,expected", [("fill_mask", 255), ("clear_mask", 0)])
def test_mask_fill_and_clear(ctl, op, expected):
    getattr(ctl, op)()
    assert np.all(ctl.doc.mask == expected)


def test_invert_mask_and_clear_depth(ctl):
    ctl.clear_mask()
    ctl.invert_mask()
    assert np.all(ctl.doc.mask == 255)
    ctl.begin_stroke("depth", 0.5, 0.5, alt=False)
    ctl.end_stroke()
    ctl.clear_depth_edits()
    assert not ctl.doc.depth_offset.any()


def test_undo_redo_bump_layer_versions(ctl):
    ctl.clear_mask()
    version = ctl.mask_version
    ctl.undo()
    assert ctl.mask_version > version


def test_first_depth_estimate_does_not_mark_the_document_dirty(ctl):
    ctl.set_disparity(np.ones((10, 15), np.float32), "small/standard")
    assert not ctl.dirty
    ctl.set_disparity(np.ones((10, 15), np.float32), "large/high")  # a re-analysis is a change
    assert ctl.dirty
