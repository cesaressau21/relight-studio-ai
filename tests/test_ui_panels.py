"""Right-hand panels are bound to the document in both directions."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from relight_studio.core.lights import LightType

from .conftest import settle


def test_light_sliders_edit_the_selected_light(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    lid = ctl.selected
    for key, value in (("x", 0.8), ("y", 0.1), ("z", 0.5), ("intensity", 3.3), ("radius", 1.4),
                       ("kelvin", 3000.0), ("softness", 0.9), ("diffusion", 0.6), ("attenuation", 2.0),
                       ("specular", 1.7)):
        panel.rows[key].spin.setValue(value)
        assert getattr(ctl.doc.light(lid), key) == pytest.approx(value, abs=0.01), key


def test_light_panel_refreshes_after_undo(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    panel.rows["intensity"].spin.setValue(4.0)
    ctl.undo()
    assert panel.rows["intensity"].value() == pytest.approx(ctl.selected_light().intensity)


def test_light_name_type_color_and_toggles(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    panel.name.setText("Hero")
    panel.name.editingFinished.emit()
    assert ctl.selected_light().name == "Hero"
    panel.type_combo.setCurrentIndex(panel.type_combo.findData(LightType.RIM.value))
    assert ctl.selected_light().type is LightType.RIM
    panel.color.colorChanged.emit((0.1, 0.2, 0.9))
    assert ctl.selected_light().color == (0.1, 0.2, 0.9)
    panel.enabled.setChecked(False)
    assert not ctl.selected_light().enabled
    panel.type_combo.setCurrentIndex(panel.type_combo.findData(LightType.POINT.value))
    panel.shadows.setChecked(False)
    assert not ctl.selected_light().shadows


def test_light_list_selects_and_toggles(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    second = ctl.add_light(LightType.AREA)
    assert panel.list.count() == 2
    panel.list.setCurrentRow(0)
    assert ctl.selected == ctl.doc.lights[0].id
    panel.list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert not ctl.doc.light(second.id).enabled


def test_add_menu_duplicate_and_delete_buttons(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    actions = panel.add_btn.menu().actions()
    assert len(actions) == len(LightType)
    actions[3].trigger()  # soft area
    assert ctl.selected_light().type is LightType.AREA
    panel.dup_btn.click()
    assert len(ctl.doc.lights) == 3
    panel.del_btn.click()
    assert len(ctl.doc.lights) == 2


def test_controls_disable_for_inapplicable_light_types(qapp, editor):
    panel, ctl = editor.lights_panel, editor.ctl
    ctl.add_light(LightType.AMBIENT)
    assert not panel.rows["x"].isEnabled() and not panel.shadows.isEnabled()
    ctl.add_light(LightType.AREA)
    assert panel.rows["size"].isEnabled() and panel.rows["x"].isEnabled()


def test_scene_panel_controls(qapp, editor):
    panel, ctl = editor.scene_panel, editor.ctl
    panel.relight_form.rows["original_light"].spin.setValue(0.3)
    assert ctl.doc.relight.original_light == pytest.approx(0.3)
    panel.relight_form.rows["shadow_steps"].spin.setValue(40)
    assert ctl.doc.relight.shadow_steps == 40 and isinstance(ctl.doc.relight.shadow_steps, int)
    panel.depth_form.rows["relief"].spin.setValue(1.1)
    assert ctl.doc.depth.relief == pytest.approx(1.1)
    panel.invert.setChecked(True)
    assert ctl.doc.depth.invert
    panel.rolloff.setChecked(False)
    assert not ctl.doc.relight.highlight_rolloff
    panel.ambient_color.colorChanged.emit((0.5, 0.5, 1.0))
    assert ctl.doc.relight.ambient_color == (0.5, 0.5, 1.0)
    panel.view.setCurrentIndex(panel.view.findData("normals"))
    assert ctl.view == "normals"


def test_scene_panel_model_choice_warns_about_licenses(qapp, editor):
    panel, ctl = editor.scene_panel, editor.ctl
    panel.model.setCurrentIndex(panel.model.findData("large"))
    assert ctl.doc.depth.model == "large"
    assert "non-commercial" in panel.source.text()
    fired = []
    panel.reanalyzeRequested.connect(lambda: fired.append(True))
    panel.reanalyze.click()
    assert fired


def test_adjust_panel_and_reset(qapp, editor):
    panel, ctl = editor.adjust_panel, editor.ctl
    for key in ("exposure", "contrast", "highlights", "shadows", "whites", "blacks", "saturation",
                "temperature", "tint"):
        panel.form.rows[key].spin.setValue(0.5)
        assert getattr(ctl.doc.adjustments, key) == pytest.approx(0.5), key
    panel.form.rows["gamma"].spin.setValue(1.5)
    ctl.reset_adjustments()
    assert ctl.doc.adjustments.is_identity()
    assert panel.form.rows["exposure"].value() == 0.0


def test_layers_panel(qapp, editor):
    panel, ctl = editor.layers_panel, editor.ctl
    panel.size.spin.setValue(0.2)
    panel.hardness.spin.setValue(0.9)
    panel.flow.spin.setValue(0.3)
    panel.depth_strength.spin.setValue(0.05)
    assert (ctl.brush.size, ctl.brush.hardness, ctl.brush.flow, ctl.brush.depth_strength) == pytest.approx(
        (0.2, 0.9, 0.3, 0.05))
    panel.mask_invert.setChecked(True)
    panel.mask_overlay.setChecked(True)
    panel.depth_enabled.setChecked(False)
    ms = ctl.doc.mask_settings
    assert ms.invert and ms.show_overlay and not ms.depth_edits_enabled
    panel.mask_enabled.setChecked(False)
    assert not ctl.doc.mask_settings.enabled


def test_presets_panel_applies_presets(qapp, editor):
    panel, ctl = editor.presets_panel, editor.ctl
    assert len(panel.cards) >= 8 and all(card.isEnabled() for card in panel.cards)
    card = panel.cards[0]
    card.clicked.emit()
    assert [l.name for l in ctl.doc.lights] == [s["name"] for s in card.preset.lights]
    settle(qapp, editor)
