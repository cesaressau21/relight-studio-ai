"""The interface fits the screen: nothing is clipped or pushed off-screen.

Screen sizes are the logical sizes Windows reports (after display scaling and
without the taskbar), e.g. 1920x1080 at 100 % -> 1920x1032 available,
1920x1080 at 150 % -> 1280x672, 1366x768 laptops -> 1366x720.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QComboBox, QScrollArea, QToolButton

from .conftest import settle, wait_until

SCREENS = [QRect(0, 0, 1920, 1032), QRect(0, 0, 1366, 720), QRect(0, 0, 1280, 672), QRect(0, 0, 2560, 1392)]
TITLE_BAR = 32


@pytest.mark.parametrize("available", SCREENS, ids=lambda r: f"{r.width()}x{r.height()}")
def test_initial_window_fits_the_available_screen(available):
    from relight_studio.ui.layout import initial_window_rect

    rect, maximize = initial_window_rect(available)
    assert available.contains(rect.adjusted(0, -TITLE_BAR, 0, 0)) or maximize
    assert rect.width() >= 1000 or maximize


def test_window_minimum_size_fits_a_small_laptop(qapp, editor):
    hint = editor.minimumSizeHint()
    assert hint.width() <= 1280 and hint.height() <= 672 - TITLE_BAR, hint


@pytest.mark.parametrize("size", [(1920, 1000), (1366, 688), (1280, 640)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_panels_never_clip_horizontally(qapp, editor, size):
    editor.resize(*size)
    qapp.processEvents()
    for i in range(editor.tabs.count()):
        editor.tabs.setCurrentIndex(i)
        qapp.processEvents()
        area = editor.tabs.widget(i).findChild(QScrollArea)
        body = area.widget()
        name = editor.tabs.tabText(i)
        assert body.minimumSizeHint().width() <= area.viewport().width(), name
        assert body.width() <= area.viewport().width(), name


def effective_min_width(widget) -> int:
    """Minimum width a layout enforces: an explicit minimum overrides the size hint."""
    return widget.minimumWidth() or widget.minimumSizeHint().width()


def test_combo_boxes_do_not_dictate_panel_width(qapp, editor):
    for combo in editor.findChildren(QComboBox):
        assert effective_min_width(combo) <= 200, combo.currentText()


@pytest.mark.parametrize("size", [(1280, 640), (1024, 640)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_top_bar_actions_stay_visible(qapp, editor, size):
    editor.resize(*size)
    qapp.processEvents()
    bar = editor.export_btn.parentWidget()
    for button in (editor.open_btn, editor.save_btn, editor.undo_btn, editor.redo_btn, editor.compare_btn,
                   editor.export_btn):
        geo = button.geometry()
        assert button.isVisible() and geo.right() <= bar.width() and geo.width() > 0, button.toolTip()


def test_every_tool_is_reachable_on_a_short_window(qapp, editor):
    editor.resize(1280, 560)
    qapp.processEvents()
    rail_area = editor.tool_rail_area
    assert isinstance(rail_area, QScrollArea)
    for button in editor.tool_rail_area.widget().findChildren(QToolButton):
        rail_area.ensureWidgetVisible(button)
        qapp.processEvents()
        top_left = button.mapTo(rail_area.viewport(), button.rect().topLeft())
        assert 0 <= top_left.y() and top_left.y() + button.height() <= rail_area.viewport().height() + 1


def test_side_panel_is_resizable_with_sane_limits(qapp, editor):
    panel = editor.side_panel
    assert panel.minimumWidth() >= 280 and panel.maximumWidth() <= 600
    sizes = editor.body_splitter.sizes()
    editor.body_splitter.setSizes([sizes[0] - 100, sizes[1] + 100])
    qapp.processEvents()
    assert editor.body_splitter.sizes()[1] > sizes[1]


def test_long_status_messages_do_not_widen_the_window(qapp, editor):
    before = editor.minimumSizeHint().width()
    editor.set_status("A very long status message " * 20)
    qapp.processEvents()
    assert editor.minimumSizeHint().width() <= before


def test_export_dialog_fits_a_small_screen(qapp):
    from pathlib import Path

    from relight_studio.ui.export_dialog import ExportDialog

    dialog = ExportDialog(6000, 4000, Path("C:/some/very/long/folder/name/that/keeps/going/photo_relit.png"))
    hint = dialog.sizeHint()
    assert hint.width() <= 1000 and hint.height() <= 600


def test_layout_survives_a_full_render_at_small_size(qapp, editor):
    editor.resize(1280, 640)
    settle(qapp, editor)
    assert wait_until(qapp, lambda: editor.canvas.result is not None)
    rect = editor.canvas.image_rect()
    assert rect.width() <= editor.canvas.width() and rect.height() <= editor.canvas.height()


def test_title_shows_the_whole_file_name_when_there_is_room(qapp, editor):
    editor.resize(1500, 900)
    qapp.processEvents()
    label = editor.title_label
    assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
