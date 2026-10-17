"""Canvas geometry: gizmos stay attached to the image under zoom/pan and stay reachable."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def canvas(qapp):
    from relight_studio.core.document import Document
    from relight_studio.core.lights import LightType
    from relight_studio.ui.canvas import Canvas
    from relight_studio.ui.controller import EditorController

    ctl = EditorController()
    doc = Document(image=np.zeros((400, 600, 3), np.uint8))
    doc.add_light(LightType.POINT, x=0.25, y=0.75)
    doc.add_light(LightType.POINT, x=-0.4, y=0.5)  # off-frame
    ctl.load_document(doc)
    widget = Canvas(ctl)
    widget.resize(800, 600)
    widget.fit()
    return widget


def test_gizmo_tracks_the_same_image_point_under_zoom_and_pan(canvas):
    light = canvas.ctl.doc.lights[0]
    for zoom, pan in ((0.5, QPointF(10, 20)), (2.0, QPointF(-300, -100)), (1.3, QPointF(50, 0))):
        canvas.zoom, canvas.pan = zoom, pan
        pos = canvas.to_widget(light.x, light.y)
        u, v = canvas.to_norm(pos)
        assert (u, v) == pytest.approx((light.x, light.y))
        assert pos.x() == pytest.approx(pan.x() + light.x * 600 * zoom)


def test_off_frame_light_is_pinned_inside_the_canvas_and_selectable(canvas):
    light = canvas.ctl.doc.lights[1]
    pos, pinned = canvas.gizmo_pos(light)
    assert pinned
    assert 0 < pos.x() < canvas.width() and 0 < pos.y() < canvas.height()
    assert canvas.light_at(pos).id == light.id
