"""Headless tests for the RectFill retouch tool.

Verifies:
  - fill bbox: overlay filled inside drag region, untouched outside
  - arbitrary size: rectangle dimensions match drag endpoints
  - clamp: partial out-of-bounds does not crash, only intersection filled
  - undo command: one drag/release = one PaintCommand, painting=False after release
  - preview: QGraphicsRectItem exists during drag, removed after release
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsRectItem

import pytest

from app.ui.canvas.image_viewer import ImageViewer
from app.ui.commands.paint import PaintCommand


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _make_viewer(app):
    v = ImageViewer(None)
    v.set_tool("paint_fill_rect")
    pix = QPixmap(200, 200)
    pix.fill(QColor(128, 128, 128, 255))
    v.setPhoto(pix)
    return v


def _ev(type_, local, button, buttons, mod=Qt.KeyboardModifier.NoModifier):
    scene = local
    return QMouseEvent(type_, local, scene, scene, button, buttons, mod)


def _view_pos(viewer, scene_pos):
    """Convert scene coordinates to viewport coordinates for mouse events."""
    return viewer.mapFromScene(scene_pos)


def _press(viewer, pos, button=Qt.LeftButton):
    viewer.mousePressEvent(
        _ev(QMouseEvent.MouseButtonPress, pos, button, Qt.MouseButton.NoButton)
    )


def _move(viewer, pos, buttons=Qt.LeftButton):
    viewer.mouseMoveEvent(
        _ev(QMouseEvent.MouseMove, pos, Qt.MouseButton.NoButton, buttons)
    )


def _release(viewer, pos, button=Qt.LeftButton):
    viewer.mouseReleaseEvent(
        _ev(QMouseEvent.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton)
    )


# ------------------------------------------------------------------
# 1. Fill bbox — inside filled, outside untouched
# ------------------------------------------------------------------

def test_rectfill_bbox_inside_outside(app):
    v = _make_viewer(app)
    emitted = []
    v.command_emitted.connect(lambda c: emitted.append(c))

    start_scene = v.mapToScene(v.viewport().rect().center())
    end_scene = start_scene + QPointF(40, 30)

    _press(v, _view_pos(v, start_scene))
    assert v.paint_manager.painting is True

    _move(v, _view_pos(v, end_scene))
    _release(v, _view_pos(v, end_scene))

    assert v.paint_manager.painting is False
    assert len(emitted) == 1
    assert isinstance(emitted[0], PaintCommand)

    overlay = v.paint_overlay
    assert overlay is not None

    # Normalise bbox to pixel coords
    x0, y0 = int(round(min(start_scene.x(), end_scene.x()))), int(round(min(start_scene.y(), end_scene.y())))
    x1, y1 = int(round(max(start_scene.x(), end_scene.x()))), int(round(max(start_scene.y(), end_scene.y())))

    # Pixel just inside bbox should have alpha == 255
    inside_x = (x0 + x1) // 2
    inside_y = (y0 + y1) // 2
    assert overlay[inside_y, inside_x, 3] == 255, "inside pixel should be painted"

    # Pixel well outside bbox should have alpha == 0
    assert overlay[0, 0, 3] == 0, "outside pixel should not be painted"


# ------------------------------------------------------------------
# 2. Arbitrary size — rectangle dimensions match drag
# ------------------------------------------------------------------

def test_rectfill_arbitrary_size(app):
    v = _make_viewer(app)
    emitted = []
    v.command_emitted.connect(lambda c: emitted.append(c))

    start_scene = QPointF(50, 50)
    end_scene = QPointF(120, 90)

    _press(v, _view_pos(v, start_scene))
    _move(v, _view_pos(v, end_scene))
    _release(v, _view_pos(v, end_scene))

    assert len(emitted) == 1
    overlay = v.paint_overlay
    assert overlay is not None

    # All pixels in the rectangle should be painted
    for y in range(50, 90):
        for x in range(50, 120):
            assert overlay[y, x, 3] == 255, f"pixel ({x},{y}) should be painted"

    # Pixel at (49, 49) should not be painted
    assert overlay[49, 49, 3] == 0, "outside pixel should be untouched"
    # Pixel at (120, 90) should not be painted (end is exclusive)
    assert overlay[90, 120, 3] == 0, "outside pixel should be untouched"


# ------------------------------------------------------------------
# 3. Clamp — partial out-of-bounds does not crash
# ------------------------------------------------------------------

def test_rectfill_clamp(app):
    v = _make_viewer(app)
    emitted = []
    v.command_emitted.connect(lambda c: emitted.append(c))

    # Drag from (180, 180) to (250, 250) — partially out of 200x200 image
    start_scene = QPointF(180, 180)
    end_scene = QPointF(250, 250)

    _press(v, _view_pos(v, start_scene))
    _move(v, _view_pos(v, end_scene))
    _release(v, _view_pos(v, end_scene))

    assert v.paint_manager.painting is False
    assert len(emitted) == 1

    overlay = v.paint_overlay
    assert overlay is not None

    # Intersection with image bounds should be filled
    assert overlay[190, 190, 3] == 255, "clamped pixel inside image should be painted"
    # Pixel outside image boundary should be untouched (edge)
    assert overlay[0, 0, 3] == 0, "far corner should be untouched"


# ------------------------------------------------------------------
# 4. Undo command — exactly one PaintCommand, painting=False
# ------------------------------------------------------------------

def test_rectfill_undo_command(app):
    v = _make_viewer(app)
    emitted = []
    v.command_emitted.connect(lambda c: emitted.append(c))

    start_scene = QPointF(30, 40)
    end_scene = QPointF(80, 90)

    _press(v, _view_pos(v, start_scene))
    assert v.paint_manager.painting is True

    _move(v, _view_pos(v, end_scene))
    _release(v, _view_pos(v, end_scene))

    assert v.paint_manager.painting is False
    assert len(emitted) == 1
    assert isinstance(emitted[0], PaintCommand)

    # Second drag should emit a separate command
    start_scene2 = QPointF(100, 100)
    end_scene2 = QPointF(150, 150)
    _press(v, _view_pos(v, start_scene2))
    _move(v, _view_pos(v, end_scene2))
    _release(v, _view_pos(v, end_scene2))

    assert len(emitted) == 2


# ------------------------------------------------------------------
# 5. Preview — QGraphicsRectItem exists during drag, removed after release
# ------------------------------------------------------------------

def test_rectfill_preview(app):
    v = _make_viewer(app)

    start_scene = QPointF(60, 60)
    end_scene = QPointF(120, 100)

    _press(v, _view_pos(v, start_scene))

    # Preview rect should exist in the scene
    rect_items = [i for i in v._scene.items() if isinstance(i, QGraphicsRectItem)]
    assert len(rect_items) >= 1, "preview rect should exist during drag"

    # Preview rect should be updated on move
    _move(v, _view_pos(v, end_scene))
    preview = v.paint_manager._rect_item
    assert preview is not None
    r = preview.rect()
    assert r.width() > 0 and r.height() > 0, "preview should have non-zero size"

    # Release should remove preview
    _release(v, _view_pos(v, end_scene))
    assert v.paint_manager._rect_item is None, "preview should be removed after release"
