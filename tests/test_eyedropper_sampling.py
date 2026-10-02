"""Headless tests for eyedropper color sampling.

The eyedropper must return the retouch paint overlay color wherever the
overlay covers the clicked point (that is what the user sees) and the base
image color everywhere else — including spots erased with the paint eraser.
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

import numpy as np
import pytest

from app.ui.canvas.image_viewer import ImageViewer


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _make_viewer(app):
    v = ImageViewer(None)
    pix = QPixmap(200, 200)
    pix.fill(QColor(128, 128, 128, 255))
    v.setPhoto(pix)
    return v


def _add_patch(v, x, y, color, w=40, h=40):
    """Add an inpaint-style patch item (QGraphicsPixmapItem with data(0)) like
    PatchCommandBase.create_patch_item does."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :] = (color.red(), color.green(), color.blue())
    qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888)
    item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
    item.setPos(x, y)
    item.setZValue(0.5)
    item.setData(0, "test-patch-hash")
    v._scene.addItem(item)
    item._test_img = img  # keep the buffer alive for the QImage wrapper
    return item


def test_eyedropper_prefers_paint_overlay(app):
    v = _make_viewer(app)
    fill = QColor(200, 30, 40, 255)
    v.paint_onto_overlay(QPointF(50, 100), QPointF(150, 100), fill, 25)

    # On the stroke the overlay color wins over the pixelated base image
    v.sample_color_at(QPointF(100, 100))
    assert v.paint_manager.paint_color == QColor(200, 30, 40, 255)

    # Off the stroke the base image is sampled as before
    v.sample_color_at(QPointF(100, 20))
    assert v.paint_manager.paint_color == QColor(128, 128, 128, 255)


def test_eyedropper_falls_back_after_paint_eraser(app):
    v = _make_viewer(app)
    fill = QColor(200, 30, 40, 255)
    v.paint_onto_overlay(QPointF(50, 100), QPointF(150, 100), fill, 25)
    v.paint_onto_overlay(
        QPointF(80, 100), QPointF(120, 100), QColor(0, 0, 0, 0), 25, erase=True
    )

    v.sample_color_at(QPointF(100, 100))
    assert v.paint_manager.paint_color == QColor(128, 128, 128, 255)


def test_eyedropper_blends_partial_overlay_alpha(app):
    v = _make_viewer(app)
    v.paint_overlay[100, 100] = (200, 30, 40, 128)

    v.sample_color_at(QPointF(100, 100))
    c = v.paint_manager.paint_color
    expected = tuple(
        (ov * 128 + 128 * 127) // 255 for ov in (200, 30, 40)
    )
    assert (c.red(), c.green(), c.blue()) == expected


def test_eyedropper_samples_inpaint_patch(app):
    v = _make_viewer(app)
    patch_color = QColor(90, 180, 120, 255)
    _add_patch(v, 60, 80, patch_color)

    # On the patch: the algorithm's fill color, not the background
    v.sample_color_at(QPointF(80, 100))
    assert v.paint_manager.paint_color == QColor(90, 180, 120, 255)

    # Off the patch: base image
    v.sample_color_at(QPointF(20, 20))
    assert v.paint_manager.paint_color == QColor(128, 128, 128, 255)


def test_eyedropper_paint_overlay_wins_over_patch(app):
    v = _make_viewer(app)
    _add_patch(v, 60, 80, QColor(90, 180, 120, 255))
    fill = QColor(200, 30, 40, 255)
    v.paint_onto_overlay(QPointF(70, 100), QPointF(110, 100), fill, 20)

    # Brush paint (z=0.6) sits above the patch (z=0.5) — sample what is on top
    v.sample_color_at(QPointF(90, 100))
    assert v.paint_manager.paint_color == QColor(200, 30, 40, 255)

    # Patch area not covered by the brush still yields the patch color
    v.sample_color_at(QPointF(70, 85))
    assert v.paint_manager.paint_color == QColor(90, 180, 120, 255)
