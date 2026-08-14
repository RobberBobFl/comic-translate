"""Manual retouch paint manager.

Coordinates the three retouch tools (eyedropper / paint / paint-eraser):
  * eyedropper  -> samples a color from the image (handled on the viewer)
  * paint       -> paints the sampled color onto the per-page paint overlay
  * paint-eraser-> erases painted pixels (restores transparency)

The actual pixel drawing lives on the ImageViewer (which owns the overlay
QImage); this manager only tracks color/size, stroke state and undo.
"""

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPen, QPainter, QCursor, QPixmap

import numpy as np


def _same_overlay(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.shape == b.shape and bool(np.array_equal(a, b))


class PaintManager:
    def __init__(self, viewer):
        self.viewer = viewer
        self.paint_color = QColor(255, 255, 255, 255)
        self.paint_size = 25
        self.painting = False
        self._erasing = False
        self._before = None
        self._last_pos = None
        self._cursor_scaled_size = 25
        self.paint_cursor = self._make_cursor(self._cursor_scaled_size, self.paint_color)
        self._rect_origin = None
        self._rect_item = None

    # ------------------------------------------------------------------ cursor
    def _make_cursor(self, size, color):
        size = max(1, int(size))
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setPen(QPen(color, max(1, max(1, size // 12))))
        p.drawEllipse(0, 0, size - 1, size - 1)
        p.end()
        return QCursor(pix, size // 2, size // 2)

    def set_paint_color(self, color: QColor):
        self.paint_color = QColor(color)
        self.paint_cursor = self._make_cursor(self._cursor_scaled_size, self.paint_color)

    def set_paint_size(self, size, scaled_size):
        self.paint_size = size
        self._cursor_scaled_size = scaled_size
        self.paint_cursor = self._make_cursor(scaled_size, self.paint_color)

    # ------------------------------------------------------------------ strokes
    def start_stroke(self, scene_pos: QPointF, erase: bool = False):
        self._before = self.viewer.get_paint_overlay()
        self._erasing = erase
        self.painting = True
        self._last_pos = scene_pos
        self.viewer.paint_onto_overlay(scene_pos, scene_pos, self.paint_color, self.paint_size, erase=erase)

    def continue_stroke(self, scene_pos: QPointF):
        if not self.painting:
            return
        self.viewer.paint_onto_overlay(
            self._last_pos, scene_pos, self.paint_color, self.paint_size, erase=self._erasing
        )
        self._last_pos = scene_pos

    def end_stroke(self):
        if not self.painting:
            return
        # Reset the live stroke state first so the stroke can never "stick"
        # even if the snapshot/undo emission below raises.
        self.painting = False
        after = self.viewer.get_paint_overlay()
        before = self._before
        self._before = None
        self._last_pos = None
        if _same_overlay(before, after):
            return
        from app.ui.commands.paint import PaintCommand

        self.viewer.command_emitted.emit(PaintCommand(self.viewer, before, after))

    # -------------------------------------------------------------- rect fill
    def start_rectfill(self, scene_pos: QPointF):
        self._before = self.viewer.get_paint_overlay()
        self.painting = True
        self._rect_origin = scene_pos
        self._last_pos = scene_pos
        self._rect_item = QtWidgets.QGraphicsRectItem(QRectF(scene_pos, scene_pos))
        self._rect_item.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine))
        self._rect_item.setBrush(Qt.NoBrush)
        self._rect_item.setZValue(0.7)
        self.viewer._scene.addItem(self._rect_item)

    def continue_rectfill(self, scene_pos: QPointF):
        if not self.painting or self._rect_item is None:
            return
        self._last_pos = scene_pos
        self._rect_item.setRect(QRectF(self._rect_origin, scene_pos).normalized())

    def end_rectfill(self):
        if not self.painting:
            return
        self.painting = False
        if self._rect_item is not None:
            self.viewer._scene.removeItem(self._rect_item)
            self._rect_item = None
        if self._rect_origin is not None and self._last_pos is not None:
            bbox = QRectF(self._rect_origin, self._last_pos).normalized()
            self.viewer.fill_rect_on_overlay(bbox, self.paint_color)
        self._rect_origin = None
        self._last_pos = None
        after = self.viewer.get_paint_overlay()
        before = self._before
        self._before = None
        if _same_overlay(before, after):
            return
        from app.ui.commands.paint import PaintCommand
        self.viewer.command_emitted.emit(PaintCommand(self.viewer, before, after))
