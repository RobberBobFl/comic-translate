"""Shared fakes for the BubbleMaskEditor / BubbleExpandController tests.

Headless: everything runs on the offscreen Qt platform. The fakes only
implement what the editor/controller actually touch -- no pipeline, no
Detection, no real image files.
"""
import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QUndoGroup, QUndoStack

import pytest

CANVAS_H, CANVAS_W = 200, 300


class _FakeDrawingManager:
    def create_inpaint_cursor(self, tool, size):
        return QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor)


class FakeViewer(QtWidgets.QGraphicsView):
    """ImageViewer stand-in: identity scene mapping, no real photo."""

    page_changed = QtCore.Signal(int)

    def __init__(self, image=None):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene()
        self.setScene(self._scene)
        self.current_tool = None
        self.webtoon_mode = False
        self.drawing_manager = _FakeDrawingManager()
        self._image = image

    def hasPhoto(self):
        return True

    def mapToScene(self, point):
        return QtCore.QPointF(point)

    def constrain_point(self, point):
        return QtCore.QPointF(
            min(max(point.x(), 0), CANVAS_W),
            min(max(point.y(), 0), CANVAS_H),
        )

    def setCursor(self, cursor):
        pass

    def set_tool(self, tool):
        self.current_tool = tool

    def get_image_array(self, include_patches=False):
        return None if self._image is None else self._image.copy()


class FakeMain(QtWidgets.QWidget):
    """MainWindow stand-in: undo storage + the bits _apply_patch reaches."""

    def __init__(self, viewer, image=None, block=None):
        super().__init__()
        self.image_viewer = viewer
        self.curr_tblock = block
        self.curr_img_idx = 0
        self.image_files = ["page.jpg"]
        stack = QUndoStack(self)
        self.undo_group = QUndoGroup(self)
        self.undo_group.addStack(stack)
        self.undo_stacks = {"page.jpg": stack}
        self.image_patches = {}
        self.in_memory_patches = {}
        self.project_file = None
        self._stack = stack
        self._image = image
    def scale_size(self, size, width, height):
        return float(size)

    def tr(self, text):
        return text


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viewer(app):
    return FakeViewer()


@pytest.fixture
def main(viewer):
    return FakeMain(viewer)


@pytest.fixture
def bubble_main(app, bubble_image, tmp_path):
    """FakeMain wired to the synthetic bubble page + a selected text block."""
    viewer = FakeViewer(bubble_image)

    class _Blk:
        xyxy = (100, 50, 200, 150)
        bubble_xyxy = (90, 40, 210, 160)

    main = FakeMain(viewer, image=bubble_image, block=_Blk())
    main.temp_dir = str(tmp_path / "ct_temp")
    return main


@pytest.fixture
def blank_mask():
    """Full-page mask with one white block covering (100,50)-(200,150)."""
    mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    mask[50:150, 100:200] = 255
    return mask


@pytest.fixture
def blank_fragments(blank_mask):
    frag = np.zeros_like(blank_mask)
    frag[150:170, 200:230] = 255
    return frag


@pytest.fixture
def bubble_image():
    """Grey page with a white bubble (black outline) at (95,45)-(205,155)."""
    import cv2

    img = np.full((CANVAS_H, CANVAS_W, 3), 220, np.uint8)
    cv2.rectangle(img, (95, 45), (205, 155), (255, 255, 255), -1)
    cv2.rectangle(img, (95, 45), (205, 155), (30, 30, 30), 6)
    return img


@pytest.fixture(autouse=True)
def _quiet_messages(monkeypatch):
    """MMessage spawns real widgets; stub it so tests stay headless."""
    from app.ui.dayu_widgets import message

    monkeypatch.setattr(message.MMessage, "info", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(message.MMessage, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(message.MMessage, "success", staticmethod(lambda *a, **k: None))
    yield
