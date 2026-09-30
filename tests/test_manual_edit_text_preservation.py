"""Regression tests for the two manual-edit data-loss bugs:

1. Nudging a detection box by a pixel wiped the block's OCR text and
   translation (rect_item.handle_rectangle_change), and Ctrl+Z restored
   only the geometry (BoxesChangeCommand).
2. Async OCR / translation results were applied to the live block list of
   whatever page the run STARTED on, even if the user had already switched
   pages (manual_workflow on_*_ready) -- cross-contaminating page states
   on the next save.
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QGraphicsScene

from modules.detection.utils.geometry import geometry_change_invalidates_text
from modules.utils.textblock import TextBlock

from app.controllers.rect_item import RectItemController
from app.controllers.manual_workflow import ManualWorkflowController
from app.ui.commands.box import BoxesChangeCommand


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_block(xyxy, text="HELLO", translation="привет"):
    blk = TextBlock(
        text_bbox=np.array(xyxy),
        bubble_bbox=np.array(xyxy),
        text_class="text_bubble",
    )
    blk.text = text
    blk.texts = [text]
    blk.translation = translation
    return blk


# ── geometry_change_invalidates_text ─────────────────────────────────────


class TestGeometryInvalidation:
    def test_tiny_nudge_keeps_text(self):
        assert not geometry_change_invalidates_text(
            (100, 100, 200, 150), (104, 102, 204, 152), 0.0, 0.0
        )

    def test_recentre_inside_region_keeps_text(self):
        # 10 px shift on a 200x80 box (12.5 % of the small side) still keeps.
        assert not geometry_change_invalidates_text(
            (100, 100, 300, 180), (110, 100, 310, 180), 0.0, 0.0
        )

    def test_small_resize_keeps_text(self):
        # 3 px on a 300 px wide box is within the 4 px / 10 % tolerance.
        assert not geometry_change_invalidates_text(
            (0, 0, 300, 100), (0, 0, 303, 100), 0.0, 0.0
        )

    def test_real_resize_invalidates(self):
        assert geometry_change_invalidates_text(
            (0, 0, 100, 50), (0, 0, 200, 100), 0.0, 0.0
        )

    def test_drag_to_another_region_invalidates(self):
        assert geometry_change_invalidates_text(
            (0, 0, 100, 50), (300, 300, 400, 350), 0.0, 0.0
        )

    def test_rotation_invalidates(self):
        assert geometry_change_invalidates_text(
            (0, 0, 100, 50), (0, 0, 100, 50), 0.0, 5.0
        )

    def test_subdegree_rotation_keeps_text(self):
        assert not geometry_change_invalidates_text(
            (0, 0, 100, 50), (0, 0, 100, 50), 0.0, 0.5
        )

    def test_angle_wrap_compared_shortest_way(self):
        assert not geometry_change_invalidates_text(
            (0, 0, 100, 50), (0, 0, 100, 50), 359.5, 0.5
        )


# ── handle_rectangle_change ──────────────────────────────────────────────


class StubMain:
    def __init__(self, blk_list):
        self.blk_list = blk_list
        self.image_viewer = SimpleNamespace()
        self.invalidate_current_page_cache = lambda: None
        self.cache_manager = SimpleNamespace()


class TestHandleRectangleChange:
    def _ctrl(self, blk):
        return RectItemController(StubMain([blk]))

    def test_nudge_keeps_text_and_updates_geometry(self, app):
        blk = make_block([100, 100, 200, 150])
        ctrl = self._ctrl(blk)
        ctrl.handle_rectangle_change(
            (100, 100, 200, 150), (104, 102, 204, 152), 0.0, None, old_angle=0.0
        )
        assert blk.text == "HELLO"
        assert blk.translation == "привет"
        assert list(blk.xyxy) == [104, 102, 204, 152]
        assert blk.manual is True
        assert list(blk.bubble_xyxy) == [104, 102, 204, 152]

    def test_resize_wipes_text(self, app):
        blk = make_block([100, 100, 200, 150])
        ctrl = self._ctrl(blk)
        ctrl.handle_rectangle_change(
            (100, 100, 200, 150), (100, 100, 260, 220), 0.0, None, old_angle=0.0
        )
        assert blk.text == ""
        assert blk.texts == []
        assert blk.translation == ""

    def test_far_drag_wipes_text(self, app):
        blk = make_block([100, 100, 200, 150])
        ctrl = self._ctrl(blk)
        ctrl.handle_rectangle_change(
            (100, 100, 200, 150), (500, 500, 600, 550), 0.0, None, old_angle=0.0
        )
        # The old box no longer matches any block (redo already applied the
        # new geometry), so the wipe happens in BoxesChangeCommand.redo --
        # emulate the full rect_change_undo sequence here.
        cmd = BoxesChangeCommand(
            SimpleNamespace(_scene=QGraphicsScene()),
            SimpleNamespace(rect=(100, 100, 200, 150), rotation=0.0,
                            transform_origin=QPointF(0, 0)),
            SimpleNamespace(rect=(500, 500, 600, 550), rotation=0.0,
                            transform_origin=QPointF(0, 0)),
            [blk],
        )
        cmd.redo()
        ctrl.handle_rectangle_change(
            (100, 100, 200, 150), (500, 500, 600, 550), 0.0, None, old_angle=0.0
        )
        assert blk.text == ""
        assert blk.translation == ""


# ── BoxesChangeCommand: undo restores the wiped text ─────────────────────


class TestBoxesChangeCommandTextSnapshot:
    def _states(self, old, new, angle=0.0):
        make = lambda rect: SimpleNamespace(
            rect=rect, rotation=angle, transform_origin=QPointF(0, 0)
        )
        return make(old), make(new)

    def test_undo_restores_wiped_text(self, app):
        blk = make_block([0, 0, 100, 50])
        old_state, new_state = self._states((0, 0, 100, 50), (0, 0, 200, 100))
        cmd = BoxesChangeCommand(SimpleNamespace(_scene=QGraphicsScene()),
                                 old_state, new_state, [blk])

        cmd.redo()
        assert blk.text == "" and blk.translation == ""
        assert list(blk.xyxy) == [0, 0, 200, 100]

        cmd.undo()
        assert blk.text == "HELLO"
        assert blk.texts == ["HELLO"]
        assert blk.translation == "привет"
        assert list(blk.xyxy) == [0, 0, 100, 50]

        cmd.redo()
        assert blk.text == "" and blk.translation == ""
        assert list(blk.xyxy) == [0, 0, 200, 100]

    def test_nudge_undo_keeps_text(self, app):
        blk = make_block([100, 100, 200, 150])
        old_state, new_state = self._states((100, 100, 200, 150), (104, 102, 204, 152))
        cmd = BoxesChangeCommand(SimpleNamespace(_scene=QGraphicsScene()),
                                 old_state, new_state, [blk])

        cmd.redo()
        assert blk.text == "HELLO"
        cmd.undo()
        assert blk.text == "HELLO"
        assert list(blk.xyxy) == [100, 100, 200, 150]


# ── async race guard: _still_displaying ──────────────────────────────────


class TestStillDisplaying:
    def _ctrl(self, curr_idx, files=("a.jpg", "b.jpg")):
        ctrl = object.__new__(ManualWorkflowController)
        ctrl.main = SimpleNamespace(curr_img_idx=curr_idx, image_files=list(files))
        return ctrl

    def test_same_page(self):
        assert self._ctrl(1)._still_displaying("b.jpg") is True

    def test_switched_page(self):
        assert self._ctrl(1)._still_displaying("a.jpg") is False

    def test_no_current_page(self):
        assert self._ctrl(-1)._still_displaying("a.jpg") is False

    def test_none_file(self):
        assert self._ctrl(0)._still_displaying(None) is False

    def test_paths_compared_via_normcase(self):
        # os.path.normcase is a no-op on Linux (case-sensitive fs) and
        # lowercases on Windows; identical paths must always match.
        ctrl = self._ctrl(0, files=["/Data/A.JPG"])
        assert ctrl._still_displaying("/Data/A.JPG") is True
        assert ctrl._still_displaying("/Data/B.JPG") is False
