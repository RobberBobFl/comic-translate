from PySide6 import QtGui
from PySide6.QtGui import QUndoCommand

import numpy as np


class PaintCommand(QUndoCommand):
    """Undo/redo command for a single retouch paint / erase stroke.

    Stores independent copies of the paint overlay (RGBA numpy array) before
    and after the stroke so undo/redo can restore the exact pixel state.
    """

    def __init__(self, viewer, before, after):
        super().__init__("Retouch paint")
        self.viewer = viewer
        self.before = before  # numpy RGBA (H, W, 4) or None
        self.after = after    # numpy RGBA (H, W, 4) or None

    def _apply(self, overlay):
        self.viewer.set_paint_overlay(overlay)

    def undo(self):
        self._apply(self.before)

    def redo(self):
        self._apply(self.after)
