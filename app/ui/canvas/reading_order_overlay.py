from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QBrush, QPen, QFont


class ReadingOrderOverlay:
    """Temporary number badges displayed on blocks during Reading Order mode."""

    BADGE_RADIUS = 28
    BADGE_COLOR = QColor(40, 120, 220, 200)
    BADGE_TEXT_COLOR = QColor(255, 255, 255)
    BADGE_BORDER = QColor(20, 80, 180)

    def __init__(self, scene: QtWidgets.QGraphicsScene):
        self.scene = scene
        self._items: list[QtWidgets.QGraphicsEllipseItem] = []
        self._text_items: list[QtWidgets.QGraphicsTextItem] = []

    def update(self, blk_list: list) -> None:
        """Rebuild all number badges from current blk_list order."""
        self.clear()
        font = QFont("sans-serif", 20, QFont.Weight.Bold)
        for i, blk in enumerate(blk_list):
            if blk.xyxy is None:
                continue
            x1, y1 = int(blk.xyxy[0]), int(blk.xyxy[1])
            badge_center_x = x1 - self.BADGE_RADIUS - 2
            badge_center_y = y1 - self.BADGE_RADIUS - 2

            # Create ellipse centered on badge_center
            ellipse = self.scene.addEllipse(
                QtCore.QRectF(
                    badge_center_x - self.BADGE_RADIUS,
                    badge_center_y - self.BADGE_RADIUS,
                    self.BADGE_RADIUS * 2,
                    self.BADGE_RADIUS * 2,
                ),
                QPen(self.BADGE_BORDER, 1.5),
                QBrush(self.BADGE_COLOR),
            )
            ellipse.setZValue(10)
            ellipse.setAcceptHoverEvents(False)
            ellipse.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            ellipse.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self._items.append(ellipse)

            # Create text, then center it inside the ellipse using its actual bounding rect
            text = self.scene.addText(str(i + 1), font)
            text.setDefaultTextColor(self.BADGE_TEXT_COLOR)
            text.setZValue(11)
            text.setAcceptHoverEvents(False)
            text.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            text.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            br = text.boundingRect()
            text.setPos(
                badge_center_x - br.width() / 2,
                badge_center_y - br.height() / 2,
            )
            self._text_items.append(text)

    def clear(self) -> None:
        """Remove all overlay items from the scene."""
        for item in self._items:
            try:
                if item.scene() is not None:
                    self.scene.removeItem(item)
            except RuntimeError:
                pass
        for item in self._text_items:
            try:
                if item.scene() is not None:
                    self.scene.removeItem(item)
            except RuntimeError:
                pass
        self._items.clear()
        self._text_items.clear()
