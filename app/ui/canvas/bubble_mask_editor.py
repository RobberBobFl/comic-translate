"""Raster mask editor for the "Expand Bubble" tool (scope-limited, self-contained).

The bubble mask recovered by ``modules.bubble.recovery`` is full-page uint8.
This widget shows it as an overlay and lets the user touch it up with a
brush (fill 255) / eraser (clear), keeping every edited pixel inside
``draw_clamp``. Nothing is written to the shared page state, undo stack or
``current_tool`` while editing; the final mask is returned via ``accepted``.

Scope note: this is a purpose-built editor for one bubble mask at a time.
It deliberately reuses only ``constrain_point``/``create_inpaint_cursor``/
``scale_size`` and the standard stroke geometry; it is not meant to grow
into a generic mask-editing framework.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPen, QPainter, QPainterPath

from app.ui.dayu_widgets.push_button import MPushButton
from app.ui.dayu_widgets.slider import MSlider
from app.ui.dayu_widgets.tool_button import MToolButton

logger = logging.getLogger(__name__)

# Measured worst-case poke of the expanded mask outside the visible zone is
# 28 px (30 pages / 361 blocks, expand +20 px). 32 px covers that plus the
# brush radius, and is the hard limit of manual editing.
DRAW_CLAMP_MARGIN = 32

# QPainter on Format_Grayscale8 ignores pen width (draws a ~1px line), so the
# mask is kept as RGBA8888 and read out on accept with a 50% alpha threshold
# (strokes are anti-aliased; any-nonzero read-out would fold the soft fringe
# into the mask and grow every stroke by ~0.5px).
_MASK_ALPHA = 200
_FRAGMENT_ALPHA = 220
_DIM_ALPHA = 150
_LIVE_Z = 0.75
_DISPLAY_Z = 0.7
_FRAME_Z = 0.69


def _overlay_qimage(arr: np.ndarray) -> QtGui.QImage:
    """Wrap a full-page RGBA numpy buffer in a QImage (no copy)."""
    h, w = arr.shape[:2]
    return QtGui.QImage(arr.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888)


class _MaskToolbar(QtWidgets.QDialog):
    """Floating (non-modal) toolbar: brush/eraser toggle, size, Accept, Cancel.

    Non-modal on purpose: a modal exec() would steal mouse input from the
    viewer and make painting impossible. Escape/X both route to rejected().
    """

    def __init__(self, editor, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Tool)
        self._editor = editor
        self.setWindowTitle(self.tr("Edit Bubble Mask"))
        self.setModal(False)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)

        self.brush_btn = MToolButton().svg("brush-fill.svg").icon_only()
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.setToolTip(self.tr("Brush (add to mask)"))

        self.eraser_btn = MToolButton().svg("eraser_fill.svg").icon_only()
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setToolTip(self.tr("Eraser (remove from mask)"))

        grp = QtWidgets.QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.brush_btn)
        grp.addButton(self.eraser_btn)
        self._mode_group = grp

        self.size_slider = MSlider()
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(editor.brush_size)
        self.size_slider.setToolTip(self.tr("Brush size"))
        self.size_slider.setFixedWidth(140)

        self.accept_btn = MPushButton(self.tr("Apply"))
        self.accept_btn.set_dayu_type(MPushButton.PrimaryType)
        self.cancel_btn = MPushButton(self.tr("Cancel"))

        for w in (self.brush_btn, self.eraser_btn):
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for w in (self.accept_btn, self.cancel_btn, self.size_slider):
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        lay.addWidget(self.brush_btn)
        lay.addWidget(self.eraser_btn)
        lay.addWidget(self.size_slider)
        lay.addStretch()
        lay.addWidget(self.accept_btn)
        lay.addWidget(self.cancel_btn)

        self.brush_btn.toggled.connect(self._on_mode_toggled)
        self.size_slider.valueChanged.connect(editor.set_size)
        self.accept_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _on_mode_toggled(self, checked: bool):
        if checked:
            self._editor.set_mode("brush")
        else:
            self._editor.set_mode("eraser")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.reject()
        super().closeEvent(event)


class BubbleMaskEditor(QObject):
    """Raster brush/eraser over one recovered bubble mask.

    Buffers are full-page (scene coords == pixel coords, no translation), but
    QPainter is clipped to ``draw_clamp`` so no mask pixel outside it can ever
    change, regardless of brush size.
    """

    accepted = Signal(object)   # edited mask, full-page uint8
    rejected = Signal()

    def __init__(self, main, viewer, mask, fragments=None, zone=None,
                 default_size=10):
        super().__init__(viewer if isinstance(viewer, QtCore.QObject) else None)
        self.main = main
        self.viewer = viewer
        self.brush_size = int(default_size)
        self.mode = "brush"

        # The editable mask lives in an RGBA buffer ( QPainter cannot draw
        # pen-width strokes on Grayscale8). "set" = opaque white, read out on
        # accept with a 50% alpha threshold (see _on_accept). Buffers are
        # full-page: scene coords == pixel coords.
        base = (np.asarray(mask) > 0)
        self._mask_rgba = np.zeros(base.shape[:2] + (4,), dtype=np.uint8)
        self._mask_rgba[base] = (255, 255, 255, 255)
        h, w = base.shape[:2]
        self._frag_np = (np.asarray(fragments) > 0) if fragments is not None \
            else np.zeros((h, w), bool)

        # Stable QImage wrappers, kept alive for the editor's lifetime so
        # painting never re-wraps a buffer whose wrapper was collected.
        self._mask_qimg = _overlay_qimage(self._mask_rgba)
        self._display = np.zeros((h, w, 4), dtype=np.uint8)
        self._display_qimg = _overlay_qimage(self._display)

        # visible zone (dim hole + frame) and the hard draw limit
        self.zone = self._resolve_zone(zone)
        self.draw_clamp = self._resolve_clamp(self.zone)

        self._rebuild_display()

        self._display_item = None
        self._dim_item = None
        self._frame_item = None
        self._live_item = None
        self._path = None
        self._last_pos = None
        self._stroke_dirty = False
        self._stopped = False
        self._toolbar = None

    # ------------------------------------------------------------------ setup
    def _resolve_zone(self, zone):
        h, w = self._mask_rgba.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in zone)
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    def _resolve_clamp(self, zone):
        x1, y1, x2, y2 = zone
        h, w = self._mask_rgba.shape[:2]
        m = DRAW_CLAMP_MARGIN
        return (max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m))

    def _rebuild_display(self):
        d = self._display
        d[:] = 0
        m = self._mask_rgba[..., 3] > 0
        d[m] = (0, 255, 0, _MASK_ALPHA)
        f = self._frag_np
        if f.any():
            d[f] = (255, 0, 0, _FRAGMENT_ALPHA)

    def _build_dim(self):
        h, w = self._mask_rgba.shape[:2]
        dim = np.full((h, w, 4), (0, 0, 0, _DIM_ALPHA), dtype=np.uint8)
        x1, y1, x2, y2 = self.zone
        dim[y1:y2, x1:x2] = 0
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(_overlay_qimage(dim)))
        item.setZValue(_FRAME_Z)
        return item

    def _build_frame(self):
        x1, y1, x2, y2 = self.zone
        item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
        item.setPen(QPen(QColor(255, 255, 255, 200), 1, Qt.DashLine))
        item.setBrush(Qt.NoBrush)
        item.setZValue(_FRAME_Z)
        return item

    # ----------------------------------------------------------------- public
    def start(self):
        scene = self.viewer._scene
        self._dim_item = self._build_dim()
        self._frame_item = self._build_frame()
        self._display_item = QtWidgets.QGraphicsPixmapItem(
            QtGui.QPixmap.fromImage(_overlay_qimage(self._display))
        )
        self._display_item.setZValue(_DISPLAY_Z)
        for it in (self._dim_item, self._frame_item, self._display_item):
            scene.addItem(it)

        self._toolbar = _MaskToolbar(self, self.viewer)
        self._toolbar.accepted.connect(self._on_accept)
        self._toolbar.rejected.connect(self._on_reject)
        self._toolbar.show()

        self.viewer.viewport().installEventFilter(self)
        self.viewer.page_changed.connect(self._on_page_changed)
        self._apply_cursor()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._last_pos = None
        self._path = None
        self._stroke_dirty = False

        try:
            self.viewer.viewport().removeEventFilter(self)
        except Exception:
            pass
        try:
            self.viewer.page_changed.disconnect(self._on_page_changed)
        except Exception:
            pass

        scene = self.viewer._scene
        for attr in ("_display_item", "_dim_item", "_frame_item", "_live_item"):
            item = getattr(self, attr, None)
            setattr(self, attr, None)
            if item is not None:
                try:
                    if item.scene() is not None:
                        scene.removeItem(item)
                except RuntimeError:
                    pass

        tb, self._toolbar = self._toolbar, None
        if tb is not None:
            tb.accepted.disconnect(self._on_accept)
            tb.rejected.disconnect(self._on_reject)
            tb.deleteLater()

        self.viewer.set_tool(self.viewer.current_tool)

    def set_mode(self, mode: str):
        self.mode = "eraser" if mode == "eraser" else "brush"
        self._apply_cursor()

    def set_size(self, size: int):
        self.brush_size = max(1, int(size))
        self._apply_cursor()

    # -------------------------------------------------------------- internals
    def _scaled_size(self) -> float:
        h, w = self._mask_rgba.shape[:2]
        return self.main.scale_size(self.brush_size, w, h)

    def _apply_cursor(self):
        cursor = self.viewer.drawing_manager.create_inpaint_cursor(
            "brush" if self.mode == "brush" else "eraser", int(self._scaled_size())
        )
        self.viewer.setCursor(cursor)

    def _contains(self, pos: QPointF, box) -> bool:
        x1, y1, x2, y2 = box
        return x1 <= pos.x() <= x2 and y1 <= pos.y() <= y2

    def _clamped(self, pos: QPointF) -> QPointF:
        x1, y1, x2, y2 = self.draw_clamp
        return QPointF(
            min(max(pos.x(), x1), x2),
            min(max(pos.y(), y1), y2),
        )

    # --------------------------------------------------------------- strokes
    def start_stroke(self, scene_pos: QPointF):
        if not self._contains(scene_pos, self.zone):
            return  # press outside the visible zone: never start
        self._path = QPainterPath()
        self._path.moveTo(scene_pos)
        pen = QPen(
            QColor(0, 255, 0, 200) if self.mode == "brush" else QColor(0, 0, 0, 160),
            self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin,
        )
        self._live_item = QtWidgets.QGraphicsPathItem(self._path)
        self._live_item.setPen(pen)
        self._live_item.setZValue(_LIVE_Z)
        self.viewer._scene.addItem(self._live_item)
        self._last_pos = scene_pos
        self._stroke_dirty = False
        self._paint_segment(scene_pos, scene_pos)

    def continue_stroke(self, scene_pos: QPointF):
        if self._path is None or self._last_pos is None:
            return
        pos = self._clamped(scene_pos)
        self._path.lineTo(pos)
        if self._live_item is not None:
            self._live_item.setPath(self._path)
        self._paint_segment(self._last_pos, pos)
        self._last_pos = pos

    def end_stroke(self, scene_pos: QPointF):
        if self._path is None:
            return
        # Paint the final segment only up to the clamp; if the release itself
        # landed outside draw_clamp we stop at the last legal point and do not
        # create a jump to the raw release position.
        if self._contains(scene_pos, self.draw_clamp):
            self._paint_segment(self._last_pos, self._clamped(scene_pos))
        self._last_pos = None
        self._path = None
        live, self._live_item = self._live_item, None
        if live is not None:
            try:
                self.viewer._scene.removeItem(live)
            except RuntimeError:
                pass
        if self._stroke_dirty:
            self._refresh_display()

    def _paint_segment(self, p0: QPointF, p1: QPointF):
        """Draw the segment into mask + display, hard-clipped to draw_clamp.

        clipRect is what actually guarantees "no mask pixel outside
        draw_clamp changes": the brush centre is clamped, but the pen radius
        can still cross the edge, so the painter itself is clipped.
        """
        if p0 is None or p1 is None:
            return
        x1, y1, x2, y2 = self.draw_clamp
        clip = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
        erase = self.mode == "eraser"
        # A zero-length line + CompositionMode_Clear paints nothing, so a
        # plain click degenerates to a filled dot instead.
        dot = (p0.x() == p1.x()) and (p0.y() == p1.y())

        # QImage wrappers must stay referenced: a wrapper that is collected
        # before its painter's end() invalidates the next wrap of another
        # buffer ("Cannot destroy paint device that is being painted").
        for buf, qimg, color in (
            (self._mask_rgba, self._mask_qimg, QColor(255, 255, 255, 255)),
            (self._display, self._display_qimg, QColor(0, 255, 0, _MASK_ALPHA)),
        ):
            painter = QPainter(qimg)
            painter.setClipRect(clip)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if erase:
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_Clear)
                painter.setBrush(QtGui.QBrush(QColor(0, 0, 0, 0)))
                pen = QPen(QColor(0, 0, 0, 0), self.brush_size)
            else:
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_Source)
                painter.setBrush(QtGui.QBrush(color))
                pen = QPen(color, self.brush_size)
            pen.setStyle(Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            if dot:
                r = self.brush_size / 2.0
                painter.drawEllipse(p0, r, r)
            else:
                painter.drawLine(QPointF(p0), QPointF(p1))
            painter.end()

        self._stroke_dirty = True

    def _refresh_display(self):
        if self._display_item is None:
            return
        self._display_item.setPixmap(QtGui.QPixmap.fromImage(self._display_qimg))

    # ---------------------------------------------------------------- events
    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QtCore.QEvent.Type.MouseButtonPress,
                  QtCore.QEvent.Type.MouseButtonDblClick):
            if event.button() == Qt.MouseButton.LeftButton:
                pos = self.viewer.mapToScene(event.position().toPoint())
                if self._contains(pos, self.zone):
                    self.start_stroke(pos)
                    return True
                return True  # outside the zone: swallow, never fall through
        elif et == QtCore.QEvent.Type.MouseMove:
            if self._path is not None:
                pos = self.viewer.mapToScene(event.position().toPoint())
                self.continue_stroke(pos)
                return True
        elif et == QtCore.QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._path is not None:
                pos = self.viewer.mapToScene(event.position().toPoint())
                self.end_stroke(pos)
                return True
        return False

    # ----------------------------------------------------------------- exits
    def _on_page_changed(self, _idx):
        logger.debug("[bubble-mask] page changed mid-edit -> cancel")
        self._on_reject()

    def _on_accept(self):
        # 50% alpha threshold: strokes are painted with Antialiasing, so the
        # buffer carries a soft fringe along every stroke edge. Reading any
        # nonzero alpha would fold that fringe into the mask as full pixels,
        # growing each stroke by ~0.5px; the threshold keeps the mask on the
        # stroke's geometric boundary (binary recover_bubble() masks pass
        # through unchanged).
        mask = (self._mask_rgba[..., 3] > 127).astype(np.uint8) * 255
        self.stop()
        if not mask.any():
            from app.ui.dayu_widgets.message import MMessage
            MMessage.warning(
                text=self.main.tr("The mask is empty; nothing to expand."),
                parent=self.main, duration=4, closable=True,
            )
            self.rejected.emit()
            return
        self.accepted.emit(mask)

    def _on_reject(self):
        self.stop()
        self.rejected.emit()
