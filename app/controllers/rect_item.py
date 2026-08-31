from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from PySide6.QtCore import QRectF, QPointF
from PySide6.QtGui import QColor

from app.ui.canvas.rectangle import MoveableRectItem
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.commands.box import (
    AddRectangleCommand,
    AddTextItemCommand,
    BoxesChangeCommand,
    TextItemMoveCommand,
)

from modules.detection.utils.geometry import do_rectangles_overlap
from modules.utils.textblock import TextBlock
from modules.utils.language_utils import get_language_code
from modules.rendering.render import is_vertical_block

if TYPE_CHECKING:
    from controller import ComicTranslate


class RectItemController:
    def __init__(self, main: ComicTranslate):
        self.main = main

    def connect_rect_item_signals(self, rect_item: MoveableRectItem, force_reconnect: bool = False):
        if getattr(rect_item, "_ct_signals_connected", False) and not force_reconnect:
            return

        if force_reconnect:
            try:
                rect_item.signals.change_undo.disconnect(self.rect_change_undo)
            except (TypeError, RuntimeError):
                pass
            if hasattr(rect_item, "_ct_ocr_slot"):
                try:
                    rect_item.signals.ocr_block.disconnect(rect_item._ct_ocr_slot)
                except (TypeError, RuntimeError):
                    pass
            if hasattr(rect_item, "_ct_translate_slot"):
                try:
                    rect_item.signals.translate_block.disconnect(rect_item._ct_translate_slot)
                except (TypeError, RuntimeError):
                    pass

        if not hasattr(rect_item, "_ct_ocr_slot"):
            rect_item._ct_ocr_slot = lambda: self.main.ocr(True)
        if not hasattr(rect_item, "_ct_translate_slot"):
            rect_item._ct_translate_slot = lambda: self.main.translate_image(True)

        rect_item.signals.change_undo.connect(self.rect_change_undo)
        rect_item.signals.ocr_block.connect(rect_item._ct_ocr_slot)
        rect_item.signals.translate_block.connect(rect_item._ct_translate_slot)
        rect_item._ct_signals_connected = True

    def handle_rectangle_selection(self, rect: QRectF):
        rect = rect.getCoords()
        self.main.curr_tblock = self.find_corresponding_text_block(rect, 0.5)
        if self.main.curr_tblock:
            self.main.s_text_edit.blockSignals(True)
            self.main.t_text_edit.blockSignals(True)
            self.main.s_text_edit.setPlainText(self.main.curr_tblock.text)
            self.main.t_text_edit.setPlainText(self.main.curr_tblock.translation)
            self.main.s_text_edit.blockSignals(False)
            self.main.t_text_edit.blockSignals(False)
        else:
            self.main.s_text_edit.clear()
            self.main.t_text_edit.clear()
            self.main.curr_tblock = None

    def handle_rectangle_creation(self, rect_item: MoveableRectItem):
        self.connect_rect_item_signals(rect_item)
        new_rect = rect_item.mapRectToScene(rect_item.rect())
        x1, y1, w, h = new_rect.getRect()
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        new_rect_coords = (x1, y1, x1 + w, y1 + h)

        new_blk = TextBlock(text_bbox=np.array(new_rect_coords))
        new_blk.manual = True
        self.main.blk_list.append(new_blk)

        is_manual_text = (self.main.image_viewer.current_tool == 'manual_text')
        if is_manual_text:
            self._create_manual_text_item(new_blk, rect_item)
        else:
            command = AddRectangleCommand(self.main, rect_item, new_blk, self.main.blk_list)
            self.main.undo_group.activeStack().push(command)

        self._sync_to_state()

        if is_manual_text:
            self.main.set_tool(None)
            self.main.t_text_edit.setFocus()

    def _create_manual_text_item(self, blk: TextBlock, rect_item: MoveableRectItem):
        """Build a TextBlockItem for a manually-drawn box without going through OCR."""
        rs = self.main.text_ctrl.render_settings()
        text_color = QColor(rs.color)
        outline_color = QColor(rs.outline_color) if rs.outline else None
        align_id = self.main.alignment_tool_group.get_dayu_checked()
        alignment = self.main.button_to_alignment[align_id]
        line_spacing = float(self.main.line_spacing_dropdown.currentText())
        target_lang = self.main.lang_mapping.get(self.main.t_combo.currentText(), None)
        trg_lng_cd = get_language_code(target_lang)
        vertical = is_vertical_block(blk, trg_lng_cd)

        bw, bh = (blk.xyxy[2] - blk.xyxy[0], blk.xyxy[3] - blk.xyxy[1])
        properties = TextItemProperties(
            text="",
            plain_text="",
            font_family=rs.font_family,
            font_size=rs.min_font_size or 12,
            text_color=text_color,
            alignment=alignment,
            line_spacing=line_spacing,
            outline_color=outline_color,
            outline_width=float(self.main.outline_width_dropdown.currentText()),
            bold=rs.bold,
            italic=rs.italic,
            underline=rs.underline,
            direction=rs.direction,
            position=(int(blk.xyxy[0]), int(blk.xyxy[1])),
            rotation=blk.angle,
            width=bw if blk.angle == 0 and not vertical else None,
            vertical=vertical,
        )
        text_item = self.main.image_viewer.add_text_item(properties)
        text_item.set_plain_text("")

        command = AddTextItemCommand(self.main, text_item)
        self.main.undo_group.activeStack().push(command)

        text_item.selected = True
        text_item.setSelected(True)
        text_item.item_selected.emit(text_item)

    def handle_rectangle_deletion(self, rect: QRectF):
        rect_coords = rect.getCoords()
        current_text_block = self.find_corresponding_text_block(rect_coords, 0.5)
        self.main.blk_list.remove(current_text_block)
        self._sync_to_state()

    def handle_rectangle_change(
            self, 
            old_rect_coords: tuple, 
            new_rect_coords: tuple, 
            new_angle: float, 
            new_tr_origin: QPointF
        ):
        # Find the corresponding TextBlock in blk_list
        # Find the TextBlock whose region contains the center of the edited
        # rectangle. When a detection box is resized the signal carries the
        # box rect, but when the (on-top) text overlay is resized/moved it
        # carries the *text* rect -- which is much smaller and sits inside the
        # detection box, so an IoU-based overlap test (threshold 0.2) fails to
        # match and the block's xyxy is never updated. Matching by center is
        # unambiguous even for bubbles that are close together.
        ox1, oy1, ox2, oy2 = old_rect_coords
        ocx, ocy = (float(ox1) + float(ox2)) / 2.0, (float(oy1) + float(oy2)) / 2.0
        target = None
        for blk in self.main.blk_list:
            bx1, by1, bx2, by2 = blk.xyxy[:4]
            if bx1 <= ocx <= bx2 and by1 <= ocy <= by2:
                target = blk
                break
        if target is None:
            for blk in self.main.blk_list:
                if do_rectangles_overlap(blk.xyxy, old_rect_coords, 0.2):
                    target = blk
                    break

        if target is not None:
            # Update the TextBlock coordinates
            target.xyxy[:] = [int(new_rect_coords[0]),
                              int(new_rect_coords[1]),
                              int(new_rect_coords[2]),
                              int(new_rect_coords[3])]
            # OCR engines crop the block using blk.bubble_xyxy when present
            # (every engine falls back to xyxy only if bubble_xyxy is None).
            # Detected blocks carry the *original* bubble box in bubble_xyxy,
            # so without syncing it here the OCR would still crop the old
            # region after a resize. Mirror the edited geometry into
            # bubble_xyxy so the crop follows the resize.
            target.bubble_xyxy = [int(new_rect_coords[0]),
                                 int(new_rect_coords[1]),
                                 int(new_rect_coords[2]),
                                 int(new_rect_coords[3])]
            target.angle = new_angle if new_angle else 0
            target.tr_origin_point = (new_tr_origin.x(), new_tr_origin.y()) if new_tr_origin else ()
            target.manual = True
            # The block was just reshaped, so any previously recognized
            # text/translation no longer matches its (edited) region.
            # Drop it so the side panel / canvas stop showing stale
            # recognition and so an explicit per-block OCR does not skip
            # this block (OCR_image returns early when a block already
            # has text).
            target.text = ''
            if getattr(target, 'texts', None) is not None:
                target.texts = []
            target.translation = ''
        self._sync_to_state()

    def _sync_to_state(self) -> None:
        """Persist manual block edits back into the page state.

        In webtoon mode ``main.blk_list`` is only a copy of the saved block
        list, so edits (add / delete / resize) must be written back to
        ``image_states`` or they are lost on the next state read.
        """
        try:
            self.main.manual_workflow_ctrl.sync_blk_list_to_state()
        except Exception:
            pass
        # Manual block edits don't change the page pixels, so the OCR /
        # translation cache (keyed by the whole-image hash) would otherwise
        # keep serving stale text for the edited blocks. Drop those entries so
        # the next Recognize / Translate re-runs on the current blocks.
        self.main.invalidate_current_page_cache()

    def rect_change_undo(self, old_state, new_state):
        command = BoxesChangeCommand(self.main.image_viewer, old_state,
                                         new_state, self.main.blk_list)
        self.main.undo_group.activeStack().push(command)
        self.handle_rectangle_change(
            old_state.rect, 
            new_state.rect,
            new_state.rotation,
            new_state.transform_origin
        )

    def text_overlay_change_undo(self, old_state, new_state):
        """Handle moving/rotating a rendered text overlay (TextBlockItem).

        Unlike rect_change_undo this must NOT clear the block's source text or
        translation, nor overwrite its bubble region (``xyxy``/``bubble_xyxy``).
        Dragging the rendered text only changes where it is drawn -- the
        recognized block and its text stay intact so the user can still
        re-translate. The overlay's new position is persisted automatically via
        the canvas text_items_state (item.pos()).
        """
        command = TextItemMoveCommand(self.main.image_viewer, old_state, new_state)
        self.main.undo_group.activeStack().push(command)


    def find_corresponding_text_block(self, rect: tuple[float], iou_threshold: int = 0.5):
        for blk in self.main.blk_list:
            if do_rectangles_overlap(rect, blk.xyxy, iou_threshold):
                return blk
        return None

    def find_corresponding_rect(self, tblock: TextBlock, iou_threshold: int):
        for rect in self.main.image_viewer.rectangles:
            mp_rect = rect.mapRectToScene(rect.rect())
            x1, y1, w, h = mp_rect.getRect()
            rect_coord = (x1, y1, x1 + w, y1 + h)
            if do_rectangles_overlap(rect_coord, tblock.xyxy, iou_threshold):
                return rect
        return None
