"""Experimental manual "Expand Bubble" tool (MVP).

Works only inside the manual editor: for the selected text block it recovers
the bubble mask (recommended research path C->C2->Cx, with tail/fragment
trimming), shows a preview overlay, asks for an expansion in px, then repaints
the enlarged bubble (flat fill + anti-aliased morphological-ring contour) and
applies it as a single inpaint patch.

The patch goes through the existing ``PatchInsertCommand`` machinery, so Undo
works and the automatic pipeline is untouched. Enable analysis dumps with
``CT_BUBBLE_DEBUG=1`` (original / mask / tail_cut / expanded_mask /
final_result, next to the project file).
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from PySide6 import QtGui, QtWidgets

import imkit as imk
from app.ui.commands.inpaint import PatchInsertCommand
from app.ui.dayu_widgets.message import MMessage
from modules.bubble import recovery as br
from modules.detection.utils.content import get_inpaint_mask

logger = logging.getLogger(__name__)


class BubbleExpandController:
    def __init__(self, main):
        self.main = main
        self._preview_item = None
        self._last_mask = None

    # ------------------------------------------------------------------ entry
    def expand_selected(self):
        main = self.main
        viewer = main.image_viewer
        if not viewer.hasPhoto() or viewer.webtoon_mode:
            return

        blk = self._target_block()
        if blk is None:
            MMessage.info(
                text=main.tr("Select a text block first."),
                parent=main,
                duration=4,
                closable=True,
            )
            return

        rgb = viewer.get_image_array(include_patches=True)
        if rgb is None:
            return

        text_xyxy = tuple(int(v) for v in blk.xyxy)
        bubble = blk.bubble_xyxy
        bubble_xyxy = tuple(int(v) for v in bubble) if bubble is not None else None
        text_mask = get_inpaint_mask(list(text_xyxy), rgb)

        res = br.recover_bubble(rgb, text_xyxy, bubble_xyxy, text_mask)
        if res is None:
            self._clear_preview()
            MMessage.warning(
                text=main.tr("No bubble found around the selected text block."),
                parent=main,
                duration=5,
                closable=True,
            )
            return

        trim = res.get("trim", {})
        logger.debug(
            "[bubble-expand] method=%s stroke_w=%s trim=%s containment=%s",
            res["method"], res["stroke_w"],
            f"{trim.get('fragments', 0.0)}%" if trim.get("trimmed") else "no",
            trim.get("containment"),
        )

        self._show_preview(res["outer"], res.get("fragments"))

        px, ok = QtWidgets.QInputDialog.getInt(
            viewer,
            main.tr("Expand Bubble"),
            main.tr("Expand by (px):"),
            2, 1, 200,
        )
        if not ok:
            self._clear_preview()
            return

        expanded = br.expand_mask(res["outer"], px)
        if not expanded.any():
            self._clear_preview()
            return
        repainted = br.repaint(rgb, expanded, res["stroke_w"])

        self._apply_patch(repainted, expanded, res["stroke_w"])
        self._clear_preview()

        if os.environ.get("CT_BUBBLE_DEBUG", "") == "1":
            self._dump_debug(rgb, res, expanded, repainted, text_xyxy)

    # -------------------------------------------------------------- internals
    def _target_block(self):
        main = self.main
        blk = getattr(main, "curr_tblock", None)
        if blk is not None:
            return blk
        rects = main.image_viewer.get_selected_rectangles()
        if rects:
            r = rects[0].mapRectToScene(rects[0].rect())
            x1, y1, w, h = r.getRect()
            return main.rect_item_ctrl.find_corresponding_text_block(
                (int(x1), int(y1), int(x1 + w), int(y1 + h)), 0.5
            )
        return None

    def _show_preview(self, mask: np.ndarray, fragments: np.ndarray | None = None):
        self._clear_preview()
        viewer = self.main.image_viewer
        h, w = mask.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[mask > 0] = (0, 255, 0, 110)      # bubble body that will expand
        if fragments is not None and fragments.any():
            overlay[fragments > 0] = (255, 0, 0, 140)  # excluded tails / merged blobs
        qimg = QtGui.QImage(
            overlay.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888
        )
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(qimg))
        item.setZValue(0.7)
        viewer._scene.addItem(item)
        self._preview_item = item
        self._last_mask = mask

    def _clear_preview(self):
        item = self._preview_item
        self._preview_item = None
        self._last_mask = None
        if item is None:
            return
        try:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        except RuntimeError:
            pass

    def _apply_patch(self, repainted: np.ndarray, expanded: np.ndarray, stroke_w: int):
        main = self.main
        idx = main.curr_img_idx
        if idx < 0 or idx >= len(main.image_files):
            return
        file_path = main.image_files[idx]

        bbox = br.mask_bbox(expanded, pad=int(stroke_w) + 4, shape=expanded.shape[:2])
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        patch = {"bbox": (x1, y1, x2 - x1, y2 - y1),
                 "image": np.ascontiguousarray(repainted[y1:y2, x1:x2])}

        stack = main.undo_stacks.get(file_path) or main.undo_group.activeStack()
        if stack is None:
            return
        command = PatchInsertCommand(main, [patch], file_path, display=True)
        stack.push(command)

    # ------------------------------------------------------------------ debug
    def _dump_debug(self, rgb, res, expanded, repainted, text_xyxy):
        main = self.main
        stem = os.path.splitext(os.path.basename(
            main.image_files[main.curr_img_idx]))[0]
        blk_id = f"blk{abs(hash(tuple(text_xyxy))) % 10000:04d}"

        # next to the project file when saved (survives closing); else temp dir
        base = main.temp_dir
        project_file = getattr(main, "project_file", None)
        if project_file:
            base = os.path.dirname(project_file)
        out_dir = os.path.join(base, "bubble_expand_debug", f"{stem}_{blk_id}")
        os.makedirs(out_dir, exist_ok=True)

        crop_box = br.mask_bbox(expanded, pad=60, shape=rgb.shape[:2])
        if crop_box is None:
            return
        cx1, cy1, cx2, cy2 = crop_box
        crop = lambda img: np.ascontiguousarray(img[cy1:cy2, cx1:cx2])

        imk.write_image(os.path.join(out_dir, "original.png"), crop(rgb))
        imk.write_image(
            os.path.join(out_dir, "mask.png"),
            crop(cv2.cvtColor(res["outer"], cv2.COLOR_GRAY2RGB)),
        )
        fragments = res.get("fragments")
        if fragments is not None and fragments.any():
            tail_overlay = crop(rgb).copy()
            tail_overlay[crop(fragments) > 0] = (255, 0, 0)
            imk.write_image(os.path.join(out_dir, "tail_cut.png"), tail_overlay)
        imk.write_image(
            os.path.join(out_dir, "expanded_mask.png"),
            crop(cv2.cvtColor(expanded, cv2.COLOR_GRAY2RGB)),
        )
        imk.write_image(os.path.join(out_dir, "final_result.png"), crop(repainted))
        logger.info("[bubble-expand] debug dumps -> %s", out_dir)
