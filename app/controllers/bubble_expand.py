"""Experimental manual "Expand Bubble" tool (MVP).

Works only inside the manual editor: for the selected text block it recovers
the bubble mask (recommended research path C->C2->Cx, with tail/fragment
trimming), shows a preview overlay, asks for an expansion in px, then repaints
the enlarged bubble (flat fill + distance-transform-ring contour at the
measured stroke width) and applies it as a single inpaint patch.

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
from PySide6 import QtCore, QtGui, QtWidgets

import imkit as imk
from app.ui.canvas.bubble_mask_editor import BubbleMaskEditor
from app.ui.commands.inpaint import PatchInsertCommand
from app.ui.dayu_widgets.message import MMessage
from app.ui.dayu_widgets.push_button import MPushButton
from modules.bubble import recovery as br
from modules.detection.utils.content import get_inpaint_mask

logger = logging.getLogger(__name__)

# QDialog.done() result for "Edit Mask" (outside the standard accept/reject range)
_EDIT_MASK_RESULT = 100


class BubbleExpandController:
    def __init__(self, main):
        self.main = main
        self._preview_item = None
        self._last_mask = None
        self._active_editor = None
        self._last_res = None
        self._last_rgb = None

    # ------------------------------------------------------------------ entry
    def expand_selected(self):
        main = self.main
        viewer = main.image_viewer
        if not viewer.hasPhoto() or viewer.webtoon_mode:
            return

        if self._active_editor is not None:
            MMessage.info(
                text=main.tr("Finish the current mask edit first."),
                parent=main,
                duration=4,
                closable=True,
            )
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
            "[bubble-expand] method=%s stroke_w=%s fill=%s trim=%s containment=%s",
            res["method"], res["stroke_w"], res.get("fill"),
            f"{trim.get('fragments', 0.0)}%" if trim.get("trimmed") else "no",
            trim.get("containment"),
        )

        self._last_res = res
        self._last_rgb = rgb

        self._show_preview(res["outer"], res.get("fragments"))

        # visible zone = inflated detector box + stroke pad (same pad the
        # patch bbox uses), clipped to the frame; the editor adds its own
        # DRAW_CLAMP_MARGIN for the hard draw limit.
        zx1, zy1, zx2, zy2 = br.inflate(bubble_xyxy, 0.12, 8, shape=rgb.shape[:2])
        pad = int(res["stroke_w"]) + 4
        h, w = rgb.shape[:2]
        zone = (max(0, zx1 - pad), max(0, zy1 - pad),
                min(w, zx2 + pad), min(h, zy2 + pad))
        if not self._ask_expand(res, zone):
            self._clear_preview()

    # --------------------------------------------------- edit-mask escalation
    def _ask_expand(self, res, zone) -> bool:
        """px + Edit Mask choice. Returns False if the user cancelled."""
        main, viewer = self.main, self.main.image_viewer

        dlg = QtWidgets.QDialog(viewer)
        dlg.setWindowTitle(main.tr("Expand Bubble"))
        form = QtWidgets.QFormLayout(dlg)
        spin = QtWidgets.QSpinBox()
        spin.setRange(1, 200)
        spin.setValue(2)
        form.addRow(main.tr("Expand by (px):"), spin)

        btns = QtWidgets.QHBoxLayout()
        edit_btn = MPushButton(main.tr("Edit Mask"))
        ok_btn = MPushButton(main.tr("Expand"))
        ok_btn.set_dayu_type(MPushButton.PrimaryType)
        cancel_btn = MPushButton(main.tr("Cancel"))
        btns.addWidget(edit_btn)
        btns.addStretch()
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        form.addRow(btns)

        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        edit_btn.clicked.connect(lambda: dlg.done(_EDIT_MASK_RESULT))

        dlg.exec()
        result = dlg.result()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._continue_expand(res, spin.value())
            return True
        if result == _EDIT_MASK_RESULT:
            self._start_editor(res, zone)
            return True
        return False

    def _start_editor(self, res, zone):
        main = self.main
        editor = BubbleMaskEditor(
            main, main.image_viewer, res["outer"], res.get("fragments"),
            zone=zone, default_size=10,
        )
        self._active_editor = editor
        self._clear_preview()
        editor.accepted.connect(self._on_mask_edited)
        editor.rejected.connect(self._on_edit_cancelled)
        editor.start()

    def _on_mask_edited(self, mask):
        res = self._last_res
        self._active_editor = None
        if res is None:
            return
        self._show_preview(mask, None)
        main = self.main
        px, ok = QtWidgets.QInputDialog.getInt(
            main.image_viewer,
            main.tr("Expand Bubble"),
            main.tr("Expand by (px):"),
            2, 1, 200,
        )
        if not ok:
            self._clear_preview()
            return
        self._continue_expand(res, px, mask=mask)

    def _on_edit_cancelled(self):
        self._active_editor = None
        self._last_res = None
        self._last_rgb = None
        self._clear_preview()

    def _continue_expand(self, res, px, mask=None):
        """expand -> repaint -> apply. Uses the edited mask when given."""
        main = self.main
        rgb = self._last_rgb
        base = mask if mask is not None else res["outer"]

        expanded = br.expand_mask(base, px)
        if not expanded.any():
            self._clear_preview()
            return
        stroke_w = res["stroke_w"]
        repainted = br.repaint(
            rgb, expanded, stroke_w, res.get("fill", (255, 255, 255)),
        )

        self._apply_patch(repainted, expanded, stroke_w)
        self._clear_preview()

        if os.environ.get("CT_BUBBLE_DEBUG", "") == "1":
            self._dump_debug(rgb, res, expanded, repainted, mask)

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
    def _dump_debug(self, rgb, res, expanded, repainted, mask=None):
        main = self.main
        stem = os.path.splitext(os.path.basename(
            main.image_files[main.curr_img_idx]))[0]
        ref = tuple(int(v) for v in br.mask_bbox(expanded, pad=0) or (0, 0, 0, 0))
        blk_id = f"blk{abs(hash(ref)) % 10000:04d}"

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
        base_mask = mask if mask is not None else res["outer"]
        imk.write_image(
            os.path.join(out_dir, "mask.png"),
            crop(cv2.cvtColor(base_mask, cv2.COLOR_GRAY2RGB)),
        )
        if mask is not None:
            edited_overlay = crop(rgb).copy()
            edited_overlay[crop(mask) > 0] = (0, 255, 0)
            imk.write_image(os.path.join(out_dir, "edited_mask.png"),
                            edited_overlay)
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
