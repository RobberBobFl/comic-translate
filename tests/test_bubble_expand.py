"""Headless tests for BubbleExpandController integration.

Covers:
  - reentrancy guard: a second Expand call while the editor is open is ignored
  - recover_bubble finds the synthetic bubble
  - editor round-trip (brush + erase -> Accept) commits exactly ONE
    PatchInsertCommand, i.e. strokes never reach the undo stack individually
  - the edited mask stays inside draw_clamp + the requested expansion
"""

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from app.controllers.bubble_expand import BubbleExpandController
from app.ui.canvas.bubble_mask_editor import BubbleMaskEditor
from modules.bubble import recovery as br

TEXT_XYXY = (100, 50, 200, 150)
BUBBLE_XYXY = (90, 40, 210, 160)


class _DummyEditor:
    """Stand-in for an open editor (only identity matters for the guard)."""


@pytest.fixture
def controller(bubble_main):
    return BubbleExpandController(bubble_main)


def test_reentrancy_guard_ignores_second_expand(controller, bubble_main):
    controller._active_editor = _DummyEditor()
    viewer = bubble_main.image_viewer

    controller.expand_selected()

    assert controller._active_editor is not None, "open editor must survive"
    assert controller._last_res is None
    assert controller._preview_item is None
    assert len(viewer._scene.items()) == 0


def test_recover_finds_synthetic_bubble(bubble_image):
    res = br.recover_bubble(bubble_image, TEXT_XYXY, BUBBLE_XYXY, None)
    assert res is not None
    assert res["method"] == "C->Cx"
    assert res["stroke_w"] > 0
    assert br.mask_bbox(res["outer"], pad=0) is not None


def test_editor_roundtrip_commits_single_undo_command(controller, bubble_main,
                                                      bubble_image):
    viewer = bubble_main.image_viewer
    res = br.recover_bubble(bubble_image, TEXT_XYXY, BUBBLE_XYXY, None)

    zx1, zy1, zx2, zy2 = br.inflate(BUBBLE_XYXY, 0.12, 8, shape=bubble_image.shape[:2])
    pad = int(res["stroke_w"]) + 4
    h, w = bubble_image.shape[:2]
    zone = (max(0, zx1 - pad), max(0, zy1 - pad),
            min(w, zx2 + pad), min(h, zy2 + pad))

    editor = BubbleMaskEditor(controller.main, viewer, res["outer"],
                              res.get("fragments"), zone=zone, default_size=10)
    controller._active_editor = editor
    controller._last_res = res
    controller._last_rgb = bubble_image
    editor.start()

    editor.set_mode("brush")
    editor.set_size(8)
    editor.start_stroke(QPointF(140, 120))
    editor.continue_stroke(QPointF(190, 140))
    editor.end_stroke(QPointF(190, 140))

    editor.set_mode("eraser")
    editor.set_size(6)
    editor.start_stroke(QPointF(150, 100))
    editor.end_stroke(QPointF(150, 100))

    captured = {}
    editor.accepted.connect(lambda m: captured.setdefault("mask", m))
    editor._on_accept()

    edited = captured["mask"]
    assert br.mask_bbox(edited, pad=0) is not None

    controller._active_editor = None
    controller._continue_expand(res, 5, mask=edited)

    stack = bubble_main.undo_stacks["page.jpg"]
    assert stack.count() == 1, "one PatchInsertCommand for the whole edit"

    # Undo must remove the patch and leave the stacks empty again
    stack.undo()
    assert bubble_main.image_patches["page.jpg"] == []
    assert stack.count() == 1  # command still on the stack, just undone


def test_repaint_transplant_preserves_stroke_thickness_and_darkness(bubble_image):
    import cv2

    res = br.recover_bubble(bubble_image, TEXT_XYXY, BUBBLE_XYXY, None)
    assert res is not None
    outer = res["outer"]
    inner = res["interior"]
    expanded = br.expand_mask(outer, 5)

    rep_outer = br.repaint_transplant(
        bubble_image, outer, expanded, 5, fill=res["fill"], stroke_w=res["stroke_w"]
    )
    rep_inner = br.repaint_transplant(
        bubble_image, inner, expanded, 5, fill=res["fill"], stroke_w=res["stroke_w"]
    )

    gray_outer = cv2.cvtColor(rep_outer, cv2.COLOR_RGB2GRAY)
    gray_inner = cv2.cvtColor(rep_inner, cv2.COLOR_RGB2GRAY)

    dist_exp = cv2.distanceTransform((expanded > 0).astype(np.uint8), cv2.DIST_L2, 5)
    # Stroke is 6-7px wide, centered around depth ~4
    stroke_core = (dist_exp >= 2.0) & (dist_exp <= 6.0)

    # Core must be solid dark ink (<= 40 luma) in both outer- and inner-base cases
    assert gray_outer[stroke_core].mean() <= 40.0
    assert gray_inner[stroke_core].mean() <= 40.0


def test_panel_border_clamping():
    import cv2
    import numpy as np

    # Synthetic mask centered in an image
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 255, -1)

    # Normal expand
    expanded_unclamped = br.expand_mask(mask, 5)
    assert np.where(expanded_unclamped > 0)[1].max() > 80

    # Expand clamped at right border = 80
    borders = {"right": 80}
    expanded_clamped = br.expand_mask(mask, 5, borders=borders)
    assert np.where(expanded_clamped > 0)[1].max() == 80

    # Panel border detection: vertical line at x=80 spanning past the bubble
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[:, 80] = 0  # straight panel frame line
    detected = br.find_panel_borders(gray, mask)
    assert detected.get("right") == 80


def test_expand_mask_smooth_tail_sdf_union():
    import cv2
    import numpy as np

    # Body circle + tail pointing downward
    mask_body = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(mask_body, (60, 50), 25, 255, -1)

    mask_tail = np.zeros((120, 120), dtype=np.uint8)
    pts = np.array([[55, 70], [65, 70], [60, 110]], dtype=np.int32)
    cv2.fillPoly(mask_tail, [pts], 255)

    expanded = br.expand_mask(mask_body, px=5, fragments=mask_tail, sigma=3.8)

    # Body expands: max x increases from 85 to ~90
    assert np.where(expanded > 0)[1].max() >= 89

    # Tail tip at y=110 does NOT expand downward past 110
    ys = np.where(expanded > 0)[0]
    assert ys.max() <= 111


def test_repaint_transplant_frame_preservation_and_interior_wipe():
    import cv2
    import numpy as np

    # Image with panel border line at x=80 and dark text inside bubble
    rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
    rgb[:, 80] = (0, 0, 0)  # Panel border
    rgb[45:55, 45:55] = (10, 10, 10)  # Text inside bubble

    outer = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(outer, (50, 50), 25, 255, -1)
    inner = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(inner, (50, 50), 20, 255, -1)

    borders = {"right": 80}
    expanded = br.expand_mask(outer, 5, borders=borders)

    rep = br.repaint_transplant(
        rgb, outer, expanded, px=5, fill=(255, 255, 255),
        stroke_w=3.0, inner=inner, borders=borders
    )

    # Frame line at x=80 and beyond is 100% bit-exact to original rgb
    assert np.array_equal(rep[:, 80:], rgb[:, 80:])

    # Inside text is completely wiped clean to fill
    assert np.all(rep[45:55, 45:55] == (255, 255, 255))



