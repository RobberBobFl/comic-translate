"""Tests for the text-curvature (arc) tool.

Covers:
  - TextBlockItem.set_curvature clamps to [-100, 100], pads boundingRect()
    only while the arc is active, and never touches the content rect
  - the curved paint bends the ink upward for positive curvature, downward
    for negative, keeps it inside the padded rect, and falls back to the
    straight render in editing mode / vertical layout / at 0
  - the layout is text-on-path: near-zero curvature reproduces the straight
    render pixel-tight (single- and multi-line), spacing is laid out along
    the arc, and |c| = 100 bends the widest line through a quarter circle
  - curvature survives the TextItemProperties round-trip (save/load) and
    defaults to 0 for legacy state that predates the field
  - the toolbar slider applies changes with one undo step per drag and
    reverse-syncs from the selected item
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import math

import pytest
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QTextCursor, QTextDocument, QUndoStack
from PySide6.QtWidgets import (QApplication, QLabel, QSlider, QStyleOptionGraphicsItem,
                               QTextEdit, QWidget)

from app.ui.canvas.image_viewer import ImageViewer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem
from app.controllers.text import TextController


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def viewer_parent(app):
    # The viewer must outlive the test: a transient QWidget() parent would be
    # garbage-collected and take the C++ scene objects with it.
    return QWidget()


def _make_item(text="Hello WORLD!", curvature=0.0, width=300):
    item = TextBlockItem(text=text, font_family="DejaVu Sans", font_size=24,
                         render_color=QColor(0, 0, 0), outline_color=None)
    item.set_font("DejaVu Sans", 24)
    item.set_text(text, width)
    item.set_curvature(curvature)
    return item


def _ink_bbox(item, canvas=700, offset=300):
    """Render *item* via its paint() and return the ink bbox in item-local
    coordinates (the canvas is offset so arcs can extend outside (0, 0))."""
    img = QImage(canvas, canvas, QImage.Format.Format_ARGB32)
    img.fill(QColor(255, 255, 255))
    painter = QPainter(img)
    painter.translate(offset, offset)
    item.paint(painter, QStyleOptionGraphicsItem(), None)
    painter.end()

    box = None
    for y in range(canvas):
        for x in range(canvas):
            if QColor(img.pixel(x, y)).red() < 200:
                if box is None:
                    box = [x, y, x, y]
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x)
                box[3] = max(box[3], y)
    if box is None:
        return None
    return [box[0] - offset, box[1] - offset, box[2] - offset, box[3] - offset]


def _expected_sagitta(content_width, curvature):
    # Same circle the painter uses: the widest line wraps onto a radius of
    # width/theta (its arc length is the line width).
    theta = abs(curvature) / 100.0 * (math.pi / 2.0)
    radius = content_width / theta
    return radius * (1.0 - math.cos(theta / 2.0))


# ── geometry ─────────────────────────────────────────────────────────────────

def test_set_curvature_clamps(app):
    item = _make_item()
    item.set_curvature(500)
    assert item.curvature == 100.0
    item.set_curvature(-500)
    assert item.curvature == -100.0
    item.set_curvature(None)
    assert item.curvature == 0.0


def test_bounding_rect_padded_only_while_curved(app):
    straight = _make_item()
    assert straight.boundingRect() == straight.contentBoundingRect()

    curved = _make_item(curvature=50)
    padded = curved.boundingRect()
    content = curved.contentBoundingRect()
    assert padded.width() > content.width()
    assert padded.left() < content.left()
    assert padded.top() < content.top()
    # Pad is sagitta + line height: at least the sagitta on every side.
    sagitta = _expected_sagitta(content.width(), 50)
    assert -padded.left() >= sagitta
    assert -padded.top() >= sagitta


def test_content_rect_stable_across_curvature(app):
    straight = _make_item()
    curved = _make_item(curvature=70)
    assert (curved.contentBoundingRect().width()
            == pytest.approx(straight.contentBoundingRect().width()))
    assert (curved.contentBoundingRect().height()
            == pytest.approx(straight.contentBoundingRect().height()))


def test_editing_mode_drops_and_restores_pad(app):
    item = _make_item(curvature=50)
    padded_width = item.boundingRect().width()

    item.enter_editing_mode()
    assert not item._curvature_active()
    assert item.boundingRect() == item.contentBoundingRect()

    item.exit_editing_mode()
    assert item._curvature_active()
    assert item.boundingRect().width() == pytest.approx(padded_width)


def test_vertical_layout_disables_arc(app):
    item = _make_item(curvature=50)
    item.set_vertical(True)
    assert not item._curvature_active()
    assert item.boundingRect() == item.contentBoundingRect()
    item.set_vertical(False)
    assert item._curvature_active()


# ── rendering ────────────────────────────────────────────────────────────────

def test_arch_and_bowl_direction(app):
    straight = _make_item(curvature=0)
    arch = _make_item(curvature=50)
    bowl = _make_item(curvature=-50)

    ink0 = _ink_bbox(straight)
    ink_up = _ink_bbox(arch)
    ink_down = _ink_bbox(bowl)

    assert ink0 and ink_up and ink_down
    # Midpoint anchoring: the line's midpoint stays on its baseline, so the
    # arch swings its ends below the straight ink and the bowl lifts its
    # ends above it, while the opposite edge stays put.
    assert ink_up[3] > ink0[3]      # arch ink reaches lower than straight
    assert ink_up[1] >= ink0[1] - 2  # arch does not rise above straight top
    assert ink_down[1] < ink0[1]     # bowl ink reaches higher than straight
    assert ink_down[3] <= ink0[3] + 2  # bowl does not dip below straight bottom


def test_rows_never_collapse_on_strong_arc(app):
    # Multi-line rows fan apart toward the ends instead of colliding: the
    # ink box must never compress below the straight block height, and the
    # anchored edge (top for the arch, bottom for the bowl) must stay put.
    text = "WHO COULD HAVE GUESSED\nTHAT MR. TINY WAS\nA GENIUS OF ADS?"
    straight = _make_item(text=text, curvature=0)
    arch = _make_item(text=text, curvature=100)
    bowl = _make_item(text=text, curvature=-100)

    ink0, ink_up, ink_down = _ink_bbox(straight), _ink_bbox(arch), _ink_bbox(bowl)
    assert ink0 and ink_up and ink_down
    height0 = ink0[3] - ink0[1]
    assert ink_up[3] - ink_up[1] >= height0 - 2
    assert ink_down[3] - ink_down[1] >= height0 - 2
    assert ink_up[1] >= ink0[1] - 2   # arch: nothing rises above straight top
    assert ink_down[3] <= ink0[3] + 2  # bowl: nothing dips below straight bottom


def test_arc_theta_quarter_circle_at_full_slider(app):
    item = _make_item()
    assert item._arc_theta(100.0) == pytest.approx(math.pi / 2)
    assert item._arc_theta(-100.0) == pytest.approx(math.pi / 2)
    assert item._arc_theta(50.0) == pytest.approx(math.pi / 4)
    assert item._arc_theta(0.0) == pytest.approx(0.0)


def test_tiny_curvature_reproduces_straight_render(app):
    # The arc-length layout must converge to the straight layout as the
    # curvature goes to 0: near-zero curvature should sit on top of the
    # native render, which also pins the per-line baseline/radius math.
    straight = _make_item(curvature=0)
    curved = _make_item(curvature=1)
    ink0, ink1 = _ink_bbox(straight), _ink_bbox(curved)
    assert ink0 and ink1
    assert all(abs(a - b) <= 3 for a, b in zip(ink0, ink1))


def test_tiny_curvature_reproduces_straight_render_multi_line(app):
    text = "Line one here\nLine two here"
    straight = _make_item(text=text, curvature=0)
    curved = _make_item(text=text, curvature=1)
    ink0, ink1 = _ink_bbox(straight), _ink_bbox(curved)
    assert ink0 and ink1
    assert all(abs(a - b) <= 3 for a, b in zip(ink0, ink1))


def test_curved_ink_stays_inside_padded_rect(app):
    for curvature in (-100, -50, -1, 1, 50, 100):
        item = _make_item(curvature=curvature)
        ink = _ink_bbox(item)
        assert ink is not None, f"no ink at curvature={curvature}"
        rect = item.boundingRect()
        assert (rect.left() <= ink[0] and rect.top() <= ink[1]
                and rect.right() >= ink[2] and rect.bottom() >= ink[3]), \
            f"curvature={curvature}: ink {ink} outside {rect}"


def test_resetting_curvature_to_zero_renders_straight(app):
    item = _make_item(curvature=60)
    item.set_curvature(0)
    assert not item._curvature_active()
    assert item.boundingRect() == item.contentBoundingRect()

    straight = _make_item(curvature=0)
    curved_ink = _ink_bbox(item)
    straight_ink = _ink_bbox(straight)
    assert curved_ink == straight_ink


def test_multi_line_arc_renders(app):
    item = _make_item(text="Line one here\nLine two here", curvature=80)
    ink = _ink_bbox(item)
    assert ink is not None
    assert item.boundingRect().contains(QtCore.QRectF(ink[0], ink[1],
                                                      ink[2] - ink[0],
                                                      ink[3] - ink[1]))


# ── persistence ──────────────────────────────────────────────────────────────

def test_properties_roundtrip_preserves_curvature(app):
    item = _make_item(curvature=45)
    props = TextItemProperties.from_text_item(item)
    assert props.curvature == 45.0
    # Saved width/height must be the content rect, not the padded one, or the
    # arc slack would compound on every reload.
    assert props.width == pytest.approx(item.contentBoundingRect().width())
    assert props.width < item.boundingRect().width()

    restored = TextItemProperties.from_dict(props.to_dict())
    assert restored.curvature == 45.0


def test_legacy_state_defaults_to_zero(app):
    props = TextItemProperties.from_dict({"text": "old"})
    assert props.curvature == 0.0
    assert props.to_dict()["curvature"] == 0.0


def test_add_text_item_applies_curvature(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(300, 300, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    props = TextItemProperties(text="curved", font_family="DejaVu Sans",
                               font_size=24, text_color=QColor(0, 0, 0),
                               width=200, position=(20, 20))
    props.curvature = 45.0
    item = viewer.add_text_item(props)

    assert item.curvature == 45.0
    assert item._curvature_active()
    assert item.boundingRect().width() > item.contentBoundingRect().width()


# ── controller / slider ──────────────────────────────────────────────────────

class _FakeMain:
    """The minimum MainWindow surface the curvature handlers touch."""

    def __init__(self, viewer):
        self.image_viewer = viewer
        self.webtoon_mode = False
        self.curr_tblock_item = None
        self.curr_tblock = None
        self.t_text_edit = QTextEdit()
        self.curvature_slider = QSlider(Qt.Horizontal)
        self.curvature_slider.setRange(-100, 100)
        self.curvature_value_label = QLabel("0")
        self._stack = QUndoStack()
        self.undo_group = type("G", (), {})()
        self.undo_group.activeStack = lambda: self._stack


def _make_controller(main):
    controller = TextController.__new__(TextController)
    controller.main = main
    controller._suspend_text_command = False
    controller._curvature_drag = None
    controller.widgets_to_block = []
    controller._refresh_toolbar_for_item = lambda item: None
    # Same wiring as controller.py.
    main.curvature_slider.sliderPressed.connect(controller.on_curvature_slider_pressed)
    main.curvature_slider.valueChanged.connect(controller.on_curvature_change)
    main.curvature_slider.sliderReleased.connect(controller.on_curvature_slider_released)
    return controller


# ── selection box / page constraint ──────────────────────────────────────────

def test_selection_rect_is_tight_under_curvature(app):
    # The dashed border, resize handles and page constraint must follow the
    # plain text box no matter the curvature; only Qt's repaint rect grows.
    item = _make_item(curvature=50)
    assert item.selectionRect() == item.contentBoundingRect()
    assert item.selectionRect().width() < item.boundingRect().width()

    item.set_curvature(100)
    assert item.selectionRect() == item.contentBoundingRect()


def test_move_reaches_page_edge_while_curved(app, viewer_parent):
    viewer, item = _viewer_with_item(viewer_parent)
    scene = viewer._scene
    scene.setSceneRect(QtCore.QRectF(0, 0, 800, 800))
    item.set_curvature(50)
    item.setPos(450, 100)
    content_right = item.pos().x() + item.contentBoundingRect().right()
    assert content_right < 800

    # Drag far past the right page edge: the text box must stop exactly at
    # the edge (the padded repaint rect would have stopped it pad-short).
    item.move_item(QtCore.QPointF(1000, 0), QtCore.QPointF(0, 0))
    content_right = item.pos().x() + item.contentBoundingRect().right()
    assert content_right == pytest.approx(800, abs=1.0)
    assert item.pos().x() + item.boundingRect().right() > 800  # slack overhangs


def test_selected_curved_paint_draws_dashed_border(app, viewer_parent):
    # QGraphicsTextItem paints Qt's dashed selection border around
    # boundingRect(); the curved path skips super().paint(), so it must draw
    # the same border around the tight selection rect itself.
    viewer, item = _viewer_with_item(viewer_parent)
    item.set_curvature(50)
    item.setSelected(True)

    off, canvas = 200, 700
    img = QImage(canvas, canvas, QImage.Format.Format_ARGB32)
    img.fill(QColor(255, 255, 255))
    painter = QPainter(img)
    painter.translate(off, off)
    item.paint(painter, QStyleOptionGraphicsItem(), None)
    painter.end()

    sel = item.selectionRect()
    pad = item.boundingRect().height() - sel.height()  # > 0 while curved

    def dashed_along(row_x, y):
        # Qt's border is 50%-alpha black: mid-gray on white. Text ink is
        # much darker; count only mid-gray pixels.
        return sum(
            1 for x in range(int(row_x[0]), int(row_x[1]))
            if 100 < QColor(img.pixel(x + off, y + off)).red() < 210
        )

    assert dashed_along((sel.left(), sel.right()), int(sel.top())) > 0
    assert dashed_along((sel.left(), sel.right()), int(sel.bottom())) > 0
    # A padded-rect border would run pad px above the selection rect.
    assert dashed_along((sel.left(), sel.right()), int(sel.top() - min(pad, 20) - 2)) == 0


def _viewer_with_item(viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    props = TextItemProperties(text="Hello WORLD!", font_family="DejaVu Sans",
                               font_size=24, text_color=QColor(0, 0, 0),
                               width=300, position=(10, 10))
    item = viewer.add_text_item(props)
    return viewer, item


def test_slider_change_applies_and_pushes_one_undo(app, viewer_parent):
    viewer, item = _viewer_with_item(viewer_parent)
    main = _FakeMain(viewer)
    controller = _make_controller(main)
    main.curr_tblock_item = item

    main.curvature_slider.setValue(40)  # programmatic/keyboard path

    assert item.curvature == 40.0
    assert main.curvature_value_label.text() == "40"
    assert main._stack.count() == 1

    main._stack.undo()
    assert item.curvature == 0.0
    main._stack.redo()
    assert item.curvature == 40.0


def test_slider_drag_is_a_single_undo_step(app, viewer_parent):
    viewer, item = _viewer_with_item(viewer_parent)
    main = _FakeMain(viewer)
    controller = _make_controller(main)
    main.curr_tblock_item = item

    main.curvature_slider.sliderPressed.emit()
    for value in (10, 25, 55):  # every pixel of the drag
        main.curvature_slider.setValue(value)
        assert item.curvature == float(value)
        assert main._stack.count() == 0  # nothing pushed mid-drag
    main.curvature_slider.sliderReleased.emit()

    assert main._stack.count() == 1
    assert main.curvature_value_label.text() == "55"

    main._stack.undo()
    assert item.curvature == 0.0
    assert item.boundingRect() == item.contentBoundingRect()
    main._stack.redo()
    assert item.curvature == 55.0


def test_slider_drag_without_change_pushes_nothing(app, viewer_parent):
    viewer, item = _viewer_with_item(viewer_parent)
    main = _FakeMain(viewer)
    controller = _make_controller(main)
    main.curr_tblock_item = item

    main.curvature_slider.sliderPressed.emit()
    main.curvature_slider.sliderReleased.emit()  # press without movement

    assert main._stack.count() == 0
    assert item.curvature == 0.0


def test_refresh_curvature_syncs_slider(app, viewer_parent):
    viewer, item = _viewer_with_item(viewer_parent)
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    item.set_curvature(-30)
    controller._refresh_curvature(item=item)
    assert main.curvature_slider.value() == -30
    assert main.curvature_value_label.text() == "-30"

    # Syncing must not emit valueChanged (that would push an undo command).
    controller._refresh_curvature(props={"curvature": 70})
    assert main.curvature_slider.value() == 70
    assert main._stack.count() == 0


def test_text_format_command_restores_geometry(app, viewer_parent):
    viewer, item = _viewer_with_item(viewer_parent)
    main = _FakeMain(viewer)
    controller = _make_controller(main)
    main.curr_tblock_item = item

    main.curvature_slider.setValue(60)
    straight_width = item.contentBoundingRect().width()
    curved_width = item.boundingRect().width()
    assert curved_width > straight_width

    main._stack.undo()
    assert item.curvature == 0.0
    assert item.boundingRect().width() == pytest.approx(straight_width)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
