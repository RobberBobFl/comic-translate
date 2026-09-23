"""Tests for the rendering controls: fixed font size and horizontal padding.

Covers:
  - the "Fixed" checkbox pins render_settings() to a single font size so every
    block on the page renders at the toolbar size instead of auto-fitting
  - the horizontal-margin dropdown shrinks the wrap width and keeps the text
    off the bubble edges
  - h_margin persists through TextItemProperties and defaults to 0 for legacy
    project state that predates the field
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QPushButton, QWidget)

from app.ui.canvas.text.text_item_properties import (
    TextItemProperties,
    get_h_margins,
    set_h_margins,
)
from app.ui.canvas.image_viewer import ImageViewer
from app.controllers.text import TextController
from modules.utils.textblock import TextBlock


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def viewer_parent(app):
    # ImageViewer must outlive each test: a transient QWidget() parent would be
    # garbage-collected and take the C++ scene objects with it.
    return QWidget()


class _FakeSettingsPage:
    def __init__(self):
        self.ui = type("ui", (), {})()
        self.ui.min_font_spinbox = QDoubleSpinBox()
        self.ui.min_font_spinbox.setValue(9)
        self.ui.max_font_spinbox = QDoubleSpinBox()
        self.ui.max_font_spinbox.setValue(40)
        self.ui.uppercase_checkbox = QCheckBox()


class _FakeMain:
    """Stand-in exposing every attribute render_settings() / on_blk_rendered()
    touch."""

    def __init__(self, viewer):
        self.settings_page = _FakeSettingsPage()
        self.fixed_font_size_checkbox = QCheckBox()
        self.font_size_dropdown = QComboBox()
        self.font_size_dropdown.addItems(["12", "14", "24"])
        self.font_size_dropdown.setEditable(True)
        self.alignment_tool_group = type(
            "A", (), {"get_dayu_checked": lambda self: 1}
        )()
        self.font_dropdown = QComboBox()
        self.font_dropdown.addItem("Sans Serif")
        self.block_font_color_button = QPushButton()
        self.block_font_color_button.setProperty("selected_color", "#000000")
        self.outline_checkbox = QCheckBox()
        self.outline_font_color_button = QPushButton()
        self.outline_font_color_button.setProperty("selected_color", "#ffffff")
        self.outline_width_dropdown = QComboBox()
        self.outline_width_dropdown.addItem("1.0")
        self.bold_button = QCheckBox()
        self.italic_button = QCheckBox()
        self.underline_button = QCheckBox()
        self.line_spacing_dropdown = QComboBox()
        self.line_spacing_dropdown.addItems(["1.0", "1.2"])
        self.line_spacing_dropdown.setEditable(True)
        self.h_margin_dropdown = QComboBox()
        self.h_margin_dropdown.addItems(["0", "4", "6"])
        self.h_margin_dropdown.setEditable(True)
        self.t_combo = QComboBox()
        self.t_combo.addItem("English")
        self.lang_mapping = {"English": "English"}
        self.image_viewer = viewer
        self.webtoon_mode = False
        self.blk_list = []
        self.curr_img_idx = 0
        self.image_files = ["x.png"]
        self.undo_group = type("G", (), {"activeStack": lambda self: None})()
        self.button_to_alignment = {1: Qt.AlignmentFlag.AlignCenter}

    def push_command(self, command):
        pass

    def mark_project_dirty(self):
        pass


def _make_controller(main):
    controller = TextController.__new__(TextController)
    controller.main = main
    controller._suspend_text_command = False
    return controller


# ── fixed font size ──────────────────────────────────────────────────────────

def test_render_settings_normal_uses_spinboxes(app):
    main = _FakeMain(None)
    controller = _make_controller(main)
    settings = controller.render_settings()
    assert settings.min_font_size == 9
    assert settings.max_font_size == 40


def test_render_settings_fixed_pins_to_toolbar_size(app):
    main = _FakeMain(None)
    main.fixed_font_size_checkbox.setChecked(True)
    main.font_size_dropdown.setCurrentText("24")
    controller = _make_controller(main)
    settings = controller.render_settings()
    assert settings.min_font_size == settings.max_font_size == 24


def test_render_settings_fixed_clamps_invalid_input(app):
    main = _FakeMain(None)
    main.fixed_font_size_checkbox.setChecked(True)
    # Non-numeric input must not crash: render_settings() falls back to the
    # settings-page spinbox values.
    main.font_size_dropdown.setEditText("abc")
    controller = _make_controller(main)
    settings = controller.render_settings()
    assert settings.min_font_size == 9
    assert settings.max_font_size == 40


def test_render_settings_fixed_clamps_below_one(app):
    main = _FakeMain(None)
    main.fixed_font_size_checkbox.setChecked(True)
    main.font_size_dropdown.setEditText("0")
    controller = _make_controller(main)
    settings = controller.render_settings()
    # max(1.0, ...) keeps the size sane; 0 would disable every fit loop.
    assert settings.min_font_size == settings.max_font_size == 1


def test_render_settings_fixed_clamps_negative(app):
    main = _FakeMain(None)
    main.fixed_font_size_checkbox.setChecked(True)
    main.font_size_dropdown.setEditText("-5")
    controller = _make_controller(main)
    settings = controller.render_settings()
    assert settings.min_font_size == settings.max_font_size == 1


def test_on_blk_rendered_honours_fixed_size(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    main.fixed_font_size_checkbox.setChecked(True)
    main.font_size_dropdown.setCurrentText("24")
    controller = _make_controller(main)

    blk = TextBlock()
    blk.xyxy = [20, 20, 80, 80]
    blk.translation = "way too long text for this tiny bubble"
    controller.on_blk_rendered("way too long text for this tiny bubble", 24, blk, "x.png")

    item = viewer.text_items[-1]
    assert item.font_size == 24
    # Overflow is allowed: fixed mode deliberately does not shrink.
    assert item.document().size().height() > 60.0


# ── horizontal margin ────────────────────────────────────────────────────────

def test_set_and_get_h_margins_roundtrip(app):
    from app.ui.canvas.text_item import TextBlockItem

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.setTextWidth(200)
    set_h_margins(item.document(), 6.0)
    assert get_h_margins(item.document()) == 6.0


def test_h_margins_shrink_the_wrap_area(app):
    from app.ui.canvas.text_item import TextBlockItem

    item = TextBlockItem("word " * 12, font_family="Sans Serif", font_size=20)
    item.setTextWidth(200)
    height_before = item.document().size().height()
    set_h_margins(item.document(), 10.0)
    item.document().setTextWidth(200)
    height_after = item.document().size().height()
    # Padding narrows the writable area, so the text wraps onto more lines.
    assert height_after > height_before


def test_on_blk_rendered_applies_h_margin(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)

    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    main = _FakeMain(viewer)
    main.h_margin_dropdown.setCurrentText("6")
    controller = _make_controller(main)

    blk = TextBlock()
    blk.xyxy = [10, 10, 170, 170]
    blk.translation = "hello world this is a test"
    controller.on_blk_rendered("hello world this is a test", 20, blk, "x.png")

    item = viewer.text_items[-1]
    assert get_h_margins(item.document()) == 6.0
    # Bubble width 160 minus 2 * 6 padding.
    assert abs(item.boundingRect().width() - 148.0) < 1.0


def test_on_blk_rendered_without_margin_keeps_full_width(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)

    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    main = _FakeMain(viewer)
    controller = _make_controller(main)

    blk = TextBlock()
    blk.xyxy = [10, 10, 170, 170]
    blk.translation = "hello world this is a test"
    controller.on_blk_rendered("hello world this is a test", 20, blk, "x.png")

    item = viewer.text_items[-1]
    assert get_h_margins(item.document()) == 0.0
    assert abs(item.boundingRect().width() - 160.0) < 1.0


# ── persistence ──────────────────────────────────────────────────────────────

def test_h_margin_roundtrips_through_properties(app):
    from app.ui.canvas.text_item import TextBlockItem

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.setTextWidth(200)
    set_h_margins(item.document(), 6.0)

    props = TextItemProperties.from_text_item(item)
    assert props.h_margin == 6.0
    assert props.to_dict()["h_margin"] == 6.0

    restored = TextItemProperties.from_dict(props.to_dict())
    assert restored.h_margin == 6.0


def test_h_margin_defaults_to_zero_for_legacy_state(app):
    props = TextItemProperties.from_dict({"font_size": 20})
    assert props.h_margin == 0.0


# ── line spacing below 1.0 ───────────────────────────────────────────────────

def test_line_spacing_below_one_is_honoured(app):
    from app.ui.canvas.text_item import TextBlockItem

    item = TextBlockItem("line one\nline two\nline three", font_family="Sans Serif", font_size=20)
    item.set_line_spacing(0.8)
    cursor = QTextCursor(item.document())
    cursor.select(QTextCursor.SelectionType.Document)
    assert cursor.blockFormat().lineHeight() == pytest.approx(80.0)


# ── multi-select: every formatting control hits all selected blocks ─────────

def _block_line_height(item):
    cursor = QTextCursor(item.document())
    cursor.select(QTextCursor.SelectionType.Document)
    return cursor.blockFormat().lineHeight()


def test_multi_select_line_spacing_hits_all_blocks(app, viewer_parent):
    """Regression: with several blocks selected, line spacing only changed on
    the last-clicked block because the handler used curr_tblock_item directly
    instead of _apply_format_to_selected."""
    viewer = ImageViewer(viewer_parent)
    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    main = _FakeMain(viewer)
    controller = _make_controller(main)

    for x, text in ((10, "aaa bbb ccc"), (60, "ddd eee fff"), (110, "ggg hhh iii")):
        blk = TextBlock()
        blk.xyxy = [x, 10, x + 50, 60]
        blk.translation = text
        controller.on_blk_rendered(text, 20, blk, "x.png")

    items = list(viewer.text_items)
    assert len(items) == 3
    for item in items:
        item.selected = True
    main.curr_tblock_item = items[0]

    main.line_spacing_dropdown.setEditText("0.8")
    controller.on_line_spacing_change("0.8")

    heights = [_block_line_height(item) for item in items]
    assert heights == [80.0, 80.0, 80.0]


def test_multi_select_h_margin_hits_all_blocks(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    main = _FakeMain(viewer)
    controller = _make_controller(main)

    for x, text in ((10, "aaa bbb ccc"), (60, "ddd eee fff"), (110, "ggg hhh iii")):
        blk = TextBlock()
        blk.xyxy = [x, 10, x + 50, 60]
        blk.translation = text
        controller.on_blk_rendered(text, 20, blk, "x.png")

    items = list(viewer.text_items)
    for item in items:
        item.selected = True
    main.curr_tblock_item = items[0]

    main.h_margin_dropdown.setEditText("6")
    controller.on_h_margin_change("6")

    margins = [get_h_margins(item.document()) for item in items]
    assert margins == [6.0, 6.0, 6.0]


def test_multi_select_font_size_hits_all_blocks(app, viewer_parent):
    """Font size already worked via _apply_format_to_selected; kept as a guard
    so the three controls stay consistent."""
    viewer = ImageViewer(viewer_parent)
    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    main = _FakeMain(viewer)
    controller = _make_controller(main)

    for x, text in ((10, "aaa bbb ccc"), (60, "ddd eee fff"), (110, "ggg hhh iii")):
        blk = TextBlock()
        blk.xyxy = [x, 10, x + 50, 60]
        blk.translation = text
        controller.on_blk_rendered(text, 20, blk, "x.png")

    items = list(viewer.text_items)
    for item in items:
        item.selected = True
    main.curr_tblock_item = items[0]

    main.font_size_dropdown.setEditText("24")
    controller.on_font_size_change("24")

    assert [item.font_size for item in items] == [24, 24, 24]
