"""Tests for the rendering controls: fixed font size and text spacing.

Covers:
  - the "Fixed" checkbox pins render_settings() to a single font size so every
    block on the page renders at the toolbar size instead of auto-fitting
  - the letter/word-spacing dropdowns change the document's font metrics so the
    text wraps wider or tighter, and the auto-fit accounts for the spacing
  - spacing persists through TextItemProperties and defaults to 0 for legacy
    project state that predates the fields
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import (QImage, QPixmap, QTextCharFormat, QTextCursor,
                           QTextDocument, QFont, QUndoStack)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QLabel, QPushButton, QSlider, QWidget)

from app.ui.canvas.text.text_item_properties import TextItemProperties
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

    def _set_spinboxes(self, min_value, max_value):
        self.ui.min_font_spinbox.setValue(min_value)
        self.ui.max_font_spinbox.setValue(max_value)


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
        self.letter_spacing_dropdown = QComboBox()
        self.letter_spacing_dropdown.addItems(["-2", "-1", "0", "1", "2", "3", "4", "5", "6"])
        self.letter_spacing_dropdown.setCurrentText("0")
        self.letter_spacing_dropdown.setEditable(True)
        self.word_spacing_dropdown = QComboBox()
        self.word_spacing_dropdown.addItems(["-4", "-2", "0", "2", "4", "6", "8", "10", "12"])
        self.word_spacing_dropdown.setCurrentText("0")
        self.word_spacing_dropdown.setEditable(True)
        self.curvature_slider = QSlider(Qt.Horizontal)
        self.curvature_slider.setRange(-100, 100)
        self.curvature_value_label = QLabel("0")
        self.t_combo = QComboBox()
        self.t_combo.addItem("English")
        self.lang_mapping = {"English": "English"}
        # The spacing handlers route through _apply_format_to_selected, which
        # transfers the translation panel's selection to the item; an empty
        # selection means "whole block", which is what these tests exercise.
        self.t_text_edit = type("P", (), {"textCursor": lambda self: QTextCursor(QTextDocument())})()
        self.image_viewer = viewer
        self.webtoon_mode = False
        self.blk_list = []
        self.curr_img_idx = 0
        self.image_files = ["x.png"]
        self.undo_group = type("G", (), {"activeStack": lambda self: self._stack})()
        self.undo_group._stack = QUndoStack()
        self.button_to_alignment = {1: Qt.AlignmentFlag.AlignCenter}

    def push_command(self, command):
        pass

    def mark_project_dirty(self):
        pass


def _make_controller(main):
    controller = TextController.__new__(TextController)
    controller.main = main
    controller._suspend_text_command = False
    controller._curvature_drag = None
    # undo/redo of a format command refreshes the toolbar; set_values_for_blk_item
    # blocks these widgets while writing back the item's values.
    controller.widgets_to_block = []
    # The toolbar refresh needs the full widget tree (alignment button group
    # etc.), which _FakeMain does not model; the tests assert on item geometry
    # directly, so skip it.
    controller._refresh_toolbar_for_item = lambda item: None
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


# ── fixed font size render mode ──────────────────────────────────────────────

def test_fixed_mode_pins_every_block_to_one_size(app):
    from modules.rendering.render import pyside_word_wrap
    from PySide6.QtCore import Qt

    text = "word " * 8
    sizes = []
    for bw, bh in ((120, 60), (400, 80), (80, 80), (300, 300)):
        _, size, _, _ = pyside_word_wrap(
            text, "Sans Serif", bw, bh, 1.2, 1.0, False, False, False,
            Qt.AlignmentFlag.AlignCenter, Qt.LayoutDirection.LeftToRight,
            init_font_size=24, min_font_size=24, vertical=False,
            no_space_language=False, return_metrics=True,
        )
        sizes.append(size)
    assert sizes == [24, 24, 24, 24]


def test_fixed_mode_off_keeps_per_block_autofit(app):
    from modules.rendering.render import pyside_word_wrap
    from PySide6.QtCore import Qt

    text = "word " * 8
    sizes = []
    for bw, bh in ((120, 60), (400, 80), (300, 300)):
        _, size, _, _ = pyside_word_wrap(
            text, "Sans Serif", bw, bh, 1.2, 1.0, False, False, False,
            Qt.AlignmentFlag.AlignCenter, Qt.LayoutDirection.LeftToRight,
            init_font_size=40, min_font_size=4, vertical=False,
            no_space_language=False, return_metrics=True,
        )
        sizes.append(size)
    # A tiny box shrinks, a big box grows: the legacy auto-fit is intact.
    assert sizes[0] < sizes[1] <= sizes[2]


def test_render_settings_fixed_pins_min_equal_to_max(app):
    main = _FakeMain(None)
    controller = _make_controller(main)
    main.settings_page._set_spinboxes(4, 40)

    main.fixed_font_size_checkbox.setChecked(False)
    settings = controller.render_settings()
    assert settings.min_font_size == 4 and settings.max_font_size == 40

    main.fixed_font_size_checkbox.setChecked(True)
    main.font_size_dropdown.setCurrentText("24")
    settings = controller.render_settings()
    assert settings.min_font_size == settings.max_font_size == 24


# ── letter / word spacing ─────────────────────────────────────────────────────

def test_render_settings_reads_spacing_dropdowns(app):
    main = _FakeMain(None)
    controller = _make_controller(main)

    main.letter_spacing_dropdown.setCurrentText("3")
    main.word_spacing_dropdown.setCurrentText("6")
    settings = controller.render_settings()
    assert settings.letter_spacing == 3.0
    assert settings.word_spacing == 6.0


def test_render_settings_spacing_falls_back_on_partial_input(app):
    main = _FakeMain(None)
    controller = _make_controller(main)

    main.letter_spacing_dropdown.setEditText("-")
    main.word_spacing_dropdown.setEditText("abc")
    settings = controller.render_settings()
    assert settings.letter_spacing == 0.0
    assert settings.word_spacing == 0.0


def test_apply_spacing_changes_document_metrics(app):
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.setPlainText(text)
    item.setTextWidth(300)
    height_before = item.document().size().height()

    item.letter_spacing = 6.0
    item.word_spacing = 12.0
    item.apply_spacing()

    # Wider text wraps onto more lines, so the document grows.
    assert item.document().size().height() > height_before
    font = item.document().defaultFont()
    assert font.letterSpacing() == 6.0
    assert font.wordSpacing() == 12.0


def test_apply_spacing_negative_tightens(app):
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.setPlainText(text)
    item.setTextWidth(300)
    height_before = item.document().size().height()

    item.letter_spacing = -1.0
    item.apply_spacing()
    assert item.document().size().height() <= height_before


def test_apply_spacing_zero_is_a_noop_on_fresh_document(app):
    from app.ui.canvas.text_item import TextBlockItem

    item = TextBlockItem("word word", font_family="Sans Serif", font_size=20)
    item.setPlainText("word word")
    item.setTextWidth(300)
    height_before = item.document().size().height()
    item.apply_spacing()  # defaults are 0; must not raise or change metrics
    assert item.document().defaultFont().letterSpacing() == 0.0
    assert item.document().size().height() == height_before


def test_apply_spacing_zero_clears_previous_spacing(app):
    """Regression: returning to 0 must restore the original metrics.

    apply_spacing() used to bail out entirely when both values were 0, leaving
    the last nonzero spacing baked into the document font.
    """
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.setPlainText(text)
    item.setTextWidth(300)
    base_ideal = item.document().idealWidth()

    item.letter_spacing = 6.0
    item.word_spacing = 12.0
    item.apply_spacing()
    spaced_ideal = item.document().idealWidth()
    assert spaced_ideal != base_ideal

    # Back to default: the document must measure exactly as it did before any
    # spacing was applied, not stay at the spaced width.
    item.letter_spacing = 0.0
    item.word_spacing = 0.0
    item.apply_spacing()
    assert item.document().idealWidth() == base_ideal
    assert item.document().defaultFont().letterSpacing() == 0.0
    assert item.document().defaultFont().wordSpacing() == 0.0


def test_apply_spacing_repeated_zero_is_stable(app):
    """Toggling spacing on and off repeatedly must always land on the same
    metrics (the fix point of the re-fit loop aside, the *document* itself must
    be deterministic)."""
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.setPlainText(text)
    item.setTextWidth(300)
    base_ideal = item.document().idealWidth()

    for _ in range(3):
        item.letter_spacing = 5.0
        item.apply_spacing()
        assert item.document().idealWidth() != base_ideal
        item.letter_spacing = 0.0
        item.apply_spacing()
        assert item.document().idealWidth() == base_ideal


def test_apply_spacing_after_set_font_char_formats(app):
    """Regression: set_font() merges a full QFont into the whole document, so
    every fragment carries its own font. Qt's layout then prefers the fragment
    font over the document's default font, and spacing applied only to the
    default font was silently ignored."""
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.set_plain_text(text)          # goes through set_font -> char formats
    item.setTextWidth(300)
    cursor = item.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    assert cursor.charFormat().fontFamilies()[0] == "Sans Serif"

    base_ideal = item.document().idealWidth()
    item.letter_spacing = 6.0
    item.word_spacing = 12.0
    item.apply_spacing()

    assert item.document().idealWidth() != base_ideal
    cursor.select(QTextCursor.SelectionType.Document)
    # The merged format keeps the family and gains the spacing.
    assert cursor.charFormat().fontFamilies()[0] == "Sans Serif"


def test_apply_spacing_preserves_per_span_formatting(app):
    """Bold/italic/colour spans must survive applying spacing; only the
    letter/word spacing changes."""
    from app.ui.canvas.text_item import TextBlockItem
    from PySide6.QtGui import QColor

    text = "bold part, then a red part"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.set_plain_text(text)
    item.setTextWidth(300)

    cursor = item.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(9, QTextCursor.MoveMode.KeepAnchor)   # "bold part"
    bold_fmt = QTextCharFormat()
    bold_fmt.setFontWeight(QFont.Bold)
    cursor.mergeCharFormat(bold_fmt)
    cursor.setPosition(18)
    cursor.setPosition(21, QTextCursor.MoveMode.KeepAnchor)  # "red"
    red_fmt = QTextCharFormat()
    red_fmt.setForeground(QColor("red"))
    cursor.mergeCharFormat(red_fmt)

    def span_report():
        out = {}
        for pos, tag in ((4, "bold"), (14, "plain"), (19, "red")):
            c = item.textCursor()
            c.setPosition(pos)
            f = c.charFormat()
            out[tag] = (f.fontWeight() == QFont.Bold,
                        f.foreground().color().name(),
                        f.fontFamilies()[0],
                        f.fontPointSize())
        return out

    before = span_report()
    item.letter_spacing = 4.0
    item.word_spacing = 8.0
    item.apply_spacing()
    after = span_report()

    assert before == after, "per-span formatting changed"
    # And the spacing really did land.
    assert item.document().defaultFont().letterSpacing() == 4.0


def test_apply_spacing_is_idempotent(app):
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.set_plain_text(text)
    item.setTextWidth(300)
    item.letter_spacing = 4.0
    item.word_spacing = 8.0
    item.apply_spacing()

    first = item.document().idealWidth()
    item.apply_spacing()
    second = item.document().idealWidth()
    assert first == second


def test_spacing_survives_set_font_size(app):
    from app.ui.canvas.text_item import TextBlockItem

    text = "word word word word word"
    item = TextBlockItem(text, font_family="Sans Serif", font_size=20)
    item.setPlainText(text)
    item.setTextWidth(300)
    item.letter_spacing = 6.0
    item.word_spacing = 12.0
    item.apply_spacing()

    item.set_font_size(14)

    font = item.document().defaultFont()
    assert font.letterSpacing() == 6.0
    assert font.wordSpacing() == 12.0


def test_spacing_roundtrips_through_properties(app):
    from app.ui.canvas.text_item import TextBlockItem

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.setTextWidth(200)
    item.letter_spacing = 2.5
    item.word_spacing = 4.0

    props = TextItemProperties.from_text_item(item)
    assert props.letter_spacing == 2.5
    assert props.word_spacing == 4.0

    d = props.to_dict()
    assert d["letter_spacing"] == 2.5
    assert d["word_spacing"] == 4.0

    restored = TextItemProperties.from_dict(d)
    assert restored.letter_spacing == 2.5
    assert restored.word_spacing == 4.0


def test_spacing_defaults_to_zero_for_legacy_state(app):
    props = TextItemProperties.from_dict({"font_size": 20})
    assert props.letter_spacing == 0.0
    assert props.word_spacing == 0.0


def test_pyside_word_wrap_accounts_for_spacing(app):
    """The measuring font must carry the spacing, otherwise the auto-fit picks
    a size that overflows once the spacing is applied."""
    from modules.rendering.render import pyside_word_wrap
    from PySide6.QtCore import Qt

    text = "word word word word word"
    # Same box and target size, with and without spacing.
    _, size_plain, _, h_plain = pyside_word_wrap(
        text, "Sans Serif", 300, 200, 1.2, 1.0, False, False, False,
        Qt.AlignmentFlag.AlignCenter, Qt.LayoutDirection.LeftToRight,
        init_font_size=20, min_font_size=8, vertical=False,
        no_space_language=False, return_metrics=True,
    )
    _, size_spaced, _, h_spaced = pyside_word_wrap(
        text, "Sans Serif", 300, 200, 1.2, 1.0, False, False, False,
        Qt.AlignmentFlag.AlignCenter, Qt.LayoutDirection.LeftToRight,
        init_font_size=20, min_font_size=8, vertical=False,
        no_space_language=False, return_metrics=True,
        letter_spacing=6.0, word_spacing=12.0,
    )
    # Spaced text is wider at the same size, so it either wraps taller or the
    # fit shrinks the font; in both cases the measured height differs.
    assert (h_spaced != h_plain) or (size_spaced != size_plain)


def test_on_blk_rendered_applies_spacing(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(300, 300, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))

    main = _FakeMain(viewer)
    main.letter_spacing_dropdown.setCurrentText("4")
    main.word_spacing_dropdown.setCurrentText("8")
    controller = _make_controller(main)

    blk = TextBlock()
    blk.xyxy = [50, 50, 250, 250]
    blk.translation = "word " * 8
    controller.on_blk_rendered("word " * 8, 20, blk, "x.png")

    item = viewer.text_items[-1]
    assert item.letter_spacing == 4.0
    assert item.word_spacing == 8.0
    font = item.document().defaultFont()
    assert font.letterSpacing() == 4.0
    assert font.wordSpacing() == 8.0


def _render_one(controller, xyxy, text, font=20):
    blk = TextBlock()
    blk.xyxy = xyxy
    blk.translation = text
    controller.on_blk_rendered(text, font, blk, "x.png")
    return controller.main.image_viewer.text_items[-1]


def test_on_letter_spacing_change_refits(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    item = _render_one(controller, [50, 50, 350, 350], "word " * 8)
    main.curr_tblock_item = item
    h_before = item.document().size().height()

    main.letter_spacing_dropdown.setEditText("5")
    controller.on_letter_spacing_change("5")

    assert item.letter_spacing == 5.0
    assert item.document().size().height() != h_before


def test_on_spacing_change_ignores_partial_input(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    item = _render_one(controller, [50, 50, 350, 350], "word " * 8)
    main.curr_tblock_item = item
    before = item.letter_spacing

    for partial in ("-", "", "--", "abc", None):
        controller.on_letter_spacing_change(partial)  # must not raise
        controller.on_word_spacing_change(partial)

    assert item.letter_spacing == before

    controller.on_letter_spacing_change("3")
    assert item.letter_spacing == 3.0


def test_on_spacing_change_noop_when_nothing_selected(app, viewer_parent):
    """Regression: with no selection the spacing dropdown must not touch any
    block — only the selected ones. Previously include_all=True rewrote every
    block on the page, so resetting the dropdown to 0 after deselecting
    changed spacing on blocks the user never edited."""
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    items = [
        _render_one(controller, [50, 50, 150, 150], "word " * 4),
        _render_one(controller, [200, 50, 300, 150], "word " * 5),
        _render_one(controller, [50, 200, 150, 300], "word " * 6),
    ]
    # The user sets -2 on one selected block, then deselects (empty click).
    main.curr_tblock_item = items[0]
    items[0].selected = True
    controller.on_word_spacing_change("-2")
    assert items[0].word_spacing == -2.0

    main.curr_tblock_item = None
    for item in items:
        item.selected = False

    # Resetting the dropdown with nothing selected must not rewrite the page.
    controller.on_word_spacing_change("0")

    assert [item.word_spacing for item in items] == [-2.0, 0.0, 0.0]
    # ...and the untouched blocks keep their rendered metrics too.
    assert items[1].document().defaultFont().wordSpacing() == 0.0


def test_on_spacing_change_applies_to_selected_only_when_others_exist(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    selected = _render_one(controller, [50, 50, 150, 150], "word " * 4)
    other = _render_one(controller, [200, 50, 300, 150], "word " * 5)
    selected.selected = True
    other.selected = False
    main.curr_tblock_item = None

    controller.on_letter_spacing_change("2")

    assert selected.letter_spacing == 2.0
    assert other.letter_spacing == 0.0


def test_on_spacing_change_undo_restores_value(app, viewer_parent):
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    item = _render_one(controller, [50, 50, 350, 350], "word " * 8)
    main.curr_tblock_item = item
    stack = main.undo_group.activeStack()

    controller.on_letter_spacing_change("4")
    assert item.letter_spacing == 4.0

    stack.undo()
    assert item.letter_spacing == 0.0

    stack.redo()
    assert item.letter_spacing == 4.0


def test_spacing_reapplied_after_load_state_roundtrip(app, viewer_parent):
    """Page switch: save_state -> clear -> load_state must restore the spacing
    to the document font, not only to the item attributes.

    Regression: add_text_item seeded letter_spacing/word_spacing *after*
    set_text() (which calls apply_spacing()) and never re-applied, so a
    revisited page rendered at default spacing while the toolbar dropdown
    still showed the saved value.
    """
    viewer = ImageViewer(viewer_parent)
    image = QImage(400, 400, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    viewer.setPhoto(QPixmap.fromImage(image))
    main = _FakeMain(viewer)
    controller = _make_controller(main)

    item = _render_one(controller, [50, 50, 350, 350], "word " * 8)
    main.curr_tblock_item = item
    controller.on_letter_spacing_change("3")
    controller.on_word_spacing_change("6")
    assert item.letter_spacing == 3.0
    assert item.word_spacing == 6.0

    state = viewer.save_state()
    viewer.clear_scene()
    viewer.load_state(state)

    restored = viewer.text_items[0]
    # The attributes survive (this part always worked)...
    assert restored.letter_spacing == 3.0
    assert restored.word_spacing == 6.0
    # ...and the document font must carry them, otherwise the canvas shows
    # default spacing while the dropdown reports the saved value.
    font = restored.document().defaultFont()
    assert font.letterSpacing() == 3.0
    assert font.wordSpacing() == 6.0
    # Every explicit fragment font must carry the spacing too (Qt layout
    # prefers fragment fonts over the default font).
    cursor = QTextCursor(restored.document())
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    fmt = cursor.charFormat()
    frag_font = fmt.font()
    if fmt.fontFamilies():
        assert frag_font.letterSpacing() == 3.0
        assert frag_font.wordSpacing() == 6.0
