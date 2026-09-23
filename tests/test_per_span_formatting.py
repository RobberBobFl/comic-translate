"""Regression tests for per-span (per-word) text formatting.

Covers the three behaviours that make "format just the selected word" work:

1. ``update_text_format`` merges the char format into the current selection
   only, leaving the rest of the block untouched.
2. A user selection survives consecutive format operations (bold -> italic ->
   size) instead of being cleared after the first one.
3. When there is no selection, the whole block is still formatted and the
   scalar attributes (``bold``, ``font_size``) keep updating, which the
   auto-fit loop in ``fit_and_center_text_item`` depends on.
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QApplication, QTextEdit, QToolButton, QComboBox,
                               QPushButton, QCheckBox, QGraphicsScene)

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _select_span(item: TextBlockItem, start: int, end: int):
    cursor = item.textCursor()
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    item.setTextCursor(cursor)


def _per_char(item: TextBlockItem):
    """Return (bold, italic, point_size) lists, one entry per character."""
    cursor = QTextCursor(item.document())
    n = len(item.toPlainText())
    bold, italic, size = [], [], []
    for pos in range(n):
        cursor.setPosition(pos)
        cursor.setPosition(pos + 1, QTextCursor.MoveMode.KeepAnchor)
        fmt = cursor.charFormat()
        bold.append(fmt.font().bold())
        italic.append(fmt.font().italic())
        size.append(fmt.fontPointSize())
    return bold, italic, size


# ── per-span application ────────────────────────────────────────────────────

def test_bold_applies_only_to_selected_span(app):
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 0, 5)

    item.set_bold(True)

    bold, _, _ = _per_char(item)
    assert bold == [True] * 5 + [False] * 6


def test_consecutive_formats_hit_the_same_span(app):
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 0, 5)

    item.set_bold(True)
    item.set_italic(True)
    item.set_font_size(40)

    bold, italic, size = _per_char(item)
    assert bold == [True] * 5 + [False] * 6
    assert italic == [True] * 5 + [False] * 6
    assert size[:5] == [40.0] * 5


def test_selection_survives_format_operations(app):
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 6, 11)

    item.set_bold(True)
    item.set_italic(True)

    cursor = item.textCursor()
    assert cursor.hasSelection()
    assert (cursor.selectionStart(), cursor.selectionEnd()) == (6, 11)


def test_scalar_attrs_untouched_when_selection_present(app):
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 0, 5)

    item.set_bold(True)
    item.set_font_size(40)

    # Span-level edit must not rewrite the block-level defaults.
    assert item.bold is False
    assert item.font_size == 20


# ── whole-block fallback ─────────────────────────────────────────────────────

def test_no_selection_formats_whole_block(app):
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()

    item.set_bold(True)

    bold, _, _ = _per_char(item)
    assert all(bold)
    assert item.bold is True


def test_no_selection_keeps_scalar_and_clears_cursor(app):
    # The auto-fit loop relies on font_size updating and on no stale selection
    # leaking into the next iteration.
    item = TextBlockItem("Hi", font_size=12)
    item.setTextWidth(200)

    item.set_font_size(14)
    item.set_font_size(16)

    assert item.font_size == 16
    assert item.textCursor().hasSelection() is False


# ── mixed selections ─────────────────────────────────────────────────────────

def test_mixed_selection_reports_indeterminate_state(app):
    item = TextBlockItem("abcdefgh", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 0, 2)
    item.set_bold(True)

    _select_span(item, 0, 8)
    props = item.get_selected_text_properties(item.textCursor())

    # AND over the span -> not uniformly bold.
    assert props["bold"] is False


# ── persistence ──────────────────────────────────────────────────────────────

def test_span_formatting_survives_html_roundtrip(app):
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 0, 5)
    item.set_bold(True)

    props = TextItemProperties.from_text_item(item)
    rebuilt = TextBlockItem()
    rebuilt.set_text(props.text, props.width)

    bold, _, _ = _per_char(rebuilt)
    assert bold == [True] * 5 + [False] * 6
    assert rebuilt.toPlainText() == "hello world"


# ── panel selection transfer ─────────────────────────────────────────────────

class _FakeMain:
    def __init__(self):
        self.t_text_edit = QTextEdit()


def _make_controller(app):
    from app.controllers.text import TextController

    controller = TextController.__new__(TextController)
    controller.main = _FakeMain()
    return controller


def test_panel_selection_maps_onto_item(app):
    controller = _make_controller(app)
    controller.main.t_text_edit.setPlainText("hello world")
    panel_cursor = controller.main.t_text_edit.textCursor()
    panel_cursor.setPosition(6)
    panel_cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    controller.main.t_text_edit.setTextCursor(panel_cursor)

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    transferred = controller._transfer_panel_selection_to_item(item)

    assert transferred is True
    # The item must NOT be left in editing mode: that would make it keep
    # intercepting input and block later panel selections for the same block.
    assert item.editing_mode is False
    cursor = item.textCursor()
    assert (cursor.selectionStart(), cursor.selectionEnd()) == (6, 11)


def test_transferred_panel_selection_limits_bold(app):
    controller = _make_controller(app)
    controller.main.t_text_edit.setPlainText("hello world")
    panel_cursor = controller.main.t_text_edit.textCursor()
    panel_cursor.setPosition(6)
    panel_cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    controller.main.t_text_edit.setTextCursor(panel_cursor)

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    controller._transfer_panel_selection_to_item(item)
    item.set_bold(True)

    bold, _, _ = _per_char(item)
    assert bold == [False] * 6 + [True] * 5


def test_consecutive_panel_selections_format_different_words(app):
    """Regression: after the first panel-driven format the block used to stay in
    editing mode, so every later panel selection was ignored and the format kept
    hitting the first word (or the whole block)."""
    controller = _make_controller(app)
    controller.main.t_text_edit.setPlainText("one two three")
    item = TextBlockItem("one two three", font_family="Sans Serif", font_size=20)

    def _panel_select(start, end):
        panel_cursor = controller.main.t_text_edit.textCursor()
        panel_cursor.setPosition(start)
        panel_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        controller.main.t_text_edit.setTextCursor(panel_cursor)
        controller._transfer_panel_selection_to_item(item)

    _panel_select(4, 7)   # "two"
    item.set_bold(True)
    _panel_select(8, 13)  # "three"
    item.set_italic(True)

    bold, italic, _ = _per_char(item)
    assert bold == [False] * 4 + [True] * 3 + [False] * 6
    assert italic == [False] * 8 + [True] * 5
    assert item.editing_mode is False


def test_panel_selection_works_when_item_already_editing(app):
    """A panel selection is an explicit user action and wins over whatever span
    the item happened to have selected on the canvas."""
    controller = _make_controller(app)
    controller.main.t_text_edit.setPlainText("hello world")
    panel_cursor = controller.main.t_text_edit.textCursor()
    panel_cursor.setPosition(0)
    panel_cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    controller.main.t_text_edit.setTextCursor(panel_cursor)

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    item.enter_editing_mode()
    _select_span(item, 6, 11)

    assert controller._transfer_panel_selection_to_item(item) is True
    cursor = item.textCursor()
    assert (cursor.selectionStart(), cursor.selectionEnd()) == (0, 5)


def test_no_panel_selection_is_noop(app):
    controller = _make_controller(app)
    controller.main.t_text_edit.setPlainText("hello world")

    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    assert controller._transfer_panel_selection_to_item(item) is False
    assert item.editing_mode is False


# ── full controller pipeline (bold/italic buttons) ───────────────────────────

class _ToolbarFakeMain:
    """Minimal stand-in exposing every widget the controller touches."""

    def __init__(self, item, scene):
        self.t_text_edit = QTextEdit()
        self.t_text_edit.setPlainText("hello world")
        self.curr_tblock_item = item
        self.image_viewer = type("V", (), {"_scene": scene})()
        self.undo_group = type(
            "G", (), {"activeStack": lambda self: type("S", (), {"push": lambda s, c: None})()}
        )()
        self.bold_button = QToolButton()
        self.bold_button.setCheckable(True)
        self.italic_button = QToolButton()
        self.italic_button.setCheckable(True)
        self.underline_button = QToolButton()
        self.underline_button.setCheckable(True)
        self.font_size_dropdown = QComboBox()
        self.line_spacing_dropdown = QComboBox()
        self.outline_width_dropdown = QComboBox()
        self.outline_checkbox = QCheckBox()
        self.block_font_color_button = QPushButton()
        self.outline_font_color_button = QPushButton()
        self.alignment_tool_group = type(
            "ATG",
            (),
            {"get_button_group": lambda self: type("BG", (), {"buttons": lambda s: [QToolButton()] * 3})()},
        )()

    def push_command(self, command):
        pass

    def set_font(self, font_family):
        pass


def test_bold_italic_buttons_format_panel_selection(app):
    scene = QGraphicsScene()
    item = TextBlockItem("hello world", font_family="Sans Serif", font_size=20)
    scene.addItem(item)

    controller = _make_controller(app)
    controller.main = _ToolbarFakeMain(item, scene)
    controller.widgets_to_block = []
    controller.block_text_item_widgets = lambda widgets: None
    controller.unblock_text_item_widgets = lambda widgets: None

    panel_cursor = controller.main.t_text_edit.textCursor()
    panel_cursor.setPosition(6)
    panel_cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    controller.main.t_text_edit.setTextCursor(panel_cursor)

    controller.main.bold_button.setChecked(True)
    controller.bold()
    controller.main.italic_button.setChecked(True)
    controller.italic()

    bold, italic, _ = _per_char(item)
    assert bold == [False] * 6 + [True] * 5
    assert italic == [False] * 6 + [True] * 5
    # Toolbar reflects the span's formatting, not the block default.
    assert controller.main.bold_button.isChecked() is True


def test_consecutive_button_formats_hit_consecutive_selections(app):
    """Regression for the reported bug: the first panel-driven format worked,
    but a second one did not, and the block stayed 'active' until the user
    clicked something else."""
    scene = QGraphicsScene()
    item = TextBlockItem("one two three", font_family="Sans Serif", font_size=20)
    scene.addItem(item)

    controller = _make_controller(app)
    controller.main = _ToolbarFakeMain(item, scene)
    controller.main.t_text_edit.setPlainText("one two three")
    controller.widgets_to_block = []
    controller.block_text_item_widgets = lambda widgets: None
    controller.unblock_text_item_widgets = lambda widgets: None

    def _panel_select(start, end):
        panel_cursor = controller.main.t_text_edit.textCursor()
        panel_cursor.setPosition(start)
        panel_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        controller.main.t_text_edit.setTextCursor(panel_cursor)

    _panel_select(4, 7)  # "two"
    controller.main.bold_button.setChecked(True)
    controller.bold()
    _panel_select(8, 13)  # "three"
    controller.main.italic_button.setChecked(True)
    controller.italic()

    bold, italic, _ = _per_char(item)
    assert bold == [False] * 4 + [True] * 3 + [False] * 6
    assert italic == [False] * 8 + [True] * 5
    assert item.editing_mode is False
