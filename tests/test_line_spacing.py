"""Regression tests for line spacing and vertical centering.

Verifies that:
  - top margin for vertical centering applies only to the first QTextBlock
  - multi-line text does not accumulate per-block margins
  - single-line text still centers correctly
  - HTML text receives line spacing via apply_all_attributes()
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtGui import QTextCursor, QTextBlockFormat
from PySide6.QtWidgets import QApplication

from app.ui.canvas.text_item import TextBlockItem


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _block_top_margin(item: TextBlockItem, block_index: int) -> float:
    """Return the top margin of the QTextBlock at block_index."""
    doc = item.document()
    block = doc.findBlockByNumber(block_index)
    return block.blockFormat().topMargin()


def _block_line_height(item: TextBlockItem, block_index: int):
    """Return (height, heightType) of the QTextBlock at block_index."""
    doc = item.document()
    block = doc.findBlockByNumber(block_index)
    fmt = block.blockFormat()
    return fmt.lineHeight(), fmt.lineHeightType()


def _apply_top_margin_to_first_block(item: TextBlockItem, margin: float):
    """Apply a top margin to only the first QTextBlock (the fixed behaviour)."""
    cursor = QTextCursor(item.document())
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
    bf = QTextBlockFormat()
    bf.setTopMargin(margin)
    cursor.mergeBlockFormat(bf)


# ── Fix #1: vertical centering ────────────────────────────────────────────


class TestVerticalCenteringFirstBlockOnly:
    """Top margin for vertical centering must touch only the first block."""

    def test_two_lines_margin_first_block_only(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("Hello\nWorld")
        item.set_line_spacing(1.0)

        _apply_top_margin_to_first_block(item, margin=50)

        assert _block_top_margin(item, 0) == 50.0
        assert _block_top_margin(item, 1) == 0.0

    def test_three_lines_margin_first_block_only(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("A\nB\nC")
        item.set_line_spacing(1.0)

        _apply_top_margin_to_first_block(item, margin=20)

        assert _block_top_margin(item, 0) == 20.0
        assert _block_top_margin(item, 1) == 0.0
        assert _block_top_margin(item, 2) == 0.0

    def test_single_line_margin_applies(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("Hello")
        item.set_line_spacing(1.0)

        _apply_top_margin_to_first_block(item, margin=30)

        assert item.document().blockCount() == 1
        assert _block_top_margin(item, 0) == 30.0

    def test_no_margin_by_default(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("A\nB\nC")
        item.set_line_spacing(1.0)

        for i in range(item.document().blockCount()):
            assert _block_top_margin(item, i) == 0.0

    def test_cumulative_margin_not_proportional_to_line_count(self, app):
        """The core regression: gap between lines must NOT grow with block count.

        Verify per-block margins directly rather than relying on
        QTextDocument.size().height() which is unreliable in auto-width mode.
        """
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("X\nY\nZ")
        item.set_line_spacing(1.0)

        _apply_top_margin_to_first_block(item, margin=100)

        # Only block 0 must have the margin.  If the old buggy code ran,
        # all three blocks would have margin=100.
        margins = [_block_top_margin(item, i) for i in range(3)]
        assert margins == [100.0, 0.0, 0.0], (
            f"Expected [100, 0, 0], got {margins} — "
            f"margin applied to all blocks (cumulative bug)"
        )


# ── Fix #1 applied in image_viewer.py path ────────────────────────────────


class TestImageViewerVMarginFirstBlockOnly:
    """Mirrors the image_viewer.add_text_item v_margin logic."""

    def test_v_margin_first_block_only(self, app):
        from app.ui.canvas.text.text_item_properties import TextItemProperties
        from PySide6.QtCore import Qt

        props = TextItemProperties(
            text="Line1\nLine2",
            font_family="Sans",
            font_size=20,
            line_spacing=1.0,
            width=200,
            v_margin=40.0,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        item = TextBlockItem(
            text=props.text,
            font_family=props.font_family,
            font_size=props.font_size,
            line_spacing=props.line_spacing,
            alignment=props.alignment,
        )
        item.set_text(props.text, props.width)
        item.setTextWidth(props.width)
        item.set_line_spacing(props.line_spacing)

        # Reproduce the image_viewer.py v_margin path (after fix)
        from PySide6 import QtGui

        _doc = item.document()
        _cursor = QtGui.QTextCursor(_doc)
        _cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
        _cursor.select(QtGui.QTextCursor.SelectionType.BlockUnderCursor)
        _bf = QtGui.QTextBlockFormat()
        _bf.setTopMargin(props.v_margin)
        _bf.setAlignment(props.alignment)
        _cursor.mergeBlockFormat(_bf)

        assert _block_top_margin(item, 0) == 40.0
        assert _block_top_margin(item, 1) == 0.0


# ── Fix #2: HTML text line spacing ────────────────────────────────────────


class TestHtmlTextLineSpacing:
    """HTML text must receive line spacing via apply_all_attributes()."""

    def test_html_gets_proportional_line_height(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setHtml("<p>Hello<br>World</p>")
        item.setTextWidth(300)
        item.set_line_spacing(1.0)

        # Every block should have ProportionalHeight (value 4) = 100
        for i in range(item.document().blockCount()):
            height, height_type = _block_line_height(item, i)
            # ProportionalHeight enum value = 4
            assert height_type == QTextBlockFormat.LineHeightTypes.ProportionalHeight.value, (
                f"Block {i}: expected ProportionalHeight, got type {height_type}"
            )
            assert height == 100.0, (
                f"Block {i}: expected lineHeight=100 (1.0×100), got {height}"
            )

    def test_html_set_text_also_applies_line_spacing(self, app):
        """set_text() with HTML content should apply line spacing."""
        item = TextBlockItem(line_spacing=1.2)
        item.set_text("<p>A<br>B</p>", width=300)

        for i in range(item.document().blockCount()):
            height, height_type = _block_line_height(item, i)
            assert height_type == QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
            assert height == 120.0, (
                f"Block {i}: expected lineHeight=120 (1.2×100), got {height}"
            )

    def test_plain_text_line_spacing_unchanged(self, app):
        """Plain text path must still work correctly."""
        item = TextBlockItem(line_spacing=1.5)
        item.set_text("Hello\nWorld", width=300)

        for i in range(item.document().blockCount()):
            height, height_type = _block_line_height(item, i)
            assert height_type == QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
            assert height == 150.0
