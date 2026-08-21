"""Regression tests for line spacing and vertical centering.

Verifies that:
  - vertical centering margin is applied via the document root frame
    (QTextBlockFormat top margins are ignored by Qt for the first paragraph)
  - the frame margin actually shifts the text down in the layout
  - multi-line text does not accumulate per-block margins
  - HTML text receives line spacing via apply_all_attributes()
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtGui import QTextCursor, QTextBlockFormat, QColor
from PySide6.QtWidgets import QApplication

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import set_center_v_margin, get_center_v_margin
from app.controllers.text import fit_and_center_text_item


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


def _apply_frame_top_margin(item: TextBlockItem, margin: float):
    """Apply a top margin to the document root frame (the fixed behaviour)."""
    set_center_v_margin(item.document(), margin)


# ── Fix #1: vertical centering ────────────────────────────────────────────


class TestVerticalCenteringViaRootFrame:
    """Vertical centering margin must live on the document root frame.

    QTextBlockFormat top margins are ignored by Qt for the first paragraph of
    a document, so the centering margin goes on the root frame, whose top
    margin the layout engine honors (the text actually moves down and
    doc.size().height() grows by exactly the margin).
    """

    def test_frame_margin_shifts_text_down(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("Hello")
        item.set_line_spacing(1.0)
        item.setTextWidth(200)
        height_before = item.document().size().height()

        _apply_frame_top_margin(item, margin=50)

        assert get_center_v_margin(item.document()) == 50.0
        # The margin is honored: doc height grows by exactly the margin, so the
        # text is pushed down rather than silently pinned to the top.
        assert item.document().size().height() == pytest.approx(height_before + 50, abs=1e-6)

    def test_blocks_keep_zero_margin(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("A\nB\nC")
        item.set_line_spacing(1.0)

        _apply_frame_top_margin(item, margin=20)

        for i in range(item.document().blockCount()):
            assert _block_top_margin(item, i) == 0.0

    def test_no_margin_by_default(self, app):
        item = TextBlockItem(line_spacing=1.0)
        item.setPlainText("A\nB\nC")
        item.set_line_spacing(1.0)

        assert get_center_v_margin(item.document()) == 0.0


# ── Fix #1 after re-translate (set_plain_text rebuilds the doc) ────────────


class TestRefitAfterRetranslate:
    """Re-translation replaces the document via set_plain_text, which drops the
    root-frame centering margin and resets the width. fit_and_center_text_item
    must re-center and grow the font to fill the box."""

    def test_set_plain_text_drops_margin_then_refit_recenters(self, app):
        item = TextBlockItem(line_spacing=1.0, font_size=20)
        item.setPlainText("Old translation")
        item.setTextWidth(200)
        fit_and_center_text_item(item, bw=200, bh=160, min_font=4.0,
                                 alignment=item.alignment)
        assert get_center_v_margin(item.document()) > 0

        # Re-translate: set_plain_text rebuilds the document and drops the
        # root-frame margin (text would be pinned to the top).
        item.set_plain_text("A brand new translation text")
        assert get_center_v_margin(item.document()) == 0.0

        # Refit re-applies the centering margin and regrows the font.
        fit_and_center_text_item(item, bw=200, bh=160, min_font=4.0,
                                 alignment=item.alignment, max_font=40.0)
        assert get_center_v_margin(item.document()) > 0

    def test_grow_fills_box(self, app):
        item = TextBlockItem(line_spacing=1.0, font_size=12)
        item.setPlainText("Hi")
        item.setTextWidth(200)
        fit_and_center_text_item(item, bw=200, bh=200, min_font=4.0,
                                 alignment=item.alignment, max_font=40.0)
        assert item.font_size > 12, "font should have grown to fill the box"
        assert item.document().size().height() <= 200
        assert get_center_v_margin(item.document()) > 0

    def test_no_grow_when_max_font_not_given(self, app):
        # Default max_font=0 keeps the old shrink-only behaviour (no grow).
        item = TextBlockItem(line_spacing=1.0, font_size=12)
        item.setPlainText("Hi")
        item.setTextWidth(200)
        fit_and_center_text_item(item, bw=200, bh=200, min_font=4.0,
                                 alignment=item.alignment)
        assert item.font_size == 12
        assert get_center_v_margin(item.document()) > 0


# ── Fix #1 applied in image_viewer.py path ────────────────────────────────


class TestImageViewerVMarginRootFrame:
    """image_viewer.add_text_item must apply v_margin via the root frame."""

    def test_v_margin_applied_via_root_frame(self, app):
        from app.ui.canvas.text.text_item_properties import TextItemProperties
        from app.ui.canvas.image_viewer import ImageViewer
        from PySide6.QtWidgets import QWidget
        from PySide6.QtCore import Qt

        props = TextItemProperties(
            text="Line1\nLine2",
            font_family="Sans",
            font_size=20,
            text_color=QColor("black"),
            line_spacing=1.0,
            width=200,
            v_margin=40.0,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        _parent = QWidget()
        viewer = ImageViewer(_parent)
        item = viewer.add_text_item(props)

        # Margin applied on the root frame (Qt default top margin 4.0 kept and
        # the requested 40.0 added on top); blocks stay untouched.
        assert get_center_v_margin(item.document()) == 40.0
        assert item.document().rootFrame().frameFormat().topMargin() == 44.0
        assert item.document().firstBlock().blockFormat().topMargin() == 0.0


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
