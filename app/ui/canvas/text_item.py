from PySide6.QtWidgets import QGraphicsTextItem, QGraphicsItem, \
     QApplication, QWidget, QStyleOptionGraphicsItem
from PySide6.QtGui import QFont, QCursor, QColor, \
     QTextCharFormat, QTextBlockFormat, QTextCursor, QPainter, QPen, QTextFormat, \
     QFontMetricsF
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
import math, copy
from dataclasses import dataclass
from enum import Enum
from .text.vertical_layout import VerticalTextDocumentLayout

# Faux-bold stroke: on fonts with no Bold face (or a Bold face barely heavier
# than Regular) weight 700 alone is not visibly heavier, so a thin outline in
# the text color is added on top of the glyph. The width is proportional to the
# font size and is re-derived whenever size/color change (see
# _sync_faux_bold_outlines), otherwise the stroke would stay baked at its old
# width while the glyph rescales.
FAUX_BOLD_WIDTH_FACTOR = 0.015
FAUX_BOLD_MIN_PX = 0.3


@dataclass
class TextBlockState:
    rect: tuple  
    rotation: float
    transform_origin: QPointF

    @classmethod
    def from_item(cls, item: QGraphicsTextItem):
        """Create TextBlockState from a TextBlockItem"""
        rect = QRectF(item.pos(), item.boundingRect().size()).getCoords()
        return cls(
            rect=rect,
            rotation=item.rotation(),
            transform_origin=item.transformOriginPoint()
        )
    
class OutlineType(Enum):
    Full_Document = 'full_document'
    Selection = 'selection'
    
@dataclass
class OutlineInfo:
    start: int
    end: int
    color: QColor
    width: float
    type: OutlineType

class TextBlockItem(QGraphicsTextItem):
    text_changed = Signal(str)
    item_selected = Signal(object)
    item_deselected = Signal()
    text_highlighted = Signal(dict)
    change_undo = Signal(TextBlockState, TextBlockState)
    
    def __init__(self, 
             text = "", 
             font_family = "", 
             font_size = 20, 
             render_color = QColor(0, 0, 0), 
             alignment = Qt.AlignmentFlag.AlignCenter, 
             line_spacing = 1.2, 
             outline_color = QColor(255, 255, 255), 
             outline_width = 1,
             bold=False, 
             italic=False, 
             underline=False,
             direction=Qt.LayoutDirection.LeftToRight,
             block_id=""):

        super().__init__(text)
        self.text_color = render_color
        self.outline = True if outline_color else False
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.font_family = font_family
        self.font_size = font_size
        self.alignment = alignment
        self.line_spacing = line_spacing
        self.direction = direction
        self.block_id = block_id
        # Extra px added between letters / between words (QFont AbsoluteSpacing
        # and QFont::setWordSpacing). Applied to the document font in
        # apply_spacing(); 0 keeps Qt's default metrics.
        self.letter_spacing = 0.0
        self.word_spacing = 0.0
        # Curvature of the text arc in [-100, 100]: 0 renders the plain straight
        # layout, +c bends each line into an upward arch (cap) and -c into a
        # downward bowl, with |c| = 100 laying the widest line along a quarter
        # circle (90 deg). The layout is text-on-path: glyph spacing measured
        # along the arc equals the straight layout exactly and the rows sit on
        # concentric circles (see _draw_document_curved). Stored on the item
        # (not on the document) like letter/word spacing, and snapshotted by
        # TextFormatCommand's __dict__ copy for undo/redo.
        self.curvature = 0.0

        self.layout = None
        self.vertical = False

        self.selected = False
        self.resizing = False
        self.resize_handle = None
        self.resize_start = None
        self.editing_mode = False
        self.last_selection = None 
        self._drag_selecting = False
        self._drag_select_anchor = None

        # Rotation properties
        self.rot_handle = None
        self.rotating = False
        self.last_rotation_angle = 0
        self.rotation_smoothing = 1.0  # rotation sensitivity
        self.center_scene_pos = None  

        self.old_state = None

        self.selection_outlines = []

        self.setAcceptHoverEvents(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.document().contentsChanged.connect(self._on_text_changed)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setZValue(1)

        # Set the initial text direction
        self._apply_text_direction()

    def set_vertical(self, vertical: bool):
        doc = self.document()
        is_already_vertical = isinstance(doc.documentLayout(), VerticalTextDocumentLayout)

        if vertical == is_already_vertical:
            return

        self.vertical = vertical

        # Disconnect signals from the old layout if it's our custom one
        if is_already_vertical:
            old_layout = doc.documentLayout()
            if old_layout:
                try:
                    old_layout.size_enlarged.disconnect(self.on_document_enlarged)
                    old_layout.documentSizeChanged.disconnect(self.setCenterTransform)
                except (TypeError, RuntimeError): # Already disconnected
                    pass
        
        # Inform the graphics system that the geometry will change
        self.prepareGeometryChange()
        current_rect = self.boundingRect()

        # Disable text interaction while changing layout
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        
        if doc.documentLayout():
            doc.documentLayout().blockSignals(True)

        if vertical:
            layout = VerticalTextDocumentLayout(
                document=doc,
                line_spacing=self.line_spacing,
            )
            self.layout = layout
            doc.setDocumentLayout(layout)
            
            # Connect signals for the new layout
            layout.size_enlarged.connect(self.on_document_enlarged)
            layout.documentSizeChanged.connect(self.setCenterTransform)
            
            # Initialize layout with the item's current size.
            # set_max_size enforces the dimensions, but a text item with no 
            # set text has negligible size, so this can collapse the layout.
            # Only uncomment if set_vertical runs after plain text is set.
            # layout.set_max_size(current_rect.width(), current_rect.height())
            # layout.update_layout()

        else:  # Switching back to horizontal
            self.layout = None
            doc.setDocumentLayout(None)  # Qt will restore the default layout.
            self.setTextWidth(current_rect.width())
        
        # After setting the new layout, update the item's state
        self.setCenterTransform()
        self.update()

    def setCenterTransform(self):
        center = self.boundingRect().center()
        self.setTransformOriginPoint(center)

    def on_document_enlarged(self):
        self.prepareGeometryChange()
        self.setCenterTransform()

    def _apply_text_direction(self):
        text_option = self.document().defaultTextOption()
        text_option.setTextDirection(self.direction)
        self.document().setDefaultTextOption(text_option)

    def set_direction(self, direction):
        if self.direction != direction:
            self.direction = direction
            self._apply_text_direction()
            self.update()

    def set_text(self, text, width):
        if self.is_html(text):
            self.setHtml(text)
            self.setTextWidth(width)
            self.set_outline(self.outline_color, self.outline_width)
            # The HTML already carries per-span character formatting; only fix
            # the outline and block-level layout. Merging the block-level char
            # attributes over the whole document would erase those spans.
            self.apply_block_attributes()
        else:
            self.set_plain_text(text)

    def set_plain_text(self, text):
        self.setPlainText(text)
        self.apply_all_attributes()

    def is_html(self, text):
        import re
        if not isinstance(text, str):
            return False
        # Improved check for HTML tags (case-insensitive, includes common tags)
        return bool(re.search(r'<(div|span|p|br|html|body|style)[^>]*>', text, re.IGNORECASE))

    def set_font(self, font_family, font_size):
        # Ensure minimum font size.
        font_size = max(1, font_size)

        if not self.textCursor().hasSelection():
            self.font_family = font_family
            self.font_size = font_size

        # Fallback to application default font family if none provided
        effective_family = font_family.strip() if isinstance(font_family, str) and font_family.strip() else QApplication.font().family()
        font = QFont(effective_family, font_size)
        self.update_text_format('font', font)
        # update_text_format sets the default font, but letter/word spacing must
        # be applied on top of it to affect the layout metrics.
        self.apply_spacing()

    def apply_spacing(self):
        """Apply letter_spacing/word_spacing to the document.

        QFont.setLetterSpacing(AbsoluteSpacing) and setWordSpacing change how the
        text is laid out, so wider spacing wraps earlier and narrower spacing
        fits more per line. Two places must carry them:

        * the document's default font, used for fragments that have no explicit
          char-format font;
        * every fragment whose char format *does* carry an explicit font. Qt's
          rich-text layout prefers the fragment's char-format font over the
          default one, so spacing set only on the default font is silently
          ignored once a fragment has its own font -- which is the normal state
          after set_font()/set_html() (and after undo/redo), which merge a full
          QFont into the whole document.

        Always run, even when both values are 0: that is how a previous nonzero
        spacing is cleared instead of left baked into the fonts.

        The spacing is baked into a copy of each fragment's own QFont
        (preserving family/size/weight/italic/underline/etc.) and merged back
        over that fragment's exact range. The ranges come from
        QTextFragment.iteration rather than a QTextCursor: a cursor reports the
        format of the fragment to its right (or the merged format of a
        selection), so cursor-driven run detection corrupts neighbouring spans.
        """
        letter_spacing = float(getattr(self, "letter_spacing", 0.0) or 0.0)
        word_spacing = float(getattr(self, "word_spacing", 0.0) or 0.0)

        doc = self.document()
        default_font = doc.defaultFont()
        default_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
        default_font.setWordSpacing(word_spacing)
        doc.setDefaultFont(default_font)

        cursor = QTextCursor(doc)
        block = doc.begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                iterator += 1
                fmt = fragment.charFormat()
                # A fragment without its own font is laid out from the default
                # font (spaced above); merging a synthesised font here would
                # clobber its family/size.
                families = fmt.fontFamilies()
                size = fmt.fontPointSize()
                if not families or not families[0] or size <= 0:
                    continue
                font = QFont(families[0])
                font.setPointSizeF(size)
                font.setWeight(QFont.Weight(fmt.fontWeight()))
                font.setItalic(fmt.fontItalic())
                font.setUnderline(fmt.fontUnderline())
                font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
                font.setWordSpacing(word_spacing)
                new_fmt = QTextCharFormat()
                new_fmt.setFont(font)
                start = fragment.position()
                cursor.setPosition(start)
                cursor.setPosition(start + fragment.length(),
                                   QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(new_fmt)
            block = block.next()

    def set_font_size(self, font_size):
        font_size = max(1, font_size)
        if not self.textCursor().hasSelection():
            self.font_size = font_size
        self.update_text_format('size', font_size)

    def update_text_width(self):
        width = self.document().size().width()
        self.setTextWidth(width)

    def set_alignment(self, alignment):
        if not self.textCursor().hasSelection():
            self.alignment = alignment
        self.update_alignment(alignment)

    def update_alignment(self, alignment):
        cursor = self.textCursor()
        has_selection = cursor.hasSelection()
        block_format = cursor.blockFormat()
        block_format.setAlignment(alignment)

        if has_selection:
            cursor.beginEditBlock()
            start, end = cursor.selectionStart(), cursor.selectionEnd()
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            cursor.mergeBlockFormat(block_format)
            cursor.endEditBlock()
        else:
            doc = self.document()
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.mergeBlockFormat(block_format)

        self.update()

    def update_text_format(self, attribute, value):
        cursor = self.textCursor()
        has_selection = cursor.hasSelection()

        if has_selection:
            fmt_start = cursor.selectionStart()
            fmt_end = cursor.selectionEnd()
        else:
            fmt_start = 0
            fmt_end = max(0, self.document().characterCount())

        def apply_bold(cf, v):
            cf.setFontWeight(QFont.Bold if v else QFont.Normal)
            # The outline is applied *and* removed per fragment by
            # _sync_faux_bold_outlines, never here. mergeCharFormat cannot
            # remove the textOutline property, and merging NoPen would still
            # make toHtml() emit -qt-stroke-* CSS that comes back as a solid
            # 1px outline after an HTML round-trip (undo/redo, save/load,
            # export), thickening text the user never made bold.

        format_operations = {
            'color': lambda cf, v: cf.setForeground(v),
            'font': lambda cf, v: cf.setFont(v),
            'size': lambda cf, v: cf.setFontPointSize(v),
            'bold': apply_bold,
            'italic': lambda cf, v: cf.setFontItalic(v),
            'underline': lambda cf, v: cf.setFontUnderline(v),
        }

        if attribute not in format_operations:
            print(f"Unsupported attribute: {attribute}")
            return

        char_format = QTextCharFormat()
        format_operations[attribute](char_format, value)

        # Only merge if we have a selection OR if it's not a color change on HTML.
        # This prevents clobbering inline span colors during project loading/global changes.
        is_global_html_color = not has_selection and attribute == 'color' and self.is_html(self.toHtml())

        if not is_global_html_color:
            if not has_selection:
                cursor.select(QTextCursor.SelectionType.Document)
            cursor.mergeCharFormat(char_format)

        # Update the document's default format only if there is no selection
        if not has_selection:
            doc_format = self.document().defaultTextOption()
            if attribute == 'color':
                self.setDefaultTextColor(value)
            elif attribute == 'font':
                self.document().setDefaultFont(value)
            elif attribute == 'size':
                font = self.document().defaultFont()
                font.setPointSize(value)
                self.document().setDefaultFont(font)
            self.document().setDefaultTextOption(doc_format)

        # Keep a genuine user selection intact so consecutive format operations
        # (bold, then italic, then size) keep hitting the same span. A synthetic
        # whole-document selection (created above when nothing was selected) is
        # cleared again so the setters' hasSelection() guards stay consistent.
        if has_selection:
            self.setTextCursor(cursor)
        else:
            cleared = QTextCursor(self.document())
            self.setTextCursor(cleared)

        # The faux-bold stroke is derived from each fragment's own size and
        # color, so it must be re-synced whenever those change -- otherwise the
        # stroke keeps its old width/color while the glyph rescales or
        # recolors. Skipped alongside a skipped color merge so outlines stay
        # matched to span colors that were deliberately left untouched.
        if attribute in ("bold", "size", "font", "color") and not is_global_html_color:
            self._sync_faux_bold_outlines(fmt_start, fmt_end)

        self.update()

    def _sync_faux_bold_outlines(self, start: int, end: int) -> None:
        """Reconcile the faux-bold outline of every fragment in [start, end).

        The synthetic stroke is what makes bold readable on fonts whose Bold
        face is missing or barely heavier than Regular. Two invariants are
        enforced per fragment:

        * bold -> a text outline whose width is proportional to the fragment's
          own point size and whose color matches its foreground, so resizing
          or recoloring a bold span rescales the stroke with it (an SFX word at
          48pt inside a 20pt block, or a whole block resized after bolding);
        * not bold -> no text outline *property* at all. The property has to be
          removed (setCharFormat with a cleared copy), not overwritten with
          NoPen: mergeCharFormat cannot drop properties, and a NoPen pen still
          makes toHtml() emit -qt-stroke-width, which Qt turns back into a
          solid 1px outline on import.

        Fragment ranges come from QTextFragment iteration, not a cursor, for
        the same reason as apply_spacing(): a cursor reports the format of the
        fragment to its right, so cursor-driven run detection corrupts
        neighbouring spans.
        """
        doc = self.document()
        cursor = QTextCursor(doc)
        block = doc.begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                iterator += 1
                f_start = fragment.position()
                f_end = f_start + fragment.length()
                if f_end <= start or f_start >= end:
                    continue
                fmt = fragment.charFormat()
                is_bold = fmt.fontWeight() >= QFont.Bold
                has_outline = fmt.hasProperty(QTextFormat.Property.TextOutline)
                if not is_bold:
                    if has_outline:
                        cleared = QTextCharFormat(fmt)
                        cleared.clearProperty(QTextFormat.Property.TextOutline)
                        cursor.setPosition(max(f_start, start))
                        cursor.setPosition(min(f_end, end),
                                           QTextCursor.MoveMode.KeepAnchor)
                        cursor.setCharFormat(cleared)
                    continue
                size = fmt.fontPointSize()
                if size <= 0:
                    size = doc.defaultFont().pointSizeF()
                if size <= 0:
                    size = self.font_size
                brush = fmt.foreground()
                if brush.style() != Qt.BrushStyle.NoBrush and brush.color().isValid():
                    color = brush.color()
                else:
                    color = self.text_color or QColor(0, 0, 0)
                pen = QPen(color, max(FAUX_BOLD_MIN_PX, size * FAUX_BOLD_WIDTH_FACTOR))
                pen.setJoinStyle(Qt.RoundJoin)
                new_fmt = QTextCharFormat()
                new_fmt.setTextOutline(pen)
                cursor.setPosition(max(f_start, start))
                cursor.setPosition(min(f_end, end), QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(new_fmt)
            block = block.next()

    def set_line_spacing(self, spacing):
        self.line_spacing = spacing
        doc = self.document()
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.SelectionType.Document)
        block_format = QTextBlockFormat()
        spacing = spacing * 100
        spacing = float(spacing)
        block_format.setLineHeight(spacing, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        cursor.mergeBlockFormat(block_format)

    def set_color(self, color):
        if not self.textCursor().hasSelection():
            self.text_color = color
        self.update_text_format('color', color)

    def update_outlines(self):
        """Update the selection outlines when text changes"""
        if self.outline:
            # Create an outline for the entire document
            doc = self.document()
            char_count = doc.characterCount()
            
            # Create an outline info for the entire document
            new_outline = OutlineInfo(  
                start = 0,
                end = max(0, char_count - 1),
                color = self.outline_color,  
                width = self.outline_width,
                type = OutlineType.Full_Document
            )
            
            # Remove any existing full document outline
            self.selection_outlines = [outline for outline in self.selection_outlines 
                                     if outline.type != OutlineType.Full_Document]
            # Add the new one
            self.selection_outlines.append(new_outline)
        else:
            # Remove only the full document outline
            self.selection_outlines = [outline for outline in self.selection_outlines 
                                     if outline.type != OutlineType.Full_Document]

        self.update() 

    def set_outline(self, outline_color, outline_width):
        # Initialize start and end variables
        start = 0
        end = 0

        if self.textCursor().hasSelection():
            # Store outline properties for the current selection
            start = self.textCursor().selectionStart()
            end = self.textCursor().selectionEnd()
        else:
            # Set global outline properties only when there's no selection
            self.outline = True if outline_color else False

            if self.outline:
                # enabling global outline: store color/width and target whole document
                self.outline_color = outline_color
                self.outline_width = outline_width

                char_count = self.document().characterCount()
                start = 0
                end = max(0, char_count - 1)

        # When disabling outlines (outline_color is falsy), remove the relevant outlines
        if not outline_color:
            if self.textCursor().hasSelection():
                # Remove any outlines that contain the current selection range
                self.selection_outlines = [
                    outline for outline in self.selection_outlines
                    if not (outline.start <= start and outline.end >= end)
                ]
            else:
                # No selection: remove only full-document outlines
                self.selection_outlines = [
                    outline for outline in self.selection_outlines
                    if outline.type != OutlineType.Full_Document
                ]
        else:
            # Adding/updating an outline for the selection or whole document
            type = OutlineType.Selection if self.textCursor().hasSelection() else OutlineType.Full_Document

            # Remove any existing outline for this exact selection range
            self.selection_outlines = [
                outline for outline in self.selection_outlines 
                if not (outline.start == start and outline.end == end)
            ]

            # Add new outline info
            self.selection_outlines.append(
                OutlineInfo(start, end, outline_color, outline_width, type)
            )
        
        self.update()

    def contentBoundingRect(self) -> QRectF:
        """The plain document rect, without the slack boundingRect() adds for
        curved rendering. Used where the *text* box itself matters (resize
        anchors, saved width/height) instead of the visual hit area."""
        return super().boundingRect()

    def selectionRect(self) -> QRectF:
        """The rect all *interaction* works against: the dashed selection
        border, resize handles, and the page-edge constraint. Identical to
        contentBoundingRect(); a separate name so call sites read as intent
        (boundingRect() carries arc slack that must not drive the UI)."""
        return self.contentBoundingRect()

    def boundingRect(self) -> QRectF:
        rect = super().boundingRect()
        pad = self._curvature_pad(rect)
        if pad <= 0:
            return rect
        return rect.adjusted(-pad, -pad, pad, pad)

    def set_curvature(self, value):
        value = max(-100.0, min(100.0, float(value or 0.0)))
        if abs(value - float(getattr(self, "curvature", 0.0) or 0.0)) < 1e-9:
            return
        # prepareGeometryChange must run while the old (old-curvature) rect is
        # still current, so the scene invalidates the right region.
        self.prepareGeometryChange()
        self.curvature = value
        self.setCenterTransform()
        self.update()

    def _curvature_active(self) -> bool:
        # Editing needs a straight layout for the caret and hit-testing, and
        # vertical text runs through VerticalTextDocumentLayout, which the
        # glyph pipeline below does not understand.
        return (abs(float(getattr(self, "curvature", 0.0) or 0.0)) > 1e-9
                and not self.editing_mode
                and not self.vertical)

    @staticmethod
    def _arc_theta(curvature: float) -> float:
        """Angle the widest line subtends on its arc: |c| = 100 bends it
        through a quarter circle (90 deg)."""
        return abs(curvature) / 100.0 * (math.pi / 2.0)

    def _curvature_pad(self, rect: QRectF) -> float:
        if not self._curvature_active():
            return 0.0
        width = rect.width()
        if width <= 0:
            return 0.0
        theta = self._arc_theta(float(getattr(self, "curvature", 0.0) or 0.0))
        if theta <= 1e-9:
            return 0.0
        # Same circle the painter uses: the widest line wraps onto a radius of
        # width/theta. Rows away from the middle baseline sit on circles up to
        # one text height farther out, and a row's sagitta grows with its
        # radius, so bound it with the outermost radius. End glyphs rotated by
        # up to theta/2 can push ink outward by about a line height in any
        # direction.
        radius = width / theta
        sagitta = (radius + rect.height()) * (1.0 - math.cos(theta / 2.0))
        return sagitta + rect.height()

    def paint(   
        self, 
        painter: QPainter, 
        option: QStyleOptionGraphicsItem, 
        widget: QWidget = None
    ):

        if self._curvature_active():
            try:
                self._draw_selection_outlines(painter, self._draw_document_curved)
                self._draw_document_curved(painter, self.document())
            except Exception:
                # Any glyph-level failure (exotic script, missing font data)
                # falls back to the straight renderer instead of blanking.
                super().paint(painter, option, widget)
                return
            # QGraphicsTextItem.paint() ends by drawing Qt's dashed selection
            # border around boundingRect(); the curved path skips super, so
            # draw the same border around the tight selection rect (pen
            # matches qt_graphicsItem_highlightSelected).
            if self.isSelected():
                painter.save()
                painter.setPen(QPen(QColor(0, 0, 0, 127), 0, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(self.selectionRect())
                painter.restore()
            return

        # Then handle any selection outlines
        self._draw_selection_outlines(
            painter, lambda p, doc: doc.drawContents(p))

        # Draw the normal text on top
        super().paint(painter, option, widget)

    def _draw_selection_outlines(self, painter: QPainter, draw_doc):
        """Paint the stored selection outlines as offset copies of a
        recoloured clone of the document. *draw_doc(painter, doc)* renders one
        such clone; the straight path passes document drawContents, the curved
        path passes the arc renderer so both stay in sync."""
        if not self.selection_outlines:
            return

        painter.save()
        try:
            for outline_info in self.selection_outlines:
                doc = self._clone_outline_document()

                cursor = QTextCursor(doc)
                cursor.select(QTextCursor.SelectionType.Document)
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(0, 0, 0, 0))
                cursor.mergeCharFormat(fmt)

                cursor.setPosition(outline_info.start)
                cursor.setPosition(outline_info.end, QTextCursor.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setForeground(outline_info.color)
                cursor.mergeCharFormat(fmt)

                # Draw the outline for this selection
                offsets = [(dx, dy) 
                    for dx in (-outline_info.width, 0, outline_info.width)
                    for dy in (-outline_info.width, 0, outline_info.width)
                    if dx != 0 or dy != 0
                ]

                for dx, dy in offsets:
                    painter.save()
                    painter.translate(dx, dy)
                    draw_doc(painter, doc)
                    painter.restore()
        finally:
            painter.restore()

    def _clone_outline_document(self):
        source = self.document()
        doc = source.clone()
        doc.setDocumentMargin(source.documentMargin())
        doc.setDefaultFont(source.defaultFont())
        doc.setDefaultTextOption(source.defaultTextOption())
        doc.setPageSize(source.pageSize())
        doc.setTextWidth(source.textWidth())

        if self.vertical and self.layout:
            vertical_layout = VerticalTextDocumentLayout(
                document=doc,
                line_spacing=self.layout.line_spacing,
            )
            doc.setDocumentLayout(vertical_layout)
            vertical_layout.set_max_size(self.layout.max_width, self.layout.max_height)

        return doc

    @staticmethod
    def _line_cursor_x(line, pos) -> float:
        # PySide6 exposes QTextLine::cursorToX(int) as a (x, cursorPos) tuple.
        result = line.cursorToX(pos)
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)

    @staticmethod
    def _fragment_color(fmt, default: QColor) -> QColor:
        brush = fmt.foreground()
        if brush.style() != Qt.BrushStyle.NoBrush and brush.color().isValid():
            return brush.color()
        return default

    def _draw_document_curved(self, painter: QPainter, doc):
        """Draw *doc* glyph-by-glyph with the block's lines on concentric
        circular arcs -- text-on-path, not a warp.

        The straight render places glyphs at layout position + glyph run
        position (verified against QGraphicsTextItem's own output); this
        walks the same structures but re-lays the text onto an arc for
        self.curvature:

        * theta = |c|/100 * pi/2 is the angle the *widest* line subtends;
          sign(c) picks arch up (cap) vs arch down (bowl).
        * The block circle has radius R = widest_width / theta, centered one
          radius beyond the block's *middle* baseline. Each row sits on its
          own concentric circle: its radius is its distance to that common
          center, so rows run parallel and short lines bend gentler than the
          wide ones.
        * Glyphs walk the arc *by arc length*: a glyph's straight offset from
          the line start is its offset along the arc. Spacing and kerning
          measured along the arc therefore match the straight layout exactly
          -- the letters stay an even rank, they do not fan apart at the ends
          (the block only narrows slightly, its chord being shorter than the
          arc).
        * Every line's arc crosses the line's straight baseline at the line's
          midpoint (Photoshop-arc anchoring): the ends swing below it (arch)
          / above it (bowl) by the sagitta, so the rows keep their spacing at
          the block's center and fan apart toward the ends instead of
          colliding. Glyphs rotate to the arc tangent; glyph shapes themselves
          are never scaled.
        """
        curvature = float(getattr(self, "curvature", 0.0) or 0.0)
        sign = 1.0 if curvature > 0 else -1.0
        theta = self._arc_theta(curvature)
        if theta < 1e-6:
            doc.drawContents(painter)
            return

        # Forces the layout: touching QTextLine/glyphRuns before the document
        # has been laid out segfaults the Qt gui thread.
        _ = doc.size()

        # Pass 1: collect every line with its width and baseline so the whole
        # block shares one circle.
        line_entries = []  # (origin, fragments, line, width, x0, baseline)
        block = doc.begin()
        while block.isValid():
            block_layout = block.layout()
            origin = block_layout.position()
            block_pos = block.position()
            fragments = []
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                iterator += 1
                if fragment.isValid():
                    frag_pos = fragment.position() - block_pos
                    fragments.append((frag_pos, frag_pos + fragment.length(),
                                      fragment.charFormat()))
            for line_index in range(block_layout.lineCount()):
                line = block_layout.lineAt(line_index)
                text_start = line.textStart()
                text_end = text_start + line.textLength()
                if text_end <= text_start:
                    continue
                x0 = self._line_cursor_x(line, text_start)
                x1 = self._line_cursor_x(line, text_end)
                baseline = origin.y() + line.position().y() + line.ascent()
                line_entries.append(
                    (origin, fragments, line, x1 - x0, x0, baseline))
            block = block.next()

        widths = [entry[3] for entry in line_entries if entry[3] > 1.0]
        if not widths:
            doc.drawContents(painter)
            return
        block_width = max(widths)
        radius = block_width / theta
        baselines = [entry[5] for entry in line_entries]
        mid_baseline = (min(baselines) + max(baselines)) / 2.0

        for origin, fragments, line, width, x0, baseline in line_entries:
            if width <= 1.0:
                # Degenerate or RTL line (cursorToX runs right-to-left): draw
                # glyphs straight rather than at a meaningless radius.
                self._draw_line_curved(
                    painter, line, doc, origin, fragments,
                    None, None, None, None, sign)
                continue
            # The row's circle is concentric with the block's: rows away from
            # the middle baseline are farther from the common center. Floored
            # at half the line width so a tall narrow block can never wrap a
            # row past half a circle.
            row_radius = max(radius + sign * (mid_baseline - baseline),
                             width / 2.0)
            row_theta = width / row_radius
            self._draw_line_curved(
                painter, line, doc, origin, fragments,
                row_radius, row_theta, x0, origin.x() + x0 + width / 2.0,
                sign)

    def _draw_line_curved(self, painter, line, doc, origin, fragments,
                          radius, theta, x0, center_x, sign):
        """Draw one text line onto its (already sized) circle, or straight
        when *radius* is None. *x0* is the line's left edge in block-layout
        coordinates; *center_x* is the circle's horizontal center in the same
        coordinates."""
        text_start = line.textStart()
        text_end = text_start + line.textLength()
        if text_end <= text_start:
            return

        default_color = self.defaultTextColor()
        for frag_start, frag_end, fmt in fragments:
            seg_from = max(frag_start, text_start)
            seg_to = min(frag_end, text_end)
            if seg_to <= seg_from:
                continue

            color = self._fragment_color(fmt, default_color)
            outline_pen = None
            if fmt.hasProperty(QTextFormat.Property.TextOutline):
                pen = fmt.textOutline()
                if pen.style() != Qt.PenStyle.NoPen and pen.widthF() > 0 \
                        and color.alpha() > 0:
                    outline_pen = pen

            underline_pen = None
            underline_offset = 0.0
            if fmt.fontUnderline():
                font = fmt.font() if fmt.fontPointSize() > 0 else None
                if font is None or font.pointSizeF() <= 0:
                    font = doc.defaultFont()
                metrics = QFontMetricsF(font)
                underline_pen = QPen(
                    color, max(1.0, metrics.underlineThickness()))
                underline_offset = metrics.underlinePosition()

            for run in line.glyphRuns(seg_from, seg_to - seg_from):
                self._draw_run_curved(
                    painter, run, origin, color, outline_pen,
                    underline_pen, underline_offset,
                    radius, theta, x0, center_x, sign)

    def _draw_run_curved(self, painter, run, origin, color, outline_pen,
                         underline_pen, underline_offset,
                         radius, theta, x0, center_x, sign):
        positions = run.positions()
        glyph_indexes = list(run.glyphIndexes())
        count = min(len(positions), len(glyph_indexes))
        if count == 0:
            return
        raw_font = run.rawFont()
        try:
            advances = [pt.x() for pt in
                        raw_font.advancesForGlyphIndexes(glyph_indexes)]
        except Exception:
            advances = []
        if len(advances) < count:
            advances = []

        for i in range(count):
            if advances:
                advance = advances[i]
            elif i + 1 < count:
                advance = positions[i + 1].x() - positions[i].x()
            else:
                advance = 0.0

            pos = positions[i]
            ox = origin.x() + pos.x()
            oy = origin.y() + pos.y()
            try:
                path = raw_font.pathForGlyph(glyph_indexes[i])
            except Exception:
                continue

            if radius is None or advance <= 0:
                # Straight fallback for degenerate slots (zero-width
                # combining marks, empty line).
                painter.save()
                painter.translate(ox, oy)
            else:
                # Text-on-path: the glyph's straight midpoint offset from the
                # line start is its offset along the arc (so spacing measured
                # on the arc equals the straight layout), the glyph rotates
                # to the tangent, and the line's midpoint stays on its
                # straight baseline (the ends swing past it). See
                # _draw_document_curved for the block-level geometry.
                mid = ox + advance / 2.0
                phi = -theta / 2.0 + (mid - (origin.x() + x0)) / radius
                chord_x = center_x + radius * math.sin(phi)
                drop = sign * radius * (1.0 - math.cos(phi))
                painter.save()
                painter.translate(chord_x, oy + drop)
                painter.rotate(sign * math.degrees(phi))
                painter.translate(-advance / 2.0, 0.0)

            if outline_pen is not None:
                painter.setPen(outline_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPath(path)
            if underline_pen is not None:
                # Half-pixel overlap hides seams between neighbouring glyphs'
                # underline segments on the arc.
                painter.setPen(underline_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(QPointF(-0.5, underline_offset),
                                 QPointF(advance + 0.5, underline_offset))
            painter.restore()

    def set_bold(self, state):
        if not self.textCursor().hasSelection():
            self.bold = state
        self.update_text_format('bold', state)

    def set_italic(self, state):
        if not self.textCursor().hasSelection():
            self.italic = state
        self.update_text_format('italic', state)

    def set_underline(self, state):
        if not self.textCursor().hasSelection():
            self.underline = state
        self.update_text_format('underline', state)

    def apply_all_attributes(self):
        self.set_font(self.font_family, self.font_size)
        self.set_color(self.text_color)
        self.set_outline(self.outline_color, self.outline_width)
        self.set_bold(self.bold)
        self.set_italic(self.italic)
        self.set_underline(self.underline)
        self.set_line_spacing(self.line_spacing)
        self.update_text_width()
        self.set_alignment(self.alignment)

    def apply_block_attributes(self):
        """Apply only block-level attributes (line spacing, alignment, width).

        Used when loading HTML that already carries per-span character
        formatting: merging the block-level char attributes over the whole
        document would erase those spans.
        """
        self.set_line_spacing(self.line_spacing)
        # Spacing is a font-level (not char-format) property, so it is safe to
        # re-apply here without clobbering per-span formats.
        self.apply_spacing()
        self.update_text_width()
        self.set_alignment(self.alignment)
        # HTML saved before the faux-bold fix carries a bold weight with no
        # -qt-stroke-* at all (or a width baked at a since-changed size).
        # Re-derive every bold fragment's outline so old projects render with
        # the same stroke as newly bolded text.
        self._sync_faux_bold_outlines(0, max(0, self.document().characterCount()))

    def mouseDoubleClickEvent(self, event):
        if not self.editing_mode:
            self.enter_editing_mode()
            if self.layout:
                hit = self.layout.hitTest(event.pos(), None)
                cursor = self.textCursor()
                cursor.setPosition(hit)
                self.setTextCursor(cursor)
        super().mouseDoubleClickEvent(event)
        # Toolbar should reflect the (possibly empty) selection after entering
        # editing mode or repositioning the caret.
        self._on_selection_changed()

    def mousePressEvent(self, event):
        # Handle single clicks in editing mode for vertical text
        if self.editing_mode and self.layout and event.button() == Qt.MouseButton.LeftButton:
            hit = self.layout.hitTest(event.pos(), None)
            cursor = self.textCursor()
            
            # Check if shift is pressed for selection
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._drag_select_anchor = cursor.anchor()
                cursor.setPosition(hit, QTextCursor.MoveMode.KeepAnchor)
            else:
                cursor.setPosition(hit)
                self._drag_select_anchor = hit
            
            self._drag_selecting = True
            self.setTextCursor(cursor)
            self.setFocus()
            event.accept()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):

        if self.editing_mode and self.vertical:
            key = event.key()
            modifiers = event.modifiers()
            
            if key == Qt.Key.Key_Down:
                # Down arrow in vertical text = move to next character
                cursor = self.textCursor()
                move_mode = QTextCursor.MoveMode.KeepAnchor if (modifiers & Qt.KeyboardModifier.ShiftModifier) else QTextCursor.MoveMode.MoveAnchor
                cursor.movePosition(QTextCursor.MoveOperation.NextCharacter, move_mode)
                self.setTextCursor(cursor)
                event.accept()
                return
            elif key == Qt.Key.Key_Up:
                # Up arrow in vertical text = move to previous character
                cursor = self.textCursor()
                move_mode = QTextCursor.MoveMode.KeepAnchor if (modifiers & Qt.KeyboardModifier.ShiftModifier) else QTextCursor.MoveMode.MoveAnchor
                cursor.movePosition(QTextCursor.MoveOperation.PreviousCharacter, move_mode)
                self.setTextCursor(cursor)
                event.accept()
                return
            elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and not (
                modifiers & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
            ):
                # Left/Right arrow in vertical text = move between paragraphs (visual columns),
                # keeping the same in-block offset when possible.
                cursor = self.textCursor()
                move_mode = QTextCursor.MoveMode.KeepAnchor if (modifiers & Qt.KeyboardModifier.ShiftModifier) else QTextCursor.MoveMode.MoveAnchor

                # Prefer layout-aware movement (handles wrapped columns).
                if self.layout and hasattr(self.layout, "move_cursor_between_columns"):
                    column_delta = 1 if key == Qt.Key.Key_Left else -1
                    new_pos = self.layout.move_cursor_between_columns(cursor.position(), column_delta)
                    if new_pos is not None and new_pos != cursor.position():
                        cursor.setPosition(new_pos, move_mode)
                        self.setTextCursor(cursor)
                        event.accept()
                        return

                # Fallback: treat each QTextBlock as a vertical "line" and move between them.
                block = cursor.block()
                target_block = block.next() if key == Qt.Key.Key_Left else block.previous()
                if target_block.isValid():
                    offset_in_block = cursor.position() - block.position()
                    target_offset = min(offset_in_block, max(0, target_block.length() - 1))
                    new_pos = target_block.position() + target_offset
                    if new_pos != cursor.position():
                        cursor.setPosition(new_pos, move_mode)
                        self.setTextCursor(cursor)
                        event.accept()
                        return
            
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # Use the current character format as the new block's char format so empty paragraphs
                # keep the same font metrics (Qt can otherwise fall back to a tiny default).
                # Currently only necessary for vertical text layouts.
                cursor = self.textCursor()
                inherited_char_format = QTextCharFormat(cursor.charFormat())
                inherited_block_format = cursor.blockFormat()
                inherited_block_char_format = QTextCharFormat(inherited_char_format)

                # Ensure we always carry a valid point size/font for layout metrics.
                if inherited_block_char_format.fontPointSize() <= 0:
                    inherited_block_char_format.setFontPointSize(max(1, float(self.font_size)))
                font = inherited_block_char_format.font()
                if font.pointSizeF() <= 0:
                    font = self.document().defaultFont()
                    if font.pointSizeF() <= 0:
                        font.setPointSizeF(max(1.0, float(self.font_size)))
                    inherited_block_char_format.setFont(font)

                cursor.beginEditBlock()
                if cursor.hasSelection():
                    cursor.removeSelectedText()

                # Create a new paragraph that keeps the current paragraph + char formatting.
                cursor.insertBlock(inherited_block_format, inherited_block_char_format)
                cursor.setCharFormat(inherited_char_format)

                cursor.endEditBlock()
                self.setTextCursor(cursor)
                event.accept()
                return
        
        # Default handling for all other cases
        super().keyPressEvent(event)

    def enter_editing_mode(self):
        # Geometry first, while the curved rect is still current: editing
        # switches the render path to straight, dropping the curvature pad.
        self.prepareGeometryChange()
        self.editing_mode = True
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
        self.setCenterTransform()
        self.setFocus()
        self.update()

    def exit_editing_mode(self):
        self.prepareGeometryChange()
        self.editing_mode = False
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setCenterTransform()
        self.clearFocus()
        self.update()

    def _on_text_changed(self):
        new_text = self.toPlainText()
        self.text_changed.emit(new_text)
        self.update_outlines()

    def mouseMoveEvent(self, event):
        # Resize/rotate/move logic is now handled by EventHandler and QGraphicsView
        if self.editing_mode and self.layout and (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_selecting:
            hit = self.layout.hitTest(event.pos(), None)
            anchor = self._drag_select_anchor
            if anchor is None:
                anchor = self.textCursor().anchor()

            cursor = self.textCursor()
            cursor.setPosition(anchor)
            cursor.setPosition(hit, QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cursor)
            event.accept()
            return

        if self.editing_mode:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.editing_mode and self.layout and event.button() == Qt.MouseButton.LeftButton:
            self._drag_selecting = False
            self._drag_select_anchor = None
            event.accept()
            self._on_selection_changed()
            return
        super().mouseReleaseEvent(event)
        if self.editing_mode:
            # A mouse-up in editing mode usually finalizes a text selection
            # (e.g. drag-select of a word); refresh the toolbar to match it.
            self._on_selection_changed()

    def contextMenuEvent(self, event):
        super().contextMenuEvent(event)
        if self.editing_mode:
            self.enter_editing_mode()
    
    def handleDeselection(self):
        if self.selected:
            self.setSelected(False)
            self.selected = False
            self.item_deselected.emit()
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            if self.editing_mode:
                self.exit_editing_mode()
            self.update()

    def init_resize(self, scene_pos: QPointF):
        self.resizing = True
        self.resize_start = scene_pos

    def init_rotation(self, scene_pos):
        self.rotating = True
        center = self.boundingRect().center()
        self.center_scene_pos = self.mapToScene(center)
        self.last_rotation_angle = math.degrees(math.atan2(
            scene_pos.y() - self.center_scene_pos.y(),
            scene_pos.x() - self.center_scene_pos.x()
        ))

    def move_item(self, local_pos: QPointF, last_local_pos: QPointF):
        delta = self.mapToParent(local_pos) - self.mapToParent(last_local_pos)
        new_pos = self.pos() + delta
        
        # Calculate the bounding rect of the rotated rectangle in scene
        # coordinates. The tight selection rect: the arc slack in
        # boundingRect() would stop the text pad-short of the page edge.
        scene_rect = self.mapToScene(self.selectionRect())
        bounding_rect = scene_rect.boundingRect()
        
        # Get constraint bounds
        parent_rect = None
        
        # Check if we're in webtoon mode by looking for the lazy webtoon manager
        scene = self.scene()
        if scene and scene.views():
            parent_rect = scene.sceneRect()
        
        # Constrain the movement
        if bounding_rect.left() + delta.x() < parent_rect.left():
            new_pos.setX(self.pos().x() - (bounding_rect.left() - parent_rect.left()))
        elif bounding_rect.right() + delta.x() > parent_rect.right():
            new_pos.setX(self.pos().x() + parent_rect.right() - bounding_rect.right())
        
        if bounding_rect.top() + delta.y() < parent_rect.top():
            new_pos.setY(self.pos().y() - (bounding_rect.top() - parent_rect.top()))
        elif bounding_rect.bottom() + delta.y() > parent_rect.bottom():
            new_pos.setY(self.pos().y() + parent_rect.bottom() - bounding_rect.bottom())
        
        self.setPos(new_pos)

    def rotate_item(self, scene_pos):
        self.setTransformOriginPoint(self.boundingRect().center())
        current_angle = math.degrees(math.atan2(
            scene_pos.y() - self.center_scene_pos.y(),
            scene_pos.x() - self.center_scene_pos.x()
        ))
        
        angle_diff = current_angle - self.last_rotation_angle
        
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        smoothed_angle = angle_diff / self.rotation_smoothing
        
        new_rotation = self.rotation() + smoothed_angle
        self.setRotation(new_rotation)
        self.last_rotation_angle = current_angle

    def resize_item(self, scene_pos: QPointF):
        if not self.resize_start:
            return

        # Calculate delta from start position in scene coordinates
        scene_start = self.resize_start
        scene_delta = scene_pos - scene_start

        # Counter-rotate the delta to align it with the item's unrotated coordinate system
        angle_rad = math.radians(-self.rotation())
        rotated_delta_x = scene_delta.x() * math.cos(angle_rad) - scene_delta.y() * math.sin(angle_rad)
        rotated_delta_y = scene_delta.x() * math.sin(angle_rad) + scene_delta.y() * math.cos(angle_rad)
        rotated_delta = QPointF(rotated_delta_x, rotated_delta_y)

        # Get the current rect and create a new one to modify. The unpadded
        # rect: curvature slack must not feed into text width / font size.
        rect = self.contentBoundingRect()
        new_rect = QRectF(rect)
        original_height = rect.height()

        # Apply the delta based on which handle is being dragged
        if self.resize_handle in ['left', 'top_left', 'bottom_left']:
            new_rect.setLeft(rect.left() + rotated_delta.x())
        if self.resize_handle in ['right', 'top_right', 'bottom_right']:
            new_rect.setRight(rect.right() + rotated_delta.x())
        if self.resize_handle in ['top', 'top_left', 'top_right']:
            new_rect.setTop(rect.top() + rotated_delta.y())
        if self.resize_handle in ['bottom', 'bottom_left', 'bottom_right']:
            new_rect.setBottom(rect.bottom() + rotated_delta.y())

        # Ensure minimum size
        min_size = 10
        if new_rect.width() < min_size:
            if 'left' in self.resize_handle: new_rect.setLeft(new_rect.right() - min_size)
            else: new_rect.setRight(new_rect.left() + min_size)
        if new_rect.height() < min_size:
            if 'top' in self.resize_handle: new_rect.setTop(new_rect.bottom() - min_size)
            else: new_rect.setBottom(new_rect.top() + min_size)

        # Determine constraint bounds
        constraint_rect = None
        scene = self.scene()
        
        if scene and scene.views():
            constraint_rect = scene.sceneRect()
        
        if constraint_rect:
            # Map the proposed new local rect to the scene to get its final footprint
            prospective_scene_rect = self.mapRectToScene(new_rect)

            # Check if the resize would push the item outside the constraint bounds
            if (prospective_scene_rect.left() < constraint_rect.left() or
                prospective_scene_rect.right() > constraint_rect.right() or
                prospective_scene_rect.top() < constraint_rect.top() or
                prospective_scene_rect.bottom() > constraint_rect.bottom()):
                return  # Abort the resize operation

        # Calculate the required shift in the parent's coordinate system.
        pos_delta = self.mapToParent(new_rect.topLeft()) - self.mapToParent(rect.topLeft())
        new_pos = self.pos() + pos_delta

        self.setPos(new_pos)

        if self.vertical:
            if self.layout:
                self.layout.set_max_size(new_rect.width(), new_rect.height())
        else: # Horizontal logic
            self.setTextWidth(new_rect.width())
            if original_height > 0:
                height_ratio = new_rect.height() / original_height
                if height_ratio > 0:
                    new_font_size = self.font_size * height_ratio
                    # Ensure minimum font size of 1pt.
                    if new_font_size >= 1:
                        self.font_size = new_font_size
                        self.set_font_size(new_font_size)
                    else:
                        # If font would become invalid, stop the resize.
                        return

        self.resize_start = scene_pos

    def _on_selection_changed(self):
        """Emit the current selection's formatting so the toolbar reflects the
        span being edited instead of always the whole-block state."""
        cursor = self.textCursor()
        properties = self.get_selected_text_properties(cursor)
        self.text_highlighted.emit(properties)

    def on_selection_changed(self):
        cursor = self.textCursor()
        properties = self.get_selected_text_properties(cursor)
        if self.editing_mode:
            self.text_highlighted.emit(properties)

    def get_selected_text_properties(self, cursor: QTextCursor):
        if not cursor.hasSelection():
            # Spacing is stored on the item, not on the document.
            return {
                'font_family': self.font_family,
                'font_size': self.font_size,
                'bold': False,
                'italic': False,
                'underline': False,
                'text_color': self.text_color.name(),
                'alignment': self.alignment,
                'outline': self.outline,
                'outline_color': self.outline_color.name() if self.outline_color else None,
                'outline_width': self.outline_width,
                'letter_spacing': float(getattr(self, "letter_spacing", 0.0) or 0.0),
                'word_spacing': float(getattr(self, "word_spacing", 0.0) or 0.0),
                'curvature': float(getattr(self, "curvature", 0.0) or 0.0),
            }

        start = cursor.selectionStart()
        end = cursor.selectionEnd()

        # Find all selections that completely contain the current selection
        containing_outlines = [
            outline for outline in self.selection_outlines
            if outline.start <= start and outline.end >= end
        ]

        # Get outline properties from the last (most recent) containing selection
        outline_properties = None
        if containing_outlines:
            latest_outline = containing_outlines[-1]  # Get the last one from the list
            outline_properties = {
                'outline': True,
                'outline_color': latest_outline.color.name(),
                'outline_width': latest_outline.width
            }
        else:
            outline_properties = {
                'outline': False,
                'outline_color': None,
                'outline_width': None
            }

        # Create a new cursor for traversing the selection
        format_cursor = QTextCursor(cursor)

        # Initialize properties with default values
        properties = {
            'font_family': set(),
            'font_size': set(),
            'bold': True,
            'italic': True,
            'underline': True,
            'text_color': set(),
            'alignment': None,
        }

        # Get initial block format for alignment
        format_cursor.setPosition(start)
        properties['alignment'] = format_cursor.blockFormat().alignment()

        # Iterate through the selection one character at a time
        for pos in range(start, end):
            format_cursor.setPosition(pos)
            format_cursor.setPosition(pos + 1, QTextCursor.KeepAnchor)
            char_format = format_cursor.charFormat()

            # Update properties
            properties['font_family'].add(char_format.font().family())
            properties['font_size'].add(char_format.fontPointSize())
            properties['bold'] &= char_format.font().bold()
            properties['italic'] &= char_format.font().italic()
            properties['underline'] &= char_format.font().underline()
            properties['text_color'].add(char_format.foreground().color().name())

        # Convert sets to single values if all elements are the same, otherwise set to None
        for key, value in properties.items():
            if isinstance(value, set):
                properties[key] = list(value)[0] if len(value) == 1 else None

        # Merge outline properties with other properties
        properties.update(outline_properties)
        # Item-level (not span-level) property: always the block's curvature.
        properties['curvature'] = float(getattr(self, "curvature", 0.0) or 0.0)

        return properties
    
    def __copy__(self):
        cls = self.__class__
        new_instance = cls(
            text=self.toHtml(),
            font_family=self.font_family,
            font_size=self.font_size,
            render_color=self.text_color,
            alignment=self.alignment,
            line_spacing=self.line_spacing,
            outline_color=self.outline_color,
            outline_width=self.outline_width,
            bold=self.bold,
            italic=self.italic,
            underline=self.underline
        )
        
        new_instance.set_text(self.toHtml(), self.contentBoundingRect().width())
        new_instance.setTransformOriginPoint(self.transformOriginPoint())
        new_instance.setPos(self.pos())
        new_instance.setRotation(self.rotation())
        new_instance.setScale(self.scale())
        new_instance.prepareGeometryChange()
        new_instance.__dict__.update(copy.copy(self.__dict__))
        return new_instance

