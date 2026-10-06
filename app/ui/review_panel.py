from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from app.ui.dayu_widgets.combo_box import MComboBox
from app.ui.dayu_widgets.push_button import MPushButton
from app.ui.dayu_widgets.tool_button import MToolButton
from app.ui.settings.utils import set_combo_box_width


CATEGORY_LABELS = {
    "calque": "Calque",
    "mistranslation": "Mistranslation",
    "terminology": "Terminology",
    "style": "Style/Tone",
    "scene_context": "Scene Context",
    "other": "Other",
}

SEVERITY_STYLES = {
    "error": (
        "#fde8e8",
        "#b42318",
    ),
    "improvement": (
        "#e8f0fd",
        "#1d4ed8",
    ),
}

# A stylesheet with any background resets the inherited text color in Qt, so
# the recommended variant must restate it explicitly or it becomes unreadable
# in the dark theme. Applied to the variant *label* (the radio indicator
# itself is drawn by the theme).
RECOMMENDED_LABEL_STYLE = (
    "VariantLabel {"
    " background: rgba(76, 175, 80, 0.22);"
    " border: 1px solid rgba(76, 175, 80, 0.45);"
    " border-radius: 4px; padding: 2px 6px;"
    " color: #d9f2dc; font-weight: bold; }"
)


def category_label(key: str) -> str:
    return CATEGORY_LABELS.get(key, CATEGORY_LABELS["other"])


class VariantLabel(QtWidgets.QLabel):
    """Selectable variant text that toggles its radio on a plain click.

    A click without dragging (and without an existing selection) picks the
    variant; dragging selects the text for copying instead.
    """

    clicked = QtCore.Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._press_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._press_pos is not None and event.button() == Qt.MouseButton.LeftButton:
            moved = (event.position() - self._press_pos).manhattanLength() > 5
            if not moved and not self.hasSelectedText():
                self.clicked.emit()
        self._press_pos = None
        super().mouseReleaseEvent(event)


class ReviewEntryWidget(QtWidgets.QFrame):
    """One reviewer suggestion: reason, original, current text and variants."""

    apply_requested = QtCore.Signal(int, str)      # entry_id, chosen text
    dismiss_requested = QtCore.Signal(int)         # entry_id
    focus_requested = QtCore.Signal(int)           # entry_id

    def __init__(self, entry_id: int, entry: dict, parent=None):
        super().__init__(parent)
        self.entry_id = entry_id
        self.entry = entry
        self._stale = False

        self.setObjectName("reviewEntry")
        self.setStyleSheet(
            "#reviewEntry { border: 1px solid rgba(127, 127, 127, 0.35);"
            " border-radius: 6px; }"
            "#reviewEntry QLabel { background: transparent; }"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header: block number + severity badge + category.
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(self.tr("Block {0}").format(entry_id + 1))
        title.setStyleSheet("font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        severity = str(entry.get("severity", "improvement"))
        bg, fg = SEVERITY_STYLES.get(severity, SEVERITY_STYLES["improvement"])
        severity_tag = QtWidgets.QLabel(
            self.tr("ERROR") if severity == "error" else self.tr("IMPROVEMENT")
        )
        severity_tag.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 4px;"
            " padding: 1px 6px; font-weight: bold; font-size: 10px;"
        )
        header.addWidget(severity_tag)
        category_tag = QtWidgets.QLabel(category_label(str(entry.get("category", "other"))))
        category_tag.setStyleSheet(
            "background: rgba(127, 127, 127, 0.2); border-radius: 4px;"
            " padding: 1px 6px; font-size: 10px;"
        )
        header.addWidget(category_tag)
        layout.addLayout(header)

        reason = str(entry.get("reason", "")).strip()
        if reason:
            reason_label = QtWidgets.QLabel(reason)
            reason_label.setWordWrap(True)
            reason_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            reason_label.setStyleSheet("color: #999999;")
            layout.addWidget(reason_label)

        def _text_box(text: str, bold: bool = False, bg: str | None = None):
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            style = ""
            if bold:
                style += "font-weight: bold;"
            if bg:
                style += f"background: {bg}; border-radius: 4px; padding: 4px;"
            if style:
                label.setStyleSheet(style)
            return label

        source = str(entry.get("source_text", ""))
        layout.addWidget(_text_box(source))
        current = str(entry.get("translation_snapshot", ""))
        self._current_label = _text_box(current)
        layout.addWidget(self._current_label)

        # Variants: recommended + alternatives as a radio group (nothing
        # selected by default - a card is only applied on explicit choice).
        self._radio_group = QtWidgets.QButtonGroup(self)
        self._radio_group.setExclusive(True)
        self._variant_rows: list = []
        variants_box = QtWidgets.QVBoxLayout()
        variants_box.setSpacing(2)
        recommended = str(entry.get("recommended", "")).strip()
        self._add_variant(variants_box, recommended, style=RECOMMENDED_LABEL_STYLE)
        for alt in entry.get("alternatives", []) or []:
            self._add_variant(variants_box, str(alt))
        layout.addLayout(variants_box)

        # Actions
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch()
        self.dismiss_btn = MPushButton(self.tr("Dismiss")).small()
        self.dismiss_btn.clicked.connect(lambda: self.dismiss_requested.emit(self.entry_id))
        actions.addWidget(self.dismiss_btn)
        self.replace_btn = MPushButton(self.tr("Replace")).small()
        self.replace_btn.set_dayu_type(MPushButton.PrimaryType)
        self.replace_btn.setEnabled(False)
        self._radio_group.buttonToggled.connect(self._update_replace_enabled)
        self.replace_btn.clicked.connect(self._emit_apply)
        actions.addWidget(self.replace_btn)
        layout.addLayout(actions)

        self._focus_hint_installed = False

    # -- helpers --------------------------------------------------------

    def _add_variant(self, layout: QtWidgets.QVBoxLayout, text: str, style: str | None = None):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        radio = QtWidgets.QRadioButton()
        radio.setCursor(Qt.CursorShape.PointingHandCursor)
        radio.setProperty("variant_text", text)
        label = VariantLabel(text)
        if style:
            label.setStyleSheet(style)
        label.clicked.connect(lambda: radio.setChecked(True))
        self._radio_group.addButton(radio)
        self._variant_rows.append((radio, label))
        row.addWidget(radio, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label, 1)
        layout.addLayout(row)

    def _update_replace_enabled(self, *_args):
        self.replace_btn.setEnabled(self.selected_variant() is not None)

    def _emit_apply(self):
        text = self.selected_variant()
        if text is not None:
            self.apply_requested.emit(self.entry_id, text)

    def selected_variant(self) -> str | None:
        radio = self._radio_group.checkedButton()
        if radio is None:
            return None
        return str(radio.property("variant_text") or radio.text())

    def has_selection(self) -> str | None:
        return self.selected_variant()

    def mousePressEvent(self, event):  # click anywhere -> show the bubble
        if event.button() == Qt.MouseButton.LeftButton:
            self.focus_requested.emit(self.entry_id)
        super().mousePressEvent(event)

    def set_stale(self, stale: bool):
        """Mark the entry as outdated (the translation changed after review)."""
        self._stale = stale
        if stale:
            self._current_label.setStyleSheet(
                "background: rgba(127, 127, 127, 0.2); border-radius: 4px;"
                " padding: 4px; color: #999999; text-decoration: line-through;"
            )
            self.replace_btn.setEnabled(False)
            self.replace_btn.setToolTip(self.tr("Translation changed after the review"))
            self.dismiss_btn.setToolTip(self.tr("Remove this outdated suggestion"))
        for radio, label in self._variant_rows:
            radio.setEnabled(not stale)
            label.setEnabled(not stale)

    @property
    def is_stale(self) -> bool:
        return self._stale


class ReviewPanel(QtWidgets.QWidget):
    """Widget listing the reviewer's suggestions for the displayed page.

    Used inside :class:`ReviewDialog`; also embeddable standalone (the header
    with the title/close button can be hidden via :meth:`set_header_visible`).
    """

    apply_entries_requested = QtCore.Signal(list)   # [(entry_id, text), ...]
    dismiss_requested = QtCore.Signal(int)          # entry_id (relayed to the controller)
    block_focus_requested = QtCore.Signal(int)      # entry_id
    close_requested = QtCore.Signal()

    def __init__(self, parent=None, show_header: bool = True):
        super().__init__(parent)
        self._entries: dict[int, ReviewEntryWidget] = {}
        self._build_ui()
        self.set_header_visible(show_header)

    def set_header_visible(self, visible: bool):
        """Hide the title/close row when the panel lives in its own window."""
        self.header_container.setVisible(visible)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header_container = QtWidgets.QWidget()
        header_row = QtWidgets.QHBoxLayout(header_container)
        header_row.setContentsMargins(6, 6, 6, 0)
        title_lbl = QtWidgets.QLabel(self.tr("Translation Review"))
        title_lbl.setStyleSheet("font-weight: bold; color: #BBBBBB;")
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        self.clear_btn = MToolButton().icon_only().svg("close_line.svg").small()
        self.clear_btn.setToolTip(self.tr("Close"))
        self.clear_btn.clicked.connect(self.close_requested)
        header_row.addWidget(self.clear_btn)
        layout.addWidget(header_container)
        self.header_container = header_container

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setContentsMargins(6, 0, 6, 0)
        self.severity_combo = MComboBox().small()
        self.severity_combo.addItem(self.tr("All"), userData="all")
        self.severity_combo.addItem(self.tr("Errors"), userData="error")
        self.severity_combo.addItem(self.tr("Improvements"), userData="improvement")
        self.severity_combo.setToolTip(self.tr("Filter by severity"))
        self.severity_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.severity_combo)

        self.category_combo = MComboBox().small()
        self.category_combo.addItem(self.tr("All categories"), userData="all")
        for key, label in CATEGORY_LABELS.items():
            self.category_combo.addItem(self.tr(label), userData=key)
        self.category_combo.setToolTip(self.tr("Filter by category"))
        self.category_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.category_combo)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self.summary_label = QtWidgets.QLabel(self.tr("No suggestions"))
        self.summary_label.setStyleSheet("color: #999999;")
        self.summary_label.setContentsMargins(6, 2, 6, 0)
        layout.addWidget(self.summary_label)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.entries_host = QtWidgets.QWidget()
        self.entries_layout = QtWidgets.QVBoxLayout(self.entries_host)
        self.entries_layout.setContentsMargins(6, 4, 6, 4)
        self.entries_layout.setSpacing(6)
        self.entries_layout.addStretch(1)
        self.scroll.setWidget(self.entries_host)
        layout.addWidget(self.scroll, 1)

        footer_row = QtWidgets.QHBoxLayout()
        footer_row.setContentsMargins(6, 0, 6, 6)
        self.apply_all_btn = MPushButton(self.tr("Replace Selected")).small()
        self.apply_all_btn.setEnabled(False)
        self.apply_all_btn.clicked.connect(self._emit_apply_selected)
        footer_row.addWidget(self.apply_all_btn)
        footer_row.addStretch()
        layout.addLayout(footer_row)

        set_combo_box_width(
            self.severity_combo,
            [self.severity_combo.itemText(i) for i in range(self.severity_combo.count())],
            padding=40,
        )
        set_combo_box_width(
            self.category_combo,
            [self.category_combo.itemText(i) for i in range(self.category_combo.count())],
            padding=40,
        )

    # -- public API ------------------------------------------------------

    def set_entries(self, entries: dict[int, dict], stale_ids: set[int] | None = None):
        """Rebuild the card list. ``entries``: {block_index: entry_dict}."""
        stale_ids = stale_ids or set()
        for widget in self._entries.values():
            widget.deleteLater()
        self._entries = {}
        for entry_id, entry in sorted(entries.items()):
            card = ReviewEntryWidget(entry_id, entry)
            card.apply_requested.connect(self._on_card_apply)
            card.dismiss_requested.connect(self.dismiss_requested)
            card.focus_requested.connect(self.block_focus_requested)
            card.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            card.set_stale(entry_id in stale_ids)
            self.entries_layout.insertWidget(self.entries_layout.count() - 1, card)
            self._entries[entry_id] = card
        self._apply_filters()

    def remove_entries(self, entry_ids: list[int]):
        for entry_id in entry_ids:
            widget = self._entries.pop(entry_id, None)
            if widget is not None:
                widget.deleteLater()
        self._apply_filters()

    def entries(self) -> dict[int, ReviewEntryWidget]:
        return dict(self._entries)

    def is_empty(self) -> bool:
        return not self._entries

    # -- internals -------------------------------------------------------

    def _on_card_apply(self, entry_id: int, text: str):
        self.apply_entries_requested.emit([(entry_id, text)])

    def _emit_apply_selected(self):
        selections = []
        for entry_id, card in self._entries.items():
            text = card.selected_variant()
            if text is not None and not card.is_stale:
                selections.append((entry_id, text))
        if selections:
            self.apply_entries_requested.emit(selections)

    def _apply_filters(self):
        severity = self.severity_combo.currentData()
        category = self.category_combo.currentData()
        shown = 0
        errors = 0
        improvements = 0
        for entry_id, card in self._entries.items():
            entry = card.entry
            card_severity = str(entry.get("severity", "improvement"))
            card_category = str(entry.get("category", "other"))
            if card_severity == "error":
                errors += 1
            else:
                improvements += 1
            visible = severity in (None, "all", card_severity) and category in (
                None,
                "all",
                card_category,
            )
            card.setVisible(visible)
            if visible:
                shown += 1
        total = len(self._entries)
        if total == 0:
            self.summary_label.setText(self.tr("No suggestions"))
        else:
            self.summary_label.setText(
                self.tr("{shown}/{total} shown - {errors} errors, {improvements} improvements").format(
                    shown=shown, total=total, errors=errors, improvements=improvements
                )
            )
        self.apply_all_btn.setEnabled(
            any(
                card.selected_variant() is not None and not card.is_stale
                for card in self._entries.values()
            )
        )


class ReviewDialog(QtWidgets.QDialog):
    """Modeless window hosting the review panel.

    A separate window (with the normal system frame) gives the suggestion
    cards room to breathe: it can be resized, moved to a second monitor or
    parked next to the main window. Non-modal by design - the user keeps
    scrolling the comic while applying suggestions.
    """

    rerun_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Translation Review"))
        # A plain top-level window (title bar, resize, min/max buttons) even
        # though the parent main window is frameless.
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setModal(False)
        self.setMinimumSize(420, 380)
        self.resize(880, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        page_row = QtWidgets.QHBoxLayout()
        self.page_label = QtWidgets.QLabel("")
        self.page_label.setStyleSheet("color: #999999;")
        self.page_label.setContentsMargins(2, 0, 2, 0)
        self.page_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        page_row.addWidget(self.page_label, 1)
        self.rerun_button = MPushButton(self.tr("Re-run Review")).small()
        self.rerun_button.setToolTip(
            self.tr("Check the displayed page again (overwrites these suggestions).")
        )
        self.rerun_button.clicked.connect(self.rerun_requested)
        page_row.addWidget(self.rerun_button)
        layout.addLayout(page_row)

        self.review_panel = ReviewPanel(self, show_header=False)
        layout.addWidget(self.review_panel, 1)
