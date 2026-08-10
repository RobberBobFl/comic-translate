from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.text_edit import MTextEdit
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.collapse import MCollapse
from ..dayu_widgets.spin_box import MSpinBox

class LlmsPage(QtWidgets.QWidget):
    DEFAULT_EXTRA_CONTEXT_LIMIT = 1000
    DEFAULT_BATCH_SIZE = 5
    DEFAULT_CONTEXT_WINDOW = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._extra_context_limit: int | None = self.DEFAULT_EXTRA_CONTEXT_LIMIT

        v = QtWidgets.QVBoxLayout(self)
        main_layout = QtWidgets.QHBoxLayout()

        self.image_checkbox = MCheckBox(self.tr("Provide Image as Input to AI"))
        self.image_checkbox.setChecked(False)

        # Left
        left_layout = QtWidgets.QVBoxLayout()
        prompt_label = MLabel(self.tr("Extra Context:"))
        self.extra_context = MTextEdit()
        self.extra_context.setMinimumHeight(200)
        left_layout.addWidget(prompt_label)
        left_layout.addWidget(self.extra_context)
        left_layout.addWidget(self.image_checkbox)
        left_layout.addSpacing(15)
        system_label = MLabel(self.tr("System Prompt:"))
        self.system_prompt = MTextEdit()
        self.system_prompt.setMinimumHeight(150)
        left_layout.addWidget(system_label)
        left_layout.addWidget(self.system_prompt)
        self.save_system_prompt_checkbox = MCheckBox(self.tr("Save System Prompt"))
        self.save_system_prompt_checkbox.setChecked(True)
        left_layout.addWidget(self.save_system_prompt_checkbox)
        left_layout.addStretch(1)

        # Right
        right_layout = QtWidgets.QVBoxLayout()

        # Advanced settings

        # Batching and sliding context only apply to the batch and semi-auto
        # runs; a manual page translation is always sent as a single request.
        batch_label = MLabel(self.tr("Batch and Context (batch mode only)")).h4()
        right_layout.addWidget(batch_label)

        batch_size_layout = QtWidgets.QHBoxLayout()
        batch_size_label = MLabel(self.tr("Blocks per Request:"))
        self.batch_size_spinbox = MSpinBox().small()
        self.batch_size_spinbox.setFixedWidth(70)
        self.batch_size_spinbox.setMaximum(100)
        self.batch_size_spinbox.setValue(self.DEFAULT_BATCH_SIZE)
        self.batch_size_spinbox.setToolTip(
            self.tr("0 sends the whole page in one request.")
        )
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(self.batch_size_spinbox)
        batch_size_layout.addStretch()
        right_layout.addLayout(batch_size_layout)

        context_window_layout = QtWidgets.QHBoxLayout()
        context_window_label = MLabel(self.tr("Context Window:"))
        self.context_window_spinbox = MSpinBox().small()
        self.context_window_spinbox.setFixedWidth(70)
        self.context_window_spinbox.setMaximum(100)
        self.context_window_spinbox.setValue(self.DEFAULT_CONTEXT_WINDOW)
        self.context_window_spinbox.setToolTip(
            self.tr(
                "How many previously translated lines are sent along for "
                "consistency. 0 turns the sliding context off."
            )
        )
        context_window_layout.addWidget(context_window_label)
        context_window_layout.addWidget(self.context_window_spinbox)
        context_window_layout.addStretch()
        right_layout.addLayout(context_window_layout)

        batch_hint = MLabel(
            self.tr(
                "Smaller requests keep reasoning models from truncating their "
                "answer, and the context window keeps names and tone "
                "consistent across pages."
            )
        )
        batch_hint.setWordWrap(True)
        right_layout.addWidget(batch_hint)

        right_layout.addSpacing(10)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        v.addLayout(main_layout)
        v.addStretch(1)

        self.extra_context.textChanged.connect(self._limit_extra_context)

    def set_extra_context_unlimited(self, enabled: bool) -> None:
        self._extra_context_limit = None if enabled else self.DEFAULT_EXTRA_CONTEXT_LIMIT
        self._limit_extra_context()

    def _limit_extra_context(self):
        max_length = self._extra_context_limit
        if max_length is None:
            return
        text = self.extra_context.toPlainText()
        if len(text) > max_length:
            # Preserve cursor position
            cursor = self.extra_context.textCursor()
            position = cursor.position()
            
            # Truncate
            self.extra_context.setPlainText(text[:max_length])
            
            # Restore cursor (clamped to end)
            new_position = min(position, max_length)
            cursor.setPosition(new_position)
            self.extra_context.setTextCursor(cursor)

