import os

from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.text_edit import MTextEdit
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.collapse import MCollapse
from ..dayu_widgets.spin_box import MSpinBox
from ..dayu_widgets.push_button import MPushButton

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
        self.use_scene_description_checkbox = MCheckBox(
            self.tr("Use Scene Description")
        )
        left_layout.addWidget(self.use_scene_description_checkbox)

        self.scene_descriptions_group = QtWidgets.QGroupBox(
            self.tr("Scene Descriptions")
        )
        scene_layout = QtWidgets.QVBoxLayout(self.scene_descriptions_group)
        scene_hint = MLabel(
            self.tr("Descriptions are generated in English and can be edited per page.")
        ).secondary()
        scene_hint.setWordWrap(True)
        scene_layout.addWidget(scene_hint)
        self.scene_descriptions_scroll = QtWidgets.QScrollArea()
        self.scene_descriptions_scroll.setWidgetResizable(True)
        self.scene_descriptions_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scene_descriptions_widget = QtWidgets.QWidget()
        self.scene_descriptions_layout = QtWidgets.QVBoxLayout(
            self.scene_descriptions_widget
        )
        self.scene_descriptions_layout.addStretch(1)
        self.scene_descriptions_scroll.setWidget(self.scene_descriptions_widget)
        scene_layout.addWidget(self.scene_descriptions_scroll)
        self.apply_scene_descriptions_button = MPushButton(self.tr("Apply"))
        self.apply_scene_descriptions_button.set_dayu_type(MPushButton.PrimaryType)
        self.apply_scene_descriptions_button.clicked.connect(
            self.apply_scene_descriptions
        )
        scene_layout.addWidget(
            self.apply_scene_descriptions_button,
            0,
            QtCore.Qt.AlignmentFlag.AlignRight,
        )
        left_layout.addWidget(self.scene_descriptions_group)
        left_layout.addStretch(1)
        self._scene_description_edits = {}

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
                "Blocks per Request splits a page into several smaller "
                "requests. Context Window carries the last translated lines "
                "into the next request. Both apply to batch and multi-page "
                "runs; manual translation always sends one request per page."
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

    def refresh_scene_descriptions(self) -> None:
        """Rebuild per-page editors from the current project state."""
        main = self.window()
        while self.scene_descriptions_layout.count() > 1:
            item = self.scene_descriptions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._scene_description_edits = {}
        image_files = list(getattr(main, "image_files", [])) if main is not None else []
        image_states = getattr(main, "image_states", {}) if main is not None else {}
        for index, file_path in enumerate(image_files, 1):
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QVBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.addWidget(MLabel(f"Page {index} - {os.path.basename(file_path)}"))
            edit = MTextEdit()
            edit.setMinimumHeight(65)
            edit.setPlainText(
                image_states.get(file_path, {}).get("scene_description", "")
            )
            row_layout.addWidget(edit)
            self.scene_descriptions_layout.insertWidget(
                self.scene_descriptions_layout.count() - 1, row
            )
            self._scene_description_edits[file_path] = edit
        self.scene_descriptions_group.setVisible(bool(image_files))

    def apply_scene_descriptions(self) -> None:
        main = self.window()
        if main is None:
            return
        changed = False
        for file_path, edit in self._scene_description_edits.items():
            state = main.image_states.setdefault(file_path, {})
            value = edit.toPlainText().strip()
            if state.get("scene_description", "") != value:
                state["scene_description"] = value
                changed = True
        if changed and hasattr(main, "mark_project_dirty"):
            main.mark_project_dirty()

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

