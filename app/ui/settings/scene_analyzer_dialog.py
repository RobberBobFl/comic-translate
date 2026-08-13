import requests
from PySide6 import QtCore, QtWidgets

from modules.translation.scene_analyzer import SceneAnalyzer
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.push_button import MPushButton
from .utils import set_label_width


class SceneAnalyzerDialog(QtWidgets.QDialog):
    """Configure the OpenAI-compatible vision provider used for scene context."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(self.tr("Scene Description Model"))
        self.setMinimumWidth(520)

        layout = QtWidgets.QVBoxLayout(self)
        info = MLabel(self.tr(
            "Connect an OpenAI-compatible vision API. The model is used only to "
            "describe comic page scenes and is separate from OCR and translation."
        )).secondary()
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addSpacing(15)

        self.url_input = self._add_line_input(layout, self.tr("API URL"))
        self.key_input = self._add_line_input(layout, self.tr("API Key"), password=True)

        model_layout = QtWidgets.QHBoxLayout()
        model_label = MLabel(self.tr("Model")).border()
        set_label_width(model_label)
        model_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.model_combo = MComboBox().small()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setFixedWidth(300)
        self.load_models_button = MPushButton(self.tr("Load Models"))
        self.load_models_button.clicked.connect(self._load_models)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_models_button)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        hint = MLabel(self.tr("Choose a model that supports image (vision) input.")).secondary()
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addSpacing(15)

        self.save_checkbox = MCheckBox(self.tr("Save API Key"))
        layout.addWidget(self.save_checkbox)
        layout.addStretch(1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = MPushButton(self.tr("Cancel"))
        cancel.clicked.connect(self.reject)
        save = MPushButton(self.tr("Save"))
        save.set_dayu_type(MPushButton.PrimaryType)
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

        self._load()

    def _add_line_input(self, layout, title: str, password: bool = False):
        row = QtWidgets.QHBoxLayout()
        label = MLabel(title).border()
        set_label_width(label)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        widget = MLineEdit()
        widget.setFixedWidth(380)
        widget.set_prefix_widget(label)
        if password:
            widget.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        row.addWidget(widget)
        row.addStretch()
        layout.addLayout(row)
        layout.addSpacing(10)
        return widget

    def _load(self):
        credentials = self.settings.get_scene_analyzer_credentials()
        self.url_input.setText(credentials.get("api_url") or SceneAnalyzer.DEFAULT_API_URL)
        self.key_input.setText(credentials.get("api_key") or "")
        model = credentials.get("model") or ""
        if model:
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(model)
        self.save_checkbox.setChecked(bool(credentials.get("save_key", False)))

    def _load_models(self):
        api_url = self.url_input.text().strip()
        if not api_url:
            return
        base = api_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[:-len("/chat/completions")]
        headers = {}
        if self.key_input.text():
            headers["Authorization"] = f"Bearer {self.key_input.text()}"
        try:
            response = requests.get(f"{base}/models", headers=headers, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Unable to Load Models"),
                self.tr("Could not load models from the configured endpoint:\n{error}").format(
                    error=str(exc)
                ),
            )
            return

        items = payload.get("data") or payload.get("models") or []
        model_ids = []
        for item in items:
            if isinstance(item, dict):
                model_ids.append(item.get("id") or item.get("name"))
            elif isinstance(item, str):
                model_ids.append(item)
        model_ids = [model_id for model_id in model_ids if model_id]
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(model_ids)
        self.model_combo.setCurrentText(current)

    def accept(self):
        self.settings.set_scene_analyzer_credentials({
            "api_url": self.url_input.text().strip(),
            "api_key": self.key_input.text(),
            "model": self.model_combo.currentText().strip(),
            "save_key": self.save_checkbox.isChecked(),
        })
        super().accept()
