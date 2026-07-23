from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.check_box import MCheckBox
from .utils import set_label_width

class CredentialsPage(QtWidgets.QWidget):
    def __init__(self, services: list[str], value_mappings: dict[str, str], parent=None):
        super().__init__(parent)
        self.services = services
        self.value_mappings = value_mappings
        self.credential_widgets: dict[str, QtWidgets.QComboBox | MLineEdit] = {}

        # main layout (no internal scroll here — outer settings scroll handles it)
        main_layout = QtWidgets.QVBoxLayout(self)
        content_layout = QtWidgets.QVBoxLayout()

        self.save_keys_checkbox = MCheckBox(self.tr("Save Keys"))

        info_label = MLabel(self.tr(
            "These settings are for advanced users who wish to use their own Custom API endpoints (e.g. Local Language Models) for translation. "
            "For most users, no configuration is needed here."
        )).secondary()
        info_label.setWordWrap(True)
        
        content_layout.addWidget(info_label)
        content_layout.addSpacing(10)
        content_layout.addWidget(self.save_keys_checkbox)
        content_layout.addSpacing(20)

        for service_label in self.services:
            service_layout = QtWidgets.QVBoxLayout()
            service_header = MLabel(service_label).strong()
            service_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            service_layout.addWidget(service_header)

            normalized = self.value_mappings.get(service_label, service_label)

            if normalized == "Microsoft Azure":
                # OCR
                ocr_label = MLabel(self.tr("OCR")).secondary()
                service_layout.addWidget(ocr_label)

                ocr_api_key_input = MLineEdit()
                ocr_api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
                ocr_api_key_input.setFixedWidth(400)
                ocr_api_key_prefix = MLabel(self.tr("API Key")).border()
                set_label_width(ocr_api_key_prefix)
                ocr_api_key_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                ocr_api_key_input.set_prefix_widget(ocr_api_key_prefix)
                service_layout.addWidget(ocr_api_key_input)
                self.credential_widgets["Microsoft Azure_api_key_ocr"] = ocr_api_key_input

                endpoint_input = MLineEdit()
                endpoint_input.setFixedWidth(400)
                endpoint_prefix = MLabel(self.tr("Endpoint URL")).border()
                set_label_width(endpoint_prefix)
                endpoint_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                endpoint_input.set_prefix_widget(endpoint_prefix)
                service_layout.addWidget(endpoint_input)
                self.credential_widgets["Microsoft Azure_endpoint"] = endpoint_input

                # Translator
                # # Translator
                # translate_label = MLabel(self.tr("Translate")).secondary()
                # service_layout.addWidget(translate_label)

                # translator_api_key_input = MLineEdit()
                # translator_api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
                # translator_api_key_input.setFixedWidth(400)
                # translator_api_key_prefix = MLabel(self.tr("API Key")).border()
                # set_label_width(translator_api_key_prefix)
                # translator_api_key_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                # translator_api_key_input.set_prefix_widget(translator_api_key_prefix)
                # service_layout.addWidget(translator_api_key_input)
                # self.credential_widgets["Microsoft Azure_api_key_translator"] = translator_api_key_input

                # region_input = MLineEdit()
                # region_input.setFixedWidth(400)
                # region_prefix = MLabel(self.tr("Region")).border()
                # set_label_width(region_prefix)
                # region_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                # region_input.set_prefix_widget(region_prefix)
                # service_layout.addWidget(region_input)
                # self.credential_widgets["Microsoft Azure_region"] = region_input

            elif normalized == "Custom":
                api_key_input = MLineEdit()
                api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
                api_key_input.setFixedWidth(400)
                api_key_prefix = MLabel(self.tr("API Key")).border()
                set_label_width(api_key_prefix)
                api_key_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                api_key_input.set_prefix_widget(api_key_prefix)
                service_layout.addWidget(api_key_input)
                self.credential_widgets[f"{normalized}_api_key"] = api_key_input

                endpoint_input = MLineEdit()
                endpoint_input.setFixedWidth(400)
                endpoint_prefix = MLabel(self.tr("Endpoint URL")).border()
                set_label_width(endpoint_prefix)
                endpoint_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                endpoint_input.set_prefix_widget(endpoint_prefix)
                service_layout.addWidget(endpoint_input)
                self.credential_widgets[f"{normalized}_api_url"] = endpoint_input

                model_container = QtWidgets.QWidget()
                model_container.setFixedWidth(400)
                model_container_layout = QtWidgets.QHBoxLayout(model_container)
                model_container_layout.setContentsMargins(0, 0, 0, 0)
                model_container_layout.setSpacing(0)
                model_prefix = MLabel(self.tr("Model")).border()
                set_label_width(model_prefix)
                model_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                model_container_layout.addWidget(model_prefix)
                model_combo = QtWidgets.QComboBox()
                model_combo.setEditable(True)
                model_combo.setMinimumContentsLength(30)
                model_combo.setPlaceholderText(self.tr("Select or type model..."))
                model_container_layout.addWidget(model_combo, 1)
                service_layout.addWidget(model_container)
                self.credential_widgets[f"{normalized}_model"] = model_combo

                load_btn = QtWidgets.QPushButton(self.tr("Load Models"))
                load_btn.setFixedWidth(120)
                service_layout.addWidget(load_btn)

                load_btn.clicked.connect(
                    lambda checked, ak=api_key_input, ep=endpoint_input, mc=model_combo, lb=load_btn:
                    self._fetch_models(ak, ep, mc, lb)
                )

            elif normalized == "Yandex":
                api_key_input = MLineEdit()
                api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
                api_key_input.setFixedWidth(400)
                api_key_prefix = MLabel(self.tr("Secret Key")).border()
                set_label_width(api_key_prefix)
                api_key_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                api_key_input.set_prefix_widget(api_key_prefix)
                service_layout.addWidget(api_key_input)
                self.credential_widgets[f"{normalized}_api_key"] = api_key_input

                folder_id_input = MLineEdit()
                folder_id_input.setFixedWidth(400)
                folder_id_prefix = MLabel(self.tr("Folder ID")).border()
                set_label_width(folder_id_prefix)
                folder_id_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                folder_id_input.set_prefix_widget(folder_id_prefix)
                service_layout.addWidget(folder_id_input)
                self.credential_widgets[f"{normalized}_folder_id"] = folder_id_input

            else:
                api_key_input = MLineEdit()
                api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
                api_key_input.setFixedWidth(400)
                api_key_prefix = MLabel(self.tr("API Key")).border()
                set_label_width(api_key_prefix)
                api_key_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                api_key_input.set_prefix_widget(api_key_prefix)
                service_layout.addWidget(api_key_input)
                self.credential_widgets[f"{normalized}_api_key"] = api_key_input

            content_layout.addLayout(service_layout)
            content_layout.addSpacing(20)

        content_layout.addStretch(1)
        main_layout.addLayout(content_layout)

    def _fetch_models(self, api_key_input, endpoint_input, model_combo, load_btn):
        """Fetch available models from the configured API endpoint.
        Tries multiple URL patterns: OpenAI (/models, /v1/models),
        Ollama (/api/tags), and the raw URL itself."""
        import requests
        from ..dayu_widgets.message import MMessage

        api_url = endpoint_input.text().strip().rstrip('/')
        api_key = api_key_input.text().strip()

        if not api_url:
            MMessage.warning(
                self.tr("Please enter an Endpoint URL first."),
                self
            )
            return

        load_btn.setEnabled(False)
        load_btn.setText(self.tr("Loading..."))
        QtWidgets.QApplication.processEvents()

        models = []
        last_error = ""

        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            # Список URL-паттернов для поиска моделей
            url_patterns = [
                f"{api_url}/models",          # OpenAI standard
                f"{api_url}/v1/models",        # OpenAI with /v1
                f"{api_url}/api/tags",         # Ollama
            ]

            # Если URL сам выглядит как полный эндпоинт (содержит /chat/completions и т.п.),
            # пробуем подняться на уровень выше
            import re
            if re.search(r'/(chat/completions|completions|generate)$', api_url):
                parent_url = api_url.rsplit('/', 2)[0]
                url_patterns.insert(0, f"{parent_url}/models")
                url_patterns.insert(1, f"{parent_url}/v1/models")

            for url in url_patterns:
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        # OpenAI format: {"data": [{"id": "model-name", ...}]}
                        if "data" in data and isinstance(data["data"], list):
                            models = [m["id"] for m in data["data"]]
                            break
                        # Ollama format: {"models": [{"name": "model-name", ...}]}
                        if "models" in data and isinstance(data["models"], list):
                            models = [m["name"] for m in data["models"]]
                            break
                    else:
                        last_error = f"HTTP {resp.status_code} at {url}"
                except requests.exceptions.RequestException as e:
                    last_error = f"{str(e)} at {url}"
                    continue

            if models:
                current_text = model_combo.currentText().strip()
                model_combo.clear()
                model_combo.addItems(sorted(models))
                # Добавляем автодополнение с фильтрацией по подстроке
                completer = QtWidgets.QCompleter(sorted(models), model_combo)
                completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
                completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
                model_combo.setCompleter(completer)
                idx = model_combo.findText(current_text)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
                elif current_text:
                    model_combo.setEditText(current_text)
                MMessage.success(
                    self.tr("Found {count} models.").format(count=len(models)),
                    self
                )
            else:
                MMessage.warning(
                    self.tr("Could not fetch models.\n{error}").format(error=last_error),
                    self
                )

        except Exception as e:
            MMessage.error(
                self.tr("Failed to load models: {error}").format(error=str(e)),
                self
            )

        finally:
            load_btn.setEnabled(True)
            load_btn.setText(self.tr("Load Models"))
