from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.check_box import MCheckBox

class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        batch_label = MLabel(self.tr("Automatic Mode")).h4()
        batch_note = MLabel(
            self.tr(
                "Selected exports are saved to comic_translate_<timestamp> in the same directory as the input file/archive."
            )
        ).secondary()
        batch_note.setWordWrap(True)
        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))

        layout.addWidget(batch_label)
        layout.addWidget(batch_note)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)

        # Text JSON export/import section
        text_json_label = MLabel(self.tr("Text JSON")).h4()
        text_json_note = MLabel(
            self.tr(
                "Export and import translations as an editable JSON file. "
                "Import updates only translations without changing block positions or other data."
            )
        ).secondary()
        text_json_note.setWordWrap(True)
        self.export_text_button = QtWidgets.QPushButton(self.tr("Export Text (JSON)"))
        self.import_text_button = QtWidgets.QPushButton(self.tr("Import Text (JSON)"))
        self.current_page_only_checkbox = MCheckBox(self.tr("Current page only"))

        layout.addWidget(text_json_label)
        layout.addWidget(text_json_note)
        layout.addWidget(self.export_text_button)
        layout.addWidget(self.import_text_button)
        layout.addWidget(self.current_page_only_checkbox)

        # Text TXT export section
        text_txt_label = MLabel(self.tr("Text TXT")).h4()
        text_txt_note = MLabel(
            self.tr(
                "Export translations as a plain text file with original and translation per block."
            )
        ).secondary()
        text_txt_note.setWordWrap(True)
        self.export_text_button_txt = QtWidgets.QPushButton(self.tr("Export Text (TXT)"))

        layout.addWidget(text_txt_label)
        layout.addWidget(text_txt_note)
        layout.addWidget(self.export_text_button_txt)

        layout.addStretch(1)
