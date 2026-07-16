from PySide6 import QtWidgets
from PySide6.QtCore import QSettings
from .utils import create_title_and_combo, set_combo_box_width


class PersonalizationPage(QtWidgets.QWidget):
    def __init__(self, languages: list[str], themes: list[str], parent=None):
        super().__init__(parent)
        self.languages = languages
        self.themes = themes

        layout = QtWidgets.QVBoxLayout(self)

        language_widget, self.lang_combo = create_title_and_combo(self.tr("Language"), self.languages)
        set_combo_box_width(self.lang_combo, self.languages)
        theme_widget, self.theme_combo = create_title_and_combo(self.tr("Theme"), self.themes)
        set_combo_box_width(self.theme_combo, self.themes)

        layout.addWidget(language_widget)
        layout.addWidget(theme_widget)

        self.stitch_webtoon_cb = QtWidgets.QCheckBox(
            self.tr("Webtoon: stitch pages into one image (experimental)")
        )
        self.stitch_webtoon_cb.setChecked(
            bool(QSettings("ComicLabs", "ComicTranslate").value("webtoon_stitch_mode", True, type=bool))
        )
        self.stitch_webtoon_cb.stateChanged.connect(self._on_stitch_mode_changed)
        layout.addWidget(self.stitch_webtoon_cb)

        layout.addStretch()

    def _on_stitch_mode_changed(self, state: int):
        QSettings("ComicLabs", "ComicTranslate").setValue("webtoon_stitch_mode", bool(state))
