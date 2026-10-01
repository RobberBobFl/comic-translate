from PySide6 import QtWidgets
from PySide6.QtCore import QSettings, Signal
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from ..dayu_widgets.browser import MClickBrowserFileToolButton
from ..dayu_widgets.check_box import MCheckBox

class TextRenderingPage(QtWidgets.QWidget):
    favorites_only_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Font section
        font_layout = QtWidgets.QVBoxLayout()
        min_font_layout = QtWidgets.QHBoxLayout()
        max_font_layout = QtWidgets.QHBoxLayout()
        min_font_label = MLabel(self.tr("Minimum Font Size:"))
        max_font_label = MLabel(self.tr("Maximum Font Size:"))

        self.min_font_spinbox = MSpinBox().small()
        self.min_font_spinbox.setFixedWidth(60)
        self.min_font_spinbox.setMaximum(100)
        self.min_font_spinbox.setValue(9)

        self.max_font_spinbox = MSpinBox().small()
        self.max_font_spinbox.setFixedWidth(60)
        self.max_font_spinbox.setMaximum(100)
        self.max_font_spinbox.setValue(40)

        min_font_layout.addWidget(min_font_label)
        min_font_layout.addWidget(self.min_font_spinbox)
        min_font_layout.addStretch()

        max_font_layout.addWidget(max_font_label)
        max_font_layout.addWidget(self.max_font_spinbox)
        max_font_layout.addStretch()

        font_label = MLabel(self.tr("Font:")).h4()

        font_browser_layout = QtWidgets.QHBoxLayout()
        import_font_label = MLabel(self.tr("Import Font:"))
        self.font_browser = MClickBrowserFileToolButton(multiple=True)
        self.font_browser.set_dayu_filters([".ttf", ".ttc", ".otf", ".woff", ".woff2"])
        self.font_browser.setToolTip(self.tr("Import the Font to use for Rendering Text on Images"))

        font_browser_layout.addWidget(import_font_label)
        font_browser_layout.addWidget(self.font_browser)
        font_browser_layout.addStretch()

        font_layout.addWidget(font_label)
        font_layout.addLayout(font_browser_layout)
        font_layout.addLayout(min_font_layout)
        font_layout.addLayout(max_font_layout)

        # Uppercase
        self.uppercase_checkbox = MCheckBox(self.tr("Render Text in UpperCase"))

        # Favorite fonts
        self.favorites_only_checkbox = MCheckBox(self.tr("Show Favorite Fonts Only"))
        self.favorites_only_checkbox.setToolTip(
            self.tr("Limit the font dropdown to the fonts starred as favorites")
        )
        self.favorites_only_checkbox.toggled.connect(self._on_favorites_only_changed)

        layout.addWidget(self.uppercase_checkbox)
        layout.addWidget(self.favorites_only_checkbox)
        layout.addSpacing(10)
        layout.addLayout(font_layout)
        layout.addSpacing(10)
        layout.addStretch(1)

    def _on_favorites_only_changed(self, checked: bool):
        # Persist immediately (same pattern as shortcut changes), the
        # dropdown listens to favorites_only_changed.
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("text_rendering")
        settings.setValue("show_favorites_only", bool(checked))
        settings.endGroup()
        self.favorites_only_changed.emit(bool(checked))
