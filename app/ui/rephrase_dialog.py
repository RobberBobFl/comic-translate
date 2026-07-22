from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt


class RephraseDialog(QtWidgets.QDialog):
    """Dialog showing the rephrased text with Accept / Cancel buttons."""

    def __init__(self, original: str, rephrased: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Rephrase"))
        self.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)

        # Original
        layout.addWidget(QtWidgets.QLabel(self.tr("Original:")))
        orig_label = QtWidgets.QLabel(original)
        orig_label.setWordWrap(True)
        orig_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        orig_label.setStyleSheet(
            "background: #f5f5f5; color: #000000; padding: 6px; border-radius: 4px;"
        )
        layout.addWidget(orig_label)

        # Rephrased
        layout.addWidget(QtWidgets.QLabel(self.tr("Rephrased:")))
        self.rephrased_label = QtWidgets.QLabel(rephrased)
        self.rephrased_label.setWordWrap(True)
        self.rephrased_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.rephrased_label.setStyleSheet(
            "background: #d4edda; color: #000000; padding: 6px; border-radius: 4px; font-weight: bold;"
        )
        layout.addWidget(self.rephrased_label)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        accept_btn = QtWidgets.QPushButton(self.tr("Apply"))
        accept_btn.setDefault(True)
        accept_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(accept_btn)
        layout.addLayout(btn_layout)