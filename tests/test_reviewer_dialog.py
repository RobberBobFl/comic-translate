"""Tests for the ReviewerDialog settings dialog.

Locks down the manual model-name entry: MComboBox's line edit is read-only by
default and only dayu's set_editable() unlocks it, so a regression here would
silently force users to pick from the loaded list again.
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6 import QtWidgets

from app.ui.settings.reviewer_dialog import ReviewerDialog


class FakeSettings:
    def __init__(self):
        self.saved = None

    def get_reviewer_credentials(self):
        return {
            "api_url": "http://localhost:11434/v1",
            "api_key": "",
            "model": "",
            "save_key": False,
        }

    def set_reviewer_credentials(self, data):
        self.saved = data


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_manual_model_name_is_accepted(app):
    settings = FakeSettings()
    dialog = ReviewerDialog(settings)

    line_edit = dialog.model_combo.lineEdit()
    assert not line_edit.isReadOnly(), "line edit must accept typing"

    dialog.model_combo.addItems(["model-a", "model-b"])
    line_edit.setText("my-custom-model")
    assert dialog.model_combo.currentText() == "my-custom-model"

    dialog.accept()
    assert settings.saved is not None
    assert settings.saved["model"] == "my-custom-model"
    assert settings.saved["api_url"] == "http://localhost:11434/v1"


def test_manual_model_name_survives_load_models(app):
    """Typing a custom name, then loading the list, keeps the typed value."""
    settings = FakeSettings()
    dialog = ReviewerDialog(settings)
    dialog.model_combo.addItems(["model-a"])
    dialog.model_combo.lineEdit().setText("my-custom-model")

    # Simulate what _load_models does after a successful fetch.
    current = dialog.model_combo.currentText()
    dialog.model_combo.clear()
    dialog.model_combo.addItems(["model-a", "model-b", "model-c"])
    dialog.model_combo.setCurrentText(current)

    assert dialog.model_combo.currentText() == "my-custom-model"


def test_existing_model_prefilled(app):
    class PrefilledSettings(FakeSettings):
        def get_reviewer_credentials(self):
            return {
                "api_url": "http://x/v1",
                "api_key": "k",
                "model": "saved-model",
                "save_key": True,
            }

    dialog = ReviewerDialog(PrefilledSettings())
    assert dialog.model_combo.currentText() == "saved-model"
    assert not dialog.model_combo.lineEdit().isReadOnly()
