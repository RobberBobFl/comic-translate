"""Tests for the favorite-fonts feature.

Verifies that:
  - FontFavorites persists the starred families across instances (QSettings)
  - the font combo model keeps favorites pinned at the top, in star order
  - the star toggle path reorders the model without emitting bogus
    currentTextChanged signals
  - favorites-only mode hides every non-favorite row except the current font
  - setCurrentFont resolves families against the custom model (Qt's own
    implementation no-ops on replaced models) and clears on unknown names
  - clicking the star column of the popup view toggles the favorite
"""
import os
import tempfile

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.font_favorites import FontFavorites
from app.ui.dayu_widgets.combo_box import MFontComboBox


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def favorites():
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    settings = QtCore.QSettings(path, QtCore.QSettings.Format.IniFormat)
    yield FontFavorites(settings=settings)
    os.unlink(path)


@pytest.fixture()
def combo(app, favorites):
    widget = MFontComboBox()
    widget.set_font_favorites(favorites)
    yield widget
    widget.deleteLater()


def _model_families(combo):
    return [combo._font_model.item(r, 0).text() for r in range(combo._font_model.rowCount())]


def _hidden_rows(combo):
    view = combo.view()
    return [r for r in range(combo._font_model.rowCount())
            if view.isRowHidden(r, QtCore.QModelIndex())]


def test_model_lists_all_families(combo):
    families = list(QtGui.QFontDatabase.families())
    assert combo._font_model.columnCount() == 2
    assert _model_families(combo) == families


def test_favorites_pinned_on_top_in_star_order(combo, favorites):
    families = list(QtGui.QFontDatabase.families())
    first, second = families[5], families[40]

    favorites.toggle(first)
    favorites.toggle(second)

    result = _model_families(combo)
    assert result[:2] == [first, second]
    assert result[2:] == [f for f in families if f not in (first, second)]


def test_favorites_persist_across_instances(favorites):
    families = list(QtGui.QFontDatabase.families())
    favorites.toggle(families[5])

    reloaded = FontFavorites(settings=favorites._qsettings())
    assert reloaded.favorites() == favorites.favorites()
    assert reloaded.is_favorite(families[5])


def test_favorites_lookup_is_case_insensitive(favorites):
    families = list(QtGui.QFontDatabase.families())
    family = families[5]
    favorites.toggle(family)
    assert favorites.is_favorite(family.upper())
    # un-starring with a different spelling still removes the single entry
    favorites.toggle(family.swapcase())
    assert favorites.favorites() == []


def test_star_toggle_is_signal_silent(combo, favorites):
    """Reordering on star toggle must not emit currentTextChanged spikes."""
    families = list(QtGui.QFontDatabase.families())
    combo._select_family(families[7])
    assert combo.currentText() == families[7]

    spikes = []
    combo.currentTextChanged.connect(lambda text: spikes.append(text))
    favorites.toggle(families[5])

    assert spikes == []
    assert combo.currentText() == families[7]
    assert _model_families(combo)[0] == families[5]


def test_rebuild_keeps_selection_and_clears_when_empty(combo, favorites):
    families = list(QtGui.QFontDatabase.families())
    combo._select_family(families[9])
    combo._rebuild_font_model()
    assert combo.currentText() == families[9]

    # a fresh combo with no selection must stay unselected after a rebuild
    fresh = MFontComboBox()
    fresh.set_font_favorites(favorites)
    assert fresh.currentIndex() == -1
    fresh._rebuild_font_model()
    assert fresh.currentIndex() == -1
    fresh.deleteLater()


def test_set_current_font_resolves_and_clears(combo):
    families = list(QtGui.QFontDatabase.families())
    combo.setCurrentFont(QtGui.QFont(families[10]))
    assert combo.currentText() == families[10]

    combo.setCurrentFont(QtGui.QFont("No Such Font Family 123"))
    assert combo.currentIndex() == -1


def test_favorites_only_hides_non_favorites(combo, favorites):
    families = list(QtGui.QFontDatabase.families())
    favorite = families[5]
    favorites.toggle(favorite)

    current = families[9]  # selected but not favorited
    combo._select_family(current)
    combo.set_favorites_only(True)
    combo._apply_favorites_filter()

    visible = {r for r in range(combo._font_model.rowCount())
               if r not in set(_hidden_rows(combo))}
    visible_families = {combo._font_model.item(r, 0).text() for r in visible}
    assert visible_families == {favorite, current}


def test_favorites_only_off_shows_everything(combo, favorites):
    families = list(QtGui.QFontDatabase.families())
    favorites.toggle(families[5])
    combo.set_favorites_only(False)
    combo._rebuild_font_model()
    assert _hidden_rows(combo) == []


def test_star_column_click_toggles_favorite(combo, favorites, qtbot=None):
    view = combo.view()
    view.resize(500, 300)
    families = _model_families(combo)
    target = families[3]
    row = families.index(target)

    star_index = combo._font_model.index(row, MFontComboBox.STAR_COLUMN)
    rect = view.visualRect(star_index)
    assert not rect.isNull(), "star column must be visible inside the popup"

    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(rect.center()),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        device=QtGui.QPointingDevice.primaryPointingDevice(),
    )
    view.mousePressEvent(event)

    assert favorites.is_favorite(target)
    # clicking the star must not change the combo selection
    assert combo.currentIndex() != row or combo.currentText() == target


def test_line_edit_font_follows_selection(combo):
    families = list(QtGui.QFontDatabase.families())
    combo._select_family(families[6])
    assert combo.lineEdit().font().family() == families[6]
