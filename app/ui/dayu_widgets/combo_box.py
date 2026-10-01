#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2019.2
# Email : muyanru345@163.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import third-party modules
import math

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# Import local modules
from . import dayu_theme
from .completer import MCompleter
from .mixin import cursor_mixin
from .mixin import focus_shadow_mixin
from .mixin import property_mixin
from . import utils as utils


@property_mixin
class MComboBoxSearchMixin(object):
    def __init__(self, *args, **kwargs):
        super(MComboBoxSearchMixin, self).__init__(*args, **kwargs)
        self.filter_model = QtCore.QSortFilterProxyModel(self)
        self.filter_model.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.filter_model.setSourceModel(self.model())
        self.completer = MCompleter(self)
        self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
        self.completer.setModel(self.filter_model)

    def search(self):
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setEditable(True)

        self.setCompleter(self.completer)

        edit = self.lineEdit()
        edit.setReadOnly(False)
        edit.returnPressed.disconnect()
        edit.textEdited.connect(self.filter_model.setFilterFixedString)
        self.completer.activated.connect(lambda t: t and self.setCurrentIndex(self.findText(t)))

    def _set_searchable(self, value):
        """search property to True then trigger search"""
        value and self.search()

    def setModel(self, model):
        super(MComboBoxSearchMixin, self).setModel(model)
        self.filter_model.setSourceModel(model)
        self.completer.setModel(self.filter_model)

    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.filter_model.setFilterKeyColumn(column)
        super(MComboBoxSearchMixin, self).setModelColumn(column)


@cursor_mixin
@focus_shadow_mixin
class MComboBox(MComboBoxSearchMixin, QtWidgets.QComboBox):
    Separator = "/"
    sig_value_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(MComboBox, self).__init__(parent)

        self._root_menu = None
        self._display_formatter = utils.display_formatter
        self.setEditable(True)
        line_edit = self.lineEdit()
        line_edit.setReadOnly(True)
        line_edit.setTextMargins(4, 0, 4, 0)
        line_edit.setStyleSheet("background-color:transparent")
        line_edit.setCursor(QtCore.Qt.PointingHandCursor)
        line_edit.installEventFilter(self)
        self._has_custom_view = False
        self.set_value("")
        self.set_placeholder(self.tr("Please Select"))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._dayu_size = dayu_theme.default_size

    # Add this new method to toggle editability
    def set_editable(self, editable: bool):
        """Set whether the combo box is editable."""
        self.lineEdit().setReadOnly(not editable) 

    def get_dayu_size(self):
        """
        Get the push button height
        :return: integer
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the avatar size.
        :param value: integer
        :return: None
        """
        self._dayu_size = value
        self.lineEdit().setProperty("dayu_size", value)
        self.style().polish(self)

    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def set_formatter(self, func):
        self._display_formatter = func

    def set_placeholder(self, text):
        """Display the text when no item selected."""
        self.lineEdit().setPlaceholderText(text)

    def set_value(self, value):
        self.setProperty("value", value)

    def _set_value(self, value):
        self.lineEdit().setProperty("text", self._display_formatter(value))
        if self._root_menu:
            self._root_menu.set_value(value)

    def set_menu(self, menu):
        self._root_menu = menu
        self._root_menu.sig_value_changed.connect(self.sig_value_changed)
        self._root_menu.sig_value_changed.connect(self.set_value)

    def setView(self, *args, **kwargs):
        """Override setView to flag _has_custom_view variable."""
        self._has_custom_view = True
        super(MComboBox, self).setView(*args, **kwargs)

    def showPopup(self):
        """Override default showPopup. When set custom menu, show the menu instead."""
        if self._has_custom_view or self._root_menu is None:
            super(MComboBox, self).showPopup()
        else:
            super(MComboBox, self).hidePopup()
            self._root_menu.popup(self.mapToGlobal(QtCore.QPoint(0, self.height())))

    # def setCurrentIndex(self, index):
    #     raise NotImplementedError

    def eventFilter(self, widget, event):
        if widget is self.lineEdit() and widget.isReadOnly():
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.showPopup()
        return super(MComboBox, self).eventFilter(widget, event)

    def huge(self):
        """Set MComboBox to huge size"""
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """Set MComboBox to large size"""
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """Set MComboBox to  medium"""
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """Set MComboBox to small size"""
        self.set_dayu_size(dayu_theme.small)
        return self

    def tiny(self):
        """Set MComboBox to tiny size"""
        self.set_dayu_size(dayu_theme.tiny)
        return self


def _make_star_pixmap(filled, size=16, dpr=2.0):
    """Draw a five-point star: filled amber for favorites, gray outline otherwise."""
    pm = QtGui.QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    polygon = QtGui.QPolygonF()
    center = size / 2.0
    outer = size / 2.0 - 1.0
    inner = outer * 0.45
    for i in range(10):
        angle = -math.pi / 2.0 + i * math.pi / 5.0
        radius = outer if i % 2 == 0 else inner
        polygon.append(QtCore.QPointF(center + radius * math.cos(angle),
                                      center + radius * math.sin(angle)))
    painter.setPen(QtGui.QPen(QtGui.QColor("#f5a623" if filled else "#8a8a8a"), 1.2))
    if filled:
        painter.setBrush(QtGui.QColor("#f5a623"))
    painter.drawPolygon(polygon)
    painter.end()
    return pm


class _StarItemDelegate(QtWidgets.QStyledItemDelegate):
    """Paints the favorite star in the last popup column."""

    def __init__(self, is_favorite_fn, parent=None):
        super(_StarItemDelegate, self).__init__(parent)
        self._is_favorite = is_favorite_fn
        self._filled = _make_star_pixmap(True)
        self._outline = _make_star_pixmap(False)

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QtGui.QIcon()
        widget = opt.widget
        style = widget.style() if widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        family = index.siblingAtColumn(0).data(QtCore.Qt.ItemDataRole.DisplayRole) or ""
        pixmap = self._filled if self._is_favorite(family) else self._outline
        rect = option.rect
        size = QtCore.QSizeF(pixmap.size()) / pixmap.devicePixelRatio()
        target = QtCore.QRectF(rect.x() + (rect.width() - size.width()) / 2.0,
                               rect.y() + (rect.height() - size.height()) / 2.0,
                               size.width(), size.height())
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, pixmap, QtCore.QRectF(pixmap.rect()))
        painter.restore()

    def sizeHint(self, option, index):
        size = super(_StarItemDelegate, self).sizeHint(option, index)
        star = max(self._filled.width(), self._outline.width())
        size.setWidth(max(size.width(), star + 8))
        return size


class _FontPopupTreeView(QtWidgets.QTreeView):
    """Popup view that turns clicks on the star column into toggle requests
    instead of row selection."""

    favorite_clicked = QtCore.Signal(str)

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and index.column() == MFontComboBox.STAR_COLUMN:
            family = index.siblingAtColumn(0).data(QtCore.Qt.ItemDataRole.DisplayRole)
            if family:
                self.favorite_clicked.emit(family)
            event.accept()
            return
        super(_FontPopupTreeView, self).mousePressEvent(event)


@cursor_mixin
@focus_shadow_mixin
class MFontComboBox(MComboBoxSearchMixin, QtWidgets.QFontComboBox):
    Separator = "/"
    STAR_COLUMN = 1
    sig_value_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(MFontComboBox, self).__init__(parent)

        self._root_menu = None
        self._display_formatter = utils.display_formatter
        self.setEditable(True)
        line_edit = self.lineEdit()
        #line_edit.setReadOnly(True)
        line_edit.setTextMargins(4, 0, 4, 0)
        line_edit.setStyleSheet("background-color:transparent")
        line_edit.setCursor(QtCore.Qt.PointingHandCursor)
        line_edit.installEventFilter(self)
        self._has_custom_view = False
        self.set_value("")
        self.set_placeholder(self.tr("Please Select"))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._dayu_size = dayu_theme.default_size
        self._font_preview_delegate = self.view().itemDelegate() if self.view() is not None else None
        self._font_favorites = None
        self._favorites_only = False
        self._known_families = ()
        self._font_model = QtGui.QStandardItemModel(0, 2, self)
        line_edit = self.lineEdit()
        if line_edit is not None:
            self._default_line_edit_family = line_edit.font().family()
        self._configure_popup_view()
        self._rebuild_font_model()
        app_instance = QtGui.QGuiApplication.instance()
        if app_instance is not None:
            app_instance.fontDatabaseChanged.connect(self._ensure_font_model_fresh)
        self.currentTextChanged.connect(self._update_selected_font_tooltip)
        self.currentIndexChanged.connect(self._update_line_edit_font)
        self._update_selected_font_tooltip(self.currentText())

    def _update_line_edit_font(self, index):
        line_edit = self.lineEdit()
        if line_edit is None:
            return
        family = ""
        if 0 <= index < self._font_model.rowCount():
            font = self._font_model.item(index, 0).data(QtCore.Qt.ItemDataRole.FontRole)
            if font is not None:
                family = font.family()
        if not family:
            family = getattr(self, "_default_line_edit_family", "") or line_edit.font().family()
        new_font = QtGui.QFont(line_edit.font())
        new_font.setFamily(family)
        if new_font != line_edit.font():
            line_edit.setFont(new_font)

    def _configure_popup_view(self):
        view = self.view()
        if not isinstance(view, _FontPopupTreeView):
            tree_view = _FontPopupTreeView(self)
            tree_view.setRootIsDecorated(False)
            tree_view.setItemsExpandable(False)
            tree_view.setUniformRowHeights(True)
            tree_view.setHeaderHidden(True)
            if self._font_preview_delegate is not None:
                tree_view.setItemDelegateForColumn(0, self._font_preview_delegate)
            tree_view.setItemDelegateForColumn(1, _StarItemDelegate(self._is_font_favorite, tree_view))
            tree_view.favorite_clicked.connect(self._on_star_clicked)
            header = tree_view.header()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(MFontComboBox.STAR_COLUMN, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self._has_custom_view = True
            # Mixin setModel(): swaps the combo model and re-binds the
            # search completer to it.
            self.setModel(self._font_model)
            super(MFontComboBox, self).setView(tree_view)
            view = tree_view

        if view is None:
            return

        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        if hasattr(view, "setWordWrap"):
            view.setWordWrap(False)

    def _sync_popup_width(self):
        view = self.view()
        if view is None:
            return

        widest_text = 0
        model = self.model()
        root_index = self.rootModelIndex()
        model_column = self.modelColumn()
        if model is not None:
            for row in range(self.count()):
                model_index = model.index(row, model_column, root_index)
                if model_index.isValid():
                    widest_text = max(widest_text, view.sizeHintForIndex(model_index).width())

        if widest_text <= 0:
            metrics = view.fontMetrics()
            for idx in range(self.count()):
                widest_text = max(widest_text, metrics.horizontalAdvance(self.itemText(idx)))

        extra_padding = 120
        is_tree = isinstance(view, QtWidgets.QTreeView)
        star_width = view.columnWidth(MFontComboBox.STAR_COLUMN) if is_tree else 0
        target_width = widest_text + extra_padding
        max_popup_width = max(self.width(), 420)
        popup_width = min(max_popup_width, max(self.width(), target_width + star_width))

        if is_tree:
            # The name column must leave room for the star column, otherwise
            # the star gets clipped outside the popup. Long names stay
            # reachable through the horizontal scrollbar.
            view.setColumnWidth(0, max(120, popup_width - star_width))

        view.setMinimumWidth(popup_width)
        view.setMaximumWidth(popup_width)

    def _update_selected_font_tooltip(self, family_name: str):
        if self.lineEdit() is not None:
            self.lineEdit().setToolTip(family_name or self.tr("Font"))

    def setCurrentFont(self, font):
        """QFontComboBox's own implementation only updates its private state
        (its model update no-ops for foreign models), so select the family in
        our model ourselves."""
        family = font.family() if isinstance(font, QtGui.QFont) else str(font)
        if not self._select_family(family or ""):
            self.setCurrentIndex(-1)

    def set_font_favorites(self, font_favorites):
        """Attach a FontFavorites instance; enables the star column."""
        self._font_favorites = font_favorites
        if font_favorites is not None:
            font_favorites.changed.connect(self._rebuild_font_model)
            self._rebuild_font_model()

    def set_favorites_only(self, enabled):
        """When True, the popup only lists favorites (plus the current font)."""
        self._favorites_only = bool(enabled)

    def _is_font_favorite(self, family: str) -> bool:
        return (self._font_favorites is not None
                and self._font_favorites.is_favorite(family))

    def _on_star_clicked(self, family: str):
        if self._font_favorites is not None:
            self._font_favorites.toggle(family)

    def _rebuild_font_model(self):
        families = list(QtGui.QFontDatabase.families())
        self._known_families = tuple(families)
        current = self.currentText()

        ordered = []
        available = {family.casefold(): family for family in families}
        if self._font_favorites is not None:
            for stored in self._font_favorites.favorites():
                family = available.get(stored.casefold())
                if family is not None and family not in ordered:
                    ordered.append(family)
        favorite_set = {family.casefold() for family in ordered}
        ordered.extend(family for family in families if family.casefold() not in favorite_set)

        # QComboBox auto-selects row 0 as soon as the first row lands and
        # clears the selection when rows go away — both emit currentTextChanged
        # with transient values. Keep the whole repopulation signal-silent and
        # restore the selection up front.
        was_blocked = self.blockSignals(True)
        try:
            self._font_model.setRowCount(0)
            for family in ordered:
                name_item = QtGui.QStandardItem(family)
                name_item.setData(QtGui.QFont(family), QtCore.Qt.ItemDataRole.FontRole)
                star_item = QtGui.QStandardItem()
                star_item.setEditable(False)
                self._font_model.appendRow([name_item, star_item])

            if current:
                self._select_family(current)
            else:
                self.setCurrentIndex(-1)
        finally:
            self.blockSignals(was_blocked)
        view = self.view()
        if isinstance(view, QtWidgets.QTreeView):
            # Column widths only stick once the model actually has columns.
            view.setColumnWidth(MFontComboBox.STAR_COLUMN, 28)
        if self._favorites_only:
            self._apply_favorites_filter()
        elif view is not None:
            for row in range(self._font_model.rowCount()):
                view.setRowHidden(row, QtCore.QModelIndex(), False)

    def _select_family(self, family: str) -> bool:
        folded = family.casefold()
        for row in range(self._font_model.rowCount()):
            name = self._font_model.item(row, 0).text()
            if name.casefold() == folded:
                self.setCurrentIndex(row)
                return True
        return False

    def _ensure_font_model_fresh(self):
        if self._known_families != tuple(QtGui.QFontDatabase.families()):
            self._rebuild_font_model()

    def _apply_favorites_filter(self):
        view = self.view()
        if view is None:
            return
        current = self.currentText().casefold()
        for row in range(self._font_model.rowCount()):
            family = self._font_model.item(row, 0).text()
            hidden = (self._favorites_only
                      and self._font_favorites is not None
                      and not self._font_favorites.is_favorite(family)
                      and family.casefold() != current)
            view.setRowHidden(row, QtCore.QModelIndex(), hidden)

    def get_dayu_size(self):
        """
        Get the push button height
        :return: integer
        """
        return self._dayu_size

    def set_dayu_size(self, value):
        """
        Set the avatar size.
        :param value: integer
        :return: None
        """
        self._dayu_size = value
        self.lineEdit().setProperty("dayu_size", value)
        self.style().polish(self)

    dayu_size = QtCore.Property(int, get_dayu_size, set_dayu_size)

    def set_formatter(self, func):
        self._display_formatter = func

    def set_placeholder(self, text):
        """Display the text when no item selected."""
        self.lineEdit().setPlaceholderText(text)

    def set_value(self, value):
        self.setProperty("value", value)

    def _set_value(self, value):
        self.lineEdit().setProperty("text", self._display_formatter(value))
        if self._root_menu:
            self._root_menu.set_value(value)

    def set_menu(self, menu):
        self._root_menu = menu
        self._root_menu.sig_value_changed.connect(self.sig_value_changed)
        self._root_menu.sig_value_changed.connect(self.set_value)

    def setView(self, *args, **kwargs):
        """Override setView to flag _has_custom_view variable."""
        self._has_custom_view = True
        super(MFontComboBox, self).setView(*args, **kwargs)
        self._configure_popup_view()

    def showPopup(self):
        """Override default showPopup. When set custom menu, show the menu instead."""
        if self._has_custom_view or self._root_menu is None:
            self._ensure_font_model_fresh()
            self._apply_favorites_filter()
            super(MFontComboBox, self).showPopup()
            self._sync_popup_width()
            if self.view() is not None:
                self.view().horizontalScrollBar().setValue(0)
        else:
            super(MFontComboBox, self).hidePopup()
            self._root_menu.popup(self.mapToGlobal(QtCore.QPoint(0, self.height())))

    # def setCurrentIndex(self, index):
    #     raise NotImplementedError

    def eventFilter(self, widget, event):
        if widget is self.lineEdit() and widget.isReadOnly():
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.showPopup()
        return super(MFontComboBox, self).eventFilter(widget, event)

    def huge(self):
        """Set MComboBox to huge size"""
        self.set_dayu_size(dayu_theme.huge)
        return self

    def large(self):
        """Set MComboBox to large size"""
        self.set_dayu_size(dayu_theme.large)
        return self

    def medium(self):
        """Set MComboBox to  medium"""
        self.set_dayu_size(dayu_theme.medium)
        return self

    def small(self):
        """Set MComboBox to small size"""
        self.set_dayu_size(dayu_theme.small)
        return self

    def tiny(self):
        """Set MComboBox to tiny size"""
        self.set_dayu_size(dayu_theme.tiny)
        return self
