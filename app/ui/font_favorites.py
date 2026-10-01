"""Persistent list of favorite font families backed by QSettings.

The favorites are stored as an ordered list of font family names (the
order in which they were starred). The dropdown shows them pinned at
the top of the font list.
"""
from PySide6 import QtCore


_shared_instance = None


class FontFavorites(QtCore.QObject):
    changed = QtCore.Signal()

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        # Tests pass a scratch QSettings so the real user config is untouched.
        self._settings = settings
        self._favorites = []
        self._load()

    def _qsettings(self) -> QtCore.QSettings:
        if self._settings is not None:
            return self._settings
        return QtCore.QSettings("ComicLabs", "ComicTranslate")

    def _load(self):
        settings = self._qsettings()
        settings.beginGroup("fonts")
        raw = settings.value("favorites", [])
        settings.endGroup()
        if isinstance(raw, str):
            raw = [raw]
        self._favorites = [str(v) for v in (raw or []) if str(v).strip()]

    def _save(self):
        settings = self._qsettings()
        settings.beginGroup("fonts")
        settings.setValue("favorites", self._favorites)
        settings.endGroup()
        self.changed.emit()

    def favorites(self):
        return list(self._favorites)

    def is_favorite(self, family: str) -> bool:
        return self._find(family) is not None

    def toggle(self, family: str) -> bool:
        """Star/unstar a font. Returns True if it is now a favorite."""
        if self._find(family) is not None:
            self.remove(family)
            return False
        self.add(family)
        return True

    def add(self, family: str):
        if not isinstance(family, str) or not family.strip():
            return
        if self._find(family) is None:
            self._favorites.append(family)
            self._save()

    def remove(self, family: str):
        found = self._find(family)
        if found is not None:
            self._favorites.remove(found)
            self._save()

    def _find(self, family: str):
        """Return the stored spelling of `family` (case-insensitive lookup)."""
        if not isinstance(family, str):
            return None
        folded = family.casefold()
        for stored in self._favorites:
            if stored.casefold() == folded:
                return stored
        return None


def get_shared_font_favorites() -> FontFavorites:
    """Process-wide FontFavorites so every widget sees the same list."""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = FontFavorites()
    return _shared_instance
