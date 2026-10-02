from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from PySide6 import QtCore

from modules.utils.common_utils import is_close

if TYPE_CHECKING:
    from controller import ComicTranslate
    from modules.utils.textblock import TextBlock

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


class TranslationReviewController(QtCore.QObject):
    """Populates the review panel from page state and applies replacements.

    Suggestions live in ``image_states[path]['translation_review']``; the panel
    shows the displayed page's entries and lets the user push a chosen variant
    into the block's translation (canvas, edits and persisted state included).
    """

    def __init__(self, main: ComicTranslate):
        super().__init__(main)
        self.main = main

        panel = getattr(self.main, "review_panel", None)
        if panel is None:
            return
        panel.apply_entries_requested.connect(self.apply_entries)
        panel.block_focus_requested.connect(self.focus_block)
        panel.close_requested.connect(panel.hide)

        # The stored review belongs to a page; repopulate when the page changes.
        self.main.page_list.currentRowChanged.connect(
            lambda _row: QtCore.QTimer.singleShot(250, self.refresh_from_state)
        )

    # ------------------------------------------------------------------
    # Panel population
    # ------------------------------------------------------------------

    def _current_file_path(self) -> Optional[str]:
        if 0 <= self.main.curr_img_idx < len(self.main.image_files):
            return self.main.image_files[self.main.curr_img_idx]
        return None

    def refresh_from_state(self) -> None:
        """Show the current page's stored review, marking drifted entries stale."""
        panel = getattr(self.main, "review_panel", None)
        if panel is None:
            return
        file_path = self._current_file_path()
        state = self.main.image_states.get(file_path, {}) if file_path else {}
        review_state = state.get("translation_review") or {}
        raw_blocks = review_state.get("blocks") or {}
        entries: dict[int, dict] = {}
        stale_ids: set[int] = set()
        for key, entry in raw_blocks.items():
            try:
                entry_id = int(key)
            except (TypeError, ValueError):
                continue
            entries[entry_id] = entry
            if not self._is_current(file_path, entry):
                stale_ids.add(entry_id)
        panel.set_entries(entries, stale_ids)
        if entries and not panel.isVisible():
            panel.show()

    def show_results(self, file_path: Optional[str]) -> None:
        """Called after a review run: show results if the reviewed page is displayed."""
        current = self._current_file_path()
        if current is None or file_path is None:
            return
        if os.path.normcase(current) != os.path.normcase(file_path):
            return
        self.refresh_from_state()

    def _is_current(self, file_path: Optional[str], entry: dict) -> bool:
        """False when the block's translation drifted since the review ran."""
        blk = self._live_block(file_path, entry)
        if blk is None:
            return False
        return _norm(blk.translation) == _norm(entry.get("translation_snapshot", ""))

    # ------------------------------------------------------------------
    # Block lookup
    # ------------------------------------------------------------------

    def _live_block(self, file_path: Optional[str], entry: dict) -> Optional[TextBlock]:
        """Find the entry's block among the displayed blocks by position.

        Falls back to the reviewed index when the position does not match but
        the source text does (blocks may be re-sorted between review and apply).
        """
        if file_path is None:
            return None
        current = self._current_file_path()
        if current is None or os.path.normcase(current) != os.path.normcase(file_path):
            return None
        blk_list = self.main.blk_list or []
        xyxy = entry.get("xyxy")
        angle = float(entry.get("angle", 0.0) or 0.0)
        if xyxy:
            for blk in blk_list:
                try:
                    blk_xyxy = (
                        int(blk.xyxy[0]), int(blk.xyxy[1]),
                        int(blk.xyxy[2]), int(blk.xyxy[3]),
                    )
                except (TypeError, ValueError, AttributeError):
                    continue
                if blk_xyxy == tuple(int(v) for v in xyxy) and is_close(
                    float(getattr(blk, "angle", 0.0) or 0.0), angle, 0.5
                ):
                    return blk
        # Position fallback: same source text at the reviewed index.
        try:
            idx = int(entry.get("block_index", -1))
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(blk_list):
            blk = blk_list[idx]
            if _norm(getattr(blk, "text", "")) == _norm(entry.get("source_text", "")):
                return blk
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def focus_block(self, entry_id: int) -> None:
        file_path = self._current_file_path()
        state = self.main.image_states.get(file_path, {}) if file_path else {}
        entry = (state.get("translation_review") or {}).get("blocks", {}).get(
            str(entry_id)
        )
        if entry is None:
            return
        blk = self._live_block(file_path, entry)
        if blk is None:
            return
        self.main.search_ctrl._select_block(blk)

    def apply_entries(self, selections: list) -> None:
        """Replace translations with the chosen variants: [(entry_id, text), ...]."""
        panel = getattr(self.main, "review_panel", None)
        if not selections:
            return
        file_path = self._current_file_path()
        if file_path is None:
            return
        state = self.main.image_states.setdefault(file_path, {})
        review_state = state.get("translation_review") or {}
        blocks = review_state.get("blocks") or {}

        applied: list[int] = []
        skipped_stale: list[int] = []
        for selection in selections:
            try:
                entry_id, text = int(selection[0]), str(selection[1])
            except (TypeError, ValueError):
                continue
            entry = blocks.get(str(entry_id))
            if entry is None:
                continue
            blk = self._live_block(file_path, entry)
            if blk is None:
                skipped_stale.append(entry_id)
                continue
            if _norm(blk.translation) != _norm(entry.get("translation_snapshot", "")):
                skipped_stale.append(entry_id)
                continue

            old_text = blk.translation
            blk.translation = text
            self._sync_state_block(state, entry, text)
            self._update_canvas(blk, text)
            applied.append(entry_id)
            logger.info(
                "Review applied for block %d: %r -> %r",
                entry_id, old_text, text,
            )

        if not applied:
            if skipped_stale:
                self._mark_stale(skipped_stale)
            return

        for entry_id in applied:
            blocks.pop(str(entry_id), None)
        if not blocks:
            state.pop("translation_review", None)

        if getattr(self.main, "webtoon_mode", False):
            try:
                self.main.manual_workflow_ctrl.sync_blk_list_to_state()
            except Exception:
                logger.exception("Failed to sync applied review to page state")

        self.main.mark_project_dirty()
        if panel is not None:
            panel.remove_entries(applied)
            self._mark_stale(skipped_stale)

    def _update_canvas(self, blk: TextBlock, text: str) -> None:
        try:
            text_ctrl = self.main.text_ctrl
            text_item = self.main.search_ctrl._find_text_item_for_block(blk)
        except Exception:
            return
        if text_item is not None:
            try:
                text_ctrl.apply_text_from_command(text_item, text, html=None, blk=blk)
                text_ctrl._refit_text_item_to_block(text_item, blk)
            except Exception:
                logger.exception("Failed to update text item after review apply")
        try:
            if self.main.curr_tblock is blk:
                self.main.t_text_edit.blockSignals(True)
                self.main.t_text_edit.setPlainText(text)
                self.main.t_text_edit.blockSignals(False)
        except Exception:
            pass

    def _sync_state_block(self, state: dict, entry: dict, text: str) -> None:
        """Mirror the replacement into the page's persisted block list."""
        blks = state.get("blk_list") or []
        xyxy = entry.get("xyxy")
        angle = float(entry.get("angle", 0.0) or 0.0)
        if not xyxy:
            return
        target = tuple(int(v) for v in xyxy)
        for blk in blks:
            try:
                blk_xyxy = (
                    int(blk.xyxy[0]), int(blk.xyxy[1]),
                    int(blk.xyxy[2]), int(blk.xyxy[3]),
                )
            except (TypeError, ValueError, AttributeError):
                continue
            if blk_xyxy == target and is_close(
                float(getattr(blk, "angle", 0.0) or 0.0), angle, 0.5
            ):
                blk.translation = text
                return

    def _mark_stale(self, entry_ids: list[int]) -> None:
        panel = getattr(self.main, "review_panel", None)
        if panel is None or not entry_ids:
            return
        for entry_id, card in panel.entries().items():
            if entry_id in entry_ids:
                card.set_stale(True)
