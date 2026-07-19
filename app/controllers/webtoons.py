from __future__ import annotations

import copy
import os
import tempfile
from typing import TYPE_CHECKING
from dataclasses import dataclass
from PySide6.QtCore import QSettings, QTimer
import numpy as np
import cv2
import imkit as imk
from PySide6 import QtGui
from PySide6 import QtWidgets


if TYPE_CHECKING:
    from controller import ComicTranslate

@dataclass
class LazyLoadingConfig:
    enabled: bool = True
    max_loaded_pages: int = 5
    viewport_buffer: int = 2
    load_timer_interval: int = 50
    scroll_debounce_delay: int = 150


class WebtoonController:
    """Webtoon controller with lazy loading support."""

    # Strip height (px) above which a stitched image is split into contiguous
    # page-height chunks (CHUNK_H) so we never blow up memory or hit image /
    # viewer / export dimension limits. Used by the "lightweight" mode.
    LIGHT_MAX_STRIP_H = 24000
    # Effectively no limit: the entire comic is stitched into one image
    # ("unlimited" mode). Memory and export-dimension limits may be hit on
    # very long webtoons.
    UNLIMITED_MAX_STRIP_H = 1_000_000_000
    CHUNK_H = 6000

    def __init__(self, main: ComicTranslate):
        self.main = main
        self._initialization_complete = False  # Track initialization state

        # Stitched-webtoon (single-image) mode bookkeeping
        self._stitch_temp: str | None = None
        self._webtoon_source_files: list[str] | None = None
        self._webtoon_source_states: dict | None = None
        # Stitched-webtoon bookkeeping used to split the export back into the
        # original pages (see _stitch_compute / export).
        self._stitched_orig_heights: list[int] | None = None
        self._stitched_chunk_bounds: list[tuple[int, int]] | None = None
        self._stitch_choice: str | None = None

        # Load lazy loading configuration
        config = LazyLoadingConfig()
        self.lazy_config = config
        self.lazy_loading_enabled = self.lazy_config.enabled
        
    @property
    def image_viewer(self):
        return self.main.image_viewer
        
    @property
    def image_files(self):
        return self.main.image_files
        
    @property
    def image_states(self):
        return self.main.image_states

    @property
    def current_file_path(self):
        """Get the current file path based on curr_img_idx."""
        curr_idx = self.main.curr_img_idx
        if 0 <= curr_idx < len(self.image_files):
            return self.image_files[curr_idx]
        return None

    def switch_to_webtoon_mode(self) -> bool:
        """Enhanced webtoon mode switch with lazy loading option."""
        if not self.image_files:
            print("No images loaded, cannot switch to webtoon mode")
            return False
            
        print(f"Switching to webtoon mode with {len(self.image_files)} images")
        return self._switch_to_lazy_webtoon_mode()

    def _switch_to_lazy_webtoon_mode(self) -> bool:
        """Switch to memory-efficient lazy loading webtoon mode."""

        self.main.image_ctrl.save_current_image_state()
        self.image_viewer.webtoon_manager.set_main_controller(self.main)
        
        # Get current page for initialization
        curr_img_idx = self.main.curr_img_idx
        current_page = max(0, min(curr_img_idx, len(self.image_files) - 1))
        
        # Load with lazy strategy, starting from current page
        success = self.image_viewer.load_images_webtoon(self.image_files, current_page)
        if not success:
            print("Failed to initialize lazy webtoon mode")
            return False
        
        # Apply configuration
        manager = self.image_viewer.webtoon_manager
        manager.max_loaded_pages = self.lazy_config.max_loaded_pages
        manager.viewport_buffer = self.lazy_config.viewport_buffer
        manager.load_timer.setInterval(self.lazy_config.load_timer_interval)
        manager.scroll_timer.setInterval(self.lazy_config.scroll_debounce_delay)
        
        manager.main = self.main
        manager.set_enhanced_controller(self)
        self.image_viewer.page_changed.connect(self.on_page_changed)
        manager.enhanced_controller = self
        manager.scene_item_manager.merge_clipped_items_back()
        self._setup_lazy_scene_items()
        self._connect_lazy_loading_events()
        self.image_viewer.webtoon_manager.restore_view_state()

        return True

    def _setup_lazy_scene_items(self):
        """Set up scene item management for lazy loading."""  
        # Clear scene items temporarily - they'll be reloaded when lazy pages load
        self.main.blk_list.clear()
        self.image_viewer.rectangles.clear()
        self.image_viewer.text_items.clear()
        
    def _connect_lazy_loading_events(self):
        """Connect events for lazy loading triggers."""
        # Connect scroll events to trigger lazy loading
        original_wheel_event = self.image_viewer.wheelEvent
        
        def enhanced_wheel_event(event):
            result = original_wheel_event(event)
            # Trigger lazy loading check after scroll
            if self.image_viewer.webtoon_manager:
                self.image_viewer.webtoon_manager.on_scroll()
            return result
        
        self.image_viewer.wheelEvent = enhanced_wheel_event
        
        # Also connect to viewport change events
        original_resizeEvent = self.image_viewer.resizeEvent
        
        def enhanced_resizeEvent(event):
            result = original_resizeEvent(event)
            # Trigger lazy loading update on viewport resize
            if self.image_viewer.webtoon_manager:
                self.image_viewer.webtoon_manager.on_scroll()
            return result
            
        self.image_viewer.resizeEvent = enhanced_resizeEvent

    def switch_to_regular_mode(self):
        """Switch back to regular mode with proper cleanup."""
        print("Switching to regular mode")
        
        if self.lazy_loading_enabled and hasattr(self.image_viewer, 'webtoon_manager'):
            self.image_viewer.webtoon_manager.save_view_state()
            # Disconnect page change signal
            try:
                self.image_viewer.page_changed.disconnect(self.on_page_changed)
            except:
                pass  # May not be connected
            
            # Transfer lazy-loaded items back to unified state before switching
            self._consolidate_lazy_items()
            
            # Clear lazy loading manager
            self.image_viewer.webtoon_manager.clear()
        
        # Continue with existing regular mode logic
        self._switch_to_regular_mode_existing()

    def _consolidate_lazy_items(self):
        """Consolidate lazy-loaded items back to unified scene state."""
        manager = self.image_viewer.webtoon_manager
        
        # Consolidate image data back to main controller
        for page_idx, img_array in manager.image_data.items():
            if page_idx < len(self.image_files):
                file_path = self.image_files[page_idx]
                self.main.image_data[file_path] = img_array
        
        # Save all currently visible scene items to their appropriate page states
        # This is crucial to ensure items from multiple pages are saved correctly
        manager.scene_item_manager.save_all_scene_items_to_states()
        
    def _switch_to_regular_mode_existing(self):
        """Use existing regular mode switch logic."""
        
        # Clear the scene to remove multi-page items
        self.main.blk_list.clear()
        self.image_viewer.clear_scene()
        self.main.image_ctrl.force_default_view_on_next_image_load()
        
        # Make sure the image viewer is the current widget
        self.main.central_stack.setCurrentWidget(self.image_viewer)
        
        # Ensure the image viewer has focus
        self.image_viewer.setFocus()
        
        # Display the current image in regular mode (this will load only the current page's items)
        curr_img_idx = self.main.curr_img_idx
        if 0 <= curr_img_idx < len(self.image_files):
            self.main.image_ctrl.display_image(curr_img_idx, switch_page=False)

    # Page change handler for lazy loading
    def on_page_changed(self, page_index: int):
        """Handle page changes in lazy loading webtoon mode."""
        # Only respond if we're actually in lazy webtoon mode and not already processing a page change
        if (self.image_viewer.webtoon_mode and
            0 <= page_index < len(self.image_files) and
            not getattr(self.main, '_processing_page_change', False) and
            getattr(self, '_initialization_complete', False)):
            
            # Ignore rapid successive changes to the same page
            if (hasattr(self.main, '_last_page_change_index') and 
                self.main._last_page_change_index == page_index):
                return
                
            # Set flag to prevent recursive calls
            self.main._processing_page_change = True
            self.main._last_page_change_index = page_index
            
            try:
                # Update the current image index
                old_index = self.main.curr_img_idx
                self.main.curr_img_idx = page_index
                
                # Update the page list selection without triggering signals.
                self.main.image_ctrl.set_page_list_current_row(page_index, emit_signal=False)
                
                # In lazy webtoon mode, do minimal state management to avoid interfering with scrolling
                # Only load language settings and basic state
                file_path = self.image_files[page_index]
                if file_path in self.image_states:
                    state = self.image_states[file_path]
                    # Only load language settings, don't load full image state
                    # Block signals to prevent triggering save when loading state
                    self.main.s_combo.blockSignals(True)
                    self.main.t_combo.blockSignals(True)
                    self.main.s_combo.setCurrentText(state.get('source_lang', ''))
                    self.main.t_combo.setCurrentText(state.get('target_lang', ''))
                    self.main.s_combo.blockSignals(False)
                    self.main.t_combo.blockSignals(False)
                    
                # Clear text edits
                self.main.text_ctrl.clear_text_edits()

                # Page-skip popup policy:
                # - show on explicit programmatic jumps (page list/report)
                # - hide during passive scrolling page changes
                explicit_navigation = bool(getattr(self.image_viewer, "_programmatic_scroll", False))
                self.main.image_ctrl.handle_webtoon_page_focus(file_path, explicit_navigation)
                    
            finally:
                # Use a timer to reset the processing flag to avoid blocking legitimate changes
                QTimer.singleShot(100, self._reset_page_change_flag)

    def _reset_page_change_flag(self):
        """Reset the page change processing flag."""
        self.main._processing_page_change = False

    def _on_lazy_manager_ready(self):
        """Called when the lazy manager has completed initialization."""
        self._initialization_complete = True

    def toggle_webtoon_mode(self):
        """Toggle between regular image viewer and webtoon mode.

        When the "stitch webtoon into a single image" setting is enabled,
        webtoon mode stitches all loaded pages into one (or a few) tall
        image(s) and processes them through the regular single-image
        pipeline instead of the lazy continuous-scene webtoon manager. This
        reuses the working detect/OCR/translate/render code paths and avoids
        the fragile viewport-fragment webtoon logic.
        """
        requested_mode = self.main.webtoon_toggle.isChecked()
        current_mode = bool(
            getattr(self.main, "webtoon_mode", False)
            or getattr(self.main, "webtoon_strip", False)
        )
        if current_mode == requested_mode:
            return

        if requested_mode:
            if self._stitch_enabled():
                choice = self._prompt_stitch_mode()
                if choice is None:
                    # User cancelled the dialog: keep webtoon mode off.
                    self._revert_webtoon_toggle()
                    return
                # Run the heavy stitching off the GUI thread so the progress
                # bar can animate and the UI stays responsive.
                self.main.webtoon_toggle.blockSignals(True)
                self.main.progress_bar.setVisible(True)
                self.main.progress_bar.setValue(0)
                self.main.progress_bar.setFormat(self.main.tr("Stitching webtoon… %p%"))
                self.main.run_threaded(
                    self._stitch_compute,
                    self._on_stitch_done,
                    self._on_stitch_error,
                    None,
                    choice,
                )
                return
            else:
                success = self.switch_to_webtoon_mode()
                if success:
                    self.main.webtoon_mode = True
                    self.main.webtoon_strip = False
            if success:
                self.main.mark_project_dirty()
            else:
                self.main.webtoon_toggle.blockSignals(True)
                self.main.webtoon_toggle.setChecked(False)
                self.main.webtoon_toggle.blockSignals(False)
        else:
            if getattr(self.main, "webtoon_strip", False):
                self._switch_from_stitched_to_regular()
            else:
                self.switch_to_regular_mode()
            self.main.webtoon_strip = False
            self.main.mark_project_dirty()

    def _prompt_stitch_mode(self) -> str | None:
        """Ask the user how to load the webtoon when entering stitched mode.

        Returns "light", "unlimited", or None if the user cancelled.
        """
        msg = QtWidgets.QMessageBox(self.main)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle(self.main.tr("Webtoon mode"))
        msg.setText(self.main.tr("Choose how to load the webtoon:"))
        light_btn = msg.addButton(
            self.main.tr("Lightweight (stitch, auto-chunk if very tall)"),
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        unlimited_btn = msg.addButton(
            self.main.tr("Unlimited (stitch entire comic into one image)"),
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(light_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is light_btn:
            return "light"
        if clicked is unlimited_btn:
            return "unlimited"
        return None

    # ------------------------------------------------------------------
    # Stitched (single-image) webtoon mode
    # ------------------------------------------------------------------
    def _stitch_enabled(self) -> bool:
        try:
            return bool(
                QSettings("ComicLabs", "ComicTranslate").value(
                    "webtoon_stitch_mode", True, type=bool
                )
            )
        except Exception:
            return True

    def _stitch_temp_dir(self) -> str:
        if self._stitch_temp is None:
            self._stitch_temp = tempfile.mkdtemp(prefix="ct_webtoon_stitch_")
        return self._stitch_temp

    @staticmethod
    def _normalize_channels(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3].copy()
        return arr

    def _stitch_compute(self, choice: str = "light") -> dict | None:
        """Heavy stitching work (runs in a background thread). Loads all pages,
        concatenates them into one tall image and writes the stitched chunk
        PNG(s). Returns the computed bookkeeping so the GUI can be rebuilt on
        the main thread. Returns None on failure.

        choice: "light" (auto-chunk the strip if very tall) or "unlimited"
        (stitch the entire comic into a single image).
        """
        if not self.image_files:
            print("No images loaded, cannot switch to stitched webtoon mode")
            return None

        # The "unlimited" mode stitches the entire comic into one image that
        # can far exceed PIL's decompression-bomb pixel limit (~178M px), so we
        # lift that limit. This disables PIL's DOS protection for the session.
        if choice == "unlimited":
            try:
                import PIL
                PIL.Image.MAX_IMAGE_PIXELS = None
            except Exception:
                pass

        # Preserve the original (per-page) project so we can switch back.
        self._webtoon_source_files = list(self.image_files)
        self._webtoon_source_states = {
            k: copy.deepcopy(v) for k, v in self.main.image_states.items()
        }

        # Read every page and stitch them vertically into one tall image.
        pages = []
        heights = []
        target_w = None
        source_files = self._webtoon_source_files
        load_total = len(source_files)
        for i, fp in enumerate(source_files):
            arr = self.main.image_ctrl.load_image(fp)
            if arr is None:
                continue
            arr = self._normalize_channels(arr)
            if target_w is None:
                target_w = arr.shape[1]
            elif arr.shape[1] != target_w:
                # Keep a consistent width so the pages tile cleanly.
                h = int(round(arr.shape[0] * target_w / arr.shape[1]))
                arr = cv2.resize(arr, (target_w, h), interpolation=cv2.INTER_AREA)
            pages.append(arr)
            heights.append(arr.shape[0])
            try:
                self.main.stitch_progress.emit(
                    i + 1, load_total, self.main.tr("Stitching webtoon…")
                )
            except Exception:
                pass
        if not pages:
            return None

        stitched = np.concatenate(pages, axis=0)

        # Safety valve: a single image taller than the chosen limit is split
        # into contiguous page-height chunks so we never blow up memory / export.
        max_strip_h = self.UNLIMITED_MAX_STRIP_H if choice == "unlimited" else self.LIGHT_MAX_STRIP_H
        paths: list[str] = []
        chunk_bounds: list[tuple[int, int]] = []
        if stitched.shape[0] > max_strip_h:
            step = self.CHUNK_H
            n_chunks = (stitched.shape[0] + step - 1) // step
            total_steps = load_total + n_chunks
            idx = 0
            i = 0
            while i < stitched.shape[0]:
                j = min(i + step, stitched.shape[0])
                chunk = stitched[i:j]
                p = os.path.join(self._stitch_temp_dir(), f"webtoon_chunk_{idx:04d}.png")
                imk.write_image(p, chunk)
                paths.append(p)
                chunk_bounds.append((i, j))
                idx += 1
                i = j
                try:
                    self.main.stitch_progress.emit(
                        load_total + idx, total_steps, self.main.tr("Stitching webtoon…")
                    )
                except Exception:
                    pass
        else:
            p = os.path.join(self._stitch_temp_dir(), "webtoon_stitched.png")
            imk.write_image(p, stitched)
            paths = [p]
            chunk_bounds = [(0, stitched.shape[0])]

        # Remember the original per-page heights and the chunk boundaries so we
        # can split the stitched image back into the original pages on export.
        return {
            "paths": paths,
            "heights": heights,
            "chunk_bounds": chunk_bounds,
            "choice": choice,
        }

    def _on_stitch_done(self, result):
        """Called on the GUI thread when background stitching finishes."""
        self.main.webtoon_toggle.blockSignals(False)
        self.main.progress_bar.setVisible(False)
        if not result:
            self._revert_webtoon_toggle()
            return
        self._stitched_orig_heights = result["heights"]
        self._stitched_chunk_bounds = result["chunk_bounds"]
        self._stitch_choice = result["choice"]
        self._init_stitched_files(result["paths"])
        self.main.webtoon_mode = False
        self.main.webtoon_strip = True
        self.main.mark_project_dirty()

    def _on_stitch_error(self, error_tuple):
        """Called on the GUI thread if background stitching raises."""
        self.main.webtoon_toggle.blockSignals(False)
        self.main.progress_bar.setVisible(False)
        self._revert_webtoon_toggle()
        exctype, value, tb = error_tuple
        print(f"Stitching failed: {value}")
        try:
            QtWidgets.QMessageBox.warning(
                self.main,
                self.main.tr("Webtoon mode"),
                self.main.tr("Failed to stitch webtoon:") + f" {value}",
            )
        except Exception:
            pass

    def _revert_webtoon_toggle(self):
        """Put the webtoon toggle back into the off state safely."""
        self.main.webtoon_toggle.blockSignals(True)
        self.main.webtoon_toggle.setChecked(False)
        self.main.webtoon_toggle.blockSignals(False)

    def _init_stitched_files(self, paths: list[str]):
        """Replace the loaded project with the stitched image(s) as regular
        page(s), rebuilding all per-image bookkeeping from scratch."""
        self.main.image_files = list(paths)
        self.main.image_states.clear()
        self.main.image_data.clear()
        self.main.image_history.clear()
        self.main.in_memory_history.clear()
        self.main.current_history_index.clear()
        self.main.undo_stacks.clear()
        self.main.undo_group = QtGui.QUndoGroup(self.main)
        self.main.image_patches.clear()
        self.main.in_memory_patches.clear()
        self.main.displayed_images.clear()
        self.main.loaded_images = []
        self.main.curr_img_idx = -1
        self.main.blk_list = []
        self.main.image_viewer.clear_scene()
        self.main.image_viewer.clear_rectangles(page_switch=True)
        self.main.image_viewer.clear_brush_strokes(page_switch=True)
        self.main.image_viewer.clear_text_items()

        for fp in paths:
            arr = self.main.image_ctrl.load_image(fp)
            self.main.image_data[fp] = arr
            self.main.image_history[fp] = [fp]
            self.main.in_memory_history[fp] = [arr.copy()] if arr is not None else []
            self.main.current_history_index[fp] = 0
            self.main.image_states[fp] = self.main.image_ctrl._build_image_state(
                fp, {}, [], [], False
            )
            stack = QtGui.QUndoStack(self.main)
            stack.cleanChanged.connect(self.main._update_window_modified)
            stack.indexChanged.connect(self.main._bump_dirty_revision)
            self.main.undo_stacks[fp] = stack
            self.main.undo_group.addStack(stack)

        self.main.page_list.blockSignals(True)
        self.main.image_ctrl.refresh_page_list()
        self.main.page_list.blockSignals(False)
        self.main.page_list.setCurrentRow(0)
        self.main.image_viewer.resetTransform()
        self.main.image_viewer.fitInView()
        self.main.mark_project_dirty()
        if self.main.image_files:
            self.main.image_ctrl.display_image(0, switch_page=True)

    def _switch_from_stitched_to_regular(self):
        """Restore the original per-page project that was preserved when we
        entered stitched webtoon mode."""
        src_files = getattr(self, "_webtoon_source_files", None)
        src_states = getattr(self, "_webtoon_source_states", None)
        if not src_files or src_states is None:
            self.main.image_ctrl.clear_state()
            return

        self.main.webtoon_mode = False
        self.main.image_files = list(src_files)
        self.main.image_states = {k: copy.deepcopy(v) for k, v in src_states.items()}

        self.main.image_data.clear()
        self.main.image_history.clear()
        self.main.in_memory_history.clear()
        self.main.current_history_index.clear()
        self.main.undo_stacks.clear()
        self.main.undo_group = QtGui.QUndoGroup(self.main)
        self.main.image_patches.clear()
        self.main.in_memory_patches.clear()
        self.main.displayed_images.clear()
        self.main.loaded_images = []
        self.main.curr_img_idx = -1
        self.main.blk_list = []
        self.main.image_viewer.clear_scene()
        self.main.image_viewer.clear_rectangles(page_switch=True)
        self.main.image_viewer.clear_brush_strokes(page_switch=True)
        self.main.image_viewer.clear_text_items()

        for fp in self.main.image_files:
            arr = self.main.image_ctrl.load_image(fp)
            self.main.image_data[fp] = arr
            self.main.image_history[fp] = [fp]
            self.main.in_memory_history[fp] = [arr.copy()] if arr is not None else []
            self.main.current_history_index[fp] = 0
            stack = QtGui.QUndoStack(self.main)
            stack.cleanChanged.connect(self.main._update_window_modified)
            stack.indexChanged.connect(self.main._bump_dirty_revision)
            self.main.undo_stacks[fp] = stack
            self.main.undo_group.addStack(stack)

        self.main.page_list.blockSignals(True)
        self.main.image_ctrl.refresh_page_list()
        self.main.page_list.blockSignals(False)
        self.main.page_list.setCurrentRow(0)
        self.main.image_viewer.resetTransform()
        self.main.image_viewer.fitInView()
        self.main.mark_project_dirty()
        if self.main.image_files:
            self.main.image_ctrl.display_image(0, switch_page=True)
