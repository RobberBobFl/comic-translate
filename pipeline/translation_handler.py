from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from modules.translation.processor import Translator
from modules.translation.exceptions import LLMInvalidResponseError
from modules.utils.translator_utils import set_upper_case
from modules.utils.language_utils import to_canonical_language_name
from pipeline.webtoon_utils import filter_and_convert_visible_blocks, restore_original_block_coordinates
from .cache_manager import CacheManager

if TYPE_CHECKING:
    from controller import ComicTranslate
    from .main_pipeline import ComicTranslatePipeline

logger = logging.getLogger(__name__)


def _translate_with_retries(translator, blk_list, image, extra_context, scene_description=None):
    """Translate ``blk_list`` retrying transient/server errors and empty results
    up to MAX_ATTEMPTS times (see :func:`modules.utils.retry_utils.retry_translate`).

    Hard failures (network/API errors after retries) propagate to the caller so
    the controller can surface them with the proper message. When the blocks end
    up entirely untranslated after all retries (e.g. malformed/empty LLM JSON),
    raise :class:`LLMInvalidResponseError` for the dedicated popup.
    """
    from modules.utils.retry_utils import retry_translate

    retry_translate(
        translator, blk_list, image, extra_context,
        scene_description=scene_description,
    )
    if blk_list and all(not (getattr(b, "translation", "") or "").strip() for b in blk_list):
        raise LLMInvalidResponseError(
            "LLM response could not be parsed as JSON or returned empty results after retries"
        )


class TranslationHandler:
    """Handles translation processing with caching support."""
    
    def __init__(
            self, 
            main_page: ComicTranslate, 
            cache_manager: CacheManager, 
            pipeline: ComicTranslatePipeline,
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.pipeline = pipeline

    def translate_image(self, single_block=False):
        source_lang = to_canonical_language_name(
            self.main_page.s_combo.currentText(),
            self.main_page.lang_mapping,
        )
        target_lang = to_canonical_language_name(
            self.main_page.t_combo.currentText(),
            self.main_page.lang_mapping,
        )
        if self.main_page.image_viewer.hasPhoto() and self.main_page.blk_list:
            settings_page = self.main_page.settings_page
            image = self.main_page.image_viewer.get_image_array()
            llm_settings = settings_page.get_llm_settings()
            extra_context = llm_settings['extra_context']
            system_prompt = llm_settings.get('system_prompt', '')
            translator_key = settings_page.get_tool_selection('translator')

            upper_case = settings_page.ui.uppercase_checkbox.isChecked()

            translator = Translator(self.main_page, source_lang, target_lang)
            current_path = None
            if 0 <= self.main_page.curr_img_idx < len(self.main_page.image_files):
                current_path = self.main_page.image_files[self.main_page.curr_img_idx]
            state = self.main_page.image_states.get(current_path, {})
            scene_description = ""
            if (
                llm_settings.get("use_scene_description", False)
                and translator.is_llm_engine
            ):
                scene_description = state.get("scene_description", "") or ""
            
            # Get translation cache key
            translation_cache_key = self.cache_manager._get_translation_cache_key(
                image,
                source_lang,
                target_lang,
                translator_key,
                extra_context,
                system_prompt,
                settings=settings_page,
                scene_description=scene_description,
            )
            
            if single_block:
                blk = self.pipeline.get_selected_block()
                if blk is None:
                    return
                
                # Check if block already has translation to avoid redundant processing
                if hasattr(blk, 'translation') and blk.translation and blk.translation.strip():
                    return
                
                # Check if we have cached translation results for this image/translator/language combination
                if self.cache_manager._is_translation_cached(translation_cache_key):
                    # Check if block exists in cache and source text matches
                    cached_translation = self.cache_manager._get_cached_translation_for_block(translation_cache_key, blk)
                    if cached_translation is not None:  # Block was processed and source text matches
                        blk.translation = cached_translation
                        logger.info(f"Using cached translation result for block: '{cached_translation}'")
                        set_upper_case([blk], upper_case)
                        return
                    else:
                        logger.info("Block not found in cache or source text changed, processing single block...")
                    
                    # If we reach here, need to process the block
                    single_block_list = [blk]
                    _translate_with_retries(
                        translator,
                        single_block_list,
                        image,
                        extra_context,
                        scene_description=scene_description,
                    )
                    
                    # Update the cache with this new result using the cache manager's method
                    self.cache_manager.update_translation_cache_for_block(translation_cache_key, blk)
                    
                    logger.info(f"Processed single block and updated cache: '{blk.translation}'")
                    set_upper_case([blk], upper_case)
                else:
                    # Run translation on all blocks and cache the results
                    logger.info("No cached translation results found, running translation on entire page...")
                    # Create a mapping between original blocks and their copies
                    all_blocks_copy = []
                    
                    for original_blk in self.main_page.blk_list:
                        copy_blk = original_blk.deep_copy()
                        all_blocks_copy.append(copy_blk)
                    
                    if all_blocks_copy:  
                        _translate_with_retries(
                            translator,
                            all_blocks_copy,
                            image,
                            extra_context,
                            scene_description=scene_description,
                        )
                        # Cache using the original blocks to maintain consistent IDs
                        self.cache_manager._cache_translation_results(translation_cache_key, self.main_page.blk_list, all_blocks_copy)
                        cached_translation = self.cache_manager._get_cached_translation_for_block(translation_cache_key, blk)
                        blk.translation = cached_translation
                        logger.info(f"Cached translation results and extracted translation for block: {cached_translation}")
                    
                    set_upper_case([blk], upper_case)
            else:
                # For full page translation, check if we can use cached results
                if self.cache_manager._can_serve_all_blocks_from_translation_cache(translation_cache_key, self.main_page.blk_list):
                    # All blocks can be served from cache with matching source text
                    self.cache_manager._apply_cached_translations_to_blocks(translation_cache_key, self.main_page.blk_list)
                    logger.info(f"Using cached translation results for all {len(self.main_page.blk_list)} blocks")
                else:
                    # Need to run translation and cache results
                    _translate_with_retries(
                        translator,
                        self.main_page.blk_list,
                        image,
                        extra_context,
                        scene_description=scene_description,
                    )
                    self.cache_manager._cache_translation_results(translation_cache_key, self.main_page.blk_list)
                    logger.info("Translation completed and cached for %d blocks", len(self.main_page.blk_list))
                
                set_upper_case(self.main_page.blk_list, upper_case)

    def translate_webtoon_visible_area(self, single_block=False):
        """Perform translation on the visible area in webtoon mode."""
        source_lang = to_canonical_language_name(
            self.main_page.s_combo.currentText(),
            self.main_page.lang_mapping,
        )
        target_lang = to_canonical_language_name(
            self.main_page.t_combo.currentText(),
            self.main_page.lang_mapping,
        )
        
        if not (self.main_page.image_viewer.hasPhoto() and 
                self.main_page.webtoon_mode):
            logger.warning("translate_webtoon_visible_area called but not in webtoon mode")
            return
        
        # Get the visible area image and mapping data
        visible_image, mappings = self.main_page.image_viewer.get_visible_area_image()
        if visible_image is None or not mappings:
            logger.warning("No visible area found for translation")
            return
        
        # Filter blocks to only those in the visible area and convert coordinates
        visible_blocks = filter_and_convert_visible_blocks(
            self.main_page, self.pipeline, mappings, single_block
        )
        if not visible_blocks:
            logger.info("No blocks found in visible area")
            return
        
        # Perform translation on the visible image with filtered blocks
        settings_page = self.main_page.settings_page
        llm_settings = settings_page.get_llm_settings()
        extra_context = llm_settings['extra_context']
        upper_case = settings_page.ui.uppercase_checkbox.isChecked()
        current_path = None
        if 0 <= self.main_page.curr_img_idx < len(self.main_page.image_files):
            current_path = self.main_page.image_files[self.main_page.curr_img_idx]
        state = self.main_page.image_states.get(current_path, {})
        
        translator = Translator(self.main_page, source_lang, target_lang)
        scene_description = ""
        if llm_settings.get("use_scene_description", False) and translator.is_llm_engine:
            scene_description = state.get("scene_description", "") or ""
        _translate_with_retries(
            translator,
            visible_blocks,
            visible_image,
            extra_context,
            scene_description=scene_description,
        )

        # Translation is set, now restore original coordinates
        restore_original_block_coordinates(visible_blocks)
        
        # Apply upper case if needed
        set_upper_case(visible_blocks, upper_case)

        # Persist the translation back into the page state. In webtoon mode
        # main.blk_list is only a copy of the saved block list, so without this
        # the translations are lost the next time the current page is rebuilt
        # from state.
        try:
            self.main_page.manual_workflow_ctrl.sync_blk_list_to_state()
        except Exception:
            logger.exception("Failed to sync webtoon translation to page state")

        logger.info(f"Translation completed for {len(visible_blocks)} blocks in visible area")
