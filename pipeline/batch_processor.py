from __future__ import annotations

import os
import json
import requests
import logging
import traceback
import imkit as imk
import time
from typing import TYPE_CHECKING
from datetime import datetime
from typing import List
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor

from modules.detection.processor import TextBlockDetector
from modules.translation.processor import Translator
from modules.translation.scene_analyzer import SceneAnalyzer
from modules.utils.textblock import sort_blk_list
from modules.utils.pipeline_config import get_config
from modules.utils.image_utils import generate_mask, get_smart_text_color
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.translator_utils import get_context_entries, get_raw_translation, get_raw_text, format_translations, is_renderable_translation
from modules.rendering.render import get_best_render_area, pyside_word_wrap, is_vertical_block, render_box_for_block
from modules.utils.device import resolve_device
from modules.utils.exceptions import InsufficientCreditsException
from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.messages import Messages
from .cache_manager import CacheManager
from .block_detection import BlockDetectionHandler
from .inpainting import InpaintingHandler, call_inpaint_image
from .ocr_handler import OCRHandler
from app.translation_memory import get_translation_memory

if TYPE_CHECKING:
    from controller import ComicTranslate

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of comic translation."""
    
    def __init__(
            self, 
            main_page: ComicTranslate, 
            cache_manager: CacheManager, 
            block_detection_handler: BlockDetectionHandler, 
            inpainting_handler: InpaintingHandler, 
            ocr_handler: OCRHandler 
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        # Use shared handlers from the main pipeline
        self.block_detection = block_detection_handler
        self.inpainting = inpainting_handler
        self.ocr_handler = ocr_handler

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        logger.info("Skipping fallback translated image save for '%s'.", base_name)

    def emit_progress(self, index, total, step, steps, change_name):
        """Wrapper around main_page.progress_update.emit that logs a human-readable stage."""
        stage_map = {
            0: 'start-image',
            1: 'text-block-detection',
            2: 'ocr-processing',
            3: 'translation',
            5: 'pre-inpaint-setup',
            7: 'inpainting',
            9: 'text-rendering-prepare',
            10: 'save-and-finish',
        }
        stage_name = stage_map.get(step, f'stage-{step}')
        logger.info(f"Progress: image_index={index}/{total} step={step}/{steps} ({stage_name}) change_name={change_name}")
        self.main_page.progress_update.emit(index, total, step, steps, change_name)

    def log_skipped_image(self, directory, timestamp, image_path, reason="", full_traceback=""):
        # Deprecated: skip details are captured by batch reporting/UI signals.
        return

    def _is_cancelled(self) -> bool:
        worker = getattr(self.main_page, "current_worker", None)
        return bool(worker and worker.is_cancelled)

    def batch_process(self, selected_paths: List[str] = None):
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total_images = len(image_list)
        batch_report = []
        try:
            if self.main_page.file_handler.should_pre_materialize(image_list):
                count = self.main_page.file_handler.pre_materialize(image_list)
                logger.info("Batch pre-materialized %d paths before full-run processing.", count)
        except Exception:
            logger.debug("Batch pre-materialization failed; continuing lazily.", exc_info=True)

        # Sliding context window: the last translated lines are carried over to
        # the next page so names, pronouns and tone stay consistent through the
        # chapter. Lives for one run only, and is dropped at a source boundary.
        sliding_buffer: list[dict] = []
        buffer_source = None

        for index, image_path in enumerate(image_list):
            if self._is_cancelled():
                return

            file_on_display = self.main_page.image_files[self.main_page.curr_img_idx]

            # index, step, total_steps, change_name
            self.emit_progress(index, total_images, 0, 10, True)

            settings_page = self.main_page.settings_page
            source_lang = self.main_page.image_states[image_path]['source_lang']
            target_lang = self.main_page.image_states[image_path]['target_lang']

            trg_lng_cd = get_language_code(target_lang)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
            extension = os.path.splitext(image_path)[1]
            directory = os.path.dirname(image_path)

            archive_bname = ""
            for archive in self.main_page.file_handler.archive_info:
                images = archive['extracted_images']
                archive_path = archive['archive_path']

                for img_pth in images:
                    if img_pth == image_path:
                        directory = os.path.dirname(archive_path)
                        archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()

            # A new archive/folder or a different language pair is a different
            # story: carrying context across would poison the translation.
            page_source = (archive_bname or directory, source_lang, target_lang)
            if page_source != buffer_source:
                sliding_buffer.clear()
                buffer_source = page_source

            ensure_path_materialized(image_path)
            image = imk.read_image(image_path)

            # skip UI-skipped images
            state = self.main_page.image_states.get(image_path, {})
            if state.get('skip', False):
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.log_skipped_image(directory, timestamp, image_path, "User-skipped")
                continue

            # Text Block Detection
            self.emit_progress(index, total_images, 1, 10, False)
            if self._is_cancelled():
                return

            # Use the shared block detector from the handler
            if self.block_detection.block_detector_cache is None:
                self.block_detection.block_detector_cache = TextBlockDetector(settings_page)
            
            blk_list = self.block_detection.block_detector_cache.detect(image)

            self.emit_progress(index, total_images, 2, 10, False)
            if self._is_cancelled():
                return

            self.block_detection.annotate_language_if_auto(image, blk_list, source_lang)

            if blk_list:
                # Get ocr cache key for batch processing
                ocr_model = settings_page.get_tool_selection('ocr')
                device = resolve_device(settings_page.is_gpu_enabled())
                cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device, settings=settings_page)
                # Use the shared OCR processor from the handler
                self.ocr_handler.ocr.initialize(self.main_page, source_lang)
                try:
                    # Retry transient server errors and empty bubbles up to
                    # MAX_ATTEMPTS times before giving up on the page.
                    from modules.utils.retry_utils import retry_ocr
                    retry_ocr(self.ocr_handler.ocr, image, blk_list, label=f"OCR:{base_name}")

                    # Cache the OCR results for potential future use
                    self.cache_manager._cache_ocr_results(cache_key, self.main_page.blk_list)
                    rtl = True if source_lang == 'Japanese' else False
                    blk_list = sort_blk_list(blk_list, rtl)
                    
                except InsufficientCreditsException:
                    raise
                except Exception as e:
                    # if it's a connection/network error, give a short message
                    if isinstance(e, requests.exceptions.ConnectionError):
                        err_msg = QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
                    # if it's an HTTPError, try to pull the "error_description" field
                    elif isinstance(e, requests.exceptions.HTTPError):
                        status_code = e.response.status_code if e.response is not None else 500
                        if status_code >= 500:
                            err_msg = Messages.get_server_error_text(status_code, context='ocr')
                        else:
                            try:
                                err_json = e.response.json()
                                if "detail" in err_json and isinstance(err_json["detail"], dict):
                                    err_msg = err_json["detail"].get("error_description", str(e))
                                else:
                                    err_msg = err_json.get("error_description", str(e))
                            except Exception:
                                err_msg = str(e)
                    else:
                        err_msg = str(e)

                    logger.exception(f"OCR processing failed: {err_msg}")
                    reason = f"OCR: {err_msg}"
                    full_traceback = traceback.format_exc()
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "OCR", err_msg)
                    self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                    batch_report.append({
                        "page": os.path.basename(image_path),
                        "page_index": index,
                        "total_blocks": len(blk_list),
                        "empty_ocr": [(list(blk.xyxy), getattr(blk, 'text', '')) for blk in blk_list],
                        "empty_translation": [],
                        "skipped": True,
                        "skip_reason": f"OCR: {err_msg}",
                    })
                    continue
            else:
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Text Blocks", "")
                self.log_skipped_image(directory, timestamp, image_path, "No text blocks detected")
                continue

            self.emit_progress(index, total_images, 3, 10, False)
            if self._is_cancelled():
                return

            # Get Translations/ Export if selected
            llm_settings = settings_page.get_llm_settings()
            extra_context = llm_settings['extra_context']
            system_prompt = llm_settings.get('system_prompt', '')
            translator_key = settings_page.get_tool_selection('translator')
            translator = Translator(self.main_page, source_lang, target_lang)
            use_scene_description = bool(llm_settings.get("use_scene_description", False))
            scene_description = ""
            scene_block_metadata = {}
            if use_scene_description and translator.is_llm_engine:
                scene_description = state.get("scene_description", "") or ""
                scene_block_metadata = state.get("scene_block_metadata", {}) or {}
                if not scene_description:
                    analyzer = SceneAnalyzer.from_settings(settings_page)
                    source_text = SceneAnalyzer.format_source_blocks(
                        blk_list, include_coords=True
                    )
                    scene_description = (
                        analyzer.analyze(
                            image,
                            source_text=source_text,
                            is_webtoon=getattr(self.main_page, "webtoon_mode", False),
                        )
                        or ""
                    )
                    if scene_description:
                        state["scene_description"] = scene_description
                        if blk_list:
                            metadata, match_log = SceneAnalyzer.match_scene_metadata(
                                scene_description, blk_list
                            )
                            for line in match_log:
                                logger.debug("Scene metadata: %s", line)
                            if metadata:
                                scene_block_metadata = {
                                    f"block_{k}": v
                                    for k, v in sorted(metadata.items())
                                }
                                state["scene_block_metadata"] = scene_block_metadata
            
            batch_size = llm_settings.get('batch_size', 5)
            context_window = llm_settings.get('context_window', 8)
            
            # Get translation cache key for batch processing
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
            
            try:
                if self.cache_manager._can_serve_all_blocks_from_translation_cache(translation_cache_key, blk_list):
                    # Same page, same translator, same settings: a re-run of the
                    # chapter costs nothing.
                    self.cache_manager._apply_cached_translations_to_blocks(translation_cache_key, blk_list)
                    logger.info("Using cached translations for '%s'", base_name)
                else:
                    context_blocks = sliding_buffer[-context_window:] if context_window else None
                    # Retry transient server errors and empty bubbles up to
                    # MAX_ATTEMPTS times. Hard failures raise and are handled by
                    # the except block below (page skipped); residual empty
                    # bubbles are reported in the final batch report instead of
                    # skipping the whole page.
                    from modules.utils.retry_utils import retry_translate
                    retry_translate(
                        translator, blk_list, image, extra_context,
                        context_blocks=context_blocks, batch_size=batch_size,
                        scene_description=scene_description,
                        scene_block_metadata=scene_block_metadata,
                        label=f"Translation:{base_name}",
                    )
                    # Cache the translation results for potential future use
                    self.cache_manager._cache_translation_results(translation_cache_key, blk_list)

                if context_window:
                    sliding_buffer.extend(get_context_entries(blk_list))
                    del sliding_buffer[:-context_window]
                # Collect translation memory: capture each block's source + model
                # output once (skips blanks; never touches user final_output).
                get_translation_memory().capture_translated_blocks(image_path, blk_list)
            except InsufficientCreditsException:
                raise
            except Exception as e:
                # if it's a connection/network error, give a short message
                if isinstance(e, requests.exceptions.ConnectionError):
                    err_msg = QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
                # if it's an HTTPError, try to pull the "error_description" field
                elif isinstance(e, requests.exceptions.HTTPError):
                    status_code = e.response.status_code if e.response is not None else 500
                    if status_code >= 500:
                        err_msg = Messages.get_server_error_text(status_code, context='translation')
                    else:
                        try:
                            err_json = e.response.json()
                            if "detail" in err_json and isinstance(err_json["detail"], dict):
                                err_msg = err_json["detail"].get("error_description", str(e))
                            else:
                                err_msg = err_json.get("error_description", str(e))
                        except Exception:
                            err_msg = str(e)
                else:
                    err_msg = str(e)

                if self.progress_callback:
                    self.progress_callback(error=err_msg)
                logger.exception(f"Translation failed: {err_msg}")
                reason = f"Translator: {err_msg}"
                full_traceback = traceback.format_exc()
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", err_msg)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                batch_report.append({
                    "page": os.path.basename(image_path),
                    "page_index": index,
                    "total_blocks": len(blk_list),
                    "empty_ocr": [],
                    "empty_translation": [
                        (list(blk.xyxy), getattr(blk, 'translation', ''))
                        for blk in blk_list
                        if not (blk.translation or "").strip()
                    ],
                    "skipped": True,
                    "skip_reason": f"Translator: {err_msg}",
                })
                continue

            if self._is_cancelled():
                return

            entire_raw_text = get_raw_text(blk_list)
            entire_translated_text = get_raw_translation(blk_list)

            # Parse JSON strings and check if they're empty objects or invalid
            try:
                raw_text_obj = json.loads(entire_raw_text)
                translated_text_obj = json.loads(entire_translated_text)
                
                if (not raw_text_obj) or (not translated_text_obj):
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "Translator", "")
                    self.log_skipped_image(directory, timestamp, image_path, "Translator: empty JSON")
                    batch_report.append({
                        "page": os.path.basename(image_path),
                        "page_index": index,
                        "total_blocks": len(blk_list),
                        "empty_ocr": [],
                        "empty_translation": [],
                        "skipped": True,
                        "skip_reason": "Translator: empty JSON",
                    })
                    continue
            except json.JSONDecodeError as e:
                # Handle invalid JSON
                error_message = str(e)
                reason = f"Translator: JSONDecodeError: {error_message}"
                logger.exception(reason)
                full_traceback = traceback.format_exc()
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", error_message)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                batch_report.append({
                    "page": os.path.basename(image_path),
                    "page_index": index,
                    "total_blocks": len(blk_list),
                    "empty_ocr": [],
                    "empty_translation": [],
                    "skipped": True,
                    "skip_reason": reason,
                })
                continue

            export_settings = settings_page.get_export_settings()

            if export_settings['export_raw_text']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "raw_texts", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                with open(
                    os.path.join(path, os.path.splitext(os.path.basename(image_path))[0] + "_raw.json"),
                    'w',
                    encoding='UTF-8',
                ) as file:
                    file.write(entire_raw_text)

            if export_settings['export_translated_text']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_texts", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                with open(
                    os.path.join(path, os.path.splitext(os.path.basename(image_path))[0] + "_translated.json"),
                    'w',
                    encoding='UTF-8',
                ) as file:
                    file.write(entire_translated_text)

            self.emit_progress(index, total_images, 5, 10, False)
            if self._is_cancelled():
                return

            # Clean Image of text
            config = get_config(settings_page)
            
            # Filter blocks to only inpaint if both OCR text and Translation are non-empty
            # and the translation will actually be rendered (single-character translations
            # like an echoed "?" are skipped at render time).
            inpaint_blk_list = [
                blk for blk in blk_list
                if blk.text and blk.text.strip() and blk.translation and blk.translation.strip()
                and is_renderable_translation(blk.translation)
            ]
            
            logger.info("pre-inpaint: generating mask (inpaint_blk_list=%d blocks out of %d)",
                        len(inpaint_blk_list), len(blk_list))
            t0 = time.time()
            mask = generate_mask(image, inpaint_blk_list)
            t1 = time.time()
            logger.info("pre-inpaint: mask generated in %.2fs (mask shape=%s)", t1 - t0, getattr(mask, 'shape', None))

            self.emit_progress(index, total_images, 7, 10, False)
            if self._is_cancelled():
                return

            inpaint_input_img = call_inpaint_image(
                self.inpainting, image, mask, config,
                blk_list=inpaint_blk_list,
            )
            inpaint_input_img = imk.convert_scale_abs(inpaint_input_img)

            # Saving cleaned image
            patches = self.inpainting.get_inpainted_patches(mask, inpaint_input_img)
            self.main_page.patches_processed.emit(patches, image_path)

            if export_settings['export_inpainted_image']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "cleaned_images", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                imk.write_image(os.path.join(path, f"{base_name}_cleaned{extension}"), inpaint_input_img)

            self.emit_progress(index, total_images, 9, 10, False)
            if self._is_cancelled():
                return

            # Text Rendering
            render_settings = self.main_page.render_settings()
            upper_case = render_settings.upper_case
            outline = render_settings.outline
            format_translations(blk_list, trg_lng_cd, upper_case=upper_case)
            get_best_render_area(blk_list, image, inpaint_input_img)

            font = render_settings.font_family
            setting_font_color = QColor(render_settings.color)

            max_font_size = render_settings.max_font_size
            min_font_size = render_settings.min_font_size
            line_spacing = float(render_settings.line_spacing) 
            outline_width = float(render_settings.outline_width)
            outline_color = QColor(render_settings.outline_color) if outline else None
            bold = render_settings.bold
            italic = render_settings.italic
            underline = render_settings.underline
            alignment_id = render_settings.alignment_id
            alignment = self.main_page.button_to_alignment[alignment_id]
            direction = render_settings.direction
                
            text_items_state = []
            for blk in blk_list:
                # Anchor the translation to the (shrunk) speech bubble when one
                # was detected, so text centers inside the bubble rather than the
                # tight text-line box (horizontal bubbles sit at the bubble top).
                x1, y1, block_width, block_height = render_box_for_block(blk, blk_list)

                translation = blk.translation
                if not is_renderable_translation(translation):
                    continue
                
                # Determine if this block should use vertical rendering
                vertical = is_vertical_block(blk, trg_lng_cd)

                translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                    translation, 
                    font, 
                    block_width, 
                    block_height,
                    line_spacing, 
                    outline_width, 
                    bold, 
                    italic, 
                    underline,
                    alignment, 
                    direction, 
                    max_font_size, 
                    min_font_size,
                    vertical,
                    is_no_space_lang(trg_lng_cd),
                    return_metrics=True
                )
                
                # Display text if on current page  
                if image_path == file_on_display:
                    self.main_page.blk_rendered.emit(translation, font_size, blk, image_path)

                # Smart Color Override
                font_color = get_smart_text_color(blk.font_color, setting_font_color)

                # Use TextItemProperties for consistent text item creation.
                # pyside_word_wrap was called with the block width, so
                # rendered_height is the exact wrapped height at that width.
                # Widen the document to the block width and vertically center
                # the text via v_margin (every non-vertical block, rotated
                # included). Vertical text keeps the rendered-size box.
                if not vertical:
                    item_width = block_width
                    item_v_margin = max(0.0, (block_height - rendered_height) / 2.0)
                else:
                    item_width = rendered_width
                    item_v_margin = 0.0

                text_props = TextItemProperties(
                    text=translation,
                    plain_text=translation,
                    font_family=font,
                    font_size=font_size,
                    text_color=font_color,
                    alignment=alignment,
                    line_spacing=line_spacing,
                    outline_color=outline_color,
                    outline_width=outline_width,
                    bold=bold,
                    italic=italic,
                    underline=underline,
                    position=(x1, y1),
                    rotation=blk.angle,
                    scale=1.0,
                    transform_origin=blk.tr_origin_point,
                    width=item_width,
                    height=rendered_height,
                    v_margin=item_v_margin,
                    direction=direction,
                    vertical=vertical,
                    selection_outlines=[
                        OutlineInfo(0, len(translation), 
                        outline_color, 
                        outline_width, 
                        OutlineType.Full_Document)
                    ] if outline else [],
                )
                text_items_state.append(text_props.to_dict())

            self.main_page.image_states[image_path]['viewer_state'].update({
                'text_items_state': text_items_state
                })
            
            self.main_page.image_states[image_path]['viewer_state'].update({
                'push_to_stack': True
                })
            
            self.emit_progress(index, total_images, 9, 10, False)
            if self._is_cancelled():
                return

            # Saving blocks with texts to history
            self.main_page.image_states[image_path].update({
                'blk_list': blk_list                   
            })

            # Notify UI that this page's render state is finalized.
            # This enables a deterministic refresh when the user navigates to this page
            # during processing and misses live blk_rendered events.
            self.main_page.render_state_ready.emit(image_path)

            if image_path == file_on_display:
                self.main_page.blk_list = blk_list

            self.emit_progress(index, total_images, 10, 10, False)

            # Collect batch report data
            empty_ocr_blocks = [
                (list(blk.xyxy), getattr(blk, 'text', ''))
                for blk in blk_list
                if not (blk.text or "").strip()
            ]
            empty_tr_blocks = [
                (list(blk.xyxy), getattr(blk, 'translation', ''))
                for blk in blk_list
                if not (blk.translation or "").strip()
            ]
            batch_report.append({
                "page": os.path.basename(image_path),
                "page_index": index,
                "total_blocks": len(blk_list),
                "empty_ocr": empty_ocr_blocks,
                "empty_translation": empty_tr_blocks,
            })

        # Final batch report
        self._log_batch_report(batch_report)

    def _log_batch_report(self, batch_report: list):
        """Log a summary report of empty OCR/translation blocks across all pages."""
        if not batch_report:
            return

        issues = [p for p in batch_report if p.get("empty_ocr") or p.get("empty_translation") or p.get("skipped")]
        total_empty_ocr = sum(len(p.get("empty_ocr", [])) for p in batch_report)
        total_empty_tr = sum(len(p.get("empty_translation", [])) for p in batch_report)
        skipped = [p for p in batch_report if p.get("skipped")]

        lines = []
        lines.append("=" * 60)
        lines.append(f"BATCH REPORT: {len(batch_report)} pages processed")
        lines.append("=" * 60)

        if not issues:
            lines.append("✓ All pages OK — no empty OCR or translations")
        else:
            ok_count = len(batch_report) - len(issues)
            lines.append(f"✓ {ok_count} page(s) fully OK")
            lines.append(f"⚠ {len(issues)} page(s) with issues:\n")
            for p in issues:
                page_name = p["page"]
                total = p["total_blocks"]
                e_ocr = len(p.get("empty_ocr", []))
                e_tr = len(p.get("empty_translation", []))
                lines.append(f"  {page_name} ({total} blocks)")
                if p.get("skipped"):
                    lines.append(f"    SKIPPED: {p.get('skip_reason', 'unknown error')}")
                if e_ocr:
                    lines.append(f"    OCR empty: {e_ocr} block(s)")
                    for coords, _ in p["empty_ocr"]:
                        lines.append(f"      → {coords}")
                if e_tr:
                    lines.append(f"    Translation empty: {e_tr} block(s)")
                    for coords, _ in p["empty_translation"]:
                        lines.append(f"      → {coords}")
                lines.append("")

        lines.append(f"TOTAL: {total_empty_ocr} empty OCR, {total_empty_tr} empty translation, {len(skipped)} skipped across {len(batch_report)} pages")
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        logger.warning(report_text)

        # Emit signal for UI dialog
        try:
            self.main_page.batch_report_ready.emit(report_text)
        except Exception as e:
            logger.error("Failed to emit batch_report_ready signal: %s", e)

