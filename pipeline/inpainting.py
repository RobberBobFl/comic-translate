import numpy as np
import time
import logging
import inspect
import imkit as imk

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush

from modules.utils.device import resolve_device
from modules.utils.image_utils import (
    build_block_mask_data, build_bubble_clip_mask, clip_mask_to_bubble,
    clip_mask_components_to_bubble, _resolve_block_crop_bounds,
)
from modules.utils.pipeline_config import inpaint_map, get_config, get_inpainter_backend
from modules.utils.textblock import adjust_text_line_coordinates
from pipeline.inpainting_boxes import merge_overlapping_padded_boxes
from pipeline.webtoon_utils import filter_and_convert_visible_blocks, restore_original_block_coordinates

logger = logging.getLogger(__name__)

FAST_FILL_BUBBLE_INSET = 7


def call_inpaint_image(
    inpainting_handler,
    image: np.ndarray,
    mask: np.ndarray,
    config,
    blk_list: list | None = None,
):
    inpaint_image = inpainting_handler.inpaint_image
    try:
        parameters = inspect.signature(inpaint_image).parameters
        accepts_blk_list = (
            "blk_list" in parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        )
    except (TypeError, ValueError):
        accepts_blk_list = True

    if accepts_blk_list:
        return inpaint_image(image, mask, config, blk_list=blk_list)
    return inpaint_image(image, mask, config)


class InpaintingHandler:
    """Handles image inpainting functionality."""
    
    def __init__(self, main_page):
        self.main_page = main_page
        self.inpainter_cache = None
        self.cached_inpainter_key = None

    def _ensure_inpainter(self):
        settings_page = self.main_page.settings_page
        inpainter_key = settings_page.get_tool_selection('inpainter')
        if self.inpainter_cache is None or self.cached_inpainter_key != inpainter_key:
            backend = get_inpainter_backend(inpainter_key)
            device = resolve_device(settings_page.is_gpu_enabled(), backend)
            InpainterClass = inpaint_map[inpainter_key]
            logger.info("pre-inpaint: initializing inpainter '%s' on device %s", inpainter_key, device)
            t0 = time.time()
            self.inpainter_cache = InpainterClass(device, backend=backend)
            self.cached_inpainter_key = inpainter_key
            logger.info("pre-inpaint: inpainter initialized in %.2fs", time.time() - t0)
        return self.inpainter_cache

    def manual_inpaint(self, respect_manual_mask: bool = True):
        image_viewer = self.main_page.image_viewer
        settings_page = self.main_page.settings_page
        mask = image_viewer.get_mask_for_inpainting()
        
        # Handle webtoon mode vs regular mode differently
        mappings = None
        if self.main_page.webtoon_mode:
            # In webtoon mode, use visible area image for inpainting
            image, mappings = image_viewer.get_visible_area_image()
        else:
            # Regular mode - get the full image
            image = image_viewer.get_image_array()

        if image is None or mask is None:
            return None

        config = get_config(settings_page)
        inpaint_blocks = self._get_manual_fast_fill_blocks(mappings)
        inpaint_input_img = self.inpaint_image(
            image,
            mask,
            config,
            blk_list=inpaint_blocks or None,
            respect_manual_mask=respect_manual_mask,
        )
        inpaint_input_img = imk.convert_scale_abs(inpaint_input_img) 

        return inpaint_input_img

    def _qimage_to_np(self, qimg: QImage):
        if qimg.width() <= 0 or qimg.height() <= 0:
            return np.zeros((max(1, qimg.height()), max(1, qimg.width())), dtype=np.uint8)
        ptr = qimg.constBits()
        arr = np.array(ptr).reshape(qimg.height(), qimg.bytesPerLine())
        return arr[:, :qimg.width()]

    def _generate_mask_from_saved_strokes(self, strokes: list[dict], image: np.ndarray):
        if image is None or not strokes:
            return None
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return None

        human_qimg = QImage(width, height, QImage.Format_Grayscale8)
        gen_qimg = QImage(width, height, QImage.Format_Grayscale8)
        human_qimg.fill(0)
        gen_qimg.fill(0)

        human_painter = QPainter(human_qimg)
        gen_painter = QPainter(gen_qimg)

        human_painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        gen_painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        human_painter.setBrush(QBrush(QColor(255, 255, 255)))
        gen_painter.setBrush(QBrush(QColor(255, 255, 255)))

        has_any = False
        for stroke in strokes:
            path = stroke.get('path')
            if path is None:
                continue
            brush_hex = QColor(stroke.get('brush', '#00000000')).name(QColor.HexArgb)
            if brush_hex == "#80ff0000":
                gen_painter.drawPath(path)
                has_any = True
                continue

            width_px = max(1, int(stroke.get('width', 25)))
            human_pen = QPen(QColor(255, 255, 255), width_px, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            human_painter.setPen(human_pen)
            human_painter.drawPath(path)
            has_any = True

        human_painter.end()
        gen_painter.end()

        if not has_any:
            return None

        human_mask = self._qimage_to_np(human_qimg)
        gen_mask = self._qimage_to_np(gen_qimg)
        kernel = np.ones((5, 5), np.uint8)
        human_mask = imk.dilate(human_mask, kernel, iterations=2)
        gen_mask = imk.dilate(gen_mask, kernel, iterations=3)
        mask = np.where((human_mask > 0) | (gen_mask > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            return None
        return mask

    def _get_manual_fast_fill_blocks(self, mappings: list[dict] | None = None) -> list:
        blocks = getattr(self.main_page, "blk_list", None) or []
        if not blocks:
            return []
        if not getattr(self.main_page, "webtoon_mode", False):
            return blocks
        if not mappings:
            return []

        visible_blocks = []
        try:
            visible_blocks = filter_and_convert_visible_blocks(
                self.main_page,
                self.main_page.pipeline,
                mappings,
                single_block=False,
            )
            return [
                block.deep_copy() if hasattr(block, "deep_copy") else block
                for block in visible_blocks
            ]
        finally:
            restore_original_block_coordinates(visible_blocks)

    def _get_regular_patches(self, mask: np.ndarray, inpainted_image: np.ndarray):
        contours, _ = imk.find_contours(mask)
        if not contours:
            return []
        boxes = [imk.bounding_rect(c) for c in contours]
        merged_boxes = merge_overlapping_padded_boxes(boxes, inpainted_image.shape, pad=8)
        if len(merged_boxes) < len(boxes):
            logger.info("Inpaint hybrid: merged %d mask contours into %d output patches", len(boxes), len(merged_boxes))
        patches = []
        for x1, y1, x2, y2 in merged_boxes:
            w = x2 - x1
            h = y2 - y1
            patch = inpainted_image[y1:y2, x1:x2]
            patches.append({'bbox': [x1, y1, w, h], 'image': patch.copy()})
        return patches

    def _inpaint_full_image(self, image: np.ndarray, mask: np.ndarray, config):
        if image is None:
            return None
        if mask is None or not np.any(mask):
            return image.copy()
        return self.inpainter_cache(image, mask, config)

    def _inpaint_by_patches(self, image: np.ndarray, mask: np.ndarray, config):
        inpainted_image = image.copy()
        contours, _ = imk.find_contours(mask)
        
        if not contours:
            return inpainted_image

        # 1. Extract and merge bounding boxes
        boxes = [imk.bounding_rect(c) for c in contours]
        merged_boxes = merge_overlapping_padded_boxes(boxes, image.shape)

        # 2. Patch inference
        for x1, y1, x2, y2 in merged_boxes:
            img_patch = image[y1:y2, x1:x2]
            mask_patch = mask[y1:y2, x1:x2]
            
            patch_inpainted = self.inpainter_cache(img_patch, mask_patch, config)
            inpainted_image[y1:y2, x1:x2] = patch_inpainted
            
        return inpainted_image

    def _get_fast_fill_regions(
        self,
        block,
        bounds: tuple[int, int, int, int],
        crop_mask: np.ndarray,
        residual_crop: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if getattr(block, "text_class", None) != "text_bubble":
            return None, None
        if crop_mask is None or residual_crop is None:
            return None, None

        masked_region = (crop_mask > 0) & (residual_crop > 0)
        if not np.any(masked_region):
            return None, None

        h, w = crop_mask.shape[:2]
        x1, y1, _, _ = bounds
        background_region = crop_mask == 0

        text_bounds = getattr(block, "xyxy", None)
        if text_bounds is not None:
            tx1, ty1, tx2, ty2 = [int(round(float(v))) for v in text_bounds]
            lx1 = max(0, min(w, tx1 - x1))
            ly1 = max(0, min(h, ty1 - y1))
            lx2 = max(lx1, min(w, tx2 - x1))
            ly2 = max(ly1, min(h, ty2 - y1))
            block_region = np.zeros((h, w), dtype=bool)
            if lx2 > lx1 and ly2 > ly1:
                block_region[ly1:ly2, lx1:lx2] = True
                # When the user stroke falls on the text itself, prefer sampling
                # only from the text block background. When the stroke is elsewhere
                # in the bubble, keep the full bubble background so those strokes
                # can still be fast-filled instead of falling through to NN inpaint.
                masked_in_text = masked_region & block_region
                masked_pixels = int(np.count_nonzero(masked_region))
                masked_in_text_pixels = int(np.count_nonzero(masked_in_text))
                overlap_ratio = (
                    masked_in_text_pixels / float(masked_pixels)
                    if masked_pixels > 0 else 0.0
                )
                if masked_in_text_pixels >= 8 and overlap_ratio >= 0.6:
                    block_background = background_region & block_region
                    if np.count_nonzero(block_background) >= 32:
                        background_region = block_background

        # Drop a thin halo around the mask so antialiased glyph edges do not vote as background.
        halo_kernel = np.ones((3, 3), np.uint8)
        halo = imk.dilate((crop_mask > 0).astype(np.uint8), halo_kernel, iterations=1) > 0
        safe_background = background_region & ~halo
        if np.count_nonzero(safe_background) >= 32:
            background_region = safe_background

        if np.count_nonzero(background_region) < 32:
            return None, None
        return masked_region, background_region

    def _get_fast_fill_bounds(self, block, image: np.ndarray) -> tuple[int, int, int, int] | None:
        base_bounds = getattr(block, "xyxy", None)
        if base_bounds is None or len(base_bounds) < 4:
            return None

        if getattr(block, "text_class", None) == "text_bubble":
            bubble_bounds = getattr(block, "bubble_xyxy", None)
            if bubble_bounds is not None and len(bubble_bounds) >= 4:
                return adjust_text_line_coordinates(bubble_bounds, 10, 10, image)

        return adjust_text_line_coordinates(base_bounds, 10, 10, image)

    @staticmethod
    def _same_bounds(
        lhs: tuple[int, int, int, int] | None,
        rhs: tuple[int, int, int, int] | None,
    ) -> bool:
        if lhs is None or rhs is None:
            return lhs is rhs
        return tuple(int(v) for v in lhs) == tuple(int(v) for v in rhs)

    def _format_block_debug_label(self, block) -> str:
        text_bounds = getattr(block, "xyxy", None)
        bubble_bounds = getattr(block, "bubble_xyxy", None)
        ocr_text = getattr(block, "text", "") or ""
        if len(ocr_text) > 60:
            ocr_text = ocr_text[:60] + "..."
        return (
            f"class={getattr(block, 'text_class', None)} "
            f"text={tuple(int(round(float(v))) for v in text_bounds) if text_bounds is not None else None} "
            f"bubble={tuple(int(round(float(v))) for v in bubble_bounds) if bubble_bounds is not None else None} "
            f"ocr=\"{ocr_text}\""
        )

    def _summarize_mask_overlap_with_blocks(
        self,
        mask: np.ndarray,
        blk_list: list | None,
        image: np.ndarray,
    ) -> list[str]:
        if mask is None or not np.any(mask) or not blk_list:
            return []

        summaries: list[str] = []
        for idx, block in enumerate(blk_list):
            bounds = self._get_fast_fill_bounds(block, image)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            overlap = int(np.count_nonzero(mask[y1:y2, x1:x2]))
            if overlap <= 0:
                continue
            summaries.append(
                f"block[{idx}] overlap={overlap} bounds={(x1, y1, x2, y2)} {self._format_block_debug_label(block)}"
            )
        return summaries

    def _estimate_fast_fill_color(
        self,
        crop: np.ndarray,
        masked_region: np.ndarray,
        background_region: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        if crop.size == 0 or not np.any(masked_region) or not np.any(background_region):
            return None, "empty-crop-or-regions"

        background_pixels = crop[background_region]
        if background_pixels.shape[0] < 64:
            return None, f"background-too-small:{background_pixels.shape[0]}"

        masked_ratio = float(np.count_nonzero(masked_region)) / float(np.count_nonzero(masked_region | background_region))
        if masked_ratio > 0.90:
            return None, f"masked-ratio-too-high:{masked_ratio:.3f}"

        pixels = background_pixels.reshape(-1, 3)
        pixels_u16 = pixels.astype(np.uint16)
        quantized = pixels_u16 // 16
        keys = (quantized[:, 0] << 8) | (quantized[:, 1] << 4) | quantized[:, 2]
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        if unique_keys.size == 0:
            return None, "no-quantized-clusters"

        min_cluster = max(48, int(pixels.shape[0] * 0.015))
        order = np.argsort(counts)[::-1][:16]
        candidates = []
        for key_index in order:
            count = int(counts[key_index])
            if count < min_cluster:
                continue
            cluster = pixels[inverse == key_index].astype(np.float32)
            if cluster.size == 0:
                continue
            median = np.median(cluster, axis=0)
            p10 = np.percentile(cluster, 10, axis=0)
            p90 = np.percentile(cluster, 90, axis=0)
            spread = p90 - p10
            candidates.append({
                "count": count,
                "coverage": count / float(pixels.shape[0]),
                "median": median,
                "brightness": float(np.mean(median)),
                "spread": float(np.max(spread)),
            })

        if not candidates:
            return None, f"no-candidates:min_cluster={min_cluster}"

        candidates.sort(key=lambda item: item["count"], reverse=True)
        largest = candidates[0]
        bright_candidates = [
            item for item in candidates
            if item["brightness"] >= 150.0 and item["count"] >= max(min_cluster, int(largest["count"] * 0.12))
        ]
        selected = max(bright_candidates, key=lambda item: item["count"]) if bright_candidates else largest

        # Check if the selected color dominates the background region, ignoring dark outlines/borders for light bubbles
        selected_brightness = selected["brightness"]
        if selected_brightness >= 80.0:
            pixel_brightness = 0.299 * pixels[:, 2] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 0]
            valid_pixels = pixels[pixel_brightness >= 50.0]
        else:
            valid_pixels = pixels

        if valid_pixels.size == 0:
            valid_pixels = pixels

        diff = np.abs(valid_pixels - selected["median"])
        is_close = np.all(diff <= 15, axis=1)
        neighborhood_coverage = np.count_nonzero(is_close) / float(valid_pixels.shape[0])

        if neighborhood_coverage < 0.55:
            return None, f"selected-too-low-coverage:coverage={neighborhood_coverage:.3f}"
        if selected["count"] < 96 and selected["coverage"] < 0.03:
            return None, f"selected-too-small:count={selected['count']},coverage={selected['coverage']:.3f}"
        if selected["spread"] > 48.0:
            return None, f"selected-too-wide:spread={selected['spread']:.3f}"

        return selected["median"], (
            "ok:"
            f"count={selected['count']},coverage={selected['coverage']:.3f},"
            f"brightness={selected['brightness']:.1f},spread={selected['spread']:.1f}"
        )

    def _apply_fast_bubble_cleanup(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        blk_list: list | None,
        respect_manual_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if image is None or mask is None or not np.any(mask) or not blk_list:
            return image.copy(), mask.copy(), 0

        cleaned_image = image.copy()
        residual_mask = mask.copy()
        cleaned_blocks = 0

        for idx, block in enumerate(blk_list):
            if getattr(block, "xyxy", None) is None or len(block.xyxy) < 4:
                continue
            if getattr(block, "text_class", None) != "text_bubble" or getattr(block, "bubble_xyxy", None) is None:
                if getattr(block, "manual", False) and getattr(block, "text_class", None) != "text_bubble":
                    block.text_class = "text_bubble"
                    if getattr(block, "bubble_xyxy", None) is None:
                        block.bubble_xyxy = np.array(block.xyxy)
                else:
                    continue
            bounds = self._get_fast_fill_bounds(block, image)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            residual_crop = residual_mask[y1:y2, x1:x2]
            if not np.any(residual_crop):
                continue
            crop_mask = np.where(residual_crop > 0, 255, 0).astype(np.uint8)
            if getattr(block, "text_class", None) == "text_bubble" and getattr(block, "bubble_xyxy", None) is not None:
                crop_mask = clip_mask_components_to_bubble(
                    crop_mask,
                    bounds,
                    block.bubble_xyxy,
                    inset=FAST_FILL_BUBBLE_INSET,
                    image=image,
                    seed_bbox=block.xyxy,
                )
                
            initial_overlap = int(np.count_nonzero(crop_mask))
            if initial_overlap <= 0:
                continue
            success, reason = self._fast_fill_block(cleaned_image, residual_mask, block, bounds, crop_mask)
            if not success:
                # Try dense_bubble_fill for high-density bubbles before
                # falling back to the heavier NN inpainting path.
                dense_ok, dense_reason = self._dense_bubble_fill(
                    image,           # original — clean top/bottom strips
                    cleaned_image,   # write result here
                    residual_mask,
                    block,
                    bounds,
                    crop_mask,
                )
                if dense_ok:
                    success = True
                    reason = f"dense:{dense_reason}"

                    # Dense-fill cleanup: fill remaining residual pixels
                    # inside bubble_clip (or bubble bbox fallback) with
                    # the same background color.
                    dense_residual = residual_mask[y1:y2, x1:x2]
                    dense_remaining = int(np.count_nonzero(dense_residual))
                    if dense_remaining > 0:
                        dense_crop = image[y1:y2, x1:x2]
                        dense_crop_h, dense_crop_w = dense_crop.shape[:2]
                        residual_in = dense_residual > 0

                        # 1) Try bubble_clip (segmented mask)
                        dense_clip = build_bubble_clip_mask(
                            (dense_crop_h, dense_crop_w), bounds,
                            block.bubble_xyxy, inset=FAST_FILL_BUBBLE_INSET,
                            image=image, seed_bbox=block.xyxy,
                        )
                        in_clip = (dense_clip & residual_in) if dense_clip is not None else np.zeros_like(residual_in)
                        n_in_clip = int(np.count_nonzero(in_clip))

                        # 2) Fallback: any residual strictly inside bubble bbox
                        n_in_bbox = 0
                        if n_in_clip == 0 and getattr(block, "bubble_xyxy", None) is not None:
                            bx1, by1, bx2, by2 = [int(v) for v in block.bubble_xyxy[:4]]
                            bbox_pad = 2
                            bx1 -= bbox_pad
                            by1 -= bbox_pad
                            bx2 += bbox_pad
                            by2 += bbox_pad
                            bbox_mask = np.zeros_like(residual_in)
                            r_y1 = max(0, by1 - y1)
                            r_y2 = min(dense_crop_h, by2 - y1)
                            r_x1 = max(0, bx1 - x1)
                            r_x2 = min(dense_crop_w, bx2 - x1)
                            if r_y2 > r_y1 and r_x2 > r_x1:
                                bbox_mask[r_y1:r_y2, r_x1:r_x2] = True
                            in_bbox = bbox_mask & residual_in
                            n_in_bbox = int(np.count_nonzero(in_bbox))

                        n_use = n_in_clip if n_in_clip > 0 else n_in_bbox
                        use_mask = in_clip if n_in_clip > 0 else in_bbox if n_in_bbox > 0 else None

                        if use_mask is not None and n_use > 0:
                            # Sample bg from dense_fill region (already clean)
                            bg_px = dense_crop[dense_residual == 0]
                            if bg_px.shape[0] > 0:
                                dense_fill_color = np.median(bg_px, axis=0).astype(np.uint8)
                            else:
                                dense_fill_color = np.array([254, 254, 254], dtype=np.uint8)
                            fill_rgb_full = np.broadcast_to(
                                dense_fill_color, dense_crop.shape
                            ).astype(np.float32)
                            soft = np.zeros_like(dense_residual, dtype=np.float32)
                            soft[use_mask] = 1.0
                            soft = imk.gaussian_blur(
                                (soft * 255).astype(np.uint8), 1.0
                            ).astype(np.float32) / 255.0
                            soft = np.clip(soft, 0.0, 1.0)[..., np.newaxis]
                            cf = dense_crop.astype(np.float32)
                            blended = cf * (1.0 - soft) + fill_rgb_full * soft
                            image[y1:y2, x1:x2] = np.clip(
                                np.round(blended), 0, 255
                            ).astype(np.uint8)
                            dense_residual[use_mask] = 0
                            n_filled = n_use
                        else:
                            n_filled = 0

                        dense_after = int(np.count_nonzero(dense_residual))
                        logger.debug(
                            "Dense fill cleanup: block overlap=%d, "
                            "remaining_before=%d, in_clip=%d, in_bbox_fallback=%d, "
                            "filled_by_bg=%d, sent_to_nn=%d",
                            initial_overlap, dense_remaining,
                            n_in_clip, n_in_bbox, n_filled, dense_after,
                        )
                        if dense_after > 0:
                            ry, rx = np.where(dense_residual > 0)
                            if ry.size > 0:
                                logger.debug(
                                    "  residual coords: count=%d, "
                                    "min=(%d,%d) max=(%d,%d), "
                                    "bounds=(%d,%d,%d,%d), "
                                    "bubble_xyxy=%s, "
                                    "clip_shape=%s",
                                    ry.size,
                                    int(rx.min()) + x1, int(ry.min()) + y1,
                                    int(rx.max()) + x1, int(ry.max()) + y1,
                                    x1, y1, x2, y2,
                                    getattr(block, "bubble_xyxy", None),
                                    dense_clip.shape if dense_clip is not None else None,
                                )
                elif respect_manual_mask:
                    logger.info(
                        "Inpaint fast-fill: block[%d] skipped (manual mask, no fallback) %s reason=%s",
                        idx,
                        self._format_block_debug_label(block),
                        reason,
                    )
                    continue
                else:
                    fallback_mask, fallback_bounds = build_block_mask_data(
                        image,
                        block,
                        require_text_or_translation=False,
                        clip_to_bubble=True,
                    )
                    if fallback_mask is None or fallback_bounds is None:
                        logger.info(
                            "Inpaint fast-fill: block[%d] failed without fallback mask (%s) overlap=%d reason=%s",
                            idx,
                            self._format_block_debug_label(block),
                            initial_overlap,
                            reason,
                        )
                        continue
                    success, fallback_reason = self._fast_fill_block(
                        cleaned_image,
                        residual_mask,
                        block,
                        fallback_bounds,
                        fallback_mask,
                    )
                    if not success:
                        logger.info(
                            "Inpaint fast-fill: block[%d] failed (%s) overlap=%d primary=%s fallback=%s fallback_bounds=%s",
                            idx,
                            self._format_block_debug_label(block),
                            initial_overlap,
                            reason,
                            fallback_reason,
                            fallback_bounds,
                        )
                        continue
                    reason = f"fallback:{fallback_reason}"

                    # If fallback only covered the text region, retry the wider bubble
                    # bounds against the remaining residual. That catches user brush
                    # strokes on bubble background after the text itself has been cleared.
                    if not self._same_bounds(bounds, fallback_bounds):
                        retry_crop = residual_mask[y1:y2, x1:x2]
                        if np.any(retry_crop):
                            retry_mask = np.where(retry_crop > 0, 255, 0).astype(np.uint8)
                            retry_success, retry_reason = self._fast_fill_block(
                                cleaned_image,
                                residual_mask,
                                block,
                                bounds,
                                retry_mask,
                            )
                            if retry_success:
                                reason = f"{reason}+retry:{retry_reason}"
                            else:
                                reason = f"{reason}+retry_failed:{retry_reason}"

            cleaned_blocks += 1
            remaining_overlap = int(np.count_nonzero(residual_mask[y1:y2, x1:x2]))
            logger.info(
                "Inpaint fast-fill: block[%d] cleaned overlap=%d remaining=%d bounds=%s %s reason=%s",
                idx,
                initial_overlap,
                remaining_overlap,
                bounds,
                self._format_block_debug_label(block),
                reason,
            )

        return cleaned_image, residual_mask, cleaned_blocks

    def _fast_fill_block(
        self,
        cleaned_image: np.ndarray,
        residual_mask: np.ndarray,
        block,
        bounds: tuple[int, int, int, int],
        crop_mask: np.ndarray,
    ) -> tuple[bool, str]:
        x1, y1, x2, y2 = bounds
        crop = cleaned_image[y1:y2, x1:x2]
        residual_crop = residual_mask[y1:y2, x1:x2]
        masked_region, background_region = self._get_fast_fill_regions(block, bounds, crop_mask, residual_crop)
        if masked_region is None or background_region is None:
            return False, "invalid-regions"

        fill_color, color_reason = self._estimate_fast_fill_color(crop, masked_region, background_region)
        if fill_color is None:
            return False, color_reason

        fill_region = self._get_associated_residual_components(residual_crop, masked_region)
        
        if getattr(block, "text_class", None) == "text_bubble" and getattr(block, "bubble_xyxy", None) is not None:
            bubble_mask = build_bubble_clip_mask(
                fill_region.shape[:2],
                bounds,
                block.bubble_xyxy,
                inset=FAST_FILL_BUBBLE_INSET,
                image=cleaned_image,
                seed_bbox=block.xyxy,
            )
            if bubble_mask is not None:
                num_labels, labeled_fill = imk.connected_components(fill_region, connectivity=4)
                overlapping_labels = np.unique(labeled_fill[bubble_mask])
                keep_labels = overlapping_labels[overlapping_labels > 0]
                if keep_labels.size > 0:
                    fill_region = np.isin(labeled_fill, keep_labels)
                else:
                    fill_region = np.zeros_like(fill_region)
        else:
            bubble_mask = None

        soft_mask = imk.gaussian_blur(fill_region.astype(np.uint8) * 255, 1.0).astype(np.float32) / 255.0
        soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., np.newaxis]
        
        if bubble_mask is not None:
            soft_mask = soft_mask * bubble_mask[..., np.newaxis]
        crop_f = crop.astype(np.float32)
        fill_rgb = np.broadcast_to(fill_color, crop.shape).astype(np.float32)
        blended = crop_f * (1.0 - soft_mask) + fill_rgb * soft_mask
        cleaned_image[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        residual_crop[fill_region] = 0
        return True, color_reason

    @staticmethod
    def _estimate_dense_fill_color(
        crop: np.ndarray,
        crop_mask: np.ndarray,
        bounds: tuple[int, int, int, int],
        seed_bbox,
        bubble_xyxy=None,
        min_px: int = 200,
    ):
        """
        Estimate fill color from seed-bg: pixels inside the text bbox
        that are NOT text (background between letters). Clips seed to
        bubble_xyxy so panel background outside the bubble is excluded.
        Returns (fill_color, max_iqr, n_px) or None if not enough pixels.
        """
        if seed_bbox is None or len(seed_bbox) < 4:
            return None
        bx1, by1, bx2, by2 = [int(v) for v in seed_bbox[:4]]
        ox, oy = int(bounds[0]), int(bounds[1])
        ch, cw = crop_mask.shape[:2]

        # Clip seed to bubble bbox so panel background is excluded
        if bubble_xyxy is not None and len(bubble_xyxy) >= 4:
            bbx1, bby1, bbx2, bby2 = [int(v) for v in bubble_xyxy[:4]]
            bx1 = max(bx1, bbx1)
            by1 = max(by1, bby1)
            bx2 = min(bx2, bbx2)
            by2 = min(by2, bby2)

        # Convert to crop coordinates
        sx1 = max(0, min(cw, bx1 - ox))
        sy1 = max(0, min(ch, by1 - oy))
        sx2 = max(sx1, min(cw, bx2 - ox))
        sy2 = max(sy1, min(ch, by2 - oy))
        if sx2 <= sx1 or sy2 <= sy1:
            return None

        seed = crop[sy1:sy2, sx1:sx2]
        seed_text = crop_mask[sy1:sy2, sx1:sx2]
        if seed.shape[0] == 0 or seed_text.shape[0] == 0:
            return None
        if seed.shape[0] != seed_text.shape[0]:
            return None
        seed_bg = seed[seed_text == 0]

        if seed_bg.shape[0] < min_px:
            return None

        # Filter out dark pixels (brightness < 50) which are likely
        # figures/shadows inside the bubble, not the bubble background.
        seed_brightness = seed_bg.mean(axis=1)
        bright_mask = seed_brightness >= 50.0
        seed_bg = seed_bg[bright_mask]

        if seed_bg.shape[0] < min_px:
            return None

        fill_color = np.median(seed_bg, axis=0).astype(np.uint8)
        seed_f = seed_bg.astype(np.float32)
        p25 = np.percentile(seed_f, 25, axis=0)
        p75 = np.percentile(seed_f, 75, axis=0)
        max_iqr = float(np.max(p75 - p25))
        return fill_color, max_iqr, seed_bg.shape[0]

    def _dense_bubble_fill(
        self,
        original_image: np.ndarray,
        cleaned_image: np.ndarray,
        residual_mask: np.ndarray,
        block,
        bounds: tuple[int, int, int, int],
        crop_mask: np.ndarray,
        *,
        density_threshold: float = 0.70,
        spread_threshold: float = 55.0,
    ) -> tuple[bool, str]:
        """
        Fast fill for extremely dense text bubbles where NN inpainting
        would fail due to insufficient background context.

        Computes median background color from seed-bg (pixels between
        letters inside the text bbox) which avoids panel background leaks.
        Falls back to clip-bg if seed-bg has too few pixels.
        Only applies if background is homogeneous (low IQR spread).
        """
        x1, y1, x2, y2 = bounds
        crop_h, crop_w = crop_mask.shape[:2]

        bubble_clip = build_bubble_clip_mask(
            (crop_h, crop_w),
            bounds,
            block.bubble_xyxy,
            inset=FAST_FILL_BUBBLE_INSET,
            image=original_image,
            seed_bbox=block.xyxy,
        )
        if bubble_clip is None:
            logger.debug("Dense fill: block skipped — no bubble clip")
            return False, "no-bubble-clip"

        mask_area = int(np.count_nonzero(crop_mask))
        bubble_area = int(np.count_nonzero(bubble_clip))
        if bubble_area == 0:
            logger.debug("Dense fill: block skipped — empty bubble clip")
            return False, "empty-bubble"

        density = mask_area / float(bubble_area)
        if density < density_threshold:
            logger.debug(
                "Dense fill: block skipped — density %.2f < %.2f",
                density, density_threshold,
            )
            return False, f"not-dense:{density:.2f}"

        crop = original_image[y1:y2, x1:x2]

        # --- Estimate fill color: try seed-bg first, fall back to clip-bg ---
        seed_bbox = getattr(block, "xyxy", None)
        bubble_xyxy = getattr(block, "bubble_xyxy", None)
        seed_est = self._estimate_dense_fill_color(crop, crop_mask, bounds, seed_bbox, bubble_xyxy)

        if seed_est is not None:
            fill_color, max_iqr, n_px = seed_est
            fill_source = "seed_bg"
            bg_n = n_px
        else:
            # Fallback: use all background pixels inside bubble clip
            bg_mask = bubble_clip & (crop_mask == 0)
            bg_pixels = crop[bg_mask]
            if bg_pixels.shape[0] == 0:
                logger.debug("Dense fill: block skipped — no bg pixels in bubble")
                return False, "no-bg-pixels"
            bg_f = bg_pixels.astype(np.float32)
            bg_n = bg_pixels.shape[0]
            p25 = np.percentile(bg_f, 25, axis=0)
            p75 = np.percentile(bg_f, 75, axis=0)
            max_iqr = float(np.max(p75 - p25))
            fill_color = np.median(bg_f, axis=0).astype(np.uint8)
            fill_source = "clip_bg"

        logger.debug(
            "Dense fill candidate: src=%s n=%d median=R%d G%d B%d "
            "max_iqr=%.1f bg_px=%d",
            fill_source, bg_n,
            fill_color[0], fill_color[1], fill_color[2],
            max_iqr, bg_n,
        )

        # Homogeneity check: IQR must be below threshold.
        # High IQR = textured/gradient/multi-colored background → skip.
        if max_iqr > spread_threshold:
            logger.debug(
                "Dense fill: block skipped — heterogeneous bg "
                "(max_iqr=%.1f > %.1f)",
                max_iqr, spread_threshold,
            )
            return False, f"heterogeneous:max_iqr={max_iqr:.1f}"

        logger.debug(
            "Dense fill: filling block density=%.2f, bg_px=%d, "
            "max_iqr=%.1f, src=%s, color=%s",
            density, bg_n, max_iqr, fill_source, fill_color,
        )

        fill_region = (crop_mask > 0).astype(bool)
        soft_mask = imk.gaussian_blur(
            fill_region.astype(np.uint8) * 255, 1.0
        ).astype(np.float32) / 255.0
        soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., np.newaxis]

        bubble_mask_float = bubble_clip.astype(np.float32)[..., np.newaxis]
        soft_mask = soft_mask * bubble_mask_float

        crop_f = crop.astype(np.float32)
        fill_rgb = np.broadcast_to(fill_color, crop.shape).astype(np.float32)
        blended = crop_f * (1.0 - soft_mask) + fill_rgb * soft_mask
        cleaned_image[y1:y2, x1:x2] = np.clip(
            np.round(blended), 0, 255
        ).astype(np.uint8)
        # Expand the zeroed region to catch Gaussian feather residual
        # that falls just outside the binary mask boundary.
        feather_kernel = imk.get_structuring_element(imk.MORPH_RECT, (7, 7))
        expanded_fill = imk.dilate(
            fill_region.astype(np.uint8), feather_kernel, iterations=1
        ) > 0
        residual_mask[y1:y2, x1:x2][expanded_fill] = 0
        return True, f"density={density:.2f},bg_px={bg_n},max_iqr={max_iqr:.1f},src={fill_source},color={fill_color}"

    def _get_associated_residual_components(self, residual_crop: np.ndarray, masked_region: np.ndarray) -> np.ndarray:
        residual_binary = (residual_crop > 0).astype(np.uint8)
        if not np.any(residual_binary):
            return masked_region

        num_labels, labels, _stats, _centroids = imk.connected_components_with_stats(
            residual_binary,
            connectivity=8,
        )
        if num_labels <= 1:
            return residual_binary > 0

        near_mask = imk.dilate(
            masked_region.astype(np.uint8),
            np.ones((17, 17), np.uint8),
            iterations=1,
        ) > 0
        overlap = near_mask & (labels > 0)
        overlap_labels = np.unique(labels[overlap])
        if overlap_labels.size == 0:
            return masked_region

        return np.isin(labels, overlap_labels)

    def _drop_tiny_residual_components(self, mask: np.ndarray, max_component_area: int = 256) -> tuple[np.ndarray, int]:
        if mask is None or not np.any(mask):
            return mask, 0

        binary = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _centroids = imk.connected_components_with_stats(
            binary,
            connectivity=8,
        )
        if num_labels <= 1:
            return mask, 0

        pruned = mask.copy()
        dropped_pixels = 0
        for label in range(1, num_labels):
            area = int(stats[label, imk.CC_STAT_AREA])
            if area <= max_component_area:
                component = labels == label
                dropped_pixels += int(np.count_nonzero(pruned[component]))
                pruned[component] = 0

        return pruned, dropped_pixels

    def _residual_component_summary(self, mask: np.ndarray) -> tuple[int, int, tuple[int, int, int, int] | None]:
        if mask is None or not np.any(mask):
            return 0, 0, None

        binary = (mask > 0).astype(np.uint8)
        num_labels, _labels, stats, _centroids = imk.connected_components_with_stats(
            binary,
            connectivity=8,
        )
        if num_labels <= 1:
            return 0, 0, None

        areas = stats[1:, imk.CC_STAT_AREA]
        largest_offset = int(np.argmax(areas))
        label = largest_offset + 1
        x = int(stats[label, imk.CC_STAT_LEFT])
        y = int(stats[label, imk.CC_STAT_TOP])
        w = int(stats[label, imk.CC_STAT_WIDTH])
        h = int(stats[label, imk.CC_STAT_HEIGHT])
        return num_labels - 1, int(stats[label, imk.CC_STAT_AREA]), (x, y, w, h)

    def inpaint_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config,
        blk_list: list | None = None,
        *,
        respect_manual_mask: bool = False,
    ) -> np.ndarray:
        """
        Intelligently chooses between full-image and patch-based inpainting
        based on image size, number of text blocks, and total mask area.
        
        Args:
            image: Input image (BGR)
            mask: Text removal mask
            config: Inpainting configuration
            blk_list: List of TextBlocks
            respect_manual_mask: If True, don't fallback on manual mask failures
        """
        if image is None:
            return None
        if mask is None or not np.any(mask):
            return image.copy()

        working_image, working_mask, cleaned_blocks = self._apply_fast_bubble_cleanup(
            image, mask, blk_list, respect_manual_mask=respect_manual_mask,
        )
        if cleaned_blocks:
            logger.info("Inpaint hybrid: fast-cleaned %d bubble blocks", cleaned_blocks)
            working_mask, dropped_pixels = self._drop_tiny_residual_components(working_mask)
            if dropped_pixels:
                logger.info("Inpaint hybrid: discarded %d tiny residual mask pixels after fast cleanup", dropped_pixels)
        if working_mask is None or not np.any(working_mask):
            return working_image

        contours, _ = imk.find_contours(working_mask)
        if not contours:
            return working_image

        if cleaned_blocks:
            component_count, largest_area, largest_bbox = self._residual_component_summary(working_mask)
            overlap_summary = self._summarize_mask_overlap_with_blocks(working_mask, blk_list, image)
            logger.info(
                "Inpaint hybrid: residual mask remains after fast cleanup; running NN inpainting "
                "(components=%d, largest_area=%d, largest_bbox=%s, overlaps=%s)",
                component_count,
                largest_area,
                largest_bbox,
                overlap_summary[:5],
            )

        if getattr(self, "main_page", None) is not None or getattr(self, "inpainter_cache", None) is None:
            self._ensure_inpainter()

        if getattr(self.inpainter_cache, "force_full_image_inpainting", False):
            logger.info("Inpaint hybrid: FULL-IMAGE forced by inpainter")
            return self._inpaint_full_image(working_image, working_mask, config)

        boxes = [imk.bounding_rect(c) for c in contours]
        merged_boxes = merge_overlapping_padded_boxes(boxes, working_image.shape)
        num_patches = len(merged_boxes)

        if num_patches == 0:
            return working_image

        # Calculate mathematically modeled cost comparison:
        # 1. Calculate the scaled dimension of the full image based on actual configuration
        h_img, w_img = working_image.shape[:2]
        hd_strategy = getattr(config, "hd_strategy", "Resize")
        if hd_strategy == "Resize":
            max_size = getattr(config, "hd_strategy_resize_limit", 1024)
        elif hd_strategy == "Original":
            max_size = max(h_img, w_img)
        else:
            max_size = 1024  # Fallback 

        scale = min(1.0, max_size / max(h_img, w_img))
        full_w = max(1, int(round(w_img * scale)))
        full_h = max(1, int(round(h_img * scale)))
        full_pixels = full_w * full_h

        # 2. Calculate the cumulative effective pixels processed by all patches,
        # accounting for minimum dimensions (128px) and padding modules (8px).
        total_patch_pixels = 0
        for x1, y1, x2, y2 in merged_boxes:
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            eff_w = max(128, int(np.ceil(w / 8.0) * 8))
            eff_h = max(128, int(np.ceil(h / 8.0) * 8))
            total_patch_pixels += eff_w * eff_h

        # 3. Add a cost penalty for session run overhead (approx. 50,000 pixels worth of CPU/GPU time per extra patch)
        session_overhead_penalty = (num_patches - 1) * 50000
        estimated_patch_cost = total_patch_pixels + session_overhead_penalty

        # Choose full-image if the estimated patch cost exceeds the full-image cost
        use_patches = estimated_patch_cost < full_pixels
        area_ratio = (sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in merged_boxes)) / (w_img * h_img)

        if use_patches:
            logger.info(
                "Inpaint hybrid: PATCH-BASED (patches=%d, ratio=%.2f%%, size=%dx%d)",
                num_patches, area_ratio * 100, w_img, h_img
            )
            return self._inpaint_by_patches(working_image, working_mask, config)
        else:
            logger.info(
                "Inpaint hybrid: FULL-IMAGE (patches=%d, ratio=%.2f%%, size=%dx%d)",
                num_patches, area_ratio * 100, w_img, h_img
            )
            return self._inpaint_full_image(working_image, working_mask, config)

    def inpaint_page_from_saved_strokes(
        self,
        image: np.ndarray,
        strokes: list[dict],
        blk_list: list | None = None,
        respect_manual_mask: bool = False,
    ):
        mask = self._generate_mask_from_saved_strokes(strokes, image)
        if mask is None:
            return []
        config = get_config(self.main_page.settings_page)
        inpainted = self.inpaint_image(
            image,
            mask,
            config,
            blk_list=blk_list or None,
            respect_manual_mask=respect_manual_mask,
        )
        inpainted = imk.convert_scale_abs(inpainted)
        return self._get_regular_patches(mask, inpainted)

    def inpaint_complete(self, patch_list):
        # Handle webtoon mode vs regular mode
        if self.main_page.webtoon_mode:
            # In webtoon mode, group patches by page and apply them
            patches_by_page = {}
            for patch in patch_list:
                if 'page_index' in patch and 'file_path' in patch:
                    file_path = patch['file_path']
                    
                    if file_path not in patches_by_page:
                        patches_by_page[file_path] = []
                    
                    # Remove page-specific keys for the patch command but keep scene_pos for webtoon mode
                    clean_patch = {
                        'bbox': patch['bbox'],
                        'image': patch['image']
                    }
                    # Add scene position info for webtoon mode positioning
                    if 'scene_pos' in patch:
                        clean_patch['scene_pos'] = patch['scene_pos']
                        clean_patch['page_index'] = patch['page_index']
                    patches_by_page[file_path].append(clean_patch)
            
            # Apply patches to each page
            for file_path, patches in patches_by_page.items():
                self.main_page.image_ctrl.on_inpaint_patches_processed(patches, file_path)
        else:
            # Regular mode - original behavior
            self.main_page.apply_inpaint_patches(patch_list)
        
        self.main_page.image_viewer.clear_brush_strokes() 
        self.main_page.undo_group.activeStack().endMacro()  
        # get_best_render_area(self.main_page.blk_list, original_image, inpainted)    

    def get_inpainted_patches(self, mask: np.ndarray, inpainted_image: np.ndarray):
        # slice mask into bounding boxes
        contours, _ = imk.find_contours(mask)
        patches = []
        # Handle webtoon mode vs regular mode
        if self.main_page.webtoon_mode:
            # In webtoon mode, we need to map patches back to their respective pages
            visible_image, mappings = self.main_page.image_viewer.get_visible_area_image()
            if visible_image is None or not mappings:
                return patches
                
            for i, c in enumerate(contours):
                x, y, w, h = imk.bounding_rect(c)
                patch_bottom = y + h

                # Find all pages that this patch overlaps with
                overlapping_mappings = []
                for mapping in mappings:
                    if (y < mapping['combined_y_end'] and patch_bottom > mapping['combined_y_start']):
                        overlapping_mappings.append(mapping)
                
                if not overlapping_mappings:
                    continue
                    
                # If patch spans multiple pages, clip and redistribute
                for mapping in overlapping_mappings:
                    # Calculate the intersection with this page
                    clip_top = max(y, mapping['combined_y_start'])
                    clip_bottom = min(patch_bottom, mapping['combined_y_end'])
                    
                    if clip_bottom <= clip_top:
                        continue
                        
                    # Extract the portion of the patch for this page
                    clipped_patch = inpainted_image[clip_top:clip_bottom, x:x+w]
                    
                    # Convert coordinates back to page-local coordinates
                    page_local_y = clip_top - mapping['combined_y_start'] + mapping['page_crop_top']
                    clipped_height = clip_bottom - clip_top
                    
                    # Calculate the correct scene position by converting from visible area coordinates to scene coordinates
                    scene_y = mapping['scene_y_start'] + (clip_top - mapping['combined_y_start'])
                    
                    patches.append({
                        'bbox': [x, int(page_local_y), w, clipped_height],
                        'image': clipped_patch.copy(),
                        'page_index': mapping['page_index'],
                        'file_path': self.main_page.image_files[mapping['page_index']],
                        'scene_pos': [x, scene_y]  # Store correct scene position for webtoon mode
                    })
        else:
            # Regular mode - original behavior
            boxes = [imk.bounding_rect(c) for c in contours]
            merged_boxes = merge_overlapping_padded_boxes(boxes, inpainted_image.shape, pad=8)
            if len(merged_boxes) < len(boxes):
                logger.info("Inpaint hybrid: merged %d mask contours into %d output patches", len(boxes), len(merged_boxes))
            for x1, y1, x2, y2 in merged_boxes:
                w = x2 - x1
                h = y2 - y1
                patch = inpainted_image[y1:y2, x1:x2]
                patches.append({
                    'bbox': [x1, y1, w, h],
                    'image': patch.copy(),
                })
                
        return patches
    
    def inpaint(self):
        mask = self.main_page.image_viewer.get_mask_for_inpainting()
        painted = self.manual_inpaint()              
        patches = self.get_inpainted_patches(mask, painted)
        return patches         
