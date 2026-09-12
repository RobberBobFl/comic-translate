import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..utils.textblock import TextBlock
from .utils.geometry import does_rectangle_fit, do_rectangles_overlap, \
    merge_overlapping_boxes
from .font.engine import extract_foreground_color
from .heuristic_lines import annotate_blocks_with_heuristic_lines
from .backend import resolve_detection_backend
from modules.utils.device import resolve_device
from .utils.content import filter_and_fix_bboxes

logger = logging.getLogger(__name__)


class DetectionEngine(ABC):
    """
    Abstract base class for all detection engines.
    Each model implementation should inherit from this class.
    """
    
    def __init__(self, settings=None):
        self.settings = settings
        self.backend = resolve_detection_backend()
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the detection model with necessary parameters.
        
        Args:
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """
        Detect text blocks in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of TextBlock objects with detected regions
        """
        pass
        
    def create_text_blocks(
        self, 
        image: np.ndarray, 
        text_boxes: np.ndarray,
        bubble_boxes: Optional[np.ndarray] = None
    ) -> list[TextBlock]:
        
        text_boxes = filter_and_fix_bboxes(text_boxes, image.shape)
        bubble_boxes = filter_and_fix_bboxes(bubble_boxes, image.shape)

        logger.debug("[create_text_blocks] text_boxes=%d bubble_boxes=%d",
                      len(text_boxes), len(bubble_boxes))
        for i, tb in enumerate(text_boxes):
            logger.debug("  text_box[%d] = %s", i, [int(v) for v in tb])
        for i, bb in enumerate(bubble_boxes):
            logger.debug("  bubble_box[%d] = %s", i, [int(v) for v in bb])

        if bubble_boxes is None or len(bubble_boxes) == 0:
            # No bubbles — full merge (dedup + overlap pruning)
            text_boxes = merge_overlapping_boxes(text_boxes)
        else:
            # Bubbles exist: only merge text boxes where one is fully
            # contained inside another (duplicate detection from the model).
            # Do NOT prune overlapping boxes — those are separate texts
            # in separate adjacent bubbles.
            from .utils.geometry import is_mostly_contained, merge_boxes as _merge_boxes
            merged = list(text_boxes)
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(merged):
                    box = merged[i]
                    for j in range(len(merged) - 1, -1, -1):
                        if i == j:
                            continue
                        other = merged[j]
                        if (is_mostly_contained(box, other, 0.8)
                                or is_mostly_contained(other, box, 0.8)):
                            merged[i] = _merge_boxes(box, other)
                            box = merged[i]
                            merged.pop(j)
                            if j < i:
                                i -= 1
                            changed = True
                    i += 1
            text_boxes = np.array(merged) if merged else np.empty((0, 4), dtype=int)

        text_blocks = []
        text_matched = [False] * len(text_boxes)  # Track matched text boxes
        
        # Set bubble_boxes to empty array if None
        if bubble_boxes is None:
            bubble_boxes = np.array([])

        if len(text_boxes) == 0:
            return text_blocks

        # Collect text foreground color without running the expensive font model.
        h, w = image.shape[:2]
        text_colors_per_box: list[tuple] = [()] * len(text_boxes)
        for txt_idx, txt_box in enumerate(text_boxes):
            x1, y1, x2, y2 = map(int, txt_box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                text_color = extract_foreground_color(image[y1:y2, x1:x2])
                if text_color is not None:
                    text_colors_per_box[txt_idx] = tuple(text_color)

        # Build TextBlock objects using pre-computed font attrs
        for txt_idx, txt_box in enumerate(text_boxes):
            text_color = text_colors_per_box[txt_idx]

            # If no bubble boxes, all text is free text
            if len(bubble_boxes) == 0:
                text_blocks.append(
                        TextBlock(
                            text_bbox=txt_box,
                            text_class='text_free',
                            font_color=text_color,
                        )
                )
                continue

            # Find the tightest (smallest area) bubble that contains or
            # overlaps this text box.  The old greedy "break on first match"
            # would assign both texts to a large neighbouring bubble when
            # a smaller, tighter bubble exists.
            best_bubble = None
            best_area = float('inf')
            for bble_box in bubble_boxes:
                if bble_box is None:
                    continue
                if does_rectangle_fit(bble_box, txt_box) or do_rectangles_overlap(bble_box, txt_box):
                    bx1, by1, bx2, by2 = [float(v) for v in bble_box]
                    area = max(1, (bx2 - bx1) * (by2 - by1))
                    if area < best_area:
                        best_area = area
                        best_bubble = bble_box

            if best_bubble is not None:
                logger.debug("  text_box[%d] %s -> best bubble %s (area=%d)",
                              txt_idx, [int(v) for v in txt_box],
                              [int(v) for v in best_bubble], int(best_area))
                text_blocks.append(
                    TextBlock(
                        text_bbox=txt_box,
                        bubble_bbox=best_bubble,
                        text_class='text_bubble',
                        font_color=text_color,
                    )
                )
                text_matched[txt_idx] = True
            
            if not text_matched[txt_idx]:
                text_blocks.append(
                    TextBlock(
                        text_bbox=txt_box,
                        text_class='text_free',
                        font_color=text_color,
                    )
                )

        logger.debug("[create_text_blocks] result: %d text_blocks (%d bubble, %d free)",
                      len(text_blocks),
                      sum(1 for b in text_blocks if b.text_class == 'text_bubble'),
                      sum(1 for b in text_blocks if b.text_class == 'text_free'))
        for i, blk in enumerate(text_blocks):
            logger.debug("  block[%d] class=%s xyxy=%s bubble=%s",
                          i, blk.text_class,
                          [int(v) for v in blk.xyxy],
                          [int(v) for v in blk.bubble_xyxy] if blk.bubble_xyxy is not None else None)
        
        try:
            backend = resolve_detection_backend(getattr(self, "backend", None))
            device = resolve_device(self.settings.is_gpu_enabled(), backend) if self.settings else "cpu"
            _ = backend, device
            annotate_blocks_with_heuristic_lines(image, text_blocks)
        except Exception as e:
            print(f"Failed to build heuristic text lines: {e}")

        return text_blocks
