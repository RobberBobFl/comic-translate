from typing import Any
import logging
import time
import numpy as np
from abc import abstractmethod
import base64
import imkit as imk

from ..base import LLMTranslation
from ...utils.exceptions import InsufficientCreditsException
from ...utils.textblock import TextBlock
from ...utils.translator_utils import get_context_entries, get_raw_text, set_texts_from_json

logger = logging.getLogger(__name__)

# Pages with few blocks are cheap enough to translate in one request; splitting
# them would only multiply the reasoning-token overhead paid per request.
SINGLE_BATCH_MAX_BLOCKS = 8

# A chunk that fails with a network/API error or comes back with malformed/empty
# JSON is retried this many times in total before the chunk is treated as failed.
CHUNK_ATTEMPTS = 3
CHUNK_RETRY_DELAY = 1.5


class _TranslationSlot:
    """Write target for set_texts_from_json used to validate a chunk response.

    The parsed translations land here first so a chunk that comes back empty or
    unparseable leaves the real TextBlocks untouched.
    """

    __slots__ = ('translation',)

    def __init__(self):
        self.translation = ''


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    
    def __init__(self):
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.model = None
        self.img_as_llm_input = False
        self.custom_system_prompt = ""
        self.temperature = None
        self.top_p = None
        self.max_tokens = None
        self.timeout = 30  
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        llm_settings = settings.get_llm_settings()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.img_as_llm_input = llm_settings.get('image_input_enabled', True)
        self.custom_system_prompt = llm_settings.get('system_prompt', '')
        self.context_window = llm_settings.get('context_window', 8)
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_tokens = 16000
        
    def translate(
        self,
        blk_list: list[TextBlock],
        image: np.ndarray,
        extra_context: str,
        context_blocks: list = None,
        batch_size: int = None,
        scene_description: str = None,
    ) -> tuple[list[TextBlock], bool]:
        """
        Translate text blocks using LLM.

        The page is sent either as a single request (manual translation) or as
        several smaller chunks (batch/semi-auto), each carrying a sliding window
        of previously translated lines so names, pronouns and style stay
        consistent across chunks and pages.

        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            context_blocks: Previously translated source/translation pairs
            batch_size: Blocks per request; falsy means one request per page

        Returns:
            Tuple of (updated TextBlock objects, success flag). The flag is True
            when at least one chunk was translated, so a page that came back
            partially translated is still rendered.
        """
        base_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
        system_prompt = f"{self.custom_system_prompt}\n{base_prompt}" if self.custom_system_prompt else base_prompt

        chunks = self._split_into_chunks(blk_list, batch_size)
        window = list(context_blocks) if context_blocks else []
        window_size = self._context_window_size()

        translated_chunks = 0
        failed_blocks = 0
        last_error = None

        for index, chunk in enumerate(chunks):
            # Only the first request of a page carries the image: later chunks
            # would pay the full image cost again for the same page.
            chunk_image = image if index == 0 else None
            try:
                translations = self._translate_chunk(
                    chunk,
                    chunk_image,
                    extra_context,
                    system_prompt,
                    window,
                    scene_description,
                )
            except InsufficientCreditsException:
                raise
            except Exception as e:
                last_error = e
                translations = None
                logger.exception("Translation chunk %d/%d failed", index + 1, len(chunks))

            if translations is None:
                failed_blocks += len(chunk)
                continue

            for blk, translation in zip(chunk, translations):
                blk.translation = translation

            translated_chunks += 1
            if window_size:
                window.extend(get_context_entries(chunk))
                del window[:-window_size]

        if translated_chunks == 0:
            # Nothing came through: surface the original API error (auth,
            # network, credits) instead of a generic failure flag.
            if last_error is not None:
                raise last_error
            return blk_list, False

        if failed_blocks:
            logger.warning(
                "Partial translation: %d of %d blocks left untranslated.",
                failed_blocks,
                len(blk_list),
            )

        return blk_list, True

    def _context_window_size(self) -> int:
        try:
            return max(0, int(getattr(self, 'context_window', 8) or 0))
        except (TypeError, ValueError):
            return 8

    @staticmethod
    def _split_into_chunks(blk_list: list[TextBlock], batch_size: int) -> list[list[TextBlock]]:
        """Split blocks into request-sized chunks, preserving reading order."""
        if not blk_list:
            return []
        try:
            size = int(batch_size or 0)
        except (TypeError, ValueError):
            size = 0
        if size <= 0 or len(blk_list) <= max(size, SINGLE_BATCH_MAX_BLOCKS):
            return [list(blk_list)]
        return [list(blk_list[i:i + size]) for i in range(0, len(blk_list), size)]

    @staticmethod
    def _format_context(context_blocks: list) -> str:
        """Render the sliding window as plain 'source -> translation' lines.

        Deliberately not JSON: the block keys of the chunk being translated
        restart at block_0, so a JSON context section would collide with them.
        """
        lines = []
        for entry in context_blocks or []:
            if isinstance(entry, dict):
                source = entry.get('source', '')
                translation = entry.get('translation', '')
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                source, translation = entry
            else:
                source = getattr(entry, 'text', '')
                translation = getattr(entry, 'translation', '')
            source = (source or '').strip().replace('\n', ' ')
            translation = (translation or '').strip().replace('\n', ' ')
            if source and translation:
                lines.append(f"{source} → {translation}")
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        chunk: list[TextBlock],
        extra_context: str,
        context_blocks: list,
        scene_description: str = None,
    ) -> str:
        target_hint = f"Target language: {self.target_lang}." if self.target_lang else ""
        parts = [extra_context, "Make the translation sound as natural as possible.", target_hint]

        if scene_description:
            parts.append(
                "SCENE CONTEXT:\n"
                f"{scene_description.strip()}\n"
                "Treat this as a potentially imperfect visual hint and use it only when relevant."
            )

        context_section = self._format_context(context_blocks)
        if context_section:
            parts.append(
                "Earlier lines of this comic, already translated. Use them for "
                "consistent names, pronouns, honorifics and tone. "
                "DO NOT translate or return them:\n"
                f"{context_section}"
            )

        parts.append(f"TEXT TO TRANSLATE:\n{get_raw_text(chunk)}")
        return "\n".join(part for part in parts if part)

    def _translate_chunk(
        self,
        chunk: list[TextBlock],
        image: np.ndarray,
        extra_context: str,
        system_prompt: str,
        context_blocks: list,
        scene_description: str = None,
    ) -> list[str] | None:
        """Translate one chunk. Returns translations, or None if the chunk failed.

        Network/API errors are retried; an unparseable or empty response is not,
        since repeating the same prompt yields the same malformed JSON.
        """
        user_prompt = self._build_user_prompt(
            chunk, extra_context, context_blocks, scene_description
        )
        expects_text = any((blk.text or '').strip() for blk in chunk)

        for attempt in range(1, CHUNK_ATTEMPTS + 1):
            try:
                response = self._perform_translation(user_prompt, system_prompt, image)
            except InsufficientCreditsException:
                raise
            except Exception:
                if attempt == CHUNK_ATTEMPTS:
                    raise
                logger.warning(
                    "Translation request failed (attempt %d/%d), retrying.",
                    attempt,
                    CHUNK_ATTEMPTS,
                )
                time.sleep(CHUNK_RETRY_DELAY * attempt)
                continue

            slots = [_TranslationSlot() for _ in chunk]
            if not set_texts_from_json(slots, response):
                if attempt < CHUNK_ATTEMPTS:
                    logger.warning(
                        "JSON parse failed (attempt %d/%d), retrying.",
                        attempt,
                        CHUNK_ATTEMPTS,
                    )
                    time.sleep(CHUNK_RETRY_DELAY * attempt)
                    continue
                return None

            translations = [
                slot.translation if isinstance(slot.translation, str) else str(slot.translation)
                for slot in slots
            ]
            if expects_text and not any(t.strip() for t in translations):
                if attempt < CHUNK_ATTEMPTS:
                    logger.warning(
                        "No usable block translations (attempt %d/%d), retrying.",
                        attempt,
                        CHUNK_ATTEMPTS,
                    )
                    time.sleep(CHUNK_RETRY_DELAY * attempt)
                    continue
                logger.warning("LLM response contained no usable block translations.")
                return None

            return translations

        return None
    
    def rephrase(self, text: str, target_lang: str, scene_description: str = None) -> str:
        """Produce a fresh, natural translation from the original source text.

        Uses the same LLM engine as translation but sends the *original*
        source text and asks for a natural, idiomatic translation — instead
        of rephrasing an existing (possibly clunky) translation.

        ``scene_description`` (optional) is the visual context for the page,
        matching what ``_build_user_prompt`` injects during batch translation,
        so the rephrase benefits from the same disambiguation cues.
        """
        system_prompt = (
            f"You are an expert translator. Translate to {target_lang}, "
            "making it sound natural and idiomatic, like a native speaker. "
            "Output ONLY the translation, no explanations, no prefixes."
        )
        parts = [
            f"Translate this to {target_lang}, phrasing it as naturally as possible:\n{text}"
        ]
        if scene_description:
            parts.append(
                "SCENE CONTEXT:\n"
                f"{scene_description.strip()}\n"
                "Treat this as a potentially imperfect visual hint and use it only when relevant."
            )
        user_prompt = "\n\n".join(parts)
        # Pass a tiny dummy image so the engine does not crash on None.
        dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        return self._perform_translation(user_prompt, system_prompt, dummy)

    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass

    def encode_image(self, image: np.ndarray, ext=".jpg"):
        """
        Encode CV2/numpy image directly to base64 string using cv2.imencode.
        
        Args:
            image: Numpy array representing the image
            ext: Extension/format to encode the image as (".png" by default for higher quality)
                
        Returns:
            Tuple of (Base64 encoded string, mime_type)
        """
        # Direct encoding from numpy/cv2 format to bytes
        buffer = imk.encode_image(image, ext.lstrip('.'))
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Map extension to mime type
        mime_types = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext.lower(), f"image/{ext[1:].lower()}")
        
        return img_str, mime_type