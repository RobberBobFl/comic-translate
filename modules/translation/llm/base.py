from typing import Any
import numpy as np
from abc import abstractmethod
import base64
import imkit as imk

from ..base import LLMTranslation
from ...utils.textblock import TextBlock
from ...utils.translator_utils import get_raw_text, set_texts_from_json


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
        self.reasoning_effort = llm_settings.get('reasoning_effort', 'Auto')
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_tokens = 5000
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        entire_raw_text = get_raw_text(blk_list)
        base_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
        system_prompt = f"{self.custom_system_prompt}\n{base_prompt}" if self.custom_system_prompt else base_prompt
        target_hint = f"Target language: {self.target_lang}." if self.target_lang else ""
        user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\n{target_hint}\nTranslate this:\n{entire_raw_text}"
        
        entire_translated_text = self._perform_translation(user_prompt, system_prompt, image)
        success = set_texts_from_json(blk_list, entire_translated_text)
            
        return blk_list, success
    
    def rephrase(self, text: str, target_lang: str) -> str:
        """Produce a fresh, natural translation from the original source text.

        Uses the same LLM engine as translation but sends the *original*
        source text and asks for a natural, idiomatic translation — instead
        of rephrasing an existing (possibly clunky) translation.
        """
        system_prompt = (
            f"You are an expert translator. Translate to {target_lang}, "
            "making it sound natural and idiomatic, like a native speaker. "
            "Output ONLY the translation, no explanations, no prefixes."
        )
        user_prompt = (
            f"Translate this to {target_lang}, phrasing it as naturally as possible:\n"
            f"{text}"
        )
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