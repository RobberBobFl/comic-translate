import base64
import json
import re
import numpy as np
from .textblock import TextBlock
import imkit as imk


MODEL_MAP = {
    "Custom": "",  
    "Deepseek": "deepseek-v4-flash", 
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Claude-4.6-Sonnet": "claude-sonnet-4-6",
    "Claude-4.5-Haiku": "claude-haiku-4-5-20251001",
    "Gemini-2.5-Flash-Lite": "gemini-2.5-flash-lite",
    "Gemini-3.1-Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini-2.5-Pro": "gemini-2.5-pro"
}

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_txts_dict[block_key] = blk.text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json

def get_context_entries(blk_list: list[TextBlock]) -> list[dict]:
    """Source/translation pairs of already translated blocks.

    Feeds the sliding context window that LLM translators send along with each
    batch so names, pronouns and tone stay consistent across batches and pages.
    """
    entries = []
    for blk in blk_list:
        source = str(blk.text or '').strip()
        translation = str(blk.translation or '').strip()
        if source and translation:
            entries.append({'source': source, 'translation': translation})
    return entries

def get_raw_translation(blk_list: list[TextBlock]):
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_translations_dict[block_key] = blk.translation
    
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (``` or ```json) if present."""
    stripped = text.strip()
    # Match an optional language tag after the opening fence.
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n```\s*$", stripped, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Fallback: just drop any ``` tokens.
    return re.sub(r"```", "", stripped).strip()


def _extract_json_value(text: str):
    """Parse the first JSON value (object or array) from arbitrary text.

    Uses a real JSON decoder (raw_decode) so braces/quotes inside string
    values are handled correctly. Returns the parsed value, or None if no
    complete JSON value is found (e.g. the response was truncated).
    """
    text = text.strip()
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i >= n:
        return None
    try:
        value, _ = decoder.raw_decode(text[i:])
        return value
    except json.JSONDecodeError:
        return None


def _parse_lenient(json_text: str):
    """Fallback tolerant parse: accept trailing commas / // line comments."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"//[^\n]*", "", json_text)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def set_texts_from_json(blk_list: list[TextBlock], json_string: str) -> bool:
    if not json_string:
        print("Warning: Empty translation response from LLM.")
        return False

    cleaned = _strip_code_fences(json_string)
    value = _extract_json_value(cleaned)
    if value is None:
        # Tolerant fallback for models that emit trailing commas / comments.
        value = _parse_lenient(cleaned)

    if value is None:
        print("Failed to parse JSON from LLM response.")
        print(f"Raw LLM response (first 2000 chars):\n{json_string[:2000]}")
        return False

    if isinstance(value, list):
        # Model returned a JSON array; map by index to block_N keys.
        translation_dict = {f"block_{i}": v for i, v in enumerate(value)}
    elif isinstance(value, dict):
        translation_dict = value
    else:
        print("LLM response JSON is neither an object nor an array.")
        return False

    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        if block_key in translation_dict:
            blk.translation = translation_dict[block_key]
        else:
            print(f"Warning: {block_key} not found in JSON string.")
    return True

def set_upper_case(blk_list: list[TextBlock], upper_case: bool):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper() 
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper()
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)

def is_renderable_translation(translation: str | None) -> bool:
    """True if the render stage should draw this translation.

    Punctuation-only translations (an echoed "?", "!?", "...") aren't worth
    redrawing — the original artwork already shows the same thing. Anything
    gated on rendering (like inpainting) must skip them too, otherwise the
    bubble gets cleaned with nothing drawn over it. Unlike a length check,
    this keeps legitimate single-character translations (e.g. "何", "5").
    """
    if not translation:
        return False
    return any(ch.isalnum() for ch in translation)
