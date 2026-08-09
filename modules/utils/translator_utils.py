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


def _extract_json_object(text: str) -> str | None:
    """Extract the outermost balanced JSON object from arbitrary text.

    Uses brace counting that respects string literals and basic escapes,
    so it won't be fooled by braces inside strings. Returns None if no
    complete object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _try_parse(json_text: str) -> dict | None:
    """Parse JSON, falling back to trailing-comma / comment tolerant cleanup."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    # Tolerant pass: drop // line comments and trailing commas before } or ].
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
    candidate = _extract_json_object(cleaned)
    if candidate is None:
        print("No JSON object found in the input string.")
        print(f"Raw LLM response (first 500 chars):\n{json_string[:500]}")
        return False

    translation_dict = _try_parse(candidate)
    if translation_dict is None:
        print("Failed to parse JSON from LLM response.")
        print(f"Extracted JSON candidate (first 500 chars):\n{candidate[:500]}")
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
