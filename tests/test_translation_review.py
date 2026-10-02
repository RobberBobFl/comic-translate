"""Tests for the translation reviewer: block formatting and response parsing.

The reviewer model receives BLOCK N entries (original + current translation)
and returns a JSON list of suggestions. These tests lock down the exact
numbering convention (blk_list indices) and the tolerant normalization of
whatever the model actually returned.
"""

import json

from modules.translation.reviewer import TranslationReviewer
from modules.utils.textblock import TextBlock


def _make_blk(text, translation=""):
    blk = TextBlock()
    blk.text = text
    blk.translation = translation
    return blk


# ---------------------------------------------------------------------------
# format_review_blocks
# ---------------------------------------------------------------------------


def test_format_keeps_blk_list_indices():
    blocks = [_make_blk("A", "А"), _make_blk("B", "Б")]
    formatted = TranslationReviewer.format_review_blocks(blocks)
    assert "BLOCK 0:" in formatted
    assert "BLOCK 1:" in formatted
    assert "ORIGINAL: A" in formatted
    assert "TRANSLATION: А" in formatted
    assert formatted.index("BLOCK 0:") < formatted.index("BLOCK 1:")


def test_format_skips_empty_source_but_keeps_index():
    blocks = [_make_blk("A", "А"), _make_blk("", "should not appear"), _make_blk("C", "С")]
    formatted = TranslationReviewer.format_review_blocks(blocks)
    assert "BLOCK 0:" in formatted
    assert "BLOCK 2:" in formatted
    assert "BLOCK 1:" not in formatted
    assert "should not appear" not in formatted


def test_format_marks_missing_translation_as_empty():
    blocks = [_make_blk("A", "")]
    formatted = TranslationReviewer.format_review_blocks(blocks)
    assert "TRANSLATION: (empty)" in formatted


def test_format_collapses_whitespace():
    blocks = [_make_blk("hello\n  world", "привет\tмир")]
    formatted = TranslationReviewer.format_review_blocks(blocks)
    assert "ORIGINAL: hello world" in formatted
    assert "TRANSLATION: привет мир" in formatted


# ---------------------------------------------------------------------------
# parse_review_response — accepted shapes
# ---------------------------------------------------------------------------


def _entry(block=0, severity="error", category="calque", reason="r",
           recommended="rec", alternatives=None):
    data = {
        "block": block,
        "severity": severity,
        "category": category,
        "reason": reason,
        "recommended": recommended,
        "alternatives": alternatives or [],
    }
    return data


def test_parse_blocks_object():
    text = json.dumps({"blocks": [_entry(block=0), _entry(block=2, severity="improvement")]})
    result = TranslationReviewer.parse_review_response(text)
    assert set(result) == {0, 2}
    assert result[0]["severity"] == "error"
    assert result[2]["severity"] == "improvement"


def test_parse_bare_list():
    text = json.dumps([_entry(block=1, recommended="one"), _entry(block=3, recommended="two")])
    result = TranslationReviewer.parse_review_response(text)
    assert set(result) == {1, 3}
    assert result[1]["recommended"] == "one"


def test_parse_block_n_mapping():
    text = json.dumps({
        "block_0": _entry(block=None, recommended="rec0"),
        "block_5": _entry(block=None, recommended="rec5"),
    })
    result = TranslationReviewer.parse_review_response(text)
    assert set(result) == {0, 5}
    assert result[5]["recommended"] == "rec5"


def test_parse_strips_code_fences():
    inner = json.dumps({"blocks": [_entry(block=4, recommended="вариант")]})
    text = f"```json\n{inner}\n```"
    result = TranslationReviewer.parse_review_response(text)
    assert result[4]["recommended"] == "вариант"


def test_parse_garbage_returns_empty():
    assert TranslationReviewer.parse_review_response("not json at all") == {}
    assert TranslationReviewer.parse_review_response("") == {}
    assert TranslationReviewer.parse_review_response(None) == {}


# ---------------------------------------------------------------------------
# parse_review_response — normalization
# ---------------------------------------------------------------------------


def test_unknown_severity_becomes_improvement():
    text = json.dumps({"blocks": [_entry(severity="catastrophe"), _entry(block=1, severity="ERROR")]})
    result = TranslationReviewer.parse_review_response(text)
    assert result[0]["severity"] == "improvement"
    assert result[1]["severity"] == "error"


def test_missing_severity_defaults_to_improvement():
    entry = _entry()
    del entry["severity"]
    result = TranslationReviewer.parse_review_response(json.dumps({"blocks": [entry]}))
    assert result[0]["severity"] == "improvement"


def test_unknown_category_becomes_other_and_is_normalized():
    text = json.dumps({"blocks": [_entry(category="Style Tone"), _entry(block=1, category="weird")]})
    result = TranslationReviewer.parse_review_response(text)
    assert result[0]["category"] == "style_tone" or result[0]["category"] == "other"
    assert result[1]["category"] == "other"


def test_known_category_survives():
    result = TranslationReviewer.parse_review_response(
        json.dumps({"blocks": [_entry(category="Calque")]})
    )
    assert result[0]["category"] == "calque"


def test_alternatives_clamped_and_cleaned():
    entry = _entry(alternatives=["a", "", 5, "b", "c"])
    result = TranslationReviewer.parse_review_response(json.dumps({"blocks": [entry]}))
    assert result[0]["alternatives"] == ["a", "5"]


def test_alternatives_single_string_accepted():
    result = TranslationReviewer.parse_review_response(
        json.dumps({"blocks": [_entry(alternatives="один вариант")]})
    )
    assert result[0]["alternatives"] == ["один вариант"]


def test_entry_without_recommended_dropped():
    text = json.dumps({"blocks": [_entry(recommended="  "), _entry(block=1, recommended="ok")]})
    result = TranslationReviewer.parse_review_response(text)
    assert set(result) == {1}


def test_entry_without_block_number_dropped():
    text = json.dumps({"blocks": [_entry(block=None), _entry(block=2, recommended="ok")]})
    result = TranslationReviewer.parse_review_response(text)
    assert set(result) == {2}


def test_negative_block_dropped():
    result = TranslationReviewer.parse_review_response(
        json.dumps({"blocks": [_entry(block=-1, recommended="x")]})
    )
    assert result == {}


def test_block_number_coercion():
    text = json.dumps({"blocks": [
        _entry(block="block_3", recommended="a"),
        _entry(block="7", recommended="b"),
        _entry(block=8.0, recommended="c"),
    ]})
    result = TranslationReviewer.parse_review_response(text)
    assert set(result) == {3, 7, 8}


def test_bool_block_index_rejected():
    result = TranslationReviewer.parse_review_response(
        json.dumps({"blocks": [_entry(block=True, recommended="x")]})
    )
    assert result == {}


# ---------------------------------------------------------------------------
# apply_placeholders
# ---------------------------------------------------------------------------


def test_placeholders_replaced():
    text = TranslationReviewer.apply_placeholders(
        "from {source_lang} to {target_lang}", "Japanese", "Russian"
    )
    assert text == "from Japanese to Russian"


def test_placeholders_keep_other_braces():
    text = TranslationReviewer.apply_placeholders(
        "review {source_lang} {keep_me}", "en", "ru"
    )
    assert "{keep_me}" in text
    assert "{source_lang}" not in text


def test_placeholders_empty_langs():
    text = TranslationReviewer.apply_placeholders("from {source_lang}", "", "")
    assert "the source language" in text
