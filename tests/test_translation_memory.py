"""Tests for the local translation-memory collection feature.

Covers: stable block_id generation/persistence, blank-skipping capture,
re-translate keeping the user's final_output, single-record-per-block upsert,
and JSONL export.
"""

import json

from app.projects.parsers import ProjectDecoder, ProjectEncoder
from modules.utils.textblock import TextBlock, ensure_block_id


# ---- block_id ----------------------------------------------------------------

def test_block_id_is_none_until_ensured():
    blk = TextBlock()
    assert blk.block_id is None


def test_ensure_block_id_generates_stable_uuid():
    blk = TextBlock()
    bid = ensure_block_id(blk)
    assert isinstance(bid, str)
    assert len(bid) == 32  # uuid4().hex
    # idempotent
    assert ensure_block_id(blk) == bid


def test_legacy_block_without_id_gets_persistent_uuid():
    blk = TextBlock()
    # Simulate a block loaded from an old project that had no block_id.
    blk.__dict__.pop("block_id", None)
    assert getattr(blk, "block_id", "MISSING") == "MISSING"
    bid = ensure_block_id(blk)
    assert bid
    assert ensure_block_id(blk) == bid


def test_block_id_persists_through_project_encode_decode():
    blk = TextBlock(text="hello", translation="bonjour")
    bid = ensure_block_id(blk)

    encoder = ProjectEncoder()
    blob = encoder.encode(blk)

    decoder = ProjectDecoder()
    restored = decoder.decode(blob)
    assert isinstance(restored, TextBlock)
    assert restored.block_id == bid
    assert restored.text == "hello"
    assert restored.translation == "bonjour"


def test_block_id_preserved_across_deep_copy():
    blk = TextBlock(text="x")
    bid = ensure_block_id(blk)
    clone = blk.deep_copy()
    assert clone.block_id == bid


# ---- TranslationMemoryStore --------------------------------------------------

def _store(tmp_path):
    from app.translation_memory import TranslationMemoryStore

    return TranslationMemoryStore(str(tmp_path / "tm.db"))


def test_record_initial_skips_blank_source_or_model_output(tmp_path):
    store = _store(tmp_path)
    store.record_initial("p1", "b1", "", "model")  # blank source
    store.record_initial("p1", "b1", "src", "")  # blank model output
    assert store.count() == 0

    store.record_initial("p1", "b1", "src", "model")
    assert store.count() == 1
    rec = store.get("b1")
    assert rec["source"] == "src"
    assert rec["model_output"] == "model"
    assert rec["final_output"] is None


def test_capture_translated_blocks_skips_blank_and_never_sets_final(tmp_path):
    store = _store(tmp_path)
    empty = TextBlock(text="", translation="")
    ensure_block_id(empty)
    store.capture_translated_blocks("p", [empty])
    assert store.count() == 0

    good = TextBlock(text="src", translation="out")
    ensure_block_id(good)
    store.capture_translated_blocks("p", [good])
    rec = store.get(good.block_id)
    assert rec is not None
    assert rec["source"] == "src"
    assert rec["model_output"] == "out"
    assert rec["final_output"] is None  # capture never writes final_output


def test_retranslate_updates_model_output_but_keeps_final_output(tmp_path):
    store = _store(tmp_path)
    store.record_initial("p", "b", "src", "model_v1")
    store.save_correction("p", "b", "USER_FIX")

    # block re-translated later
    store.record_initial("p", "b", "src", "model_v2")

    rec = store.get("b")
    assert rec["model_output"] == "model_v2"
    assert rec["final_output"] == "USER_FIX"  # preserved, never auto-overwritten


def test_save_correction_upserts_single_record_per_block(tmp_path):
    store = _store(tmp_path)
    store.record_initial("p", "b", "src", "model")
    store.save_correction("p", "b", "first_fix")
    store.save_correction("p", "b", "second_fix")

    assert store.count() == 1
    assert store.get("b")["final_output"] == "second_fix"


def test_ordinary_edit_does_not_auto_write_final_output(tmp_path):
    # The store only ever receives final_output via save_correction. Recording
    # (the auto-capture path) leaves final_output NULL even if a block's
    # .translation is later changed by manual editing.
    store = _store(tmp_path)
    blk = TextBlock(text="src", translation="model")
    ensure_block_id(blk)
    store.capture_translated_blocks("p", [blk])

    # Simulate the user editing the translation in the UI.
    blk.translation = "user typed something"

    # Without an explicit save_correction call, the stored final_output is still
    # NULL -- manual edits never reach the dataset automatically.
    assert store.get(blk.block_id)["final_output"] is None


def test_export_jsonl_shape(tmp_path):
    store = _store(tmp_path)
    blk = TextBlock(text="s", translation="m")
    ensure_block_id(blk)
    store.record_initial("p", blk.block_id, "s", "m")
    store.save_correction("p", blk.block_id, "f")

    out = tmp_path / "out.jsonl"
    written = store.export_jsonl(str(out))
    assert written == 1

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    obj = json.loads(lines[0])
    assert obj == {"source": "s", "model_output": "m", "final_output": "f"}
