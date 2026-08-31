"""Tests for final_output capture at confirmed export time.

Covers: normal export confirm/cancel, repeat export, model_output
preservation, skipping blocks without a TM record, webtoon modes, and that
Ctrl+S never touches translation memory.
"""

import numpy as np
from types import SimpleNamespace
from unittest import mock

from PySide6.QtWidgets import QApplication

from app.controllers.projects import ProjectController
from app.translation_memory import TranslationMemoryStore
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem
from modules.utils.textblock import TextBlock, ensure_block_id


def _make_blocks():
    blk1 = TextBlock(text="src1", translation="model1")
    ensure_block_id(blk1)
    blk1.xyxy = np.array([0, 0, 100, 40], dtype=float)  # center (50, 20)
    blk2 = TextBlock(text="src2", translation="model2")
    ensure_block_id(blk2)
    blk2.xyxy = np.array([200, 200, 300, 240], dtype=float)  # center (250, 220)
    return blk1, blk2


def _pages_state(final_text="FINAL1"):
    # One text item whose center (50, 20) lands inside blk1.
    return {
        "p": {
            "viewer_state": {
                "text_items_state": [
                    {"position": (10, 5), "width": 80, "height": 30, "text": final_text},
                ]
            }
        }
    }


def _pages_state_both(final1="FINAL1", final2="FINAL2"):
    # Two text items: one inside blk1 (center 50,20), one inside blk2 (center 250,220).
    return {
        "p": {
            "viewer_state": {
                "text_items_state": [
                    {"position": (10, 5), "width": 80, "height": 30, "text": final1, "plain_text": final1},
                    {"position": (210, 205), "width": 80, "height": 30, "text": final2, "plain_text": final2},
                ]
            }
        }
    }


def test_store_save_correction_preserves_model_output(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    store.record_initial("p", "b", "src", "MODEL_V1")
    store.save_correction("p", "b", "FINAL_V2")
    rec = store.get("b")
    assert rec["model_output"] == "MODEL_V1"  # original model answer untouched
    assert rec["final_output"] == "FINAL_V2"


def test_capture_final_translations_updates_only_existing(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")  # blk1 has a record

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state("FINAL1"))

    assert store.get(blk1.block_id)["final_output"] == "FINAL1"
    assert store.get(blk1.block_id)["model_output"] == "model1"  # unchanged
    # A block with no TM record (e.g. edited manually) is now captured too.
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state_both("FINAL1", "FINAL2"))
    assert store.get(blk2.block_id)["final_output"] == "FINAL2"
    assert store.get(blk2.block_id)["model_output"] == ""  # no LLM output for manual block


def test_capture_creates_record_for_manual_block(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    # blk2 has NO LLM record at all (manual block, like the reported C/D case).

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state_both("X", "Ладно, ребята, в путь!"))

    rec = store.get(blk2.block_id)
    assert rec["final_output"] == "Ладно, ребята, в путь!"
    assert rec["model_output"] == ""          # manual translation is NOT model_output
    assert rec["source"] == "src2"


def test_capture_manual_block_keeps_model_output_empty_on_repeat_export(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    # First export creates the manual record.
    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state_both("X", "Мстители, сбор!"))
        obj._capture_final_translations(_pages_state_both("X", "Мстители, сбор!"))  # repeat export

    rec = store.get(blk2.block_id)
    assert rec["final_output"] == "Мстители, сбор!"
    assert rec["model_output"] == ""          # still empty, never overwritten


def test_capture_existing_llm_record_updates_final_only(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")  # LLM answer

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state("Пользовательский финал"))

    rec = store.get(blk1.block_id)
    assert rec["final_output"] == "Пользовательский финал"
    assert rec["model_output"] == "model1"     # original LLM output preserved


def test_capture_skips_blank_final(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    # text item whose translation is blank -> must not create/overwrite.
    blank_state = {
        "p": {"viewer_state": {"text_items_state": [
            {"position": (10, 5), "width": 80, "height": 30, "text": "", "plain_text": "   "},
        ]}}
    }
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(blank_state)

    assert store.get(blk1.block_id)["final_output"] is None  # untouched
    assert store.get(blk2.block_id) is None                   # no record created


def test_capture_same_source_different_block_ids_stays_separate(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk_a = TextBlock(text="SAME SOURCE", translation="m1")
    ensure_block_id(blk_a)
    blk_a.xyxy = np.array([0, 0, 100, 40], dtype=float)
    blk_b = TextBlock(text="SAME SOURCE", translation="m2")
    ensure_block_id(blk_b)
    blk_b.xyxy = np.array([200, 200, 300, 240], dtype=float)

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk_a, blk_b]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state("F1"))  # only blk_a matches geometry

    assert store.get(blk_a.block_id)["final_output"] == "F1"
    assert store.get(blk_b.block_id) is None  # distinct block_id -> not merged


def test_capture_final_translations_repeat_export_updates_final(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state("FINAL_A"))
        obj._capture_final_translations(_pages_state("FINAL_B"))  # re-export

    assert store.get(blk1.block_id)["final_output"] == "FINAL_B"
    assert store.get(blk1.block_id)["model_output"] == "model1"  # still untouched


def _pages_state_html_final(plain_final="FINAL1"):
    # Mirror real webtoon serialization: `text` is render HTML, `plain_text` is
    # the translation. The capture must use plain_text, never the HTML.
    html = (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
        '"http://www.w3.org/TR/REC-html40/strict.dtd">\n'
        "<html><head></head><body>" + plain_final + "</body></html>"
    )
    return {
        "p": {
            "viewer_state": {
                "text_items_state": [
                    {"position": (10, 5), "width": 80, "height": 30,
                     "text": html, "plain_text": plain_final},
                ]
            }
        }
    }


def _pages_state_legacy_html_final():
    # Project saved before plain_text existed: only render HTML is present.
    html = (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
        '"http://www.w3.org/TR/REC-html40/strict.dtd">\n'
        "<html><head></head><body>FINAL1</body></html>"
    )
    return {
        "p": {
            "viewer_state": {
                "text_items_state": [
                    {"position": (10, 5), "width": 80, "height": 30, "text": html},
                ]
            }
        }
    }


def _pages_state_undecodable_html():
    # No plain_text and the HTML cannot yield a translation (no body text).
    html = (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
        '"http://www.w3.org/TR/REC-html40/strict.dtd">\n'
        "<html><head></head><body></body></html>"
    )
    return {
        "p": {
            "viewer_state": {
                "text_items_state": [
                    {"position": (10, 5), "width": 80, "height": 30, "text": html},
                ]
            }
        }
    }


def test_capture_uses_plain_text_not_html(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state_html_final("Невероятная архитектура!"))

    assert store.get(blk1.block_id)["final_output"] == "Невероятная архитектура!"
    assert "<!DOCTYPE" not in store.get(blk1.block_id)["final_output"]
    assert "<html" not in store.get(blk1.block_id)["final_output"]


def test_capture_decodes_legacy_html(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state_legacy_html_final())

    assert store.get(blk1.block_id)["final_output"] == "FINAL1"
    assert "<!DOCTYPE" not in store.get(blk1.block_id)["final_output"]


def test_capture_skips_undecodable_html(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    main = SimpleNamespace(image_states={"p": {"blk_list": [blk1, blk2]}})
    obj = SimpleNamespace(main=main)
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item

    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._capture_final_translations(_pages_state_undecodable_html())

    # Undecodable markup is skipped, not stored.
    assert store.get(blk1.block_id)["final_output"] is None


def test_store_refuses_html_final_output(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    store.record_initial("p", "b", "src", "MODEL")
    html = (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
        '"http://www.w3.org/TR/REC-html40/strict.dtd">\n<html></html>'
    )
    store.save_correction("p", "b", html)
    # Nothing corrupting was written; existing model_output preserved.
    assert store.get("b")["final_output"] is None
    assert store.get("b")["model_output"] == "MODEL"


def test_store_accepts_ordinary_translation_with_angle_brackets(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    store.record_initial("p", "b", "src", "MODEL")
    # Ordinary text containing < and > must NOT be rejected.
    store.save_correction("p", "b", "a < b and c > d")
    assert store.get("b")["final_output"] == "a < b and c > d"


def test_manual_record_is_valid_exported_and_distinguishable(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    # LLM record (has model_output).
    store.record_initial("p", "llm", "src", "LLM OUT")
    # Manual record created the way capture does (no model_output passed).
    store.save_correction("p", "manual", "ручной финал", source="src2")

    # Both are valid corpus entries.
    assert store.count() == 2

    rec_llm = store.get("llm")
    rec_man = store.get("manual")
    assert rec_llm["model_output"] == "LLM OUT"
    assert rec_man["model_output"] == ""            # empty, never the manual final
    assert rec_man["final_output"] == "ручной финал"

    # export_jsonl includes both and marks them distinctly.
    out = tmp_path / "out.jsonl"
    written = store.export_jsonl(str(out))
    assert written == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert any('"model_output": "LLM OUT"' in line for line in lines)
    assert any('"model_output": ""' in line for line in lines)
    # Manual final never leaked into a model_output field (its value is "").
    assert not any('"model_output": "ручной финал"' in line for line in lines)


def _make_fake_export_controller(store, blocks, webtoon=False, normal_result=(True, True),
                                  webtoon_result=("pages", True), pages=None):
    calls = []
    main = SimpleNamespace()
    main.tr = lambda s: s
    main.image_states = {"p": {"blk_list": blocks}}
    main.webtoon_strip = webtoon
    main.loading = SimpleNamespace(setVisible=lambda *a: None)
    main.progress_bar = SimpleNamespace(setVisible=lambda *a: None, setValue=lambda *a: None, setFormat=lambda *a: None)
    main.default_error_handler = lambda *a: None
    main.run_threaded = lambda *a, **k: calls.append(a)
    main.image_ctrl = SimpleNamespace(save_current_image_state=lambda: None)

    obj = SimpleNamespace(main=main)
    obj._build_all_pages_current_state = lambda: pages
    obj._capture_final_translations = ProjectController._capture_final_translations.__get__(obj)
    obj._plain_text_for_item = ProjectController._plain_text_for_item
    obj._prompt_normal_export_confirm = lambda: normal_result
    obj._prompt_webtoon_export_confirm = lambda: webtoon_result
    obj.save_and_make_worker = lambda *a, **k: None
    obj._run_export_plan = ProjectController._run_export_plan.__get__(obj)
    return obj, calls


def test_run_export_plan_confirm_captures(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    obj, calls = _make_fake_export_controller(store, [blk1, blk2], normal_result=(True, True), pages=_pages_state())
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._run_export_plan([])

    assert store.get(blk1.block_id)["final_output"] == "FINAL1"
    assert len(calls) == 1  # export proceeded


def test_run_export_plan_cancel_does_nothing(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    obj, calls = _make_fake_export_controller(store, [blk1, blk2], normal_result=(False, False), pages=_pages_state())
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._run_export_plan([])

    assert store.get(blk1.block_id)["final_output"] is None  # untouched
    assert len(calls) == 0  # export aborted


def test_run_export_plan_webtoon_pages_confirm_captures(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    obj, calls = _make_fake_export_controller(
        store, [blk1, blk2], webtoon=True, webtoon_result=("pages", True), pages=_pages_state()
    )
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._run_export_plan([])

    assert store.get(blk1.block_id)["final_output"] == "FINAL1"
    assert len(calls) == 1


def test_run_export_plan_webtoon_cancel_does_nothing(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    obj, calls = _make_fake_export_controller(
        store, [blk1, blk2], webtoon=True, webtoon_result=(None, False), pages=_pages_state()
    )
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._run_export_plan([])

    assert store.get(blk1.block_id)["final_output"] is None
    assert len(calls) == 0


def test_run_export_plan_normal_export_html_text_uses_plain(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    obj, calls = _make_fake_export_controller(
        store, [blk1, blk2], normal_result=(True, True), pages=_pages_state_html_final("Обычный экспорт OK")
    )
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._run_export_plan([])

    assert store.get(blk1.block_id)["final_output"] == "Обычный экспорт OK"
    assert "<!DOCTYPE" not in store.get(blk1.block_id)["final_output"]
    assert len(calls) == 1


def test_run_export_plan_webtoon_export_html_text_uses_plain(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    blk1, blk2 = _make_blocks()
    store.record_initial("p", blk1.block_id, "src1", "model1")

    obj, calls = _make_fake_export_controller(
        store, [blk1, blk2], webtoon=True, webtoon_result=("pages", True),
        pages=_pages_state_html_final("Вебтун экспорт OK"),
    )
    with mock.patch("app.translation_memory.get_translation_memory", lambda: store):
        obj._run_export_plan([])

    assert store.get(blk1.block_id)["final_output"] == "Вебтун экспорт OK"
    assert "<!DOCTYPE" not in store.get(blk1.block_id)["final_output"]
    assert len(calls) == 1


def test_ctrl_s_save_does_not_touch_translation_memory(tmp_path):
    store = TranslationMemoryStore(str(tmp_path / "tm.db"))
    spy = mock.MagicMock(return_value=store)

    main = SimpleNamespace(project_file="project.ctpr")
    obj = SimpleNamespace(main=main)
    obj.save_current_state = lambda: None
    obj.run_save_proj = lambda *a, **k: None
    obj.thread_save_project = ProjectController.thread_save_project.__get__(obj)

    with mock.patch("app.translation_memory.get_translation_memory", spy):
        obj.thread_save_project()

    spy.assert_not_called()  # Ctrl+S must never write translation memory


def test_text_item_properties_plain_text_is_translation():
    app = QApplication.instance() or QApplication([])

    # `text` is the render HTML; `plain_text` is the displayed translation.
    item = TextBlockItem(text="Привет мир", font_size=20)
    props = TextItemProperties.from_text_item(item)

    assert props.plain_text == "Привет мир"   # translation, HTML-free
    assert "<!DOCTYPE" not in props.plain_text
    assert "<!DOCTYPE" in props.text           # render HTML preserved
    assert props.plain_text != props.text


def test_text_item_properties_round_trip_preserves_plain_text():
    app = QApplication.instance() or QApplication([])

    item = TextBlockItem(text="Здание двигается", font_size=20)
    props = TextItemProperties.from_text_item(item)
    data = props.to_dict()
    restored = TextItemProperties.from_dict(data)

    assert restored.plain_text == "Здание двигается"
    assert "<!DOCTYPE" in restored.text
