"""Tests for TranslationReviewController apply/staleness logic.

Headless (offscreen Qt): a fake main window with one page, one block and a
stored translation_review state. Covers the apply path (translation replaced,
entry removed from state, canvas updated) and the staleness guard (the block's
translation changed after the review ran).
"""

import os

if os.environ.get("QT_QPA_PLATFORM", "") == "":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6 import QtCore, QtWidgets

from app.controllers.translation_review import TranslationReviewController
from modules.utils.textblock import TextBlock

PAGE = "/comic/page.png"


class FakePageList(QtCore.QObject):
    currentRowChanged = QtCore.Signal(int)


class FakeTextCtrl:
    """Records apply calls instead of touching a real canvas."""

    def __init__(self, main):
        self.main = main
        self.applied = []

    def apply_text_from_command(self, text_item, text, html=None, blk=None):
        self.applied.append((blk, text))

    def _refit_text_item_to_block(self, text_item, blk):
        pass


class FakeMain(QtCore.QObject):
    def __init__(self, blk_list, image_states):
        super().__init__()
        self.image_files = [PAGE]
        self.curr_img_idx = 0
        self.blk_list = blk_list
        self.image_states = image_states
        self.webtoon_mode = False
        self.curr_tblock = None
        self.page_list = FakePageList()
        self.dirty = False
        self.applied_to_canvas = []
        self.text_ctrl = FakeTextCtrl(self)

    def mark_project_dirty(self):
        self.dirty = True


def _make_blk(text, translation, xyxy=(10, 10, 100, 40), angle=0.0):
    blk = TextBlock()
    blk.text = text
    blk.translation = translation
    blk.xyxy = list(xyxy)
    blk.angle = angle
    return blk


def _review_state(blk, **overrides):
    entry = {
        "block_index": 0,
        "source_text": blk.text,
        "translation_snapshot": blk.translation,
        "xyxy": [int(v) for v in blk.xyxy],
        "angle": float(blk.angle),
        "severity": "improvement",
        "category": "calque",
        "reason": "sounds like a calque",
        "recommended": "Естественный вариант",
        "alternatives": ["Другой вариант"],
    }
    entry.update(overrides)
    return {"target_lang": "Russian", "blocks": {"0": entry}}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def setup(app):
    def _setup(blk, review_state=None, live_translation=None):
        if live_translation is not None:
            blk.translation = live_translation
        # The persisted page state holds its own copy of the block, exactly
        # like a saved project does.
        state_blk = blk.deep_copy()
        if live_translation is not None:
            state_blk.translation = blk.translation
        state = {"blk_list": [state_blk]}
        if review_state is not None:
            state["translation_review"] = review_state
        main = FakeMain([blk], {PAGE: state})
        ctrl = TranslationReviewController(main)
        return main, ctrl
    return _setup


def test_refresh_populates_panel(setup):
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk))
    ctrl.show_results(PAGE)
    assert not ctrl.panel.is_empty()
    cards = ctrl.panel.entries()
    assert 0 in cards
    assert not cards[0].is_stale


def test_refresh_marks_drifted_translation_stale(setup):
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk), live_translation="Пользователь правил")
    ctrl.show_results(PAGE)
    card = ctrl.panel.entries()[0]
    assert card.is_stale


def test_apply_replaces_translation_and_clears_entry(setup):
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk))
    ctrl.apply_entries([(0, "Это сейчас не важно!")])

    assert blk.translation == "Это сейчас не важно!"
    state = main.image_states[PAGE]
    assert "translation_review" not in state
    assert main.dirty
    assert ctrl.panel.is_empty()


def test_apply_syncs_state_blk_list_copy(setup):
    """The persisted state holds a *different* TextBlock object; it must be updated too."""
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk))
    state_blk = main.image_states[PAGE]["blk_list"][0]
    assert state_blk is not blk
    ctrl.apply_entries([(0, "Это сейчас не важно!")])
    assert blk.translation == "Это сейчас не важно!"
    assert state_blk.translation == "Это сейчас не важно!"


def test_apply_skips_stale_and_marks_card(setup):
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk), live_translation="Пользователь правил")
    ctrl.show_results(PAGE)
    ctrl.apply_entries([(0, "Это сейчас не важно!")])

    # Nothing was applied and the entry stays in state.
    assert blk.translation == "Пользователь правил"
    assert "translation_review" in main.image_states[PAGE]
    assert not main.dirty
    card = ctrl.panel.entries().get(0)
    assert card is not None and card.is_stale


def test_apply_skips_missing_entry(setup):
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk))
    ctrl.apply_entries([(42, "Что-то")])
    assert blk.translation == "Придирки к словам!"
    assert "translation_review" in main.image_states[PAGE]


def test_review_window_is_lazy_and_reopenable(setup):
    """The review window is created on demand and reopens after being closed."""
    blk = _make_blk("Semantics!", "Придирки к словам!")
    main, ctrl = setup(blk, _review_state(blk))
    assert ctrl._dialog is None
    ctrl.refresh_from_state()
    assert ctrl._dialog is None  # no window is created just by switching pages

    ctrl.show_results(PAGE)
    assert ctrl._dialog is not None and ctrl._dialog.isVisible()
    assert 0 in ctrl.panel.entries()
    # In its own window the panel hides its duplicate title row.
    assert not ctrl.panel.header_container.isVisible()
    assert ctrl._dialog.page_label.text()

    ctrl._dialog.close()
    assert not ctrl._dialog.isVisible()
    ctrl.show_results(PAGE)
    assert ctrl._dialog.isVisible()


def test_live_block_fallback_matches_by_source_text(setup):
    """Blocks re-sorted after the review: xyxy no longer matches, but the
    reviewed index still holds a block with the same source text."""
    blk = _make_blk("Semantics!", "Придирки к словам!")
    state = _review_state(blk)
    # Move the block elsewhere; entry xyxy now points at a different block.
    other = _make_blk("Other", "Другое", xyxy=(200, 10, 260, 40))
    moved = _make_blk(blk.text, blk.translation, xyxy=(500, 10, 560, 40))
    main = FakeMain([other, moved], {PAGE: {"blk_list": [other, moved]}})
    # Patch entry to point at index 1 where the same source text now lives.
    state["blocks"]["0"]["block_index"] = 1
    main.image_states[PAGE]["translation_review"] = state
    ctrl = TranslationReviewController(main)

    blk_found = ctrl._live_block(PAGE, state["blocks"]["0"])
    assert blk_found is moved


# ---------------------------------------------------------------------------
# manual_workflow.review_translation end-to-end (fake reviewer, no network)
# ---------------------------------------------------------------------------

import numpy as np
from modules.translation import reviewer as reviewer_module
from modules.translation.reviewer import TranslationReviewer
from app.controllers.manual_workflow import ManualWorkflowController


FAKE_REVIEW_JSON = """Here is my review:
{"blocks": [{"block": 0, "severity": "improvement", "category": "calque",
"reason": "sounds like a calque", "recommended": "Это сейчас не важно!",
"alternatives": ["Не важно"]}]}"""


class FakeSettingsPage:
    def __init__(self):
        self.llm = {
            "use_translation_review": True,
            "review_send_image": False,
            "review_prompt": "",
        }
        self.creds = {"api_url": "http://localhost:11434/v1", "model": "fake", "api_key": ""}

    def get_llm_settings(self):
        return dict(self.llm)

    def get_reviewer_credentials(self):
        return dict(self.creds)


class FakeReviewer:
    calls = []

    def __init__(self):
        self.response = FAKE_REVIEW_JSON

    @classmethod
    def from_settings(cls, settings):
        return cls()

    def review(self, blk_list, image=None, source_lang="", target_lang="",
               instructions="", scene_description="", is_webtoon=False):
        FakeReviewer.calls.append(
            {"target_lang": target_lang, "image": image, "blocks": len(blk_list)}
        )
        return self.response

    parse_review_response = staticmethod(TranslationReviewer.parse_review_response)
    apply_placeholders = staticmethod(TranslationReviewer.apply_placeholders)


class FakeCombo(QtCore.QObject):
    def __init__(self, text):
        super().__init__()
        self._text = text

    def currentText(self):
        return self._text


def _make_review_main(blk):
    main = FakeMain([blk], {PAGE: {"blk_list": [blk.deep_copy()]}})
    main.settings_page = FakeSettingsPage()
    main.lang_mapping = {}
    main.s_combo = FakeCombo("English")
    main.t_combo = FakeCombo("Russian")
    main.semi_auto_mode = False
    main.review_ctrl = TranslationReviewController(main)

    class _Loading:
        def setVisible(self, visible):
            pass

    main.loading = _Loading()
    main.hbutton_group = None

    def _noop(*_args, **_kwargs):
        pass

    main.disable_hbutton_group = _noop
    main.default_error_handler = lambda error_tuple: None
    main.on_manual_finished = lambda: None

    def _sync_run_threaded(callback, result_callback=None, error_callback=None,
                           finished_callback=None, *args, **kwargs):
        result = callback()
        if result_callback is not None:
            result_callback(result)
        if finished_callback is not None:
            finished_callback()

    main.run_threaded = _sync_run_threaded
    return main


def test_manual_workflow_review_translation(app, monkeypatch):
    FakeReviewer.calls = []
    monkeypatch.setattr(reviewer_module, "TranslationReviewer", FakeReviewer)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )

    blk = _make_blk("Semantics!", "Придирки к словам!")
    main = _make_review_main(blk)
    ctrl = ManualWorkflowController(main)

    ctrl.review_translation()

    assert len(FakeReviewer.calls) == 1
    assert FakeReviewer.calls[0]["target_lang"] == "Russian"
    assert FakeReviewer.calls[0]["image"] is None  # send_image off
    state = main.image_states[PAGE]
    assert "translation_review" in state
    entry = state["translation_review"]["blocks"]["0"]
    assert entry["recommended"] == "Это сейчас не важно!"
    assert entry["translation_snapshot"] == "Придирки к словам!"
    assert entry["xyxy"] == [10, 10, 100, 40]
    assert main.dirty
    # The panel shows the results for the displayed page.
    assert 0 in main.review_ctrl.panel.entries()


def test_manual_workflow_review_requires_translations(app, monkeypatch):
    FakeReviewer.calls = []
    monkeypatch.setattr(reviewer_module, "TranslationReviewer", FakeReviewer)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )

    blk = _make_blk("Semantics!", "")  # not translated yet
    main = _make_review_main(blk)
    ctrl = ManualWorkflowController(main)

    ctrl.review_translation()

    assert FakeReviewer.calls == []
    assert "translation_review" not in main.image_states[PAGE]


def test_manual_workflow_review_gate_by_checkbox(app, monkeypatch):
    monkeypatch.setattr(reviewer_module, "TranslationReviewer", FakeReviewer)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )

    blk = _make_blk("Semantics!", "Придирки к словам!")
    main = _make_review_main(blk)
    main.settings_page.llm["use_translation_review"] = False
    ctrl = ManualWorkflowController(main)

    ctrl.review_translation()
    assert FakeReviewer.calls == []
