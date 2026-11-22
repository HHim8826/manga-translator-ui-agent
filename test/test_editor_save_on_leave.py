"""The open editor page is exported and saved before it is left, not only on an image switch.

The user's report (2026-11-23): the last page of a chapter was often not saved. Saving and
exporting happened only when switching to another image, so the last page (no next image) was
lost when the window closed; a page saved with Ctrl+S was never exported (ch39 p14, ch40 p12 kept
their old output images); an export without a save left the project JSON old (ch31 p9). Also: a
prefetched neighbour page read before an AI Agent run wrote it was shown as it was, and the save
wrote over a JSON another program had changed."""
import _bootstrap  # noqa: F401, I001

import json
import os
import types
from types import SimpleNamespace

import numpy as np
import pytest

from editor.controller_document_service import EditorControllerDocumentService
from editor.controller_export_service import EditorControllerExportService
from editor.document_files import document_files, json_stamp
from editor.document_state import DocumentSnapshot
from test_editor_save_state import _Controller as _SaveController
from test_editor_save_state import _PersistenceService, _RenderParameterService, _ready_artifact

_LOG = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None,
                       debug=lambda *a, **k: None, error=lambda *a, **k: None)


class _State:
    def __init__(self, dirty, stale):
        self.dirty, self.stale = dirty, stale

    def has_unsaved_changes(self):
        return self.dirty

    def needs_export(self):
        return self.stale


class _Controller:
    def __init__(self, *, auto_save, auto_export, suppress=False, dirty=True, stale=False, save_ok=True):
        self.export_service = _State(dirty, stale)
        app = SimpleNamespace(editor_auto_save_on_switch=auto_save, editor_auto_export_on_switch=auto_export,
                              editor_suppress_unsaved_warning=suppress)
        self.config_service = SimpleNamespace(get_config=lambda: SimpleNamespace(app=app))
        self.logger = _LOG
        self.saved, self.exported, self.save_ok = 0, [], save_ok

    def commit_pending_edits(self):
        pass

    def save_editor_state(self):
        self.saved += 1
        return self.save_ok

    def export_image(self, automatic=False):
        self.exported.append(automatic)
        return object()


def _settle(controller, purpose, monkeypatch, answer="discard"):
    service = EditorControllerDocumentService(controller)
    asked = []
    monkeypatch.setattr(service, "_ask_unsaved_action", lambda p="switch": asked.append(p) or answer)
    try:
        return service.settle_current_document(purpose), asked
    finally:
        service.shutdown()


def test_a_page_saved_with_ctrl_s_is_still_exported_when_it_is_left(monkeypatch):
    controller = _Controller(auto_save=True, auto_export=True, dirty=False, stale=True)
    assert _settle(controller, "switch", monkeypatch) == (True, [])
    assert controller.exported == [True] and controller.saved == 0  # saved already: only the export
    untouched = _Controller(auto_save=True, auto_export=True, dirty=False, stale=False)
    assert _settle(untouched, "close", monkeypatch) == (True, []) and untouched.exported == []


def test_the_last_page_is_saved_when_the_window_closes(monkeypatch):
    controller = _Controller(auto_save=True, auto_export=True)
    assert _settle(controller, "close", monkeypatch) == (True, [])
    assert controller.saved == 1 and controller.exported == [True]
    failing = _Controller(auto_save=True, auto_export=False, save_ok=False)
    # the save failed: asked, so the window is not stuck open
    assert _settle(failing, "close", monkeypatch, answer="cancel") == (False, ["close"])
    assert _settle(failing, "close", monkeypatch, answer="discard") == (True, ["close"])
    assert _settle(failing, "switch", monkeypatch) == (False, [])  # as before: the switch stops
    assert _settle(failing, "leave", monkeypatch) == (True, [])
    reloaded = _Controller(auto_save=True, auto_export=False, save_ok=False)
    reloaded.save_editor_state = lambda: setattr(reloaded.export_service, "dirty", False) or False
    assert _settle(reloaded, "close", monkeypatch) == (False, [])  # chose to reload the newer files


def test_an_export_that_cannot_be_made_does_not_keep_the_window_open(monkeypatch):
    controller = _Controller(auto_save=True, auto_export=True)
    controller.export_image = lambda automatic=False: None
    assert _settle(controller, "close", monkeypatch) == (True, []) and controller.saved == 1
    assert _settle(controller, "switch", monkeypatch)[0] is False  # as before


@pytest.mark.parametrize("purpose", ["switch", "close", "remove"])
def test_without_auto_save_it_asks_before_the_page_is_left(monkeypatch, purpose):
    controller = _Controller(auto_save=False, auto_export=False)
    assert _settle(controller, purpose, monkeypatch, answer="cancel") == (False, [purpose])
    assert _settle(controller, purpose, monkeypatch, answer="save") == (True, [purpose])
    assert controller.saved == 1


def test_leaving_the_editor_never_asks_and_the_choice_not_to_be_asked_holds(monkeypatch):
    assert _settle(_Controller(auto_save=False, auto_export=False), "leave", monkeypatch) == (True, [])
    quiet = _Controller(auto_save=False, auto_export=False, suppress=True)
    assert _settle(quiet, "close", monkeypatch) == (True, [])


# --- the export service: what the open page needs, and a JSON changed by another program ---

def _page(tmp_path):
    source = tmp_path / "001.png"
    source.write_bytes(b"png")
    json_path = tmp_path / "manga_translator_work" / "json" / "001_translations.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(json.dumps({str(source): {"regions": []}}), encoding="utf-8")
    return source, json_path


def _service(monkeypatch, tmp_path):
    from editor import controller_export_service as export_module
    from services import export_service

    _PersistenceService.calls.clear()
    _PersistenceService.saved_images.clear()
    _PersistenceService.sidecar_path = None
    monkeypatch.setattr(export_service, "ExportService", _PersistenceService)
    monkeypatch.setattr(export_module, "get_render_parameter_service", lambda: _RenderParameterService())
    source, json_path = _page(tmp_path)
    artifact = _ready_artifact()
    controller = _SaveController(source, mask=artifact.mask.copy(), artifact=artifact)
    revision = [0]
    controller.history_service.current_revision = lambda: revision[0]
    controller.reloaded = []
    controller.document_service = SimpleNamespace(do_load_image=controller.reloaded.append)
    service = EditorControllerExportService(controller)
    service.begin_document(DocumentSnapshot(source_path=str(source), image=None,
                                            files_signature=document_files(str(source))))
    return service, controller, revision, json_path


def test_the_page_needs_an_export_after_an_edit_until_it_is_exported(monkeypatch, tmp_path):
    service, controller, revision, _json = _service(monkeypatch, tmp_path)
    submitted = []
    monkeypatch.setattr(service, "_submit_job", lambda job: submitted.append(job) or object())
    try:
        assert not service.needs_export()  # as loaded
        revision[0] = 4  # an edit (then Ctrl+S: the history is clean, the output image is not)
        assert service.needs_export()
        assert service.export_image(automatic=True) is not None and len(submitted) == 1
        assert not service.needs_export()
        service.export_failed(str(controller.model.source_path))
        assert service.needs_export()  # exported again when the page is left
        service.end_document()
        assert not service.needs_export()
    finally:
        for job in submitted:
            job.release_resources()
        service.shutdown()


def test_a_json_written_by_another_program_is_not_saved_over_without_asking(monkeypatch, tmp_path):
    service, controller, _revision, json_path = _service(monkeypatch, tmp_path)
    try:
        assert service.save_editor_state() is True  # nobody else wrote it
        assert len(_PersistenceService.calls) == 1
        # (the fake does not write: the stamp after the save is the file as it is)
        json_path.write_text(json.dumps({"agent": {"regions": [{"translation": "new"}]}}), encoding="utf-8")
        os.utime(json_path, ns=(1, 1))
        assert service.json_changed_elsewhere(str(controller.model.source_path))
        answers = iter(["cancel", "reload", "overwrite"])
        monkeypatch.setattr(service, "_ask_changed_on_disk_action", lambda _path: next(answers))
        assert service.save_editor_state() is False and len(_PersistenceService.calls) == 1  # cancel
        assert service.save_editor_state() is False and controller.reloaded == [str(controller.model.source_path)]
        assert service.save_editor_state() is True and len(_PersistenceService.calls) == 2  # overwrite
        assert not service.json_changed_elsewhere(str(controller.model.source_path))
    finally:
        service.shutdown()


def test_a_page_read_without_stamps_is_saved_as_before(monkeypatch, tmp_path):
    service, controller, _revision, json_path = _service(monkeypatch, tmp_path)
    try:
        service.begin_document(DocumentSnapshot(source_path=str(controller.model.source_path), image=None))
        json_path.write_text("{}", encoding="utf-8")
        assert not service.json_changed_elsewhere(str(controller.model.source_path))
        assert service.save_editor_state() is True
    finally:
        service.shutdown()


def test_a_prefetched_page_older_than_its_files_is_read_again(tmp_path):
    source, json_path = _page(tmp_path)
    controller = SimpleNamespace(logger=_LOG)
    service = EditorControllerDocumentService(controller)
    try:
        for changed in (False, True):
            snapshot = DocumentSnapshot(source_path=str(source), image=None,
                                        files_signature=document_files(str(source)))
            service._prefetched_documents[service._prefetch_key(str(source))] = snapshot
            if changed:  # an AI Agent run wrote the page after it was prefetched
                json_path.write_text(json.dumps({str(source): {"regions": [{"translation": "x"}]}}),
                                     encoding="utf-8")
            taken = service._take_prefetched_document(str(source))
            assert (taken is None) is changed
        assert json_stamp(str(source))[1] is not None
    finally:
        service.shutdown()


# --- the main window: closing, leaving the editor ---

def _window(settle_result, unfinished_after=0):
    events = []
    counts = iter([0, unfinished_after, unfinished_after])
    export_service = SimpleNamespace(unfinished_count=lambda: next(counts))
    controller = SimpleNamespace(export_service=export_service, shutdown=lambda: events.append("shutdown"),
                                 document_service=SimpleNamespace(
                                     settle_current_document=lambda p: events.append(p) or settle_result))
    window = SimpleNamespace(
        editor_controller=controller, logger=_LOG,
        main_view=SimpleNamespace(shutdown_background_threads=lambda _t: events.append("threads")),
        editor_logic=SimpleNamespace(shutdown=lambda: events.append("logic")),
        file_list_data_service=SimpleNamespace(shutdown=lambda: None),
        app_logic=SimpleNamespace(shutdown=lambda: events.append("app")),
    )
    from ui.main_window import MainWindow

    window._settle_editor = types.MethodType(MainWindow._settle_editor, window)
    return window, events


class _Event:
    def __init__(self):
        self.state = None

    def ignore(self):
        self.state = "ignored"

    def accept(self):
        self.state = "accepted"


def test_the_window_stays_open_when_the_page_is_not_saved():
    from ui.main_window import MainWindow

    window, events = _window(False)
    event = _Event()
    MainWindow.closeEvent(window, event)
    assert event.state == "ignored" and events == ["close"]  # nothing else was shut down


def test_the_export_the_close_started_is_waited_for_without_a_question():
    from ui.main_window import MainWindow

    window, events = _window(True, unfinished_after=1)
    event = _Event()
    MainWindow.closeEvent(window, event)
    assert event.state == "accepted"
    assert events[:2] == ["close", "shutdown"]


def test_leaving_the_editor_page_settles_its_page():
    from ui.main_window import MainWindow

    editor, other = object(), object()
    settled = []
    widgets = {0: editor, 1: other}
    window = SimpleNamespace(
        stacked_widget=SimpleNamespace(widget=widgets.get), editor_view=editor, _main_pages_by_widget={},
        _settle_editor=settled.append, _on_main_page_activated=lambda key: None)
    MainWindow._on_fluent_page_changed(window, 0)  # into the editor
    MainWindow._on_fluent_page_changed(window, 1)  # out of it
    MainWindow._on_fluent_page_changed(window, 1)
    assert settled == ["leave"]


def test_removing_the_open_page_from_the_editor_list_settles_it_first(tmp_path):
    from editor.editor_logic import EditorLogic

    page = str(tmp_path / "001.png")
    calls = []
    logic = SimpleNamespace(
        controller=SimpleNamespace(model=SimpleNamespace(get_source_image_path=lambda: page),
                                   document_service=SimpleNamespace(
                                       settle_current_document=lambda p: calls.append(p) or False)),
        _source_by_key={}, _path_is_within=EditorLogic._path_is_within)
    logic.removal_takes_current = types.MethodType(EditorLogic.removal_takes_current, logic)
    assert EditorLogic.settle_before_removal(logic, page) is False and calls == ["remove"]
    assert EditorLogic.settle_before_removal(logic, str(tmp_path / "002.png")) is True  # another page
    assert EditorLogic.settle_before_removal(logic, str(tmp_path)) is False  # its folder


def test_a_page_saved_while_its_repair_runs_gets_the_finished_repair_written(monkeypatch, tmp_path):
    service, controller, _revision, _json = _service(monkeypatch, tmp_path)
    finished = controller.model.artifact
    controller.model.artifact = None  # the repair of the new mask is still running
    controller.model.display_image = np.full((8, 8, 3), 5, dtype=np.uint8)  # the older repair
    controller.history_service.is_clean = lambda: controller.history_service.clean
    try:
        assert service.save_editor_state() is True
        assert np.array_equal(_PersistenceService.saved_images[-1], controller.model.display_image)
        controller.history_service.clean = False  # edited again: the next save writes it
        controller.model.artifact = finished
        service.inpaint_installed()
        assert len(_PersistenceService.saved_images) == 1
        controller.history_service.clean = True
        service.inpaint_installed()
        assert np.array_equal(_PersistenceService.saved_images[-1], finished.image)
        service.inpaint_installed()  # once
        assert len(_PersistenceService.saved_images) == 2
    finally:
        service.shutdown()
