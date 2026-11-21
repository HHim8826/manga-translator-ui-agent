"""What the AI Agent page says about a run: every rendered page ends saved, skipped, failed or stopped
(review 07), the steps before the first chapter starts (review 16), and the stop button stops only
the live AI Agent run (review 15)."""
import asyncio
import os
from types import SimpleNamespace

import pytest

from manga_translator.agent import client as agent_client
from manga_translator.agent import events as agent_events
from manga_translator.agent.events import read_events

from test_agent_core import FakeTranslator, ScriptedClient, _finish_page, run_agent_workflow
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


class Cfg:
    class translator:
        target_lang = "CHT"


def _chapter_log(folder):
    runs = folder / "manga_translator_work" / "agent" / "runs"
    return read_events(str(runs / sorted(os.listdir(runs))[-1]))


class MixedRender(FakeTranslator):
    """The render saves 001, finds an output of 002 already there (overwriting off) and fails to
    save 003; with ``stop_at`` it is stopped after saving that many pages."""
    reports_saved = False

    def __init__(self, tmp_path, stop_at=None):
        super().__init__(tmp_path)
        self.on_result_saved = None
        self.stop_at = stop_at

    async def translate_batch(self, images_with_configs, save_info=None):
        names = await super().translate_batch(images_with_configs, save_info)
        if self.template:
            return names
        results = []
        for n, (path, _) in enumerate(images_with_configs):
            if self.stop_at is not None and n == self.stop_at:
                raise asyncio.CancelledError()
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem == "002":
                results.append(SimpleNamespace(image_name=path, success=True, skipped=True, skip_reason="existing_output",
                                               skip_message="output exists", output_path="out/002.png"))
            elif stem == "003":
                results.append(SimpleNamespace(image_name=path, success=False, skipped=False,
                                               translation_error="Failed to save output file: 003.png"))
            else:
                ctx = SimpleNamespace(image_name=path, success=True, output_path=f"out/{stem}.png")
                self.on_result_saved(ctx)
                results.append(ctx)
        return results


def _three_pages(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    return folder, [str(folder / f"00{i}.png") for i in (1, 2, 3)]


def test_every_rendered_page_ends_saved_skipped_or_failed(tmp_path):
    folder, paths = _three_pages(tmp_path)
    asyncio.run(run_agent_workflow(MixedRender(tmp_path), [(p, Cfg) for p in paths], {},
                                   client=ScriptedClient(_finish_page() * 3), log=lambda m: None,
                                   agent_settings={"inpaint_review": False, "recall_pass": False}))
    done = {e["page"]: e for e in _chapter_log(folder) if e["type"] == "render_done"}
    assert {page: e["result"] for page, e in done.items()} == {"001": "saved", "002": "skipped", "003": "failed"}
    assert done["002"]["reason"] == "existing_output" and done["002"]["output"] == "out/002.png"
    assert "Failed to save" in done["003"]["error"]


def test_a_stopped_render_ends_the_pages_it_did_not_save(tmp_path):
    folder, paths = _three_pages(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_agent_workflow(MixedRender(tmp_path, stop_at=1), [(p, Cfg) for p in paths], {},
                                       client=ScriptedClient(_finish_page() * 3), log=lambda m: None,
                                       agent_settings={"inpaint_review": False, "recall_pass": False}))
    logged = _chapter_log(folder)
    done = {e["page"]: e["result"] for e in logged if e["type"] == "render_done"}
    assert done == {"001": "saved", "002": "stopped", "003": "stopped"}
    assert logged[-1]["type"] == "stopped"


def _stages(run):
    heard = []

    def listener(event):
        if event["type"] == "workflow_stage":
            heard.append(event)

    agent_events.add_listener(listener)
    try:
        run()
    finally:
        agent_events.remove_listener(listener)
    return heard


def test_the_steps_before_the_first_chapter_are_heard_live(tmp_path, monkeypatch):
    from test_agent_render import ProbeClient

    folder = tmp_path / "chapter"
    folder.mkdir()
    image = [(str(folder / "001.png"), Cfg)]
    probe = ProbeClient(error=RuntimeError("model does not support image input"))
    monkeypatch.setattr(agent_client, "build_client", lambda config, settings: probe)

    def failing():
        with pytest.raises(Exception):
            asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), image, {}, log=lambda m: None,
                                           agent_settings={"inpaint_review": False}))

    heard = _stages(failing)
    assert [e["stage"] for e in heard] == ["start", "api_check", "failed"]
    assert heard[1]["model"] == "probe-model" and "image input" in heard[2]["error"]
    assert not (folder / "manga_translator_work" / "agent" / "runs").exists()  # live only, not in a log

    def stopped():
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), image, {}, client=ScriptedClient([]),
                                           log=lambda m: None, should_stop=lambda: True,
                                           agent_settings={"inpaint_review": False}))

    assert [e["stage"] for e in _stages(stopped)] == ["start", "preprocess", "stopped"]


# --- the page ------------------------------------------------------------------------
def _page():
    from ui.main_page.pages.agents_page import AgentsPage

    return AgentsPage(_t)


def _start(run="r1", batch="b1", pages=("001.jpg", "002.jpg", "003.jpg")):
    return {"type": "run_start", "run": run, "batch": batch, "chapter": "第35話", "pages": list(pages), "model": "m"}


def test_the_page_shows_how_each_rendered_page_ended(app):  # noqa: F811
    page = _page()
    page.replay([
        _start(),
        *[{"type": "page_done", "run": "r1", "page": p, "status": "done", "flags": 0} for p in ("001", "002", "003")],
        {"type": "run_end", "run": "r1", "pages": 3, "total": 3, "done": 3, "totals": {}},
        {"type": "render_start", "run": "r1", "agent": "render", "pages": 3},
        {"type": "render_done", "run": "r1", "page": "001", "agent": "render", "output": "out/001.png"},  # older log
        {"type": "render_done", "run": "r1", "page": "002", "agent": "render", "result": "skipped",
         "reason": "existing_output", "message": "output exists", "output": "out/002.png"},
    ])
    assert page.status_label.text() == "Agents Render Progress page=002"
    page.handle_event({"type": "render_done", "run": "r1", "page": "003", "agent": "render", "result": "failed",
                       "error": "disk full"})
    assert not page._rendering  # no page left "being rendered"
    status = page.status_label.text()
    assert status.startswith("Agents Finished done=3")
    assert status.endswith("Agents Finished Render Skipped count=1Agents Finished Render Failed count=1")
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(3) for r in rows[-3:]] == ["Agents Render Detail file=001.png",
                                              "Agents Render Skipped Existing file=002.png",
                                              "Agents Render Failed error=disk full"]

    page = _page()
    page.replay([_start(pages=("001.jpg",)), {"type": "run_end", "run": "r1", "pages": 1, "total": 1, "done": 1},
                 {"type": "render_start", "run": "r1", "agent": "render", "pages": 1},
                 {"type": "render_done", "run": "r1", "page": "001", "agent": "render", "result": "stopped"},
                 {"type": "stopped", "run": "r1", "agent": "workflow", "stage": "render"}])
    assert not page._rendering and page.status_label.text().startswith("Agents Stopped Status")


def test_the_status_says_what_the_run_does_before_its_first_chapter(app):  # noqa: F811
    page = _page()
    page.replay([_start(run="r0", batch="b0", pages=("009.jpg",)),
                 {"type": "run_end", "run": "r0", "pages": 1, "total": 1, "done": 1}])
    assert page.status_label.text().startswith("Agents Finished")
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 3})
    assert page.status_label.text() == "Agents Stage Start pages=3" and page.page_list.count() == 1  # old run gone
    page._handle_live({"type": "workflow_stage", "stage": "api_check", "model": "gpt"})
    assert page.status_label.text() == "Agents Stage API Check model=gpt"
    page._handle_live({"type": "workflow_stage", "stage": "preprocess", "pages": 3})
    assert page.status_label.text() == "Agents Stage Preprocess pages=3"
    page._handle_live(_start())
    assert page.status_label.text() == "Agents Running model=m"

    page = _page()
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 1})
    page._handle_live({"type": "workflow_stage", "stage": "failed", "error": "APICheckError: no key"})
    assert page.status_label.text() == "Agents Stage Failed error=APICheckError: no key"
    page.set_task_running(False)
    assert page.status_label.text() == "Agents Stage Failed error=APICheckError: no key"

    page = _page()  # the task ended without saying why
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "api_check", "model": "gpt"})
    page.set_task_running(False)
    assert page.status_label.text() == "Agents Stage Ended"


def test_the_stop_button_stops_only_the_live_agent_run(app, tmp_path):  # noqa: F811
    from test_agent_page_actions import _log

    page = _page()
    asked = []
    page.stop_requested.connect(lambda: asked.append(True))
    page.set_task_running(True)  # a normal translation: nothing to stop here
    assert not page.stop_button.isEnabled()
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 3})
    assert page.stop_button.isEnabled() and page.stop_button.text() == "Agents Stop"
    page._handle_live(_start())
    assert page.stop_button.toolTip() == "Agents Stop Tip chapters=第35話"

    old = _log(tmp_path / "old.jsonl", [_start(run="r0", batch="b0", pages=("009.jpg",))])
    page.open_run_log(old)
    assert not page.stop_button.isEnabled() and not page.live_button.isHidden()
    page._on_stop_clicked()  # even if it were clicked: the old log has nothing to stop
    assert asked == []
    page.live_button.click()
    assert page.stop_button.isEnabled()
    page.stop_button.click()
    assert asked == [True] and page.stop_button.text() == "Stopping..." and not page.stop_button.isEnabled()
    page.set_task_running(False)
    assert not page.stop_button.isEnabled() and page.stop_button.text() == "Agents Stop"

    # an old log opened while a run checks the API: back to the current run shows that, not the log
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 3})
    page._handle_live({"type": "workflow_stage", "stage": "api_check", "model": "gpt"})
    page.open_run_log(old)
    assert page.status_label.text() != "Agents Stage API Check model=gpt" and not page.live_button.isHidden()
    page.live_button.click()
    assert page.status_label.text() == "Agents Stage API Check model=gpt" and page.page_list.count() == 1
