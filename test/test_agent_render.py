"""What the AI Agent workflow typesets: only pages the agent translated, untranslated boxes left as
they are, and an API check before the preprocess."""
import asyncio
import json
import os

import numpy as np
import pytest

from manga_translator.agent import client as agent_client
from manga_translator.agent.client import APICheckError, check_api
from manga_translator.agent.events import read_events
from manga_translator.agent.page import PageDoc
from manga_translator.agent.runner import ChatReply, ToolCall

from test_agent_core import FakeTranslator, ScriptedClient, _decode, _finish_page, make_page, run_agent_workflow
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


class Cfg:
    class translator:
        target_lang = "CHT"


class RecordingTranslator(FakeTranslator):
    """Also records the pages and the keep_empty_translations flag of each call."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.batches = []

    async def translate_batch(self, images_with_configs, save_info=None):
        self.batches.append(([os.path.basename(p) for p, _ in images_with_configs],
                             getattr(self, "keep_empty_translations", False)))
        return await super().translate_batch(images_with_configs, save_info)


def _chapter_log(folder):
    runs = folder / "manga_translator_work" / "agent" / "runs"
    return read_events(str(runs / sorted(os.listdir(runs))[-1]))


def test_only_pages_the_agent_translated_are_rendered(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / f"00{i}.png") for i in range(1, 6)]
    translator = RecordingTranslator(tmp_path)
    # page 1 done; pages 2-4 fail (three in a row stop the run); page 5 is never reached
    client = ScriptedClient(_finish_page() + [RuntimeError("connection lost")] * 3)
    logs = []
    result = asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {}, client=client,
                                            log=logs.append, agent_settings={"inpaint_review": False,
                                                                              "recall_pass": False}))
    assert translator.batches[0] == (["001.png", "002.png", "003.png", "004.png", "005.png"], False)  # preprocess
    assert translator.batches[1:] == [(["001.png"], True)]  # the render: page 1 only, empty boxes stay empty
    assert not getattr(translator, "keep_empty_translations")  # flags restored
    assert result[0] == "ctx:001.png"
    skipped = {os.path.basename(c.image_name): c.skip_reason for c in result[1:]}
    assert skipped == {"002.png": "agent_error", "003.png": "agent_error", "004.png": "agent_error",
                       "005.png": "agent_not_reached"}
    assert all(c.skipped and "not rendered" in c.skip_message for c in result[1:])
    assert any("Not rendered (no agent translation): 002.png (error)" in line for line in logs)
    events = _chapter_log(folder)
    start = next(e for e in events if e["type"] == "render_start")
    assert start["pages"] == 1
    assert start["skipped"] == {"002": "error", "003": "error", "004": "error", "005": "not_reached"}
    end = next(e for e in events if e["type"] == "run_end")
    assert end["pages"] == 4 and end["total"] == 5 and end["done"] == 1


def test_nothing_is_rendered_when_every_page_failed(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    translator = RecordingTranslator(tmp_path)
    client = ScriptedClient([RuntimeError("bad request")])
    result = asyncio.run(run_agent_workflow(translator, [(str(folder / "001.png"), Cfg)], {}, client=client,
                                            log=lambda m: None, agent_settings={"inpaint_review": False}))
    assert len(translator.batches) == 1  # the preprocess only
    assert [c.skip_reason for c in result] == ["agent_error"]
    start = next(e for e in _chapter_log(folder) if e["type"] == "render_start")
    assert start["pages"] == 0 and start["skipped"] == {"001": "error"}


def test_an_incomplete_page_keeps_the_original_text_of_untranslated_boxes(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    doc.set_text(1, "a", "甲", "dialogue", 0.9)
    doc.status = "incomplete"
    doc.save()
    saved = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert [r["translation"] for r in saved["regions"]] == ["甲", "", ""]  # still there for the editor
    mask = _decode(saved["mask_raw"])
    assert mask[60, 40] == 255  # the translated box is erased
    assert mask[60, 120] == 0 and mask[165, 240] == 0  # the others keep their original text

    # a failed page is not rendered: its JSON keeps the whole mask for the editor
    image_path, json_path = make_page(tmp_path / "error")
    doc = PageDoc.load(image_path, json_path)
    doc.set_text(1, "a", "甲", "dialogue", 0.9)
    doc.status = "error"
    doc.save()
    mask = _decode(next(iter(json.load(open(json_path, encoding="utf-8")).values()))["mask_raw"])
    assert mask[60, 120] == 255 and mask[165, 240] == 255


class ProbeClient:
    model_label = "probe-model"

    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error
        self.requests = []

    async def complete(self, messages, tools, tool_choice, effort=""):
        self.requests.append((messages, tools, tool_choice))
        if self.error is not None:
            raise self.error
        return self.reply


def test_api_check_sends_an_image_and_a_tool():
    client = ProbeClient(ChatReply(content=None, tool_calls=[ToolCall("c1", "report_shape", '{"shape": "circle"}')]))
    asyncio.run(check_api(client))
    messages, tools, choice = client.requests[0]
    assert choice == "required" and tools[0]["function"]["name"] == "report_shape"
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image/")

    with pytest.raises(APICheckError, match="without calling the tool.*API Management"):
        asyncio.run(check_api(ProbeClient(ChatReply(content="a circle"))))
    with pytest.raises(APICheckError, match=r"model probe-model\): PermissionError: no key.*OPENAI_API_BASE"):
        asyncio.run(check_api(ProbeClient(error=PermissionError("no key"))))


def test_a_wrong_api_setting_stops_before_the_preprocess(tmp_path, monkeypatch):
    folder = tmp_path / "chapter"
    folder.mkdir()
    translator = RecordingTranslator(tmp_path)
    probe = ProbeClient(error=RuntimeError("model does not support image input"))
    monkeypatch.setattr(agent_client, "build_client", lambda config, settings: probe)
    with pytest.raises(APICheckError, match="does not support image input"):
        asyncio.run(run_agent_workflow(translator, [(str(folder / "001.png"), Cfg)], {}, log=lambda m: None,
                                       agent_settings={"inpaint_review": False}))
    assert translator.batches == [] and len(probe.requests) == 1  # no preprocess


def test_agents_page_names_the_pages_that_are_not_rendered(app):  # noqa: F811
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "run": "r1", "pages": ["001.jpg", "002.jpg", "003.jpg"], "model": "m"},
        {"type": "page_done", "run": "r1", "page": "001", "status": "done", "flags": 0},
        {"type": "page_done", "run": "r1", "page": "002", "status": "error", "flags": 1},
        {"type": "run_end", "run": "r1", "pages": 2, "total": 3, "done": 1, "totals": {"requests": 4}},
        {"type": "render_start", "run": "r1", "agent": "render", "pages": 1,
         "skipped": {"002": "error", "003": "not_reached"}},
    ])
    assert page._pages["003"]["state"] == "not_reached"
    row = page.timeline.topLevelItem(page.timeline.topLevelItemCount() - 1)
    assert row.text(2) == "Agents Render" and row.text(3) == "Agents Render Start pages=1"
    assert row.text(4) == ("Agents Render Skipped pages=002（Agent Page Error）Agent Result Separator"
                           "003（Agent Page Not Reached）")
    assert page.status_label.text() == "Agents Render Start pages=1"
    page.handle_event({"type": "render_done", "run": "r1", "page": "001", "agent": "render", "output": "x.png"})
    status = page.status_label.text()
    assert status.startswith("Agents Finished done=1 flagged=1 pages=3")
    assert status.endswith("Agents Finished Unrendered count=2")

    # nothing to render: the finished status right away
    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "run": "r2", "pages": ["001.jpg"], "model": "m"},
        {"type": "run_end", "run": "r2", "pages": 1, "total": 1, "done": 0, "totals": {}},
        {"type": "render_start", "run": "r2", "agent": "render", "pages": 0, "skipped": {"001": "error"}},
    ])
    assert page.status_label.text().startswith("Agents Finished") and not page._rendering
