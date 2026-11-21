"""Optional proofreading pass: one conversation over the whole chapter's translations."""
import asyncio
import json
import os

import pytest

from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.proofread import Proofreader, page_key
from manga_translator.agent.runner import AgentRunner
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore
from manga_translator.agent.work_prompt import parse

from test_agent_core import FakeTranslator, ScriptedClient, _calls, _finish_page, make_page


def _translated_chapter(tmp_path):
    pages = [make_page(tmp_path, "001"), make_page(tmp_path, "002")]
    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    runner = AgentRunner(AgentSettings(), ScriptedClient(_finish_page() + _finish_page()), store, events,
                         [p for p, _ in pages], "CHT")
    asyncio.run(runner.run())
    return pages, store, events, runner


def test_proofreader_revises_flags_and_records_the_old_text(tmp_path):
    pages, store, events, runner = _translated_chapter(tmp_path)
    work = parse({"glossary": {"Person": [{"original": "text2", "aliases": [
        {"original": "text2", "translations": [{"text": "二號", "condition": ""}]}]}]}}, "work.yaml")
    client = ScriptedClient([], proof=[
        _calls(("revise_text", {"page": "002", "id": 2, "translation": "改二", "reason": "語氣太硬"}),
               ("view_page", {"page": "001"}),
               ("flag_for_review", {"page": "002", "id": 3, "reason": "雙關沒翻出來"}),
               ("revise_text", {"page": "009", "id": 1, "translation": "x", "reason": "r"})),
        _calls(("finish_proofread", {"note": "整體通順"}))])
    proofreader = Proofreader(AgentSettings(), client, store, events, "CHT", runner.memory, work)
    result = asyncio.run(proofreader.run([p for p, _ in pages]))
    events.close()
    assert result == {"status": "done", "revised": 1, "flagged": 1, "steps": 2}
    first = client.proof_requests[0][1]["content"]
    assert "Page 002 (3 region(s))" in first and "「text2」 → 「譯2」" in first and "[dialogue, ~" in first
    assert "work.yaml" in first and "text2 → 二號" in first
    tool_results = [m for m in client.proof_requests[1] if m.get("role") == "tool"]
    assert "Check: " in tool_results[0]["content"] and "二號" in tool_results[0]["content"]  # glossary warning
    assert tool_results[1]["content"][1]["type"] == "image_url"
    assert tool_results[3]["content"].startswith("Error: there is no page")
    saved = next(iter(json.load(open(pages[1][1], encoding="utf-8")).values()))["regions"]
    assert [r["translation"] for r in saved] == ["譯1", "改二", "譯3"] and saved[1]["translation_raw"] == "改二"
    side = store.load_sidecar("002")
    assert side["proofread"]["revisions"] == [{"id": 2, "source": "text2", "old": "譯2", "new": "改二",
                                               "reason": "語氣太硬"}]
    assert side["proofread"]["note"] == "整體通順"
    assert side["regions"][1]["translation"] == "改二" and side["regions"][1]["reasons"][-1] == "proofread: 語氣太硬"
    assert side["review_flags"][-1] == {"region": 3, "type": "uncertain_translation", "reason": "雙關沒翻出來",
                                       "agent": "proofread", "center": [240.0, 165.0], "bbox": [200, 150, 280, 180],
                                       "text": "譯3"}  # where the box is: found again after a reorder
    assert "proofread" not in store.load_sidecar("001")
    logged = [e for e in read_events(events.path) if e.get("agent") == "proofread"]
    kinds = [e["type"] for e in logged]
    assert kinds[0] == "proofread_start" and kinds[-1] == "proofread_done"
    flag_at = kinds.index("flag")
    assert kinds[flag_at - 1] == "tool_call" and kinds[flag_at + 1] == "tool_result"  # flag before its result
    assert logged[-1]["revised"] == 1 and logged[-1]["pages"] == {"002": 1}


def _chapter(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()

    class Cfg:
        class translator:
            target_lang = "CHT"

    return [str(folder / "001.png")], Cfg, folder


def test_workflow_proofreads_only_when_asked(tmp_path):
    from manga_translator.agent.workflow import run_agent_workflow

    paths, Cfg, _ = _chapter(tmp_path)
    for proofread, expected in ((False, 0), (True, 1)):
        client = ScriptedClient(_finish_page())
        asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {}, client=client,
                                       log=lambda m: None,
                                       agent_settings={"inpaint_review": False, "proofread": proofread}))
        assert len(client.proof_requests) == expected


def test_proofreading_failure_does_not_stop_the_chapter(tmp_path):
    from manga_translator.agent.workflow import run_agent_workflow

    paths, Cfg, folder = _chapter(tmp_path)

    class Failing(ScriptedClient):
        async def complete(self, messages, tools, tool_choice, effort="", cache_key=""):
            if tools and tools[0]["function"]["name"] == "view_page":
                raise RuntimeError("proxy down")
            return await super().complete(messages, tools, tool_choice, effort)

    result = asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                            client=Failing(_finish_page()), log=lambda m: None,
                                            agent_settings={"inpaint_review": False, "proofread": True}))
    assert result == ["ctx:001.png"]
    store = AgentStore(str(folder / "manga_translator_work"))
    log = os.path.join(store.runs_dir, sorted(os.listdir(store.runs_dir))[-1])
    assert any(e["type"] == "error" and e.get("agent") == "proofread" for e in read_events(log))


# --- review 08: a proofreading that stops halfway ------------------------------------------------
@pytest.mark.parametrize("failure", [RuntimeError("proxy down"), asyncio.CancelledError()])
def test_an_interrupted_proofreading_keeps_the_record_of_what_it_changed(tmp_path, failure):
    pages, store, events, runner = _translated_chapter(tmp_path)
    client = ScriptedClient([], proof=[
        _calls(("revise_text", {"page": "002", "id": 2, "translation": "改二", "reason": "語氣太硬"}),
               ("flag_for_review", {"page": "001", "id": 1, "reason": "雙關"})),
        failure])
    proofreader = Proofreader(AgentSettings(), client, store, events, "CHT", runner.memory, None)
    with pytest.raises(type(failure)):
        asyncio.run(proofreader.run([p for p, _ in pages]))
    events.close()
    saved = next(iter(json.load(open(pages[1][1], encoding="utf-8")).values()))["regions"]
    assert saved[1]["translation"] == "改二"  # in the JSON ...
    side = store.load_sidecar("002")  # ... and in the sidecar: the editor and a rerun see the same text
    assert side["regions"][1]["translation"] == "改二"
    assert side["proofread"]["revisions"] == [{"id": 2, "source": "text2", "old": "譯2", "new": "改二",
                                               "reason": "語氣太硬"}]
    assert side["proofread"]["finished"] is False and side["proofread"]["steps"] == 2
    assert store.load_sidecar("001")["review_flags"][-1]["reason"] == "雙關"


# --- review 12: page names with dots -------------------------------------------------------------
def test_page_names_with_dots_are_found():
    pages = {"v1.2": 1, "001": 2, "a.b.c": 3}
    assert page_key("v1.2", pages) == "v1.2" and page_key("v1.2.png", pages) == "v1.2"
    assert page_key("001.JPG", pages) == "001" and page_key(" a.b.c ", pages) == "a.b.c"
    assert page_key("v1", pages) is None and page_key("v1.2.txt", pages) is None and page_key(None, pages) is None


def test_the_proofreader_revises_a_page_whose_name_has_a_dot(tmp_path):
    from test_agent_core import make_page as _make

    pages = [_make(tmp_path, "v1.2")]
    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    runner = AgentRunner(AgentSettings(), ScriptedClient(_finish_page()), store, events, [pages[0][0]], "CHT")
    asyncio.run(runner.run())
    client = ScriptedClient([], proof=[
        _calls(("revise_text", {"page": "v1.2", "id": 1, "translation": "改一", "reason": "r"}),
               ("revise_text", {"page": "v1.2.png", "id": 2, "translation": "改二", "reason": "r"})),
        _calls(("finish_proofread", {}))])
    result = asyncio.run(Proofreader(AgentSettings(), client, store, events, "CHT", runner.memory, None)
                         .run([pages[0][0]]))
    events.close()
    assert result["revised"] == 2
    saved = next(iter(json.load(open(pages[0][1], encoding="utf-8")).values()))["regions"]
    assert [r["translation"] for r in saved[:2]] == ["改一", "改二"]
    calls = [e for e in read_events(events.path) if e["type"] == "tool_call" and e.get("tool") == "revise_text"]
    assert {e["page"] for e in calls} == {"v1.2"}
