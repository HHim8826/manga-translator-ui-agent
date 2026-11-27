"""What happened in the work's earlier chapters (2026-11-28): the story summary is one text of about 500
characters that the agent rewrites as it goes; by chapter 47 of a real work it told only of chapters
41-47. Each earlier chapter now has a synopsis of its own (``synopses.py``) in the translator's system
prompt and the proofreader's first message; each page leaves a note on what happens on it for the later
pages; the proofreader searches the earlier chapters too; and the chapters translated before the agent
(27-32 of that work: translations, no run log) count as the work's chapters."""
import asyncio
import json
import os

from manga_translator.agent import runner as runner_module
from manga_translator.agent import synopses as synopses_module
from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.proofread import CONTEXT_SEARCH, ProofSession, Proofreader, _execute
from manga_translator.agent.runner import AgentRunner, ChatReply, ToolCall
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.synopses import Synopses, prompt_text, update
from manga_translator.agent.work_memory import EarlierTranslations, earlier_chapters, save_chapter

from test_agent_chapter_context import _chapter
from test_agent_core import ScriptedClient, _calls, _finish_page, _runner, make_page


def _plain_chapter(parent, name, pages):
    """A chapter translated without the agent: page JSON files, no run log."""
    work = parent / name / "manga_translator_work"
    (work / "json").mkdir(parents=True)
    for stem, regions in pages.items():
        (work / "json" / f"{stem}_translations.json").write_text(json.dumps(
            {str(parent / name / f"{stem}.jpg"): {"regions": [{"text": t, "translation": tr} for t, tr in regions]}},
            ensure_ascii=False), encoding="utf-8")
    return work


# --- the chapters translated before the agent ----------------------------------------------------
def test_chapters_translated_without_the_agent_count_next_to_the_works_chapters(tmp_path):
    works = tmp_path / "works"
    _plain_chapter(works, "第２７話", {"003": [("ショッピング！", "去逛街！")]})
    (works / "Output").mkdir(parents=True)  # no translations: not a chapter
    _chapter(works, "第３３話", pages={"004": [("まだショッピング", "還在逛街")]})
    this = _chapter(works, "第４７話")
    assert [name for name, _ in earlier_chapters("", str(this), "work.yaml")] == ["第３３話", "第２７話"]
    hits = [e["translation"] for e in EarlierTranslations("", str(this), "work.yaml").entries()]
    assert hits == ["還在逛街", "去逛街！"]

    # a folder with none of the work's chapters: the chapters there are not taken for the work's
    lonely = tmp_path / "lonely"
    _plain_chapter(lonely, "第１話", {"001": [("こんにちは", "你好")]})
    assert earlier_chapters("", str(_chapter(lonely, "第２話")), "work.yaml") == []


def test_the_chapters_next_to_the_work_memorys_chapters_count_too(tmp_path):
    vol1 = tmp_path / "vol1"
    _plain_chapter(vol1, "第２７話", {"003": [("ショッピング！", "去逛街！")]})
    elsewhere = _chapter(vol1, "第３３話", prompt="old-name.yaml", pages={"004": [("ショッピング", "逛街")]})
    work_dir = str(tmp_path / "dict" / "agent" / "work.memory")
    memory = AgentMemory(str(elsewhere / "agent" / "memory"))
    memory.story_update("Myaa 在遺跡裡逛街。")
    save_chapter(memory, work_dir, "第３３話", str(elsewhere.parent))
    this = _chapter(tmp_path / "vol2", "第４７話")
    assert [name for name, _ in earlier_chapters(work_dir, str(this), "work.yaml")] == ["第３３話", "第２７話"]


# --- synopses ------------------------------------------------------------------------------------
class SynopsisClient:
    """Answers each synopsis request with the chapter it was for; ``fail`` names chapters whose request fails."""

    def __init__(self, fail=()):
        self.requests = []
        self.fail = set(fail)

    async def complete(self, messages, tools, tool_choice, effort="", cache_key=""):
        assert tools[0]["function"]["name"] == "save_synopsis" and cache_key == "synopsis"
        self.requests.append(messages)
        chapter = messages[1]["content"].split(":", 1)[0][len("Chapter "):]
        if chapter in self.fail:
            raise ConnectionError("cut off")
        return ChatReply(content=None, prompt_tokens=900, completion_tokens=80, cached_tokens=0,
                         tool_calls=[ToolCall(id="c1", name="save_synopsis",
                                              arguments=json.dumps({"synopsis": f"{chapter}的摘要"}, ensure_ascii=False))])


def _work(tmp_path):
    works = tmp_path / "works"
    _plain_chapter(works, "第２７話", {"003": [("ショッピング！", "去逛街！")]})
    _chapter(works, "第３３話", pages={"004": [("まだショッピング", "還在逛街")]})
    ch40 = _chapter(works, "第４０話", pages={"002": [("ドラゴン", "龍")]})
    (ch40 / "agent" / "pages").mkdir(parents=True)
    (ch40 / "agent" / "pages" / "002.agent.json").write_text(json.dumps(
        {"page_note": "龍在火山上醒來，眾人逃走。"}, ensure_ascii=False), encoding="utf-8")
    return str(tmp_path / "dict" / "agent" / "work.memory"), _chapter(works, "第４７話"), ch40


def test_each_earlier_chapter_gets_its_synopsis_once_and_again_when_it_changed(tmp_path):
    work_dir, this, ch40 = _work(tmp_path)
    client = SynopsisClient()
    lines = []
    text, info = asyncio.run(update(client, AgentSettings(), "CHT", work_dir, str(this), "work.yaml", lines.append))
    assert info["written"] == ["第２７話", "第３３話", "第４０話"] and info["failed"] == []
    assert info["usage"]["requests"] == 3 and info["usage"]["prompt_tokens"] == 2700
    assert text.startswith("What happened in the work's earlier chapters, oldest first.")
    assert text.index("- 第２７話: 第２７話的摘要") < text.index("- 第３３話: 第３３話的摘要") < text.index("- 第４０話")
    sent = {m[1]["content"].split(":", 1)[0]: m[1]["content"] for m in client.requests}
    assert "- まだショッピング → 還在逛街" in sent["Chapter 第３３話"]
    assert "Page 002 (what happens: 龍在火山上醒來，眾人逃走。):" in sent["Chapter 第４０話"]
    assert "into Chinese (Traditional)" in client.requests[0][0]["content"]
    assert any("Chapter synopses: 3 written, 2700 prompt tokens" in line for line in lines)
    assert os.path.exists(os.path.join(work_dir, "synopses.json"))

    again, info = asyncio.run(update(client, AgentSettings(), "CHT", work_dir, str(this), "work.yaml"))
    assert again == text and info == {} and len(client.requests) == 3  # nothing changed: no request

    page = ch40 / "json" / "002_translations.json"
    page.write_text(page.read_text(encoding="utf-8").replace("龍", "巨龍"), encoding="utf-8")  # the user's fix
    _text, info = asyncio.run(update(client, AgentSettings(), "CHT", work_dir, str(this), "work.yaml"))
    assert info["written"] == ["第４０話"] and len(client.requests) == 4


def test_a_failed_synopsis_leaves_that_chapter_out(tmp_path):
    work_dir, this, _ = _work(tmp_path)
    text, info = asyncio.run(update(SynopsisClient(fail={"第３３話"}), AgentSettings(), "CHT", work_dir, str(this),
                                    "work.yaml"))
    assert info["written"] == ["第２７話", "第４０話"]
    assert info["failed"] == ["第３３話 (ConnectionError: cut off)"]
    assert "第２７話的摘要" in text and "第３３話" not in text
    # the next chapter's start tries it again
    _text, info = asyncio.run(update(SynopsisClient(), AgentSettings(), "CHT", work_dir, str(this), "work.yaml"))
    assert info["written"] == ["第３３話"]


def test_the_nearest_synopses_are_shown_within_the_limit(tmp_path, monkeypatch):
    known = {}
    chapters = []
    for n in (46, 45, 44, 30):  # nearest first
        folder = str(tmp_path / f"第{n}話")
        known[os.path.normcase(os.path.abspath(folder))] = {"name": f"第{n}話", "synopsis": "x" * 40}
        chapters.append((f"第{n}話", folder))
    monkeypatch.setattr(synopses_module, "SYNOPSES_CHARS", 100)
    text = prompt_text(known, chapters)
    assert "- 第44話" not in text and text.index("- 第45話") < text.index("- 第46話")
    assert text.endswith("Older chapters, not summarized here: 第30話, 第44話.")
    assert prompt_text({}, chapters) == ""


def test_a_moved_chapter_keeps_its_synopsis(tmp_path):
    store = Synopses(str(tmp_path / "work.memory"))
    store.put("第３３話", str(tmp_path / "old" / "第３３話"), "摘要", "f1")
    assert store.find(store.load(), "第３３話", str(tmp_path / "new" / "第３３話"))["synopsis"] == "摘要"


# --- page notes ----------------------------------------------------------------------------------
def _finish_with_note(note, n_regions=3):
    saves, done = _finish_page(n_regions)
    return [saves, _calls(("mark_page_done", {"confidence": 0.9, "page_note": note}))]


def test_each_page_leaves_a_note_for_the_later_pages(tmp_path, monkeypatch):
    pages = [make_page(tmp_path, f"{i:03d}") for i in range(1, 5)]
    script = sum((_finish_with_note(f"第{i}頁：Myaa 逛街。") for i in range(1, 5)), [])
    runner, client, store, events = _runner(tmp_path, pages, script)
    asyncio.run(runner.run())
    events.close()
    assert store.load_sidecar("002")["page_note"] == "第2頁：Myaa 逛街。"
    later = json.dumps(client.requests[-2], ensure_ascii=False)  # page 004's first request
    assert "Page 003 as translated (what happens: 第3頁：Myaa 逛街。):" in later
    assert "page_note" in json.dumps(runner.registry.schemas())
    assert "a page_note: what happens on it" in runner.system_prompt()

    # a later run of the chapter reads them from the sidecars; the pages the recap does not repeat
    # keep their note
    again, *_rest, events2 = _runner(tmp_path, [pages[3]], [])
    events2.close()
    monkeypatch.setattr(runner_module, "RECAP_CHARS", 0)
    text = again.recap("004")
    assert "Not repeated here (search_past_translations finds them): 001." in text
    assert "What happens on them:\n- 001: 第1頁：Myaa 逛街。" in text


# --- the proofreader -----------------------------------------------------------------------------
def test_the_proofreader_gets_the_synopses_and_searches_the_earlier_chapters(tmp_path):
    works = tmp_path / "works"
    _plain_chapter(works, "第２７話", {"003": [("ショッピング！", "去逛街！")]})
    _chapter(works, "第３３話", pages={"004": [("まだショッピング", "還在逛街")]})
    this = _chapter(works, "第４７話")
    earlier = EarlierTranslations("", str(this), "work.yaml")
    session = ProofSession({}, [], AgentMemory(str(tmp_path / "mem")), "CHT", earlier=earlier)
    text, summary, _ = _execute(session, "search_past_translations", {"query": "ショッピング"})
    assert text == "第３３話 004: 「まだショッピング」 → 「還在逛街」\n第２７話 003: 「ショッピング！」 → 「去逛街！」"
    assert summary == "ショッピング: 2"

    store = AgentStore(str(this))
    events = EventLog(store.run_log_path("r1"), "r1")
    proof = Proofreader(AgentSettings(), ScriptedClient([]), store, events, "CHT", AgentMemory(str(tmp_path / "m2")),
                        synopses="What happened in the work's earlier chapters, oldest first.\n- 第３３話: 逛街",
                        earlier=earlier)
    events.close()
    assert [t["function"]["name"] for t in proof.tools()][-1] == "search_past_translations"
    assert CONTEXT_SEARCH in proof.system_prompt() and "また, いつもの, あの時" in proof.system_prompt()
    assert "- 第３３話: 逛街" in proof.first_message({})
    plain = Proofreader(AgentSettings(), ScriptedClient([]), store, events, "CHT", AgentMemory(str(tmp_path / "m3")))
    assert "search_past_translations" not in [t["function"]["name"] for t in plain.tools()]
    assert CONTEXT_SEARCH not in plain.system_prompt()


# --- the workflow --------------------------------------------------------------------------------
def test_the_next_chapter_starts_with_the_synopsis_of_the_first(tmp_path):
    from test_agent_work_glossary import WORK, _run

    prompt_path = tmp_path / "work.yaml"
    prompt_path.write_bytes(WORK.encode("utf-8"))
    client, lines = _run(tmp_path, str(prompt_path), _finish_page() + _finish_page())
    assert len(client.synopsis_requests) == 1  # chapter 1's, at the start of chapter 2
    assert "→ 譯1" in client.synopsis_requests[0][1]["content"]
    first_system = client.requests[0][0]["content"]
    second_system = client.requests[-1][0]["content"]
    assert "What happened in the work's earlier chapters" not in first_system
    assert "What happened in the work's earlier chapters" in second_system and "synopsis 1" in second_system
    store = AgentStore(str(tmp_path / "ch2" / "chapter" / "manga_translator_work"))
    start = [e for e in read_events(os.path.join(store.runs_dir, sorted(os.listdir(store.runs_dir))[-1]))
             if e["type"] == "run_start"][0]
    assert start["synopses"]["written"] == ["chapter"] and start["synopses"]["usage"]["requests"] == 1
    if client.proof_requests:  # proofreading on: the second chapter's proofreader has them too
        assert "synopsis 1" in client.proof_requests[-1][1]["content"]

    # a second run of chapter 2 writes nothing again
    again, _ = _run(tmp_path, str(prompt_path), _finish_page() + _finish_page())
    assert again.synopsis_requests == []
