"""Context for a page that starts afresh (2026-11-28): from chapter 44 on, each page of a real work saw
only the last two pages' translations, and a line of chapter 47 that pointed back to chapter 33's
running joke (「またショッピングが始まったりしないよね」, Myaa going through ruins: 「還在逛街」) was
translated word for word (「買東西」). A page now gets every earlier page of the chapter within a size
limit, and search_past_translations also searches the work's earlier chapters."""
import asyncio
import json

from manga_translator.agent import runner as runner_module
from manga_translator.agent.events import EventLog
from manga_translator.agent.page import PageDoc
from manga_translator.agent.runner import AgentRunner
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore
from manga_translator.agent.work_memory import EarlierTranslations, earlier_chapters

from test_agent_core import ScriptedClient, _finish_page, _runner, make_page


def test_older_pages_beyond_the_limit_are_named_not_repeated(tmp_path, monkeypatch):
    pages = [make_page(tmp_path, f"{i:03d}") for i in range(1, 6)]
    runner, client, store, events = _runner(tmp_path, pages, _finish_page() * 5)
    asyncio.run(runner.run())
    events.close()
    one_page = len(runner.recap("002")) - len("Pages before this one, already translated: 001.") - 100
    monkeypatch.setattr(runner_module, "RECAP_CHARS", int(one_page * 2.5))  # room for two pages and a half
    text = runner.recap("005")
    assert "Not repeated here (search_past_translations finds them): 001, 002." in text
    assert "Page 001 as translated" not in text and "Page 002 as translated" not in text
    assert text.index("Page 003 as translated") < text.index("Page 004 as translated")
    monkeypatch.setattr(runner_module, "RECAP_CHARS", 0)  # the last RECAP_PAGES always
    text = runner.recap("005")
    assert "Not repeated here (search_past_translations finds them): 001, 002." in text
    assert "Page 003 as translated" in text and "Page 004 as translated" in text


def _chapter(parent, name, prompt="work.yaml", pages=None):
    work = parent / name / "manga_translator_work"
    (work / "json").mkdir(parents=True)
    (work / "agent" / "runs").mkdir(parents=True)
    (work / "agent" / "runs" / "r1.jsonl").write_text(
        json.dumps({"type": "run_start", "work_prompt": {"name": prompt}}) + "\n", encoding="utf-8")
    for stem, regions in (pages or {}).items():
        (work / "json" / f"{stem}_translations.json").write_text(json.dumps(
            {str(parent / name / f"{stem}.jpg"): {"regions": [{"text": t, "translation": tr} for t, tr in regions]}},
            ensure_ascii=False), encoding="utf-8")
    return work


def test_the_search_finds_the_earlier_chapters_nearest_first(tmp_path):
    works = tmp_path / "works"
    _chapter(works, "第３１話", prompt="other.yaml", pages={"004": [("ショッピング中", "逛街中")]})  # another work's
    _chapter(works, "第３３話", pages={"004": [("…そして、みゃーさんはまだショッピングしてた", "︙而Myaa小姐︐[BR]還在逛街︒")]})
    _chapter(works, "第４０話", pages={"002": [("ショッピングの続き", "繼續逛")]})
    _chapter(works, "第５０話", pages={"001": [("ショッピング", "未來的譯文")]})  # after this one
    this = _chapter(works, "第４７話")
    found = earlier_chapters("", str(this), "work.yaml")
    assert [name for name, _ in found] == ["第４０話", "第３３話"]
    entries = EarlierTranslations("", str(this), "work.yaml").entries()
    assert entries[1] == {"chapter": "第３３話", "page": "004", "source": "…そして、みゃーさんはまだショッピングしてた",
                          "translation": "︙而Myaa小姐︐還在逛街︒"}

    image, page_json = make_page(tmp_path / "elsewhere", "001")  # a page for the run; its chapter is ``this``
    store = AgentStore(str(this))
    events = EventLog(store.run_log_path("r2"), "r2")
    runner = AgentRunner(AgentSettings(), ScriptedClient([]), store, events, [str(image)], "CHT",
                         prompt_name="work.yaml")
    runner.session.doc = PageDoc.load(image, page_json)
    result = runner.registry.execute("search_past_translations", json.dumps({"query": "ショッピング"}))
    events.close()
    hits = json.loads(result.text)
    assert [(h["chapter"], h["translation"]) for h in hits] == [("第４０話", "繼續逛"), ("第３３話", "︙而Myaa小姐︐還在逛街︒")]
    assert "No earlier translation matches (this chapter and the work's earlier chapters)." == \
        runner.registry.execute("search_past_translations", json.dumps({"query": "存在しない"})).text


def test_the_work_memorys_chapters_count_wherever_they_are(tmp_path):
    from manga_translator.agent.store import AgentMemory
    from manga_translator.agent.work_memory import save_chapter

    elsewhere = _chapter(tmp_path / "vol1", "第３３話", prompt="old-name.yaml",
                         pages={"004": [("ショッピング", "逛街")]})
    work_dir = str(tmp_path / "dict" / "agent" / "work.memory")
    memory = AgentMemory(str(elsewhere / "agent" / "memory"))
    memory.story_update("Myaa 在遺跡裡逛街。")
    save_chapter(memory, work_dir, "第３３話", str(elsewhere.parent))
    this = _chapter(tmp_path / "vol2", "第４７話")
    assert [name for name, _ in earlier_chapters(work_dir, str(this), "work.yaml")] == ["第３３話"]


def test_the_prompt_says_to_look_up_a_line_that_points_back(tmp_path):
    pages = [make_page(tmp_path, "001")]
    for whole in (False, True):
        runner, _client, _store, events = _runner(tmp_path, pages, [], whole_chapter_conversation=whole)
        text = runner.system_prompt()
        events.close()
        assert "また, いつもの, あの時" in text and "searches the work's earlier chapters" in text
