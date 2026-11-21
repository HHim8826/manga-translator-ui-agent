"""The chapters of a work share their memory (``manga_translator.agent.work_memory``): a chapter's
first run starts with what the chapters before it learned, and adds its own when it ends."""
import json
import os

from manga_translator.agent.events import read_events
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.work_glossary import candidates
from manga_translator.agent.work_memory import bootstrap, chapter_order, save_chapter, seed_chapter, work_memory_dir

from test_agent_core import _finish_page
from test_agent_work_glossary import WORK, _learn, _run


def test_the_work_folder_is_next_to_the_prompt_file(tmp_path):
    assert work_memory_dir(str(tmp_path / "dict" / "作品.yaml")) == str(tmp_path / "dict" / "agent" / "作品.memory")
    assert work_memory_dir("") == ""


def test_chapters_sort_by_their_number():
    names = ["第１００話", "第３４話_10002_1-10", "第2話", "第３３話_10001_1-11"]
    assert sorted(names, key=chapter_order) == ["第2話", "第３３話_10001_1-11", "第３４話_10002_1-10", "第１００話"]


def _chapter_memory(folder, n):
    memory = AgentMemory(str(folder))
    memory.glossary_upsert("魔剣", f"魔劍{n}", "", "003", "Item")
    memory.character_upsert("りょうた", {"target_name": "Riyota", "speech": f"語氣{n}"}, "002")
    memory.story_update(f"第{n}話的劇情")
    return memory


def test_a_chapter_starts_with_the_work_memory_and_adds_its_own(tmp_path):
    work = str(tmp_path / "work.memory")
    first = _chapter_memory(tmp_path / "ch1", 1)
    first.glossary_upsert("ドン", "咚", "", "004")  # a sound: not for the prompt file, but shared
    assert save_chapter(first, work, "第1話") == {"glossary_added": 2, "glossary_updated": 0,
                                                 "characters_added": 1, "story": 6}

    assert seed_chapter(AgentMemory(str(tmp_path / "ch2")), work) == {"glossary": 2, "characters": 1, "story": 6}
    second = AgentMemory(str(tmp_path / "ch2"))  # as the runner reads it
    assert [e["source"] for e in second.glossary] == ["魔剣", "ドン"] and second.glossary[0]["chapter"] == "第1話"
    assert second.story == "第1話的劇情" and "chapter:" not in second.snapshot_text()

    second.glossary_upsert("魔剣", "魔劍2", "", "005", "Item")
    second.character_upsert("みゃー", {"target_name": "Myaa"}, "001")
    second.story_update("第2話的劇情")
    assert save_chapter(second, work, "第2話") == {"glossary_added": 0, "glossary_updated": 1,
                                                  "characters_added": 1, "story": 6}
    merged = AgentMemory(work)
    assert merged.glossary[0]["target"] == "魔劍2" and merged.story == "第2話的劇情"
    assert merged.characters["みゃー"]["chapter"] == "第2話" and merged.characters["りょうた"]["chapter"] == "第1話"
    meta = json.loads((tmp_path / "work.memory" / "chapters.json").read_text(encoding="utf-8"))
    assert list(meta["chapters"]) == ["第1話", "第2話"] and meta["story_from"] == "第2話"

    # a rerun of chapter 1 keeps what it knew then, not chapter 2's story
    assert seed_chapter(AgentMemory(str(tmp_path / "ch1")), work) == {}
    assert AgentMemory(str(tmp_path / "ch1")).story == "第1話的劇情"


def _old_chapter(parent, name, prompt, n):
    work_dir = parent / name / "manga_translator_work"
    runs = work_dir / "agent" / "runs"
    runs.mkdir(parents=True)
    start = {"type": "run_start", "run": "r1", "work_prompt": {"name": prompt, "terms": 0}}
    (runs / "r1.jsonl").write_text(json.dumps(start) + "\n", encoding="utf-8")
    _chapter_memory(work_dir / "agent" / "memory", n)


def test_the_first_time_the_work_memory_comes_from_the_earlier_chapters(tmp_path):
    parent = tmp_path / "作品"
    _old_chapter(parent, "第３３話_1", "作品.yaml", 33)
    _old_chapter(parent, "第３４話_2", "作品.yaml", 34)
    _old_chapter(parent, "第３２話_0", "別的.yaml", 32)  # run with another prompt file: another work
    _old_chapter(parent, "第３６話_4", "作品.yaml", 36)  # comes after this chapter
    this = str(parent / "第３５話_3" / "manga_translator_work")
    work = str(tmp_path / "dict" / "agent" / "作品.memory")
    assert bootstrap(work, this, "作品.yaml") == ["第３３話_1", "第３４話_2"]
    memory = AgentMemory(work)
    assert memory.story == "第34話的劇情"
    assert memory.glossary[0]["target"] == "魔劍34" and memory.glossary[0]["chapter"] == "第３４話_2"
    assert bootstrap(work, this, "作品.yaml") == []  # only while the work has no memory


def test_a_term_from_an_earlier_chapter_names_that_chapter(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory"))
    memory.replace(glossary=[{"source": "魔剣", "target": "魔劍", "note": "", "page": "003", "category": "Item",
                              "chapter": "第33話"}], characters={}, story="")
    terms, _ = candidates(memory, {}, ["拿著魔劍"], "第34話")
    assert "第33話 p.003" in terms[0].description


def _first_request(client, index):
    return json.dumps(client.requests[index], ensure_ascii=False)


def test_the_next_chapter_starts_with_what_the_first_learned(tmp_path):
    prompt_path = tmp_path / "work.yaml"
    prompt_path.write_bytes(WORK.encode("utf-8"))
    client, lines = _run(tmp_path, str(prompt_path), _learn() + _finish_page() + _finish_page())
    # chapter 1: three requests; chapter 2 starts with the sound term chapter 1 recorded
    assert "ドン → 譯2" not in _first_request(client, 0) and "ドン → 譯2" in _first_request(client, 3)
    folder = tmp_path / "agent" / "work.memory"
    assert [e["source"] for e in AgentMemory(str(folder)).glossary] == ["魔剣", "ドン"]
    assert any(f"Memory shared by the work's chapters: {folder}" in line for line in lines)

    def events(chapter):
        store = AgentStore(str(tmp_path / chapter / "chapter" / "manga_translator_work"))
        return read_events(os.path.join(store.runs_dir, sorted(os.listdir(store.runs_dir))[-1]))

    saved = [e for e in events("ch1") if e["type"] == "work_memory"]
    assert len(saved) == 1 and saved[0]["glossary_added"] == 2 and saved[0]["action"] == "saved"
    start = [e for e in events("ch2") if e["type"] == "run_start"][0]
    assert start["work_memory"] == {"glossary": 2, "characters": 0, "story": 0, "from": []}


def test_chapters_keep_their_own_memory_when_sharing_is_off(tmp_path):
    prompt_path = tmp_path / "work.yaml"
    prompt_path.write_bytes(WORK.encode("utf-8"))
    client, _ = _run(tmp_path, str(prompt_path), _learn() + _finish_page() + _finish_page(), share_memory=False)
    assert "ドン → 譯2" not in _first_request(client, 3)
    assert not (tmp_path / "agent" / "work.memory").exists()
