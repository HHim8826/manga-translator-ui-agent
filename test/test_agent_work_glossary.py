"""New names and terms written back into the work's prompt file (setting update_work_glossary)."""
import asyncio
import json
import os

import pytest
import yaml

from manga_translator.agent.events import read_events
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.work_glossary import candidates, insert_yaml, update_work_glossary

from test_agent_core import FakeTranslator, ScriptedClient, _calls, _finish_page, _session, make_page

# the layout of a prompt file the editor wrote and the user commented: CRLF, no newline at the end
WORK = "\r\n".join([
    "# 作品：試驗用",
    "system_prompt: |",
    "  譯成{{{target_lang}}}。",
    "",
    "glossary:",
    "  # 主要角色",
    "  Person:",
    "    - original: \"ミヤムトゥー\"",
    "      aliases:",
    "        - original: \"ミヤムトゥー\"",
    "          translations:",
    "            - text: \"Mi Yam Twu\"",
    "              condition: \"\"",
    "        - original: \"みゃー\"",
    "          translations:",
    "            - text: \"Myaa\"",
    "              condition: \"\"",
    "      description: \"女主角。\"  # 不要改",
    "      overwrite: false",
    "",
    "  Location: []",
    "  Creature:",
    "    - original: \"キバ魚\"",
    "      aliases:",
    "        - original: \"キバ魚\"",
    "          translations:",
    "            - text: \"牙魚\"",
    "              condition: \"\"",
    "      description: \"\"",
    "      overwrite: false",
    "# 檔案結尾的註解",
])

TRANSLATIONS = ["Myaa，去Dodado村吧", "那隻咕欸在叫", "拿著星之劍", "Riyota說"]


def _memory(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory"))
    memory.character_upsert("みゃー", {"target_name": "Myaa"}, "001")  # an alias of an entry
    memory.character_upsert("りょうた", {"target_name": "Riyota", "notes": "主角"}, "002")
    memory.character_upsert("ディメ", {"target_name": "咕欸獸", "category": "Creature"}, "003")  # not used
    memory.glossary_upsert("ドダド村", "Dodado村", "村名", "001", "Location")
    memory.glossary_upsert("くえ", "咕欸", "叫聲", "002")  # no category: a sound
    memory.glossary_upsert("星の剣", "星之劍", "", "004", "Item")
    memory.glossary_upsert("キバ魚", "尖牙魚", "", "004", "Creature")  # already in the file
    return memory


def test_new_terms_go_into_their_categories_and_the_file_keeps_its_layout(tmp_path):
    path = tmp_path / "work.yaml"
    path.write_bytes(WORK.encode("utf-8"))
    result = update_work_glossary(str(path), _memory(tmp_path), TRANSLATIONS, "第１話")
    assert result["added"] == [{"category": "Person", "source": "りょうた", "target": "Riyota"},
                               {"category": "Location", "source": "ドダド村", "target": "Dodado村"},
                               {"category": "Item", "source": "星の剣", "target": "星之劍"}]
    assert result["skipped"] == {"known": ["みゃー", "キバ魚"], "other": ["くえ"], "unused": ["ディメ"]}

    raw = path.read_bytes()
    assert (tmp_path / "work.yaml.bak").read_bytes() == WORK.encode("utf-8")
    assert raw.count(b"\r\n") == raw.count(b"\n") and not raw.endswith(b"\n")
    old, new = WORK.split("\r\n"), raw.decode("utf-8").split("\r\n")
    # every old line is still there and in order, except the empty Location list that now has an entry
    it = iter(new)
    assert all(line in it for line in old if line != "  Location: []")
    assert "  Location:" in new and new[-1] == "# 檔案結尾的註解"
    i = new.index("    - original: \"りょうた\"")
    assert new[i - 2:i] == ["      overwrite: false", ""]  # after the last Person entry, a blank line apart
    assert new[i + 6] == "      description: \"主角（AI agent 自動加入：第１話 p.002）\""
    assert new[new.index("  Location:") + 1] == "    - original: \"ドダド村\""
    assert new[new.index("  Item:") - 2:new.index("  Item:")] == ["      overwrite: false", ""]  # a new section at the end

    data = yaml.safe_load(raw.decode("utf-8"))
    assert [e["original"] for e in data["glossary"]["Person"]] == ["ミヤムトゥー", "りょうた"]
    assert data["glossary"]["Person"][0]["description"] == "女主角。"
    assert data["glossary"]["Location"][0]["aliases"] == [
        {"original": "ドダド村", "translations": [{"text": "Dodado村", "condition": ""}]}]
    assert list(data["glossary"]) == ["Person", "Location", "Creature", "Item"]

    # the next chapter finds them in the file and adds nothing
    again = update_work_glossary(str(path), _memory(tmp_path), TRANSLATIONS, "第２話")
    assert again["added"] == [] and path.read_bytes() == raw


def test_an_inline_list_is_not_rewritten(tmp_path):
    path = tmp_path / "work.yaml"
    text = "glossary:\n  Person: [{original: A}]\n"
    path.write_text(text, encoding="utf-8")
    memory = AgentMemory(str(tmp_path / "memory"))
    memory.character_upsert("りょうた", {"target_name": "Riyota"})
    with pytest.raises(ValueError):
        update_work_glossary(str(path), memory, ["Riyota"], "c")
    assert path.read_text(encoding="utf-8") == text and not (tmp_path / "work.yaml.bak").exists()


def test_files_without_a_glossary_and_json_files(tmp_path):
    term = AgentMemory(str(tmp_path / "memory"))
    term.glossary_upsert("ギルド", "公會", "", "001", "Org")
    path = tmp_path / "work.yaml"
    path.write_bytes("﻿system_prompt: 譯好\n\n".encode("utf-8"))  # a BOM, LF, two newlines at the end
    assert update_work_glossary(str(path), term, ["去公會"], "c")["added"]
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbfsystem_prompt") and raw.endswith(b"overwrite: false\n\n")
    assert yaml.safe_load(raw.decode("utf-8-sig"))["glossary"]["Org"][0]["original"] == "ギルド"

    path = tmp_path / "work.json"
    path.write_text(json.dumps({"system_prompt": "x", "glossary": {"Person": [], "Org": []}}), encoding="utf-8")
    assert update_work_glossary(str(path), term, ["去公會"], "c")["added"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["glossary"]["Org"][0]["aliases"][0]["translations"][0]["text"] == "公會"
    assert json.loads((tmp_path / "work.json.bak").read_text(encoding="utf-8"))["glossary"]["Org"] == []
    # an empty layout, and one whose glossary is not divided into categories
    assert insert_yaml("", []) == "glossary:"
    path.write_text(json.dumps({"glossary": [{"original": "A"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        update_work_glossary(str(path), term, ["去公會"], "c")


def test_memory_tools_record_the_category(tmp_path):
    session, tools = _session(tmp_path)
    assert tools.execute("glossary_upsert", json.dumps({"source": "ギルド", "target": "公會", "category": "Org",
                                                        "reason": "term"})).ok
    assert tools.execute("glossary_upsert", json.dumps({"source": "くえ", "target": "咕欸", "category": "Other",
                                                        "reason": "a sound"})).ok
    missing = tools.execute("glossary_upsert", json.dumps({"source": "X", "target": "Y", "reason": "r"}))
    assert not missing.ok and "category" in missing.text
    assert [e.get("category") for e in session.memory.glossary] == ["Org", None]
    assert tools.execute("character_upsert", json.dumps({"name": "ディメ", "target_name": "Dime",
                                                         "category": "Creature", "reason": "a dragon"})).ok
    assert session.memory.characters["ディメ"]["category"] == "Creature"


class TwoChapters(FakeTranslator):
    """The preprocess writes each page next to its own image (``<base>/chapter/<stem>.png``)."""

    async def translate_batch(self, images_with_configs, save_info=None):
        if self.template:
            for path, _ in images_with_configs:
                make_page(type(self.tmp_path)(os.path.dirname(os.path.dirname(path))),
                          os.path.splitext(os.path.basename(path))[0])
            self.template = False
            try:
                return await super().translate_batch(images_with_configs, save_info)
            finally:
                self.template = True
        return await super().translate_batch(images_with_configs, save_info)


def _run(tmp_path, prompt_path, script, **agent_settings):
    from manga_translator.agent.workflow import run_agent_workflow

    paths = []
    for name in ("ch1", "ch2"):
        (tmp_path / name / "chapter").mkdir(parents=True, exist_ok=True)
        paths.append(str(tmp_path / name / "chapter" / "001.png"))

    class Cfg:
        class translator:
            target_lang = "CHT"
            high_quality_prompt_path = prompt_path

    client = ScriptedClient(script)
    lines = []
    asyncio.run(run_agent_workflow(TwoChapters(tmp_path), [(p, Cfg) for p in paths], {}, client=client,
                                   log=lines.append, agent_settings={"inpaint_review": False, **agent_settings}))
    return client, lines


def _learn():
    return [_calls(("glossary_upsert", {"source": "魔剣", "target": "譯1", "category": "Item", "reason": "an item"}),
                   ("glossary_upsert", {"source": "ドン", "target": "譯2", "category": "Other", "reason": "a sound"}))]


def test_workflow_saves_the_chapter_terms_and_the_next_chapter_reads_them(tmp_path):
    prompt_path = tmp_path / "work.yaml"
    prompt_path.write_bytes(WORK.encode("utf-8"))
    client, lines = _run(tmp_path, str(prompt_path), _learn() + _finish_page() + _finish_page())

    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    assert [e["original"] for e in data["glossary"]["Item"]] == ["魔剣"]
    assert "AI agent 自動加入：chapter p.001" in data["glossary"]["Item"][0]["description"]
    assert (tmp_path / "work.yaml.bak").read_bytes() == WORK.encode("utf-8")
    systems = [r[0]["content"] for r in client.requests]
    assert "魔剣 → 譯1" not in systems[0] and "魔剣 → 譯1" in systems[-1]
    assert any("1 new term(s) saved to work.yaml: 魔剣 → 譯1" in line for line in lines)

    store = AgentStore(str(tmp_path / "ch1" / "chapter" / "manga_translator_work"))
    log = os.path.join(store.runs_dir, sorted(os.listdir(store.runs_dir))[-1])
    saved = [e for e in read_events(log) if e["type"] == "work_glossary"]
    assert saved == [dict(saved[0], agent="memory", file="work.yaml",
                          added=[{"category": "Item", "source": "魔剣", "target": "譯1"}], skipped={"other": 1})]


def test_workflow_leaves_the_file_alone_when_off_or_broken(tmp_path):
    prompt_path = tmp_path / "work.yaml"
    prompt_path.write_bytes(WORK.encode("utf-8"))
    _run(tmp_path, str(prompt_path), _learn() + _finish_page() + _finish_page(), update_work_glossary=False)
    assert prompt_path.read_bytes() == WORK.encode("utf-8") and not (tmp_path / "work.yaml.bak").exists()

    broken = "glossary:\n  Item: [{original: A}]\n"
    prompt_path.write_text(broken, encoding="utf-8")
    client, lines = _run(tmp_path / "again", str(prompt_path), _learn() + _finish_page() + _finish_page())
    assert prompt_path.read_text(encoding="utf-8") == broken
    assert any("new terms not saved (ValueError" in line for line in lines)
    assert len(client.requests) == 5 and not client.script  # the second chapter was still translated


def test_the_users_own_version_stays_in_the_bak_and_the_last_versions_are_kept(tmp_path, monkeypatch):
    from manga_translator.agent import work_glossary

    path = tmp_path / "work.yaml"
    path.write_bytes(WORK.encode("utf-8"))
    stamps = iter(f"20261121-1200{i:02d}" for i in range(20))
    monkeypatch.setattr(work_glossary.time, "strftime", lambda fmt: next(stamps))
    versions = []
    for i in range(7):  # seven chapters, each adds a term
        versions.append(path.read_bytes())
        memory = AgentMemory(str(tmp_path / f"memory{i}"))
        memory.glossary_upsert(f"剣{i}", f"劍{i}", "", "001", "Item")
        assert update_work_glossary(str(path), memory, [f"劍{i}"], f"第{i}話")["added"]
    assert (tmp_path / "work.yaml.bak").read_bytes() == WORK.encode("utf-8")  # never replaced
    backups = tmp_path / "agent" / "backups"
    kept = sorted(os.listdir(backups))
    assert kept == [f"work.20261121-1200{i:02d}.yaml" for i in range(2, 7)]  # the last five
    assert [(backups / name).read_bytes() for name in kept] == versions[2:]
    # two changes in one second do not overwrite each other
    monkeypatch.setattr(work_glossary.time, "strftime", lambda fmt: "20261121-120006")
    assert work_glossary.backup(str(path)).endswith("work.20261121-120006-01.yaml")


def test_the_note_of_an_added_term_is_in_the_target_language(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory"))
    memory.glossary_upsert("星の剣", "Star Sword", "a weapon", "002", "Item")
    data = {"glossary": {"Item": []}}
    english = candidates(memory, data, ["the Star Sword"], "ch1", "ENG")[0][0]
    assert english.description == "a weapon (added by the AI agent: ch1 p.002)"
    simplified = candidates(memory, data, ["Star Sword"], "第1话", "CHS")[0][0]
    assert simplified.description == "a weapon（AI agent 自动加入：第1话 p.002）"
