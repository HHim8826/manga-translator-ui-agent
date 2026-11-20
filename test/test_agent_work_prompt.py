"""The work's prompt file (high_quality_prompt_path) in the agent: system prompt, checks, workflow."""
import asyncio
import os

from manga_translator.agent import checks
from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.runner import AgentRunner
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore
from manga_translator.agent.work_prompt import load_work_prompt, parse

from test_agent_core import FakeTranslator, ScriptedClient, _calls, _finish_page, _session, make_page

YAML = """system_prompt: |
  譯成{{{target_lang}}}，語氣輕鬆。
glossary:
  Person:
    - original: ミヤムトゥー
      aliases:
        - original: ミヤムトゥー
          translations:
            - text: Mi Yam Twu
              condition: ""
        - original: みゃーさん
          translations:
            - text: Myaa小姐
              condition: 主角稱呼她時
        - original: みゃー
          translations:
            - text: Myaa
              condition: ""
      description: 女主角。
      overwrite: false
  Location: []
  Creature:
    - original: キバ魚
      aliases:
        - original: キバ魚
          translations:
            - text: 牙魚
              condition: 自擬
      description: ""
"""


def _prompt_file(tmp_path):
    path = tmp_path / "work.yaml"
    path.write_text(YAML, encoding="utf-8")
    return str(path)


def test_prompt_file_becomes_text_and_terms(tmp_path):
    prompt = load_work_prompt(_prompt_file(tmp_path), "Traditional Chinese")
    assert prompt.name == "work.yaml"
    assert prompt.instructions == "譯成Traditional Chinese，語氣輕鬆。"
    assert "- ミヤムトゥー → Mi Yam Twu; みゃーさん → Myaa小姐 (主角稱呼她時); みゃー → Myaa" in prompt.text
    assert "  女主角。" in prompt.text and "[Creature]" in prompt.text and "[Location]" not in prompt.text
    assert [(t.source, t.targets) for t in prompt.terms] == [
        ("ミヤムトゥー", ["Mi Yam Twu"]), ("みゃーさん", ["Myaa小姐"]), ("みゃー", ["Myaa"]), ("キバ魚", ["牙魚"])]
    # the same path without the extension is found, like the translators do; a missing file is None
    assert load_work_prompt(_prompt_file(tmp_path)[:-5] + ".json") is not None
    assert load_work_prompt(str(tmp_path / "nothing.yaml")) is None
    assert parse({"system_prompt": "", "glossary": {"Person": []}}, "empty").text == ""


def test_checks_use_the_work_glossary_and_prefer_the_longer_term(tmp_path):
    session, _ = _session(tmp_path)
    session.work_terms = load_work_prompt(_prompt_file(tmp_path)).terms
    region = session.doc.regions[1]

    def codes(source, translation):
        region.source_text, region.translation = source, translation
        return [(i.code, i.detail.get("source")) for i in checks.region_issues(session, region)
                if i.code == "glossary_mismatch"]

    assert codes("みゃーさん、キバ魚だよ", "Myaa小姐，是牙魚喔") == []
    assert codes("みゃーさん、行こう", "Myaa，走吧") == [("glossary_mismatch", "みゃーさん")]  # not also みゃー
    assert codes("ミヤムトゥー！", "米雅！") == [("glossary_mismatch", "ミヤムトゥー")]
    # the memory glossary does not repeat a term the work file already checked
    session.memory.glossary_upsert("キバ魚", "尖牙魚")
    assert codes("キバ魚", "牙魚") == []


def test_system_prompt_and_glossary_search_include_the_work_file(tmp_path):
    pages = [make_page(tmp_path, "001")]
    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    prompt = load_work_prompt(_prompt_file(tmp_path), "Traditional Chinese")
    runner = AgentRunner(AgentSettings(extra_instructions="短一點"), ScriptedClient([]), store, events,
                         [p for p, _ in pages], "CHT", work_prompt=prompt)
    text = runner.system_prompt()
    assert "user's prompt file work.yaml" in text and "みゃーさん → Myaa小姐" in text
    assert text.index("work.yaml") < text.index("短一點")
    found = runner.registry.execute("glossary_search", '{"query": "Myaa"}')
    events.close()
    assert "work prompt file" in found.text and "みゃーさん" in found.text


def test_workflow_reads_the_translator_prompt_path(tmp_path):
    from manga_translator.agent.workflow import run_agent_workflow

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png")]
    prompt_path = _prompt_file(tmp_path)

    class Cfg:
        class translator:
            target_lang = "CHT"
            high_quality_prompt_path = prompt_path

    client = ScriptedClient(_finish_page())
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {}, client=client,
                                   log=lambda m: None, agent_settings={"inpaint_review": False}))
    assert "みゃーさん → Myaa小姐" in client.requests[0][0]["content"]
    store = AgentStore(str(folder / "manga_translator_work"))
    log = os.path.join(store.runs_dir, sorted(os.listdir(store.runs_dir))[-1])
    start = read_events(log)[0]
    assert start["type"] == "run_start" and start["work_prompt"] == {"name": "work.yaml", "terms": 4}
    # off: the agent's prompt does not carry it
    client = ScriptedClient(_finish_page())
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {}, client=client,
                                   log=lambda m: None, agent_settings={"inpaint_review": False,
                                                                       "use_work_prompt": False}))
    assert "Myaa" not in client.requests[0][0]["content"]
