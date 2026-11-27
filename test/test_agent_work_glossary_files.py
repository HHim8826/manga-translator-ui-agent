"""Review 2026-11-28 B17: a comment at the start of a line inside the glossary ended the block (a second
``Location:`` was written and every later write failed its check), ``glossary: {}`` got a second
``glossary:`` at the end that the check let through, two runs wrote through the same ``.tmp`` with no
lock, and an empty prompt file or a JSON one with a BOM was not read."""
import json
import os
import threading

import pytest
import yaml

from manga_translator.agent import work_glossary
from manga_translator.agent.events import EventLog
from manga_translator.agent.runner import AgentRunner
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.work_glossary import NewTerm, insert_yaml, update_work_glossary, write_terms
from manga_translator.agent.work_prompt import load_work_prompt

from test_agent_core import ScriptedClient, make_page

COMMENTED = "\n".join([
    "glossary:",
    "  Person:",
    "    - original: \"A\"",
    "      description: \"\"",
    "# 地點",
    "  Location:",
    "    - original: \"B\"",
    "      description: \"\"",
    "# 其他",
    "other: 1",
])


def _place(source="ドダド村", target="Dodado村"):
    return NewTerm("Location", source, target)


def _keys(text, key):
    return sum(1 for line in text.split("\n") if line.strip() == key)


def test_a_comment_at_the_start_of_a_line_stays_inside_the_glossary():
    out = insert_yaml(COMMENTED, [_place(), NewTerm("Item", "剣", "劍")])
    assert _keys(out, "Location:") == 1 and _keys(out, "glossary:") == 1
    data = yaml.safe_load(out)
    assert [e["original"] for e in data["glossary"]["Location"]] == ["B", "ドダド村"]
    assert data["glossary"]["Item"][0]["original"] == "剣" and data["other"] == 1
    lines = out.split("\n")
    assert lines[lines.index("# 其他") + 1] == "other: 1"  # the comment stays over its key
    assert lines.index("  Item:") < lines.index("# 其他")


@pytest.mark.parametrize("empty", ["{}", "null", "~", "{ }  # 還沒有"])
def test_an_empty_glossary_written_inline_gets_the_categories(tmp_path, empty):
    path = tmp_path / "work.yaml"
    path.write_text(f"system_prompt: x\nglossary: {empty}\n", encoding="utf-8")
    write_terms(str(path), [_place()])
    text = path.read_text(encoding="utf-8")
    assert text.count("glossary:") == 1
    assert yaml.safe_load(text)["glossary"]["Location"][0]["original"] == "ドダド村"


def test_a_key_written_twice_is_not_saved(tmp_path, monkeypatch):
    path = tmp_path / "work.yaml"
    text = "glossary:\n  Person: []\n"
    path.write_text(text, encoding="utf-8")
    # an edit that writes the glossary again (the reader keeps the last one, so it reads back right)
    monkeypatch.setattr(work_glossary, "insert_yaml",
                        lambda old, terms: old + "\nglossary:\n  Person: []\n  Location:\n"
                        + "\n".join(work_glossary._entry_lines(terms[0], 4)))
    with pytest.raises(ValueError, match="twice"):
        write_terms(str(path), [_place()])
    assert path.read_text(encoding="utf-8") == text
    assert work_glossary._duplicate_keys("a: 1\nb:\n  c: 1\n  c: 2\na: 3\n") == 2


def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "work.yaml"
    path.write_text("glossary:\n", encoding="utf-8")

    def fail(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(work_glossary.os, "replace", fail)
    with pytest.raises(PermissionError):
        write_terms(str(path), [_place()])
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []
    assert not os.path.exists(str(path) + ".tmp")


def test_two_runs_write_the_prompt_file_one_after_the_other(tmp_path):
    path = str(tmp_path / "work.yaml")
    open(path, "w", encoding="utf-8").write("glossary:\n")
    failed = []
    with work_glossary.locked(path):
        def other():
            try:
                with work_glossary.locked(path, wait=0.3):
                    pass
            except TimeoutError:
                failed.append(True)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert failed == [True]
    with work_glossary.locked(path, wait=0.3):  # let go
        pass

    memory = AgentMemory(str(tmp_path / "memory"))
    memory.glossary_upsert("ドダド村", "Dodado村", "", "001", "Location")
    reads = []
    real = work_glossary.read_prompt_file

    def read(p):
        # the read is made with the lock held: another run cannot take it in between
        with pytest.raises(TimeoutError):
            with work_glossary.locked(p, wait=0):
                pass
        reads.append(p)
        return real(p)

    work_glossary.read_prompt_file, saved = read, work_glossary.read_prompt_file
    try:
        assert update_work_glossary(path, memory, ["去Dodado村"], "c")["added"]
    finally:
        work_glossary.read_prompt_file = saved
    assert reads


@pytest.mark.parametrize("name, content", [("work.yaml", ""), ("work.yaml", "{}\n"), ("work.json", ""),
                                           ("work.json", "{}")])
def test_an_empty_prompt_file_is_a_work_prompt(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    prompt = load_work_prompt(str(path), "Traditional Chinese")
    assert prompt is not None and prompt.text == "" and prompt.path == str(path)

    pages = [make_page(tmp_path, "001")]
    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    runner = AgentRunner(AgentSettings(), ScriptedClient([]), store, events, [p for p, _ in pages], "CHT",
                         work_prompt=prompt)
    text = runner.system_prompt()
    events.close()
    assert "Notes for this work" not in text  # nothing to tell the model


def test_a_json_prompt_file_with_a_bom_is_read(tmp_path):
    from manga_translator.translators.prompt_loader import load_prompt_file

    path = tmp_path / "work.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"system_prompt": "譯好"}, ensure_ascii=False).encode("utf-8"))
    assert load_prompt_file(str(path)) == {"system_prompt": "譯好"}
    assert "譯好" in load_work_prompt(str(path)).text
