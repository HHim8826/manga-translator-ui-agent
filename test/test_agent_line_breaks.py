"""AI line breaks and paragraphs in the agent workflow.

With the render setting disable_auto_wrap ("AI line breaks") the renderer breaks lines only at [BR], so
the agent gets the same rules as the classic translators and every source line count. Separate
paragraphs (also the two blocks of a stepped narration box) stay separate regions.
"""
import asyncio
import json
import os
import types

import numpy as np
import pytest

from manga_translator.agent import checks, workflow
from manga_translator.agent.events import EventLog
from manga_translator.agent.page import (PageDoc, Region, RegionError, needs_break, paragraph_break,
                                         with_breaks)
from manga_translator.agent.proofread import PROOF_BREAKS, ProofError, ProofPage, Proofreader, ProofSession, _execute
from manga_translator.agent.runner import LINE_BREAK_RULES, AgentRunner
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.tools import AgentSession, ToolRegistry

from test_agent_core import ScriptedClient, _quad, make_page

LONG = "村裡大概就忙到這時候。"
"""A translation long enough to need a line break when its source has two lines."""


def _two_line_page(tmp_path, stem="001", translations=None):
    """make_page with region 1 made of two columns (a two-line source) right next to each other."""
    image_path, json_path = make_page(tmp_path, stem)
    raw = json.load(open(json_path, encoding="utf-8"))
    page = next(iter(raw.values()))
    page["regions"][0]["lines"] = _quad(40, 20, 60, 120) + _quad(20, 20, 38, 120)
    for region, text in zip(page["regions"], translations or ()):
        region["translation"] = text
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, ensure_ascii=False)
    return image_path, json_path


def _region(rid, *boxes, direction="v"):
    lines = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in boxes]
    return Region(id=rid, data={"lines": lines, "direction": direction})


# --- the [BR] marker -------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("村裡\n大概", "村裡[BR]大概"),
    ("村裡\r\n大概", "村裡[BR]大概"),
    ("[BR]村裡[BR][BR]大概[BR]", "村裡[BR]大概"),
    ("村裡 【BR】 大概<br/>就是", "村裡[BR]大概[BR]就是"),
    ("村裡<BR>大概", "村裡[BR]大概"),
    ("\n村裡大概\n", "村裡大概"),
    ("", ""),
])
def test_breaks_are_written_as_the_apps_marker(raw, expected):
    assert with_breaks(raw) == expected


def test_a_multi_line_source_needs_a_break_unless_the_translation_is_short():
    assert needs_break(2, LONG)
    assert not needs_break(1, LONG)
    assert not needs_break(2, "村裡大概[BR]就忙到這時候。")
    assert not needs_break(3, "嗯……？")  # too short to break


# --- paragraphs --------------------------------------------------------------------
def test_side_by_side_paragraphs_are_not_one_block():
    """Chapter 34 p.7: two paragraphs of a narration box, the screenshot of the user's report."""
    first = _region(5, (258, 130, 295, 470), (219, 130, 251, 438))
    second = _region(6, (136, 149, 171, 357), (95, 150, 127, 525))
    split = paragraph_break([first, second])
    assert split is not None and {split[0].id, split[1].id} == {5, 6}


def test_the_blocks_of_a_stepped_narration_box_are_not_one_block():
    """Chapter 34 p.6: an upper right block and a lower left one, sharing columns."""
    upper = _region(1, (160, 60, 200, 245), (120, 60, 155, 230), (80, 60, 115, 240))
    lower = _region(4, (100, 290, 135, 435), (60, 290, 95, 400))
    assert paragraph_break([upper, lower]) is not None


def test_one_paragraph_cut_into_boxes_is_one_block():
    right = _region(1, (258, 130, 295, 470))
    left = _region(2, (219, 130, 251, 438), (180, 140, 212, 400))
    assert paragraph_break([right, left]) is None
    # horizontal rows: the lines follow each other downwards
    top = _region(1, (20, 20, 300, 50), direction="h")
    bottom = _region(2, (20, 56, 260, 86), direction="h")
    assert paragraph_break([top, bottom]) is None
    far = _region(3, (20, 140, 260, 170), direction="h")
    assert paragraph_break([top, bottom, far]) is not None


def test_a_chain_of_close_boxes_is_one_block():
    """The first and last box are far apart, but each one touches the next."""
    boxes = [_region(i, (300 - 40 * i, 20, 334 - 40 * i, 300)) for i in range(1, 5)]
    assert paragraph_break(boxes) is None


def test_merge_refuses_separate_paragraphs_and_changes_nothing(tmp_path):
    image_path, json_path = make_page(tmp_path)  # regions 1 and 2 have a column of blank space between them
    doc = PageDoc.load(image_path, json_path)
    with pytest.raises(RegionError, match="link_regions"):
        doc.merge([1, 2], "同一個框")
    assert doc.regions[2].merged_into is None
    with pytest.raises(RegionError, match="nothing to undo"):  # no checkpoint was taken
        doc.undo()


# --- the translation agent ---------------------------------------------------------
def _session(tmp_path, line_breaks=True):
    image_path, json_path = _two_line_page(tmp_path)
    session = AgentSession(AgentSettings(double_read="none"), AgentMemory(str(tmp_path / "mem")), [image_path],
                           target_lang="CHT")
    session.line_breaks = line_breaks
    session.start_page(0, PageDoc.load(image_path, json_path))
    return session, ToolRegistry(session)


def test_candidates_give_the_source_line_count_only_with_ai_line_breaks(tmp_path):
    session, _ = _session(tmp_path)
    lines = session.candidate_lines().splitlines()
    assert "2 source line(s)" in lines[0] and "1 source line(s)" in lines[1]
    session.line_breaks = False
    assert "source line" not in session.candidate_lines()


def test_set_text_saves_breaks_as_br_and_points_out_a_missing_one(tmp_path):
    session, registry = _session(tmp_path)
    saved = registry.execute("set_text", {"id": 1, "source_text": "バタバタ", "translation": LONG,
                                          "category": "narration", "confidence": 0.9})
    assert saved.ok and "has no [BR]" in saved.text and "set_line_breaks" in saved.text
    fixed = registry.execute("set_line_breaks", {"id": 1, "text": "村裡大概就忙到\n這時候。"})
    assert fixed.ok and session.doc.regions[1].translation == "村裡大概就忙到[BR]這時候。"
    assert "村裡大概就忙到 / 這時候。" in fixed.summary
    changed = registry.execute("set_line_breaks", {"id": 1, "text": "村裡大概[BR]很忙"})
    assert not changed.ok and "line breaks only" in changed.text
    again = registry.execute("set_text", {"id": 1, "source_text": "バタバタ", "translation": "村裡大概\n就忙到這時候。",
                                          "category": "narration", "confidence": 0.9})
    assert "has no [BR]" not in again.text and session.doc.regions[1].translation == "村裡大概[BR]就忙到這時候。"
    one_line = registry.execute("set_text", {"id": 2, "source_text": "x", "translation": LONG,
                                             "category": "dialogue", "confidence": 0.9})
    assert "[BR]" not in one_line.text


def test_mark_page_done_lists_a_missing_break_once(tmp_path):
    session, registry = _session(tmp_path)
    for rid, text in ((1, LONG), (2, "好"), (3, "走")):
        registry.execute("set_text", {"id": rid, "source_text": "x", "translation": text, "category": "dialogue",
                                      "confidence": 0.95})
    first = registry.execute("mark_page_done", {"confidence": 0.9})
    assert not first.ok and "#1: the source has 2 lines but the translation has no [BR]" in first.text
    assert first.summary == "checks missing_line_break:1"
    done = registry.execute("mark_page_done", {"confidence": 0.9})
    assert done.ok and session.doc.confidence == pytest.approx(0.9)  # a light weight: 1 * (1 - 0.1)
    assert [c["code"] for c in session.doc.checks] == ["missing_line_break"]
    assert not [f for f in session.doc.flags if f.get("region") == 1]  # not on the review list


def test_no_line_break_check_without_ai_line_breaks(tmp_path):
    session, registry = _session(tmp_path, line_breaks=False)
    registry.execute("set_text", {"id": 1, "source_text": "x", "translation": LONG, "category": "dialogue",
                                  "confidence": 0.95})
    assert "[BR]" not in registry.execute("set_text", {"id": 1, "source_text": "x", "translation": LONG,
                                                       "category": "dialogue", "confidence": 0.95}).text
    assert not [i for i in checks.page_issues(session) if i.code == "missing_line_break"]


def test_breaks_do_not_count_as_letters_or_split_terms(tmp_path):
    memory = AgentMemory(str(tmp_path / "mem"))
    memory.glossary_upsert("ヤムパヒテナ", "亞姆帕希特納", "", "Person")
    issues = checks.term_issues(1, "ヤムパヒテナさん", "這位是亞姆帕[BR]希特納先生", [], memory)
    assert issues == []
    region = _region(1, (0, 0, 30, 30))
    assert checks._capacity(region) >= 1 and checks.text_length("好[BR]啊") == 2


def test_the_agent_gets_the_rules_only_with_ai_line_breaks(tmp_path):
    image_path, _ = _two_line_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    on = AgentRunner(AgentSettings(), ScriptedClient([]), store, events, [image_path], "CHT", line_breaks="RULES-X")
    prompt = on.system_prompt()
    assert "RULES-X" in prompt and "original_region_count" in prompt and on.session.line_breaks
    off = AgentRunner(AgentSettings(), ScriptedClient([]), store, events, [image_path], "CHT")
    assert "AI line breaks" not in off.system_prompt() and not off.session.line_breaks
    # merging: one paragraph only, separate paragraphs stay apart
    assert "merge_regions only for one paragraph" in prompt and "Separate paragraphs stay separate" in prompt
    events.close()


def test_rules_come_from_the_line_break_prompt_file(monkeypatch):
    def config(on):
        return types.SimpleNamespace(render=types.SimpleNamespace(disable_auto_wrap=on))

    assert workflow.line_break_rules(config(False)) == ""
    assert workflow.line_break_rules(types.SimpleNamespace()) == ""
    rules = workflow.line_break_rules(config(True))
    assert "original_region_count" in rules and "[BR]" in rules
    from manga_translator.translators import prompt_loader
    monkeypatch.setattr(prompt_loader, "load_line_break_prompt", lambda _dir: None)
    assert workflow.line_break_rules(config(True)) == LINE_BREAK_RULES


# --- proofreading ------------------------------------------------------------------
def _proof_session(tmp_path, line_breaks=True):
    image_path, json_path = _two_line_page(tmp_path, translations=("村裡大概就忙到[BR]這時候。", "好", "走"))
    page = ProofPage(image_path)
    return ProofSession({"001": page}, [], AgentMemory(str(tmp_path / "mem")), "CHT", line_breaks), page, json_path


def test_proofreading_shows_and_keeps_the_breaks(tmp_path):
    session, page, json_path = _proof_session(tmp_path)
    listing = page.listing(line_counts=True)
    assert "2 line(s)]" in listing[0] and "→ 「村裡大概就忙到[BR]這時候。」" in listing[0]
    assert "line(s)" not in page.listing()[0] and "[BR]" in page.listing()[0]
    with pytest.raises(ProofError, match=r"needs \[BR\]"):
        _execute(session, "revise_text", {"page": "001", "id": 1, "translation": "村裡大概只忙到這個時候。", "reason": "r"})
    text, summary, _ = _execute(session, "revise_text", {"page": "001", "id": 1,
                                                         "translation": "村裡大概只忙到\n這個時候。", "reason": "r"})
    assert "revised" in text and "/" in summary
    saved = next(iter(json.load(open(json_path, encoding="utf-8")).values()))["regions"]
    assert saved[0]["translation"] == "村裡大概只忙到[BR]這個時候。"
    # a one-line source needs no break
    _execute(session, "revise_text", {"page": "001", "id": 2, "translation": "好吧，就這麼辦。", "reason": "r"})


def test_proofreading_without_ai_line_breaks_does_not_ask_for_breaks(tmp_path):
    session, _, _ = _proof_session(tmp_path, line_breaks=False)
    _execute(session, "revise_text", {"page": "001", "id": 1, "translation": "村裡大概只忙到這個時候。", "reason": "r"})


def test_the_proofreader_prompt_explains_the_breaks():
    def make(on):
        return Proofreader(AgentSettings(), None, None, None, "CHT", None, line_breaks=on)

    assert PROOF_BREAKS.strip() in make(True).system_prompt()
    off = make(False).system_prompt()
    assert "AI line breaks" not in off and "[BR] in a translation is a line break" in off
