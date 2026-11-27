"""Review 2026-11-28 B19 and U2: the tool_result of flag_for_review wrote over the flag's text (the
inpaint review's area 4 became #4, the proofreader's flag lost its kind) and the rows of two agents on
one page were mixed up, a redo counted its requests twice, the flag count went down while the
proofreading ran beside the inpaint review, a render that failed set the preview's output, and some
tool summaries were shown in English."""
import json

import _bootstrap  # noqa: F401,I001
import pytest

from ui.main_page.pages.agent_results import result_text

from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)

CATALOG = json.loads((_bootstrap.ROOT / "desktop_qt_ui" / "locales" / "zh_TW.json").read_text(encoding="utf-8"))


def _tw(key, **kwargs):
    return CATALOG[key].format(**kwargs)


def _page():
    from ui.main_page.pages.agents_page import AgentsPage

    return AgentsPage(_tw)


def _rows(page):
    return [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]


def test_the_flag_text_stays_and_two_agents_keep_their_own_rows(app):
    page = _page()
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "tool_call", "page": "001", "agent": "inpaint", "tool": "flag_for_review", "args": {"id": 4}},
        {"type": "tool_call", "page": "001", "agent": "proofread", "tool": "flag_for_review", "args": {"id": 4}},
        {"type": "flag", "page": "001", "agent": "proofread", "region": 4, "flag": "uncertain_translation"},
        {"type": "tool_result", "page": "001", "agent": "proofread", "tool": "flag_for_review", "ok": True,
         "summary": "001 #4"},
        {"type": "flag", "page": "001", "agent": "inpaint", "region": 4, "flag": "inpaint"},
        {"type": "tool_result", "page": "001", "agent": "inpaint", "tool": "flag_for_review", "ok": True,
         "summary": "#4 inpaint"},
    ])
    inpaint, proofread = _rows(page)
    assert inpaint.text(3) == "區域 4：修復" and inpaint.toolTip(3) == "#4 inpaint"
    assert proofread.text(3) == "第 4 框：譯文不確定" and proofread.toolTip(3) == "001 #4"
    # with no flag event (already listed) the result is shown, in the interface language
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "tool_call", "page": "001", "agent": "proofread", "tool": "flag_for_review", "args": {"id": 4}},
        {"type": "tool_result", "page": "001", "agent": "proofread", "tool": "flag_for_review", "ok": True,
         "summary": "001 #4: already listed"},
    ])
    assert _rows(page)[-1].text(3) == "001 第 4 框: 已在審查清單上"


def test_a_redo_counts_each_request_once(app):
    page = _page()
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m", "redo": True},
        {"type": "inpaint_done", "page": "001", "agent": "inpaint", "usage": {"requests": 3}, "flags": 0},
        {"type": "proofread_done", "agent": "proofread", "usage": {"requests": 2}},
        {"type": "run_end", "pages": 1, "done": 1, "totals": {"requests": 5}, "redo": True},
    ])
    assert page._requests == 5
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "inpaint_done", "page": "001", "agent": "inpaint", "usage": {"requests": 3}, "flags": 0},
        {"type": "run_end", "pages": 1, "done": 1, "totals": {"requests": 4}},  # the translator's own
    ])
    assert page._requests == 7


def test_the_flag_count_does_not_go_down_while_the_proofreading_runs(app):
    page = _page()
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "page_done", "page": "001", "status": "done", "flags": 1},
        {"type": "proofread_start", "pages": 1, "regions": 3},
        {"type": "flag", "page": "001", "agent": "proofread", "region": 2, "flag": "uncertain_translation"},
        # the inpaint review saved the page before the proofreading did: its count has no proofread flag
        {"type": "inpaint_done", "page": "001", "agent": "inpaint", "flags": 2, "status": "done"},
    ])
    assert page._pages["001"]["flags"] == 3
    page.handle_event({"type": "proofread_done", "agent": "proofread", "flagged": 1})
    page.handle_event({"type": "page_done", "page": "001", "status": "done", "flags": 3})
    assert page._pages["001"]["flags"] == 3


@pytest.mark.parametrize("result", ["failed", "skipped"])
def test_a_render_without_output_leaves_the_preview(app, result):
    page = _page()
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "page_done", "page": "001", "status": "done", "image": "C:/x/001.jpg"},
        {"type": "render_done", "page": "001", "result": result, "output": "C:/x/out/001.png"},
    ])
    assert "output" not in page.preview.pages["001"]
    page.handle_event({"type": "render_done", "page": "001", "result": "saved", "output": "C:/x/out/001.png"})
    assert page.preview.pages["001"]["output"] == "C:/x/out/001.png"


@pytest.mark.parametrize("tool, summary, agent, expected", [
    ("undo", "fit #3", "translator", "復原：貼齊文字框 #3"),
    ("adjust_mask", "#1 dilate 3px (M1,M2)", "inpaint", "#1 遮罩擴大 3px（只在 M1,M2）"),
    ("note_area", "#2 note", "inpaint", "區域 2 已加註記"),
    ("fit_region", "#4 120x80 font 28", "translator", "#4 120x80 字級 28"),
    ("fit_region", "#4 font 28", "translator", "#4 字級 28"),
    ("revise_text", "page 01 #3: old → new", "proofread", "page 01 第 3 框: old → new"),
    ("flag_for_review", "001 #3: already listed", "proofread", "001 第 3 框: 已在審查清單上"),
    ("split_region", "#3 → #3 + #7 (cut)", "translator", "#3 → #3 + #7 （切開框）"),
    ("mask_paint", "#10 paint add B3 : D5", "inpaint", "#10 加入遮罩 B3:D5"),
    ("undo", "#10 paint remove C4", "inpaint", "復原：#10 移出遮罩 C4"),
    ("flag_for_review", "#4 inpaint", "inpaint", "區域 4：修復"),
    ("flag_for_review", "#4 inpaint", "translator", "#4：修復"),
])
def test_summaries_in_the_interface_language(tool, summary, agent, expected):
    from ui.main_page.pages.agents_page import TOOL_KEYS

    assert result_text(_tw, tool, summary, True, TOOL_KEYS, agent=agent)[0] == expected
