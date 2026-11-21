"""Agents page: live events fill the page list and the timeline; filters work."""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "desktop_qt_ui"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from manga_translator.agent.events import EventLog  # noqa: E402


_APP = []


@pytest.fixture(scope="module")
def app():
    if not _APP:  # kept for the whole session: a second QApplication breaks qfluentwidgets' config object
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _t(key, **kwargs):
    return key + (" " + " ".join(f"{k}={v}" for k, v in sorted(kwargs.items())) if kwargs else "")


def _feed(page):
    events = [
        {"type": "run_start", "pages": ["001.jpg", "002.jpg"], "model": "m", "memory": {"glossary": 1}},
        {"type": "page_start", "page": "001", "index": 0, "total": 2},
        {"type": "tool_call", "page": "001", "tool": "zoom", "args": {"id": 3}, "reason": "small text",
         "ts": "2026-11-20T21:14:09+08:00"},
        {"type": "tool_result", "page": "001", "tool": "zoom", "ok": True, "summary": "#3 ×3"},
        {"type": "tool_call", "page": "001", "tool": "mark_page_done", "args": {"confidence": 0.9}, "reason": ""},
        {"type": "tool_result", "page": "001", "tool": "mark_page_done", "ok": False, "summary": "missing #4"},
        {"type": "flag", "page": "001", "region": 4, "flag": "uncertain_ocr", "reason": "handwriting"},
        {"type": "usage", "page": "001", "prompt_tokens": 100_000, "limit": 400_000},
        {"type": "page_done", "page": "001", "status": "done", "flags": 1, "memory": {"glossary": 2, "characters": 1}},
        {"type": "run_end", "pages": 1, "done": 1, "flagged": 1, "totals": {"requests": 4}},
    ]
    page.replay(events)


def test_events_fill_timeline_and_pages(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    _feed(page)
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(2) for r in rows] == ["Agent Tool zoom", "Agent Tool mark_page_done", "Agents Flag"]
    assert rows[0].text(3) == "#3 ×3" and rows[0].text(4) == "small text" and rows[0].text(0) == "21:14:09"
    assert rows[1].text(3) == "Agent Result Missing ids=#4" and rows[1].toolTip(3) == "missing #4"
    items = [page.page_list.item(i).text() for i in range(page.page_list.count())]
    assert items[0] == "Agents All Pages"
    assert items[1] == "001 · Agent Page Done · Agents Flag Count count=1"
    assert items[2] == "002 · Agent Page Waiting"
    assert page.context_bar.value() == 250
    assert "glossary=2" in page.memory_label.text()
    assert "done=1" in page.status_label.text()

    page.problems_only.setChecked(True)
    assert [r.isHidden() for r in rows] == [True, False, False]
    page.problems_only.setChecked(False)
    page.page_list.setCurrentRow(2)  # page 002 has no events
    assert all(r.isHidden() for r in rows)
    page.page_list.setCurrentRow(0)
    assert not any(r.isHidden() for r in rows)


def test_flag_for_review_call_is_one_highlighted_row(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["003.jpg"], "model": "m"},
        {"type": "tool_call", "page": "003", "tool": "flag_for_review", "args": {"type": "missing_text"},
         "reason": "handwritten sigh"},
        {"type": "flag", "page": "003", "region": None, "flag": "missing_text", "reason": "handwritten sigh"},
        {"type": "tool_result", "page": "003", "tool": "flag_for_review", "ok": True, "summary": "missing_text"},
        {"type": "flag", "page": "003", "region": None, "flag": "step_limit", "reason": "40 steps"},
    ])
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(2) for r in rows] == ["Agent Tool flag_for_review", "Agents Flag"]
    page.problems_only.setChecked(True)
    assert not any(r.isHidden() for r in rows)
    assert page.page_list.item(1).text().endswith("count=2")


def test_recall_events_show_added_regions_and_failures(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["004.jpg"], "model": "m"},
        {"type": "recall", "page": "004", "added": [12, 13], "ignored": {"punctuation": 3, "duplicate": 1}},
        {"type": "recall", "page": "004", "added": [], "error": "RuntimeError: down"},
        {"type": "tool_call", "page": "004", "tool": "add_region", "args": {"box": [1, 2, 3, 4]}, "reason": "手寫"},
    ])
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(2) for r in rows] == ["Agents Recall", "Agents Recall", "Agent Tool add_region"]
    assert rows[0].text(3) == "Agents Recall Detail added=2 ignored=4 · #12, #13"
    assert rows[1].text(3) == "RuntimeError: down"
    page.problems_only.setChecked(True)
    assert [r.isHidden() for r in rows] == [True, False, True]
    page.handle_event({"type": "page_done", "page": "004", "status": "done", "confidence": 0.846, "flags": 0})
    assert page.page_list.item(1).text() == "004 · Agent Page Done · Agents Page Confidence percent=85"


def test_live_events_arrive_through_the_listener(app, tmp_path):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    log = EventLog(str(tmp_path / "r.jsonl"), "r")
    log.emit("run_start", pages=["009.jpg"], model="m")
    log.emit("error", page="009", error="boom")
    log.close()
    app.processEvents()
    assert page.timeline.topLevelItemCount() == 1
    assert page.timeline.topLevelItem(0).text(3) == "boom"
    page.deleteLater()
    app.processEvents()


def test_inpaint_review_events(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["005.jpg"], "model": "m"},
        {"type": "page_done", "page": "005", "status": "done", "confidence": 0.99, "flags": 0},
        {"type": "inpaint_start", "page": "005", "agent": "inpaint", "units": 9, "issues": 2},
        {"type": "tool_call", "page": "005", "agent": "inpaint", "tool": "protect_lines", "args": {"id": 3},
         "reason": "泡泡邊框被擦斷"},
        {"type": "tool_result", "page": "005", "agent": "inpaint", "tool": "protect_lines", "ok": True,
         "summary": "#3 protect lines 2px"},
    ])
    assert page.page_list.item(1).text() == "005 · Agent Page Inpainting · Agents Page Confidence percent=99"
    page.handle_event({"type": "inpaint_done", "page": "005", "agent": "inpaint", "status": "done", "units": 9,
                       "retried": 1, "flagged": 1, "confidence": 0.731, "flags": 1})
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(2) for r in rows] == ["Agents Inpaint", "Agent Tool protect_lines", "Agents Inpaint"]
    assert rows[0].text(3) == "Agents Inpaint Start issues=2 units=9"
    assert rows[2].text(3) == "Agents Inpaint Detail flagged=1 retried=1 units=9"
    page.problems_only.setChecked(True)
    assert [r.isHidden() for r in rows] == [True, True, False]
    page.handle_event({"type": "inpaint_done", "page": "006", "agent": "inpaint", "status": "done", "units": 12,
                       "filled": 9, "retried": 0, "flagged": 0, "confidence": 0.99, "flags": 0})
    row = page.timeline.topLevelItem(page.timeline.topLevelItemCount() - 1)
    assert row.text(3) == "Agents Inpaint Detail Filled filled=9 flagged=0 retried=0 units=12" and row.isHidden()
    assert page.page_list.item(1).text() == "005 · Agent Page Done · Agents Page Confidence percent=73 · Agents Flag Count count=1"


def test_status_follows_the_agent_while_pages_are_reviewed_alongside(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["001.jpg", "002.jpg"], "model": "m"},
        {"type": "page_start", "page": "001", "index": 0, "total": 2},
        {"type": "page_done", "page": "001", "status": "done", "confidence": 0.99, "flags": 0},
        {"type": "page_start", "page": "002", "index": 1, "total": 2},
        {"type": "inpaint_start", "page": "001", "agent": "inpaint", "units": 3, "issues": 0},
    ])
    assert page.status_label.text().startswith("Agents Page Progress") and "page=002" in page.status_label.text()
    page.handle_event({"type": "inpaint_done", "page": "001", "agent": "inpaint", "status": "done", "units": 3,
                       "retried": 0, "flagged": 0, "confidence": 0.99, "flags": 0})
    page.handle_event({"type": "page_done", "page": "002", "status": "done", "confidence": 0.99, "flags": 0})
    page.handle_event({"type": "inpaint_start", "page": "002", "agent": "inpaint", "units": 2, "issues": 0})
    page.handle_event({"type": "run_end", "pages": 2, "done": 2, "flagged": 0, "totals": {"requests": 6}})
    assert page.status_label.text() == "Agents Inpaint Progress page=002"
    page.handle_event({"type": "error", "page": "002", "agent": "inpaint", "error": "RuntimeError: boom"})
    assert page.status_label.text().startswith("Agents Finished") and "done=2" in page.status_label.text()
    assert page.page_list.item(2).text().startswith("002 · Agent Page Done")


def test_proofread_events(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "page_done", "page": "001", "status": "done", "confidence": 0.99, "flags": 0},
        {"type": "run_end", "pages": 1, "done": 1, "flagged": 0, "totals": {"requests": 3}},
        {"type": "proofread_start", "agent": "proofread", "pages": 1, "regions": 7},
        {"type": "tool_call", "page": "001", "agent": "proofread", "tool": "revise_text", "args": {"id": 2},
         "reason": "語氣"},
        {"type": "tool_result", "page": "001", "agent": "proofread", "tool": "revise_text", "ok": True,
         "summary": "001 #2: 甲 → 乙"},
    ])
    assert page.status_label.text() == "Agents Proofread Progress"
    page.handle_event({"type": "proofread_done", "agent": "proofread", "revised": 1, "flagged": 1, "note": "ok"})
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(2) for r in rows] == ["Agents Proofread", "Agent Tool revise_text", "Agents Proofread"]
    assert rows[0].text(3) == "Agents Proofread Start pages=1 regions=7"
    assert rows[2].text(3) == "Agents Proofread Detail flagged=1 revised=1" and rows[2].text(4) == "ok"
    assert page.status_label.text().startswith("Agents Finished")


def test_font_events(app):
    from PyQt6.QtCore import Qt
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "fonts", "page": "001", "agent": "fonts", "changed": 2, "unfixed": 1, "changes": [
            {"region": 5, "from": "A", "to": "Noto Sans TC::Bold", "reason": "missing", "missing": "值"},
            {"region": 2, "from": "B", "to": "A", "reason": "layout"}]},
    ])
    row = page.timeline.topLevelItem(0)
    assert row.text(2) == "Agents Fonts" and row.text(3) == "Agents Fonts Detail changed=2 unfixed=1"
    assert row.text(4) == ("#5 → Noto Sans TC::Bold (Agents Fonts Missing chars=值); "
                           "#2 → A (Agents Fonts Layout)")
    assert row.data(0, Qt.ItemDataRole.UserRole + 1) is True

    # this version: only the missing characters come from another font; the styles are kept for the work
    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "fonts", "page": "001", "agent": "fonts", "changed": 1, "borrowed": 1, "unfixed": 0, "changes": [
            {"region": 2, "from": "B", "to": "A", "reason": "layout"},
            {"region": 4, "font": "A", "fallback": "Yu Mincho::Light", "reason": "fallback", "missing": "值︒"}]},
        {"type": "font_styles", "agent": "memory", "file": "作品.fonts.json",
         "styles": {"旁白": "Noto Serif TC", "手寫": "AR TianniuB5"}},
    ])
    row = page.timeline.topLevelItem(0)
    assert row.text(3) == "Agents Fonts Detail Borrowed borrowed=1 changed=1 unfixed=0"
    assert row.text(4) == "#2 → A (Agents Fonts Layout); #4 Agents Fonts Fallback chars=值︒ font=Yu Mincho Light"
    assert row.data(0, Qt.ItemDataRole.UserRole + 1) is not True
    styles = page.timeline.topLevelItem(1)
    assert styles.text(2) == "Agents Font Styles"
    assert styles.text(3) == "Agents Font Styles Detail count=2 file=作品.fonts.json"
    assert styles.text(4) == "旁白 → Noto Serif TC; 手寫 → AR TianniuB5"
