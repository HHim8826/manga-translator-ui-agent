"""Agents page: live events fill the page list and the timeline; filters work."""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "desktop_qt_ui"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from manga_translator.agent.events import EventLog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


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
    assert rows[1].text(3) == "missing #4"
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
