"""A run that the user stops, a run over several chapters, and the AI Agent page while it fills:
the stopped page is saved, every chapter keeps its own pages, and the lists keep their scroll."""
import asyncio
import os
from pathlib import Path

import pytest

from manga_translator.agent.events import read_events

from test_agent_core import FakeTranslator, ScriptedClient, _finish_page, _runner, make_page, run_agent_workflow
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


class Cfg:
    class translator:
        target_lang = "CHT"


class Chapters(FakeTranslator):
    """The preprocess writes each page in its own chapter folder (``<base>/chapter/001.png``)."""

    async def translate_batch(self, images_with_configs, save_info=None):
        if self.template:
            for path, _ in images_with_configs:
                make_page(Path(path).parent.parent, os.path.splitext(os.path.basename(path))[0])
        return [f"ctx:{os.path.basename(p)}" for p, _ in images_with_configs]


def _workflow(tmp_path, paths, script):
    return run_agent_workflow(Chapters(tmp_path), [(p, Cfg) for p in paths], {}, client=ScriptedClient(script),
                              log=lambda m: None, agent_settings={"inpaint_review": False, "recall_pass": False})


def _log(folder):
    runs = Path(folder) / "manga_translator_work" / "agent" / "runs"
    return read_events(str(runs / os.listdir(runs)[0]))


# --- runner and workflow -------------------------------------------------------------
def test_a_stopped_page_is_saved_and_the_run_end_says_why(tmp_path):
    pages = [make_page(tmp_path, "001"), make_page(tmp_path, "002"), make_page(tmp_path, "003")]
    # page 002 is cancelled after its first step, like the stop button does
    script = _finish_page() + [_finish_page()[0], asyncio.CancelledError()]
    runner, _client, store, events = _runner(tmp_path, pages, script)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run())
    events.close()
    assert store.load_sidecar("001")["status"] == "done"
    assert store.load_sidecar("002")["status"] == "stopped"
    assert store.load_sidecar("003") is None
    end = [e for e in read_events(events.path) if e["type"] == "run_end"]
    assert len(end) == 1 and end[0]["reason"] == "stopped" and end[0]["done"] == 1 and end[0]["total"] == 3


def test_three_failed_pages_end_the_run_as_failed(tmp_path):
    pages = [make_page(tmp_path, f"00{i}") for i in range(1, 5)]
    runner, _client, _store, events = _runner(tmp_path, pages, [RuntimeError("refused")] * 3)
    outcomes = asyncio.run(runner.run())
    events.close()
    assert [o["status"] for o in outcomes] == ["error"] * 3
    end = [e for e in read_events(events.path) if e["type"] == "run_end"][0]
    assert end["reason"] == "failed" and end["total"] == 4


def test_the_workflow_logs_the_stop(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_workflow(tmp_path, paths, _finish_page() + [asyncio.CancelledError()]))
    types = [e["type"] for e in _log(folder)]
    assert types[-2:] == ["run_end", "stopped"]
    assert [e for e in _log(folder) if e["type"] == "stopped"][0]["stage"] == "agent"


def test_the_chapters_of_one_run_share_a_batch(tmp_path):
    first, second = tmp_path / "a" / "chapter", tmp_path / "b" / "chapter"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    paths = [str(first / "001.png"), str(second / "001.png")]
    asyncio.run(_workflow(tmp_path, paths, _finish_page() + _finish_page()))
    starts = [[e for e in _log(folder) if e["type"] == "run_start"][0] for folder in (first, second)]
    assert starts[0]["batch"] == starts[1]["batch"] and starts[0]["batch"].startswith("b")
    assert starts[0]["run"] != starts[1]["run"]  # also when both chapters start in the same second
    assert [s["chapter"] for s in starts] == ["chapter", "chapter"]


# --- the AI Agent page ----------------------------------------------------------------
def _chapter(run, chapter, batch="b1", pages=("001.jpg", "002.jpg")):
    return [{"type": "run_start", "run": run, "batch": batch, "chapter": chapter, "pages": list(pages), "model": "m"}]


def _texts(page):
    return [page.page_list.item(i).text() for i in range(page.page_list.count())]


def test_two_chapters_keep_their_own_pages(app):  # noqa: F811
    from PyQt6.QtCore import Qt
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay(_chapter("r1", "第33話") + [
        {"type": "page_start", "run": "r1", "page": "001", "index": 0, "total": 2, "image": "a/001.jpg"},
        {"type": "flag", "run": "r1", "page": "001", "region": 2, "flag": "uncertain_ocr"},
        {"type": "page_done", "run": "r1", "page": "001", "status": "done", "flags": 1},
        {"type": "page_done", "run": "r1", "page": "002", "status": "done", "flags": 0},
        {"type": "run_end", "run": "r1", "pages": 2, "total": 2, "done": 2, "totals": {"requests": 5}},
    ] + _chapter("r2", "第34話") + [
        {"type": "page_start", "run": "r2", "page": "001", "index": 0, "total": 2, "image": "b/001.jpg"},
        {"type": "page_done", "run": "r2", "page": "001", "status": "incomplete", "flags": 0},
    ])
    assert _texts(page) == ["Agents All Pages", "第33話", "001 · Agent Page Done · Agents Flag Count count=1",
                            "002 · Agent Page Done", "第34話", "001 · Agent Page Incomplete", "002 · Agent Page Waiting"]
    assert page.preview.pages["001"]["image"] == "a/001.jpg"
    assert page.preview.pages["r2|001"]["image"] == "b/001.jpg"
    assert page.preview.title_label.text() == "Agents Preview · 第34話 · 001"
    assert page.status_label.text() == "Agents Page Progress current=1 page=第34話 · 001 total=2"
    assert not page.page_list.item(1).flags() & Qt.ItemFlag.ItemIsSelectable  # the headings are not pages
    page.page_list.setCurrentRow(5)  # 第34話 001: only its rows
    assert page.preview.shown == "r2|001"
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.isHidden() for r in rows] == [True]  # the flag row belongs to 第33話's 001
    page.handle_event({"type": "run_end", "run": "r2", "pages": 2, "total": 2, "done": 0, "totals": {"requests": 3}})
    assert page.status_label.text() == "Agents Finished done=2 flagged=1 pages=4 requests=8"
    # a new run (another batch) starts over
    page.replay(_chapter("r3", "第35話", batch="b2", pages=("001.jpg",)))
    assert _texts(page) == ["Agents All Pages", "001 · Agent Page Waiting"]


def test_a_run_log_opened_on_its_own_starts_over(app):  # noqa: F811
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay(_chapter("r1", "第33話"))
    page.replay(_chapter("r1", "第33話"))  # the same chapter again: not a second chapter
    assert _texts(page) == ["Agents All Pages", "001 · Agent Page Waiting", "002 · Agent Page Waiting"]


def test_a_stopped_run_shows_where_it_stopped(app):  # noqa: F811
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay(_chapter("r1", "第33話", pages=("001.jpg", "002.jpg", "003.jpg")) + [
        {"type": "page_done", "run": "r1", "page": "001", "status": "done", "flags": 0},
        {"type": "inpaint_start", "run": "r1", "page": "001", "agent": "inpaint", "units": 2, "issues": 0},
        {"type": "page_start", "run": "r1", "page": "002", "index": 1, "total": 3},
        {"type": "run_end", "run": "r1", "pages": 1, "total": 3, "done": 1, "reason": "stopped",
         "totals": {"requests": 4}},
        {"type": "stopped", "run": "r1", "agent": "workflow", "stage": "agent"},
    ])
    assert _texts(page)[1:] == ["001 · Agent Page Stopped", "002 · Agent Page Stopped", "003 · Agent Page Not Reached"]
    assert page.status_label.text() == "Agents Stopped Status done=1 flagged=0 pages=3 requests=4"


def test_a_failed_run_says_so(app):  # noqa: F811
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay(_chapter("r1", "第33話") + [
        {"type": "run_end", "run": "r1", "pages": 2, "total": 2, "done": 0, "reason": "failed", "totals": {}},
    ])
    assert page.status_label.text() == "Agents Finished done=0 flagged=0 pages=2 requests=0Agents Finished Failed"


def test_the_lists_keep_their_scroll_position(app):  # noqa: F811
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.resize(1200, 300)
    page.show()
    pages = [f"{i:03}.jpg" for i in range(1, 61)]
    page.replay(_chapter("r1", "第33話", pages=pages))
    for i in range(80):
        page.handle_event({"type": "tool_call", "run": "r1", "page": "001", "tool": "zoom", "args": {"id": 1}})
    app.processEvents()
    pages_bar, rows_bar = page.page_list.verticalScrollBar(), page.timeline.verticalScrollBar()
    assert pages_bar.maximum() > 10 and rows_bar.maximum() > 10
    assert rows_bar.maximum() - rows_bar.value() <= page.timeline.sizeHintForRow(0)  # at the bottom: follows
    pages_bar.setValue(5)
    rows_bar.setValue(3)
    page.handle_event({"type": "page_start", "run": "r1", "page": "040", "index": 39, "total": 60})
    page.handle_event({"type": "flag", "run": "r1", "page": "040", "region": 1, "flag": "uncertain_ocr"})
    page.handle_event({"type": "page_done", "run": "r1", "page": "040", "status": "done", "flags": 1})
    page.handle_event({"type": "render_start", "run": "r1", "agent": "render", "pages": 59,
                       "skipped": {"041": "error"}})
    app.processEvents()
    assert pages_bar.value() == 5 and rows_bar.value() == 3
    assert _texts(page)[40] == "040 · Agent Page Done · Agents Flag Count count=1"
    assert _texts(page)[41] == "041 · Agent Page Error"
    page.replay(_chapter("r2", "第34話", pages=pages))  # the next chapter rebuilds the list
    app.processEvents()
    assert pages_bar.value() == 5 and _texts(page)[1] == "第33話"
    page.close()
