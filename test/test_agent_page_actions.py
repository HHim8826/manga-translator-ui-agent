"""AI Agent page: stop, open in the editor, the output folder, old run logs while a run goes on,
the work memory rows, and zooming the preview."""
import json
import os
import types

import cv2
import numpy as np

from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


def _page():
    from ui.main_page.pages.agents_page import AgentsPage

    return AgentsPage(_t)


def _start(run="r1", batch="b1", pages=("001.jpg", "002.jpg"), **extra):
    return {"type": "run_start", "run": run, "batch": batch, "chapter": "第35話", "pages": list(pages), "model": "m",
            **extra}


def _texts(page):
    return [page.page_list.item(i).text() for i in range(page.page_list.count())]


def test_the_stop_button_follows_the_task(app):  # noqa: F811
    page = _page()
    asked = []
    page.stop_requested.connect(lambda: asked.append(True))
    assert not page.stop_button.isEnabled()
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 2})  # an AI Agent run
    assert page.stop_button.isEnabled() and page.stop_button.text() == "Agents Stop"
    page.stop_button.click()
    assert asked == [True] and not page.stop_button.isEnabled() and page.stop_button.text() == "Stopping..."
    page.set_task_running(False)
    assert not page.stop_button.isEnabled() and page.stop_button.text() == "Agents Stop"


def test_the_stop_button_stays_while_a_check_makes_its_backups(app):  # noqa: F811
    """Review item 15: start, API check, then a check's backups (``revise``): the status says so and the
    stop button works until the first chapter starts."""
    page = _page()
    asked = []
    page.stop_requested.connect(lambda: asked.append(True))
    page.set_task_running(True)
    enabled = []
    for event in ({"stage": "start", "pages": 2}, {"stage": "api_check", "model": "m"},
                  {"stage": "revise", "pages": 2}):
        page._handle_live({"type": "workflow_stage", **event})
        enabled.append(page.stop_button.isEnabled())
    assert enabled == [True, True, True]
    assert page.status_label.text() == "Agents Stage Revise pages=2"
    page.stop_button.click()
    assert asked == [True]
    page.set_task_running(False)
    page._handle_live({"type": "workflow_stage", "stage": "restore_failed", "pages": ["002.png"]})
    assert page.status_label.text() == "Agents Stage Restore Failed count=1 pages=002.png"


def test_open_the_page_in_the_editor_and_the_output_folder(app, tmp_path):  # noqa: F811
    page = _page()
    opened = []
    page.open_in_editor_requested.connect(opened.append)
    assert not page.editor_button.isEnabled() and not page.output_button.isEnabled()
    page.replay([_start(),
                 {"type": "page_start", "run": "r1", "page": "001", "index": 0, "total": 2, "image": "c/001.jpg"},
                 {"type": "page_start", "run": "r1", "page": "002", "index": 1, "total": 2, "image": "c/002.jpg"}])
    assert page.editor_button.isEnabled() and not page.output_button.isEnabled()
    page.editor_button.click()  # the preview follows 002
    page.page_list.setCurrentRow(1)  # the user picks 001
    page.editor_button.click()
    assert opened == ["c/002.jpg", "c/001.jpg"]
    output = tmp_path / "out" / "001.png"
    page.handle_event({"type": "render_done", "run": "r1", "page": "001", "agent": "render", "output": str(output)})
    assert page.output_button.isEnabled() and page._output_folder() == str(tmp_path / "out")


def _log(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    return str(path)


def test_an_old_run_log_does_not_mix_with_the_live_run(app, tmp_path):  # noqa: F811
    page = _page()
    page._handle_live(_start(pages=("001.jpg",)))
    page._handle_live({"type": "page_start", "run": "r1", "page": "001", "index": 0, "total": 1})
    old = _log(tmp_path / "old.jsonl", [_start(run="r0", batch="b0", pages=("009.jpg",)),
                                        {"type": "page_done", "run": "r0", "page": "009", "status": "done"}])
    page.open_run_log(old)
    assert _texts(page)[1:] == ["009 · Agent Page Done"]
    assert page.subtitle_label.text() == "Agents Viewing Log file=old.jsonl" and not page.live_button.isHidden()
    page._handle_live({"type": "page_done", "run": "r1", "page": "001", "status": "done", "flags": 2})
    assert _texts(page)[1:] == ["009 · Agent Page Done"]  # kept, not shown
    page.live_button.click()
    assert _texts(page)[1:] == ["001 · Agent Page Done · Agents Flag Count count=2"]
    assert page.subtitle_label.text() == "Agents Page Subtitle" and page.live_button.isHidden()

    # the dialog opens in the folder of the last log; a new run takes the page back from an old log
    assert page._log_folder() == str(tmp_path)
    page.open_run_log(old)
    page._handle_live(_start(run="r2", batch="b2", pages=("003.jpg",)))
    assert _texts(page)[1:] == ["003 · Agent Page Waiting"] and page.subtitle_label.text() == "Agents Page Subtitle"


def test_the_run_log_dialog_starts_in_the_chapter_runs(app, tmp_path):  # noqa: F811
    runs = tmp_path / "第35話" / "manga_translator_work" / "agent" / "runs"
    runs.mkdir(parents=True)
    page = _page()
    page.replay([_start(), {"type": "page_start", "run": "r1", "page": "001", "index": 0, "total": 2,
                            "image": str(tmp_path / "第35話" / "001.jpg")}])
    assert page._log_folder() == str(runs)


def test_the_work_memory_rows(app):  # noqa: F811
    page = _page()
    page.replay([
        _start(work_memory={"glossary": 12, "characters": 3, "story": 240, "from": ["第33話", "第34話"]}),
        {"type": "work_memory", "run": "r1", "agent": "memory", "action": "saved", "folder": "dict/agent/x.memory",
         "glossary_added": 2, "glossary_updated": 1, "characters_added": 0, "story": 250},
    ])
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [(r.text(2), r.text(3), r.text(4)) for r in rows] == [
        ("Agents Work Memory", "Agents Work Memory Seeded characters=3 glossary=12 story=240",
         "第33話Agent Result Separator第34話"),
        ("Agents Work Memory", "Agents Work Memory Saved added=2 characters=0 updated=1", ""),
    ]
    assert rows[1].toolTip(4) == "dict/agent/x.memory"


def _tools(n, start=0, ok=True):
    events = []
    for i in range(start, start + n):
        events += [{"type": "tool_call", "run": "r1", "page": "001", "tool": "set_text", "args": {"id": 1},
                    "reason": f"r{i}"},
                   {"type": "tool_result", "run": "r1", "page": "001", "tool": "set_text", "ok": ok,
                    "summary": f"s{i}"}]
    return events


def test_a_long_replay_scrolls_once_and_ends_where_row_by_row_would(app):  # noqa: F811
    """Review perf item: a replay (an old log, back to the current run) built and followed each row;
    1,000 rows took about 0.75 s. Rows are now added without following each one, with the same end:
    at the bottom after a new run or when it was there, where the user scrolled to otherwise."""
    page = _page()
    page.resize(800, 600)
    page.show()
    bar = page.timeline.verticalScrollBar()
    page.replay([_start(pages=("001.jpg",)), *_tools(300)])
    app.processEvents()
    assert page.timeline.topLevelItemCount() == 300 and 0 < bar.maximum() == bar.value()

    bar.setValue(0)  # the user scrolls up to read
    page.replay(_tools(50, start=300, ok=False))
    app.processEvents()
    assert page.timeline.topLevelItemCount() == 350 and bar.value() == 0
    assert page.timeline.topLevelItem(349).text(4) == "r349"

    bar.setValue(bar.maximum())
    page.handle_event(_tools(1, start=350)[0])  # a live row follows again
    app.processEvents()
    assert bar.value() == bar.maximum() and page.timeline.topLevelItemCount() == 351

    page.problems_only.setChecked(True)  # a replay keeps the filter
    page.replay([_start(run="r2", batch="b2", pages=("001.jpg",)), *_tools(5), *_tools(2, start=5, ok=False)])
    app.processEvents()
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(4) for r in rows if not r.isHidden()] == ["r5", "r6"]
    page.hide()


def _canvas(tmp_path):
    from PyQt6.QtCore import QPointF
    from ui.main_page.pages.agent_preview import PreviewCanvas, ImageCache

    path = str(tmp_path / "page.png")
    cv2.imwrite(path, np.full((300, 200, 3), 255, np.uint8))
    pixmap, size = ImageCache().get(path)
    canvas = PreviewCanvas()
    canvas.resize(200, 300)
    canvas.set_content(pixmap, size)
    return canvas, QPointF


def _image_point(canvas, pos):
    rect = canvas.image_rect()
    return ((pos.x() - rect.left()) / rect.width(), (pos.y() - rect.top()) / rect.height())


def test_the_preview_zooms_at_the_cursor_and_moves(app, tmp_path):  # noqa: F811
    from ui.main_page.pages.agent_preview import MAX_ZOOM

    canvas, QPointF = _canvas(tmp_path)
    pos = QPointF(50, 60)
    before = _image_point(canvas, pos)
    canvas.zoom_at(pos, 2)
    assert abs(canvas.zoom - 1.5625) < 1e-9
    after = _image_point(canvas, pos)
    assert abs(after[0] - before[0]) < 1e-6 and abs(after[1] - before[1]) < 1e-6  # the spot stays under the cursor
    rect = canvas.image_rect()
    canvas.pan(-30, -20)
    moved = canvas.image_rect()
    assert (moved.left(), moved.top()) == (rect.left() - 30, rect.top() - 20)
    canvas.pan(10_000, 10_000)  # never past the image's edge
    assert (canvas.image_rect().left(), canvas.image_rect().top()) == (0, 0)
    canvas.zoom_at(pos, 100)
    assert canvas.zoom == MAX_ZOOM
    canvas.zoom_at(pos, -100)
    assert canvas.zoom == 1.0 and canvas.image_rect().width() == 200
    canvas.zoom_at(pos, 3)
    canvas.reset_zoom()
    assert canvas.zoom == 1.0 and (canvas.center.x(), canvas.center.y()) == (0.5, 0.5)


def test_another_page_resets_the_zoom_a_stage_keeps_it(app, tmp_path):  # noqa: F811
    from PyQt6.QtCore import QPointF
    from ui.main_page.pages.agent_preview import PagePreview

    for stem in ("001", "002"):
        cv2.imwrite(str(tmp_path / f"{stem}.png"), np.full((300, 200, 3), 255, np.uint8))
        cv2.imwrite(str(tmp_path / f"{stem}_in.png"), np.full((300, 200, 3), 128, np.uint8))
    preview = PagePreview(_t)
    preview.resize(300, 500)
    for stem in ("001", "002"):
        preview.update_page(stem, image=str(tmp_path / f"{stem}.png"), inpainted=str(tmp_path / f"{stem}_in.png"))
    preview.show_page("001")
    preview.canvas.zoom_at(QPointF(40, 40), 2)
    preview._on_stage_clicked("original")
    assert preview.canvas.zoom > 1
    preview.show_page("002")
    assert preview.canvas.zoom == 1.0


def test_an_agent_run_shows_its_page(app):  # noqa: F811
    """Review 2026-11-24 X1: the page was shown when the start button was pressed, before the
    question whether to redo pages (a cancelled start was left on the AI Agent page, showing the last
    run). It is shown when the run starts: its ``workflow_stage`` ``start`` event."""
    from ui.main_window import MainWindow

    switched = []

    def window(show):
        config = types.SimpleNamespace(agent=types.SimpleNamespace(show_page_on_start=show))
        return types.SimpleNamespace(config_service=types.SimpleNamespace(get_config=lambda: config),
                                     _switch_main_page=switched.append, logger=None)

    MainWindow._show_agents_page_on_start(window(True))
    MainWindow._show_agents_page_on_start(window(False))  # the option is off
    assert switched == ["agents"]

    page = _page()
    started = []
    page.run_started.connect(lambda: started.append(True))
    page._handle_live({"type": "workflow_stage", "stage": "api_check", "model": "m"})
    assert started == []
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 3})
    assert started == [True]


def test_flag_rows_number_regions_the_way_each_agent_does(app):  # noqa: F811
    page = _page()
    page.replay([
        _start(pages=("001.jpg",)),
        {"type": "flag", "run": "r1", "page": "001", "region": 4, "flag": "uncertain_ocr", "reason": "a"},
        {"type": "flag", "run": "r1", "page": "001", "agent": "proofread", "region": 4,
         "flag": "uncertain_translation", "reason": "b"},
        {"type": "flag", "run": "r1", "page": "001", "agent": "fonts", "region": 2, "flag": "layout", "reason": "c"},
        {"type": "flag", "run": "r1", "page": "001", "agent": "inpaint", "region": 3, "flag": "inpaint", "reason": "d"},
    ])
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(3) for r in rows if r.text(2) == "Agents Flag"] == [
        "Agent Result With Kind id=#4 kind=Agent Flag uncertain_ocr",  # the translator's own ids: the preview's boxes
        "Agent Result With Kind id=Agent Result Editor Box n=4 kind=Agent Flag uncertain_translation",
        "Agent Result With Kind id=Agent Result Editor Box n=2 kind=Agent Flag layout",
        "Agent Result With Kind id=Agent Review Area id=3 kind=Agent Flag inpaint",
    ]


def test_a_stop_from_the_translation_page_also_shows_on_the_agents_page(app, monkeypatch):  # noqa: F811
    """Review 2026-11-28 X4: the AI Agent page's stop button stayed on after a stop from the translation
    page, and the translation page's button said "Stop Translation" behind the rerun question, before
    the translation started."""
    from ui.main_page import runtime

    page = _page()
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "start", "pages": 2})
    assert page.stop_button.isEnabled()
    button = types.SimpleNamespace(state=[], setEnabled=lambda on: button.state.append(on),
                                   setText=lambda text: button.state.append(text),
                                   setProperty=lambda *a: None, style=lambda: None,
                                   clicked=types.SimpleNamespace(disconnect=lambda: None, connect=lambda f: None))
    monkeypatch.setattr(runtime, "_set_start_button_state", lambda view, state: None)
    view = types.SimpleNamespace(start_button=button, agents_page=page, _t=lambda key: key)
    runtime.set_stopping_state(view)
    assert not page.stop_button.isEnabled() and page.stop_button.text() == "Stopping..."
    assert button.state[:2] == [False, "Stopping..."]

    later = []
    monkeypatch.setattr(runtime.QTimer, "singleShot", lambda ms, callback: later.append(ms))
    logic = types.SimpleNamespace(state_manager=types.SimpleNamespace(is_translating=lambda: True),
                                  current_worker=None, stop_task=lambda: None)
    button.state.clear()
    view = types.SimpleNamespace(start_button=button, controller=logic, _t=lambda key: key,
                                 _enable_stop_button=lambda: None)
    runtime.enable_stop_button(view)  # the scan is over, the rerun question is open: nothing to stop yet
    assert button.state == [] and later == [runtime.STOP_RETRY_MS]
    logic.current_worker = object()  # the translation started
    runtime.enable_stop_button(view)
    assert button.state[:2] == [True, "Stop Translation"]
