"""A rerun of a chapter the AI Agent already did: what is known about each page, the backup of
edited JSON, and the question before the run."""
import asyncio
import json
import os

import pytest

from manga_translator.agent import resume
from manga_translator.agent.resume import PageState, previous_states

from test_agent_core import FakeTranslator, ScriptedClient, _finish_page, run_agent_workflow
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


class Cfg:
    class translator:
        target_lang = "CHT"


def _run(tmp_path, paths, script):
    return asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                          client=ScriptedClient(script), log=lambda m: None,
                                          agent_settings={"inpaint_review": False, "recall_pass": False}))


def _json(folder, stem):
    return folder / "manga_translator_work" / "json" / f"{stem}_translations.json"


def _edit(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    next(iter(raw.values()))["regions"][0]["translation"] = "我改的"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def test_states_after_a_run_and_an_edit(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png"), str(folder / "003.png")]
    assert [s.status for s in previous_states(paths)] == ["", "", ""]  # never run
    # 001 done, 002 fails three times... only once here: the run goes on to 003, which is done
    _run(tmp_path, paths, _finish_page() + [RuntimeError("refused")] + _finish_page())
    states = previous_states(paths)
    assert [(s.stem, s.status, s.edited) for s in states] == [("001", "done", False), ("002", "error", False),
                                                              ("003", "done", False)]
    _edit(_json(folder, "003"))
    assert [s.edited for s in previous_states(paths)] == [False, False, True]


def test_runs_without_fingerprints_compare_the_time(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png")]
    _run(tmp_path, paths, _finish_page())
    os.remove(folder / "manga_translator_work" / "agent" / "saved_json.json")
    runs = folder / "manga_translator_work" / "agent" / "runs"
    logged = os.path.getmtime(runs / os.listdir(runs)[0])
    path = _json(folder, "001")
    os.utime(path, (logged + 5, logged + 5))  # the render right after the agent
    assert not previous_states(paths)[0].edited
    os.utime(path, (logged + resume.EDIT_MARGIN + 60, logged + resume.EDIT_MARGIN + 60))
    assert previous_states(paths)[0].edited


def test_a_rerun_keeps_a_copy_of_the_edited_json(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    _run(tmp_path, paths, _finish_page() + _finish_page())
    _edit(_json(folder, "002"))
    edited = _json(folder, "002").read_bytes()
    _run(tmp_path, paths, _finish_page() + _finish_page())
    backups = folder / "manga_translator_work" / "agent" / "backups"
    copies = os.listdir(backups)
    assert len(copies) == 1 and copies[0].startswith("002_translations.") and copies[0].endswith(".json")
    assert (backups / copies[0]).read_bytes() == edited
    assert not any(s.edited for s in previous_states(paths))  # the new run's JSON is recorded again


def _states(tmp_path):
    return [PageState(str(tmp_path / "001.png"), "done"), PageState(str(tmp_path / "002.png"), "done", True),
            PageState(str(tmp_path / "003.png"), "error"), PageState(str(tmp_path / "004.png"), "")]


def test_the_question_names_the_edited_pages():
    from ui.main_page.pages.agent_rerun import rerun_question

    states = _states(__import__("pathlib").Path("x"))
    question = rerun_question(_t, states)
    assert question["text"] == ("Agent Rerun Text done=2 pages=4 seen=3\n\n"
                                "Agent Rerun Edited count=1 pages=002")
    assert [os.path.basename(p) for p in question["unfinished"]] == ["003.png", "004.png"]
    assert rerun_question(_t, [PageState("a/001.png", "")]) is None


@pytest.mark.parametrize("button,expected", [
    ("Agent Rerun Unfinished count=2", ["003.png", "004.png"]),
    ("Agent Rerun All count=4", ["001.png", "002.png", "003.png", "004.png"]),
    ("Cancel", None),
])
def test_the_user_picks_the_pages(tmp_path, monkeypatch, app, button, expected):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    from ui.main_page.pages.agent_rerun import ask_rerun

    seen = []

    def click(box):
        seen.append([b.text() for b in box.buttons()])
        next(b for b in box.buttons() if b.text() == button).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", click)
    states = _states(tmp_path)
    chosen = ask_rerun(None, _t, [s.path for s in states], states)
    assert (None if chosen is None else [os.path.basename(p) for p in chosen]) == expected
    assert set(seen[0]) == {"Agent Rerun Unfinished count=2", "Agent Rerun All count=4", "Cancel"}


def test_no_question_for_pages_the_agent_never_saw(tmp_path, monkeypatch, app):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    from ui.main_page.pages.agent_rerun import ask_rerun

    monkeypatch.setattr(QMessageBox, "exec", lambda box: pytest.fail("asked"))
    folder = tmp_path / "chapter"
    folder.mkdir()
    files = [str(folder / "001.png")]
    assert ask_rerun(None, _t, files) == files
    # every page done: no "only the pages not done" button
    states = [PageState(files[0], "done")]
    monkeypatch.setattr(QMessageBox, "exec", lambda box: [b for b in box.buttons() if "All" in b.text()][0].click())
    assert ask_rerun(None, _t, files, states) == files
