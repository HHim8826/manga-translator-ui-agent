"""A rerun of a chapter the AI Agent already did: what is known about each page, the backup of
edited JSON, and the question before the run."""
import asyncio
import json
import os

import pytest

from manga_translator.agent import resume
from manga_translator.agent.resume import PageState, previous_states

from test_agent_core import FakeTranslator, ScriptedClient, _calls, _finish_page, make_page, run_agent_workflow
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


class Cfg:
    class translator:
        target_lang = "CHT"


def _run(tmp_path, paths, script, translator=None, client=None):
    return asyncio.run(run_agent_workflow(translator or FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                          client=client or ScriptedClient(script), log=lambda m: None,
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
                                "Agent Rerun Edited count=1 pages=002\n\nAgent Rerun Revise Note\n\n"
                                "Agent Rerun Stop Note")
    assert [os.path.basename(p) for p in question["unfinished"]] == ["003.png", "004.png"]
    # a check goes over the finished pages only: 003 failed, 004 was never run
    assert [os.path.basename(p) for p in question["revise"]] == ["001.png", "002.png"]
    assert rerun_question(_t, [PageState("a/001.png", "")]) is None


@pytest.mark.parametrize("button,expected", [
    ("Agent Rerun Unfinished count=2", (["003.png", "004.png"], False)),
    ("Agent Rerun All count=4", (["001.png", "002.png", "003.png", "004.png"], False)),
    ("Agent Rerun Revise count=2", (["001.png", "002.png"], True)),
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
    assert (None if chosen is None else ([os.path.basename(p) for p in chosen[0]], chosen[1])) == expected
    assert set(seen[0]) == {"Agent Rerun Unfinished count=2", "Agent Rerun All count=4", "Agent Rerun Revise count=2",
                            "Cancel"}


def test_no_question_for_pages_the_agent_never_saw(tmp_path, monkeypatch, app):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    from ui.main_page.pages.agent_rerun import ask_rerun

    monkeypatch.setattr(QMessageBox, "exec", lambda box: pytest.fail("asked"))
    folder = tmp_path / "chapter"
    folder.mkdir()
    files = [str(folder / "001.png")]
    assert ask_rerun(None, _t, files) == (files, False)
    # every page done: no "only the pages not done" button
    states = [PageState(files[0], "done")]
    monkeypatch.setattr(QMessageBox, "exec", lambda box: [b for b in box.buttons() if "All" in b.text()][0].click())
    assert ask_rerun(None, _t, files, states) == (files, False)


def _translations(folder, stem):
    raw = json.loads(_json(folder, stem).read_text(encoding="utf-8"))
    return [r.get("translation") for r in next(iter(raw.values()))["regions"]]


def _agent_dir(folder):
    return folder / "manga_translator_work" / "agent"


def test_a_first_run_asks_about_translations_made_elsewhere_and_keeps_a_copy(tmp_path):
    from ui.main_page.pages.agent_rerun import rerun_question

    folder = tmp_path / "chapter"
    path, json_path = make_page(tmp_path, "001")
    paths = [path, str(folder / "002.png")]  # 002 has no JSON yet
    _edit(_json(folder, "001"))  # translated outside the AI Agent: no sidecar
    states = previous_states(paths)
    assert [(s.status, s.translated) for s in states] == [("", True), ("", False)]
    question = rerun_question(_t, states)
    assert question["text"] == "Agent Rerun Translated count=1 pages=001\n\nAgent Rerun Stop Note"
    assert [os.path.basename(p) for p in question["unfinished"]] == ["002.png"]  # 001 counts as done
    mine = _json(folder, "001").read_bytes()
    _run(tmp_path, paths, _finish_page() + _finish_page())  # the user picked "redo all"
    backups = _agent_dir(folder) / "backups"
    assert [(backups / name).read_bytes() for name in os.listdir(backups)] == [mine]
    assert not (_agent_dir(folder) / "run_journal.json").exists()


def _set_texts(word):
    calls = [("set_text", {"id": i, "source_text": f"text{i}", "translation": f"{word}{i}", "category": "dialogue",
                           "confidence": 0.9, "reason": "讀圖"}) for i in (1, 2, 3)]
    return [_calls(*calls), _calls(("mark_page_done", {"confidence": 0.9}))]


def test_a_stopped_rerun_gives_the_unfinished_pages_their_translations_back(tmp_path):
    from ui.main_page.pages.agent_rerun import rerun_question

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png"), str(folder / "003.png")]
    _run(tmp_path, paths, _finish_page() * 3)
    before = {stem: _json(folder, stem).read_bytes() for stem in ("001", "002", "003")}
    # redo all: 001 is redone, the stop comes at 002's first model turn, 003 is never reached
    with pytest.raises(asyncio.CancelledError):
        _run(tmp_path, paths, _set_texts("新") + [asyncio.CancelledError()])
    assert _translations(folder, "001") == ["新1", "新2", "新3"]
    assert _json(folder, "002").read_bytes() == before["002"]  # stopped in the middle: as before the run
    assert _json(folder, "003").read_bytes() == before["003"]
    states = previous_states(paths)
    assert [(s.stem, s.status, s.edited) for s in states] == [("001", "done", False), ("002", "done", False),
                                                              ("003", "done", False)]
    assert rerun_question(_t, states)["unfinished"] == []
    assert not (_agent_dir(folder) / "run_journal.json").exists()
    assert not os.listdir(_agent_dir(folder) / "backups")  # the agent's own copies are not kept


class _FailingPreprocess(FakeTranslator):
    """The preprocess fails on 002: no JSON written, the translator's context says why."""

    async def translate_batch(self, images_with_configs, save_info=None):
        if not self.template:
            return await super().translate_batch(images_with_configs, save_info)
        from types import SimpleNamespace

        ok = [(p, c) for p, c in images_with_configs if not p.endswith("002.png")]
        failed = [SimpleNamespace(image_name=p, success=False, translation_error="detection failed")
                  for p, _ in images_with_configs if p.endswith("002.png")]
        return await super().translate_batch(ok, save_info) + failed


def test_a_page_the_preprocess_failed_on_is_left_alone(tmp_path):
    from manga_translator.agent.events import read_events

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    _run(tmp_path, paths, _finish_page() * 2)
    before = _json(folder, "002").read_bytes()
    client = ScriptedClient(_finish_page())  # only 001 gets model calls
    results = _run(tmp_path, paths, None, translator=_FailingPreprocess(tmp_path), client=client)
    assert len(client.requests) == 2
    assert _json(folder, "002").read_bytes() == before
    assert previous_states(paths)[1].status == "done"
    failed = results[-1]
    assert failed.success is False and "preprocess failed: detection failed" in failed.translation_error
    runs = _agent_dir(folder) / "runs"
    events = read_events(str(runs / sorted(os.listdir(runs))[-1]))
    assert any(e["type"] == "error" and e.get("page") == "002" and e.get("stage") == "preprocess" for e in events)


def test_a_run_that_was_killed_is_put_back_before_the_next_one(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    _run(tmp_path, paths, _finish_page() * 2)
    before = {stem: _json(folder, stem).read_bytes() for stem in ("001", "002")}
    guard = resume.RunGuard.begin(paths)
    for stem in ("001", "002"):  # the preprocess wrote both again
        make_page(tmp_path, stem)
    guard.page_finished(paths[0])
    assert (_agent_dir(folder) / "run_journal.json").exists()
    # the app was killed here: no guard.finish()
    states = previous_states(paths)
    assert _json(folder, "002").read_bytes() == before["002"]
    assert _json(folder, "001").read_bytes() != before["001"]  # the run finished 001
    assert [s.status for s in states] == ["done", "done"]
    assert not (_agent_dir(folder) / "run_journal.json").exists()
