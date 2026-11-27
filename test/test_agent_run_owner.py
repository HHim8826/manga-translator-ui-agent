"""Review 2026-11-28 B18: the journal of a killed run whose owner was on another computer (the folder was
moved or synced), or whose process id the system gave to another process, locked the chapter for good
although the run's lock was free; and one JSON file cut short stopped the run."""
import json
import logging
import os

import pytest

from manga_translator.agent import resume
from manga_translator.agent.resume import previous_states
from manga_translator.agent.store import AgentMemory, _read_json

from test_agent_core import _finish_page, make_page
from test_agent_resume import _agent_dir, _json, _run


def _killed(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    _run(tmp_path, paths, _finish_page() * 2)
    before = _json(folder, "002").read_bytes()
    guard = resume.RunGuard.begin(paths)
    for stem in ("001", "002"):
        make_page(tmp_path, stem)
    guard.page_finished(paths[0])
    resume._ACTIVE.discard(guard.owner["token"])  # killed: the system let go of its lock
    guard.release()
    return folder, paths, before


def _owner(folder, **changes):
    path = _agent_dir(folder) / "run_journal.json"
    journal = json.loads(path.read_text(encoding="utf-8"))
    journal["owner"].update(changes)
    path.write_text(json.dumps(journal), encoding="utf-8")
    assert resume.owner_alive(journal["owner"])  # what it looks like from its owner alone


@pytest.mark.parametrize("changes", [{"host": "another-computer"},
                                     {"pid": os.getppid(), "process_started": None}],  # an id given again
                         ids=["moved", "pid-reused"])
def test_a_killed_runs_journal_does_not_lock_the_chapter_when_its_lock_is_free(tmp_path, changes):
    folder, paths, before = _killed(tmp_path)
    _owner(folder, **changes)
    assert not any(s.busy for s in previous_states(paths))
    resume.RunGuard.begin(paths).finish()  # starts, and puts the killed run's page back first
    assert _json(folder, "002").read_bytes() == before
    assert not (_agent_dir(folder) / "run_journal.json").exists()


def test_a_run_that_holds_the_lock_still_keeps_the_chapter(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png")]
    _run(tmp_path, paths, _finish_page())
    guard = resume.RunGuard.begin(paths)
    try:
        _owner(folder, host="another-computer")
        assert all(s.busy for s in previous_states(paths))
        assert resume.recover(paths) == []
        with pytest.raises(resume.ChapterBusy):
            resume.RunGuard.begin(paths)
        assert (_agent_dir(folder) / "run_journal.json").exists()
    finally:
        guard.finish()


def test_a_file_cut_short_is_read_as_empty_and_kept(tmp_path, caplog):
    path = tmp_path / "glossary.json"
    path.write_text('[{"source": "ドダド', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="manga_translator.agent"):
        assert _read_json(str(path), []) == []
    assert "cannot read" in caplog.text and (tmp_path / "glossary.json.broken").read_text(encoding="utf-8")
    assert AgentMemory(str(tmp_path)).glossary == []
    path.write_bytes(b"\xff\xfe\x00")  # not UTF-8
    assert _read_json(str(path), {"x": 1}) == {"x": 1}
    assert _read_json(str(tmp_path / "none.json"), None) is None


def test_a_broken_journal_and_sidecar_do_not_stop_the_next_run(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png")]
    _run(tmp_path, paths, _finish_page())
    (_agent_dir(folder) / "run_journal.json").write_text('{"pages": {"001": {', encoding="utf-8")
    side = next(p for p in (_agent_dir(folder)).rglob("*.json") if "001" in p.name and "journal" not in p.name)
    side.write_text("{", encoding="utf-8")
    assert [s.path for s in previous_states(paths)] == paths
    resume.RunGuard.begin(paths).finish()
