"""The command line: ``local --agent-revise`` checks a finished chapter again with the AI Agent even when
``cli.agent_workflow`` is off, and ``--subprocess`` (normal translation only) refuses it before it touches a
file (review item 11). The config service, the translator and the agent run are fakes: nothing reads or
writes config.json, loads a model or calls an API."""
import _bootstrap  # noqa: F401,I001
import asyncio
import logging
import sys
import types

import cv2
import numpy as np
import pytest

from core.config_models import AppSettings


class FakeConfigService:
    agent = False
    source_dir = False

    def __init__(self, root):
        self.settings = AppSettings()  # the defaults: cli.agent_workflow is False
        self.settings.cli.agent_workflow = self.agent
        self.settings.cli.save_to_source_dir = self.source_dir

    def load_config_file(self, path):
        return False

    def get_config(self):
        return self.settings


@pytest.fixture
def cli(monkeypatch, tmp_path):
    import manga_translator
    from manga_translator.agent import workflow
    from manga_translator.mode import local, subprocess_manager

    calls = {"translate_batch": 0, "agent": [], "redo": [], "subprocess": 0, "save_info": []}

    class FakeTranslator:
        def __init__(self, params):
            pass

        async def translate_batch(self, images, save_info=None):
            calls["translate_batch"] += 1
            return []

    async def fake_agent(translator, images, save_info, **kwargs):
        calls["agent"].append(kwargs.get("revise"))
        calls["redo"].append(kwargs.get("redo"))
        calls["save_info"].append(save_info)
        return []

    async def fake_subprocess(**kwargs):
        calls["subprocess"] += 1
        return 0, 0, 0

    monkeypatch.setitem(sys.modules, "desktop_qt_ui.services.config_service",
                        types.SimpleNamespace(ConfigService=FakeConfigService))
    monkeypatch.setattr(manga_translator, "MangaTranslator", FakeTranslator)
    monkeypatch.setattr(workflow, "run_agent_workflow", fake_agent)
    monkeypatch.setattr(subprocess_manager, "translate_with_subprocess", fake_subprocess)
    monkeypatch.setattr(local, "ROOT_DIR", tmp_path)  # its log file goes here, not into the program folder

    loggers = [logging.root, logging.getLogger("manga_translator")]
    kept = [(lg, lg.handlers[:], lg.level) for lg in loggers]
    chapter = tmp_path / "第1話"
    chapter.mkdir()
    cv2.imwrite(str(chapter / "001.png"), np.full((40, 30, 3), 255, np.uint8))

    def run(*argv):
        from manga_translator.args import create_parser

        args = create_parser().parse_args(["local", *argv, "-i", str(chapter)])
        asyncio.run(local.run_local_mode(args))
        return calls

    yield run, chapter, calls
    for lg, handlers, level in kept:
        for handler in lg.handlers[:]:
            if handler not in handlers:
                lg.removeHandler(handler)
                handler.close()
        lg.setLevel(level)


def test_agent_revise_checks_again_with_the_default_config(cli, tmp_path):
    run, _chapter, _calls = cli
    calls = run("--agent-revise", "-o", str(tmp_path / "out"))
    assert calls["translate_batch"] == 0 and calls["agent"] == [True]


def test_without_the_flag_the_default_config_translates_as_before(cli, tmp_path):
    run, _chapter, _calls = cli
    calls = run("-o", str(tmp_path / "out"))
    assert calls["translate_batch"] == 1 and calls["agent"] == []


def test_subprocess_mode_refuses_agent_revise_before_touching_files(cli, capsys):
    run, chapter, calls = cli
    with pytest.raises(SystemExit) as stop:
        run("--agent-revise", "--subprocess")
    assert stop.value.code == 2
    assert calls["subprocess"] == 0 and calls["translate_batch"] == 0 and calls["agent"] == []
    assert "--agent-revise" in capsys.readouterr().out
    assert not (chapter.parent / "第1話-translated").exists()  # the default output folder was never made
    assert sorted(p.name for p in chapter.iterdir()) == ["001.png"]


def test_agent_redo_does_the_steps_left_with_the_default_config(cli, tmp_path):
    """``--agent-redo``: only the steps an earlier run could not do (``agent/redo.py``)."""
    run, _chapter, _calls = cli
    calls = run("--agent-redo", "-o", str(tmp_path / "out"))
    assert calls["translate_batch"] == 0 and calls["agent"] == [False] and calls["redo"] == [True]


def test_subprocess_mode_refuses_agent_redo_before_touching_files(cli, capsys):
    run, chapter, calls = cli
    with pytest.raises(SystemExit) as stop:
        run("--agent-redo", "--subprocess")
    assert stop.value.code == 2 and calls["subprocess"] == 0 and calls["agent"] == []
    assert "--agent-redo" in capsys.readouterr().out
    assert sorted(p.name for p in chapter.iterdir()) == ["001.png"]


def test_subprocess_mode_refuses_a_config_with_the_agent_workflow(cli, capsys, monkeypatch):
    """Review 2026-11-28 B12: with cli.agent_workflow on, --subprocess ran a normal translation."""
    run, chapter, calls = cli
    monkeypatch.setattr(FakeConfigService, "agent", True)
    with pytest.raises(SystemExit) as stop:
        run("--subprocess")
    assert stop.value.code == 2 and calls["subprocess"] == 0 and calls["agent"] == []
    assert "cli.agent_workflow" in capsys.readouterr().out
    assert sorted(p.name for p in chapter.iterdir()) == ["001.png"]


def test_the_agent_run_gets_the_save_to_source_dir_setting(cli, tmp_path, monkeypatch):
    """Review 2026-11-28 B12: save_info had no save_to_source_dir, so a CLI run wrote elsewhere and a
    check did not find (and back up) the outputs in manga_translator_work/result."""
    run, _chapter, _calls = cli
    monkeypatch.setattr(FakeConfigService, "source_dir", True)
    calls = run("--agent-revise", "-o", str(tmp_path / "out"))
    assert calls["save_info"][0]["save_to_source_dir"] is True


def test_the_local_module_takes_agent_redo_on_its_own(monkeypatch):
    """Review 2026-11-28 B12: ``python -m manga_translator.mode.local --agent-redo`` was an unknown argument."""
    from manga_translator.mode import local

    monkeypatch.setattr(sys, "argv", ["local", "-i", "ch01", "--agent-redo"])
    assert local.parse_args().agent_redo is True
