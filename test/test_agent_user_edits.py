"""Pages the user saves in the editor while an AI Agent run works on them (``json_guard``), and an
inpainted image the review could not save."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from manga_translator.agent import json_guard
from manga_translator.agent.events import read_events
from manga_translator.agent.store import AgentStore

from test_agent_core import FakeTranslator, ScriptedClient, _finish_page, run_agent_workflow


class Cfg:
    class translator:
        target_lang = "CHT"


def _json(folder, stem):
    return folder / "manga_translator_work" / "json" / f"{stem}_translations.json"


def _save_in_editor(path, text="我改的"):
    """What the editor's save does: the whole JSON written again, with the user's translation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    for region in next(iter(raw.values()))["regions"]:
        region["translation"] = text
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=4), encoding="utf-8")
    return path.read_bytes()


class EditingClient(ScriptedClient):
    """The user saves ``target`` in the editor when the translator sends request ``at``."""

    def __init__(self, script, target, at):
        super().__init__(script)
        self.target, self.at, self.saved = target, at, None

    async def complete(self, messages, tools, tool_choice, effort="", cache_key=""):
        reply = await super().complete(messages, tools, tool_choice, effort)
        if len(self.requests) == self.at and self.saved is None:
            self.saved = _save_in_editor(self.target)
        return reply


def _run(tmp_path, paths, client, translator=None, **settings):
    return asyncio.run(run_agent_workflow(translator or FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                          client=client, log=lambda m: None,
                                          agent_settings={"inpaint_review": False, "recall_pass": False, **settings}))


def _events(folder):
    runs = folder / "manga_translator_work" / "agent" / "runs"
    return read_events(str(runs / sorted(os.listdir(runs))[-1]))


@pytest.fixture(autouse=True)
def _clean():
    json_guard.reset()
    yield
    json_guard.reset()


def test_a_save_while_the_agent_works_on_the_page_keeps_both_versions(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png")]
    client = EditingClient(_finish_page(), _json(folder, "001"), at=1)  # saved while the model answers
    results = _run(tmp_path, paths, client)
    assert _json(folder, "001").read_bytes() == client.saved  # the user's version stays
    conflicts = folder / "manga_translator_work" / "agent" / "conflicts"
    copies = os.listdir(conflicts)
    agent_version = json.loads((conflicts / copies[0]).read_text(encoding="utf-8"))
    assert [r["translation"] for r in next(iter(agent_version.values()))["regions"]] == ["譯1", "譯2", "譯3"]
    side = AgentStore(str(folder / "manga_translator_work")).load_sidecar("001")
    assert side["status"] == "edited" and side["review_flags"][-1]["type"] == "conflict"
    assert copies[0] in side["review_flags"][-1]["reason"]
    assert results[0].skip_reason == "agent_edited"  # not typeset: the user exports it from the editor
    assert not json_guard.users_files()  # forgotten after the run


def test_a_page_saved_before_the_agent_gets_to_it_is_left_alone(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    client = EditingClient(_finish_page(), _json(folder, "002"), at=1)  # 002 saved during 001
    translator = FakeTranslator(tmp_path)
    results = _run(tmp_path, paths, client, translator=translator)
    assert len(client.requests) == 2  # no model call for 002
    assert _json(folder, "002").read_bytes() == client.saved
    done = [e for e in _events(folder) if e["type"] == "page_done"]
    assert [(e["page"], e["status"]) for e in done] == [("001", "done"), ("002", "edited")]
    assert results[0] == "ctx:001.png" and results[1].skip_reason == "agent_edited"


def test_a_page_saved_during_a_stopped_run_is_not_put_back(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    _run(tmp_path, paths, ScriptedClient(_finish_page() * 2))
    json_guard.reset()
    # rerun: the user saves 002 while 001 is translated, then stops the run at 002
    client = EditingClient(_finish_page()[:1] + [asyncio.CancelledError()], _json(folder, "002"), at=1)
    with pytest.raises(asyncio.CancelledError):
        _run(tmp_path, paths, client)
    assert _json(folder, "002").read_bytes() == client.saved


def test_the_proofreading_and_the_font_check_leave_the_users_page_alone(tmp_path):
    from manga_translator.agent.proofread import ProofPage, ProofError

    from test_agent_core import make_page

    from manga_translator.agent.fonts import FontService

    image, json_path = make_page(tmp_path, "001")
    json_guard.note(json_path)
    page = ProofPage(image)
    saved = _save_in_editor(tmp_path / "chapter" / "manga_translator_work" / "json" / "001_translations.json")
    with pytest.raises(ProofError, match="saved in the editor"):
        page.write(1, "改")
    service = FontService.__new__(FontService)  # leaves the page before it looks at a font
    result = asyncio.run(service.fix_page(image, store=None, events=None))
    assert result == {"page": "001", "changed": 0, "unfixed": 0}
    assert open(json_path, "rb").read() == saved


def test_the_render_does_not_write_the_users_json_again(tmp_path):
    from manga_translator.manga_translator import MangaTranslator

    translator = SimpleNamespace(json_save_guard=lambda image: False, text_output_file=None)
    assert MangaTranslator._save_text_to_file(translator, str(tmp_path / "001.png"), ctx=None) is True
    assert not os.path.exists(tmp_path / "manga_translator_work")


# --- review 10: the reviewed image was not saved -------------------------------------------------
def test_the_pipeline_backend_says_when_the_image_was_not_saved():
    from manga_translator.agent.workflow import PipelineInpaintBackend

    backend = PipelineInpaintBackend.__new__(PipelineInpaintBackend)
    backend.translator = SimpleNamespace(_save_inpainted_image=lambda path, image: None)
    with pytest.raises(OSError, match="not saved"):
        backend.save("x/001.png", object())
    backend.translator = SimpleNamespace(_save_inpainted_image=lambda path, image: "x/inpainted/001.png")
    assert backend.save("x/001.png", object()) == "x/inpainted/001.png"


def test_a_review_whose_image_was_not_saved_is_not_reused_by_the_render(tmp_path):
    from test_agent_inpaint import FakeBackend

    class Unsaved(FakeBackend):
        def save(self, path, image):
            raise OSError("the inpainted image of 001.png was not saved")

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png")]
    translator = FakeTranslator(tmp_path)
    client = ScriptedClient(_finish_page())
    asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {}, client=client, log=lambda m: None,
                                   inpaint_backend=Unsaved(), agent_settings={"recall_pass": False}))
    store = AgentStore(str(folder / "manga_translator_work"))
    assert "inpaint" not in store.load_sidecar("001")  # nothing recorded as reviewed
    assert [c["ignore_existing_inpainted"] for c in translator.calls[1:]] == [True]  # inpainted again
    errors = [e for e in _events(folder) if e["type"] == "error" and e.get("agent") == "inpaint"]
    assert errors and "not saved" in errors[0]["error"]


def test_the_next_run_tells_the_pages_saved_in_the_editor_during_a_run_the_users(tmp_path):
    """Review 2026-11-28 B3: the run fingerprinted the user's JSON as its own, so the next run took
    the page for the agent's, offered no warning and dropped its copy; a page saved before the agent
    got to it had no sidecar of its own, so the editor's AI Agent tab never showed the conflict."""
    from manga_translator.agent.resume import FINGERPRINTS, USERS_SAVE, previous_states

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png"), str(folder / "003.png")]
    _run(tmp_path, paths, ScriptedClient(_finish_page() * 3))
    json_guard.reset()
    # rerun: 001 is saved while the model works on it, 002 while 001 is worked on (before its turn)
    client = EditingClient(_finish_page() * 2, _json(folder, "001"), at=1)
    first = client.complete

    async def complete(messages, tools, tool_choice, effort="", cache_key=""):
        reply = await first(messages, tools, tool_choice, effort, cache_key)
        if len(client.requests) == 1:
            _save_in_editor(_json(folder, "002"), "也是我改的")
        return reply

    client.complete = complete
    _run(tmp_path, paths, client)
    json_guard.reset()
    store = AgentStore(str(folder / "manga_translator_work"))
    side = store.load_sidecar("002")
    assert side["status"] == "edited" and side["review_flags"][-1]["type"] == "conflict"
    prints = json.loads(open(os.path.join(store.agent_dir, FINGERPRINTS), encoding="utf-8").read())
    assert prints["001"] == USERS_SAVE and prints["002"] == USERS_SAVE and prints["003"] != USERS_SAVE
    states = {s.stem: s for s in previous_states(paths)}
    assert states["001"].edited and states["002"].edited and states["001"].users
    assert not states["003"].edited and not states["003"].users
