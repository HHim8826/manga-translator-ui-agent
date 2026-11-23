"""A step that failed on the API's usage limit is done again by hand (``redo``), not waited for: the run
ends as before, the step is written down with the time the limit resets, and when the user starts the
chapter again the rerun question offers to redo only that step. The API clients are closed in the
run's own event loop."""
import asyncio
import datetime as dt
import json
import os

import httpx
import openai
import pytest

from manga_translator import api_key_rotation
from manga_translator.agent import redo
from manga_translator.agent.client import OpenAIChatClient, quota_reset_seconds, quota_resets
from manga_translator.agent.events import read_events
from manga_translator.agent.resume import INPAINTED, PageState, previous_states, record_saved
from manga_translator.utils.path_manager import get_inpainted_path
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore

from test_agent_core import FakeTranslator, ScriptedClient, _calls, _finish_page, run_agent_workflow
from test_agent_inpaint import FakeBackend
from test_agent_revise import PaintBackend
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


def _limit(seconds=792, resets_at=None, headers=None, code=429):
    """The error an API proxy sends when its usage limit is used up (ch37 proofread,
    2026-11-22)."""
    error = {"type": "usage_limit_reached", "message": "The usage limit has been reached",
             "limit_window_minutes": 300}
    if seconds is not None:
        error["resets_in_seconds"] = seconds
    if resets_at is not None:
        error["resets_at"] = resets_at
    response = httpx.Response(code, headers=headers or {},
                              request=httpx.Request("POST", "http://api/chat/completions"))
    cls = openai.RateLimitError if code == 429 else openai.InternalServerError
    return cls(f"Error code: {code} - {{'error': {error}}}", response=response, body=error)


def test_a_reset_after_midnight_is_later_than_one_before_it():
    """The review report of 2026-11-22: two steps of a page reset at 23:55 and at 00:10 the next day."""
    zone = dt.timezone(dt.timedelta(hours=8))
    evening = dt.datetime(2026, 11, 22, 23, 50, tzinfo=zone)
    data = {"inpaint_review": {"001": {"resets": "2026-11-22T23:55+08:00", "at": "2026-11-22T23:40:00+08:00"}},
            "proofread": {"001": {"resets": "2026-11-23T00:10+08:00", "at": "2026-11-22T23:45:00+08:00"}}}
    assert redo.resets_of(data, "001", evening) == "2026-11-23T00:10+08:00"
    assert redo.reset_label("2026-11-23T00:10+08:00", evening) == "11/23 00:10"
    assert redo.reset_label("2026-11-22T23:55+08:00", evening) == "23:55"
    assert redo.resets_of(data, "001", evening + dt.timedelta(hours=1)) == ""  # both have passed
    # the first records kept only HH:MM: the first such time after the step failed
    old = {"proofread": {"001": {"resets": "00:10", "at": "2026-11-22T23:45:00+08:00"},
                         "002": {"resets": "23:55", "at": "2026-11-22T23:45:00+08:00"}}}
    assert redo.resets_of(old, "001", evening) == "2026-11-23T00:10+08:00"
    assert redo.resets_of(old, "002", evening) == "2026-11-22T23:55+08:00"
    assert redo.reset_time({"resets": "nonsense", "at": "2026-11-22T23:45:00+08:00"}) is None
    assert redo.latest_reset(["2026-11-22T23:55+08:00", "2026-11-23T00:10+08:00", ""], evening) ==         "2026-11-23T00:10+08:00"


def test_the_reset_time_is_read_from_the_error_the_api_sent():
    now = 1_795_357_000.0
    assert quota_reset_seconds(_limit(792), now) == 792
    try:  # as the agent sees it: the key rotation gave up, the 429 is its cause
        try:
            raise _limit(792)
        except openai.RateLimitError as exc:
            raise api_key_rotation.APIRotationExhaustedError("agent chat completion failed") from exc
    except api_key_rotation.APIRotationExhaustedError as wrapped:
        assert quota_reset_seconds(wrapped, now) == 792
        fields = quota_resets(wrapped, now)
        at = dt.datetime.fromisoformat(fields["resets_at"])
        assert set(fields) == {"resets", "resets_at"} and at.tzinfo is not None
        assert 792 <= at.timestamp() - now < 792 + 60  # rounded up to the minute
        assert fields["resets"] == redo.reset_label(at, dt.datetime.fromtimestamp(now).astimezone())
    assert quota_reset_seconds(_limit(None, resets_at=now + 600), now) == 600
    assert quota_reset_seconds(_limit(None, resets_at=(now + 60) * 1000), now) == pytest.approx(60)
    assert quota_reset_seconds(_limit(None, headers={"retry-after": "120"}), now) == 120
    # only the text of the error is left (another client library)
    text = RuntimeError("Error code: 429 - {'error': {'type': 'usage_limit_reached', 'resets_in_seconds': 792}}")
    assert quota_reset_seconds(text, now) == 792
    assert quota_reset_seconds(_limit(None), now) is None  # a 429 that does not say when
    assert quota_reset_seconds(_limit(792, code=500), now) is None
    assert quota_reset_seconds(RuntimeError("the streamed reply stopped before its end"), now) is None
    assert quota_resets(RuntimeError("boom"), now) == {}


def test_the_api_clients_are_closed_in_the_loop_that_opened_them():
    endpoint = api_key_rotation.APIEndpoint(feature="translator", provider="openai", slot=1, api_key="x",
                                            base_url="http://127.0.0.1:9/v1", model_name="m",
                                            status_key="test-redo-close", label="close")
    client = OpenAIChatClient((endpoint,), "failover", AgentSettings())

    async def run():
        api = client._client(endpoint)
        await client.aclose()
        return api

    api = asyncio.run(run())
    assert api.is_closed() and not client._clients


# --- redo: the step is written down, and done again by hand ------------------------------------
SETTINGS = {"inpaint_review": False, "proofread": True, "recall_pass": False}


class Cfg:
    class translator:
        target_lang = "CHT"


def _used_up():
    """What the agent's client raises when every key hit the usage limit: the rotation's error, the 429
    as its cause."""
    error = api_key_rotation.APIRotationExhaustedError("agent chat completion failed")
    error.__cause__ = _limit(792)
    return error


class LimitedClient(ScriptedClient):
    """The proofreading (its ``view_page`` tool) or the inpaint review (``view_region``) hits the limit."""

    def __init__(self, script, fail=("view_page",), **kwargs):
        super().__init__(script, **kwargs)
        self.fail = set(fail)

    async def complete(self, messages, tools, tool_choice, effort="", cache_key=""):
        if tools and tools[0]["function"]["name"] in self.fail:
            raise _used_up()
        return await super().complete(messages, tools, tool_choice, effort, cache_key)


def _chapter(tmp_path, n=2):
    folder = tmp_path / "chapter"
    folder.mkdir(exist_ok=True)
    return folder, [str(folder / f"{i:03d}.png") for i in range(1, n + 1)]


def _store(folder):
    return AgentStore(str(folder / "manga_translator_work"))


def _log(folder):
    """The latest run's events (runs of a test start in the same second: they share a log)."""
    runs = folder / "manga_translator_work" / "agent" / "runs"
    events = read_events(str(runs / sorted(os.listdir(runs))[-1]))
    return events[max(i for i, e in enumerate(events) if e["type"] == "run_start"):]


def _json(folder, stem):
    return folder / "manga_translator_work" / "json" / f"{stem}_translations.json"


def _regions(folder, stem):
    raw = json.loads(_json(folder, stem).read_text(encoding="utf-8"))
    return [r["translation"] for r in next(iter(raw.values()))["regions"]]


def _edit(folder, stem, text="我改的"):
    raw = json.loads(_json(folder, stem).read_text(encoding="utf-8"))
    region = next(iter(raw.values()))["regions"][0]
    region["translation"] = region["translation_raw"] = text
    _json(folder, stem).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def _limited_run(tmp_path, n=2, **settings):
    """A chapter whose proofreading hit the usage limit (ch37, 2026-11-22): the pages are rendered
    without it."""
    folder, paths = _chapter(tmp_path, n)
    translator = FakeTranslator(tmp_path)
    results = asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {},
                                             client=LimitedClient(_finish_page() * n), log=lambda m: None,
                                             agent_settings={**SETTINGS, **settings}))
    return folder, paths, results


def _redo(tmp_path, paths, client, backend=None, **settings):
    translator = FakeTranslator(tmp_path)
    logs = []
    results = asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {}, client=client,
                                             log=logs.append, redo=True, inpaint_backend=backend or PaintBackend(),
                                             agent_settings={**SETTINGS, **settings}))
    return results, translator, logs


def test_a_proofreading_that_hit_the_usage_limit_is_written_down(tmp_path):
    folder, paths, results = _limited_run(tmp_path)
    assert results == ["ctx:001.png", "ctx:002.png"]  # rendered as before, without the proofreading
    pending = redo.read(_store(folder))
    assert set(pending) == {"proofread"} and set(pending["proofread"]) == {"001", "002"}
    entry = pending["proofread"]["001"]
    assert redo.reset_time(entry) > dt.datetime.now().astimezone()  # with its date and time zone
    assert "agent chat completion failed" in entry["error"] and entry["at"]
    error = next(e for e in _log(folder) if e["type"] == "error" and e.get("agent") == "proofread")
    assert error["redo"] is True and error["resets_at"] == entry["resets"] and error["resets"]
    states = previous_states(paths)
    assert [(s.status, s.pending, s.resets) for s in states] == [("done", ("proofread",), entry["resets"])] * 2


def test_the_rerun_question_offers_to_redo_the_steps(tmp_path, monkeypatch, app):  # noqa: F811
    from ui.main_page.pages import agent_rerun
    from ui.main_page.pages.agent_rerun import REDO, ask_rerun, rerun_question

    soon = dt.datetime.now().astimezone() + dt.timedelta(minutes=30)
    later = soon + dt.timedelta(days=1)
    states = [PageState(str(tmp_path / "001.png"), "done", pending=("proofread",), resets=soon.isoformat()),
              PageState(str(tmp_path / "002.png"), "done", pending=("inpaint_review", "proofread"),
                        resets=later.isoformat()),
              PageState(str(tmp_path / "003.png"), "done"),
              PageState(str(tmp_path / "004.png"), "error", pending=("proofread",))]  # not rendered: no redo
    question = rerun_question(_t, states)
    assert [os.path.basename(p) for p in question["redo"]] == ["001.png", "002.png"]
    assert ("Agent Rerun Redo Note count=2 pages=001Agent Result Separator002 "
            "steps=Agents InpaintAgent Result SeparatorAgents Proofread") in question["text"]
    assert f"Agent Rerun Redo Resets time={later.strftime('%m/%d %H:%M')}" in question["text"]  # another day
    states[1].resets = ""
    assert f"Agent Rerun Redo Resets time={redo.reset_label(soon)}" in rerun_question(_t, states)["text"]
    states[0].resets = (dt.datetime.now().astimezone() - dt.timedelta(minutes=1)).isoformat()
    assert "Agent Rerun Redo Resets" not in rerun_question(_t, states)["text"]  # passed: the limit is back
    assert "Agent Rerun Redo" not in rerun_question(_t, states[2:])["text"]

    seen = []

    def click(box):
        seen.append(([b.text() for b in box.choice_buttons], box.default_button.text()))
        next(b for b in box.choice_buttons if b.text() == "Agent Rerun Redo count=2").click()
        return 0

    monkeypatch.setattr(agent_rerun, "_exec", click)
    chosen = ask_rerun(None, _t, [s.path for s in states], states)
    assert ([os.path.basename(p) for p in chosen[0]], chosen[1]) == (["001.png", "002.png"], REDO)
    assert seen[0][1] == "Agent Rerun Redo count=2"  # the first choice when there are steps to redo


def test_a_redo_does_only_the_proofreading_and_leaves_the_users_boxes(tmp_path):
    folder, paths, _results = _limited_run(tmp_path)
    _edit(folder, "002")  # in the editor, after the run
    proof = [_calls(("revise_text", {"page": "001", "id": 2, "translation": "改二", "reason": "語氣"}),
                    ("revise_text", {"page": "002", "id": 1, "translation": "不行", "reason": "r"})),
             _calls(("finish_proofread", {}))]
    client = ScriptedClient([], proof=proof)
    results, translator, _logs = _redo(tmp_path, paths, client)
    assert client.requests == [] and len(client.proof_requests) == 2  # no translation, only the proofreading
    assert "LOCKED" in client.proof_requests[0][1]["content"]
    assert translator.calls == [{"template": False, "save_text": True, "load_text": True, "skip_txt_import": True,
                                 "ignore_existing_inpainted": False}]  # no preprocess; the render reuses
    assert results == ["ctx:001.png", "ctx:002.png"]
    assert _regions(folder, "001") == ["譯1", "改二", "譯3"]
    assert _regions(folder, "002") == ["我改的", "譯2", "譯3"]  # the user's box stays
    assert redo.read(_store(folder)) == {}
    assert not os.path.exists(os.path.join(_store(folder).agent_dir, redo.PENDING))
    log = _log(folder)
    assert next(e for e in log if e["type"] == "run_start")["redo"] == {"proofread": 2}
    assert next(e for e in log if e["type"] == "run_end")["redo"] is True
    backups = os.listdir(folder / "manga_translator_work" / "agent" / "backups")
    assert any(n.startswith("002_translations.") for n in backups)
    assert not (folder / "manga_translator_work" / "agent" / "run_journal.json").exists()
    assert previous_states(paths)[0].pending == ()


def test_a_redo_that_fails_again_keeps_the_pages_and_the_step(tmp_path):
    folder, paths, _results = _limited_run(tmp_path)
    results, translator, _logs = _redo(tmp_path, paths, LimitedClient([]))
    assert not any(c["load_text"] for c in translator.calls)  # nothing rendered over the outputs
    assert [getattr(r, "skip_reason", None) for r in results] == ["agent_redo_failed"] * 2
    assert set(redo.read(_store(folder))["proofread"]) == {"001", "002"}  # still to do
    render = next(e for e in _log(folder) if e["type"] == "render_start")
    assert render["skipped"] == {"001": "redo_failed", "002": "redo_failed"}


def test_pages_translated_again_forget_their_steps(tmp_path):
    folder, paths, _results = _limited_run(tmp_path)
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(paths[0], Cfg)], {},
                                   client=ScriptedClient(_finish_page()), log=lambda m: None,
                                   agent_settings={**SETTINGS, "proofread": False}))
    assert set(redo.read(_store(folder))["proofread"]) == {"002"}  # 001 was translated again


def test_an_inpaint_review_is_redone_except_on_a_page_the_user_changed(tmp_path):
    folder, paths = _chapter(tmp_path)
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                   client=LimitedClient(_finish_page() * 2, fail=("view_region",)),
                                   log=lambda m: None, inpaint_backend=FakeBackend(),
                                   agent_settings={**SETTINGS, "inpaint_review": True, "proofread": False}))
    pending = redo.read(_store(folder))
    assert set(pending) == {"inpaint_review"} and set(pending["inpaint_review"]) == {"001", "002"}
    assert pending["inpaint_review"]["001"]["resets"]
    _edit(folder, "002")  # its inpainted image may hold the user's repairs now
    client = ScriptedClient([])
    backend = FakeBackend()
    results, translator, logs = _redo(tmp_path, paths, client, backend, inpaint_review=True, proofread=False)
    assert len(client.review_requests) == 1 and client.requests == [] and client.proof_requests == []
    assert any("002: changed in the editor after the run" in line for line in logs)
    assert results[0] == "ctx:001.png" and getattr(results[1], "skip_reason", None) == "agent_redo_yours"
    assert [c["ignore_existing_inpainted"] for c in translator.calls if c["load_text"]] == [False]
    assert list(backend.saved) == [paths[0]]  # the review saved 001's inpainted image, the render reuses it
    assert redo.read(_store(folder)) == {}  # 001 done, 002's review no longer wanted


def test_a_redo_leaves_an_inpainted_image_the_user_changed(tmp_path):
    """The review report of 2026-11-22 (P1-06): only the inpainted PNG was repaired, the JSON kept."""
    folder, paths = _chapter(tmp_path)
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                   client=LimitedClient(_finish_page() * 2, fail=("view_region",)),
                                   log=lambda m: None, inpaint_backend=FakeBackend(),
                                   agent_settings={**SETTINGS, "inpaint_review": True, "proofread": False}))
    for path in paths:  # as the render left them
        with open(get_inpainted_path(path), "wb") as fh:
            fh.write(b"the pipeline's inpainting of " + os.path.basename(path).encode())
    record_saved(paths)
    mine = get_inpainted_path(paths[1])
    with open(mine, "wb") as fh:
        fh.write(b"the user's repairs: (13, 88, 156)")
    assert [(s.edited, s.inpaint_edited) for s in previous_states(paths)] == [(False, False), (False, True)]
    client, backend = ScriptedClient([]), FakeBackend()
    results, translator, logs = _redo(tmp_path, paths, client, backend, inpaint_review=True, proofread=False)
    assert len(client.review_requests) == 1 and list(backend.saved) == [paths[0]]  # only 001 reviewed
    assert any("002: its inpainted image changed after the run" in line for line in logs)
    assert results[0] == "ctx:001.png" and getattr(results[1], "skip_reason", None) == "agent_redo_yours"
    assert open(mine, "rb").read() == b"the user's repairs: (13, 88, 156)"
    assert redo.read(_store(folder)) == {}
    render = next(e for e in _log(folder) if e["type"] == "render_start")
    assert render["skipped"] == {"002": "redo_yours"}


def test_without_a_record_an_inpainted_image_written_after_the_run_is_the_users(tmp_path):
    folder, paths = _chapter(tmp_path)
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                   client=LimitedClient(_finish_page() * 2, fail=("view_region",)),
                                   log=lambda m: None, inpaint_backend=FakeBackend(),
                                   agent_settings={**SETTINGS, "inpaint_review": True, "proofread": False}))
    record = os.path.join(_store(folder).agent_dir, INPAINTED)
    assert not os.path.exists(record) or "001" not in json.loads(open(record, encoding="utf-8").read())
    logged = os.path.getmtime(_store(folder).latest_run_log())
    for path, when in zip(paths, (logged - 100, logged + 120)):  # a run before the records; then the user
        with open(get_inpainted_path(path), "wb") as fh:
            fh.write(b"x")
        os.utime(get_inpainted_path(path), (when, when))
    assert [s.inpaint_edited for s in previous_states(paths)] == [False, True]


def test_a_proofreading_that_stopped_at_its_step_limit_stays_to_redo(tmp_path):
    """The review report of 2026-11-22 (P2-01): ``proofread_steps`` 1, the proofreader did not finish."""
    folder, paths = _chapter(tmp_path)
    client = ScriptedClient(_finish_page() * 2, proof=[
        _calls(("revise_text", {"page": "001", "id": 2, "translation": "改二", "reason": "語氣"}))])
    results = asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {}, client=client,
                                             log=lambda m: None, agent_settings={**SETTINGS, "proofread_steps": 1}))
    assert results == ["ctx:001.png", "ctx:002.png"]  # rendered, with the revision it made
    assert _regions(folder, "001") == ["譯1", "改二", "譯3"]
    pending = redo.read(_store(folder))["proofread"]
    assert set(pending) == {"001", "002"} and "step limit (1 steps)" in pending["001"]["error"]
    done = next(e for e in _log(folder) if e["type"] == "proofread_done")
    assert done["finished"] is False and done["redo"] is True
    # the redo goes on with it, and a proofreading that finishes takes it off the list
    _redo(tmp_path, paths, ScriptedClient([], proof=[_calls(("finish_proofread", {}))]))
    assert redo.read(_store(folder)) == {} and _regions(folder, "001") == ["譯1", "改二", "譯3"]


def test_a_rerun_that_is_stopped_puts_back_the_steps_to_redo(tmp_path):
    """The review report of 2026-11-22 (P2-02): every page again, stopped at the first translation."""
    folder, paths, _results = _limited_run(tmp_path)
    before = [_json(folder, stem).read_bytes() for stem in ("001", "002")]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                       client=ScriptedClient([asyncio.CancelledError()]), log=lambda m: None,
                                       agent_settings=SETTINGS))
    assert [_json(folder, stem).read_bytes() for stem in ("001", "002")] == before
    pending = redo.read(_store(folder))
    assert set(pending) == {"proofread"} and set(pending["proofread"]) == {"001", "002"}
    assert "agent chat completion failed" in pending["proofread"]["001"]["error"]  # the entry as it was
    assert [s.pending for s in previous_states(paths)] == [("proofread",)] * 2


def test_the_page_says_how_to_redo_a_step(app):  # noqa: F811
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "run": "r1", "batch": "b1", "chapter": "第37話", "pages": ["001.jpg", "002.jpg"],
         "model": "m"},
        {"type": "error", "run": "r1", "agent": "proofread", "error": "APIRotationExhaustedError: x", "redo": True,
         "resets": "13:05"},
        {"type": "error", "run": "r1", "agent": "inpaint", "page": "002", "error": "RuntimeError: x", "redo": True},
        {"type": "error", "run": "r1", "agent": "translator", "page": "001", "error": "boom"},
        {"type": "proofread_done", "run": "r1", "agent": "proofread", "revised": 1, "flagged": 0,
         "finished": False, "steps": 1, "note": "", "redo": True},
    ])
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert [r.text(4) for r in rows[-4:]] == ["Agents Redo Hint Resets time=13:05", "Agents Redo Hint", "",
                                              "Agents Proofread Unfinished"]

    page = AgentsPage(_t)
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "redo", "pages": 2})
    assert page.status_label.text() == "Agents Stage Redo pages=2" and page._agent_task
    page._handle_live({"type": "run_start", "run": "r2", "batch": "b2", "chapter": "第37話",
                       "pages": ["001.jpg", "002.jpg"], "model": "m", "redo": {"proofread": 2}})
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert rows[-1].text(3) == "Agents Redo Start proofread=2 review=0"
    page._handle_live({"type": "page_done", "run": "r2", "page": "001", "status": "redo_failed", "flags": 0})
    page._handle_live({"type": "render_start", "run": "r2", "agent": "render", "pages": 0,
                       "skipped": {"001": "redo_failed", "002": "redo_none"}})
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert rows[-1].text(4) == ("Agents Render Held pages=001（Agent Page Redo Failed）Agent Result Separator"
                                "002（Agent Page Redo None）")
    assert "Agent Page Redo Failed" in page._page_text(page._order[0])
