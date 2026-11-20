"""Agent workflow core: page document, memory, events, tools, runner and workflow (no network)."""
import asyncio
import base64
import json
import os

import cv2
import numpy as np
import pytest

from manga_translator.agent import events as agent_events
from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.page import PageDoc, RegionError
from manga_translator.agent.runner import AgentRunner, ChatReply, ToolCall, strip_images
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.tools import AgentSession, ToolRegistry
from manga_translator.agent.workflow import run_agent_workflow


def _quad(x1, y1, x2, y2):
    return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]


def _mask_b64(mask):
    ok, png = cv2.imencode(".png", mask)
    return base64.b64encode(png.tobytes()).decode()


def _decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)


def make_page(tmp_path, stem="001", boxes=((20, 20, 60, 120), (100, 20, 140, 120), (200, 150, 280, 180))):
    """An image, its work dir JSON (like Export Original Text writes) and the mask."""
    folder = tmp_path / "chapter"
    json_dir = folder / "manga_translator_work" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((240, 320, 3), 255, np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    regions = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(image, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (0, 0, 0), 2)
        mask[y1 - 4:y2 + 4, x1 - 4:x2 + 4] = 255
        regions.append({"lines": _quad(x1, y1, x2, y2), "texts": [f"text{i + 1}"], "text": f"text{i + 1}",
                        "translation": "", "font_size": 24, "direction": "vertical", "prob": 0.9,
                        "render_box_rect_local": [0, 0, 1, 1]})
    image_path = str(folder / f"{stem}.png")
    cv2.imwrite(image_path, image)
    data = {image_path: {"regions": regions, "mask_raw": _mask_b64(mask), "mask_is_refined": True,
                         "skip_font_scaling": False}}
    json_path = json_dir / f"{stem}_translations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return image_path, str(json_path)


# --- page document ----------------------------------------------------------------
def test_page_doc_edits_and_save(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    assert [r.id for r in doc.active_regions()] == [1, 2, 3]
    assert doc.missing() == [1, 2, 3]

    doc.set_text(1, "うーん", "嗯……", "dialogue", 0.9, reason="read from the page")
    kept = doc.merge([2, 1], "one bubble")
    assert kept.id == 1 and doc.regions[2].merged_into == 1
    assert kept.translation is None  # merged text must be set again
    with pytest.raises(RegionError, match="merged into #1"):
        doc.get(2)
    doc.set_text(1, "うーんそうだね", "嗯……是啊", "dialogue", 0.8)
    doc.skip(3, "sign", "background sign stays")
    assert doc.missing() == []

    doc.save()
    saved = json.load(open(json_path, encoding="utf-8"))
    page = next(iter(saved.values()))
    assert len(page["regions"]) == 1
    region = page["regions"][0]
    assert region["translation"] == "嗯……是啊" and region["texts"] == ["うーんそうだね"]
    assert len(region["lines"]) == 2 and "render_box_rect_local" not in region
    mask = _decode(page["mask_raw"])
    assert mask[160, 240] == 0, "skipped sign must leave the mask"
    assert mask[60, 40] == 255 and mask[60, 120] == 255, "kept text stays in the mask"


def test_page_doc_keeps_unfinished_regions_and_undo(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    doc.set_text(2, "a", "甲", "dialogue", 1.0)
    doc.set_text(2, "a", "乙", "dialogue", 1.0)
    assert doc.undo() == "set_text #2" and doc.regions[2].translation == "甲"
    with pytest.raises(RegionError):
        doc.set_text(1, "x", "  ", "dialogue", 1.0)
    with pytest.raises(RegionError):
        doc.set_line_breaks(2, "丙", "")
    doc.set_order([3, 1])
    assert doc.order == [3, 1, 2]
    doc.save()
    page = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert [r["translation"] for r in page["regions"]] == ["", "", "甲"]
    assert page["mask_raw"] == next(iter(json.load(open(json_path, encoding="utf-8")).values()))["mask_raw"]


def test_sidecar_matches_regions_by_center_and_bbox(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    doc.set_text(1, "x", "叉", "sfx", 0.5, ruby="", reason="r")
    doc.flag(1, "uncertain_ocr", "handwritten")
    side = doc.sidecar()
    first = side["regions"][0]
    assert first["match"] == {"center": [40.0, 70.0], "bbox": [20, 20, 60, 120]}
    assert first["reasons"] == ["r"] and side["review_flags"][0]["type"] == "uncertain_ocr"


# --- memory and events ------------------------------------------------------------
def test_memory_round_trip(tmp_path):
    memory = AgentMemory(str(tmp_path / "mem"))
    assert memory.glossary_upsert("即死パンチ", "即死拳", page="003") == "added"
    assert memory.glossary_upsert("即死ﾊﾟﾝﾁ", "必殺即死拳", page="004") == "updated"  # NFKC match
    assert memory.character_upsert("ユウ", {"target_name": "悠", "gender": "male"}, "003") == "added"
    memory.story_update("A summary.")
    again = AgentMemory(str(tmp_path / "mem"))
    assert again.glossary_search("即死")[0]["target"] == "必殺即死拳"
    assert again.character_get("悠")["ユウ"]["gender"] == "male"
    assert "A summary." in again.snapshot_text() and "即死パンチ → 必殺即死拳" in again.snapshot_text()


def test_event_log_writes_jsonl_and_notifies(tmp_path):
    seen = []
    agent_events.add_listener(seen.append)
    try:
        log = EventLog(str(tmp_path / "runs" / "r1.jsonl"), "r1")
        log.emit("tool_call", page="003", tool="zoom", args={"id": 7}, reason="字太小")
        log.close()
    finally:
        agent_events.remove_listener(seen.append)
    events = read_events(str(tmp_path / "runs" / "r1.jsonl"))
    assert events[0]["tool"] == "zoom" and events[0]["run"] == "r1" and seen[0]["reason"] == "字太小"


# --- tools --------------------------------------------------------------------------
def _session(tmp_path, **settings):
    image_path, json_path = make_page(tmp_path)
    session = AgentSession(AgentSettings(**settings), AgentMemory(str(tmp_path / "mem")), [image_path])
    session.start_page(0, PageDoc.load(image_path, json_path))
    return session, ToolRegistry(session)


def test_tools_validate_and_report_errors(tmp_path):
    session, tools = _session(tmp_path)
    assert "no tool named" in tools.execute("run_shell", "{}").text
    assert "not valid JSON" in tools.execute("zoom", "{oops").text
    assert "missing required" in tools.execute("set_text", json.dumps({"id": 1})).text
    assert "no region #9" in tools.execute("zoom", json.dumps({"id": 9})).text
    result = tools.execute("zoom", json.dumps({"id": 1, "scale": 2}))
    assert result.ok and result.images[0].startswith("data:image/jpeg;base64,")
    assert tools.execute("get_candidates", "{}").images
    assert "40x100px" in tools.execute("measure_bubble", json.dumps({"id": 1})).text


def test_mark_page_done_requires_every_region(tmp_path):
    session, tools = _session(tmp_path, low_confidence_threshold=0.7)
    first = tools.execute("mark_page_done", json.dumps({"confidence": 0.9}))
    assert not first.ok and "#1, #2, #3" in first.text
    for rid in (1, 2):
        assert tools.execute("set_text", json.dumps({"id": rid, "source_text": "s", "translation": "t",
                                                     "category": "dialogue", "confidence": 0.9})).ok
    assert tools.execute("skip_region", json.dumps({"id": 3, "category": "not_text", "reason": "no text"})).ok
    done = tools.execute("mark_page_done", json.dumps({"confidence": 0.5, "note": "puns"}))
    assert done.page_done and session.doc.flags[-1]["type"] == "low_confidence"


def test_memory_tools_use_page_name(tmp_path):
    session, tools = _session(tmp_path)
    tools.execute("glossary_upsert", json.dumps({"source": "ギルド", "target": "公會", "reason": "term"}))
    assert session.memory.glossary[0]["page"] == "001"
    session.past.append({"page": "000", "id": 1, "source": "ギルドへ", "translation": "去公會", "category": "dialogue"})
    assert "去公會" in tools.execute("search_past_translations", json.dumps({"query": "公會"})).text


# --- runner -------------------------------------------------------------------------
class ScriptedClient:
    model_label = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def complete(self, messages, tools, tool_choice):
        self.requests.append(messages)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _calls(*calls):
    return ChatReply(content=None, prompt_tokens=1000, completion_tokens=50,
                     tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(calls)])


def _finish_page(n_regions=3):
    calls = [("set_text", {"id": i, "source_text": f"s{i}", "translation": f"譯{i}", "category": "dialogue",
                           "confidence": 0.9, "reason": "讀圖"}) for i in range(1, n_regions + 1)]
    return [_calls(*calls), _calls(("mark_page_done", {"confidence": 0.9}))]


def _runner(tmp_path, pages, script, **settings):
    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    client = ScriptedClient(script)
    runner = AgentRunner(AgentSettings(**settings), client, store, events, [p for p, _ in pages], "CHT")
    return runner, client, store, events


def test_runner_finishes_pages_and_strips_old_images(tmp_path):
    pages = [make_page(tmp_path, "001"), make_page(tmp_path, "002")]
    runner, client, store, events = _runner(tmp_path, pages, _finish_page() + _finish_page(),
                                            keep_images_last_n_pages=1)
    outcomes = asyncio.run(runner.run())
    events.close()
    assert [o["status"] for o in outcomes] == ["done", "done"]
    saved = next(iter(json.load(open(pages[1][1], encoding="utf-8")).values()))
    assert [r["translation"] for r in saved["regions"]] == ["譯1", "譯2", "譯3"]
    assert store.load_sidecar("001")["status"] == "done"
    # page 2's requests carry no image from page 1
    page2_request = client.requests[2]
    page1_parts = [m for m in page2_request if isinstance(m.get("content"), list)
                   and any("001.png" in p.get("text", "") for p in m["content"] if p.get("type") == "text")]
    assert page1_parts and not any(p.get("type") == "image_url" for p in page1_parts[0]["content"])
    types = [e["type"] for e in read_events(events.path)]
    assert types[0] == "run_start" and types[-1] == "run_end" and types.count("page_done") == 2
    # every assistant tool call is answered before the next request
    for request in client.requests:
        pending = set()
        for message in request:
            if message.get("tool_calls"):
                pending |= {c["id"] for c in message["tool_calls"]}
            if message.get("role") == "tool":
                pending.discard(message["tool_call_id"])
        assert not pending


def test_runner_flags_failed_page_and_continues(tmp_path):
    pages = [make_page(tmp_path, "001"), make_page(tmp_path, "002")]
    script = [_calls(("set_text", {"id": 1, "source_text": "s", "translation": "t", "category": "dialogue",
                                   "confidence": 1})), RuntimeError("content refused")] + _finish_page()
    runner, client, store, events = _runner(tmp_path, pages, script)
    outcomes = asyncio.run(runner.run())
    events.close()
    assert [o["status"] for o in outcomes] == ["error", "done"]
    assert store.load_sidecar("001")["review_flags"][0]["type"] == "error"
    # the failed page's half turn is not sent with page 2
    assert all(m.get("role") != "tool" for m in client.requests[-2][:3])


def test_runner_step_limit_and_compaction(tmp_path):
    pages = [make_page(tmp_path, "001"), make_page(tmp_path, "002")]
    big = ChatReply(content="thinking", prompt_tokens=5000)
    runner, client, store, events = _runner(tmp_path, pages, [big, big] + _finish_page(),
                                            max_steps_per_page=2, compact_threshold_tokens=4000)
    outcomes = asyncio.run(runner.run())
    events.close()
    assert outcomes[0]["status"] == "incomplete" and outcomes[0]["missing"] == [1, 2, 3]
    assert outcomes[1]["status"] == "done"
    logged = read_events(events.path)
    assert any(e["type"] == "compact" for e in logged)
    assert "Pages already finished in this run: 001" in client.requests[2][1]["content"]


def test_strip_images_keeps_text():
    message = {"role": "user", "content": [{"type": "text", "text": "hi"},
                                           {"type": "image_url", "image_url": {"url": "x"}}]}
    stripped = strip_images(message, "page 1")
    assert stripped["content"][0]["text"] == "hi" and "omitted" in stripped["content"][1]["text"]


# --- workflow -----------------------------------------------------------------------
class FakeTranslator:
    """Records the mode flags of each translate_batch call; the first call
    plays the preprocess and writes the JSON files."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.calls = []
        self.deleted = []
        for name in ("template", "save_text", "load_text", "skip_txt_import", "ignore_existing_inpainted"):
            setattr(self, name, False)

    async def translate_batch(self, images_with_configs, save_info=None):
        self.calls.append({k: getattr(self, k) for k in ("template", "save_text", "load_text", "skip_txt_import",
                                                          "ignore_existing_inpainted")})
        if self.template:
            for path, _ in images_with_configs:
                make_page(self.tmp_path, os.path.splitext(os.path.basename(path))[0])
        return [f"ctx:{os.path.basename(p)}" for p, _ in images_with_configs]

    def _delete_original_txt_after_json_translation(self, path):
        self.deleted.append(path)


def test_workflow_runs_preprocess_agent_render(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]

    class Cfg:
        class translator:
            target_lang = "CHT"

    translator = FakeTranslator(tmp_path)
    client = ScriptedClient(_finish_page() + _finish_page())
    result = asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {}, client=client, log=lambda m: None))
    assert result == ["ctx:001.png", "ctx:002.png"]
    assert translator.calls[0]["template"] and translator.calls[0]["save_text"]
    assert translator.calls[1] == {"template": False, "save_text": True, "load_text": True, "skip_txt_import": True,
                                   "ignore_existing_inpainted": True}
    assert translator.deleted == paths
    assert not translator.load_text and not translator.template  # flags restored
    runs = os.listdir(folder / "manga_translator_work" / "agent" / "runs")
    assert len(runs) == 1 and runs[0].endswith(".jsonl")
