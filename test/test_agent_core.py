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
from manga_translator.agent.snap import snap_text
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


def _write_column(image, x, y, letters="ABCD", step=24):
    """Vertical 'text': letters stacked in one column (strokes the detector missed)."""
    for i, letter in enumerate(letters):
        cv2.putText(image, letter, (x, y + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def make_page(tmp_path, stem="001", boxes=((20, 20, 60, 120), (100, 20, 140, 120), (200, 150, 280, 180)),
              unboxed=False):
    """An image, its work dir JSON (like Export Original Text writes) and the mask.
    ``unboxed`` adds a column of text at x 165-185, y 22-115 that no region covers."""
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
    if unboxed:
        _write_column(image, 165, 40)
    image_path = str(folder / f"{stem}.png")
    cv2.imwrite(image_path, image)
    data = {image_path: {"regions": regions, "mask_raw": _mask_b64(mask), "mask_is_refined": True,
                         "skip_font_scaling": False, "original_width": 320, "original_height": 240}}
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
        assert tools.execute("set_text", json.dumps({"id": rid, "source_text": f"text{rid}", "translation": "t",
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
    """Main-conversation replies come from ``script`` in order; recall checks
    (their own tool) take the next entry of ``recall`` (default: nothing found)."""
    model_label = "fake"

    def __init__(self, script, recall=None):
        self.script = list(script)
        self.recall = list(recall or [])
        self.requests = []
        self.recall_requests = []

    async def complete(self, messages, tools, tool_choice):
        if tools and tools[0]["function"]["name"] == "report_unboxed_text":
            self.recall_requests.append(messages)
            items = self.recall.pop(0) if self.recall else []
            return _calls(("report_unboxed_text", {"items": items}))
        self.requests.append(messages)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _calls(*calls):
    return ChatReply(content=None, prompt_tokens=1000, completion_tokens=50,
                     tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(calls)])


def _finish_page(n_regions=3):
    calls = [("set_text", {"id": i, "source_text": f"text{i}", "translation": f"譯{i}", "category": "dialogue",
                           "confidence": 0.9, "reason": "讀圖"}) for i in range(1, n_regions + 1)]
    return [_calls(*calls), _calls(("mark_page_done", {"confidence": 0.9}))]


def _runner(tmp_path, pages, script, recall=None, **settings):
    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    client = ScriptedClient(script, recall)
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


# --- recall: snapping, add_region, recall check ----------------------------------------
def _bubble_page():
    """A speech bubble with two columns of text, screentone outside the bubble."""
    image = np.full((300, 400, 3), 255, np.uint8)
    inside = np.zeros((300, 400), np.uint8)
    cv2.ellipse(inside, (200, 150), (80, 115), 0, 0, 360, 1, -1)
    for y in range(2, 300, 6):
        for x in range(2, 400, 6):
            if not inside[y, x]:
                cv2.circle(image, (x, y), 1, (60, 60, 60), -1)
    cv2.ellipse(image, (200, 150), (80, 115), 0, 0, 360, (0, 0, 0), 3)
    _write_column(image, 210, 90, "ABCDE")
    _write_column(image, 180, 90, "FGHIJ")
    return image


def test_snap_text_tightens_a_loose_box_and_ignores_outline_and_tone():
    image = _bubble_page()
    # text spans x 180-228, y 73-190; the box is loose and reaches past the bubble into the screentone
    snap = snap_text(image, (150, 20, 250, 215), "v")
    assert snap is not None
    x1, y1, x2, y2 = snap.bbox
    assert 176 <= x1 <= 182 and 224 <= x2 <= 232 and 70 <= y1 <= 76 and 186 <= y2 <= 194
    assert len(snap.lines) == 2 and 14 <= snap.font_size <= 24
    assert snap.fg[0] < 80 and snap.bg[0] > 200
    ox, oy = snap.erase_origin
    assert ox >= x1 - 8 and oy >= y1 - 8 and snap.erase.any()
    # a box that cuts the text off still gets the whole column
    cut = snap_text(image, (205, 85, 235, 160), "v")
    assert cut.bbox[1] <= 76 and cut.bbox[3] >= 186
    # strokes that belong to another region are left out
    taken = np.zeros(image.shape[:2], np.uint8)
    taken[60:200, 200:240] = 1
    other = snap_text(image, (170, 60, 240, 200), "v", exclude=taken)
    assert other.bbox[2] <= 202
    # a flat area holds no text
    assert snap_text(image, (150, 230, 190, 255), "v") is None


def test_add_region_tool_snaps_saves_and_undoes(tmp_path):
    image_path, json_path = make_page(tmp_path, unboxed=True)
    session = AgentSession(AgentSettings(), AgentMemory(str(tmp_path / "mem")), [image_path])
    session.start_page(0, PageDoc.load(image_path, json_path))
    registry = ToolRegistry(session)
    bad = registry.execute("add_region", {"direction": "v", "reason": "r"})
    assert not bad.ok and "box" in bad.text
    nothing = registry.execute("add_region", {"box": [150, 200, 190, 230], "direction": "v", "reason": "r"})
    assert not nothing.ok and "no text strokes" in nothing.text
    added = registry.execute("add_region", {"box": [158, 30, 196, 100], "direction": "v", "reason": "手寫字",
                                            "after": 2})
    assert added.ok and added.images and "Added #4" in added.text
    doc = session.doc
    region = doc.get(4)
    x1, y1, x2, y2 = region.bbox
    assert 160 <= x1 <= 168 and 178 <= x2 <= 192 and y1 <= 26 and y2 >= 108
    assert doc.order == [1, 2, 4, 3] and region.origin == "agent" and region.direction == "v"
    assert "added by you" in session.candidate_lines()
    registry.execute("set_text", {"id": 4, "source_text": "ABCD", "translation": "甲乙", "category": "handwritten",
                                  "confidence": 0.8})
    doc.save()
    saved = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert [r["translation"] for r in saved["regions"]] == ["", "", "甲乙", ""]
    assert saved["regions"][2]["direction"] == "v" and saved["regions"][2]["font_size"] > 0
    mask = _decode(saved["mask_raw"])
    assert mask[60, 172] == 255 and mask[200, 172] == 0  # the new text is erased, nothing else
    assert {r["id"]: r["origin"] for r in doc.sidecar()["regions"]}[4] == "agent"
    # merged into a detector region, the added strokes are still erased
    registry.execute("merge_regions", {"ids": [2, 4], "reason": "同一個框"})
    doc.save()
    merged = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert len(merged["regions"]) == 3 and _decode(merged["mask_raw"])[60, 172] == 255
    registry.execute("undo", {})
    # undo the text, then the region
    registry.execute("undo", {})
    registry.execute("undo", {})
    assert 4 not in doc.regions and doc.order == [1, 2, 3]
    # grid cells of a zoom on an area work as well
    zoomed = registry.execute("zoom", {"box": [150, 10, 200, 120]})
    assert zoomed.ok and "add_region accepts these cells" in zoomed.text
    by_cells = registry.execute("add_region", {"cells": "A1:H8", "direction": "v", "reason": "r"})
    assert by_cells.ok and doc.get(4).bbox[0] >= 158


def test_recall_check_adds_missed_text_before_the_page(tmp_path):
    pages = [make_page(tmp_path, "001", unboxed=True), make_page(tmp_path, "002")]
    recall = [[
        {"box": [160, 25, 195, 115], "text": "ABCD", "kind": "handwritten", "direction": "v"},
        {"box": [20, 20, 60, 120], "text": "text1", "kind": "dialogue", "direction": "v"},   # already boxed
        {"box": [290, 200, 300, 210], "text": "…", "kind": "punctuation", "direction": "v"},
        {"box": [150, 200, 190, 230], "text": "?", "kind": "other", "direction": "v"},      # nothing there
        {"box": "bad", "text": "", "kind": "other", "direction": "v"},
    ], []]
    # #4 came from the recall check and was never zoomed: the first mark_page_done lists it
    script = [_calls(*[("set_text", {"id": i, "source_text": f"text{i}", "translation": f"譯{i}",
                                     "category": "dialogue", "confidence": 0.9}) for i in range(1, 5)]),
              _calls(("mark_page_done", {"confidence": 0.9})),
              _calls(("mark_page_done", {"confidence": 0.9}))] + _finish_page()
    runner, client, store, events = _runner(tmp_path, pages, script, recall=recall)
    outcomes = asyncio.run(runner.run())
    events.close()
    assert [o["status"] for o in outcomes] == ["done", "done"]
    assert len(client.recall_requests) == 2
    first_page = client.requests[0][-1]["content"][0]["text"]
    assert "A recall check added #4" in first_page and "added by the recall check (handwritten)" in first_page
    assert "(320x240 px)" in first_page
    logged = [e for e in read_events(events.path) if e["type"] == "recall"]
    assert logged[0]["added"] == [4]
    assert logged[0]["ignored"] == {"punctuation": 1, "duplicate": 1, "no_strokes": 1, "invalid": 1}
    assert logged[1]["added"] == []
    saved = next(iter(json.load(open(pages[0][1], encoding="utf-8")).values()))
    assert [r["translation"] for r in saved["regions"]] == ["譯1", "譯2", "譯3", "譯4"]
    assert store.load_sidecar("001")["regions"][3]["origin"] == "recall"
    assert runner.totals["recall_requests"] == 2


def test_recall_failure_does_not_stop_the_page(tmp_path):
    pages = [make_page(tmp_path, "001")]

    class FailingRecall(ScriptedClient):
        async def complete(self, messages, tools, tool_choice):
            if tools and tools[0]["function"]["name"] == "report_unboxed_text":
                raise RuntimeError("recall down")
            return await super().complete(messages, tools, tool_choice)

    store = AgentStore(os.path.join(os.path.dirname(pages[0][0]), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    client = FailingRecall(_finish_page())
    runner = AgentRunner(AgentSettings(), client, store, events, [pages[0][0]], "CHT")
    outcomes = asyncio.run(runner.run())
    events.close()
    assert outcomes[0]["status"] == "done"
    recall = [e for e in read_events(events.path) if e["type"] == "recall"]
    assert recall and "recall down" in recall[0]["error"]


def test_system_prompt_keeps_sound_effects_as_drawn(tmp_path):
    pages = [make_page(tmp_path, "001")]
    runner, client, store, events = _runner(tmp_path, pages, [])
    events.close()
    prompt = runner.system_prompt()
    assert "never translated: skip_region with category sfx" in prompt
    assert "Text inside a speech bubble is dialogue and gets translated" in prompt



# --- program checks and page confidence -----------------------------------------------
def _checked_session(tmp_path):
    image_path, json_path = make_page(tmp_path)
    memory = AgentMemory(str(tmp_path / "mem"))
    memory.glossary_upsert("即死パンチ", "即死拳")
    memory.character_upsert("ディメっち", {"target_name": "小迪梅（暱稱）"})
    session = AgentSession(AgentSettings(), memory, [image_path], target_lang="CHT")
    session.start_page(0, PageDoc.load(image_path, json_path))
    return session, ToolRegistry(session)


def _say(tools, rid, source, translation, confidence=0.95):
    result = tools.execute("set_text", {"id": rid, "source_text": source, "translation": translation,
                                        "category": "dialogue", "confidence": confidence})
    assert result.ok, result.text


def test_program_checks_are_shown_once_then_set_the_confidence(tmp_path):
    session, tools = _checked_session(tmp_path)
    _say(tools, 1, "ディメっちが即死パンチ", "迪梅っち用了即死之拳")   # kana left, glossary + character not used
    _say(tools, 2, "text2", "好")
    _say(tools, 3, "全然違う", "這是一段非常非常非常長的翻譯文字")    # reading differs from local OCR, too long
    first = tools.execute("mark_page_done", {"confidence": 0.97})
    assert not first.ok and not first.page_done
    assert first.summary.startswith("1 個框留有假名；1 個框沒用術語表譯法；1 個框沒用角色譯名")
    assert first.summary.endswith("2 個框沒有放大確認；1 個框譯文過長")
    for expected in ("#1: the translation still contains Japanese kana 「っち」", "glossary translates it as 「即死拳」",
                     "the character list says 「小迪梅（暱稱）」", "#3: your reading differs from the local OCR 「text3」",
                     "#3: the translation has 16 characters"):
        assert expected in first.text
    # the agent fixes #1, zooms #1 and #3 but keeps the long translation
    _say(tools, 1, "ディメっちが即死パンチ", "小迪梅用了即死拳")
    assert tools.execute("zoom", {"id": 3}).ok and tools.execute("zoom", {"id": 1}).ok
    done = tools.execute("mark_page_done", {"confidence": 0.97})
    assert done.page_done and "program checks 0.85" in done.text
    doc = session.doc
    assert doc.check_score == 0.85 and doc.confidence == 0.85 and doc.agent_confidence == 0.97
    assert [c["code"] for c in doc.checks] == ["overflow"] and not doc.flags
    side = doc.sidecar()
    assert side["check_score"] == 0.85 and side["regions"][2]["checks"] == ["overflow"]


def test_unfixed_serious_checks_go_to_the_review_list(tmp_path):
    session, tools = _checked_session(tmp_path)
    _say(tools, 1, "text1", "はい")
    _say(tools, 2, "text2", "好")
    assert tools.execute("zoom", {"box": [0, 0, 320, 240]}).ok   # an area zoom counts for every region in it
    _say(tools, 3, " ", "嗯")   # whitespace only: nothing was transcribed
    assert not tools.execute("mark_page_done", {"confidence": 0.9}).ok
    done = tools.execute("mark_page_done", {"confidence": 0.9})
    assert done.page_done
    doc = session.doc
    assert {(f["region"], f["type"]) for f in doc.flags} == {(1, "uncertain_translation"), (3, "uncertain_ocr")}
    assert doc.flags[0]["reason"] == "譯文裡還有日文假名「はい」"
    assert doc.confidence == 0.5 and not any(f["type"] == "low_confidence" for f in doc.flags)


def test_low_score_without_other_flags_adds_a_page_flag(tmp_path):
    session, tools = _checked_session(tmp_path)
    for rid in (1, 2, 3):
        _say(tools, rid, f"text{rid}", "好", confidence=0.4)
    assert tools.execute("zoom", {"box": [0, 0, 320, 240]}).ok
    done = tools.execute("mark_page_done", {"confidence": 0.9})
    assert done.page_done and session.doc.check_score == 0.5
    flag = session.doc.flags[-1]
    assert flag["type"] == "low_confidence" and flag["reason"].startswith("程式檢查的信心分數 0.50")


def test_overflow_counts_letters_not_punctuation(tmp_path):
    from types import SimpleNamespace

    from manga_translator.agent import checks

    session = SimpleNamespace(target_lang="CHT", memory=AgentMemory(str(tmp_path / "mem")),
                              settings=AgentSettings(double_read="none"))

    def codes(translation):
        region = SimpleNamespace(id=6, translation=translation, source_text="無理", ocr_local="無理",
                                 category="dialogue", confidence=0.95, origin="detector",
                                 bbox=[0, 0, 57, 103], font_size=57, direction="v")   # room for one character
        return [i.code for i in checks.region_issues(session, region)]

    assert codes("不行。") == [] and codes("不行……！") == []
    assert codes("絕對不行") == ["overflow"]
