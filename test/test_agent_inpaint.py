"""M3 inpaint review: units, attempts, precise mask operations, the review conversation."""
import asyncio
import base64
import json
import os

import cv2
import numpy as np

from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.inpaint import FILL, InpaintError, InpaintPage, decode_mask, fill_patch
from manga_translator.agent.inpaint_review import InpaintReviewer
from manga_translator.agent.runner import ChatReply, ToolCall
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore

import pytest


def _letters(image, x, y, letters="ABC", step=26):
    for i, letter in enumerate(letters):
        cv2.putText(image, letter, (x, y + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def _page():
    """A bubble (outline) with a column of text, a caption outside it, and a stray mask blob."""
    image = np.full((240, 320, 3), 255, np.uint8)
    cv2.ellipse(image, (90, 110), (60, 95), 0, 0, 360, (0, 0, 0), 3)
    _letters(image, 78, 70)
    _letters(image, 230, 150, "XY")
    ink = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) < 128).astype(np.uint8)
    outline = np.zeros_like(ink)
    cv2.ellipse(outline, (90, 110), (60, 95), 0, 0, 360, 1, 7)
    text = ink & (1 - outline)
    mask = cv2.dilate(text, np.ones((7, 7), np.uint8)) * 255
    mask[20:34, 280:300] = 255  # a mask blob without a region
    regions = [(1, [np.array([[70, 45], [110, 45], [110, 130], [70, 130]], float)], (70, 45, 110, 130)),
               (2, [np.array([[225, 128], [260, 128], [260, 180], [225, 180]], float)], (225, 128, 260, 180))]
    return image, mask, regions


class FakeBackend:
    """Fills the mask with white; ``holes`` leaves boxes untouched on the first call (residual text)."""

    def __init__(self, holes=()):
        self.holes = list(holes)
        self.calls = []
        self.saved = {}

    def methods(self):
        return ["lama_large"]

    async def inpaint(self, method, image, mask):
        self.calls.append((method, int((mask > 0).sum())))
        out = image.copy()
        fill = mask > 0
        if self.holes and len(self.calls) == 1:
            for x1, y1, x2, y2 in self.holes:
                fill[y1:y2, x1:x2] = False
        out[fill] = 255
        return out

    async def text_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.where(gray < 128, 255, 0).astype(np.uint8)

    def save(self, path, image):
        self.saved[path] = image


def _built(holes=()):
    image, mask, regions = _page()
    backend = FakeBackend(holes)
    page = InpaintPage(image, mask, regions, backend, "001")
    asyncio.run(page.initial("lama_large"))
    return page, backend, image, mask


def test_units_follow_regions_and_the_first_pass_is_checked():
    page, backend, image, mask = _built()
    assert [u.id for u in page.units] == [1, 2, 101]
    assert page.by_id[101].extra and page.by_id[1].regions == [1]
    assert backend.calls == [("lama_large", int((mask > 0).sum()))]
    for unit in page.units:
        assert unit.current.checked and unit.current.issues == []
    result = page.composite()
    assert (result[mask > 0] == 255).all()
    assert np.array_equal(result[mask == 0], image[mask == 0]), "pixels outside the mask never change"
    assert page.score() == 1.0


def test_residual_text_is_found_and_a_retry_starts_from_the_original():
    page, backend, image, mask = _built(holes=[(70, 40, 110, 80)])  # letter A stays in #1
    unit = page.by_id[1]
    assert unit.current.residual and unit.current.issues == ["residual_text"]
    assert not page.by_id[2].current.issues
    assert page.score() == 0.65
    # dilating around the residual grows the mask only there
    label, before, after = page.adjust(unit, "dilate", 3, "residual")
    assert label == "dilate 3px (residual)" and after > before
    grown = unit.work & ~unit.base_mask
    ys, xs = np.nonzero(grown)
    assert ys.min() + unit.box[1] >= 40 - 7 and ys.max() + unit.box[1] < 80 + 7
    attempt = asyncio.run(page.retry(unit, "lama_large"))
    assert attempt.n == 2 and attempt.issues == [] and attempt.ops == ["dilate 3px (residual)"]
    assert backend.calls[-1] == ("lama_large", int(unit.work.sum())), "only this area is inpainted again"
    result = page.composite()
    assert np.array_equal(result[page.final_mask() == 0], image[page.final_mask() == 0])
    # going back to attempt 1 restores its pixels and mask
    page.pick(unit, 1)
    assert unit.current.residual and page.final_mask().sum() == (mask > 0).sum()
    with pytest.raises(InpaintError):
        page.pick(unit, 3)


def test_protect_lines_takes_the_outline_out_of_the_mask():
    image, mask, regions = _page()
    outline = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(outline, (90, 110), (60, 95), 0, 0, 360, 1, 3)
    mask[100:140, 20:60] = 255  # the mask reaches over the bubble outline on the left
    page = InpaintPage(image, mask, regions, FakeBackend(), "001")
    asyncio.run(page.initial("lama_large"))
    unit = page.by_id[1]
    assert unit.current.broken, "the cut outline is reported"
    x1, y1, x2, y2 = unit.box
    label, before, after, lines = page.protect_lines(unit, 2)
    assert lines >= 1 and after < before
    assert not (unit.work & (outline[y1:y2, x1:x2] > 0)).any(), "no outline pixel stays in the mask"
    attempt = asyncio.run(page.retry(unit, "lama_large"))
    assert not attempt.broken
    with pytest.raises(InpaintError, match="no mask edit"):
        page.undo(unit)


def test_components_select_paint_and_undo():
    page, backend, image, mask = _built()
    unit = page.by_id[1]
    # a stroke just outside the mask: a stroke candidate
    original = page.original
    cv2.line(original, (84, 129), (96, 129), (0, 0, 0), 2)
    mask_comps, strokes = page.components(unit)
    assert mask_comps and [s["name"] for s in strokes] == ["S1"]
    label, before, after = page.select(unit, [1], [])
    assert label == "select +S1" and after > before
    with pytest.raises(InpaintError, match="S2 does not exist"):
        page.select(unit, [2], [])
    w, h = unit.size
    page.paint(unit, (0, 0, w / 8, h / 8), "remove")
    assert page.undo(unit) == "paint remove"
    assert page.undo(unit) == "select +S1"
    assert np.array_equal(unit.work, unit.base_mask)
    with pytest.raises(InpaintError, match="M9 does not exist"):
        page.select(unit, [], [9])


def test_fill_takes_the_paper_color_and_warns_on_texture():
    crop = np.full((60, 60, 3), 240, np.uint8)
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    crop[25:35, 25:35] = 0
    out, note = fill_patch(crop, mask)
    assert (out[mask] == 240).all() and note == ""
    noisy = np.random.default_rng(0).integers(0, 255, (60, 60, 3)).astype(np.uint8)
    _out, note = fill_patch(noisy, mask)
    assert "not flat" in note


# --- the review conversation -----------------------------------------------------------
class ReviewClient:
    model_label = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def complete(self, messages, tools, tool_choice):
        assert tools[0]["function"]["name"] == "view_region"
        self.requests.append(messages)
        return self.script.pop(0)


def _calls(*calls):
    return ChatReply(content=None, prompt_tokens=800, completion_tokens=40,
                     tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(calls)])


def _review_page(tmp_path):
    image, mask, regions = _page()
    folder = tmp_path / "chapter"
    (folder / "manga_translator_work" / "json").mkdir(parents=True)
    path = str(folder / "001.png")
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ok, png = cv2.imencode(".png", mask)
    data = {path: {"regions": [{"lines": [p.tolist() for p in polys], "texts": [f"t{rid}"], "text": f"t{rid}",
                                "translation": f"譯{rid}", "font_size": 20, "direction": "v"}
                               for rid, polys, _box in regions],
                   "mask_raw": base64.b64encode(png.tobytes()).decode(), "mask_is_refined": True,
                   "original_width": 320, "original_height": 240}}
    json_path = folder / "manga_translator_work" / "json" / "001_translations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    store = AgentStore(str(folder / "manga_translator_work"))
    store.save_sidecar("001", {"page": "001.png", "status": "done", "confidence": 0.95, "regions": [],
                               "review_flags": []})
    return path, str(json_path), store


def _reviewer(tmp_path, script, holes=(), **settings):
    path, json_path, store = _review_page(tmp_path)
    backend = FakeBackend(holes)
    events = EventLog(store.run_log_path("r1"), "r1")
    client = ReviewClient(script)
    reviewer = InpaintReviewer(AgentSettings(**settings), client, store, events, "CHT", backend,
                               save_image=backend.save)
    return reviewer, client, backend, store, events, path, json_path


def test_review_fixes_residual_text_and_saves_everything(tmp_path):
    script = [_calls(("adjust_mask", {"id": 1, "op": "dilate", "px": 3, "scope": "residual", "reason": "殘字"}),
                     ("inpaint_region", {"id": 1, "reason": "重修"})),
              _calls(("finish_review", {}))]
    reviewer, client, backend, store, events, path, json_path = _reviewer(tmp_path, script, holes=[(70, 40, 110, 80)])
    outcome = asyncio.run(reviewer.review_page(path))
    events.close()
    assert outcome["status"] == "reviewed" and outcome["retried"] == 1 and outcome["flagged"] == 0
    first = client.requests[0][1]["content"]
    assert "#1 (region #1" in first[0]["text"] and "residual text at 1 spot(s)" in first[0]["text"]
    assert sum(1 for part in first if part["type"] == "image_url") >= 2  # overview + sheet
    # the result went to the inpainted image, the JSON mask and the sidecar
    saved = backend.saved[path]
    assert (cv2.cvtColor(saved, cv2.COLOR_RGB2GRAY)[40:80, 70:110] > 200).all()
    data = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert data["mask_is_refined"] and decode_mask(data["mask_raw"]).sum() > 0
    side = store.load_sidecar("001")
    unit = side["inpaint"]["units"]["1"]
    assert unit["chosen"] == 2 and [a["verdict"] for a in unit["attempts"]] == [["residual_text"], ["pass"]]
    assert os.path.exists(os.path.join(store.candidates_dir("001"), unit["attempts"][0]["file"]))
    assert side["inpaint_score"] == 1.0 and side["confidence"] == 0.95
    usage = {"prompt_tokens": 1600, "completion_tokens": 80, "cached_tokens": 0, "requests": 2}
    assert side["inpaint"]["usage"] == usage
    logged = read_events(events.path)
    assert [e["type"] for e in logged if e["type"].startswith("inpaint")] == ["inpaint_start", "inpaint_done"]
    assert logged[-1]["usage"] == usage
    calls = [e for e in logged if e["type"] == "tool_call"]
    assert calls and all(e["agent"] == "inpaint" for e in calls)


def test_review_warns_once_then_flags_what_is_left(tmp_path):
    script = [_calls(("finish_review", {})), _calls(("finish_review", {"note": "ok"}))]
    reviewer, client, backend, store, events, path, json_path = _reviewer(tmp_path, script, holes=[(70, 40, 110, 80)])
    outcome = asyncio.run(reviewer.review_page(path))
    events.close()
    tool_messages = [m for m in client.requests[1] if m.get("role") == "tool"]
    assert "residual text is still detected in #1" in tool_messages[0]["content"]
    side = store.load_sidecar("001")
    assert side["review_flags"] == [{"region": 1, "type": "inpaint", "reason": "修圖後仍偵測到殘留文字"}]
    assert side["inpaint_score"] == round(0.65 * 0.75, 3) and side["confidence"] == side["inpaint_score"]
    assert outcome["flagged"] == 1


def test_review_limits_attempts_and_steps(tmp_path):
    retry = ("inpaint_region", {"id": 2, "method": FILL, "reason": "試填色"})
    script = [_calls(retry), _calls(retry), _calls(("view_region", {"id": 2}))]
    reviewer, client, backend, store, events, path, json_path = _reviewer(
        tmp_path, script, inpaint_attempts_per_region=1, inpaint_review_steps=3)
    outcome = asyncio.run(reviewer.review_page(path))
    events.close()
    results = [e for e in read_events(events.path) if e["type"] == "tool_result"]
    assert results[0]["ok"] and not results[1]["ok"] and results[2]["ok"]
    side = store.load_sidecar("001")
    assert side["inpaint"]["units"]["2"]["attempts"][1]["method"] == FILL
    assert side["review_flags"][-1]["reason"] == "修圖審查在 3 步後停止，沒有完成"
    assert not side["inpaint"]["finished"] and outcome["status"] == "reviewed"
