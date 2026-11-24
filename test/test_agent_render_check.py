"""The size the render draws each region at, set against its source strokes and the same text elsewhere
on the page (``render_check.py``), checked before mark_page_done."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from manga_translator.agent.checks import Issue
from manga_translator.agent.page import PageDoc
from manga_translator.agent.render_check import Drawn, _box, lines_across, measured_size, size_issues
from test_agent_core import _calls, _runner, make_page

FOUR = ((20, 20, 60, 120), (100, 20, 140, 120), (200, 150, 280, 180), (160, 20, 190, 120))


def _snap(size, lines, bbox, angle=0.0):
    return SimpleNamespace(font_size=size, lines=[[[0, 0], [1, 0], [1, 1], [0, 1]]] * lines, bbox=bbox, angle=angle)


def test_two_touching_lines_read_as_one_are_measured_per_line():
    # two lines of 26 px glyphs, 62 px across, found as one line 60 px thick
    assert measured_size(_snap(60, 1, (0, 0, 200, 62)), 2, "h") == pytest.approx(62 / 2.2)
    assert measured_size(_snap(60, 1, (0, 0, 62, 200)), 2, "v") == pytest.approx(62 / 2.2)  # columns: across is x
    assert measured_size(_snap(26, 2, (0, 0, 200, 62)), 2, "h") == 26  # both lines found
    assert measured_size(_snap(24, 1, (0, 0, 120, 90), angle=-38), 3, "h") == 24  # a slanted line is one line
    assert measured_size(_snap(4, 1, (0, 0, 10, 4)), 1, "h") is None  # a dot, not text
    assert measured_size(None, 1, "h") is None
    # two columns 67 px across, the snap found one of them, 28 px across: that one is the size
    assert measured_size(_snap(28, 1, (389, 0, 417, 90)), 2, "v", extent=67) == 28
    assert measured_size(_snap(60, 1, (0, 0, 62, 200)), 2, "v", extent=64) == pytest.approx(62 / 2.2)


def test_lines_across_count_a_column_cut_in_two_once():
    cut = [[[455, 1491], [477, 1491], [477, 1523], [455, 1523]], [[453, 1529], [476, 1529], [476, 1569], [453, 1569]]]
    assert lines_across(cut, "v") == (1, 24)  # バ / リ: one column the detector cut in two
    assert lines_across(cut, "h") == (2, 78)  # read across rows, the same pieces are two lines
    two = [[[389, 1825], [417, 1825], [417, 1918], [389, 1918]], [[413, 1804], [456, 1804], [456, 1841], [413, 1841]]]
    assert lines_across(two, "v") == (2, 67)
    assert lines_across([], "v") == (1, 0.0)


def test_the_source_is_measured_in_the_detected_lines_own_box(tmp_path):
    # ch42 p1 #14: Region.bbox (565, 1096, 721, 1225) took in a stroke beside the slanted label
    image_path, json_path = make_page(tmp_path, boxes=FOUR[:1])
    doc = PageDoc.load(image_path, json_path)
    doc.regions[1].data["lines"] = [[[566.0, 1206.0], [706.2, 1096.3], [720.4, 1115.0], [581.0, 1224.2]]]
    assert _box(doc.regions[1]) == (566.0, 1096.3, 720.4, 1224.2)
    assert doc.regions[1].bbox == [566, 1096, 721, 1225]


def _doc(tmp_path, sources):
    image_path, json_path = make_page(tmp_path, boxes=FOUR[:len(sources)])
    doc = PageDoc.load(image_path, json_path)
    for rid, source in enumerate(sources, start=1):
        doc.regions[rid].source_text = source
        doc.regions[rid].translation = f"譯{rid}"
    return doc


def test_drawn_too_big_or_too_small_for_its_source(tmp_path):
    doc = _doc(tmp_path, ["ケード", "（資材置き場？）", "井戸"])
    issues = size_issues(doc, {1: Drawn(54, 25), 2: Drawn(10, 21), 3: Drawn(19, 22)})
    assert [(i.code, i.region) for i in issues] == [("drawn_large", 1), ("drawn_small", 2)]
    assert issues[0].detail == {"drawn": 54, "source": 25, "times": 2.2}
    assert issues[1].detail == {"drawn": 10, "source": 21, "percent": 48}
    # strokes too small to measure: no verdict
    assert size_issues(doc, {1: Drawn(54, 8)}) == []


def test_a_dotted_ellipsis_is_not_small_text(tmp_path):
    doc = _doc(tmp_path, ["……あ", "ッ", "は、ハロー…"])
    doc.regions[1].translation, doc.regions[2].translation, doc.regions[3].translation = "︙啊", "！", "H-Hello︙"
    issues = size_issues(doc, {1: Drawn(10, 24), 2: Drawn(6, 15), 3: Drawn(14, 29)})
    assert [(i.code, i.region) for i in issues] == [("drawn_small", 2), ("drawn_small", 3)]


def test_the_same_label_drawn_unlike_its_copies(tmp_path):
    doc = _doc(tmp_path, ["テント屋根"] * 4)
    # #1: upright in a box too big for it; its strokes measure about as big as the others'
    issues = size_issues(doc, {1: Drawn(35, 25), 2: Drawn(19, 22), 3: Drawn(20, 25), 4: Drawn(19, 24)})
    assert [(i.code, i.region) for i in issues] == [("drawn_unlike", 1)]
    assert issues[0].detail == {"drawn": 35, "other": 19, "text": "テント屋根"}
    # strokes twice as big in the source: drawn bigger on purpose
    assert size_issues(doc, {1: Drawn(35, 45), 2: Drawn(19, 22), 3: Drawn(20, 25), 4: Drawn(19, 24)}) == []
    # a copy already listed against its own strokes does not count among the others
    issues = size_issues(doc, {1: Drawn(19, 22), 2: Drawn(40, 22), 3: Drawn(20, 25), 4: Drawn(19, 24)})
    assert [(i.code, i.region) for i in issues] == [("drawn_large", 2)]


class FakeCheck:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = 0

    async def issues(self, doc):
        self.calls += 1
        return self.rounds.pop(0) if self.rounds else []


def _say(i):
    return ("set_text", {"id": i, "source_text": f"text{i}", "translation": f"譯{i}", "category": "dialogue",
                         "confidence": 0.9, "reason": "讀圖"})


def test_the_agent_hears_the_drawn_sizes_and_what_is_left_goes_to_review(tmp_path):
    pages = [make_page(tmp_path, "001")]
    script = [_calls(_say(1), _say(2), _say(3)), _calls(("mark_page_done", {"confidence": 0.9})),
              _calls(("mark_page_done", {"confidence": 0.9}))]
    runner, client, store, events = _runner(tmp_path, pages, script, recall_pass=False)
    large = Issue("drawn_large", 2, {"drawn": 54, "source": 25, "times": 2.2}, 0.25)
    runner.render_check = FakeCheck([[large], [large]])
    outcomes = asyncio.run(runner.run())
    events.close()
    assert outcomes[0]["status"] == "done" and runner.render_check.calls == 2
    first = [m for m in client.requests[2] if m.get("role") == "tool"][-1]["content"]
    assert "Not finished yet" in first and "#2: rendered, its glyphs are about 54 px" in first
    side = store.load_sidecar("001")
    flags = [f for f in side["review_flags"] if f.get("type") == "layout"]
    assert len(flags) == 1 and flags[0]["reason"] == "渲染後字約 54 px，原文約 25 px，字太大"


def test_no_check_while_a_region_is_left_and_a_failing_check_is_skipped(tmp_path):
    pages = [make_page(tmp_path, "001")]
    script = [_calls(_say(1), _say(2)), _calls(("mark_page_done", {"confidence": 0.9})),
              _calls(_say(3)), _calls(("mark_page_done", {"confidence": 0.9}))]
    runner, client, store, events = _runner(tmp_path, pages, script, recall_pass=False)

    class Broken:
        calls = 0

        async def issues(self, doc):
            Broken.calls += 1
            raise RuntimeError("no renderer")

    runner.render_check = Broken()
    outcomes = asyncio.run(runner.run())
    events.close()
    assert outcomes[0]["status"] == "done"
    assert Broken.calls == 1  # not while #3 had no translation
    logged = [json.loads(line) for line in open(events.path, encoding="utf-8")]
    assert any(e["type"] == "render_check" and e["ok"] is False for e in logged)


def test_the_renderer_draws_the_page_the_agent_is_about_to_save(tmp_path, monkeypatch):
    """``PipelineFontBackend.load_page`` reads the JSON from memory, with the page's own flags."""
    from manga_translator.agent.fonts import PipelineFontBackend
    from manga_translator.config import Config, TranslatorConfig
    from manga_translator.manga_translator import MangaTranslator

    doc = _doc(tmp_path, ["text1", "text2", "text3"])
    translator = MangaTranslator(params={"load_text": True, "translator": "none"})
    monkeypatch.setattr(translator, "_prime_bubble_detection_cache", lambda config, ctx: None)
    backend = PipelineFontBackend(translator, Config(translator=TranslatorConfig(translator="none")))
    loaded = asyncio.run(backend.load_page(doc.image_path, doc.json_data()[doc.key]))
    assert loaded is not None
    blocks, image = loaded
    assert [b.translation for b in blocks] == ["譯1", "譯2", "譯3"] and image.shape == (240, 320, 3)
    assert translator._preloaded_load_text_payloads == {}  # nothing left behind for the render
    drawn = asyncio.run(backend.render_alone(doc.image_path, blocks[2]))
    ink = np.argwhere(drawn.min(axis=2) < 128)
    assert len(ink) and 140 <= ink[:, 0].mean() <= 190 and 190 <= ink[:, 1].mean() <= 290  # in box #3


def test_the_size_check_has_a_renderer_with_the_font_features_off():
    from manga_translator.agent.fonts import PipelineFontBackend
    from manga_translator.agent.workflow import _render_backend

    class Pipeline:
        def _load_text_and_regions_from_file(self, path, config):
            return None

        def set_preloaded_load_text_payload(self, path, payload):
            pass

    own = SimpleNamespace(backend=object())
    assert _render_backend(Pipeline(), None, own) is own.backend  # the font service's
    assert isinstance(_render_backend(Pipeline(), None, None), PipelineFontBackend)  # choose_fonts / fix_missing_glyphs off
    assert _render_backend(object(), None, None) is None  # no page from memory: no check


def test_the_runner_takes_the_renderer_it_is_given(tmp_path):
    pages = [make_page(tmp_path, "001")]
    runner, client, store, events = _runner(tmp_path, pages, [])
    events.close()
    assert runner.render_check is None
    from manga_translator.agent.runner import AgentRunner
    from manga_translator.agent.settings import AgentSettings

    backend = SimpleNamespace(load_page=None)
    with_backend = AgentRunner(AgentSettings(), client, store, events, [pages[0][0]], "CHT", render_backend=backend)
    assert with_backend.render_check is not None and with_backend.render_check.backend is backend


def test_a_slanted_label_drawn_upright(tmp_path):
    doc = _doc(tmp_path, ["置き", "下り", "にっ", "スッ"])
    sizes = {1: Drawn(24, 25, source_angle=-26, angle=0),  # ch42 p1 「堆」: upright over 置き slanted 26°
             2: Drawn(15, 15, source_angle=39, angle=37),  # drawn along the slant
             3: Drawn(30, 28, source_angle=2, angle=30),  # the agent's slant over strokes measured upright: not said
             4: Drawn(30, 28, source_angle=None, angle=0)}
    issues = size_issues(doc, sizes)
    assert [(i.code, i.region, i.detail) for i in issues] == [("drawn_turned", 1, {"source": -26, "drawn": 0})]
    # a region already listed for its size is not listed twice; 85° and -85° are 10° apart
    assert [i.code for i in size_issues(doc, {1: Drawn(54, 25, source_angle=-26, angle=0)})] == ["drawn_large"]
    assert size_issues(doc, {1: Drawn(24, 25, source_angle=85, angle=-85)}) == []
