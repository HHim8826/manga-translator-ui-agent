"""Fonts in the agent workflow: missing glyphs, similar fonts, the agent's font sheet, the layout check."""
import asyncio
import json
import os
import re

import pytest

from manga_translator.agent import fonts
from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.fonts import FontCatalog, FontService, Ink
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore
from manga_translator.utils import BASE_PATH

from test_agent_core import AgentRunner, ScriptedClient, _calls, make_page

FONT_DIR = os.path.join(BASE_PATH, "fonts")
needs_fonts = pytest.mark.skipif(not all(os.path.exists(os.path.join(FONT_DIR, f)) for f in ("msyh.ttc", "msgothic.ttc")),
                                 reason="bundled CJK fonts not found")


@pytest.fixture(scope="module")
def catalog():
    return FontCatalog(fonts.scan_fonts([FONT_DIR]))


@pytest.fixture
def vertical_rules(monkeypatch):
    rules = {"common": [], "horizontal": [], "vertical": [(re.compile("。"), "︒")]}
    monkeypatch.setattr(fonts, "_replacements", lambda: rules)


def test_rendered_chars_are_what_the_renderer_draws(vertical_rules):
    assert len(fonts.language_chars("CHT")) == 5401 and "值" in fonts.language_chars("CHT")
    assert fonts.rendered_chars("好。[BR]值 ♥", "h") == "。值好"  # no break marker, space or symbol
    assert fonts.rendered_chars("好。", "v") == "好︒"
    assert fonts.rendered_chars("好。", "auto") == "。好︒"  # either direction


@needs_fonts
def test_catalog_finds_fonts_and_ranks_similar_ones(catalog):
    gothic = catalog.find("MS Gothic")
    assert gothic is not None and catalog.find("ＭＳ ゴシック") is gothic  # localized name
    assert gothic.missing("值得") == "值" and not gothic.missing("得")
    regular = catalog.find("Microsoft YaHei")
    assert regular.style == "Regular" and catalog.find("Microsoft YaHei::Bold") is regular  # no bold file here
    assert catalog.usable(regular) and not catalog.usable(catalog.find("Microsoft YaHei UI"))  # same glyphs
    ranked = catalog.similar(gothic, "值得", "CHT")
    assert ranked and all(not face.missing("值得") for _, face in ranked)
    assert [s for s, _ in ranked] == sorted((s for s, _ in ranked), reverse=True)
    assert catalog.similarity(gothic, gothic, "cjk") == pytest.approx(1.0)


@needs_fonts
def test_font_sheet_lists_complete_fonts(catalog):
    service = FontService(catalog, None, "MS Gothic", "CHT")
    service.prepare_sheet()
    assert service.choices and all(f.coverage(fonts.language_chars("CHT")) >= 0.999 for f in service.choices)
    assert all("Gothic" not in f.family for f in service.choices)  # 97 % of Big5 level 1: not on the sheet
    assert service.sheet_url.startswith("data:image/")
    lines = service.sheet_text().splitlines()
    assert lines[0].startswith("0: MS Gothic") and lines[1].startswith("1: ")
    assert FontService(catalog, None, "anime ace", "ENG").catalog.sheet_faces("ENG", None) == []


class FakeBackend:
    """Ink boxes per font; ``moved`` fonts land somewhere else."""

    def __init__(self, moved=()):
        self.moved = set(moved)
        self.calls = []

    async def ink(self, image_path, index, font):
        self.calls.append((index, font))
        return Ink((100, 40, 140, 160)) if font in self.moved else Ink((10, 20, 50, 140))


def _page(tmp_path, regions):
    image_path, json_path = make_page(tmp_path, boxes=[(20, 20, 60, 120)] * len(regions))
    raw = json.load(open(json_path, encoding="utf-8"))
    for saved, (translation, font) in zip(next(iter(raw.values()))["regions"], regions):
        saved.update(translation=translation, translation_raw=translation, font_family=font, direction="h")
    json.dump(raw, open(json_path, "w", encoding="utf-8"), ensure_ascii=False)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    return image_path, json_path, store, EventLog(store.run_log_path("r1"), "r1")


def _saved_fonts(json_path):
    return [r.get("font_family") for r in next(iter(json.load(open(json_path, encoding="utf-8")).values()))["regions"]]


@needs_fonts
def test_missing_glyphs_get_the_most_similar_complete_font(tmp_path, catalog, vertical_rules):
    image_path, json_path, store, events = _page(tmp_path, [("這個值得", "MS Gothic"), ("得了", "MS Gothic"),
                                                            ("", "MS Gothic")])
    service = FontService(catalog, FakeBackend(), "MS Gothic", "CHT")
    best = catalog.similar(catalog.find("MS Gothic"), fonts.rendered_chars("這個值得", "h"), "CHT")[0][1]
    result = asyncio.run(service.fix_page(image_path, store, events))
    events.close()
    assert result["changed"] == 1 and result["unfixed"] == 0
    assert _saved_fonts(json_path) == [best.value, "MS Gothic", "MS Gothic"]
    side = store.load_sidecar("001")
    assert side["fonts"]["changes"] == [{"region": 1, "from": "MS Gothic", "to": best.value, "reason": "missing",
                                         "missing": "值"}]
    logged = [e for e in read_events(events.path) if e["type"] == "fonts"]
    assert logged and logged[0]["changed"] == 1 and logged[0]["changes"][0]["missing"] == "值"


@needs_fonts
def test_a_font_that_moves_the_text_is_not_used(tmp_path, catalog, vertical_rules):
    image_path, json_path, store, events = _page(tmp_path, [("值得", "MS Gothic")])
    everything = {face.value for face in catalog.faces if face.value}
    backend = FakeBackend(moved=everything - {"MS Gothic"})
    result = asyncio.run(FontService(catalog, backend, "MS Gothic", "CHT").fix_page(image_path, store, events))
    events.close()
    assert result["changed"] == 0 and result["unfixed"] == 1
    assert _saved_fonts(json_path) == ["MS Gothic"]  # missing glyphs beat a moved text
    side = store.load_sidecar("001")
    unfixed = side["fonts"]["unfixed"][0]
    assert unfixed["missing"] == "值" and unfixed["tried"] and "moved" in unfixed["tried"][0]
    assert side["review_flags"][-1]["type"] == "layout" and "值" in side["review_flags"][-1]["reason"]
    assert any(e["type"] == "flag" and e.get("agent") == "fonts" for e in read_events(events.path))


@needs_fonts
def test_an_agent_font_that_moves_the_text_goes_back_to_the_users_font(tmp_path, catalog, vertical_rules):
    other = next(f.value for f in catalog.faces if f.value and f.family == "Arial Unicode MS") \
        if catalog.find("Arial Unicode MS") else catalog.find("Microsoft YaHei").value
    image_path, json_path, store, events = _page(tmp_path, [("得了", other), ("得了", other)])
    backend = FakeBackend(moved={other})
    result = asyncio.run(FontService(catalog, backend, "MS Gothic", "CHT").fix_page(image_path, store, events))
    events.close()
    assert result["changed"] == 2 and _saved_fonts(json_path) == ["MS Gothic", "MS Gothic"]
    assert store.load_sidecar("001")["fonts"]["changes"][0]["reason"] == "layout"


def test_layout_check_rules():
    ref = Ink((100, 100, 140, 300))
    assert fonts.layout_ok(ref, Ink((101, 102, 141, 298)))[0]
    assert not fonts.layout_ok(ref, Ink((130, 100, 170, 300)))[0]  # moved
    assert not fonts.layout_ok(ref, Ink((95, 80, 145, 340)))[0]  # grew
    assert not fonts.layout_ok(ref, Ink((110, 140, 130, 260)))[0]  # shrank
    assert not fonts.layout_ok(ref, Ink((100, 140, 140, 270)))[0]  # a shorter column: the renderer dropped a character
    assert not fonts.layout_ok(ref, Ink((100, 100, 140, 300), outside=0.2))[0]
    assert fonts.layout_ok(None, None)[0] and not fonts.layout_ok(ref, None)[0]


def test_layout_check_failure_keeps_the_font(tmp_path):
    class Broken:
        async def ink(self, image_path, index, font):
            raise fonts.LayoutUnknown("cannot render")

    service = FontService.__new__(FontService)
    ok, why = asyncio.run(FontService._fits(service, "x.png", 0, "a", "b", {}))
    assert not ok  # no backend at all
    service.backend = Broken()
    ok, why = asyncio.run(FontService._fits(service, "x.png", 0, "a", "b", {}))
    assert not ok and "cannot render" in why


class SheetFonts:
    """A font service with a sheet, for the translation agent."""

    class Face:
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.choices = [self.Face("Noto Serif TC"), self.Face("AR TianniuB5")]
        self.default_font = "A-OTF Shin Go Pr6N"
        self.sheet_url = "data:image/png;base64,AAAA"

    def sheet_text(self):
        return "0: A-OTF Shin Go Pr6N\n1: Noto Serif TC\n2: AR TianniuB5"


def test_agent_picks_fonts_from_the_sheet(tmp_path):
    image_path, json_path = make_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    save = lambda i, **extra: ("set_text", {"id": i, "source_text": f"text{i}", "translation": f"譯{i}",
                                            "category": "narration", "confidence": 0.9, **extra})
    client = ScriptedClient([
        _calls(save(1, font=1), save(2, font=2), save(3, font=7)),
        _calls(save(3), save(2, font=0)),
        _calls(("mark_page_done", {"confidence": 0.9}))])
    runner = AgentRunner(AgentSettings(recall_pass=False), client, store, events, [image_path], "CHT",
                         fonts=SheetFonts())
    asyncio.run(runner.run())
    events.close()
    first = client.requests[0]
    assert "Fonts: every region is typeset in the user's font (A-OTF Shin Go Pr6N)" in first[0]["content"]
    intro = first[1]["content"]
    assert "Font sheet (set_text font):" in intro[0]["text"] and "2: AR TianniuB5" in intro[0]["text"]
    assert intro[1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    results = [m for m in client.requests[1] if m.get("role") == "tool"]
    assert "there is no font 7" in results[2]["content"]
    saved = next(iter(json.load(open(json_path, encoding="utf-8")).values()))["regions"]
    assert [r.get("font_family") for r in saved] == ["Noto Serif TC", None, None]  # 0 = the user's font again
    assert store.load_sidecar("001")["regions"][0]["font"] == "Noto Serif TC"
    start = next(e for e in read_events(events.path) if e["type"] == "run_start")
    assert start["fonts"] == ["Noto Serif TC", "AR TianniuB5"]


def test_no_sheet_without_fonts(tmp_path):
    image_path, _ = make_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    client = ScriptedClient([_calls(("set_text", {"id": 1, "source_text": "t", "translation": "譯", "category": "dialogue",
                                                  "confidence": 0.9, "font": 1})),
                             _calls(("mark_page_done", {"confidence": 0.9}))])
    runner = AgentRunner(AgentSettings(recall_pass=False, max_steps_per_page=2), client, store, events, [image_path],
                         "CHT")
    asyncio.run(runner.run())
    events.close()
    assert isinstance(client.requests[0][1]["content"], str) and "Fonts:" not in client.requests[0][0]["content"]
    assert "no font sheet" in [m for m in client.requests[1] if m.get("role") == "tool"][0]["content"]


def test_workflow_checks_fonts_after_the_agents(tmp_path):
    from manga_translator.agent.workflow import run_agent_workflow
    from test_agent_core import FakeTranslator, _finish_page

    folder = tmp_path / "chapter"
    folder.mkdir()

    class Cfg:
        class translator:
            target_lang = "CHT"

    class Service:
        choices = []
        default_font = "A-OTF Shin Go Pr6N"

        def __init__(self):
            self.pages = []

        async def fix_page(self, path, store, events, fix_missing=True):
            self.pages.append((os.path.basename(path), fix_missing))
            if len(self.pages) == 2:
                raise RuntimeError("renderer gone")
            return {"changed": 1, "unfixed": 0}

    service = Service()
    paths = [str(folder / "001.png"), str(folder / "002.png")]
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {},
                                   client=ScriptedClient(_finish_page() + _finish_page()), log=lambda m: None,
                                   agent_settings={"inpaint_review": False, "fix_missing_glyphs": False},
                                   fonts=service))
    assert service.pages == [("001.png", False), ("002.png", False)]
    store = AgentStore(str(folder / "manga_translator_work"))
    log = os.path.join(store.runs_dir, sorted(os.listdir(store.runs_dir))[-1])
    assert any(e["type"] == "error" and e.get("agent") == "fonts" for e in read_events(log))
