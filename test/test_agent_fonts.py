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


def _saved_regions(json_path):
    return next(iter(json.load(open(json_path, encoding="utf-8")).values()))["regions"]


def _runs(region):
    """(text, fontFamily) of every run of a saved region's rich text, paragraph by paragraph."""
    return [[(i["text"], (i.get("style") or {}).get("fontFamily")) for i in block["inlines"]]
            for block in region["translation_rich"]["blocks"]]


@needs_fonts
def test_only_the_missing_characters_come_from_a_similar_font(tmp_path, catalog, vertical_rules):
    image_path, json_path, store, events = _page(tmp_path, [("這個值得[BR]了", "MS Gothic"), ("得了", "MS Gothic"),
                                                            ("", "MS Gothic"), ("值了", "MS Gothic")])
    service = FontService(catalog, FakeBackend(), "MS Gothic", "CHT")
    best = catalog.similar(catalog.find("MS Gothic"), "值", "CHT")[0][1]
    result = asyncio.run(service.fix_page(image_path, store, events))
    events.close()
    assert result["changed"] == 0 and result["borrowed"] == 2 and result["unfixed"] == 0
    assert _saved_fonts(json_path) == ["MS Gothic"] * 4  # every box keeps its font
    saved = _saved_regions(json_path)
    assert _runs(saved[0]) == [[("這個", None), ("值", best.value), ("得", None)], [("了", None)]]
    assert saved[0]["translation"] == "這個值得[BR]了" and saved[0]["translation_raw"] == "這個值得[BR]了"
    assert "translation_rich" not in saved[1]
    assert _runs(saved[3]) == [[("值", best.value), ("了", None)]]  # the same fallback for every box
    side = store.load_sidecar("001")
    assert side["fonts"]["changes"] == [] and side["fonts"]["borrowed"][0] == {
        "region": 1, "font": "MS Gothic", "fallback": best.value, "reason": "fallback", "missing": "值"}
    assert not side.get("review_flags")
    logged = [e for e in read_events(events.path) if e["type"] == "fonts"]
    assert logged and logged[0]["borrowed"] == 2 and logged[0]["changes"][0]["fallback"] == best.value
    # a second check finds nothing left to do
    again = EventLog(store.run_log_path("r2"), "r2")
    result = asyncio.run(service.fix_page(image_path, store, again))
    again.close()
    assert result["borrowed"] == result["unfixed"] == result["changed"] == 0
    assert _runs(_saved_regions(json_path)[0]) == _runs(saved[0])


@needs_fonts
def test_letters_that_no_font_has_go_on_the_review_list(tmp_path, catalog, vertical_rules):
    image_path, json_path, store, events = _page(tmp_path, [("\U00020000得", "MS Gothic")])
    result = asyncio.run(FontService(catalog, FakeBackend(), "MS Gothic", "CHT").fix_page(image_path, store, events))
    events.close()
    assert result["unfixed"] == 1 and result["borrowed"] == 0
    assert "translation_rich" not in _saved_regions(json_path)[0]
    side = store.load_sidecar("001")
    assert side["fonts"]["unfixed"][0]["missing"] == "\U00020000"
    assert side["review_flags"][-1]["type"] == "layout" and "\U00020000" in side["review_flags"][-1]["reason"]
    assert side["review_flags"][-1]["bbox"] == [20, 20, 60, 120]  # found again after a reorder
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
    side = store.load_sidecar("001")
    assert side["fonts"]["changes"][0]["reason"] == "layout"
    # the box now differs from the same kind of text elsewhere: on the review list, in the target language
    assert [f["region"] for f in side["review_flags"]] == [1, 2]
    assert side["review_flags"][0]["type"] == "layout" and "跑位" in side["review_flags"][0]["reason"]


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


def test_agent_defines_font_styles_and_uses_them(tmp_path):
    image_path, json_path = make_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    save = lambda i, category="narration", **extra: ("set_text", {
        "id": i, "source_text": f"text{i}", "translation": f"譯{i}", "category": category, "confidence": 0.9, **extra})
    style = lambda name, font, reason="明體的旁白框": ("set_font_style", {"name": name, "font": font, "reason": reason})
    client = ScriptedClient([
        _calls(style("旁白", 1), style("x", 7), save(1, style="旁白"), save(2, style="手寫"),
               save(3, "dialogue", style="旁白")),
        _calls(save(2), save(3, "dialogue", style="旁白", reason="心聲用和旁白一樣的明體"), style("旁白", 2)),
        _calls(save(3, "dialogue", style="none"), ("mark_page_done", {"confidence": 0.9}))])
    runner = AgentRunner(AgentSettings(recall_pass=False), client, store, events, [image_path], "CHT",
                         fonts=SheetFonts())
    asyncio.run(runner.run())
    events.close()
    first = client.requests[0]
    assert "Fonts: every region is typeset in the user's font (A-OTF Shin Go Pr6N) unless it has a font style" \
        in first[0]["content"]
    intro = first[1]["content"]
    assert "Font sheet (set_font_style font):" in intro[0]["text"] and "2: AR TianniuB5" in intro[0]["text"]
    assert "Font styles (set_text style):\n(no font styles yet)" in intro[0]["text"]
    assert intro[1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    results = [m["content"] for m in client.requests[1] if m.get("role") == "tool"]
    assert "Font style '旁白' = Noto Serif TC (added)" in results[0]
    assert "there is no font 7" in results[1]
    assert "there is no font style '手寫'" in results[3]
    assert "normal dialogue keeps the user's font" in results[4]  # dialogue needs a reason
    later = [m["content"] for m in client.requests[2] if m.get("role") == "tool"]
    assert "already sets Noto Serif TC on page(s) 001" in later[-1]  # a used style keeps its font
    saved = _saved_regions(json_path)
    assert [r.get("font_family") for r in saved] == ["Noto Serif TC", None, None]  # none = the user's font again
    side = store.load_sidecar("001")["regions"]
    assert side[0]["font"] == "Noto Serif TC" and side[0]["style"] == "旁白" and side[2]["style"] == ""
    assert runner.memory.font_styles["旁白"]["font"] == "Noto Serif TC"
    assert json.load(open(os.path.join(store.memory_dir, "font_styles.json"), encoding="utf-8"))["旁白"]["reason"] \
        == "明體的旁白框"
    start = next(e for e in read_events(events.path) if e["type"] == "run_start")
    assert start["fonts"] == ["Noto Serif TC", "AR TianniuB5"]


def test_styles_of_earlier_chapters_are_listed_with_their_font_number(tmp_path):
    image_path, _ = make_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    work = {"手寫": {"font": "AR TianniuB5", "reason": "圓珠筆手寫", "chapter": "第33話"},
            "招牌": {"font": "Not On The Sheet", "reason": "r"}}
    runner = AgentRunner(AgentSettings(recall_pass=False), ScriptedClient([]), store, events, [image_path], "CHT",
                         fonts=SheetFonts(), font_styles=work)
    text = runner.preamble[1]["content"][0]["text"]
    assert "- 手寫: font 2 (AR TianniuB5) (圓珠筆手寫)" in text and "- 招牌: Not On The Sheet (r)" in text
    assert runner.memory.font_styles["手寫"]["from"] == "第33話" and runner.memory.font_styles["招牌"]["from"] == "work"
    # without the font sheet there are no styles
    plain = AgentRunner(AgentSettings(recall_pass=False), ScriptedClient([]), AgentStore(str(tmp_path / "w2")),
                        events, [image_path], "CHT", font_styles=work)
    assert plain.memory.font_styles == {} and "set_font_style" not in plain.registry.tools
    assert "set_font_style" in runner.registry.tools
    events.close()


def test_no_sheet_without_fonts(tmp_path):
    image_path, _ = make_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    client = ScriptedClient([_calls(("set_text", {"id": 1, "source_text": "t", "translation": "譯", "category": "dialogue",
                                                  "confidence": 0.9, "style": "旁白", "reason": "r"})),
                             _calls(("mark_page_done", {"confidence": 0.9}))])
    runner = AgentRunner(AgentSettings(recall_pass=False, max_steps_per_page=2), client, store, events, [image_path],
                         "CHT")
    asyncio.run(runner.run())
    events.close()
    assert isinstance(client.requests[0][1]["content"], str) and "Fonts:" not in client.requests[0][0]["content"]
    assert "no font styles" in [m for m in client.requests[1] if m.get("role") == "tool"][0]["content"]
    assert "set_font_style" not in runner.registry.tools


def test_font_styles_are_kept_for_the_work(tmp_path):
    from manga_translator.agent import font_styles

    assert font_styles.style_name("  Hand   Written ") == "hand written"
    with pytest.raises(ValueError):
        font_styles.style_name(" ")
    with pytest.raises(ValueError):
        font_styles.style_name("x" * 31)
    prompt = tmp_path / "dict" / "作品.yaml"
    path = font_styles.work_fonts_path(str(prompt))
    assert path == os.path.join(str(tmp_path / "dict"), "agent", "作品.fonts.json") and font_styles.work_fonts_path("") == ""
    assert font_styles.load_work_styles(path) == {}
    styles = {"旁白": {"font": "Noto Serif TC", "reason": "明體", "page": "003"}}
    assert font_styles.save_work_styles(path, styles, "第34話") == ["旁白"]
    assert font_styles.save_work_styles(path, styles, "第35話") == []  # nothing new
    assert font_styles.load_work_styles(path) == {"旁白": {"font": "Noto Serif TC", "reason": "明體", "chapter": "第34話"}}
    styles["手寫"] = {"font": "AR TianniuB5", "reason": "手寫"}
    assert font_styles.save_work_styles(path, styles, "第35話") == ["手寫"]
    assert set(font_styles.load_work_styles(path)) == {"旁白", "手寫"}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{broken")
    assert font_styles.load_work_styles(path) == {}


def test_workflow_keeps_installed_work_styles_and_saves_them(tmp_path):
    from manga_translator.agent import workflow
    from manga_translator.agent.store import AgentMemory

    class Catalog:
        def find(self, value):
            return object() if value == "Noto Serif TC" else None

    class Service:
        catalog = Catalog()

    path = str(tmp_path / "agent" / "w.fonts.json")
    with open(os.path.join(str(tmp_path), "seed.json"), "w", encoding="utf-8") as fh:
        json.dump({"styles": {"旁白": {"font": "Noto Serif TC", "reason": "明體"},
                              "招牌": {"font": "Gone", "reason": "r"}}}, fh)
    os.makedirs(os.path.dirname(path))
    os.replace(os.path.join(str(tmp_path), "seed.json"), path)
    logs = []
    assert set(workflow._work_styles(path, Service(), logs.append)) == {"旁白"}
    assert any("Gone" in line for line in logs)
    memory = AgentMemory(str(tmp_path / "memory"))
    memory.font_style_set("手寫", "AR TianniuB5", "手寫字", "002")
    events = EventLog(str(tmp_path / "run.jsonl"), "r1")
    workflow._save_work_styles(path, memory, str(tmp_path / "第34話" / "manga_translator_work"), events, logs.append)
    events.close()
    assert set(json.load(open(path, encoding="utf-8"))["styles"]) == {"旁白", "招牌", "手寫"}
    logged = [e for e in read_events(str(tmp_path / "run.jsonl")) if e["type"] == "font_styles"]
    assert logged and logged[0]["styles"] == {"手寫": "AR TianniuB5"} and logged[0]["file"] == "w.fonts.json"


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


def test_workflow_scans_fonts_while_preprocessing(tmp_path, monkeypatch):
    from manga_translator.agent import workflow
    from test_agent_core import FakeTranslator, _finish_page

    folder = tmp_path / "chapter"
    folder.mkdir()
    order = []

    class Cfg:
        class translator:
            target_lang = "CHT"

    async def font_service(translator, config, settings, target_lang, log):
        order.append("fonts start")
        await asyncio.sleep(0)
        order.append("fonts done")
        return None

    class Translator(FakeTranslator):
        async def translate_batch(self, images_with_configs, save_info=None):
            if self.template:
                order.append("preprocess start")
                await asyncio.sleep(0.01)
                order.append("preprocess done")
            return await super().translate_batch(images_with_configs, save_info)

    monkeypatch.setattr(workflow, "_font_service", font_service)
    asyncio.run(workflow.run_agent_workflow(Translator(tmp_path), [(str(folder / "001.png"), Cfg)], {},
                                            client=ScriptedClient(_finish_page()), log=lambda m: None,
                                            agent_settings={"inpaint_review": False, "recall_pass": False}))
    assert order == ["preprocess start", "fonts start", "fonts done", "preprocess done"]


def _face(family, *names):
    return fonts.FontFace("x.ttf", 0, family, "Regular", 400, tuple(names), frozenset())


def test_design_classes_come_from_the_names():
    cases = {
        ("MingLiU", "細明體"): "serif", ("PMingLiU",): "serif", ("Yu Mincho", "游明朝"): "serif",
        ("SimSun", "宋体"): "serif", ("Source Han Serif TC",): "serif",
        ("Microsoft JhengHei", "微軟正黑體"): "sans", ("A-OTF Shin Go Pr6N", "A-OTF 新ゴ Pr6N"): "sans",
        ("FZLanTingHei-M-GBK",): "sans", ("Microsoft Sans Serif",): "sans", ("Yu Gothic",): "sans",
        ("FangSong", "仿宋"): "fangsong", ("DFKai-SB", "標楷體"): "kai", ("KaiTi",): "kai",
        ("FZZhunYuan-M02",): "round", ("LiSu", "隶书"): "clerical",
        ("FOT-Humming Std",): None, ("Lolita",): None,  # "Ming" in Humming is not a Ming face
    }
    for names, kind in cases.items():
        assert fonts.design_class(_face(*names)) == kind, names
    ming, mincho, hei = _face("MingLiU"), _face("Yu Mincho"), _face("SimHei")
    assert fonts.class_term(ming, mincho) == fonts.CLASS_BONUS
    assert fonts.class_term(ming, hei) == -fonts.CLASS_BONUS
    assert fonts.class_term(ming, _face("Lolita")) == 0.0


class _Face:
    def __init__(self, value, lacks=""):
        self.value = self.label = self.family = value
        self.lacks = lacks

    def has(self, ch):
        return ch not in self.lacks

    def missing(self, chars):
        return "".join(sorted(c for c in set(chars) if c in self.lacks))


class _OneChoice:
    """A catalog whose only font for what MingLiU lacks is Yu Mincho, or none at all."""

    def __init__(self, has_choice=True):
        self.has_choice = has_choice
        self.by_name = {"MingLiU": _Face("MingLiU", "︒值"), "Yu Mincho": _Face("Yu Mincho")}
        self.asked = []

    def find(self, value):
        return self.by_name.get(value.split("::")[0])

    def similar(self, ref, chars, target_lang):
        self.asked.append(chars)
        return [(0.1, self.by_name["Yu Mincho"])] if self.has_choice else []


def _vertical(json_path):
    raw = json.load(open(json_path, encoding="utf-8"))
    for region in next(iter(raw.values()))["regions"]:
        region["direction"] = "v"
    json.dump(raw, open(json_path, "w", encoding="utf-8"), ensure_ascii=False)


def test_vertical_punctuation_a_font_lacks_is_borrowed_too(tmp_path, vertical_rules):
    texts = [("好。", "MingLiU"), ("值。", "MingLiU")]
    image_path, json_path, store, events = _page(tmp_path, texts)
    _vertical(json_path)
    catalog = _OneChoice()
    result = asyncio.run(FontService(catalog, FakeBackend(), "MingLiU", "CHT").fix_page(image_path, store, events))
    events.close()
    assert result["changed"] == 0 and result["borrowed"] == 2 and result["unfixed"] == 0
    assert _saved_fonts(json_path) == ["MingLiU", "MingLiU"]
    saved = _saved_regions(json_path)
    # the mark the renderer draws for vertical text (︒), not the one in the translation
    assert _runs(saved[0]) == [[("好", None), ("︒", "Yu Mincho")]]
    assert saved[0]["translation"] == "好︒" and saved[0]["translation_raw"] == "好。"
    assert _runs(saved[1]) == [[("值︒", "Yu Mincho")]]
    assert catalog.asked == ["︒"]  # the second box reuses the first fallback: it has 值 too
    assert not (store.load_sidecar("001") or {}).get("review_flags")

    # no font has them: punctuation is left to Qt, missing letters go on the review list
    image_path, json_path, store, events = _page(tmp_path / "none", texts)
    _vertical(json_path)
    result = asyncio.run(FontService(_OneChoice(False), FakeBackend(), "MingLiU", "CHT").fix_page(
        image_path, store, events))
    events.close()
    assert result["borrowed"] == 0 and result["unfixed"] == 1
    side = store.load_sidecar("001")
    assert side["fonts"]["kept"] == [{"region": 1, "font": "MingLiU", "missing": "︒", "tried": []}]
    assert [f["region"] for f in side["review_flags"]] == [2]
    assert "translation_rich" not in _saved_regions(json_path)[0]


def test_borrowed_glyphs_keep_the_rich_text_and_its_fonts(vertical_rules):
    from manga_translator.agent.fonts import borrow_glyphs
    from manga_translator.rendering.rich_text import RICH_TEXT_FORMAT

    face = _Face("MingLiU", "︒值")
    rich = {"format": RICH_TEXT_FORMAT, "blocks": [{"type": "paragraph", "inlines": [
        {"type": "text", "text": "值得", "style": {}},
        {"type": "text", "text": "值", "style": {"fontFamily": "DFKai-SB"}}]}]}
    fields = borrow_glyphs({"translation": "值得值", "translation_rich": rich}, face, "Yu Mincho")
    assert _runs(fields) == [[("值", "Yu Mincho"), ("得", None), ("值", "DFKai-SB")]]  # the user's font stays
    assert "translation" not in fields  # rich text already holds the replaced text
    assert borrow_glyphs({"translation": "得了", "direction": "v"}, face, "Yu Mincho") is None
    # auto direction: tall lines are vertical, so 。 is drawn as ︒
    tall = {"translation": "好。", "direction": "auto", "lines": [[[0, 0], [30, 0], [30, 200], [0, 200]]]}
    assert _runs(borrow_glyphs(tall, face, "Yu Mincho")) == [[("好", None), ("︒", "Yu Mincho")]]
    wide = dict(tall, lines=[[[0, 0], [200, 0], [200, 30], [0, 30]]])
    assert borrow_glyphs(wide, face, "Yu Mincho") is None


def test_a_font_flag_keeps_where_its_box_is():
    from manga_translator.agent.fonts import _where

    region = {"lines": [[[20, 20], [60, 20], [60, 120], [20, 120]]], "translation": "譯"}
    assert _where(region) == {"center": [40.0, 70.0], "bbox": [20, 20, 60, 120], "text": "譯"}
    assert _where({"lines": "broken"}) == {}
